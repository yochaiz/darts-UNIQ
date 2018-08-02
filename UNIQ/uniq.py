import torch.nn as nn
from UNIQ.quantize import backup_weights, quantize, add_noise, restore_weights
from UNIQ.actquant import ActQuant


def save_state(self, _):
    self.full_parameters = {}
    layers_list = self.get_layers_list()
    layers_steps = self.get_layers_steps(layers_list)

    if self.quant and not self.training:
        self.full_parameters = backup_weights(layers_list, {})
        for i in range(len(layers_steps)):
            quantize(layers_steps[i], bitwidth=self.bitwidth[i])
    elif self.noise and self.training:
        self.full_parameters = backup_weights(layers_steps[self.training_stage], {})
        add_noise(layers_steps[self.training_stage], bitwidth=self.bitwidth[self.training_stage],
                  training=self.training)

        for i in range(self.training_stage):
            self.full_parameters = backup_weights(layers_steps[i], self.full_parameters)
            quantize(layers_steps[i], bitwidth=self.bitwidth[i])


def restore_state(self, _, __):
    layers_list = self.get_layers_list()
    layers_steps = self.get_layers_steps(layers_list)

    if self.quant and not self.training:
        restore_weights(layers_list, self.full_parameters)
    elif self.noise and self.training:
        restore_weights(layers_steps[self.training_stage],
                        self.full_parameters)  # Restore the noised layers
        for i in range(self.training_stage):
            restore_weights(layers_steps[i], self.full_parameters)  # Restore the quantized layers


class UNIQNet(nn.Module):
    def __init__(self, quant=False, noise=False, bitwidth=[], quant_edges=True, act_noise=True, step_setup=[15, 9],
                 act_bitwidth=[], act_quant=False):
        super(UNIQNet, self).__init__()
        self.quant = quant
        self.noise = noise
        # bitwidth from now on must be a list
        assert (isinstance(bitwidth, list))
        self.bitwidth = bitwidth
        self.training_stage = 0

        # step is number of groups to divide the network
        # if step is number of layers, than each layer is in its own group
        # self.step = len(self.bitwidth)

        self.act_noise = act_noise
        self.act_quant = act_quant
        assert (isinstance(act_bitwidth, list))
        self.act_bitwidth = act_bitwidth
        self.quant_edges = quant and quant_edges
        self.stages = list(range(step_setup[0], 1000, step_setup[1]))
        self.register_forward_pre_hook(save_state)
        self.register_forward_hook(restore_state)

    def get_layers_list(self):
        modules_list = list(self.modules())
        return [x for x in modules_list if
                isinstance(x, nn.Conv2d) or isinstance(x, nn.Linear) or isinstance(x, ActQuant)]

    def get_layers_steps(self, layers_list):
        layers_steps = []
        l = []
        for layer in layers_list:
            if isinstance(layer, ActQuant):
                l.append(layer)
                layers_steps.append(l)
                l = []
            elif len(l) == 0:
                l = [layer]
            else:
                layers_steps.append(l)
                l = [layer]
        if len(l) > 0:
            layers_steps.append(l)

        # remove edge layers if we don't quantize edges
        # TODO: need to consider quant_edges or not ???
        # if not self.quant_edges:
        #     layers_steps = layers_steps[1:-1]

        return layers_steps

    def switch_stage(self, logger):
        """
        Switches the stage of network to the next one.
        :return:
        """
        if self.training_stage + 1 >= len(self.layers_steps):
            return

        if logger:
            logger.info("Switching stage")

        self.training_stage += 1
        for step in self.layers_steps[:self.training_stage]:
            for layer in step:
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    for param in layer.parameters():
                        param.requires_grad = False
                elif isinstance(layer, ActQuant):
                    layer.quatize_during_training = True
                    layer.noise_during_training = False

        if self.act_noise:
            for layer in self.layers_steps[self.training_stage]:  # Turn on noise only for current stage
                if isinstance(layer, ActQuant):
                    layer.noise_during_training = True

    def prepare_uniq(self):
        """
        Prepares UNIQ network. Divides layers by stages and turns noise in first activation layer if needed
        :return:
        """
        # collect layers
        self.layers_list = self.get_layers_list()

        # set full precision to all layers
        for layer in self.layers_list:
            layer.__param_bitwidth__ = 32
            layer.__act_bitwidth__ = 32

        # collect layers by steps
        self.layers_steps = self.get_layers_steps(self.layers_list)
        # TODO: merge downsample with its conv to single step???

        # remove edge layers if we don't quantize edges
        if not self.quant_edges:
            self.layers_steps = self.layers_steps[1:-1]

        # set number of train steps
        self.step = len(self.layers_steps)

        # collect activations layers
        self.act_list = []
        for step in self.layers_steps:
            if len(step) > 1:
                for ind in range(len(step) - 1):
                    if (not isinstance(step[ind], ActQuant)) and (isinstance(step[ind + 1], ActQuant)):
                        self.act_list.append(step[ind])
                        # TODO: act_bitwidth is only if we have ReLU ??? it must go together ???

        if self.act_noise:
            for layer in self.layers_steps[0]:  # Turn on noise for first stage
                if isinstance(layer, ActQuant):
                    layer.noise_during_training = True

        if (self.quant is False) or (len(self.bitwidth) == 0):
            self.bitwidth = [32] * len(self.layers_steps)

        if (self.act_quant is False) or (len(self.act_bitwidth) == 0):
            self.act_bitwidth = [32] * len(self.act_list)

        # set qunatization bitwidth for layers we want to quantize
        for index, step in enumerate(self.layers_steps):
            for layer in step:
                layer.__param_bitwidth__ = self.bitwidth[index]
                layer.__act_bitwidth__ = 32

        # set qunatization bitwidth for activations we want to quantize
        for index, layer in enumerate(self.act_list):
            layer.__act_bitwidth__ = self.act_bitwidth[index]
