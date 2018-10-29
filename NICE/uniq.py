from abc import abstractmethod
from torch.nn import Module, Conv2d, Linear

from NICE.quantize import quantize, backup_weights, restore_weights
from NICE.actquant import ActQuant


# def save_state(self, _):
#     self.full_parameters = {}
#     layers_list = self.layers_list()
#     layers_steps = self.layers_steps()
#
#     self.full_parameters = backup_weights(layers_list, {})
#
#     if self.quant and not self.training and not self.statistics_phase:
#         for i in range(len(layers_steps)):
#             self.quantize.quantize_uniform_improved(layers_steps[i])
#
#         if self.quantize.hardware_clamp:
#             self.quantize.assign_act_clamp_during_val(layers_list)
#             self.quantize.assign_weight_clamp_during_val(layers_list)
#
#     elif self.quant and self.training:
#
#         if self.allow_grad:
#             for i in range(self.quant_stage_for_grads):
#                 self.quantize.quantize_uniform_improved(layers_steps[i])
#
#         else:
#             if self.noise:
#                 self.quantize.add_improved_uni_noise(layers_steps[self.training_stage])
#             for i in range(self.training_stage):
#                 self.quantize.quantize_uniform_improved(layers_steps[i])
#
#
# def restore_state(self, _, __):
#     layers_list = self.layers_list()
#     restore_weights(layers_list, self.full_parameters)


class UNIQNet(Module):
    def __init__(self, bitwidth, act_bitwidth, params, std_act_clamp=3, std_weight_clamp=3.45, noise_mask=0.05):
        super(UNIQNet, self).__init__()
        # self.quant_epoch_step = quant_epoch_step
        # self.quant_start_stage = quant_start_stage
        self.quant = False
        self.noise = False
        # self.wrpn = False

        assert (isinstance(bitwidth, list))
        assert (len(bitwidth) == 1)
        self.bitwidth = bitwidth
        assert (isinstance(act_bitwidth, list))
        assert (len(act_bitwidth) <= 1)
        self.act_bitwidth = act_bitwidth

        # self.training_stage = 0
        # self.step = step
        self.num_of_layers_each_step = 1
        self.act_noise = False
        self.act_quant = True
        # self.quant_edges = True
        # self.quant_first_layer = True
        # self.register_forward_pre_hook(save_state)
        # self.register_forward_hook(restore_state)
        self.layers_b_dict = None
        # self.noise_mask_init = 0. if not noise else noise_mask
        self.noise_mask_init = noise_mask

        quantize_act_bitwidth = self.act_bitwidth[0] if len(act_bitwidth) > 0 else 16
        self.quantize = quantize(bitwidth[0], quantize_act_bitwidth, None, std_act_clamp=std_act_clamp, std_weight_clamp=std_weight_clamp,
                                 noise_mask=self.noise_mask_init)

        self.derivedClassSpecific(params)

        # set conv bitwidth & act_bitwidth
        convList = [x for x in self.modules() if isinstance(x, Conv2d)]
        for index, layer in enumerate(convList):
            layer.__param_bitwidth__ = self.bitwidth[index]
            layer.__act_bitwidth__ = self.act_bitwidth

        self.full_parameters = {}
        self.layers_list = self.build_layers_list()

        # self.statistics_phase = False
        # self.allow_grad = False
        # self.random_noise_injection = False

        # self.open_grad_after_each_stage = True
        # self.quant_stage_for_grads = quant_start_stage

        # self.noise_level = 0
        # self.noise_batch_counter = 0

    # specific code for derived classes
    @abstractmethod
    def derivedClassSpecific(self, params):
        raise NotImplementedError('subclasses must override derivedClassSpecific()!')

    def build_layers_list(self):
        modules_list = list(self.modules())
        return [x for x in modules_list if isinstance(x, Conv2d) or isinstance(x, Linear) or isinstance(x, ActQuant)]

    def quantizeFunc(self):
        assert (len(self.bitwidth) == 1)
        assert (self.full_parameters == {})
        assert (self.quant is True)
        assert (self.noise is False)
        # check if we are in inference mode or we are training with switching stage and this op has already changed stage
        assert ((self.training is False) or ((self.training is True) and (self.noise is False)))

        layers_list = self.layers_list
        self.full_parameters = backup_weights(layers_list, {})
        self.quantize.quantize_uniform_improved(layers_list)

    def add_noise(self):
        assert (len(self.bitwidth) == 1)
        assert (self.full_parameters == {})
        assert (self.noise is True)
        assert (self.training is True)
        assert (self.quant is False)

        layers_list = self.layers_list
        self.full_parameters = backup_weights(layers_list, {})
        self.quantize.add_improved_uni_noise(layers_list)

    def restore_state(self):
        assert (self.full_parameters != {})
        restore_weights(self.layers_list, self.full_parameters)
        self.full_parameters = {}

    # def layers_steps(self):
    #     split_layers = self.split_one_layer_with_parameter_in_step()
    #     return split_layers

    # def count_of_parameters_layer_in_list(self, list):
    #     counter = 0
    #     for layer in list:
    #         if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
    #             counter += 1
    #     return counter

    # def split_one_layer_with_parameter_in_step(self):
    #     layers = self.layers_list()
    #     splited_layers = []
    #     split_step = []
    #     for layer in layers:
    #
    #         if (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)) \
    #                 and self.count_of_parameters_layer_in_list(split_step) == self.num_of_layers_each_step:
    #             splited_layers.append(split_step)
    #             split_step = []
    #             split_step.append(layer)
    #         else:
    #             split_step.append(layer)
    #
    #     # add left layers
    #     if len(split_step) > 0:
    #         splited_layers.append(split_step)
    #
    #     return splited_layers

    # def switch_stage(self, epoch_progress):
    #     """
    #     Switches the stage of network to the next one.
    #     :return:
    #     """
    #
    #     layers_steps = self.layers_steps()
    #     max_stage = len(layers_steps)
    #     if self.training_stage >= max_stage + 1:
    #         return
    #
    #     if self.open_grad_after_each_stage == False:
    #         if (np.floor(
    #                 epoch_progress / self.quant_epoch_step) + self.quant_start_stage > self.training_stage and self.training_stage < max_stage - 1):
    #             self.training_stage += 1
    #             print("Switching stage, new stage is: ", self.training_stage)
    #             for step in layers_steps[:self.training_stage]:
    #                 for layer in step:
    #                     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) \
    #                             or isinstance(layer, nn.BatchNorm2d):
    #
    #                         for param in layer.parameters():
    #                             param.requires_grad = False
    #                     elif isinstance(layer, ActQuant) or isinstance(layer, ActQuantDeepIspPic) or isinstance(layer, ActQuantWRPN):
    #                         layer.quatize_during_training = True
    #                         layer.noise_during_training = False
    #
    #             if self.act_noise:
    #                 for layer in layers_steps[self.training_stage]:  # Turn on noise only for current stage
    #                     if isinstance(layer, ActQuant) or isinstance(layer, ActQuantDeepIspPic) or isinstance(layer, ActQuantWRPN):
    #                         layer.noise_during_training = True
    #             return True
    #
    #         elif (np.floor(epoch_progress / self.quant_epoch_step) + self.quant_start_stage > max_stage - 1 and self.allow_grad == False):
    #             self.allow_grad = True
    #             self.quant_stage_for_grads = self.training_stage + 1
    #             self.random_noise_injection = False
    #             print("Switching stage, allowing all grad to propagate. new stage is: ", self.training_stage)
    #             for step in layers_steps[:self.training_stage]:
    #                 for layer in step:
    #                     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
    #                         for param in layer.parameters():
    #                             param.requires_grad = True
    #             return True
    #         return False
    #
    #     else:
    #
    #         if (np.floor(epoch_progress / self.quant_epoch_step) + self.quant_start_stage > self.training_stage and
    #                 self.training_stage < max_stage - 1):
    #
    #             self.training_stage += 1
    #             print("Switching stage, new stage is: ", self.training_stage)
    #             for step in layers_steps[:self.training_stage]:
    #                 for layer in step:
    #                     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) \
    #                             or isinstance(layer, nn.BatchNorm2d):
    #                         for param in layer.parameters():
    #                             param.requires_grad = True
    #                     elif isinstance(layer, ActQuant) or isinstance(layer, ActQuantDeepIspPic) or isinstance(layer, ActQuantWRPN):
    #                         layer.quatize_during_training = True
    #                         layer.noise_during_training = False
    #
    #             if self.act_noise:
    #                 for layer in layers_steps[self.training_stage]:  # Turn on noise only for current stage
    #                     if isinstance(layer, ActQuant) or isinstance(layer, ActQuantDeepIspPic) or isinstance(layer, ActQuantWRPN):
    #                         layer.noise_during_training = True
    #
    #             self.allow_grad = False
    #             return True
    #
    #         if (np.floor(epoch_progress / self.quant_epoch_step) + self.quant_start_stage > max_stage - 1 and self.allow_grad == False):
    #             self.allow_grad = True
    #             self.quant_stage_for_grads = self.training_stage + 1
    #             self.random_noise_injection = False
    #             print("Switching stage, allowing all grad to propagate. new stage is: ", self.training_stage)
    #
    #         return False
