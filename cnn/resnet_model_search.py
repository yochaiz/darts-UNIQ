from torch import load as loadModel
from torch.nn import Module, Conv2d, AvgPool2d
import torch.nn.functional as F
from UNIQ.actquant import ActQuant
from UNIQ.quantize import backup_weights, restore_weights, quantize
from UNIQ.flops_benchmark import count_flops
from cnn.MixedOp import MixedConv, MixedConvWithReLU, MixedLinear, MixedOp, QuantizedOp
from collections import OrderedDict
from torch import float32


def save_quant_state(self, _):
    assert (self.noise is False)
    if self.quant and not self.noise and self.training:
        self.full_parameters = {}
        layers_list = self.get_layers_list()
        layers_steps = self.get_layers_steps(layers_list)
        assert (len(layers_steps) == 1)

        self.full_parameters = backup_weights(layers_steps[0], self.full_parameters)
        quantize(layers_steps[0], bitwidth=self.bitwidth[0])


def restore_quant_state(self, _, __):
    assert (self.noise is False)
    if self.quant and not self.noise and self.training:
        layers_list = self.get_layers_list()
        layers_steps = self.get_layers_steps(layers_list)
        assert (len(layers_steps) == 1)

        restore_weights(layers_steps[0], self.full_parameters)  # Restore the quantized layers


class BasicBlock(Module):
    def __init__(self, bitwidths, in_planes, out_planes, kernel_size, stride):
        super(BasicBlock, self).__init__()

        stride1 = stride if in_planes == out_planes else (stride + 1)

        self.block1 = MixedConvWithReLU(bitwidths, in_planes, out_planes, kernel_size, stride1, useResidual=False)
        self.block2 = MixedConvWithReLU(bitwidths, out_planes, out_planes, kernel_size, stride, useResidual=True)

        self.downsample = MixedConv(bitwidths, in_planes, out_planes, kernel_size, stride1) \
            if in_planes != out_planes else None

    def forward(self, x):
        residual = self.downsample(x) if self.downsample else x

        out = self.block1(x)
        out = self.block2(out, residual)

        return out


class ResNet(Module):
    nClasses = 10  # cifar-10

    def __init__(self, criterion, bitwidths, kernel_sizes):
        super(ResNet, self).__init__()

        # init MixedConvWithReLU layers list
        self.layersList = []

        self.block1 = MixedConvWithReLU(bitwidths, 3, 16, kernel_sizes, 1)
        self.layersList.append(self.block1)

        layers = [
            BasicBlock(bitwidths, 16, 16, kernel_sizes, 1),
            BasicBlock(bitwidths, 16, 16, kernel_sizes, 1),
            BasicBlock(bitwidths, 16, 16, kernel_sizes, 1),
            BasicBlock(bitwidths, 16, 32, kernel_sizes, 1),
            BasicBlock(bitwidths, 32, 32, kernel_sizes, 1),
            BasicBlock(bitwidths, 32, 32, kernel_sizes, 1),
            BasicBlock(bitwidths, 32, 64, kernel_sizes, 1),
            BasicBlock(bitwidths, 64, 64, kernel_sizes, 1),
            BasicBlock(bitwidths, 64, 64, kernel_sizes, 1)
        ]

        i = 2
        for l in layers:
            setattr(self, 'block{}'.format(i), l)
            i += 1

        self.avgpool = AvgPool2d(8)
        self.fc = MixedLinear(bitwidths, 64, self.nClasses)

        # build mixture layers list
        self.layersList = [m for m in self.modules() if isinstance(m, MixedOp)]

        # build alphas list, i.e. architecture parameters
        self._arch_parameters = [l.alphas for l in self.layersList]

        # set noise=True for 1st layer
        for op in self.layersList[0].ops:
            if op.quant:
                op.noise = True

        # init criterion
        self._criterion = criterion

        # set learnable parameters
        self.learnable_params = [param for param in self.parameters() if param.requires_grad]
        # update model parameters() function
        self.parameters = self.getLearnableParams

        # init number of layers we have completed its quantization
        self.nLayersQuantCompleted = 0

    def nLayers(self):
        return len(self.layersList)

    def forward(self, x):
        out = self.block1(x)

        blockNum = 2
        b = getattr(self, 'block{}'.format(blockNum))
        while b is not None:
            out = b(out)

            # move to next block
            blockNum += 1
            b = getattr(self, 'block{}'.format(blockNum), None)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target, self.countBops())

    def arch_parameters(self):
        return self._arch_parameters

    def getLearnableParams(self):
        return self.learnable_params

    # create list of lists of alpha with its corresponding operation
    def alphas_state(self):
        res = []
        for layer in self.layersList:
            layerAlphas = [(a, op) for a, op in zip(layer.alphas, layer.ops)]
            res.append(layerAlphas)

        return res

    # counts the entire model bops
    def countBops(self):
        totalBops = 0
        for layer in self.layersList:
            weights = F.softmax(layer.alphas, dim=-1)
            for w, b in zip(weights, layer.bops):
                totalBops += (w * b)

        return totalBops

    # return top k operations per layer
    def topOps(self, k):
        top = []
        for layer in self.layersList:
            # calc weights from alphas and sort them
            weights = F.softmax(layer.alphas, dim=-1)
            # weights = layer.alphas
            wSorted, wIndices = weights.sort(descending=True)
            # keep only top-k
            wSorted = wSorted[:k]
            wIndices = wIndices[:k]
            # add to top
            top.append([(i, w.item(), layer.alphas[i], layer.ops[i]) for w, i in zip(wSorted, wIndices)])

        return top

    def switch_stage(self, logger=None):
        # TODO: freeze stage alphas as well ???
        if self.nLayersQuantCompleted + 1 < len(self.layersList):
            layer = self.layersList[self.nLayersQuantCompleted]
            for op in layer.ops:
                # turn off noise in op
                assert (op.noise is True)
                op.noise = False

                # set pre & post quantization hooks, from now on we want to quantize these ops
                op.register_forward_pre_hook(save_quant_state)
                op.register_forward_hook(restore_quant_state)

                # turn off gradients
                for m in op.modules():
                    if isinstance(m, Conv2d):
                        for param in m.parameters():
                            param.requires_grad = False
                    elif isinstance(m, ActQuant):
                        m.quatize_during_training = True
                        m.noise_during_training = False

            # update learnable parameters
            self.learnable_params = [param for param in self.parameters() if param.requires_grad]

            # we have completed quantization of one more layer
            self.nLayersQuantCompleted += 1

            # turn on noise in the new layer we want to quantize
            layer = self.layersList[self.nLayersQuantCompleted]
            for op in layer.ops:
                op.noise = True

            if logger:
                logger.info('Switching stage, nLayersQuantCompleted:[{}], learnable_params:[{}]'
                            .format(self.nLayersQuantCompleted, len(self.learnable_params)))

    # load original pre_trained model of UNIQ
    def loadUNIQPre_trained(self, path, logger, gpu):
        checkpoint = loadModel(path, map_location=lambda storage, loc: storage.cuda(gpu))
        chckpntDict = checkpoint['state_dict']
        newStateDict = OrderedDict()

        map = {}
        map['conv1'] = 'block1.ops.0.op.0.0'
        map['bn1'] = 'block1.ops.0.op.0.1'

        layersNumberMap = [(1, 0, 2), (1, 1, 3), (1, 2, 4), (2, 1, 6), (2, 2, 7), (3, 1, 9), (3, 2, 10)]
        for n1, n2, m in layersNumberMap:
            map['layer{}.{}.conv1'.format(n1, n2)] = 'block{}.block1.ops.0.op.0.0'.format(m)
            map['layer{}.{}.bn1'.format(n1, n2)] = 'block{}.block1.ops.0.op.0.1'.format(m)
            map['layer{}.{}.conv2'.format(n1, n2)] = 'block{}.block2.ops.0.op.0.0'.format(m)
            map['layer{}.{}.bn2'.format(n1, n2)] = 'block{}.block2.ops.0.op.0.1'.format(m)

        downsampleLayersMap = [(2, 5), (3, 8)]
        for n, m in downsampleLayersMap:
            map['layer{}.0.conv1'.format(n)] = 'block{}.block1.ops.0.op.0.0'.format(m)
            map['layer{}.0.bn1'.format(n)] = 'block{}.block1.ops.0.op.0.1'.format(m)
            map['layer{}.0.conv2'.format(n)] = 'block{}.block2.ops.0.op.0.0'.format(m)
            map['layer{}.0.bn2'.format(n)] = 'block{}.block2.ops.0.op.0.1'.format(m)
            map['layer{}.0.downsample.0'.format(n)] = 'block{}.downsample.ops.0.op.0'.format(m)
            map['layer{}.0.downsample.1'.format(n)] = 'block{}.downsample.ops.0.op.1'.format(m)

        map['fc'] = 'fc.ops.0.op'

        token = '.ops.'
        for key in chckpntDict.keys():
            prefix = key[:key.rindex('.')]
            suffix = key[key.rindex('.'):]
            newKey = map[prefix]
            # find new key layer
            newKeyOp = newKey[:newKey.index(token)]
            # init path to layer
            layerPath = [p for p in newKeyOp.split('.')]
            # get layer by walking through path
            layer = self
            for p in layerPath:
                layer = getattr(layer, p)
            # update layer ops
            for i in range(len(layer.ops)):
                newStateDict[newKey + suffix] = chckpntDict[key]
                newKey = newKey.replace(newKeyOp + token + '{}.'.format(i), newKeyOp + token + '{}.'.format(i + 1))

        # load model weights
        self.load_state_dict(newStateDict)

        logger.info('Loaded model from [{}]'.format(path))
        logger.info('checkpoint validation accuracy:[{:.5f}]'.format(checkpoint['best_prec1']))

    def loadPreTrainedModel(self, path, logger, gpu):
        checkpoint = loadModel(path, map_location=lambda storage, loc: storage.cuda(gpu))
        chckpntDict = checkpoint['state_dict']
        curStateDict = self.state_dict()
        newStateDict = OrderedDict()
        # split state_dict keys by model layers
        layerKeys = {}  # collect ALL layer keys in state_dict
        layerOp0Keys = {}  # collect ONLY layer.ops.0. keys in state_dict, for duplication to the rest of layer ops
        token = '.ops.'
        for key, item in chckpntDict.items():
            prefix = key[:key.index(token)]
            isKey0 = prefix + token + '0.' in key
            if prefix in layerKeys:
                layerKeys[prefix].append(key)
                if isKey0: layerOp0Keys[prefix].append(key)
            else:
                layerKeys[prefix] = [key]
                if isKey0: layerOp0Keys[prefix] = [key]
        # duplicate state_dict values according to number of ops in each model layer
        for layerKey in layerKeys.keys():
            # init path to layer
            layerPath = [p for p in layerKey.split('.')]
            # get layer by walking through path
            layer = self
            for p in layerPath:
                layer = getattr(layer, p)
            # init stateKey prefix
            srcPrefix = layerKey + token + '0.op.'
            dstPrefix = layerKey + token + '{}.op.'
            # add missing layer operations to state_dict
            for stateKey in layerOp0Keys[layerKey]:
                if srcPrefix not in stateKey:
                    srcPrefix = layerKey + token + '0.'
                    dstPrefix = layerKey + token + '{}.'
                for i in range(len(layer.ops)):
                    # update newKey to fit current model structure, i.e. with '.op.' in keys
                    newKey = stateKey.replace(srcPrefix, layerKey + token + '{}.op.'.format(i))
                    j = 0
                    keyOldFormat = stateKey.replace(srcPrefix, dstPrefix.format(j))
                    foundMatch = False
                    while (keyOldFormat in layerKeys[layerKey]) and (not foundMatch):
                        if chckpntDict[keyOldFormat].size() == curStateDict[newKey].size():
                            newStateDict[newKey] = chckpntDict[keyOldFormat]
                            foundMatch = True
                        j += 1
                        keyOldFormat = stateKey.replace(srcPrefix, dstPrefix.format(j))

        # load model weights
        self.load_state_dict(newStateDict)

        # # load model alphas
        # if 'alphas' in checkpoint:
        #     chkpntAlphas = checkpoint['alphas']
        #     for i, l in enumerate(self.layersList):
        #         layerChkpntAlphas = chkpntAlphas[i]
        #         # init corresponding layers with its alphas
        #         jStart = 0
        #         jEnd = len(layerChkpntAlphas)
        #         while jEnd < len(l.alphas):
        #             l.alphas[jStart:jEnd].copy_(layerChkpntAlphas)
        #             jStart = jEnd
        #             jEnd += len(layerChkpntAlphas)

        logger.info('Loaded model from [{}]'.format(path))
        logger.info('checkpoint validation accuracy:[{:.5f}]'.format(checkpoint['best_prec1']))

    def loadFromCheckpoint(self, path, logger, gpu):
        checkpoint = loadModel(path, map_location=lambda storage, loc: storage.cuda(gpu))
        # split state_dict keys by model layers
        layerKeys = {}  # collect ALL layer keys in state_dict
        layerOp0Keys = {}  # collect ONLY layer.ops.0. keys in state_dict, for duplication to the rest of layer ops
        token = '.ops.'
        for key in checkpoint['state_dict'].keys():
            prefix = key[:key.index(token)]
            isKey0 = prefix + token + '0.' in key
            if prefix in layerKeys:
                layerKeys[prefix].append(key)
                if isKey0: layerOp0Keys[prefix].append(key)
            else:
                layerKeys[prefix] = [key]
                if isKey0: layerOp0Keys[prefix] = [key]
        # duplicate state_dict values according to number of ops in each model layer
        for layerKey in layerKeys.keys():
            # init path to layer
            layerPath = [p for p in layerKey.split('.')]
            # get layer by walking through path
            layer = self
            for p in layerPath:
                layer = getattr(layer, p)
            # add missing layer operations to state_dict
            for stateKey in layerOp0Keys[layerKey]:
                for i in range(len(layer.ops)):
                    newKey = stateKey.replace(layerKey + token + '0.', layerKey + token + '{}.'.format(i))
                    if newKey not in layerKeys[layerKey]:
                        checkpoint['state_dict'][newKey] = checkpoint['state_dict'][stateKey]

        # load model weights
        self.load_state_dict(checkpoint['state_dict'])
        # load model alphas
        # if 'alphas' in checkpoint:
        #     for i, l in enumerate(self.layersList):
        #         layerChkpntAlphas = checkpoint['alphas'][i]
        #         assert (layerChkpntAlphas.size() <= l.alphas.size())
        #         l.alphas = layerChkpntAlphas.expand_as(l.alphas)

        # load nLayersQuantCompleted
        # if 'nLayersQuantCompleted' in checkpoint:
        #     self.nLayersQuantCompleted = checkpoint['nLayersQuantCompleted']

        logger.info('Loaded model from [{}]'.format(path))
        logger.info('checkpoint validation accuracy:[{:.5f}]'.format(checkpoint['best_prec1']))
