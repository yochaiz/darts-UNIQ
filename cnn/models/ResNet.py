from collections import OrderedDict

from torch.nn import Module, Conv2d, AvgPool2d, Linear, ModuleList

from UNIQ.actquant import ActQuant

from cnn.MixedOp import MixedOp, MixedConv, MixedConvWithReLU, MixedLinear, Block
from cnn.models import BaseNet
from cnn.models.BaseNet import save_quant_state, restore_quant_state


class BasicBlock(Block):
    def __init__(self, bitwidths, in_planes, out_planes, kernel_size, stride, input_size, input_bitwidth, prevLayer):
        super(BasicBlock, self).__init__()

        stride1 = stride if in_planes == out_planes else (stride + 1)

        self.block1 = MixedConvWithReLU(bitwidths[0] if isinstance(bitwidths[0], list) else bitwidths, in_planes, out_planes, kernel_size, stride1,
                                        input_size[0], input_bitwidth, prevLayer, useResidual=False)

        bitwidthIdx = 1
        self.downsample = None
        if in_planes != out_planes:
            downsampleBitwidth = [(b, None) for b, _ in (bitwidths[1] if isinstance(bitwidths[0], list) else bitwidths)]
            self.downsample = MixedConv(downsampleBitwidth, in_planes, out_planes, [1], stride1, input_size[0], input_bitwidth, prevLayer)
            bitwidthIdx += 1

        self.block2 = MixedConvWithReLU(bitwidths[bitwidthIdx] if isinstance(bitwidths[0], list) else bitwidths,
                                        out_planes, out_planes, kernel_size, stride, input_size[-1], self.block1.getOutputBitwidthList(),
                                        prevLayer=self.block1, useResidual=True)

    def forward(self, x):
        residual = self.downsample(x) if self.downsample else x

        out = self.block1(x)
        out = self.block2(out, residual)

        return out

    def outputLayer(self):
        return self.block2

    def getBops(self, input_bitwidth):
        bops = self.block1.getBops(input_bitwidth)
        if self.downsample:
            bops += self.downsample.getBops(input_bitwidth)

        input_bitwidth = self.block1.getCurrentOutputBitwidth()
        bops += self.block2.getBops(input_bitwidth)

        return bops

    def getCurrentOutputBitwidth(self):
        return self.block2.getCurrentOutputBitwidth()

    def getOutputBitwidthList(self):
        return self.block2.getOutputBitwidthList()

    # select random alpha
    def chooseRandomPath(self):
        if self.downsample:
            self.downsample.chooseRandomPath()

        self.block1.chooseRandomPath()
        self.block2.chooseRandomPath()

    # select alpha based on alphas distribution
    def choosePathByAlphas(self):
        if self.downsample:
            self.downsample.choosePathByAlphas()

        self.block1.choosePathByAlphas()
        self.block2.choosePathByAlphas()

    def evalMode(self):
        if self.downsample:
            self.downsample.evalMode()

        self.block1.evalMode()
        self.block2.evalMode()

    def numOfOps(self):
        return self.block2.numOfOps()


class ResNet(BaseNet):
    def __init__(self, args):
        super(ResNet, self).__init__(args, initLayersParams=(args.bitwidth, args.kernel))

        # set noise=True for 1st layer
        if len(self.layersList) > 0:
            for op in self.layersList[0].getOps():
                op.noise = op.quant

        # update model parameters() function
        self.parameters = self.getLearnableParams

    # init layers (type, in_planes, out_planes)
    def initLayersPlanes(self):
        return [(MixedConvWithReLU, 3, 16, 32),
                (BasicBlock, 16, 16, [32]), (BasicBlock, 16, 16, [32]), (BasicBlock, 16, 16, [32]),
                (BasicBlock, 16, 32, [32, 16]), (BasicBlock, 32, 32, [16]), (BasicBlock, 32, 32, [16]),
                (BasicBlock, 32, 64, [16, 8]), (BasicBlock, 64, 64, [8]), (BasicBlock, 64, 64, [8])]

    def initLayers(self, params):
        bitwidths, kernel_sizes = params
        bitwidths = bitwidths.copy()

        layersPlanes = self.initLayersPlanes()

        # init 1st layer input bitwidth which is 8-bits
        input_bitwidth = [8]
        # init previous layer
        prevLayer = None

        # create list of layers from layersPlanes
        # supports bitwidth as list of ints, i.e. same bitwidths to all layers
        # supports bitwidth as list of lists, i.e. specific bitwidths to each layer
        layers = ModuleList()
        for i, (layerType, in_planes, out_planes, input_size) in enumerate(layersPlanes):
            # build layer
            l = layerType(bitwidths, in_planes, out_planes, kernel_sizes, 1, input_size, input_bitwidth, prevLayer)
            # add layer to layers list
            layers.append(l)
            # remove layer specific bitwidths, in case of different bitwidths to layers
            if isinstance(bitwidths[0], list):
                nMixedOpLayers = 1 if isinstance(l, MixedOp) \
                    else sum(1 for _, m in l._modules.items() if isinstance(m, MixedOp))
                del bitwidths[:nMixedOpLayers]
            # update input_bitwidth for next layer
            input_bitwidth = l.getOutputBitwidthList()
            # update previous layer
            prevLayer = l.outputLayer()

        self.avgpool = AvgPool2d(8)
        # self.fc = MixedLinear(bitwidths, 64, 10)
        self.fc = Linear(64, 10).cuda()

        return layers

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

    def getLearnableParams(self):
        return self.learnable_params

    def turnOnWeights(self):
        for layer in self.layersList:
            for op in layer.getOps():
                # turn off operations noise
                op.noise = False
                # remove hooks
                for handler in op.hookHandlers:
                    handler.remove()
                # clear hooks handlers list
                op.hookHandlers.clear()
                # turn on operations gradients
                for m in op.modules():
                    if isinstance(m, Conv2d):
                        for param in m.parameters():
                            param.requires_grad = True
                    elif isinstance(m, ActQuant):
                        m.quatize_during_training = False
                        m.noise_during_training = True

        # set noise=True for 1st layer
        if len(self.layersList) > 0:
            layer = self.layersList[0]
            for op in layer.getOps():
                op.noise = op.quant

        # update learnable parameters
        self.learnable_params = [param for param in self.parameters() if param.requires_grad]
        # reset nLayersQuantCompleted
        self.nLayersQuantCompleted = 0

    def switch_stage(self, loggerFuncs=[]):
        # check whether we have to perform a switching stage, or there are no more stages left
        conditionFlag = self.nLayersQuantCompleted < len(self.layersList)
        if conditionFlag:
            layer = self.layersList[self.nLayersQuantCompleted]
            assert (layer.alphas.requires_grad is False)

            for op in layer.getOps():
                # turn off noise in op
                assert (op.noise is True)
                op.noise = False

                # set pre & post quantization hooks, from now on we want to quantize these ops
                op.hookHandlers.append(op.register_forward_pre_hook(save_quant_state))
                op.hookHandlers.append(op.register_forward_hook(restore_quant_state))

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

            if self.nLayersQuantCompleted < len(self.layersList):
                layer = self.layersList[self.nLayersQuantCompleted]
                # turn on noise in the new layer we want to quantize
                for op in layer.getOps():
                    op.noise = True

            logMsg = 'nLayersQuantCompleted:[{}], learnable_params:[{}], learnable_alphas:[{}]' \
                .format(self.nLayersQuantCompleted, len(self.learnable_params), len(self.learnable_alphas))

            # log message to all loggers
            for f in loggerFuncs:
                f(logMsg)

        return conditionFlag

    def buildStateDictMap(self, chckpntDict):
        map = {}
        map['conv1'] = 'layers.0.ops.0.0.op.0.0'
        map['bn1'] = 'layers.0.ops.0.0.op.0.1'

        layersNumberMap = [(1, 0, 1), (1, 1, 2), (1, 2, 3), (2, 1, 5), (2, 2, 6), (3, 1, 8), (3, 2, 9)]
        for n1, n2, m in layersNumberMap:
            map['layer{}.{}.conv1'.format(n1, n2)] = 'layers.{}.block1.ops.0.0.op.0.0'.format(m)
            map['layer{}.{}.bn1'.format(n1, n2)] = 'layers.{}.block1.ops.0.0.op.0.1'.format(m)
            map['layer{}.{}.conv2'.format(n1, n2)] = 'layers.{}.block2.ops.0.0.op.0.0'.format(m)
            map['layer{}.{}.bn2'.format(n1, n2)] = 'layers.{}.block2.ops.0.0.op.0.1'.format(m)

        downsampleLayersMap = [(2, 0, 4), (3, 0, 7)]
        for n1, n2, m in downsampleLayersMap:
            map['layer{}.{}.conv1'.format(n1, n2)] = 'layers.{}.block1.ops.0.0.op.0.0'.format(m)
            map['layer{}.{}.bn1'.format(n1, n2)] = 'layers.{}.block1.ops.0.0.op.0.1'.format(m)
            map['layer{}.{}.conv2'.format(n1, n2)] = 'layers.{}.block2.ops.0.0.op.0.0'.format(m)
            map['layer{}.{}.bn2'.format(n1, n2)] = 'layers.{}.block2.ops.0.0.op.0.1'.format(m)
            map['layer{}.{}.downsample.0'.format(n1, n2)] = 'layers.{}.downsample.ops.0.0.op.0'.format(m)
            map['layer{}.{}.downsample.1'.format(n1, n2)] = 'layers.{}.downsample.ops.0.0.op.1'.format(m)

        map['fc'] = 'fc'

        return map

    # load original pre_trained model of UNIQ
    def loadUNIQPreTrained(self, chckpntDict):
        map = self.buildStateDictMap(chckpntDict)
        newStateDict = OrderedDict()

        # make sure all chckpntDict keys exist in map, otherwise quit
        for key in chckpntDict.keys():
            prefix = key[:key.rindex('.')]
            if prefix not in map:
                return False

        token = '.ops.'
        for key in chckpntDict.keys():
            if key.startswith('fc.'):
                newStateDict[key] = chckpntDict[key]
                continue

            prefix = key[:key.rindex('.')]
            suffix = key[key.rindex('.'):]
            newKey = map[prefix]
            # find new key layer
            idx = newKey.find(token)
            if idx >= 0:
                newKeyOp = newKey[:idx]
                # init path to layer
                layerPath = [p for p in newKeyOp.split('.')]
                # get layer by walking through path
                layer = self
                for p in layerPath:
                    layer = getattr(layer, p)
                # update layer ops
                for j in range(layer.nOpsCopies()):
                    for i in range(layer.numOfOps()):
                        newKey = map[prefix].replace(newKeyOp + token + '0.0.', newKeyOp + token + '{}.{}.'.format(j, i))
                        newStateDict[newKey + suffix] = chckpntDict[key]

        # load model weights
        self.load_state_dict(newStateDict)

    # load weights from the same model but with single operation per layer, like a uniform bitwidth trained model
    def loadSingleOpPreTrained(self, chckpntDict):
        newStateDict = OrderedDict()

        token = '.ops.'
        for key in chckpntDict.keys():
            if key.startswith('fc.'):
                newStateDict[key] = chckpntDict[key]
                continue

            prefix = key[:key.rindex('.')]
            suffix = key[key.rindex('.'):]

            # find new key layer
            newKeyOp = prefix[:prefix.index(token)]
            # init path to layer
            layerPath = [p for p in newKeyOp.split('.')]
            # get layer by walking through path
            layer = self
            for p in layerPath:
                layer = getattr(layer, p)
            # update layer ops
            for j in range(layer.nOpsCopies()):
                for i in range(layer.numOfOps()):
                    newKey = prefix.replace(newKeyOp + token + '0.0.', newKeyOp + token + '{}.{}.'.format(j, i))
                    newStateDict[newKey + suffix] = chckpntDict[key]

        # load model weights
        self.load_state_dict(newStateDict)
