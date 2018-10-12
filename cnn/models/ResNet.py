from collections import OrderedDict

from torch.nn import Conv2d, AvgPool2d, Linear, ModuleList

from cnn.MixedFilter import MixedConv, MixedConvWithReLU, Block
from cnn.MixedLayer import MixedLayer
from cnn.models import BaseNet


# init layers creation functions
def createMixedConvWithReLU(bitwidths, in_planes, kernel_size, stride, input_size, prevLayer):
    def f():
        return MixedConvWithReLU(bitwidths, in_planes, 1, kernel_size, stride, input_size, prevLayer)

    return f


def createMixedConv(bitwidths, in_planes, kernel_size, stride, input_size, prevLayer):
    def f():
        return MixedConv(bitwidths, in_planes, 1, kernel_size, stride, input_size, prevLayer)

    return f


class BasicBlock(Block):
    def __init__(self, bitwidths, in_planes, out_planes, kernel_size, stride, input_size, prevLayer):
        super(BasicBlock, self).__init__()

        stride1 = stride if in_planes == out_planes else (stride + 1)

        self.block1 = MixedLayer(out_planes,
                                 createMixedConvWithReLU(bitwidths[0] if isinstance(bitwidths[0], list) else bitwidths, in_planes, kernel_size,
                                                         stride1, input_size[0], prevLayer), useResidual=False)

        bitwidthIdx = 1
        self.downsample = None
        if in_planes != out_planes:
            downsampleBitwidth = [(b, None) for b, _ in (bitwidths[1] if isinstance(bitwidths[0], list) else bitwidths)]
            self.downsample = MixedLayer(out_planes,
                                         createMixedConv(downsampleBitwidth, in_planes, [1], stride1, input_size[0], prevLayer))
            bitwidthIdx += 1

        self.block2 = MixedLayer(out_planes,
                                 createMixedConvWithReLU(bitwidths[bitwidthIdx] if isinstance(bitwidths[0], list) else bitwidths, out_planes,
                                                         kernel_size, stride, input_size[-1], prevLayer=self.block1), useResidual=True)

    def forward(self, x):
        residual = self.downsample(x) if self.downsample else x

        out = self.block1(x)
        out = self.block2(out, residual)

        return out

    def outputLayer(self):
        return self.block2

    # input_bitwidth is a list of bitwidth per feature map
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

    # # select random alpha
    # def chooseRandomPath(self):
    #     if self.downsample:
    #         self.downsample.chooseRandomPath()
    #
    #     self.block1.chooseRandomPath()
    #     self.block2.chooseRandomPath()

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
        super(ResNet, self).__init__(args, initLayersParams=(args.bitwidth, args.kernel, args.nClasses))

        # turn on noise in 1st layer
        if len(self.layersList) > 0:
            layerIdx = 0
            self.layersList[layerIdx].turnOnNoise(layerIdx)
            # for op in self.layersList[0].getOps():
            #     op.noise = op.quant

        # update model parameters() function
        self.parameters = self.getLearnableParams

    @staticmethod
    def createMixedLayer(bitwidths, in_planes, out_planes, kernel_sizes, stride, input_size, prevLayer):
        f = createMixedConvWithReLU(bitwidths, in_planes, kernel_sizes, stride, input_size, prevLayer)
        layer = MixedLayer(out_planes, f)

        if layer.numOfOps() > 1:
            layer.setAlphas([0., 0., 0., 0.25, 0.75])
            # layer.setFiltersPartition()

        return layer

    # init layers (type, in_planes, out_planes)
    def initLayersPlanes(self):
        return [(self.createMixedLayer, 3, 16, 32),
                (BasicBlock, 16, 16, [32]), (BasicBlock, 16, 16, [32]), (BasicBlock, 16, 16, [32]),
                (BasicBlock, 16, 32, [32, 16]), (BasicBlock, 32, 32, [16]), (BasicBlock, 32, 32, [16]),
                (BasicBlock, 32, 64, [16, 8]), (BasicBlock, 64, 64, [8]), (BasicBlock, 64, 64, [8])]

    def initLayers(self, params):
        bitwidths, kernel_sizes, nClasses = params
        bitwidths = bitwidths.copy()

        layersPlanes = self.initLayersPlanes()

        # init previous layer
        prevLayer = None

        # create list of layers from layersPlanes
        # supports bitwidth as list of ints, i.e. same bitwidths to all layers
        # supports bitwidth as list of lists, i.e. specific bitwidths to each layer
        layers = ModuleList()
        for i, (layerType, in_planes, out_planes, input_size) in enumerate(layersPlanes):
            # build layer
            l = layerType(bitwidths, in_planes, out_planes, kernel_sizes, 1, input_size, prevLayer)
            # add layer to layers list
            layers.append(l)
            # remove layer specific bitwidths, in case of different bitwidths to layers
            # if isinstance(bitwidths[0], list):
            #     nMixedOpLayers = 1 if isinstance(l, MixedFilter) \
            #         else sum(1 for _, m in l._modules.items() if isinstance(m, MixedFilter))
            #     del bitwidths[:nMixedOpLayers]
            # # update previous layer
            # prevLayer = l.outputLayer()

        self.avgpool = AvgPool2d(8)
        # self.fc = MixedLinear(bitwidths, 64, 10)
        self.fc = Linear(64, nClasses).cuda()

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
        for layerIdx, layer in enumerate(self.layersList):
            assert (layer.added_noise is False)
            assert (layer.quantized is True)
            # remove quantization
            layer.unQuantize(layerIdx)
            # turn on gradients
            layer.turnOnGradients(layerIdx)

        # turn on noise in 1st layer
        if len(self.layersList) > 0:
            layerIdx = 0
            layer = self.layersList[layerIdx]
            layer.turnOnNoise(layerIdx)

        # update learnable parameters
        self.learnable_params = [param for param in self.parameters() if param.requires_grad]
        # reset nLayersQuantCompleted
        self.nLayersQuantCompleted = 0

    # def turnOnWeights(self):
    #     for layer in self.layersList:
    #         for op in layer.getOps():
    #             # turn off operations noise
    #             op.noise = False
    #             # remove hooks
    #             for handler in op.hookHandlers:
    #                 handler.remove()
    #             # clear hooks handlers list
    #             op.hookHandlers.clear()
    #             # turn on operations gradients
    #             for m in op.modules():
    #                 if isinstance(m, Conv2d):
    #                     for param in m.parameters():
    #                         param.requires_grad = True
    #                 elif isinstance(m, ActQuant):
    #                     m.quatize_during_training = False
    #                     m.noise_during_training = True
    #
    #     # set noise=True for 1st layer
    #     if len(self.layersList) > 0:
    #         layer = self.layersList[0]
    #         for op in layer.getOps():
    #             op.noise = op.quant
    #
    #     # update learnable parameters
    #     self.learnable_params = [param for param in self.parameters() if param.requires_grad]
    #     # reset nLayersQuantCompleted
    #     self.nLayersQuantCompleted = 0

    def switch_stage(self, loggerFuncs=[]):
        print('*** switch_stage() ***')
        # check whether we have to perform a switching stage, or there are no more stages left
        conditionFlag = self.nLayersQuantCompleted < len(self.layersList)
        if conditionFlag:
            layer = self.layersList[self.nLayersQuantCompleted]
            # assert (layer.alphas.requires_grad is False)

            # turn off noise in layers ops
            layer.turnOffNoise(self.nLayersQuantCompleted)
            # quantize layer
            layer.quantize(self.nLayersQuantCompleted)
            # turn off gradients
            layer.turnOffGradients(self.nLayersQuantCompleted)

            # for op in layer.getOps():
            #     # turn off noise in op
            #     assert (op.noise is True)
            #     op.noise = False
            #
            #     # # set pre & post quantization hooks, from now on we want to quantize these ops
            #     # op.hookHandlers.append(op.register_forward_pre_hook(save_quant_state))
            #     # op.hookHandlers.append(op.register_forward_hook(restore_quant_state))
            #
            #     # turn off gradients
            #     for m in op.modules():
            #         if isinstance(m, Conv2d):
            #             for param in m.parameters():
            #                 param.requires_grad = False
            #         elif isinstance(m, ActQuant):
            #             m.quatize_during_training = True
            #             m.noise_during_training = False

            # update learnable parameters
            self.learnable_params = [param for param in self.parameters() if param.requires_grad]

            # we have completed quantization of one more layer
            self.nLayersQuantCompleted += 1

            if self.nLayersQuantCompleted < len(self.layersList):
                layer = self.layersList[self.nLayersQuantCompleted]
                # turn on noise in the new layer we want to quantize
                layer.turnOnNoise(self.nLayersQuantCompleted)
                # for op in layer.getOps():
                #     assert (op.noise is False)
                #     op.noise = True

            logMsg = 'nLayersQuantCompleted:[{}/{}], learnable_params:[{}], learnable_alphas:[{}]' \
                .format(self.nLayersQuantCompleted, self.nLayers(), len(self.learnable_params), len(self.learnable_alphas))

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
            if (key.startswith('fc.')) or (token not in key):
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
