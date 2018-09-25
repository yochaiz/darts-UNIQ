from collections import OrderedDict

from torch.nn import Module, Conv2d, AvgPool2d, Linear, ModuleList

from UNIQ.actquant import ActQuant

from cnn.MixedOp import MixedOp, MixedConv, MixedConvWithReLU, MixedLinear
from cnn.models import BaseNet
from cnn.models.BaseNet import save_quant_state, restore_quant_state


class BasicBlock(Module):
    def __init__(self, bitwidths, in_planes, out_planes, kernel_size, stride, input_size, input_bitwidth, nOpsCopies=1):
        super(BasicBlock, self).__init__()

        stride1 = stride if in_planes == out_planes else (stride + 1)

        self.block1 = MixedConvWithReLU(bitwidths[0] if isinstance(bitwidths[0], list) else bitwidths, in_planes,
                                        out_planes, kernel_size, stride1, input_size[0], input_bitwidth,
                                        nOpsCopies=nOpsCopies, useResidual=False)

        self.block2 = MixedConvWithReLU(bitwidths[1] if isinstance(bitwidths[0], list) else bitwidths, out_planes,
                                        out_planes, kernel_size, stride, input_size[-1],
                                        self.block1.getOutputBitwidthList(), nOpsCopies=self.block1.numOfOps(),
                                        useResidual=True)

        downsampleBitwidth = [(b, None) for b, _ in (bitwidths[2] if isinstance(bitwidths[0], list) else bitwidths)]
        self.downsample = MixedConv(downsampleBitwidth, in_planes, out_planes, [1], stride1, input_size[0],
                                    input_bitwidth, nOpsCopies=self.block1.numOfOps()) \
            if in_planes != out_planes else None

    def forward(self, x):
        residual = self.downsample(x) if self.downsample else x

        out = self.block1(x)
        out = self.block2(out, residual)

        return out

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
    def chooseRandomPath(self, prev_alpha_idx):
        if self.downsample:
            self.downsample.chooseRandomPath(prev_alpha_idx)

        prev_alpha_idx = self.block1.chooseRandomPath(prev_alpha_idx)
        return self.block2.chooseRandomPath(prev_alpha_idx)

    # select alpha based on alphas distribution
    def choosePathByAlphas(self, prev_alpha_idx):
        if self.downsample:
            self.downsample.choosePathByAlphas(prev_alpha_idx)

        prev_alpha_idx = self.block1.choosePathByAlphas(prev_alpha_idx)
        return self.block2.choosePathByAlphas(prev_alpha_idx)

    def evalMode(self, prev_alpha_idx):
        if self.downsample:
            self.downsample.evalMode(prev_alpha_idx)

        prev_alpha_idx = self.block1.evalMode(prev_alpha_idx)
        return self.block2.evalMode(prev_alpha_idx)

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

        layersPlanes = self.initLayersPlanes()

        # init 1st layer input bitwidth which is 8-bits
        input_bitwidth = [8]
        # init 1st layer ops nCopies
        nCopies = 1

        # create list of layers from layersPlanes
        # supports bitwidth as list of ints, i.e. same bitwidths to all layers
        # supports bitwidth as list of lists, i.e. specific bitwidths to each layer
        layers = ModuleList()
        for i, (layerType, in_planes, out_planes, input_size) in enumerate(layersPlanes):
            # build layer
            l = layerType(bitwidths, in_planes, out_planes, kernel_sizes, 1, input_size, input_bitwidth, nCopies)
            # add layer to layers list
            layers.append(l)
            # remove layer specific bitwidths, in case of different bitwidths to layers
            if isinstance(bitwidths[0], list):
                nMixedOpLayers = sum(1 for m in l.modules() if isinstance(m, MixedOp))
                del bitwidths[:nMixedOpLayers]
            # update input_bitwidth for next layer
            input_bitwidth = l.getOutputBitwidthList()
            # update ops nCopies for next layer
            nCopies = l.numOfOps()

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

    # def _loss(self, input, target):
    #     logits = self(input)
    #     return self._criterion(logits, target, self.countBops())

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

    def switch_stage(self, logger=None):
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

            if logger:
                logger.info(
                    'Switching stage, nLayersQuantCompleted:[{}], learnable_params:[{}], learnable_alphas:[{}]'
                        .format(self.nLayersQuantCompleted, len(self.learnable_params), len(self.learnable_alphas)))

        return conditionFlag

    # load original pre_trained model of UNIQ
    def loadUNIQPre_trained(self, chckpntDict):
        newStateDict = OrderedDict()

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

        # map['fc'] = 'fc'

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
                        newKey = map[prefix].replace(newKeyOp + token + '0.0.',
                                                     newKeyOp + token + '{}.{}.'.format(j, i))
                        newStateDict[newKey + suffix] = chckpntDict[key]

        # load model weights
        self.load_state_dict(newStateDict)

    # def loadPreTrainedModel(self, path, logger, gpu):
    #     checkpoint = loadModel(path, map_location=lambda storage, loc: storage.cuda(gpu))
    #     chckpntDict = checkpoint['state_dict']
    #     curStateDict = self.state_dict()
    #     newStateDict = OrderedDict()
    #     # split state_dict keys by model layers
    #     layerKeys = {}  # collect ALL layer keys in state_dict
    #     layerOp0Keys = {}  # collect ONLY layer.ops.0. keys in state_dict, for duplication to the rest of layer ops
    #     token = '.ops.'
    #     for key, item in chckpntDict.items():
    #         prefix = key[:key.index(token)]
    #         isKey0 = prefix + token + '0.' in key
    #         if prefix in layerKeys:
    #             layerKeys[prefix].append(key)
    #             if isKey0: layerOp0Keys[prefix].append(key)
    #         else:
    #             layerKeys[prefix] = [key]
    #             if isKey0: layerOp0Keys[prefix] = [key]
    #     # duplicate state_dict values according to number of ops in each model layer
    #     for layerKey in layerKeys.keys():
    #         # init path to layer
    #         layerPath = [p for p in layerKey.split('.')]
    #         # get layer by walking through path
    #         layer = self
    #         for p in layerPath:
    #             layer = getattr(layer, p)
    #         # init stateKey prefix
    #         srcPrefix = layerKey + token + '0.op.'
    #         dstPrefix = layerKey + token + '{}.op.'
    #         # add missing layer operations to state_dict
    #         for stateKey in layerOp0Keys[layerKey]:
    #             if srcPrefix not in stateKey:
    #                 srcPrefix = layerKey + token + '0.'
    #                 dstPrefix = layerKey + token + '{}.'
    #             for i in range(len(layer.ops)):
    #                 # update newKey to fit current model structure, i.e. with '.op.' in keys
    #                 newKey = stateKey.replace(srcPrefix, layerKey + token + '{}.op.'.format(i))
    #                 j = 0
    #                 keyOldFormat = stateKey.replace(srcPrefix, dstPrefix.format(j))
    #                 foundMatch = False
    #                 while (keyOldFormat in layerKeys[layerKey]) and (not foundMatch):
    #                     if chckpntDict[keyOldFormat].size() == curStateDict[newKey].size():
    #                         newStateDict[newKey] = chckpntDict[keyOldFormat]
    #                         foundMatch = True
    #                     j += 1
    #                     keyOldFormat = stateKey.replace(srcPrefix, dstPrefix.format(j))
    #
    #     # load model weights
    #     self.load_state_dict(newStateDict)
    #
    #     # # load model alphas
    #     # if 'alphas' in checkpoint:
    #     #     chkpntAlphas = checkpoint['alphas']
    #     #     for i, l in enumerate(self.layersList):
    #     #         layerChkpntAlphas = chkpntAlphas[i]
    #     #         # init corresponding layers with its alphas
    #     #         jStart = 0
    #     #         jEnd = len(layerChkpntAlphas)
    #     #         while jEnd < len(l.alphas):
    #     #             l.alphas[jStart:jEnd].copy_(layerChkpntAlphas)
    #     #             jStart = jEnd
    #     #             jEnd += len(layerChkpntAlphas)
    #
    #     logger.info('Loaded model from [{}]'.format(path))
    #     logger.info('checkpoint validation accuracy:[{:.5f}]'.format(checkpoint['best_prec1']))
    #
    # def loadFromCheckpoint(self, path, logger, gpu):
    #     checkpoint = loadModel(path, map_location=lambda storage, loc: storage.cuda(gpu))
    #     # split state_dict keys by model layers
    #     layerKeys = {}  # collect ALL layer keys in state_dict
    #     layerOp0Keys = {}  # collect ONLY layer.ops.0. keys in state_dict, for duplication to the rest of layer ops
    #     token = '.ops.'
    #     for key in checkpoint['state_dict'].keys():
    #         prefix = key[:key.index(token)]
    #         isKey0 = prefix + token + '0.' in key
    #         if prefix in layerKeys:
    #             layerKeys[prefix].append(key)
    #             if isKey0: layerOp0Keys[prefix].append(key)
    #         else:
    #             layerKeys[prefix] = [key]
    #             if isKey0: layerOp0Keys[prefix] = [key]
    #     # duplicate state_dict values according to number of ops in each model layer
    #     for layerKey in layerKeys.keys():
    #         # init path to layer
    #         layerPath = [p for p in layerKey.split('.')]
    #         # get layer by walking through path
    #         layer = self
    #         for p in layerPath:
    #             layer = getattr(layer, p)
    #         # add missing layer operations to state_dict
    #         for stateKey in layerOp0Keys[layerKey]:
    #             for i in range(len(layer.ops)):
    #                 newKey = stateKey.replace(layerKey + token + '0.', layerKey + token + '{}.'.format(i))
    #                 if newKey not in layerKeys[layerKey]:
    #                     checkpoint['state_dict'][newKey] = checkpoint['state_dict'][stateKey]
    #
    #     # load model weights
    #     self.load_state_dict(checkpoint['state_dict'])
    #     # load model alphas
    #     # if 'alphas' in checkpoint:
    #     #     for i, l in enumerate(self.layersList):
    #     #         layerChkpntAlphas = checkpoint['alphas'][i]
    #     #         assert (layerChkpntAlphas.size() <= l.alphas.size())
    #     #         l.alphas = layerChkpntAlphas.expand_as(l.alphas)
    #
    #     # load nLayersQuantCompleted
    #     # if 'nLayersQuantCompleted' in checkpoint:
    #     #     self.nLayersQuantCompleted = checkpoint['nLayersQuantCompleted']
    #
    #     logger.info('Loaded model from [{}]'.format(path))
    #     logger.info('checkpoint validation accuracy:[{:.5f}]'.format(checkpoint['best_prec1']))
