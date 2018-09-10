from collections import OrderedDict

from torch import load as loadModel
from torch.nn import Module, Conv2d, AvgPool2d

from UNIQ.actquant import ActQuant

from cnn.MixedOp import MixedConv, MixedConvWithReLU, MixedLinear
from cnn.models import BaseNet
from cnn.models.BaseNet import save_quant_state, restore_quant_state


class BasicBlock(Module):
    def __init__(self, bitwidths, in_planes, out_planes, kernel_size, stride):
        super(BasicBlock, self).__init__()

        stride1 = stride if in_planes == out_planes else (stride + 1)

        self.block1 = MixedConvWithReLU(bitwidths, in_planes, out_planes, kernel_size, stride1, useResidual=False)
        self.block2 = MixedConvWithReLU(bitwidths, out_planes, out_planes, kernel_size, stride, useResidual=True)

        self.downsample = MixedConv(bitwidths, in_planes, out_planes, [1], stride1) \
            if in_planes != out_planes else None

    def forward(self, x):
        residual = self.downsample(x) if self.downsample else x

        out = self.block1(x)
        out = self.block2(out, residual)

        return out


class ResNet(BaseNet):
    def __init__(self, lmbda, maxBops, bitwidths, kernel_sizes, bopsFuncKey, saveFolder=None):
        super(ResNet, self).__init__(lmbda=lmbda, maxBops=maxBops, initLayersParams=(bitwidths, kernel_sizes),
                                     bopsFuncKey=bopsFuncKey, saveFolder=saveFolder)

        # set noise=True for 1st layer
        if len(self.layersList) > 0:
            for op in self.layersList[0].ops:
                if op.quant:
                    op.noise = True

        # turn on grads in 1st layer alphas
        # self.layersList[0].alphas.requires_grad = True
        # self.update_learnable_alphas()
        # update model parameters() function
        self.parameters = self.getLearnableParams

    def initLayers(self, params):
        bitwidths, kernel_sizes = params

        self.block1 = MixedConvWithReLU(bitwidths, 3, 16, kernel_sizes, 1)

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
        self.fc = MixedLinear(bitwidths, 64, 10)

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

    # def _loss(self, input, target):
    #     logits = self(input)
    #     return self._criterion(logits, target, self.countBops())

    def update_learnable_alphas(self):
        self.learnable_alphas = [l.alphas for l in self.layersList if l.alphas.requires_grad is True]

    def getLearnableParams(self):
        return self.learnable_params

    def switch_stage(self, logger=None):
        if self.nLayersQuantCompleted < len(self.layersList):
            layer = self.layersList[self.nLayersQuantCompleted]
            assert (layer.alphas.requires_grad is False)

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

            if self.nLayersQuantCompleted < len(self.layersList):
                layer = self.layersList[self.nLayersQuantCompleted]
                # turn on noise in the new layer we want to quantize
                for op in layer.ops:
                    op.noise = True

                # # turn on layer alphas gradients
                # layer.alphas.requires_grad = True
                # # update learnable alphas
                # self.update_learnable_alphas()
                # self.update_learnable_alphas()

            if logger:
                logger.info(
                    'Switching stage, nLayersQuantCompleted:[{}], learnable_params:[{}], learnable_alphas size:[{}]'
                        .format(self.nLayersQuantCompleted, len(self.learnable_params), len(self.learnable_alphas)))

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
