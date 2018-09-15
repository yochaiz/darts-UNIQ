from collections import OrderedDict

from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, Module, AvgPool2d, Linear

from .ResNet import ResNet, BasicBlock
from cnn.MixedOp import MixedConvWithReLU


class BasicBlockFullPrecision(Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(BasicBlockFullPrecision, self).__init__()

        stride1 = stride if in_planes == out_planes else (stride + 1)

        self.block1 = Sequential(
            Conv2d(in_planes, out_planes, kernel_size, stride=stride1, padding=1, bias=False),
            BatchNorm2d(out_planes),
            ReLU(inplace=True)
        )

        self.block2 = Sequential(
            Conv2d(out_planes, out_planes, kernel_size, stride=stride, padding=1, bias=False),
            BatchNorm2d(out_planes)
        )

        self.relu = ReLU(inplace=True)

        self.downsample = Sequential(
            Conv2d(in_planes, out_planes, kernel_size=1, stride=stride1, bias=False),
            BatchNorm2d(out_planes)
        ) if in_planes != out_planes else None

    def forward(self, x):
        residual = self.downsample(x) if self.downsample else x

        out = self.block1(x)
        out = self.block2(out)
        out += residual
        out = self.relu(out)

        return out


class ThinResNet(ResNet):
    def __init__(self, args):
        super(ThinResNet, self).__init__(args)

    def initLayers(self, params):
        bitwidths, kernel_sizes = params

        # init layers (type, in_planes, out_planes)
        layersPlanes = [(MixedConvWithReLU, 3, 16, 32), (BasicBlock, 16, 16, [32]),
                        (BasicBlock, 16, 32, [32, 16]), (BasicBlock, 32, 64, [16, 8])]

        # create list of layers from layersPlanes
        # supports bitwidth as list of ints, i.e. same bitwidths to all layers
        # supports bitwidth as list of lists, i.e. specific bitwidths to each layer
        layers = [layerType(bitwidths if isinstance(bitwidths[0], int) else bitwidths[i],
                            in_planes, out_planes, kernel_sizes, 1, input_size)
                  for i, (layerType, in_planes, out_planes, input_size) in enumerate(layersPlanes)]

        i = 1
        for l in layers:
            setattr(self, 'block{}'.format(i), l)
            i += 1

        self.avgpool = AvgPool2d(8)
        self.fc = Linear(64, 10).cuda()

    def loadUNIQPre_trained(self, chckpntDict):
        def iterateKey(chckpntDict, map, key1, key2, dstKey):
            keyIdx = 0
            key = '{}.{}.{}'.format(key1, key2, keyIdx)
            while '{}.weight'.format(key) in chckpntDict:
                map[key] = dstKey.format(key1, key2, keyIdx)
                keyIdx += 1
                key = '{}.{}.{}'.format(key1, key2, keyIdx)

        def getNextblock(obj, key, blockNum):
            block = getattr(obj, key, None)
            return block, (blockNum + 1)

        # =============================================================================
        map = {
            'block1.0': 'block1.ops.0.op.0.0',
            'block1.1': 'block1.ops.0.op.0.1'
        }

        newStateDict = OrderedDict()

        blockNum = 2
        key1 = 'block{}'.format(blockNum)
        b, blockNum = getNextblock(self, key1, blockNum)
        while b:
            innerBlockNum = 1
            key2 = 'block{}'.format(innerBlockNum)
            c, innerBlockNum = getNextblock(b, key2, innerBlockNum)
            while c:
                iterateKey(chckpntDict, map, key1, key2, '{}.{}.ops.0.op.0.{}')
                key2 = 'block{}'.format(innerBlockNum)
                c, innerBlockNum = getNextblock(b, key2, innerBlockNum)

            # copy downsample
            key2 = 'downsample'
            c, innerBlockNum = getNextblock(b, key2, innerBlockNum)
            if c:
                iterateKey(chckpntDict, map, key1, key2, '{}.{}.ops.0.op.{}')

            key1 = 'block{}'.format(blockNum)
            b, blockNum = getNextblock(self, key1, blockNum)

        token = '.ops.'
        for key in chckpntDict.keys():
            if key.startswith('fc.'):
                newStateDict[key] = chckpntDict[key]
                continue

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

# # ====================================================
# # for training pre_trained, i.e. full precision
# # ====================================================
# def initLayers(self, params):
#     self.block1 = Sequential(
#         Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False), BatchNorm2d(16), ReLU(inplace=True)
#     )
#
#     layers = [
#         BasicBlockFullPrecision(16, 16, kernel_size=3, stride=1),
#         BasicBlockFullPrecision(16, 32, kernel_size=3, stride=1),
#         BasicBlockFullPrecision(32, 64, kernel_size=3, stride=1)
#     ]
#
#     i = 2
#     for l in layers:
#         setattr(self, 'block{}'.format(i), l)
#         i += 1
#
#     self.avgpool = AvgPool2d(8)
#     self.fc = Linear(64, 10)
#
# def switch_stage(self, logger=None):
#     pass
#
# def countBops(self):
#     return 10
#
# def loadUNIQPre_trained(self, path, logger, gpu):
#     checkpoint = loadModel(path, map_location=lambda storage, loc: storage.cuda(gpu))
#     chckpntDict = checkpoint['state_dict']
#
#     # load model weights
#     self.load_state_dict(chckpntDict)
#
#     logger.info('Loaded model from [{}]'.format(path))
#     logger.info('checkpoint validation accuracy:[{:.5f}]'.format(checkpoint['best_prec1']))
