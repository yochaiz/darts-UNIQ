# ResNetFLQ - ResNet First Layer Quantization
# implementation of ResNet where only the 1st layer is quantized, the following layers are in full-precision

from .ResNet import ResNet, ModuleList
from .ThinResNet import BasicBlockFullPrecision
from cnn.MixedFilter import Block, count_flops, ceil, log2


class BlocksFullPrecision(Block):
    def __init__(self, bitwidths, in_planes_, out_planes_, kernel_size, stride, input_size, prevLayer):
        super(BlocksFullPrecision, self).__init__()

        self.layers = self.initLayers(in_planes_, kernel_size[0], stride)
        # init bops map
        self.bopsMap = {}

    def initLayers(self, in_planes_, kernel_size, stride):
        layers = ModuleList()
        # init BasicBlocks in_planes & out_planes
        layersPlanes = [(in_planes_, 16, [32]), (16, 16, [32]), (16, 16, [32]),
                        (16, 32, [32, 16]), (32, 32, [16]), (32, 32, [16]),
                        (32, 64, [16, 8]), (64, 64, [8]), (64, 64, [8])]
        # build layers
        for in_planes, out_planes, input_size in layersPlanes:
            layers.append(BasicBlockFullPrecision(in_planes, out_planes, kernel_size, stride))

        return layers

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)

        return out

    def getLayers(self):
        return []

    # input_bitwidth is a list of bitwidth per feature map
    def getBops(self, input_bitwidth):
        return 0

    def getCurrentOutputBitwidth(self):
        return None

    def choosePathByAlphas(self):
        pass

    def numOfOps(self):
        return 1


class ResNetFLQ(ResNet):
    def __init__(self, args):
        super(ResNetFLQ, self).__init__(args)

    def initLayersPlanes(self):
        return [(self.createMixedLayer, 3, 16, 32), (BlocksFullPrecision, 16, 16, 32)]

    def buildStateDictMap(self, chckpntDict):
        map = {}
        map['conv1'] = 'layers.0.ops.0.0.op.0'
        map['bn1'] = 'layers.0.ops.0.0.op.1'
        map['relu'] = 'layers.0.ops.0.0.op.2'

        actQuantKeys = ['running_mean', 'running_std', 'clamp_val']
        convNICEkeys = ['initial_clamp_value', 'layer_basis', 'layer_b']

        layersNumberMap = [(1, 0, 0), (1, 1, 1), (1, 2, 2), (2, 1, 4), (2, 2, 5), (3, 1, 7), (3, 2, 8)]
        for n1, n2, m in layersNumberMap:
            map['layer{}.{}.conv1'.format(n1, n2)] = 'layers.1.layers.{}.block1.0'.format(m)
            map['layer{}.{}.bn1'.format(n1, n2)] = 'layers.1.layers.{}.block1.1'.format(m)
            map['layer{}.{}.conv2'.format(n1, n2)] = 'layers.1.layers.{}.block2.0'.format(m)
            map['layer{}.{}.bn2'.format(n1, n2)] = 'layers.1.layers.{}.block2.1'.format(m)

            # remove ActQuant keys from chckpntDict
            for suffix in actQuantKeys:
                del chckpntDict['layer{}.{}.relu1.{}'.format(n1, n2, suffix)]
                del chckpntDict['layer{}.{}.relu2.{}'.format(n1, n2, suffix)]

            # remove Conv2d NICE keys from chckpntDict
            for suffix in convNICEkeys:
                del chckpntDict['layer{}.{}.conv1.{}'.format(n1, n2, suffix)]
                del chckpntDict['layer{}.{}.conv2.{}'.format(n1, n2, suffix)]

        downsampleLayersMap = [(2, 0, 3), (3, 0, 6)]
        for n1, n2, m in downsampleLayersMap:
            map['layer{}.{}.conv1'.format(n1, n2)] = 'layers.1.layers.{}.block1.0'.format(m)
            map['layer{}.{}.bn1'.format(n1, n2)] = 'layers.1.layers.{}.block1.1'.format(m)
            map['layer{}.{}.conv2'.format(n1, n2)] = 'layers.1.layers.{}.block2.0'.format(m)
            map['layer{}.{}.bn2'.format(n1, n2)] = 'layers.1.layers.{}.block2.1'.format(m)
            map['layer{}.{}.downsample.0'.format(n1, n2)] = 'layers.1.layers.{}.downsample.0'.format(m)
            map['layer{}.{}.downsample.1'.format(n1, n2)] = 'layers.1.layers.{}.downsample.1'.format(m)

            # remove ActQuant keys from chckpntDict
            for suffix in actQuantKeys:
                del chckpntDict['layer{}.{}.relu1.{}'.format(n1, n2, suffix)]
                del chckpntDict['layer{}.{}.relu2.{}'.format(n1, n2, suffix)]

            # remove Conv2d NICE keys from chckpntDict
            for suffix in convNICEkeys:
                del chckpntDict['layer{}.{}.conv1.{}'.format(n1, n2, suffix)]
                del chckpntDict['layer{}.{}.conv2.{}'.format(n1, n2, suffix)]
                del chckpntDict['layer{}.{}.downsample.0.{}'.format(n1, n2, suffix)]

        map['fc'] = 'fc'

        return map
