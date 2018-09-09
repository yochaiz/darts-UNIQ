from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, Module, AvgPool2d, Linear
from torch import load as loadModel

from .ResNet import ResNet


class BasicBlock(Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(BasicBlock, self).__init__()

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
    def __init__(self, lmbda, maxBops, bitwidths, kernel_sizes, bopsFuncKey, saveFolder=None):
        super(ThinResNet, self).__init__(lmbda, maxBops, bitwidths, kernel_sizes, bopsFuncKey, saveFolder)

    def initLayers(self, params):
        self.block1 = Sequential(
            Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False), BatchNorm2d(16), ReLU(inplace=True)
        )

        layers = [
            BasicBlock(16, 16, kernel_size=3, stride=1),
            BasicBlock(16, 32, kernel_size=3, stride=1),
            BasicBlock(32, 64, kernel_size=3, stride=1)
        ]

        i = 2
        for l in layers:
            setattr(self, 'block{}'.format(i), l)
            i += 1

        self.avgpool = AvgPool2d(8)
        self.fc = Linear(64, 10)

    def switch_stage(self, logger=None):
        pass

    def countBops(self):
        return 10

    def loadUNIQPre_trained(self, path, logger, gpu):
        checkpoint = loadModel(path, map_location=lambda storage, loc: storage.cuda(gpu))
        chckpntDict = checkpoint['state_dict']

        # load model weights
        self.load_state_dict(chckpntDict)

        logger.info('Loaded model from [{}]'.format(path))
        logger.info('checkpoint validation accuracy:[{:.5f}]'.format(checkpoint['best_prec1']))
