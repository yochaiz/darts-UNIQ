from torch.nn import Module, AvgPool2d, MaxPool2d, Sequential, ReLU, Conv2d, BatchNorm2d
from torch import cat
import logging
from UNIQ.uniq import UNIQNet
from UNIQ.actquant import ActQuant
from abc import abstractmethod


class QuantizedOp(UNIQNet):
    # layers is an array of (layer, bitwidth) elements
    def __init__(self, C_in, C_out, kernel_size, stride, bitwidth=[], act_bitwidth=[]):
        super(QuantizedOp, self).__init__(quant=True, noise=True, quant_edges=True,
                                          act_quant=True, act_noise=False,
                                          step_setup=[1, 1],
                                          bitwidth=bitwidth, act_bitwidth=act_bitwidth)

        self.initLayers(C_in, C_out, kernel_size, stride)
        self.prepare_uniq()

    @abstractmethod
    def initLayers(self, C_in, C_out, kernel_size, stride):
        raise NotImplementedError('subclasses must override initLayers()!')

    def forward(self, x):
        return self.op(x)


class QuantizedConv(QuantizedOp):
    def __init__(self, C_in, C_out, kernel_size, stride, bitwidth, act_bitwidth=[]):
        super(QuantizedConv, self).__init__(C_in, C_out, kernel_size, stride, bitwidth=bitwidth, act_bitwidth=act_bitwidth)

    def initLayers(self, C_in, C_out, kernel_size, stride):
        self.op = Sequential(
            Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=1, bias=False),
            BatchNorm2d(C_out)
        )


class QuantizedConvWithReLU(QuantizedOp):
    def __init__(self, C_in, C_out, kernel_size, stride, bitwidth, act_bitwidth):
        super(QuantizedConvWithReLU, self).__init__(C_in, C_out, kernel_size, stride,
                                                    bitwidth=bitwidth, act_bitwidth=act_bitwidth)

    def initLayers(self, C_in, C_out, kernel_size, stride):
        self.op = Sequential(
            Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=1, bias=False),
            BatchNorm2d(C_out),
            ActQuant(quant=True, noise=False, bitwidth=self.act_bitwidth[0])
        )


class ReLUConvBN(Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = Sequential(
            ReLU(inplace=False),
            Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = Sequential(
            ReLU(inplace=False),
            Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in,
                   bias=False),
            Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = Sequential(
            ReLU(inplace=False),
            Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            BatchNorm2d(C_in, affine=affine),
            ReLU(inplace=False),
            Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = ReLU(inplace=False)
        self.conv_1 = Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


def createOpFunction(classRef, kernel_size, bitwidth, act_bitwidth):
    return lambda C, stride, affine: classRef(C, C, kernel_size, stride, bitwidth=bitwidth, act_bitwidth=act_bitwidth)


def originalOPS():
    OPS = {
        'none': lambda C, stride, affine: Zero(stride),
        'avg_pool_3x3': lambda C, stride, affine: AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
        'max_pool_3x3': lambda C, stride, affine: MaxPool2d(3, stride=stride, padding=1),
        'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
        'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
        'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
        'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
        'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
        'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
        'conv_7x1_1x7': lambda C, stride, affine: Sequential(
            ReLU(inplace=False),
            Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
            Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
            BatchNorm2d(C, affine=affine)
        ),
    }

    return OPS


def UNIQ_OPS(nBitsMin, nBitsMax):
    OPS = {
        'none': lambda C, stride, affine: Zero(stride),
        'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine)
    }
    # add quantized operations to OPS
    for kernel_size in [3]:
        for bitwidth in range(nBitsMin, nBitsMax + 1):
            OPS['conv_{}x{}_bitwidth_{}'.format(kernel_size, kernel_size, bitwidth)] = \
                createOpFunction(QuantizedConv, kernel_size, [bitwidth], [])

            for act_bitwidth in range(max(nBitsMin, bitwidth - 1), min(nBitsMax, bitwidth) + 1):
                OPS['conv_{}x{}_bitwidth_{}_act_bitwidth_{}'.format(kernel_size, kernel_size, bitwidth, act_bitwidth)] = \
                    createOpFunction(QuantizedConvWithReLU, kernel_size, [bitwidth], [act_bitwidth])

    return OPS


# OPS = originalOPS()
OPS = UNIQ_OPS(nBitsMin=1, nBitsMax=4)
