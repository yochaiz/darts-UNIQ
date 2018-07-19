from torch.nn import Module, AvgPool2d, MaxPool2d, Sequential, ReLU, Conv2d, BatchNorm2d
from torch import cat

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
