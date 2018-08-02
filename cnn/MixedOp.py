from UNIQ.uniq import UNIQNet
from UNIQ.actquant import ActQuant
from UNIQ.flops_benchmark import count_flops
from torch import tensor, ones
from torch.nn import Module, ModuleList, Conv2d, BatchNorm2d, Sequential, Linear, ReLU
import torch.nn.functional as F
from math import floor
from abc import abstractmethod


class QuantizedOp(UNIQNet):
    def __init__(self, op, bitwidth=[], act_bitwidth=[], useResidual=False):
        # noise=False because we want to noise only specific layer in the entire (ResNet) model
        super(QuantizedOp, self).__init__(quant=True, noise=False, quant_edges=True,
                                          act_quant=True, act_noise=False,
                                          step_setup=[1, 1],
                                          bitwidth=bitwidth, act_bitwidth=act_bitwidth)

        self.useResidual = useResidual
        self.forward = self.residualForward if useResidual else self.standardForward

        self.op = op.cuda()
        self.prepare_uniq()

    def standardForward(self, x):
        return self.op(x)

    def residualForward(self, x, residual):
        out = self.op[0](x)
        out += residual
        out = self.op[1](out)

        return out


class MixedOp(Module):
    def __init__(self):
        super(MixedOp, self).__init__()

        # init operations mixture
        self.ops = self.initOps()
        # init operations alphas (weights)
        # self.alphas = tensor(randn(self.numOfOps()).cuda(), requires_grad=True)
        value = 1.0 / self.numOfOps()
        self.alphas = tensor((ones(self.numOfOps()) * value).cuda(), requires_grad=True)
        # init bops for operation
        self.bops = self.countBops()

    @abstractmethod
    def initOps(self):
        raise NotImplementedError('subclasses must override initOps()!')

    @abstractmethod
    def countBops(self):
        raise NotImplementedError('subclasses must override countBops()!')

    def forward(self, x):
        weights = F.softmax(self.alphas, dim=-1)
        return sum(w * op(x) for w, op in zip(weights, self.ops))

    def numOfOps(self):
        return len(self.ops)

    def getBops(self):
        return self.bops


class MixedLinear(MixedOp):
    def __init__(self, bitwidths, in_features, out_features):
        self.bitwidths = bitwidths
        self.in_features = in_features
        self.out_features = out_features

        super(MixedLinear, self).__init__()

    def initOps(self):
        ops = ModuleList()
        for bitwidth in self.bitwidths:
            op = Linear(self.in_features, self.out_features)
            op = QuantizedOp(op, bitwidth=[bitwidth], act_bitwidth=[])
            ops.append(op)

        return ops

    def countBops(self):
        return [count_flops(op, self.in_features, 1) for op in self.ops]


class MixedConv(MixedOp):
    def __init__(self, bitwidths, in_planes, out_planes, kernel_size, stride):
        assert (isinstance(kernel_size, list))
        self.bitwidths = bitwidths
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride

        super(MixedConv, self).__init__()

    def initOps(self):
        ops = ModuleList()
        for bitwidth in self.bitwidths:
            for ker_sz in self.kernel_size:
                op = Sequential(
                    Conv2d(self.in_planes, self.out_planes, kernel_size=ker_sz,
                           stride=self.stride, padding=floor(ker_sz / 2), bias=False),
                    BatchNorm2d(self.out_planes)
                )
                op = QuantizedOp(op, bitwidth=[bitwidth], act_bitwidth=[])
                ops.append(op)

        return ops

    def countBops(self):
        return [count_flops(op, 1, self.in_planes) for op in self.ops]


class MixedConvWithReLU(MixedOp):
    def __init__(self, bitwidths, in_planes, out_planes, kernel_size, stride, useResidual=False):
        assert (isinstance(kernel_size, list))
        self.bitwidths = bitwidths
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.useResidual = useResidual

        super(MixedConvWithReLU, self).__init__()

        if useResidual:
            self.forward = self.residualForward

    def initOps(self):
        ops = ModuleList()
        for bitwidth in self.bitwidths:
            for act_bitwidth in self.bitwidths:
                for ker_sz in self.kernel_size:
                    op = Sequential(
                        Sequential(
                            Conv2d(self.in_planes, self.out_planes, kernel_size=ker_sz,
                                   stride=self.stride, padding=floor(ker_sz / 2), bias=False),
                            BatchNorm2d(self.out_planes)
                        ),
                        ActQuant(quant=True, noise=False, bitwidth=act_bitwidth)
                        # ReLU(inplace=True)
                    )
                    op = QuantizedOp(op, bitwidth=[bitwidth], act_bitwidth=[act_bitwidth], useResidual=self.useResidual)
                    ops.append(op)

        return ops

    def countBops(self):
        return [count_flops(op, 1, self.in_planes) for op in self.ops]

    def residualForward(self, x, residual):
        weights = F.softmax(self.alphas, dim=-1)
        return sum(w * op(x, residual) for w, op in zip(weights, self.ops))

    # # for standard op, without QuantizedOp wrapping
    # def residualForward(self, x, residual):
    #     weights = F.softmax(self.alphas, dim=-1)
    #     opsForward = []
    #     for op in self.ops:
    #         out = op[0](x)
    #         out += residual
    #         out = op[1](out)
    #         opsForward.append(out)
    #
    #     return sum(w * p for w, p in zip(weights, opsForward))
