from UNIQ.uniq import UNIQNet
from UNIQ.actquant import ActQuant
from UNIQ.flops_benchmark import count_flops

from torch import tensor, ones, zeros, sum, LongTensor, cat
from torch.nn import Module, ModuleList, Conv2d, BatchNorm2d, Sequential, Linear, ReLU
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from torch.autograd.function import Function

from math import floor
from random import randint
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
        self.hookHandlers = []

        self.op = op.cuda()
        self.prepare_uniq()

    def reset_flops_count(self):
        pass

    def compute_average_flops_cost(self):
        pass

    def stop_flops_count(self):
        pass

    def start_flops_count(self):
        pass

    def compute_average_bops_cost(self):
        pass

    def standardForward(self, x):
        return self.op(x)

    def residualForward(self, x, residual):
        out = self.op[0](x)
        out += residual
        out = self.op[1](out)

        return out


class Chooser(Function):
    chosen = None

    @staticmethod
    def forward(ctx, results, alpha):
        # choose alphas based on alphas distribution
        dist = Categorical(probs=alpha)
        chosen = dist.sample()

        # # choose alphas randomly, i.e. uniform distribution
        # chosen = tensor(randint(0, len(alpha) - 1))

        result = results.select(1, chosen)
        ctx.save_for_backward(LongTensor([results.shape]), result, chosen)
        Chooser.chosen = chosen
        return result

    @staticmethod
    def backward(ctx, grad_output):
        #  print('inside', grad_output.shape)
        assert (False)  # assure we do not use this backward
        results_shape, result, chosen = ctx.saved_tensors
        results_shape = tuple(results_shape.numpy().squeeze())
        grads_x = zeros(results_shape, device=grad_output.device, dtype=grad_output.dtype)
        grads_x.select(1, chosen).copy_(grad_output)
        grads_alpha = zeros((results_shape[1],), device=grad_output.device, dtype=grad_output.dtype)
        grads_alpha[chosen] = sum(grad_output * result)
        # print(chosen)
        # print('grads_inside', grads_alpha)
        # print('grad x: ', grads_x)
        return grads_x, grads_alpha


class MixedOp(Module):
    def __init__(self, bitwidths, params, coutBopsParams):
        super(MixedOp, self).__init__()

        # assure bitwidths is a list of integers
        if isinstance(bitwidths[0], list):
            bitwidths = bitwidths[0]
        # init operations mixture
        self.ops = self.initOps(bitwidths, params)
        # init operations alphas (weights)
        value = 1.0 / self.numOfOps()
        self.alphas = tensor((ones(self.numOfOps()) * value).cuda(), requires_grad=False)

        # self.curr_alpha_idx = 0 if self.numOfOps() == 1 else None
        self.curr_alpha_idx = 0
        self.forward = self.trainForward

        # init bops for operation
        self.bops = self.countBops(coutBopsParams)
        self.countBops = None

    @abstractmethod
    def initOps(self):
        raise NotImplementedError('subclasses must override initOps()!')

    @abstractmethod
    def countBops(self, params):
        raise NotImplementedError('subclasses must override countBops()!')

    # select random alpha
    def chooseRandomPath(self):
        self.curr_alpha_idx = randint(0, len(self.alphas) - 1)

    # select alpha based on alphas distribution
    def choosePathByAlphas(self):
        dist = Categorical(logits=self.alphas)
        chosen = dist.sample()
        self.curr_alpha_idx = chosen.item()

    def trainMode(self):
        self.forward = self.trainForward

    def evalMode(self):
        # update current alpha to max alpha value
        dist = F.softmax(self.alphas, dim=-1)
        self.curr_alpha_idx = dist.argmax().item()
        # update forward function
        self.forward = self.evalForward

    # select op index based on desired bitwidth
    def uniformMode(self, bitwidth):
        for i, op in enumerate(self.ops):
            if bitwidth in op.bitwidth:
                self.curr_alpha_idx = i
                break

        # update forward function
        self.forward = self.evalForward

    def trainForward(self, x):
        if self.alphas.requires_grad:
            results = [op(x).unsqueeze(1) for op in self.ops]
            probs = F.softmax(self.alphas, 0)
            # print('alpha: ', self.alphas)
            result = Chooser.apply(cat(results, 1), probs)
            self.curr_alpha_idx = Chooser.chosen.item()
            # print(self.ops[self.curr_alpha_idx])
        else:
            result = self.ops[self.curr_alpha_idx](x)
        return result

    def evalForward(self, x):
        return self.ops[self.curr_alpha_idx](x)

    def numOfOps(self):
        return len(self.ops)

    def getBops(self):
        return self.bops


class MixedLinear(MixedOp):
    def __init__(self, bitwidths, in_features, out_features):
        params = in_features, out_features

        super(MixedLinear, self).__init__(bitwidths, params, coutBopsParams=in_features)

    def initOps(self, bitwidths, params):
        in_features, out_features = params

        ops = ModuleList()
        for bitwidth in bitwidths:
            op = Linear(in_features, out_features)
            op = QuantizedOp(op, bitwidth=[bitwidth], act_bitwidth=[])
            ops.append(op)

        return ops

    def countBops(self, input_size):
        return [count_flops(op, input_size, 1) for op in self.ops]


class MixedConv(MixedOp):
    def __init__(self, bitwidths, in_planes, out_planes, kernel_size, stride, input_size):
        assert (isinstance(kernel_size, list))
        params = in_planes, out_planes, kernel_size, stride
        coutBopsParams = input_size, in_planes

        super(MixedConv, self).__init__(bitwidths, params, coutBopsParams)

        self.in_planes = in_planes

    def initOps(self, bitwidths, params):
        in_planes, out_planes, kernel_size, stride = params

        ops = ModuleList()
        for bitwidth, _ in bitwidths:
            for ker_sz in kernel_size:
                op = Sequential(
                    Conv2d(in_planes, out_planes, kernel_size=ker_sz,
                           stride=stride, padding=floor(ker_sz / 2), bias=False),
                    BatchNorm2d(out_planes)
                )
                op = QuantizedOp(op, bitwidth=[bitwidth], act_bitwidth=[])
                ops.append(op)

        return ops

    def countBops(self, coutBopsParams):
        input_size, in_planes = coutBopsParams
        return [count_flops(op, input_size, in_planes) for op in self.ops]


class MixedConvWithReLU(MixedOp):
    def __init__(self, bitwidths, in_planes, out_planes, kernel_size, stride, input_size, useResidual=False):
        assert (isinstance(kernel_size, list))
        params = in_planes, out_planes, kernel_size, stride, useResidual
        coutBopsParams = input_size, in_planes

        if useResidual:
            self.trainForward = self.trainResidualForward
            self.evalForward = self.evalResidualForward

        super(MixedConvWithReLU, self).__init__(bitwidths, params, coutBopsParams)

        self.in_planes = in_planes

    def initOps(self, bitwidths, params):
        in_planes, out_planes, kernel_size, stride, useResidual = params

        ops = ModuleList()
        for bitwidth, act_bitwidth in bitwidths:
            for ker_sz in kernel_size:
                op = Sequential(
                    Sequential(
                        Conv2d(in_planes, out_planes, kernel_size=ker_sz,
                               stride=stride, padding=floor(ker_sz / 2), bias=False),
                        BatchNorm2d(out_planes)
                    ),
                    ActQuant(quant=True, noise=False, bitwidth=act_bitwidth)
                    # ReLU(inplace=True)
                )
                op = QuantizedOp(op, bitwidth=[bitwidth], act_bitwidth=[act_bitwidth], useResidual=useResidual)
                ops.append(op)

        return ops

    def countBops(self, coutBopsParams):
        input_size, in_planes = coutBopsParams
        return [count_flops(op, input_size, in_planes) for op in self.ops]

    def trainResidualForward(self, x, residual):
        if self.alphas.requires_grad:
            results = [op(x, residual).unsqueeze(1) for op in self.ops]
            probs = F.softmax(self.alphas, 0)
            result = Chooser.apply(cat(results, 1), probs)
            self.curr_alpha_idx = Chooser.chosen.item()
        else:
            result = self.ops[self.curr_alpha_idx](x, residual)
        return result

    def evalResidualForward(self, x, residual):
        return self.ops[self.curr_alpha_idx](x, residual)
