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


class Block(Module):
    @abstractmethod
    def getBops(self, input_bitwidth):
        raise NotImplementedError('subclasses must override getBops()!')

    @abstractmethod
    def getCurrentOutputBitwidth(self):
        raise NotImplementedError('subclasses must override getCurrentOutputBitwidth()!')

    @abstractmethod
    def getOutputBitwidthList(self):
        raise NotImplementedError('subclasses must override getOutputBitwidthList()!')

    @abstractmethod
    def chooseRandomPath(self):
        raise NotImplementedError('subclasses must override chooseRandomPath()!')

    @abstractmethod
    def choosePathByAlphas(self):
        raise NotImplementedError('subclasses must override choosePathByAlphas()!')

    @abstractmethod
    def evalMode(self):
        raise NotImplementedError('subclasses must override evalMode()!')

    @abstractmethod
    def numOfOps(self):
        raise NotImplementedError('subclasses must override numOfOps()!')

    @abstractmethod
    def outputLayer(self):
        raise NotImplementedError('subclasses must override outputLayer()!')


# class Chooser(Function):
#     chosen = None
#
#     @staticmethod
#     def forward(ctx, results, alpha):
#         # choose alphas based on alphas distribution
#         dist = Categorical(probs=alpha)
#         chosen = dist.sample()
#
#         # # choose alphas randomly, i.e. uniform distribution
#         # chosen = tensor(randint(0, len(alpha) - 1))
#
#         result = results.select(1, chosen)
#         ctx.save_for_backward(LongTensor([results.shape]), result, chosen)
#         Chooser.chosen = chosen
#         return result
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         #  print('inside', grad_output.shape)
#         assert (False)  # assure we do not use this backward
#         results_shape, result, chosen = ctx.saved_tensors
#         results_shape = tuple(results_shape.numpy().squeeze())
#         grads_x = zeros(results_shape, device=grad_output.device, dtype=grad_output.dtype)
#         grads_x.select(1, chosen).copy_(grad_output)
#         grads_alpha = zeros((results_shape[1],), device=grad_output.device, dtype=grad_output.dtype)
#         grads_alpha[chosen] = sum(grad_output * result)
#         # print(chosen)
#         # print('grads_inside', grads_alpha)
#         # print('grad x: ', grads_x)
#         return grads_x, grads_alpha


class MixedOp(Block):
    def __init__(self, bitwidths, input_bitwidth, params, coutBopsParams, prevLayer, nOpsCopies=1):
        super(MixedOp, self).__init__()

        # assure bitwidths is a list of integers
        if isinstance(bitwidths[0], list):
            bitwidths = bitwidths[0]
        # init operations mixture
        self.ops = ModuleList()
        for _ in range(nOpsCopies):
            self.ops.append(self.initOps(bitwidths, params))

        # init list of all operations (including copies) as single long list
        # for cases we have to modify all ops
        self.opsList = []
        for ops in self.ops:
            for op in ops:
                self.opsList.append(op)

        # init ops forward counters
        self.opsForwardCounters = self.buildOpsForwardCounters()

        # init operations alphas (weights)
        value = 1.0 / self.numOfOps()
        self.alphas = tensor((ones(self.numOfOps()) * value).cuda(), requires_grad=False)

        self.curr_alpha_idx = 0
        # init previous layer
        assert ((prevLayer is None) or (isinstance(prevLayer, MixedOp)))
        self.prevLayer = prevLayer

        # init bops for operation
        self.bops = self.buildBopsMap(bitwidths, input_bitwidth, params, coutBopsParams)
        self.countBops = None

    @abstractmethod
    def initOps(self, bitwidths, params):
        raise NotImplementedError('subclasses must override initOps()!')

    @abstractmethod
    def countOpsBops(self, ops, coutBopsParams):
        raise NotImplementedError('subclasses must override countBops()!')

    @abstractmethod
    def getBitwidth(self):
        raise NotImplementedError('subclasses must override getBitwidth()!')

    @abstractmethod
    def getCurrentOutputBitwidth(self):
        raise NotImplementedError('subclasses must override getCurrentOutputBitwidth()!')

    @abstractmethod
    def getOutputBitwidthList(self):
        raise NotImplementedError('subclasses must override getOutputBitwidthList()!')

    def buildOpsForwardCounters(self):
        return [[0] * len(ops) for ops in self.ops]

    def resetOpsForwardCounters(self):
        self.opsForwardCounters = self.buildOpsForwardCounters()

    def buildBopsMap(self, bitwidths, input_bitwidth, initOpsParams, coutBopsParams):
        # init bops map
        bops = {}

        # build ops
        for in_bitwidth in input_bitwidth:
            bops_bitwidth = [(b, in_bitwidth) for b, _ in bitwidths]
            ops = self.initOps(bops_bitwidth, initOpsParams)
            bops[in_bitwidth] = self.countOpsBops(ops, coutBopsParams)

        return bops

    def outputLayer(self):
        return self

    def getBops(self, input_bitwidth):
        return self.bops[input_bitwidth][self.curr_alpha_idx]

    # select random alpha
    def chooseRandomPath(self):
        self.curr_alpha_idx = randint(0, len(self.alphas) - 1)

    # select alpha based on alphas distribution
    def choosePathByAlphas(self):
        dist = Categorical(logits=self.alphas)
        chosen = dist.sample()
        self.curr_alpha_idx = chosen.item()

    def evalMode(self):
        # update current alpha to max alpha value
        dist = F.softmax(self.alphas, dim=-1)
        self.curr_alpha_idx = dist.argmax().item()

    # select op index based on desired bitwidth
    def uniformMode(self, bitwidth):
        # it doesn't matter which copy of ops we take, the attributes are the same in all copies
        for i, op in enumerate(self.ops[0]):
            if bitwidth in op.bitwidth:
                self.curr_alpha_idx = i
                break

    def forward(self, x):
        prev_alpha_idx = self.prevLayer.curr_alpha_idx if self.prevLayer else 0
        self.opsForwardCounters[prev_alpha_idx][self.curr_alpha_idx] += 1
        return self.ops[prev_alpha_idx][self.curr_alpha_idx](x)

    # def trainForward(self, x):
    #     if self.alphas.requires_grad:
    #         probs = F.softmax(self.alphas, 0)
    #         # choose alphas based on alphas distribution
    #         dist = Categorical(probs=probs)
    #         self.curr_alpha_idx = dist.sample().item()
    #         result = self.ops[self.prev_alpha_idx][self.curr_alpha_idx](x)
    #
    #         # results = [op(x).unsqueeze(1) for op in self.ops]
    #         # probs = F.softmax(self.alphas, 0)
    #         # # print('alpha: ', self.alphas)
    #
    #         # result = Chooser.apply(cat(results, 1), probs)
    #         # self.curr_alpha_idx = Chooser.chosen.item()
    #         # print(self.ops[self.curr_alpha_idx])
    #     else:
    #         result = self.ops[self.prev_alpha_idx][self.curr_alpha_idx](x)
    #     return result
    #
    # def evalForward(self, x):
    #     return self.ops[self.prev_alpha_idx][self.curr_alpha_idx](x)

    def numOfOps(self):
        # it doesn't matter which copy of ops we take, length is the same in all copies
        return len(self.ops[0])

    def nOpsCopies(self):
        return len(self.ops)

    def getOps(self):
        return self.opsList


class MixedLinear(MixedOp):
    def __init__(self, bitwidths, in_features, out_features, input_bitwidth):
        params = in_features, out_features

        super(MixedLinear, self).__init__(bitwidths, input_bitwidth, params, coutBopsParams=in_features)

    def initOps(self, bitwidths, params):
        in_features, out_features = params

        ops = ModuleList()
        for bitwidth in bitwidths:
            op = Linear(in_features, out_features)
            op = QuantizedOp(op, bitwidth=[bitwidth], act_bitwidth=[])
            ops.append(op)

        return ops

    def countOpsBops(self, ops, input_size):
        return [count_flops(op, input_size, 1) for op in ops]


class MixedConv(MixedOp):
    def __init__(self, bitwidths, in_planes, out_planes, kernel_size, stride, input_size, input_bitwidth, prevLayer,
                 nOpsCopies=1):
        assert (isinstance(kernel_size, list))
        params = in_planes, out_planes, kernel_size, stride
        coutBopsParams = input_size, in_planes

        super(MixedConv, self).__init__(bitwidths, input_bitwidth, params, coutBopsParams, prevLayer, nOpsCopies)

        self.in_planes = in_planes

    def initOps(self, bitwidths, params):
        in_planes, out_planes, kernel_size, stride = params

        ops = ModuleList()
        for bitwidth, act_bitwidth in bitwidths:
            for ker_sz in kernel_size:
                op = Sequential(
                    Conv2d(in_planes, out_planes, kernel_size=ker_sz,
                           stride=stride, padding=floor(ker_sz / 2), bias=False),
                    BatchNorm2d(out_planes)
                )
                op = QuantizedOp(op, bitwidth=[bitwidth], act_bitwidth=[] if act_bitwidth is None else [act_bitwidth])
                ops.append(op)

        return ops

    def countOpsBops(self, ops, coutBopsParams):
        input_size, in_planes = coutBopsParams
        return [count_flops(op, input_size, in_planes) for op in ops]

    def getBitwidth(self):
        # it doesn't matter which copy of ops we take, the attributes are the same in all copies
        return self.ops[0][self.curr_alpha_idx].bitwidth[0], None


class MixedConvWithReLU(MixedOp):
    def __init__(self, bitwidths, in_planes, out_planes, kernel_size, stride,
                 input_size, input_bitwidth, prevLayer, nOpsCopies=1, useResidual=False):
        assert (isinstance(kernel_size, list))
        params = in_planes, out_planes, kernel_size, stride, useResidual
        coutBopsParams = in_planes, input_size

        if useResidual:
            self.forward = self.residualForward
            # self.trainForward = self.trainResidualForward
            # self.evalForward = self.evalResidualForward

        super(MixedConvWithReLU, self).__init__(bitwidths, input_bitwidth, params, coutBopsParams, prevLayer,
                                                nOpsCopies)

        self.in_planes = in_planes

        # init output bitwidths list
        self.outputBitwidth = []
        # it doesn't matter which copy of ops we take, the attributes are the same in all copies
        for op in self.ops[0]:
            self.outputBitwidth.extend(op.act_bitwidth)

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

    def countOpsBops(self, ops, coutBopsParams):
        in_planes, input_size = coutBopsParams
        return [count_flops(op, input_size, in_planes) for op in ops]

    def getBitwidth(self):
        # it doesn't matter which copy of ops we take, the attributes are the same in all copies
        op = self.ops[0][self.curr_alpha_idx]
        return op.bitwidth[0], op.act_bitwidth[0]

    def getCurrentOutputBitwidth(self):
        return self.outputBitwidth[self.curr_alpha_idx]

    def getOutputBitwidthList(self):
        return self.outputBitwidth

    def residualForward(self, x, residual):
        prev_alpha_idx = self.prevLayer.curr_alpha_idx if self.prevLayer else 0
        self.opsForwardCounters[prev_alpha_idx][self.curr_alpha_idx] += 1
        return self.ops[prev_alpha_idx][self.curr_alpha_idx](x, residual)

    # def trainResidualForward(self, x, residual):
    #     if self.alphas.requires_grad:
    #         probs = F.softmax(self.alphas, 0)
    #         # choose alphas based on alphas distribution
    #         dist = Categorical(probs=probs)
    #         self.curr_alpha_idx = dist.sample().item()
    #         result = self.ops[self.prev_alpha_idx][self.curr_alpha_idx](x, residual)
    #
    #         # results = [op(x, residual).unsqueeze(1) for op in self.ops]
    #         # probs = F.softmax(self.alphas, 0)
    #         # result = Chooser.apply(cat(results, 1), probs)
    #         # self.curr_alpha_idx = Chooser.chosen.item()
    #     else:
    #         result = self.ops[self.prev_alpha_idx][self.curr_alpha_idx](x, residual)
    #     return result
    #
    # def evalResidualForward(self, x, residual):
    #     return self.ops[self.prev_alpha_idx][self.curr_alpha_idx](x, residual)
