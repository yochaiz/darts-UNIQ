from UNIQ.uniq import UNIQNet
from UNIQ.actquant import ActQuant
from UNIQ.flops_benchmark import count_flops

from torch.nn import ModuleList, Conv2d, Sequential, Linear
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

from cnn.block import Block

from math import floor
from random import randint
from abc import abstractmethod


class QuantizedOp(UNIQNet):
    def __init__(self, op, bitwidth=[], act_bitwidth=[]):
        # noise=False because we want to noise only specific layer in the entire (ResNet) model
        super(QuantizedOp, self).__init__(quant=True, noise=False, quant_edges=True, act_quant=True, act_noise=False,
                                          step_setup=[1, 1], bitwidth=bitwidth, act_bitwidth=act_bitwidth)

        # self.useResidual = useResidual
        # self.forward = self.residualForward if useResidual else self.standardForward
        # self.hookHandlers = []

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

    def forward(self, x):
        return self.op(x)

    # def standardForward(self, x):
    #     return self.op(x)
    #
    # def residualForward(self, x, residual):
    #     out = self.op[0](x)
    #     out += residual
    #     out = self.op[1](out)
    #
    #     return out


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

def preForward(self, _):
    # update previous layer index
    prevLayer = self.prevLayer[0]
    self.prev_alpha_idx = prevLayer.curr_alpha_idx if prevLayer else 0
    # update forward counter
    self.opsForwardCounters[self.prev_alpha_idx][self.curr_alpha_idx] += 1
    # get current op
    op = self.ops[self.prev_alpha_idx][self.curr_alpha_idx]
    # check if we need to add noise
    if op.noise is True:
        op.add_noise()


def postForward(self, _, __):
    op = self.ops[self.prev_alpha_idx][self.curr_alpha_idx]
    # check if we need to remove noise
    if op.noise is True:
        op.restore_state()


class MixedFilter(Block):
    def __init__(self, bitwidths, params, coutBopsParams, prevLayer):
        super(MixedFilter, self).__init__()

        # save params for counting bops
        self.initOpsParams = params
        self.coutBopsParams = coutBopsParams

        # assure bitwidths is a list of integers
        if isinstance(bitwidths[0], list):
            bitwidths = bitwidths[0]

        # init previous layer, put it in a list, in order to ignore it as a model in this instance
        prevLayer = None
        assert ((prevLayer is None) or (isinstance(prevLayer, MixedFilter)))
        self.prevLayer = [prevLayer]

        # init operations mixture
        self.ops = ModuleList()
        # ops must have at least one copy
        self.ops.append(self.initOps(bitwidths, params))
        # add more copies if prevLayer exists
        if prevLayer:
            for _ in range(prevLayer.numOfOps() - 1):
                self.ops.append(self.initOps(bitwidths, params))

        # init list of all operations (including copies) as single long list
        # for cases we have to modify all ops
        self.opsList = []
        for ops in self.ops:
            for op in ops:
                self.opsList.append(op)

        # init ops forward counters
        self.opsForwardCounters = self.buildOpsForwardCounters()

        self.curr_alpha_idx = 0
        self.prev_alpha_idx = 0
        # init counter for number of consecutive times optimal alpha reached optimal probability limit
        self.optLimitCounter = 0

        # assign pre & post forward hooks
        self.register_forward_pre_hook(preForward)
        self.register_forward_hook(postForward)

    @abstractmethod
    def initOps(self, bitwidths, params):
        raise NotImplementedError('subclasses must override initOps()!')

    @abstractmethod
    def countOpsBops(self, ops, coutBopsParams):
        raise NotImplementedError('subclasses must override countBops()!')

    @abstractmethod
    def getOpBitwidth(self, op):
        raise NotImplementedError('subclasses must override getOpBitwidth()!')

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

    def outputLayer(self):
        return self

    # input_bitwidth is a list of input feature maps bitwidth
    def getBops(self, input_bitwidth):
        # get filter current bitwidth
        bitwidth, _ = self.getCurrentBitwidth()
        # create (bitwidth, act_bitwidth) tuple for count_flops(), act_bitwidth is a list of input feature maps bitwidth
        bops_bitwidth = [(bitwidth, input_bitwidth)]
        # generate op with this bitwidth
        ops = self.initOps(bops_bitwidth, self.initOpsParams)
        op = ops[0]
        # count op bops
        bops = self.countOpsBops(op, self.coutBopsParams)

        return bops

    # return list of tuples of all filter bitwidths
    def getAllBitwidths(self):
        return [self.getOpBitwidth(op) for op in self.ops[0]]

    # returns current op bitwidth
    def getCurrentBitwidth(self):
        # it doesn't matter which copy of ops we take, the attributes are the same in all copies
        return self.getOpBitwidth(self.ops[0][self.curr_alpha_idx])

    # from UNIQ.quantize import check_quantization
    # op = self.ops[0][self.curr_alpha_idx]
    # v1 = check_quantization(op.op[0].weight)
    # v2 = 2 ** (op.bitwidth[0])
    # assert (v1 <= v2)

    # # select random alpha
    # def chooseRandomPath(self):
    #     self.curr_alpha_idx = randint(0, len(self.alphas) - 1)

    # # select alpha based on alphas distribution
    # def choosePathByAlphas(self):
    #     dist = Categorical(logits=self.alphas)
    #     chosen = dist.sample()
    #     self.curr_alpha_idx = chosen.item()
    #
    # def evalMode(self):
    #     # update current alpha to max alpha value
    #     dist = F.softmax(self.alphas, dim=-1)
    #     self.curr_alpha_idx = dist.argmax().item()
    #
    # # select op index based on desired bitwidth
    # def uniformMode(self, bitwidth):
    #     # it doesn't matter which copy of ops we take, the attributes are the same in all copies
    #     for i, op in enumerate(self.ops[0]):
    #         if bitwidth in op.bitwidth:
    #             self.curr_alpha_idx = i
    #             break

    # def forward(self, x):
    #     prev_alpha_idx = self.preForward()
    #     op = self.ops[prev_alpha_idx][self.curr_alpha_idx]
    #     return op(x)

    def numOfOps(self):
        # it doesn't matter which copy of ops we take, length is the same in all copies
        return len(self.ops[0])

    def nOpsCopies(self):
        return len(self.ops)

    def getOps(self):
        return self.opsList

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


class MixedLinear(MixedFilter):
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

    def countOpsBops(self, ops, input_size):
        return [count_flops(op, input_size, 1) for op in ops]


class MixedConv(MixedFilter):
    def __init__(self, bitwidths, in_planes, out_planes, kernel_size, stride, input_size, prevLayer):
        assert (isinstance(kernel_size, list))
        params = in_planes, out_planes, kernel_size, stride
        coutBopsParams = input_size, in_planes

        self.forward = self.forwardConv

        super(MixedConv, self).__init__(bitwidths, params, coutBopsParams, prevLayer)

        self.in_planes = in_planes

        self.forwardReLU = None

    def initOps(self, bitwidths, params):
        in_planes, out_planes, kernel_size, stride = params

        ops = ModuleList()
        for bitwidth, act_bitwidth in bitwidths:
            for ker_sz in kernel_size:
                op = Sequential(
                    Conv2d(in_planes, out_planes, kernel_size=ker_sz, stride=stride, padding=floor(ker_sz / 2), bias=False)
                )
                op = QuantizedOp(op, bitwidth=[bitwidth], act_bitwidth=[] if act_bitwidth is None else [act_bitwidth])
                ops.append(op)

        return ops

    def forwardConv(self, x):
        op = self.ops[self.prev_alpha_idx][self.curr_alpha_idx].op[0]
        return op(x)

    def countOpsBops(self, op, coutBopsParams):
        input_size, in_planes = coutBopsParams
        return count_flops(op, input_size, in_planes)

    def getOpBitwidth(self, op):
        return op.bitwidth[0], None


class MixedConvWithReLU(MixedFilter):
    def __init__(self, bitwidths, in_planes, out_planes, kernel_size, stride, input_size, prevLayer):
        assert (isinstance(kernel_size, list))
        params = in_planes, out_planes, kernel_size, stride
        coutBopsParams = in_planes, input_size

        self.forward = self.forwardConv
        # if useResidual:
        #     self.forward = self.residualForward

        super(MixedConvWithReLU, self).__init__(bitwidths, params, coutBopsParams, prevLayer)

        self.in_planes = in_planes

        # init output (activations) bitwidths list
        self.outputBitwidth = []
        # it doesn't matter which copy of ops we take, the attributes are the same in all copies
        for op in self.ops[0]:
            self.outputBitwidth.extend(op.act_bitwidth)

    def initOps(self, bitwidths, params):
        in_planes, out_planes, kernel_size, stride = params

        ops = ModuleList()
        for bitwidth, act_bitwidth in bitwidths:
            for ker_sz in kernel_size:
                op = Sequential(
                    Conv2d(in_planes, out_planes, kernel_size=ker_sz, stride=stride, padding=floor(ker_sz / 2), bias=False),
                    ActQuant(quant=True, noise=False, bitwidth=act_bitwidth)
                )
                op = QuantizedOp(op, bitwidth=[bitwidth], act_bitwidth=[act_bitwidth])
                ops.append(op)

        return ops

    def forwardConv(self, x):
        op = self.ops[self.prev_alpha_idx][self.curr_alpha_idx].op[0]
        return op(x)

    def forwardReLU(self, x):
        op = self.ops[self.prev_alpha_idx][self.curr_alpha_idx].op[1]
        return op(x)

    def countOpsBops(self, op, coutBopsParams):
        input_planes, input_size = coutBopsParams
        return count_flops(op, input_size, input_planes)

    def getOpBitwidth(self, op):
        return op.bitwidth[0], op.act_bitwidth[0]

    def getCurrentOutputBitwidth(self):
        return self.outputBitwidth[self.curr_alpha_idx]

    def getOutputBitwidthList(self):
        return self.outputBitwidth

    # def residualForward(self, x, residual):
    #     prev_alpha_idx = self.preForward()
    #     return self.ops[prev_alpha_idx][self.curr_alpha_idx](x, residual)

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

    # # count bops for 1d filter, i.e. each value in map is the #bops for a 1d feature map in input
    # def buildBopsMap(self, bitwidths, input_bitwidth_list, params):
    #     initOpsParams, coutBopsParams = params
    #     # init bops map
    #     bops = {}
    #     # create set of input bitwidth
    #     input_bitwidth = set(input_bitwidth_list)
    #
    #     # build ops
    #     for in_bitwidth in input_bitwidth:
    #         bops_bitwidth = [(b, in_bitwidth) for b, _ in bitwidths]
    #         ops = self.initOps(bops_bitwidth, initOpsParams)
    #         bops[in_bitwidth] = self.countOpsBops(ops, coutBopsParams)
    #
    #     return bops
