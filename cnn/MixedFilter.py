from UNIQ.flops_benchmark import count_flops
from NICE.uniq import UNIQNet
from NICE.actquant import ActQuantBuffers

from torch import ones
from torch.nn import ModuleList, Conv2d, Sequential, BatchNorm2d

from cnn.block import Block

from math import floor, ceil, log2
from abc import abstractmethod


class QuantizedOp(UNIQNet):
    def __init__(self, op, bitwidth, act_bitwidth, modulesIdxDict):
        super(QuantizedOp, self).__init__(bitwidth=bitwidth, act_bitwidth=act_bitwidth, params=op)

        self.modulesIdxDict = modulesIdxDict

    def derivedClassSpecific(self, op):
        self.op = op.cuda()

        # self.useResidual = useResidual
        # self.forward = self.residualForward if useResidual else self.standardForward
        # self.hookHandlers = []

    def getBitwidth(self):
        bitwidth = self.bitwidth[0] if len(self.bitwidth) > 0 else None
        act_bitwidth = self.act_bitwidth[0] if len(self.act_bitwidth) > 0 else None
        return bitwidth, act_bitwidth

    # get module by type inside filter, else return None
    # modulesIdxDict is a dictionary, where the key is the module type, like Conv2d, ActQuant
    def getModule(self, moduleType):
        module = None
        idx = self.modulesIdxDict.get(moduleType, None)
        if idx is not None:
            module = self.op[idx]

        return module

    def __getDeviceName(self):
        return str(self.op[0].weight.device)

    def quantizeFunc(self):
        self._quantizeFunc(self.__getDeviceName())

    def add_noise(self):
        self._add_noise(self.__getDeviceName())

    def restore_state(self):
        self._restore_state(self.__getDeviceName())

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

    def compute_bops_mults_adds(self, batch_size):
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

def preForward(self, input):
    deviceID = input[0].device.index
    assert (deviceID not in self.hookDevices)
    self.hookDevices.append(deviceID)

    # update previous layer index
    prevLayer = self.prevLayer[0]
    self.prev_alpha_idx = prevLayer.curr_alpha_idx if prevLayer else 0
    # update forward counter
    self.opsForwardCounters[self.prev_alpha_idx][self.curr_alpha_idx] += 1
    # # get current op
    # op = self.ops[self.prev_alpha_idx][self.curr_alpha_idx]
    # # check if we need to add noise
    # if op.noise is True:
    #     op.add_noise()


def postForward(self, input, __):
    deviceID = input[0].device.index
    assert (deviceID in self.hookDevices)
    self.hookDevices.remove(deviceID)


#
#     # get current op
#     op = self.ops[self.prev_alpha_idx][self.curr_alpha_idx]
#     # check if we need to remove noise
#     if op.noise is True:
#         op.restore_state()


class MixedFilter(Block):
    def __init__(self, bitwidths, params, countBopsParams, prevLayer):
        super(MixedFilter, self).__init__()

        # assure bitwidths is a list of integers
        if isinstance(bitwidths[0], list):
            bitwidths = bitwidths[0]
        # remove duplicate values in bitwidth
        bitwidths = self.__removeDuplicateValues(bitwidths)

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

        # init ops forward counters
        self.opsForwardCounters = self.buildOpsForwardCounters()

        self.curr_alpha_idx = 0
        self.prev_alpha_idx = 0
        # init counter for number of consecutive times optimal alpha reached optimal probability limit
        self.optLimitCounter = 0

        # set forward function in order to assure that hooks will take place
        self.forwardFunc = self.setForwardFunc()
        # assign pre & post forward hooks
        self.register_forward_pre_hook(preForward)
        self.register_forward_hook(postForward)
        # set hook flag, to make sure hook happens
        # turn it on on pre-forward hook, turn it off on post-forward hook
        self.hookDevices = []

        # list of (mults, adds, calc_mac_value, batch_size) per op
        self.bops = self.countOpsBops(countBopsParams)

    def forward(self, input):
        forwardFunc = self.setForwardFunc()
        return forwardFunc(input)

    @abstractmethod
    def initOps(self, bitwidths, params):
        raise NotImplementedError('subclasses must override initOps()!')

    @abstractmethod
    def setForwardFunc(self):
        raise NotImplementedError('subclasses must override setForwardFunc()!')

    @abstractmethod
    def getCurrentOutputBitwidth(self):
        raise NotImplementedError('subclasses must override getCurrentOutputBitwidth()!')

    def buildOpsForwardCounters(self):
        return [[0] * len(ops) for ops in self.ops]

    def resetOpsForwardCounters(self):
        self.opsForwardCounters = self.buildOpsForwardCounters()

    def outputLayer(self):
        return self

    @staticmethod
    def __removeDuplicateValues(values):
        assert (isinstance(values, list))
        newValues = []
        for v in values:
            if v not in newValues:
                newValues.append(v)

        return newValues

    def countOpsBops(self, countBopsParams):
        input_size, in_planes = countBopsParams
        return [count_flops(op, input_size, in_planes) for op in self.ops[0]]

    def getBops(self, input_bitwidth, bopsMap):
        # save op bitwidth for efficient bops calculation in layer resolution
        opBitwidth = self.getCurrentBitwidth()
        # check if opBitwidth is in map, to save calculations
        if opBitwidth in bopsMap:
            bops = bopsMap[opBitwidth]
        else:
            # get filter current bitwidth
            bitwidth, _ = opBitwidth
            # bops calculation is for weight bitwidth > 1
            assert (bitwidth > 1)
            # get bops values
            mults, adds, calcMacObj, batch_size = self.bops[self.curr_alpha_idx]
            # calc max_mac_value
            max_mac_value = 0
            for act_bitwidth in input_bitwidth:
                max_mac_value += calcMacObj.calc(bitwidth, act_bitwidth)
            # init log2_max_mac_value
            log2_max_mac_value = ceil(log2(max_mac_value))
            # calc bops
            bops = 0
            # calc bops mults
            for act_bitwidth in input_bitwidth:
                bops += mults * (bitwidth - 1) * act_bitwidth
            # add bops adds
            bops += adds * log2_max_mac_value
            # divide by batch size
            bops /= batch_size
            # add bops value to map
            bopsMap[opBitwidth] = bops

        return bops

    # return list of tuples of all filter bitwidths
    def getAllBitwidths(self):
        return [op.getBitwidth() for op in self.ops[0]]

    # returns current op bitwidth
    def getCurrentBitwidth(self):
        # it doesn't matter which copy of ops we take, the attributes are the same in all copies
        return self.ops[0][self.curr_alpha_idx].getBitwidth()

    def numOfOps(self):
        # it doesn't matter which copy of ops we take, length is the same in all copies
        return len(self.ops[0])

    def nOpsCopies(self):
        return len(self.ops)

    def opsList(self):
        for ops in self.ops:
            for op in ops:
                yield op

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


class MixedConv(MixedFilter):
    def __init__(self, bitwidths, in_planes, out_planes, kernel_size, stride, input_size, prevLayer):
        assert (isinstance(kernel_size, list))
        params = in_planes, out_planes, kernel_size, stride
        coutBopsParams = input_size, in_planes

        super(MixedConv, self).__init__(bitwidths, params, coutBopsParams, prevLayer)

        self.in_planes = in_planes

        self.postResidualForward = None

    def setForwardFunc(self):
        return self.preResidualForward

    def initOps(self, bitwidths, params):
        in_planes, out_planes, kernel_size, stride = params

        ops = ModuleList()
        for bitwidth, act_bitwidth in bitwidths:
            for ker_sz in kernel_size:
                conv = Conv2d(in_planes, out_planes, kernel_size=ker_sz, stride=stride, padding=floor(ker_sz / 2), bias=False)
                conv.register_buffer('layer_b', ones(1))  # Attempt to enable multi-GPU
                conv.register_buffer('initial_clamp_value', ones(1))  # Attempt to enable multi-GPU
                conv.register_buffer('layer_basis', ones(1))  # Attempt to enable multi-GPU

                op = Sequential(conv)
                op = QuantizedOp(op, bitwidth=[bitwidth], act_bitwidth=[] if act_bitwidth is None else [act_bitwidth], modulesIdxDict={Conv2d: 0})
                ops.append(op)

        return ops

    def preResidualForward(self, x):
        assert (x.device.index in self.hookDevices)
        op = self.ops[self.prev_alpha_idx][self.curr_alpha_idx].getModule(Conv2d)
        return op(x)


class MixedConvBN(MixedFilter):
    def __init__(self, bitwidths, in_planes, out_planes, kernel_size, stride, input_size, prevLayer):
        assert (isinstance(kernel_size, list))
        params = in_planes, out_planes, kernel_size, stride
        coutBopsParams = input_size, in_planes

        super(MixedConvBN, self).__init__(bitwidths, params, coutBopsParams, prevLayer)

        self.in_planes = in_planes

        self.postResidualForward = None

    def setForwardFunc(self):
        return self.forward

    def initOps(self, bitwidths, params):
        in_planes, out_planes, kernel_size, stride = params

        ops = ModuleList()
        for bitwidth, act_bitwidth in bitwidths:
            for ker_sz in kernel_size:
                conv = Conv2d(in_planes, out_planes, kernel_size=ker_sz, stride=stride, padding=floor(ker_sz / 2), bias=False)
                conv.register_buffer('layer_b', ones(1))  # Attempt to enable multi-GPU
                conv.register_buffer('initial_clamp_value', ones(1))  # Attempt to enable multi-GPU
                conv.register_buffer('layer_basis', ones(1))  # Attempt to enable multi-GPU

                op = Sequential(conv, BatchNorm2d(out_planes))
                op = QuantizedOp(op, bitwidth=[bitwidth], act_bitwidth=[] if act_bitwidth is None else [act_bitwidth],
                                 modulesIdxDict={Conv2d: 0, BatchNorm2d: 1})
                ops.append(op)

        return ops

    def forward(self, x):
        assert (x.device.index in self.hookDevices)
        op = self.ops[self.prev_alpha_idx][self.curr_alpha_idx]
        return op(x)


class MixedConvBNWithReLU(MixedFilter):
    def __init__(self, bitwidths, in_planes, out_planes, kernel_size, stride, input_size, prevLayer):
        assert (isinstance(kernel_size, list))
        params = in_planes, out_planes, kernel_size, stride
        coutBopsParams = input_size, in_planes

        super(MixedConvBNWithReLU, self).__init__(bitwidths, params, coutBopsParams, prevLayer)

        self.in_planes = in_planes

        # init list of possible output (activations) bitwidth
        self.outputBitwidth = []
        # it doesn't matter which copy of ops we take, the attributes are the same in all copies
        for op in self.ops[0]:
            self.outputBitwidth.extend(op.act_bitwidth)

    def setForwardFunc(self):
        return self.preResidualForward

    def __initOps(self, bitwidths, params, buildOpFunc):
        in_planes, out_planes, kernel_size, stride = params

        ops = ModuleList()
        for bitwidth, act_bitwidth in bitwidths:
            for ker_sz in kernel_size:
                op = buildOpFunc(bitwidth, act_bitwidth, (in_planes, out_planes, ker_sz, stride))
                op = QuantizedOp(op, bitwidth=[bitwidth], act_bitwidth=[act_bitwidth],
                                 modulesIdxDict={Conv2d: 0, BatchNorm2d: 1, ActQuantBuffers: 2})
                ops.append(op)

        return ops

    def initOps(self, bitwidths, params):
        def buildOpFunc(bitwidth, act_bitwidth, params):
            in_planes, out_planes, kernel_size, stride = params

            conv = Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=floor(kernel_size / 2), bias=False)
            conv.register_buffer('layer_b', ones(1))  # Attempt to enable multi-GPU
            conv.register_buffer('initial_clamp_value', ones(1))  # Attempt to enable multi-GPU
            conv.register_buffer('layer_basis', ones(1))  # Attempt to enable multi-GPU

            return Sequential(
                conv,
                BatchNorm2d(out_planes),
                ActQuantBuffers(bitwidth=act_bitwidth)
            )

        return self.__initOps(bitwidths, params, buildOpFunc)

    def preResidualForward(self, x):
        assert (x.device.index in self.hookDevices)
        op = self.ops[self.prev_alpha_idx][self.curr_alpha_idx]
        conv = op.getModule(Conv2d)
        bn = op.getModule(BatchNorm2d)

        out = conv(x)
        out = bn(out)
        return out

    def postResidualForward(self, x):
        op = self.ops[self.prev_alpha_idx][self.curr_alpha_idx].getModule(ActQuantBuffers)
        return op(x)

    def getCurrentOutputBitwidth(self):
        return self.outputBitwidth[self.curr_alpha_idx]


class MixedConvWithReLU(MixedFilter):
    def __init__(self, bitwidths, in_planes, out_planes, kernel_size, stride, input_size, prevLayer):
        assert (isinstance(kernel_size, list))
        params = in_planes, out_planes, kernel_size, stride
        coutBopsParams = input_size, in_planes

        super(MixedConvWithReLU, self).__init__(bitwidths, params, coutBopsParams, prevLayer)

        self.in_planes = in_planes

        # init list of possible output (activations) bitwidth
        self.outputBitwidth = []
        # it doesn't matter which copy of ops we take, the attributes are the same in all copies
        for op in self.ops[0]:
            self.outputBitwidth.extend(op.act_bitwidth)

    def setForwardFunc(self):
        return self.preResidualForward

    def __initOps(self, bitwidths, params, buildOpFunc):
        in_planes, out_planes, kernel_size, stride = params

        ops = ModuleList()
        for bitwidth, act_bitwidth in bitwidths:
            for ker_sz in kernel_size:
                op = buildOpFunc(bitwidth, act_bitwidth, (in_planes, out_planes, ker_sz, stride))
                op = QuantizedOp(op, bitwidth=[bitwidth], act_bitwidth=[act_bitwidth], modulesIdxDict={Conv2d: 0, ActQuantBuffers: 1})
                ops.append(op)

        return ops

    def initOps(self, bitwidths, params):
        def buildOpFunc(bitwidth, act_bitwidth, params):
            in_planes, out_planes, kernel_size, stride = params

            conv = Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=floor(kernel_size / 2), bias=False)
            conv.register_buffer('layer_b', ones(1))  # Attempt to enable multi-GPU
            conv.register_buffer('initial_clamp_value', ones(1))  # Attempt to enable multi-GPU
            conv.register_buffer('layer_basis', ones(1))  # Attempt to enable multi-GPU

            return Sequential(
                conv,
                ActQuantBuffers(bitwidth=act_bitwidth)
            )

        return self.__initOps(bitwidths, params, buildOpFunc)

    def preResidualForward(self, x):
        assert (x.device.index in self.hookDevices)
        op = self.ops[self.prev_alpha_idx][self.curr_alpha_idx].getModule(Conv2d)
        return op(x)

    def postResidualForward(self, x):
        op = self.ops[self.prev_alpha_idx][self.curr_alpha_idx].getModule(ActQuantBuffers)
        return op(x)

    def getCurrentOutputBitwidth(self):
        return self.outputBitwidth[self.curr_alpha_idx]

    # def getOutputBitwidthList(self):
    #     return self.outputBitwidth

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
