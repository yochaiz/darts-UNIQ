from torch import tensor, randn
from torch.nn import Module, ModuleList, Conv2d, BatchNorm2d, Sequential, AvgPool2d, Linear
from UNIQ.uniq import UNIQNet
from UNIQ.actquant import ActQuant
from UNIQ.quantize import backup_weights, restore_weights, quantize


def save_quant_state(self, _):
    if self.quant and not self.noise and self.training:
        self.full_parameters = {}
        layers_list = self.get_layers_list()
        layers_steps = self.get_layers_steps(layers_list)
        assert (len(layers_steps) == 1)

        self.full_parameters = backup_weights(layers_steps[0], self.full_parameters)
        quantize(layers_steps[0], bitwidth=self.bitwidth[0])


def restore_quant_state(self, _, __):
    if self.quant and not self.noise and self.training:
        layers_list = self.get_layers_list()
        layers_steps = self.get_layers_steps(layers_list)
        assert (len(layers_steps) == 1)

        restore_weights(layers_steps[0], self.full_parameters)  # Restore the quantized layers


class QuantizedOp(UNIQNet):
    def __init__(self, op, bitwidth=[], act_bitwidth=[], useResidual=False):
        # noise=False because we want to noise only specific layer in the entire (ResNet) model
        super(QuantizedOp, self).__init__(quant=True, noise=False, quant_edges=True,
                                          act_quant=True, act_noise=False,
                                          step_setup=[1, 1],
                                          bitwidth=bitwidth, act_bitwidth=act_bitwidth)

        self.forward = self.residualForward if useResidual else self.standardForward

        self.op = op.cuda()
        self.prepare_uniq()

    def standardForward(self, x):
        assert (x.is_cuda)
        return self.op(x)

    def residualForward(self, x, residual):
        assert (x.is_cuda)
        out = self.op[0](x)
        assert (out.size() == residual.size())
        out += residual
        out = self.op[1](out)

        return out


class MixedLinear(Module):
    def __init__(self, nBitsMin, nBitsMax, in_features, out_features):
        super(MixedLinear, self).__init__()

        self.ops = ModuleList()
        for bitwidth in range(nBitsMin, nBitsMax + 1):
            op = Linear(in_features, out_features)
            self.ops.append(QuantizedOp(op, bitwidth=[bitwidth], act_bitwidth=[]))

    def forward(self, x, alphas):
        assert (x.is_cuda)
        assert (alphas.is_cuda)
        assert (len(alphas.size()) == 1)

        return sum(a * op(x) for a, op in zip(alphas, self.ops))

    def numOfOps(self):
        return len(self.ops)


class MixedConv(Module):
    def __init__(self, nBitsMin, nBitsMax, in_planes, out_planes, kernel_size, stride):
        super(MixedConv, self).__init__()

        self.ops = ModuleList()
        for bitwidth in range(nBitsMin, nBitsMax + 1):
            op = Sequential(
                Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1, bias=False),
                BatchNorm2d(out_planes)
            )
            self.ops.append(QuantizedOp(op, bitwidth=[bitwidth], act_bitwidth=[]))

    def forward(self, x, alphas):
        assert (x.is_cuda)
        assert (alphas.is_cuda)
        assert (len(alphas.size()) == 1)

        return sum(a * op(x) for a, op in zip(alphas, self.ops))

    def numOfOps(self):
        return len(self.ops)


class MixedConvWithReLU(Module):
    def __init__(self, nBitsMin, nBitsMax, in_planes, out_planes, kernel_size, stride, useResidual=False):
        super(MixedConvWithReLU, self).__init__()

        self.ops = ModuleList()
        for bitwidth in range(nBitsMin, nBitsMax + 1):
            for act_bitwidth in range(nBitsMin, nBitsMax + 1):
                op = Sequential(
                    Sequential(
                        Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1, bias=False),
                        BatchNorm2d(out_planes)
                    ),
                    ActQuant(quant=True, noise=False, bitwidth=act_bitwidth)
                )
                self.ops.append(QuantizedOp(op, bitwidth=[bitwidth], act_bitwidth=[act_bitwidth], useResidual=useResidual))

        self.forward = self.residualForward if useResidual else self.standardForward

    def standardForward(self, x, alphas):
        assert (x.is_cuda)
        assert (alphas.is_cuda)
        assert (len(alphas.size()) == 1)

        return sum(a * op(x) for a, op in zip(alphas, self.ops))

    def residualForward(self, x, alphas, residual):
        assert (x.is_cuda)
        assert (alphas.is_cuda)
        assert (len(alphas.size()) == 1)

        return sum(a * op(x, residual) for a, op in zip(alphas, self.ops))

        # # out = sum(a * op._modules['op'][0](x) for a, op in zip(alphas, self.ops))
        # # assert (out.size() == residual.size())
        # # out = out + residual
        # # out = out + sum(a * op._modules['op'][1](out) for a, op in zip(alphas, self.ops))
        #
        # return out

    def numOfOps(self):
        return len(self.ops)


class BasicBlock(Module):
    def __init__(self, nBitsMin, nBitsMax, in_planes, out_planes, kernel_size, stride):
        super(BasicBlock, self).__init__()

        stride1 = stride if in_planes == out_planes else (stride + 1)

        self.block1 = MixedConvWithReLU(nBitsMin, nBitsMax, in_planes, out_planes, kernel_size, stride1, useResidual=False)
        self.block2 = MixedConvWithReLU(nBitsMin, nBitsMax, out_planes, out_planes, kernel_size, stride, useResidual=True)

        self.downsample = MixedConv(nBitsMin, nBitsMax, in_planes, out_planes, kernel_size, stride1) \
            if in_planes != out_planes else None

    def forward(self, x, alphas, downsampleAlphas=None):
        assert (x.is_cuda)
        assert (alphas.is_cuda)
        assert (len(alphas.size()) == 2)

        residual = x if self.downsample is None else self.downsample(x, downsampleAlphas)

        out = self.block1(x, alphas[0])
        out = self.block2(out, alphas[1], residual)

        return out

    def getLayers(self):
        layers = [self.block1, self.block2]
        if self.downsample is not None:
            layers.append(self.downsample)

        return layers


class ResNet(Module):
    nClasses = 10  # cifar-10

    def __init__(self, criterion, nBitsMin, nBitsMax):
        super(ResNet, self).__init__()

        # init MixedConvWithReLU layers list
        self.layersList = []

        self.block1 = MixedConvWithReLU(nBitsMin, nBitsMax, 3, 16, 3, 1)
        self.layersList.append(self.block1)

        layers = [
            BasicBlock(nBitsMin, nBitsMax, 16, 16, 3, 1),
            BasicBlock(nBitsMin, nBitsMax, 16, 16, 3, 1),
            BasicBlock(nBitsMin, nBitsMax, 16, 16, 3, 1),
            BasicBlock(nBitsMin, nBitsMax, 16, 32, 3, 1),
            BasicBlock(nBitsMin, nBitsMax, 32, 32, 3, 1),
            BasicBlock(nBitsMin, nBitsMax, 32, 32, 3, 1),
            BasicBlock(nBitsMin, nBitsMax, 32, 64, 3, 1),
            BasicBlock(nBitsMin, nBitsMax, 64, 64, 3, 1),
            BasicBlock(nBitsMin, nBitsMax, 64, 64, 3, 1)
        ]

        i = 2
        for l in layers:
            setattr(self, 'block{}'.format(i), l)
            i += 1

        self.avgpool = AvgPool2d(8)
        self.fc = MixedLinear(nBitsMin, nBitsMax, 64, self.nClasses)

        # build mixture layers list
        for b in layers: self.layersList.extend(b.getLayers())
        self.layersList.append(self.fc)

        # set noise=True for 1st layer
        for op in self.layersList[0].ops:
            op.noise = True

        # init alphas (operations weights)
        nConvMixtureLayers, nLinearMixtureLayers, nDownsampleLayers = 0, 0, 0
        nOpsConvMixture, nOpsLinearMixture, nOpsDownsampleMixture = 0, 0, 0
        for l in self.layersList:
            if isinstance(l, MixedConvWithReLU):
                nConvMixtureLayers += 1
                nOpsConvMixture = l.numOfOps()
            elif isinstance(l, MixedLinear):
                nLinearMixtureLayers += 1
                nOpsLinearMixture = l.numOfOps()
            elif isinstance(l, MixedConv):
                nDownsampleLayers += 1
                nOpsDownsampleMixture = l.numOfOps()

        self.alphasConv = self.initialize_alphas(nConvMixtureLayers, nOpsConvMixture)
        self.alphasDownsample = self.initialize_alphas(nDownsampleLayers, nOpsLinearMixture)
        self.alphasLinear = self.initialize_alphas(nLinearMixtureLayers, nOpsDownsampleMixture)

        self._arch_parameters = [self.alphasConv, self.alphasLinear, self.alphasDownsample]

        # init criterion
        self._criterion = criterion

        # set learnable parameters
        self.learnable_params = [param for param in self.parameters() if param.requires_grad]
        # update model parameters() function
        self.parameters = self.getLearnableParams

        # init number of layers we have completed its quantization
        self.nLayersQuantCompleted = 0

    def nLayers(self):
        return len(self.layersList)

    def forward(self, x):
        assert (x.is_cuda)
        out = self.block1(x, self.alphasConv[0])

        alphasConvIdx = 1
        alphasDownsampleIdx = 0
        blockNum = 2
        b = getattr(self, 'block{}'.format(blockNum))
        while b is not None:
            alphasDownsample = None
            if b.downsample is not None:
                alphasDownsample = self.alphasDownsample[alphasDownsampleIdx]
                alphasDownsampleIdx += 1

            out = b(out, self.alphasConv[alphasConvIdx:alphasConvIdx + 2], alphasDownsample)

            alphasConvIdx += 2
            blockNum += 1
            b = getattr(self, 'block{}'.format(blockNum), None)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out, self.alphasLinear[0])

        return out

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    @staticmethod
    def initialize_alphas(nLayers, numOfOps):
        return tensor(1e-3 * randn(nLayers, numOfOps).cuda(), requires_grad=True)

    def arch_parameters(self):
        return self._arch_parameters

    def getLearnableParams(self):
        return self.learnable_params

    def switch_stage(self, logger=None):
        layer = self.layersList[self.nLayersQuantCompleted]
        for op in layer.ops:
            # turn of noise in op
            assert (op.noise is True)
            op.noise = False

            # set pre & post quantization hooks, from now on we want to quantize these ops
            op.register_forward_pre_hook(save_quant_state)
            op.register_forward_hook(restore_quant_state)

            # turn of gradients
            for m in op.modules():
                if isinstance(m, Conv2d):
                    for param in m.parameters():
                        param.requires_grad = False
                elif isinstance(m, ActQuant):
                    m.quatize_during_training = True
                    m.noise_during_training = False

        # update learnable parameters
        self.learnable_params = [param for param in self.parameters() if param.requires_grad]

        # we have completed quantization of one more layer
        self.nLayersQuantCompleted += 1

        # turn on noise in the new layer we want to quantize
        layer = self.layersList[self.nLayersQuantCompleted]
        for op in layer.ops:
            op.noise = True

        if logger:
            logger.info('Switching stage, nLayersQuantCompleted:[{}], learnable_params:[{}]'
                        .format(self.nLayersQuantCompleted, len(self.learnable_params)))
