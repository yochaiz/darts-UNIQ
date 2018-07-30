from torch.nn import Module, ModuleList, Conv2d, BatchNorm2d, Sequential, AvgPool2d, Linear
from UNIQ.uniq import UNIQNet
from UNIQ.actquant import ActQuant
import torch.nn.functional as F
import itertools

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


class DiscreteLinear(Module):
    def __init__(self, bitwidth, in_features, out_features):
        super(DiscreteLinear, self).__init__()

        self.ops = ModuleList()
        op = Linear(in_features, out_features)
        self.ops.append(QuantizedOp(op, bitwidth=[bitwidth], act_bitwidth=[]))

    def forward(self, x):
        return self.ops(x)


class DiscreteConv(Module):
    def __init__(self, bitwidth,  in_planes, out_planes, kernel_size, stride):
        super(DiscreteConv, self).__init__()
        self.ops = ModuleList()
        op = Sequential(
            Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1, bias=False),
            BatchNorm2d(out_planes)
        )
        self.ops.append(QuantizedOp(op, bitwidth=[bitwidth], act_bitwidth=[]))

    def forward(self, x):
        assert (x.is_cuda)
        return self.ops(x)


class DiscreteConvWithReLU(Module):
    def __init__(self, chosen_bitwidth, in_planes, out_planes, kernel_size, stride, useResidual=False):
        super(DiscreteConvWithReLU, self).__init__()
        self.ops = ModuleList()
        op = Sequential(
            Sequential(
                Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1, bias=False),
                BatchNorm2d(out_planes)
            ),
            ActQuant(quant=True, noise=False, bitwidth=chosen_bitwidth[1])
        )
        self.ops.append(QuantizedOp(op, bitwidth=[chosen_bitwidth[0]], act_bitwidth=[chosen_bitwidth[1]], useResidual=useResidual))

        self.forward = self.residualForward if useResidual else self.standardForward

    def standardForward(self, x):
        assert (x.is_cuda)
        return self.ops(x)

    def residualForward(self, x, residual):
        assert (x.is_cuda)
        return self.ops(x, residual)

class DiscreteBasicBlock(Module):
    def __init__(self, chosen_conv_bitwidth, chosen_downsample_bitwidth, in_planes, out_planes, kernel_size, stride):
        super(DiscreteBasicBlock, self).__init__()

        stride1 = stride if in_planes == out_planes else (stride + 1)

        self.block1 = DiscreteConvWithReLU(chosen_conv_bitwidth[0], in_planes, out_planes, kernel_size, stride1, useResidual=False)
        self.block2 = DiscreteConvWithReLU(chosen_conv_bitwidth[1], out_planes, out_planes, kernel_size, stride, useResidual=True)

        self.downsample = DiscreteConv(chosen_downsample_bitwidth, in_planes, out_planes, kernel_size, stride1) \
            if in_planes != out_planes else None

    def forward(self, x):
        assert (x.is_cuda)

        residual = x if self.downsample is None else self.downsample(x)

        out = self.block1(x)
        out = self.block2(out, residual)

        return out

    def getLayers(self):
        layers = [self.block1, self.block2]
        if self.downsample is not None:
            layers.append(self.downsample)

        return layers


class DiscreteResNet(Module):
    nClasses = 10  # cifar-10

    def __init__(self, criterion, nBitsMin, nBitsMax,module_search):
        super(DiscreteResNet, self).__init__()

        #all combination of possible bitwidth and act bitwidth
        possible_bitwidth = range (nBitsMin,nBitsMax+1)
        possible_act_conv_bitwidth= list(itertools.product(possible_bitwidth, possible_bitwidth))

        #get learnable alphas from module_search
        best_alphasConv_idx = F.softmax(module_search.alphasConv).max(1)[1]
        best_alphasDownsample_idx = F.softmax(module_search.alphasDownsample).max(1)[1]
        best_alphasLinear_idx = F.softmax(module_search.alphasLinear).max(1)[1]

        #Chosen bitwidth
        chosen_conv_bitwidth = [possible_act_conv_bitwidth[i] for i in best_alphasConv_idx]
        chosen_downsample_bitwidth = [possible_bitwidth[i] for i in best_alphasDownsample_idx]
        chosen_linear_bitwidth =  [possible_bitwidth[i] for i in best_alphasLinear_idx]

        # init MixedConvWithReLU layers list
        self.layersList = []

        self.block1 = DiscreteConvWithReLU(chosen_conv_bitwidth[0] , 3, 16, 3, 1)
        self.layersList.append(self.block1)

        layers = [
            DiscreteBasicBlock(chosen_conv_bitwidth[1:3], None, 16, 16, 3, 1),
            DiscreteBasicBlock(chosen_conv_bitwidth[3:5], None, 16, 16, 3, 1),
            DiscreteBasicBlock(chosen_conv_bitwidth[5:7], None, 16, 16, 3, 1),
            DiscreteBasicBlock(chosen_conv_bitwidth[7:9], chosen_downsample_bitwidth[0], 16, 32, 3, 1),
            DiscreteBasicBlock(chosen_conv_bitwidth[9:11], None, 32, 32, 3, 1),
            DiscreteBasicBlock(chosen_conv_bitwidth[11:13], None, 32, 32, 3, 1),
            DiscreteBasicBlock(chosen_conv_bitwidth[13:15], chosen_downsample_bitwidth[1], 32, 64, 3, 1),
            DiscreteBasicBlock(chosen_conv_bitwidth[15:17], None, 64, 64, 3, 1),
            DiscreteBasicBlock(chosen_conv_bitwidth[17:19], None, 64, 64, 3, 1)
        ]

        i = 2
        for l in layers:
            setattr(self, 'block{}'.format(i), l)
            i += 1

        self.avgpool = AvgPool2d(8)
        self.fc = DiscreteLinear(chosen_linear_bitwidth[0], 64, self.nClasses)

        # build mixture layers list
        for b in layers: self.layersList.extend(b.getLayers())
        self.layersList.append(self.fc)

         # init criterion
        self._criterion = criterion

        # set learnable parameters
        self.learnable_params = [param for param in self.parameters() if param.requires_grad]
        # update model parameters() function
        self.parameters = self.getLearnableParams


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


