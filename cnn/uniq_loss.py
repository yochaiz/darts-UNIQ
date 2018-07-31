from torch.nn import CrossEntropyLoss
from torch.nn import Module, ModuleList, Conv2d, BatchNorm2d, Sequential, AvgPool2d, Linear
import torch.nn.functional as F

from cnn.resnet_model_search import ResNet, MixedConv, MixedConvWithReLU, MixedLinear
from UNIQ.flops_benchmark import count_flops


class UniqLoss(Module):
    def __init__(self, lmdba, MaxBopsBits, batch_size):
        super(UniqLoss, self).__init__()
        self.lmdba = lmdba
        self.batch_size = batch_size
        self.search_loss = CrossEntropyLoss().cuda()
        #self.resnet_input_size = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 16, 16, 16, 16, 16, 16, 16, 8, 8, 8, 8, 1]

        #build model for uniform distribution of bits
        self.uniform_model = ResNet(self.search_loss , MaxBopsBits, MaxBopsBits,batch_size)

    def forward(self, input, target,model):
        bops_input = MaxBops = 0

        alphasConvIdx = alphasDownsampleIdx = alphasLinearIdx = 0

        for idx,l in enumerate(model.layersList):
            bops = []
            if isinstance(l, MixedConvWithReLU):
                alphas = model.alphasConv[alphasConvIdx]
                alphasConvIdx += 1
            elif isinstance(l, MixedLinear):
                alphas = model.alphasLinear[alphasLinearIdx]
                alphasLinearIdx += 1
            elif isinstance(l, MixedConv):
                alphas = model.alphasDownsample[alphasDownsampleIdx]
                alphasDownsampleIdx += 1
            _ , uniform_bops = count_flops(self.uniform_model.layersList[idx].ops[0], self.batch_size, 1, l.in_planes)
            MaxBops += uniform_bops

            for op in l.ops:
                bops_op = 'operation_{}_kernel_{}x{}_bit_{}_act_{}_channels_{}_residual_{}'.format(
                    str(type(l)).split(".")[-1], l.kernel_size, l.kernel_size, op.bitwidth, op.act_bitwidth,
                    l.in_planes, op.useResidual)
                bops.append(model.bopsDict[bops_op])

            weights = F.softmax(alphas)
            bops_input += sum(a * op for a, op in zip(weights, bops))

        #big penalization if bops over MaxBops
        penalization_factor = 1
        if (bops_input > MaxBops):
            penalization_factor = 5 #TODO - change penalization factor
        quant_loss = penalization_factor * (bops_input / MaxBops)

        return self.search_loss(input, target) + (self.lmdba * quant_loss)






       #calculate BOPS per operation in each layer
  #      self.bopsDict = {}
  #      for l in self.layersList:
  #          for op in l.ops:
  #              bops_op = 'operation_{}_kernel_{}x{}_bit_{}_act_{}_channels_{}_residual_{}'. format(str(type(l)).split(".")[-1], l.kernel_size, l.kernel_size, op.bitwidth, op.act_bitwidth, l.in_planes,op.useResidual)
  #              if bops_op not in self.bopsDict:
  #                  _, curr_bops = count_flops(op, batch_size, 1, l.in_planes)
  #                  self.bopsDict[bops_op] = curr_bops
