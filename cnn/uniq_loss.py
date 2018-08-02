from torch.nn import CrossEntropyLoss, Module, Tanh
from torch import tensor
from cnn.resnet_model_search import ResNet

class UniqLoss(Module):
    def __init__(self, lmdba, MaxBopsBits, kernel_sizes):
        super(UniqLoss, self).__init__()
        self.lmdba = lmdba
        self.search_loss = CrossEntropyLoss().cuda()
        # self.resnet_input_size = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 16, 16, 16, 16, 16, 16, 16, 8, 8, 8, 8, 1]

        # build model for uniform distribution of bits
        uniform_model = ResNet(self.search_loss, bitwidths=[MaxBopsBits], kernel_sizes=kernel_sizes)
        self.maxBops = uniform_model.countBops()
        self.penalization_func = Tanh()

    def forward(self, input, target, modelBops):
        # big penalization if bops over MaxBops
        quant_loss = self._bops_loss(modelBops)

        return self.search_loss(input, target) + (self.lmdba * quant_loss)


    def _bops_loss(self, modelBops):
        #Parameters that were found emphirical

        scale_diff = (modelBops - self.maxBops) / self.maxBops
        strech_factor = 20
        reward = tensor(-1.1).cuda()
        return (self.penalization_func((strech_factor * scale_diff) + reward) + 1) * 2.5


    # structure =
    # @(acc_, x, scale, stretchFactor) (tanh((acc_ - x) * stretchFactor) * scale);
    #
    # acc = 1;
    # scale = 1;
    # stretchFactor = 50;
    # reward = 1.6;
    #
    # v = acc + (reward / (stretchFactor * scale));
    #
    # f = @(acc_)    ((structure(acc_, v, scale, stretchFactor) + 1) * 2.5);

    # def forward(self, input, target, model):
    #     bops_input = MaxBops = 0
    #
    #     alphasConvIdx = alphasDownsampleIdx = alphasLinearIdx = 0
    #
    #     for idx, l in enumerate(model.layersList):
    #         bops = []
    #         if isinstance(l, MixedConvWithReLU):
    #             alphas = model.alphasConv[alphasConvIdx]
    #             alphasConvIdx += 1
    #         elif isinstance(l, MixedLinear):
    #             alphas = model.alphasLinear[alphasLinearIdx]
    #             alphasLinearIdx += 1
    #         elif isinstance(l, MixedConv):
    #             alphas = model.alphasDownsample[alphasDownsampleIdx]
    #             alphasDownsampleIdx += 1
    #         _, uniform_bops = count_flops(self.uniform_model.layersList[idx].ops[0], self.batch_size, 1, l.in_planes)
    #         MaxBops += uniform_bops
    #
    #         for op in l.ops:
    #             bops_op = 'operation_{}_kernel_{}x{}_bit_{}_act_{}_channels_{}_residual_{}'.format(
    #                 str(type(l)).split(".")[-1], l.kernel_size, l.kernel_size, op.bitwidth, op.act_bitwidth,
    #                 l.in_planes, op.useResidual)
    #             bops.append(model.bopsDict[bops_op])
    #
    #         weights = F.softmax(alphas)
    #         bops_input += sum(a * op for a, op in zip(weights, bops))
    #
    #     # big penalization if bops over MaxBops
    #     penalization_factor = 1
    #     if (bops_input > MaxBops):
    #         penalization_factor = 5
    #     quant_loss = penalization_factor * (bops_input / MaxBops)
    #
    #     return self.search_loss(input, target) + (self.lmdba * quant_loss)

    # calculate BOPS per operation in each layer
#      self.bopsDict = {}
#      for l in self.layersList:
#          for op in l.ops:
#              bops_op = 'operation_{}_kernel_{}x{}_bit_{}_act_{}_channels_{}_residual_{}'. format(str(type(l)).split(".")[-1], l.kernel_size, l.kernel_size, op.bitwidth, op.act_bitwidth, l.in_planes,op.useResidual)
#              if bops_op not in self.bopsDict:
#                  _, curr_bops = count_flops(op, batch_size, 1, l.in_planes)
#                  self.bopsDict[bops_op] = curr_bops
