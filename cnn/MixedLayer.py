from torch import cat, chunk, tensor, ones
from torch.nn import Module, ModuleList, BatchNorm2d

from cnn.MixedFilter import MixedFilter


class MixedLayer(Module):
    def __init__(self, nFilters, createMixedFilterFunc, useResidual=False):
        super(MixedLayer, self).__init__()

        # create mixed filters
        self.filters = ModuleList()
        for _ in range(nFilters):
            self.filters.append(createMixedFilterFunc())
        # make sure mixed filters are subclasses of MixedFilter
        assert (isinstance(self.filters[0], MixedFilter))

        # init batch norm
        self.bn = BatchNorm2d(nFilters)

        # init operations alphas (weights)
        value = 1.0 / self.numOfOps()
        self.alphas = tensor((ones(self.numOfOps()) * value).cuda(), requires_grad=True)

        # init list of all operations (including copies) as single long list
        # for cases we have to modify all ops
        self.opsList = []
        for filter in self.filters:
            for ops in filter.ops:
                for op in ops:
                    self.opsList.append(op)

        # set filters allocation
        ratio = [0.3125, 0.3125, 0.1875, 0.125, 0.0625]
        self.setFiltersRatio(ratio)

        # set forward function
        if useResidual:
            self.forward = self.residualForward

    def nFilters(self):
        return len(self.filters)

    def setFiltersRatio(self, ratio):
        ratio = [int(r * self.nFilters()) for r in ratio]
        ratio[-1] = self.nFilters() - sum(ratio[:-1])
        idx = 0
        for i, r in enumerate(ratio):
            for _ in range(r):
                self.filters[idx].curr_alpha_idx = i
                idx += 1

    def forward(self, x):
        out1 = []
        # apply selected op in each filter
        for f in self.filters:
            f.preForward()
            op = f.getCurrentOp().op
            res = op[0](x)
            out1.append(res)
        # concat filters output
        out1 = cat(out1, 1)
        # apply batch norm
        out1 = self.bn(out1)
        out = out1
        # apply op 2nd part if exists
        if len(op) > 1:
            out2 = []
            # split out1 to chunks again
            out1 = chunk(out1, self.nFilters(), dim=1)
            for i, f in enumerate(self.filters):
                op = f.getCurrentOp().op
                res = op[1](out1[i])
                out2.append(res)
            # concat filters output
            out2 = cat(out2, 1)
            out = out2

        return out

    def residualForward(self, x, residual):
        out1 = []
        # apply selected op in each filter
        for f in self.filters:
            f.preForward()
            op = f.getCurrentOp().op
            res = op[0](x)
            out1.append(res)
        # concat filters output
        out1 = cat(out1, 1)
        # apply batch norm
        out1 = self.bn(out1)
        out1 += residual
        out = out1
        # apply op 2nd part if exists
        if len(op) > 1:
            out2 = []
            # split out1 to chunks again
            out1 = chunk(out1, self.nFilters(), dim=1)
            for i, f in enumerate(self.filters):
                op = f.getCurrentOp().op
                res = op[1](out1[i])
                out2.append(res)
            # concat filters output
            out2 = cat(out2, 1)
            out = out2

        return out

    ## functions that need to be examined about their correctness
    def getOutputBitwidthList(self):
        return self.filters[0].getOutputBitwidthList()

    def outputLayer(self):
        return self

    def getBops(self, input_bitwidth):
        return 10

    def getCurrentOutputBitwidth(self):
        return self.filters[0].getCurrentOutputBitwidth()

    # select random alpha
    def chooseRandomPath(self):
        pass

    # select alpha based on alphas distribution
    def choosePathByAlphas(self):
        pass

    def evalMode(self):
        pass

    def numOfOps(self):
        return self.filters[0].numOfOps()

    def getOps(self):
        return self.opsList
