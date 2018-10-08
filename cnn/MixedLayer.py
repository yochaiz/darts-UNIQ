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

    def forwardConv(self, x):
        out = []
        # apply selected op in each filter
        for f in self.filters:
            res = f.forwardConv(x)
            out.append(res)
        # concat filters output
        out = cat(out, 1)

        return out

    def forwardReLU(self, x):
        out = []
        # apply selected op in each filter
        for i, f in enumerate(self.filters):
            res = f.forwardReLU(x[i])
            out.append(res)
        # concat filters output
        out = cat(out, 1)

        return out

    def preResidualForward(self, x):
        out = self.forwardConv(x)
        # apply batch norm
        out = self.bn(out)

        return out

    def postResidualForward(self, out):
        # apply ReLU if exists
        if self.filters[0].forwardReLU:
            # split out1 to chunks again
            out = chunk(out, self.nFilters(), dim=1)
            out = self.forwardReLU(out)

        return out

    def forward(self, x):
        out = self.preResidualForward(x)
        out = self.postResidualForward(out)

        return out

    def residualForward(self, x, residual):
        out = self.preResidualForward(x)
        # add residual
        out += residual
        out = self.postResidualForward(out)

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
