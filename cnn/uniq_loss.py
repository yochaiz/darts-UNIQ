from torch.nn import CrossEntropyLoss, Module, Tanh
from torch import tensor, float32
from numpy import arctanh, linspace

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class BopsLoss:
    def __init__(self, bopsFunc, v, factor, yDelta, scale):
        self.bopsFunc = bopsFunc
        self.v = v
        self.factor = factor
        self.yDelta = yDelta
        self.scale = scale

    def calcLoss(self, x):
        return (self.bopsFunc((x - self.v) * self.factor) + self.yDelta) * self.scale


class UniqLoss(Module):
    def __init__(self, lmdba, maxBops, folderName):
        super(UniqLoss, self).__init__()
        self.lmdba = lmdba
        self.search_loss = CrossEntropyLoss().cuda()

        self.maxBops = maxBops

        # init bops loss function and plot it
        self.bops_base_func = Tanh()
        self.bopsLoss = self._bops_loss(xDst=1, yDst=0.2, yMin=0, yMax=5)
        self.plotFunction(self.bopsLoss, folderName)

        # init values
        self.bopsRatio = -1
        self.quant_loss = -1

    def calcBopsRatio(self, modelBops):
        return modelBops / self.maxBops

    def calcBopsLoss(self, bopsRatio):
        return self.bopsLoss(bopsRatio)

    def forward(self, input, target, modelBops):
        # big penalization if bops over MaxBops
        self.bopsRatio = self.calcBopsRatio(modelBops)
        self.quant_loss = self.calcBopsLoss(self.bopsRatio)

        return self.search_loss(input, target) + (self.lmdba * self.quant_loss.cuda(input.device.index))
        # return self.search_loss(input, target)
        # TODO: sort all the GPU stuff in this class

    # given the 4 values, generate the appropriate tanh() function, s.t. t(xDst)=yDst & max{t}=yMax & min{t}=yMin
    def _bops_loss(self, xDst, yDst, yMin, yMax):
        factor = 20

        yDelta = yMin - (-1)
        scale = yMax / (1 + yDelta)

        v = (yDst / scale) - yDelta
        v = arctanh(v)
        v /= factor
        v = round(v, 5)
        v = xDst - v

        yDelta = tensor(yDelta, dtype=float32).cuda()
        scale = tensor(scale, dtype=float32).cuda()
        v = tensor(v, dtype=float32).cuda()
        factor = tensor(factor, dtype=float32).cuda()

        bopsLoss = BopsLoss(self.bops_base_func, v, factor, yDelta, scale)
        return bopsLoss.calcLoss

        # def t(x):
        #     # out = (x - v) * factor
        #     # out = tanh(out)
        #     # out = (out + yDelta) * scale
        #     return (bopsFunc((x - v) * factor) + yDelta) * scale
        #
        # return t

    def plotFunction(self, func, folderName):
        # build data for function
        nPts = 201
        ptsGap = 5

        pts = linspace(0, 2, nPts).tolist()
        y = [round(func(x).item(), 5) for x in pts]
        data = [[pts, y, 'bo']]
        pts = [pts[x] for x in range(0, nPts, ptsGap)]
        y = [y[k] for k in range(0, nPts, ptsGap)]
        data.append([pts, y, 'go'])

        # plot
        fig, ax = plt.subplots(nrows=1, ncols=1)
        for x, y, style in data:
            ax.plot(x, y, style)

        ax.set_xticks(pts)
        ax.set_yticks(y)
        ax.set_xlabel('bops/maxBops')
        ax.set_ylabel('Loss')
        ax.set_title('Bops ratio loss function')
        fig.set_size_inches(25, 10)

        if folderName:
            fig.savefig('{}/bops_loss_func.png'.format(folderName))

    # def _bops_loss(self, bops):
    #     # Parameters that were found emphirical
    #
    #     scale_diff = (bops - self.maxBops) / self.maxBops
    #     strech_factor = 10
    #     reward = tensor(-1.1, dtype=float32).cuda()
    #     return (self.bops_base_func((strech_factor * scale_diff) + reward) + 1) * 2.5
