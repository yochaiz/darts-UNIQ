from scipy.stats import entropy
from os import makedirs, path
from math import ceil, floor, sqrt

from torch import tensor, float32
import torch.nn.functional as F

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Statistics:
    entropyKey = 'alphas_entropy'
    weightedAvgKey = 'alphas_weighted_average'
    alphaDistributionKey = 'alphas_distribution'
    alphaLossVarianceKey = 'alphas_loss_variance'
    alphaLossAvgKey = 'alphas_loss_avg'
    allSamplesLossVarianceKey = 'all_samples_loss_variance'
    allSamplesLossAvgKey = 'all_samples_loss_avg'
    gradNormKey = 'alphas_gradient_norm'
    bopsKey = 'bops'
    batchOptModelBopsRatioKey = 'optimal_model_bops_ratio'

    # set plot points style
    ptsStyle = '-'

    # set maxCols & minRows for multiplot
    nColsMax = 7
    nRowsDefault = 3

    def __init__(self, layersList, nLayers, saveFolder):
        self.nLayers = nLayers
        # create plot folder
        plotFolderPath = '{}/plots'.format(saveFolder)
        if not path.exists(plotFolderPath):
            makedirs(plotFolderPath)

        self.saveFolder = plotFolderPath
        # init list of batch labels for y axis
        self.batchLabels = []
        # collect op bitwidth per layer in model
        self.layersBitwidths = [tensor([op.bitwidth[0] for op in layer.ops[0]], dtype=float32).cuda()
                                for layer in layersList]
        # plot bops plot
        self.plotBops(layersList)
        # init containers
        self.containers = {
            self.entropyKey: [[] for _ in range(nLayers)],
            self.weightedAvgKey: [[] for _ in range(nLayers)],
            self.alphaLossVarianceKey: [[[] for _ in range(layer.numOfOps())] for layer in layersList],
            self.alphaLossAvgKey: [[[] for _ in range(layer.numOfOps())] for layer in layersList],
            self.alphaDistributionKey: [[[] for _ in range(layer.numOfOps())] for layer in layersList],
            self.allSamplesLossVarianceKey: [[]],
            self.allSamplesLossAvgKey: [[]],
            self.batchOptModelBopsRatioKey: [[]],
            self.gradNormKey: [[] for _ in range(nLayers)]
        }
        # map each list we plot for all layers on single plot to filename
        self.plotAllLayersKeys = [self.entropyKey, self.allSamplesLossVarianceKey, self.allSamplesLossAvgKey,
                                  self.gradNormKey, self.batchOptModelBopsRatioKey]
        self.plotLayersSeparateKeys = [self.alphaLossAvgKey, self.alphaLossVarianceKey, self.alphaDistributionKey]

    def addBatchData(self, model, optBopsRatio, nEpoch, nBatch):
        assert (self.nLayers == model.nLayers())
        # add batch label
        self.batchLabels.append('[{}]_[{}]'.format(nEpoch, nBatch))
        self.containers[self.batchOptModelBopsRatioKey][0].append(optBopsRatio)
        # add data per layer
        for i, layer in enumerate(model.layersList):
            # calc layer alphas distribution
            probs = F.softmax(layer.alphas, dim=-1).detach()
            # save distribution
            for j, p in enumerate(probs):
                self.containers[self.alphaDistributionKey][i][j].append(p.item())
            # calc entropy
            self.containers[self.entropyKey][i].append(entropy(probs))

            # # collect weight bitwidth of each op in layer
            # weightBitwidth = self.layersBitwidths[i]
            # # calc weighted average of weights bitwidth
            # res = probs * weightBitwidth
            # res = res.sum().item()
            # # add layer weighted average
            # self.containers[self.weightedAvgKey][i].append(res)

        # plot data
        self.plotData()

        return optBopsRatio

    @staticmethod
    def __setAxesProperties(ax, xLabel, yLabel, yMax, title):
        # ax.set_xticks(xValues)
        # ax.set_xticklabels(self.batchLabels)
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        ax.set_ylim(ymax=yMax, ymin=0.0)
        ax.set_title(title)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.005), ncol=5, fancybox=True, shadow=True)

    def __setFigProperties(self, fig, fileName, figSize=(15, 10)):
        fig.set_size_inches(figSize)
        fig.tight_layout()
        # save to file
        fig.savefig('{}/{}.png'.format(self.saveFolder, fileName))
        # close plot
        plt.close(fig)

    def __setPlotProperties(self, fig, ax, xLabel, yLabel, yMax, title, fileName):
        self.__setAxesProperties(ax, xLabel, yLabel, yMax, title)
        self.__setFigProperties(fig, fileName)

    def __plotContainer(self, data, xValues, xLabel, yLabel, title, fileName, labelFunc, axOther=None, scale=True,
                        annotate=None):
        # create plot
        fig, ax = plt.subplots(nrows=1, ncols=1)
        # init ylim values
        dataMax = 0
        dataSum = []
        # init flag to check whether we have plotted something or not
        isPlotEmpty = True

        for i, layerData in enumerate(data):
            if len(xValues) == len(layerData):
                ax.plot(xValues, layerData, self.ptsStyle, label=labelFunc(i))
                if axOther:
                    axOther.plot(xValues, layerData, '-', label=labelFunc(i))
                isPlotEmpty = False
                dataMax = max(dataMax, max(layerData))
                dataSum.append(sum(layerData) / len(layerData))

        if not isPlotEmpty:
            # add annotations if exists
            if annotate:
                for txt, pt in annotate:
                    ax.annotate(txt, pt)

            # set yMax
            yMax = dataMax * 1.1

            # axOther doesn't scale
            if axOther:
                if yLabel == self.alphaDistributionKey:
                    yMax = 1.1

                axOther.grid()
                self.__setAxesProperties(axOther, xLabel, yLabel, yMax, title)

            if scale:
                yMax = min(yMax, (sum(dataSum) / len(dataSum)) * 1.5)

            self.__setPlotProperties(fig, ax, xLabel, yLabel, yMax, title, fileName)

    # find optimal grid
    def __findGrid(self, nPlots):
        nRowsOpt, nColsOpt = 0, 0
        optDiff = None

        # iterate over options
        for nRows in range(1, self.nRowsDefault + 1):
            nCols = ceil(nPlots / nRows)
            # calc how many empty plots will be in grid
            diff = nRows * nCols - nPlots
            # update if it is a better grid
            if (nCols <= self.nColsMax) and ((optDiff is None) or (diff < optDiff)):
                nRowsOpt = nRows
                nColsOpt = nCols
                optDiff = diff

        # if we haven't found a valid grid, use nColsMax as number of cols and adjust number of rows accordingly
        if optDiff is None:
            nRowsOpt = ceil(nPlots, self.nColsMax)
            nColsOpt = self.nColsMax

        return nRowsOpt, nColsOpt

    def plotData(self):
        # set x axis values
        xValues = list(range(len(self.batchLabels)))
        # generate different plots
        for fileName in self.plotAllLayersKeys:
            data = self.containers[fileName]
            self.__plotContainer(data, xValues, xLabel='Batch #', yLabel=fileName,
                                 title='{} over epochs'.format(fileName), fileName=fileName, labelFunc=lambda x: x)

        for fileName in self.plotLayersSeparateKeys:
            data = self.containers[fileName]
            # build subplot for all plots
            nPlots = len(data)
            nRows, nCols = self.__findGrid(nPlots)
            fig, ax = plt.subplots(nrows=nRows, ncols=nCols)
            axRow, axCol = 0, 0
            # add each layer alphas data to plot
            for i, layerData in enumerate(data):
                self.__plotContainer(layerData, xValues, xLabel='Batch #', fileName='{}_{}'.format(fileName, i),
                                     title='{} --layer:[{}]-- over epochs'.format(fileName, i), yLabel=fileName,
                                     labelFunc=lambda x: int(self.layersBitwidths[i][x].item()),
                                     axOther=ax[axRow, axCol])
                # update next axes indices
                axCol = (axCol + 1) % nCols
                if axCol == 0:
                    axRow += 1
            # save fig
            self.__setFigProperties(fig, fileName, figSize=(30, 15))

    def plotBops(self, layersList):
        # create plot
        fig, ax = plt.subplots(nrows=1, ncols=1)
        # init axis values
        xValues = [[[] for _ in range(layer.numOfOps())] for layer in layersList]
        yValues = [[[] for _ in range(layer.numOfOps())] for layer in layersList]
        # init y axis max value
        yMax = 0
        for i, layer in enumerate(layersList):
            for input_bitwidth in layer.bops.keys():
                for j, bops in enumerate(layer.bops[input_bitwidth]):
                    v = bops / 1E6
                    xValues[i][j].append(i)
                    yValues[i][j].append(v)
                    ax.annotate('input:[{}]'.format(input_bitwidth), (i, v))
                    yMax = max(yMax, v)

        colors = {}
        for i, (xLayerValues, yLayerValues) in enumerate(zip(xValues, yValues)):
            for j, (x, y) in enumerate(zip(xLayerValues, yLayerValues)):
                label = int(self.layersBitwidths[i][j].item())
                if label in colors.keys():
                    ax.plot(x, y, 'o', label=label, color=colors[label])
                else:
                    info = ax.plot(x, y, 'o', label=label)
                    colors[label] = info[0].get_color()

        yMax *= 1.1
        self.__setPlotProperties(fig, ax, xLabel='Layer #', yLabel='M-bops', yMax=yMax, title='bops per op in layer',
                                 fileName=self.bopsKey)
