from scipy.stats import entropy
from os import makedirs, path

from torch import tensor, float32
import torch.nn.functional as F

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Statistics:
    entropyKey = 'alphas_entropy'
    weightedAvgKey = 'alphas_weighted_average'
    alphaLossVarianceKey = 'alphas_loss_variance'
    alphaLossAvgKey = 'alphas_loss_avg'
    allSamplesLossVarianceKey = 'all_samples_loss_variance'
    allSamplesLossAvgKey = 'all_samples_loss_avg'
    gradNormKey = 'alphas_gradient_norm'
    bopsKey = 'bops'

    # set plot points style
    ptsStyle = 'o'

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
        self.layersBitwidths = [tensor([op.bitwidth[0] for op in layer.ops], dtype=float32).cuda()
                                for layer in layersList]
        # plot bops plot
        self.plotBops(layersList)
        # init containers
        self.containers = {
            self.entropyKey: [[] for _ in range(nLayers)],
            self.weightedAvgKey: [[] for _ in range(nLayers)],
            self.alphaLossVarianceKey: [[[] for _ in range(layer.numOfOps())] for layer in layersList],
            self.alphaLossAvgKey: [[[] for _ in range(layer.numOfOps())] for layer in layersList],
            self.allSamplesLossVarianceKey: [[]],
            self.allSamplesLossAvgKey: [[]],
            self.gradNormKey: [[] for _ in range(nLayers)]
        }
        # map each list we plot for all layers on single plot to filename
        self.plotAllLayersKeys = [self.entropyKey, self.weightedAvgKey, self.allSamplesLossVarianceKey,
                                  self.allSamplesLossAvgKey, self.gradNormKey]
        self.plotLayersSeparateKeys = [self.alphaLossAvgKey, self.alphaLossVarianceKey]

    def addBatchData(self, model, nEpoch, nBatch):
        assert (self.nLayers == model.nLayers())
        # add batch label
        self.batchLabels.append('[{}]_[{}]'.format(nEpoch, nBatch))
        # add data per layer
        for i, layer in enumerate(model.layersList):
            # calc layer alphas probabilities
            probs = F.softmax(layer.alphas, dim=-1).detach()
            # calc entropy
            self.containers[self.entropyKey][i].append(entropy(probs))
            # collect weight bitwidth of each op in layer
            weightBitwidth = self.layersBitwidths[i]
            # calc weighted average of weights bitwidth
            res = probs * weightBitwidth
            res = res.sum().item()
            # add layer weighted average
            self.containers[self.weightedAvgKey][i].append(res)

        # plot data
        self.plotData()

    def __setPlotProperties(self, fig, ax, xLabel, yLabel, yMax, title, fileName):
        # ax.set_xticks(xValues)
        # ax.set_xticklabels(self.batchLabels)
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        ax.set_ylim(ymax=yMax, ymin=0.0)
        ax.set_title(title)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=5, fancybox=True, shadow=True)
        fig.set_size_inches((12, 8))

        # save to file
        fig.savefig('{}/{}.png'.format(self.saveFolder, fileName))
        # close plot
        plt.close()

    def __plotContainer(self, data, xValues, xLabel, yLabel, title, fileName, labelFunc, scale=True):
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
                isPlotEmpty = False
                dataMax = max(dataMax, max(layerData))
                dataSum.append(sum(layerData) / len(layerData))

        if not isPlotEmpty:
            # set yMax
            yMax = dataMax * 1.1
            if scale:
                yMax = min(yMax, (sum(dataSum) / len(dataSum)) * 1.5)

            self.__setPlotProperties(fig, ax, xLabel=xLabel, yLabel=yLabel, yMax=yMax,
                                     title=title, fileName=fileName)

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
            # add each layer alphas data to plot
            for i, layerVariance in enumerate(data):
                self.__plotContainer(layerVariance, xValues, xLabel='Batch #', fileName='{}_{}'.format(fileName, i),
                                     title='{} --layer:[{}]-- over epochs'.format(fileName, i), yLabel=fileName,
                                     labelFunc=lambda x: int(self.layersBitwidths[i][x].item()))

    def plotBops(self, layersList):
        xValues = list(range(len(layersList)))
        data = [[] for _ in range(layersList[0].numOfOps())]
        for layer in layersList:
            for j, bops in enumerate(layer.bops):
                data[j].append(bops / 1E6)

        self.__plotContainer(data, xValues, xLabel='Layer #', yLabel='M-bops', title='bops per op in layer',
                             fileName=self.bopsKey, labelFunc=lambda x: int(self.layersBitwidths[0][x].item()),
                             scale=False)
