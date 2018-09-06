from scipy.stats import entropy
from os import makedirs, path

from torch import tensor, float32
import torch.nn.functional as F

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Statistics:
    def __init__(self, layersList, nLayers, saveFolder):
        self.nLayers = nLayers
        # create plot folder
        plotFolderPath = '{}/plots'.format(saveFolder)
        if not path.exists(plotFolderPath):
            makedirs(plotFolderPath)

        self.saveFolder = plotFolderPath
        # init list of entropy, weighted average per layer
        self.entropyPerLayer = [[] for _ in range(nLayers)]
        self.weightedAveragePerLayer = [[] for _ in range(nLayers)]
        self.lossVariancePerLayer = [[[] for _ in range(layer.numOfOps())] for layer in layersList]
        # init list of batch labels for y axis
        self.batchLabels = []
        # map each list we plot for all layers on single plot to filename
        self.plotAllLayersMap = {
            'alphas_entropy': self.entropyPerLayer,
            'alphas_weighted_average': self.weightedAveragePerLayer
        }
        # map each list we plot each layer on different plot to filename
        self.plotLayersSeparateMap = {
            'alphas_loss_variance': self.lossVariancePerLayer
        }
        # collect op bitwidth per layer in model
        self.layersBitwidths = [tensor([op.bitwidth[0] for op in layer.ops], dtype=float32).cuda()
                                for layer in layersList]

    def addBatchData(self, model, nEpoch, nBatch):
        assert (self.nLayers == model.nLayers())
        # add batch label
        self.batchLabels.append('[{}]_[{}]'.format(nEpoch, nBatch))
        # add data per layer
        for i, layer in enumerate(model.layersList):
            # calc layer alphas probabilities
            probs = F.softmax(layer.alphas, dim=-1).detach()
            # calc entropy
            self.entropyPerLayer[i].append(entropy(probs))
            # collect weight bitwidth of each op in layer
            weightBitwidth = self.layersBitwidths[i]
            # calc weighted average of weights bitwidth
            res = probs * weightBitwidth
            res = res.sum().item()
            # add layer weighted average
            self.weightedAveragePerLayer[i].append(res)
            # add layer alphas loss variance
            for j, alphaVariance in enumerate(layer.alphasLossVariance):
                self.lossVariancePerLayer[i][j].append(alphaVariance.item())

        # plot data
        self.plotData()

    def plotData(self):
        # set x axis values
        xValues = list(range(len(self.batchLabels)))
        # generate different plots
        for fileName, data in self.plotAllLayersMap.items():
            # create plot
            fig, ax = plt.subplots(nrows=1, ncols=1)
            # add each layer alphas data to plot
            for i, layerData in enumerate(data):
                ax.plot(xValues, layerData, 'o-', label=i)

            # set graph style
            # ax.set_xticks(xValues)
            # ax.set_xticklabels(self.batchLabels)
            ax.set_xlabel('Batch #')
            ax.set_ylabel(fileName)
            ax.set_title('{} over epochs'.format(fileName))
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=5, fancybox=True, shadow=True)

            # save to file
            fig.savefig('{}/{}.png'.format(self.saveFolder, fileName))
            # close plot
            plt.close()

        for fileName, data in self.plotLayersSeparateMap.items():
            # add each layer alphas data to plot
            for i, layerVariance in enumerate(data):
                # create plot
                fig, ax = plt.subplots(nrows=1, ncols=1)
                for j, alphaVariance in enumerate(layerVariance):
                    ax.plot(xValues, alphaVariance, 'o-', label=int(self.layersBitwidths[i][j].item()))

                # set graph style
                # ax.set_xticks(xValues)
                # ax.set_xticklabels(self.batchLabels)
                ax.set_xlabel('Batch #')
                ax.set_ylabel(fileName)
                ax.set_title('{} --layer:[{}]-- over epochs'.format(fileName, i))
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=5, fancybox=True, shadow=True)

                # save to file
                fig.savefig('{}/{}_{}.png'.format(self.saveFolder, fileName, i))
                # close plot
                plt.close()
