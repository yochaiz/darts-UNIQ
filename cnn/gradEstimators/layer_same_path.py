from .random_path import RandomPath, set_device, no_grad

from torch import ones, tensor
from torch.nn import functional as F


# select same paths to calculate loss for a layer.

class LayerSamePath(RandomPath):
    def __init__(self, model, modelClass, args, logger):
        super(LayerSamePath, self).__init__(model, modelClass, args, logger)

    def lossPerReplication(self, args):
        cModel, input, target, nSamples, gpu = args
        # switch to process GPU
        set_device(gpu)
        assert (cModel.training is False)

        # init samples data list, each elements is a tuple (loss,partition)
        samplesData = []

        with no_grad():
            # calc losses and add to list
            for _ in range(nSamples):
                # choose path in model based on alphas distribution
                cModel.choosePathByAlphas()
                # forward input in model
                logits = cModel(input)
                # calc loss
                loss, crossEntropyLoss, bopsLoss = cModel._criterion(logits, target, cModel.countBops())
                # get sample model partition
                modelPartition = [layer.getCurrentFiltersPartition() for layer in cModel.layersList]
                # add sample data to list
                samplesData.append((loss.item(), crossEntropyLoss.item(), bopsLoss.item(), modelPartition))

        return samplesData

    def processResults(self, model, results):
        # init  samples data  list
        samplesData = []
        # merge all samples losses to same list
        for partialSamplesData in results:
            samplesData.extend(partialSamplesData)

        assert (len(samplesData) == self.nSamples)
        # calc total losses
        totalLoss, crossEntropyLoss, bopsLoss = 0.0, 0.0, 0.0
        for l, c, b, _ in samplesData:
            totalLoss += l
            crossEntropyLoss += c
            bopsLoss += b
        # calc loss average
        lossAvg = totalLoss / self.nSamples
        crossEntropyAvg = crossEntropyLoss / self.nSamples
        bopsAvg = bopsLoss / self.nSamples
        # calc gradient for all alphas
        for layerIdx, layer in enumerate(model.layersList):
            # calc layer alphas softmax
            probs = F.softmax(layer.alphas, dim=-1)
            # grad = E[I_ni*Loss] - E[I_ni]*E[Loss] = v2 - v1
            # calc v1
            v1 = lossAvg * layer.nFilters() * probs
            # calc v2
            v2 = []
            for alphaIdx in range(layer.numOfOps()):
                # calc weighted loss average
                weightedLossAvg = sum([l * p[layerIdx][alphaIdx] for l, _, _, p in samplesData]) / self.nSamples
                v2.append(weightedLossAvg)

            # convert v2 to tensor
            v2 = tensor(v2).type(v1.type())
            # update layer alphas grad
            layer.alphas.grad = v2 - v1

        # add statistics
        stats = model.stats
        # add average
        stats.containers[stats.lossAvgKey][0].append(lossAvg)
        # add variance
        lossVariance = [((l - lossAvg) ** 2) for l, _, _, p in samplesData]
        lossVariance = sum(lossVariance) / (self.nSamples - 1)
        stats.containers[stats.lossVarianceKey][0].append(lossVariance)

        return lossAvg, crossEntropyAvg, bopsAvg

    # def lossPerReplication(self, args):
    #     cModel, input, target, nSamples, gpu = args
    #     # switch to process GPU
    #     set_device(gpu)
    #     assert (cModel.training is False)
    #
    #     with no_grad():
    #         # init total loss
    #         totalLoss = 0.0
    #         # init loss samples list for ALL alphas
    #         allLossSamples = []
    #         # init layers alphas grad
    #         alphasGrad = []
    #         # save stats data
    #         gradNorm = []
    #         alphaLossVariance = []
    #         for layerIdx in layersIndices:
    #             layer = cModel.layersList[layerIdx]
    #             # turn off coin toss for this layer
    #             layer.alphas.requires_grad = False
    #             # init layer alphas gradient
    #             layerAlphasGrad = zeros(layer.numOfOps()).cuda()
    #             # calc layer alphas softmax
    #             probs = F.softmax(layer.alphas, dim=-1)
    #
    #             # init loss samples list for layer alphas
    #             alphaLossSamples = [[] for _ in range(layer.numOfOps())]
    #             # for each sample select path through the specific alpha and calc the path loss
    #             for _ in range(nSamples):
    #                 # choose path in model based on alphas distribution, while current layer alpha is [i]
    #                 cModel.choosePathByAlphas()
    #                 for i in range(layer.numOfOps()):
    #                     # select the specific alpha in this layer
    #                     layer.curr_alpha_idx = i
    #                     # forward input in model
    #                     logits = cModel(input)
    #                     # calc loss
    #                     loss = cModel._criterion(logits, target, cModel.countBops()).detach()
    #                     # print('{} - {:.5f}'.format([layer.getBitwidth() for layer in cModel.layersList], loss))
    #                     # add loss to statistics list
    #                     alphaLossSamples[i].append(loss.item())
    #
    #             # process loss results for layer alphas
    #             for i in range(layer.numOfOps()):
    #                 # add layer alphas loss samples to all loss samples list
    #                 allLossSamples.extend(alphaLossSamples[i])
    #                 # calc alpha average loss
    #                 alphaAvgLoss = tensor(sum(alphaLossSamples[i]) / nSamples).cuda()
    #                 layerAlphasGrad[i] = alphaAvgLoss
    #                 # add alpha loss to total loss
    #                 totalLoss += (alphaAvgLoss * probs[i])
    #
    #                 # calc loss samples variance
    #                 lossVariance = [((x - alphaAvgLoss) ** 2) for x in alphaLossSamples[i]]
    #                 lossVariance = sum(lossVariance) / (nSamples - 1)
    #                 # add alpha loss variance to statistics
    #                 alphaLossVariance.append((layerIdx, i, alphaAvgLoss.item(), lossVariance.item()))
    #
    #             # turn in coin toss for this layer
    #             layer.alphas.requires_grad = True
    #             # add layer alphas grad to container
    #             alphasGrad.append(layerAlphasGrad)
    #             # add gradNorm to statistics
    #             gradNorm.append((layerIdx, layerAlphasGrad.norm().item()))
    #
    #         return alphasGrad, allLossSamples, layersIndices, totalLoss, gradNorm, alphaLossVariance
