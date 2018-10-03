from torch import zeros
from torch.nn import functional as F

from cnn.model_replicator import ModelReplicator


class RandomPath(ModelReplicator):
    def __init__(self, model, modelClass, args):
        super(RandomPath, self).__init__(model, modelClass, args)

    def getModel(self, args):
        return args[0]

    def buildArgs(self, inputPerGPU, targetPerGPU, layersIndicesPerModel):
        args = ((cModel, inputPerGPU[gpu], targetPerGPU[gpu], layersIndices)
                for layersIndices, (cModel, gpu) in zip(layersIndicesPerModel, self.replications))

        return args

    def lossPerReplication(self, args):
        cModel, input, target, layersIndices = args

        cModel.eval()
        # init total loss
        totalLoss = 0.0
        # init loss samples list for ALL alphas
        allLossSamples = []
        # init how many samples per alpha
        nSamplesPerAlpha = cModel.nSamplesPerAlpha
        # init layers alphas grad
        alphasGrad = []
        # save stats data
        gradNorm = []
        alphaLossVariance = []
        for layerIdx in layersIndices:
            layer = cModel.layersList[layerIdx]
            # turn off coin toss for this layer
            layer.alphas.requires_grad = False
            # init layer alphas gradient
            layerAlphasGrad = zeros(len(layer.alphas)).cuda()
            # calc layer alphas softmax
            probs = F.softmax(layer.alphas, dim=-1)

            for i, alpha in enumerate(layer.alphas):
                # # select the specific alpha in this layer
                # layer.curr_alpha_idx = i

                # init loss samples list
                alphaLossSamples = []
                for _ in range(nSamplesPerAlpha):
                    # choose path in model based on alphas distribution, while current layer alpha is [i]
                    cModel.choosePathByAlphas(layerIdx=layerIdx, alphaIdx=i)
                    # forward input in model
                    logits = cModel(input)
                    # alphaLoss += cModel._criterion(logits, target, cModel.countBops()).detach()
                    alphaLossSamples.append(cModel._criterion(logits, target, cModel.countBops()).detach())

                # add current alpha loss samples to all loss samples list
                allLossSamples.extend(alphaLossSamples)
                # calc alpha average loss
                alphaAvgLoss = sum(alphaLossSamples) / nSamplesPerAlpha
                layerAlphasGrad[i] = alphaAvgLoss
                # add alpha loss to total loss
                totalLoss += (alphaAvgLoss * probs[i])

                # calc loss samples variance
                lossVariance = [((x - alphaAvgLoss) ** 2) for x in alphaLossSamples]
                lossVariance = sum(lossVariance) / (nSamplesPerAlpha - 1)
                # add alpha loss variance to statistics
                alphaLossVariance.append((layerIdx, i, alphaAvgLoss.item(), lossVariance.item()))

            # turn in coin toss for this layer
            layer.alphas.requires_grad = True
            # add layer alphas grad to container
            alphasGrad.append(layerAlphasGrad)
            # add gradNorm to statistics
            gradNorm.append((layerIdx, layerAlphasGrad.norm().item()))

        return alphasGrad, allLossSamples, layersIndices, totalLoss.cuda(), gradNorm, alphaLossVariance

    def processResults(self, model, results):
        stats = model.stats
        # init total loss
        totalLoss = 0.0
        # init loss samples list for ALL alphas
        allLossSamples = []
        # process returned results
        for alphasGrad, partialLossSamples, layersIndices, partialLoss, gradNorm, alphaLossVariance in results:
            # add alphas loss samples to all loss samples list
            allLossSamples.extend(partialLossSamples)
            # calc total loss & total number of samples
            totalLoss += partialLoss
            # update statistics
            for layerIdx, v in gradNorm:
                stats.containers[stats.gradNormKey][layerIdx].append(v)
            for layerIdx, j, avg, variance in alphaLossVariance:
                stats.containers[stats.alphaLossAvgKey][layerIdx][j].append(avg)
                stats.containers[stats.alphaLossVarianceKey][layerIdx][j].append(variance)
            # update layers alphas gradients
            for layerAlphasGrads, layerIdx in zip(alphasGrad, layersIndices):
                alphas = model.layersList[layerIdx].alphas
                alphas.grad = layerAlphasGrads

        # average total loss
        totalLoss /= model.nLayers()
        # calc all loss samples average
        nTotalSamples = len(allLossSamples)
        allLossSamplesAvg = sum(allLossSamples) / nTotalSamples
        # calc all loss samples variance
        allLossSamples = [((x - allLossSamplesAvg) ** 2) for x in allLossSamples]
        allLossSamplesVariance = (sum(allLossSamples) / (nTotalSamples - 1)).item()
        # add all samples loss average & variance to statistics
        stats.containers[stats.allSamplesLossAvgKey][0].append(allLossSamplesAvg)
        stats.containers[stats.allSamplesLossVarianceKey][0].append(allLossSamplesVariance)

        # subtract average total loss from every alpha gradient
        for layer in model.layersList:
            layer.alphas.grad -= totalLoss
            # calc layer alphas softmax
            probs = F.softmax(layer.alphas, dim=-1)
            # multiply each grad by its probability
            layer.alphas.grad *= probs

        return totalLoss
