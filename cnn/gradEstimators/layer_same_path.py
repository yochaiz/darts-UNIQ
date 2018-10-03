from .random_path import RandomPath

from torch import zeros
from torch.nn import functional as F


# select same paths to calculate loss for a layer.

class LayerSamePath(RandomPath):
    def __init__(self, model, modelClass, args):
        super(LayerSamePath, self).__init__(model, modelClass, args)

    def lossPerReplication(self, args):
        cModel, input, target, layersIndices = args

        assert(cModel.training is False)
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

            # init loss samples list for layer alphas
            alphaLossSamples = [[] for _ in range(layer.numOfOps())]
            for _ in range(nSamplesPerAlpha):
                # choose path in model based on alphas distribution, while current layer alpha is [i]
                cModel.choosePathByAlphas()
                for i, alpha in enumerate(layer.alphas):
                    # select the specific alpha in this layer
                    layer.curr_alpha_idx = i
                    # forward input in model
                    logits = cModel(input)
                    # alphaLoss += cModel._criterion(logits, target, cModel.countBops()).detach()
                    alphaLossSamples[i].append(cModel._criterion(logits, target, cModel.countBops()).detach())

            for i, alpha in enumerate(layer.alphas):
                # add layer alphas loss samples to all loss samples list
                allLossSamples.extend(alphaLossSamples[i])
                # calc alpha average loss
                alphaAvgLoss = sum(alphaLossSamples[i]) / nSamplesPerAlpha
                layerAlphasGrad[i] = alphaAvgLoss
                # add alpha loss to total loss
                totalLoss += (alphaAvgLoss * probs[i])

                # calc loss samples variance
                lossVariance = [((x - alphaAvgLoss) ** 2) for x in alphaLossSamples[i]]
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
