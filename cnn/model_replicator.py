from numpy import array_split
from multiprocessing import Pool

from torch import zeros
from torch.nn import functional as F


class ModelReplicator:
    def __init__(self, modelClass, args, crit):
        self.modelClass = modelClass
        self.args = args
        self.crit = crit

    def copyModel(self, model):
        # create new model instance
        cModel = self.modelClass(self.crit, self.args.bitwidth, self.args.kernel, self.args.bopsCounter)
        cModel = cModel.cuda()
        # copy src model weights
        cModel.load_state_dict(model.state_dict())
        # copy src model alphas
        for cLayer, mLayer in zip(cModel.layersList, model.layersList):
            cLayer.alphas.data.copy_(mLayer.alphas.data)

        return cModel

    def loss(self, model, input, target, nCopies):
        if nCopies > 0:
            # init total loss
            totalLoss = 0.0
            # init number of total samples
            nSamplesTotal = 0
            # split layers indices between models
            layersIndicesPerModel = array_split(range(model.nLayers()), nCopies)
            with Pool(processes=nCopies, maxtasksperchild=1) as pool:
                # run pool
                results = [pool.apply(self.lossPerReplication, args=(model, input, target, layersIndices,))
                           for layersIndices in layersIndicesPerModel]
                # process returned results
                for alphasGrad, layersIndices, partialLoss, nSamplesPartial in results:
                    # calc total loss & total number of samples
                    totalLoss += partialLoss
                    nSamplesTotal += nSamplesPartial
                    # update layers alphas gradients
                    for layerAlphasGrads, layerIdx in zip(alphasGrad, layersIndices):
                        alphas = model.layersList[layerIdx].alphas
                        alphas.grad = layerAlphasGrads

                # average total loss
                totalLoss /= nSamplesTotal

                # subtract average total loss from every alpha gradient
                for layer in model.layersList:
                    layer.alphas.grad -= totalLoss
                    # calc layer alphas softmax
                    probs = F.softmax(layer.alphas)
                    # multiply each grad by its probability
                    layer.alphas.grad *= probs

                return totalLoss

    def lossPerReplication(self, model, input, target, layersIndices):
        # create model replication (copy)
        print('www:{}'.format(layersIndices))
        cModel = self.copyModel(model)
        # init total loss
        totalLoss = 0.0
        # init number of total samples
        nSamplesTotal = 0
        # init how many samples per alpha
        # init layers alphas grad
        alphasGrad = []
        nSamplesPerAlpha = 50
        for i in layersIndices:
            layer = cModel.layersList[i]
            # turn off coin toss for this layer
            layer.alphas.requires_grad = False
            # init layer alphas gradient
            layerAlphasGrad = zeros(len(layer.alphas)).cuda()

            for i, alpha in enumerate(layer.alphas):
                # select the specific alpha in this layer
                layer.curr_alpha_idx = i
                # init alpha loss
                alphaLoss = 0.0
                for _ in range(nSamplesPerAlpha):
                    logits = cModel.forward(input)
                    alphaLoss += cModel._criterion(logits, target, cModel.countBops()).detach()

                # add alpha loss to total loss
                totalLoss += alphaLoss
                # update number of total samples
                nSamplesTotal += nSamplesPerAlpha
                # update alpha average loss (S_l,i)
                layerAlphasGrad[i] = (alphaLoss / nSamplesPerAlpha)

            # turn in coin toss for this layer
            layer.alphas.requires_grad = True
            # add layer alphas grad to container
            alphasGrad.append(layerAlphasGrad)

        return alphasGrad, layersIndices, totalLoss, nSamplesTotal

        # def copyModel(self, model, nCopies):
    #     if nCopies > 0:
    #         # init list of relications
    #         replications = []
    #         # load model state dict
    #         modelStateDict = model.state_dict()
    #         for _ in range(nCopies):
    #             # create new model instance
    #             cModel = self.modelConstructor()
    #             # copy src model weights
    #             cModel.load_state_dict(modelStateDict)
    #             # copy src model alphas
    #             for cLayer, mLayer in zip(cModel.layersList, model.layersList):
    #                 cLayer.alphas.copy_(mLayer.alphas)
    #
    #             replications.append(cModel)
    #
    #         return replications

    # def loss(self, model, input, target, nCopies):
    #     if nCopies > 0:
    #         # create replications
    #         replications = self.copyModel(model, nCopies)
    #         # add model to replications in order to use it as well
    #         replications.append(model)
    #         # split layers indices between models
    #         layersIndicesPerModel = array_split(range(model.nLayers()), len(replications))
    #         # create processes
    #         for cModel, layerIndices in zip(replications, layersIndicesPerModel):
    #             p = Process(target=ModelReplicator.lossPerReplication, args=(cModel, input, target, layerIndices,))
    #             p.start()
    #             p.join()
