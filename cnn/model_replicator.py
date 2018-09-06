from numpy import array_split
from multiprocessing import Pool

from torch import zeros
from torch.nn import functional as F


class ModelReplicator:
    def __init__(self, model, modelClass, args):
        self.gpuIDs = args.gpu
        # init replications list
        self.replications = []
        # count number of replications and assign each of them to a GPU
        gpus = [gpu for gpu in args.gpu for _ in range(args.nCopies)]
        # load model state dict
        modelStateDict = model.state_dict()
        # create replications
        for gpu in gpus:
            # create model new instance
            cModel = modelClass(args.lmbda, args.maxBops, args.bitwidth, args.kernel, args.bopsCounter)
            # set model to cuda on specific GPU
            cModel = cModel.cuda(gpu)
            # set model criterion to its GPU
            cModel._criterion.cuda(gpu)
            cModel._criterion.search_loss.cuda(gpu)
            # load model weights
            cModel.load_state_dict(modelStateDict)
            # add model to replications
            self.replications.append((cModel, gpu))

    def loss(self, model, input, target):
        nCopies = len(self.replications)
        if nCopies > 0:
            # clone input & target to all GPUs
            inputPerGPU = {}
            targetPerGPU = {}
            for id in self.gpuIDs:
                inputPerGPU[id] = input if (id == input.device.index) else input.clone().cuda(id)
                targetPerGPU[id] = target if (id == target.device.index) else target.clone().cuda(id)
            # init total loss
            totalLoss = 0.0
            # split layers indices between models
            layersIndicesPerModel = array_split(range(model.nLayers()), nCopies)

            args = ((model, cModel, inputPerGPU[gpu], targetPerGPU[gpu], layersIndices, gpu)
                    for layersIndices, (cModel, gpu) in zip(layersIndicesPerModel, self.replications))

            with Pool(processes=nCopies, maxtasksperchild=1) as pool:
                results = pool.map(self.lossPerReplication, args)

            # process returned results
            for alphasGrad, layersIndices, partialLoss in results:
                # calc total loss & total number of samples
                totalLoss += partialLoss
                # update layers alphas gradients
                for layerAlphasGrads, layerIdx in zip(alphasGrad, layersIndices):
                    alphas = model.layersList[layerIdx].alphas
                    alphas.grad = layerAlphasGrads

            # average total loss
            totalLoss /= model.nLayers()

            # subtract average total loss from every alpha gradient
            for layer in model.layersList:
                layer.alphas.grad -= totalLoss
                # calc layer alphas softmax
                probs = F.softmax(layer.alphas, dim=-1)
                # multiply each grad by its probability
                layer.alphas.grad *= probs

            return totalLoss

    # def lossPerReplication(self, model, cModel, input, target, layersIndices, gpu):
    def lossPerReplication(self, args):
        model, cModel, input, target, layersIndices, gpu = args
        # copy model alphas
        for cLayer, mLayer in zip(cModel.layersList, model.layersList):
            cLayer.alphas.data.copy_(mLayer.alphas.data)

        # init total loss
        totalLoss = 0.0
        # init how many samples per alpha
        nSamplesPerAlpha = 50
        # init layers alphas grad
        alphasGrad = []
        for _ in layersIndices:
            orgLayer = model.layersList[i]
            layer = cModel.layersList[i]
            # turn off coin toss for this layer
            layer.alphas.requires_grad = False
            # init layer alphas gradient
            layerAlphasGrad = zeros(len(layer.alphas)).cuda()
            # calc layer alphas softmax
            probs = F.softmax(layer.alphas, dim=-1)

            for i, alpha in enumerate(layer.alphas):
                # select the specific alpha in this layer
                layer.curr_alpha_idx = i
                # init alpha loss
                alphaLoss = 0.0
                # init loss samples list
                lossSamples = []
                for _ in range(nSamplesPerAlpha):
                    logits = cModel.forward(input)
                    # alphaLoss += cModel._criterion(logits, target, cModel.countBops()).detach()
                    lossSamples.append(cModel._criterion(logits, target, cModel.countBops()).detach())

                # update alpha average loss
                # alphaLoss /= nSamplesPerAlpha
                # calc alpha average loss
                alphaAvgLoss = sum(lossSamples) / nSamplesPerAlpha
                layerAlphasGrad[i] = alphaAvgLoss
                # add alpha loss to total loss
                totalLoss += (alphaAvgLoss * probs[i])

                # calc loss samples variance
                lossVariance = [((x - alphaAvgLoss) ** 2) for x in lossSamples]
                orgLayer.alphasLossVariance.data[i] = sum(lossVariance) / (nSamplesPerAlpha - 1)

            # turn in coin toss for this layer
            layer.alphas.requires_grad = True
            # add layer alphas grad to container
            alphasGrad.append(layerAlphasGrad)

        return alphasGrad, layersIndices, totalLoss.cuda()

    # def copyModel(self, model, gpu):
    #     # create new model instance
    #     cModel = self.modelClass(self.crit, self.args.bitwidth, self.args.kernel, self.args.bopsCounter)
    #     cModel = cModel.cuda(gpu)
    #     # copy src model weights
    #     cModel.load_state_dict(model.state_dict())
    #     # copy src model alphas
    #     for cLayer, mLayer in zip(cModel.layersList, model.layersList):
    #         cLayer.alphas.data.copy_(mLayer.alphas.data)
    #
    #     return cModel

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
