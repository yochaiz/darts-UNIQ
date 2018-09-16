from time import time
from numpy import argmin

from torch.autograd.variable import Variable
from torch.nn import functional as F

from .regime import TrainRegime, infer
from cnn.utils import AvgrageMeter, logDominantQuantizedOp, save_checkpoint, initTrainLogger
from cnn.model_replicator import ModelReplicator


class Replicator(ModelReplicator):
    def __init__(self, model, modelClass, args):
        super(Replicator, self).__init__(model, modelClass, args)

    def buildArgs(self, inputPerGPU, targetPerGPU, layersIndicesPerModel):
        args = ((cModel, inputPerGPU[gpu], targetPerGPU[gpu], layersIndices)
                for layersIndices, (cModel, gpu) in zip(layersIndicesPerModel, self.replications))

        return args

    def lossPerReplication(self, args):
        cModel, input, target, layersIndices = args

        # init total loss
        totalLoss = 0.0
        # init alphas average loss list, each element is (layerIdx,alphaIdx,alphaAvgLoss)
        alphasLoss = []
        for layerIdx in layersIndices:
            layer = cModel.layersList[layerIdx]
            # turn off coin toss for this layer
            layer.alphas.requires_grad = False
            for i, alpha in enumerate(layer.alphas):
                # calc layer alphas softmax
                probs = F.softmax(layer.alphas, dim=-1)
                # select the specific alpha in this layer
                layer.curr_alpha_idx = i
                # init loss samples list
                alphaLossSamples = []
                for _ in range(cModel.nSamplesPerAlpha):
                    # forward through some path in model
                    logits = cModel.forward(input)
                    alphaLossSamples.append(cModel._criterion(logits, target, cModel.countBops()).detach())

                # calc current alpha batch average loss
                alphaAvgLoss = sum(alphaLossSamples) / len(alphaLossSamples)
                # add current alpha batch average loss to container
                alphasLoss.append((layerIdx, i, alphaAvgLoss))
                # add alpha loss to total loss
                totalLoss += (alphaAvgLoss * probs[i])

                # calc loss samples variance
                lossVariance = [((x - alphaAvgLoss) ** 2) for x in alphaLossSamples]
                lossVariance = sum(lossVariance) / (len(lossVariance) - 1)
                # add alpha loss average to statistics
                cModel.stats.containers[cModel.stats.alphaLossAvgKey][layerIdx][i].append(alphaAvgLoss.item())
                # add alpha loss variance to statistics
                cModel.stats.containers[cModel.stats.alphaLossVarianceKey][layerIdx][i].append(lossVariance.item())

            # turn in coin toss for this layer
            layer.alphas.requires_grad = True

        return totalLoss.cuda(), alphasLoss

    def processResults(self, model, results):
        # init list of model alphas loss average
        alphasLoss = [[[] for _ in range(layer.numOfOps())] for layer in model.layersList]
        # init total loss
        totalLoss = 0.0
        for partialLoss, layersAlphasLoss in results:
            # sum total loss
            totalLoss += partialLoss
            # update alphas average loss
            for layerIdx, alphaIdx, avgLoss in layersAlphasLoss:
                alphasLoss[layerIdx][alphaIdx].append(avgLoss)

        # average total loss
        totalLoss /= model.nLayers()

        return totalLoss, alphasLoss


class MinimalAlphaSamplesLoss(TrainRegime):
    def __init__(self, args, model, modelClass, logger):
        super(MinimalAlphaSamplesLoss, self).__init__(args, model, modelClass, logger)

        self.replicator = Replicator(model, modelClass, args)

    def trainSamplesAlphas(self, loggers):
        loss_container = AvgrageMeter()

        trainLogger = loggers.get('train')
        model = self.model

        model.train()

        nBatches = len(self.search_queue)

        # init list of model alphas loss average
        alphasLoss = [[[] for _ in range(layer.numOfOps())] for layer in model.layersList]

        for step, (input, target) in enumerate(self.search_queue):
            startTime = time()

            input = Variable(input, requires_grad=False).cuda()
            target = Variable(target, requires_grad=False).cuda(async=True)

            model.trainMode()
            batchLoss, batchAlphasLoss = self.calcBatchAlphaAvgLoss(input, target)
            # batchLoss, batchAlphasLoss = self.replicator.loss(model, input, target)

            # copy batch alphas average loss to main alphas average loss
            for batchLayerLoss, layerLoss in zip(batchAlphasLoss, alphasLoss):
                for batchAlphaLoss, alphaLoss in zip(batchLayerLoss, layerLoss):
                    alphaLoss.extend(batchAlphaLoss)

            # save loss to container
            loss_container.update(batchLoss, input.size(0))
            # add alphas data to statistics
            model.stats.addBatchData(model, self.epoch, step)

            endTime = time()

            if trainLogger:
                trainLogger.info('train [{}/{}] arch_loss:[{:.5f}] time:[{:.5f}]'
                                 .format(step, nBatches, loss_container.avg, (endTime - startTime)))

        if trainLogger:
            trainLogger.info('==== Average loss per layer ====')
        # average all losses per alpha over all batches
        for i, layerLoss in enumerate(alphasLoss):
            layerLossNew = []
            for j, alphaLoss in enumerate(layerLoss):
                print('alphaLoss length:[{}]'.format(len(alphaLoss)))
                avgLoss = (sum(alphaLoss) / len(alphaLoss)).item()
                layerLossNew.append(avgLoss)

            alphasLoss[i] = layerLossNew
            if trainLogger:
                trainLogger.info('Layer [{}]: {}'.format(i, layerLossNew))
        if trainLogger:
            trainLogger.info('===============================================')

        # select optimal alpha (lowest average loss) in each layer
        for layer, layerLoss in zip(model.layersList, alphasLoss):
            # choose the alpha with minimum avg loss
            layer.curr_alpha_idx = argmin(layerLoss)
            # make curr_alpha_idx to be the maximum alpha so we can use regular eval mode
            layer.alphas[layer.curr_alpha_idx] = max(layer.alphas) + 1

        # count current optimal model bops
        bopsRatio = model.evalMode()

        # log accuracy, loss, etc.
        message = 'Epoch:[{}] , arch_loss:[{:.3f}] , OptBopsRatio:[{:.3f}]' \
            .format(self.epoch, loss_container.avg, bopsRatio)

        for _, logger in loggers.items():
            logger.info(message)

        # log dominant QuantizedOp in each layer
        logDominantQuantizedOp(model, k=3, logger=trainLogger)

    # calc average samples loss for each alpha on given batch
    def calcBatchAlphaAvgLoss(self, input, target):
        model = self.model
        # init list of model alphas loss average
        alphasLoss = [[[] for _ in range(layer.numOfOps())] for layer in model.layersList]
        # init total loss
        totalLoss = 0.0
        for j, layer in enumerate(model.layersList):
            # turn off coin toss for this layer
            layer.alphas.requires_grad = False
            for i, alpha in enumerate(layer.alphas):
                # calc layer alphas softmax
                probs = F.softmax(layer.alphas, dim=-1)
                # select the specific alpha in this layer
                layer.curr_alpha_idx = i
                # init loss samples list
                alphaLossSamples = []
                for _ in range(model.nSamplesPerAlpha):
                    # forward through some path in model
                    logits = model.forward(input)
                    alphaLossSamples.append(model._criterion(logits, target, model.countBops()).detach())

                # calc current alpha batch average loss
                alphaAvgLoss = sum(alphaLossSamples) / len(alphaLossSamples)
                # add current alpha batch average loss to container
                alphasLoss[j][i].append(alphaAvgLoss)
                # add alpha loss to total loss
                totalLoss += (alphaAvgLoss * probs[i])

                # calc loss samples variance
                lossVariance = [((x - alphaAvgLoss) ** 2) for x in alphaLossSamples]
                lossVariance = sum(lossVariance) / (len(lossVariance) - 1)
                # add alpha loss average to statistics
                model.stats.containers[model.stats.alphaLossAvgKey][j][i].append(alphaAvgLoss.item())
                # add alpha loss variance to statistics
                model.stats.containers[model.stats.alphaLossVarianceKey][j][i].append(lossVariance.item())

            # turn in coin toss for this layer
            layer.alphas.requires_grad = True

        # average total loss
        totalLoss /= model.nLayers()

        return totalLoss, alphasLoss

    def train(self):
        model = self.model
        # turn on alphas
        model.turnOnAlphas()

        trainLogger = initTrainLogger(str(self.epoch), self.trainFolderPath, self.args.propagate)
        # set loggers dictionary
        loggersDict = dict(train=trainLogger, main=self.logger)
        # train alphas
        self.trainSamplesAlphas(loggersDict)

        # validation on current optimal model
        valid_acc = infer(self.valid_queue, model, model.evalMode, self.cross_entropy, self.epoch, loggersDict)

        # save model checkpoint
        save_checkpoint(self.trainFolderPath, model, self.epoch, valid_acc, True)
