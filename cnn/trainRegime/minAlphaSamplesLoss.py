from time import time
from numpy import argmin

from torch.autograd.variable import Variable
from torch.nn import functional as F

from .regime import TrainRegime, infer
from cnn.utils import AvgrageMeter, logDominantQuantizedOp, save_checkpoint, initTrainLogger


class MinimalAlphaSamplesLoss(TrainRegime):
    def __init__(self, args, model, modelClass, logger):
        super(MinimalAlphaSamplesLoss, self).__init__(args, model, modelClass, logger)

        self.nSamplesPerAlpha = model.nSamplesPerAlpha

    def trainSamplesAlphas(self, loggers):
        loss_container = AvgrageMeter()

        trainLogger = loggers.get('train')
        model = self.model

        model.train()

        nBatches = len(self.search_queue)

        # init list of model alphas loss average
        self.alphasLoss = [[[] for _ in range(layer.numOfOps())] for layer in model.layersList]

        for step, (input, target) in enumerate(self.search_queue):
            startTime = time()

            input = Variable(input, requires_grad=False).cuda()
            target = Variable(target, requires_grad=False).cuda(async=True)

            model.trainMode()
            batchLoss = self.calcBatchAlphaAvgLoss(input, target)

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
        for i, layerLoss in enumerate(self.alphasLoss):
            layerLossNew = []
            for j, alphaLoss in enumerate(layerLoss):
                avgLoss = (sum(alphaLoss) / len(alphaLoss)).item()
                layerLossNew.append(avgLoss)

            self.alphasLoss[i] = layerLossNew
            if trainLogger:
                trainLogger.info('Layer [{}]: {}'.format(i, layerLossNew))
        if trainLogger:
            trainLogger.info('===============================================')

        # select optimal alpha (lowest average loss) in each layer
        for layer, layerLoss in zip(model.layersList, self.alphasLoss):
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
                for _ in range(self.nSamplesPerAlpha):
                    # forward through some path in model
                    logits = model.forward(input)
                    alphaLossSamples.append(model._criterion(logits, target, model.countBops()).detach())

                # calc current alpha batch average loss
                alphaAvgLoss = sum(alphaLossSamples) / len(alphaLossSamples)
                # add current alpha batch average loss to container
                self.alphasLoss[j][i].append(alphaAvgLoss)
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

        return totalLoss

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
