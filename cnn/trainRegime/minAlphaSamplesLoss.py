from .regime import TrainRegime, infer
from torch.autograd.variable import Variable
from time import time
from cnn.utils import  logDominantQuantizedOp, AvgrageMeter, save_checkpoint, initTrainLogger
from numpy import argmin

class MinimalAlphaSamplesLoss(TrainRegime):
    def __init__(self, args, model, modelClass, logger):
        super(MinimalAlphaSamplesLoss, self).__init__(args, model, modelClass, logger)

        self.nSamplesPerAlpha = model.nSamplesPerAlpha


    def trainSamplesAlphas(self,loggers):
        # loss_container = AvgrageMeter()

        trainLogger = loggers.get('train')

        self.model.train()

        nBatches = len(self.search_queue)
        allAlphaAvgLoss = []

        for step, (input, target) in enumerate(self.search_queue):
            startTime = time()

            input = Variable(input, requires_grad=False).cuda()
            target = Variable(target, requires_grad=False).cuda(async=True)

            self.model.trainMode()
            currAllAlphaAvgLoss = self.alphaAvgLoss(input, target)

            if not allAlphaAvgLoss:
                allAlphaAvgLoss = currAllAlphaAvgLoss #for first iteration
            else:
                for j, layer in enumerate(self.model.layersList):
                    allAlphaAvgLoss[j] =  [sum(x) for x in zip(currAllAlphaAvgLoss[j], allAlphaAvgLoss[j])]


            # add alphas data to statistics
            # self.model.stats.addBatchData(self.model, self.epoch, step)

            # save alphas to csv
            self.model.save_alphas_to_csv(data=[self.epoch, step])

            endTime = time()

            if trainLogger:
                trainLogger.info('train [{}/{}]  time:[{:.5f}]'
                                 .format(step, nBatches,  endTime - startTime))

            # loss_container.update(loss, input.size(0))

        for j, layer in enumerate(self.model.layersList):
            # choose the alpha with minimum avg loss
            layer.curr_alpha_idx = argmin(allAlphaAvgLoss[j])
            #make curr_alpha_idx to be the maximum alpha so we can use regular eval mode
            self.model.layersList[j].alphas[layer.curr_alpha_idx] = max(layer.alphas) + 1

        # count current optimal model bops
        bopsRatio = self.model.evalMode()

        # log dominant QuantizedOp in each layer
        logDominantQuantizedOp(self.model, k=3, logger=trainLogger)

        # # log accuracy, loss, etc.
        message = ' OptBopsRatio:[{:.3f}] ]' \
            .format(bopsRatio)

        for _, logger in loggers.items():
            logger.info(message)

    def alphaAvgLoss(self, input, target):
        # init loss samples list for ALL alphas
        allAlphaAvgLoss = []
        for j, layer in enumerate(self.model.layersList):
            # turn off coin toss for this layer
            layer.alphas.requires_grad = False
            alphaAvgLoss = []
            for i, alpha in enumerate(layer.alphas):
                # select the specific alpha in this layer
                layer.curr_alpha_idx = i
                # init loss samples list
                alphaLossSamples = []
                for _ in range(self.nSamplesPerAlpha):
                    # forward through some path in model
                    logits = self.model.forward(input)
                    alphaLossSamples.append(self.model._criterion(logits, target, self.model.countBops()).detach())

                # calc alpha average loss
                currAlphaAvgLoss = sum(alphaLossSamples) / self.nSamplesPerAlpha
                alphaAvgLoss.append(currAlphaAvgLoss)

                # calc loss samples variance
                lossVariance = [((x - currAlphaAvgLoss) ** 2) for x in alphaLossSamples]
                lossVariance = sum(lossVariance) / (self.nSamplesPerAlpha - 1)
                # add alpha loss average to statistics
                self.model.stats.containers[self.model.stats.alphaLossAvgKey][j][i].append(currAlphaAvgLoss.item())
                # add alpha loss variance to statistics
                self.model.stats.containers[self.model.stats.alphaLossVarianceKey][j][i].append(lossVariance.item())

            #save all alpha avg loss
            allAlphaAvgLoss.append(alphaAvgLoss)
            # turn in coin toss for this layer
            layer.alphas.requires_grad = True

        return allAlphaAvgLoss




    def train(self):
            # turn on alphas
            self.model.turnOnAlphas()

            trainLogger = initTrainLogger(str(self.epoch), self.trainFolderPath, self.args.propagate)
            # set loggers dictionary
            loggersDict = dict(train=trainLogger, main=self.logger)
            # train alphas
            self.trainSamplesAlphas(loggersDict)

            # validation on current optimal model
            valid_acc = infer(self.valid_queue, self.model, self.model.evalMode, self.cross_entropy, self.epoch, loggersDict)

            # save model checkpoint
            save_checkpoint(self.trainFolderPath, self.model, self.epoch, valid_acc, True)











