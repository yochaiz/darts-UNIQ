from torch import zeros
from torch.optim import SGD
from torch.nn import functional as F

from .regime import TrainRegime, infer, trainWeights, initTrainLogger, save_checkpoint
from cnn.architect import Architect
from cnn.model_replicator import ModelReplicator


class Replicator(ModelReplicator):
    def __init__(self, model, modelClass, args):
        super(Replicator, self).__init__(model, modelClass, args)

    def getModel(self, args):
        return args[0]

    def buildArgs(self, inputPerGPU, targetPerGPU, layersIndicesPerModel):
        args = ((cModel, inputPerGPU[gpu], targetPerGPU[gpu], layersIndices)
                for layersIndices, (cModel, gpu) in zip(layersIndicesPerModel, self.replications))

        return args

    def lossPerReplication(self, args):
        cModel, input, target, layersIndices = args

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
            # # turn off coin toss for this layer
            # layer.alphas.requires_grad = False
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
                    logits = cModel.forward(input)
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

            # # turn in coin toss for this layer
            # layer.alphas.requires_grad = True
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


class AlphasWeightsLoop(TrainRegime):
    def __init__(self, args, model, modelClass, logger):
        super(AlphasWeightsLoop, self).__init__(args, model, modelClass, logger)

        # init model replicator
        replicator = Replicator(model, modelClass, args)
        # init architect
        self.architect = Architect(replicator, args)

    def train(self):
        epoch = self.epoch
        model = self.model
        args = self.args
        # init number of epochs
        nEpochs = self.model.nLayers()
        # init validation best precision value
        best_prec1 = 0.0

        for epoch in range(epoch + 1, epoch + nEpochs + 1):
            # turn on alphas
            model.turnOnAlphas()
            print('========== Epoch:[{}] =============='.format(epoch))
            # init epoch train logger
            trainLogger = initTrainLogger(str(epoch), self.trainFolderPath, args.propagate)
            # set loggers dictionary
            loggersDict = dict(train=trainLogger, main=self.logger)
            # train alphas
            self.trainAlphas(self.search_queue, model, self.architect, epoch, loggersDict)

            # validation on current optimal model
            valid_acc = infer(self.valid_queue, model, model.evalMode, self.cross_entropy, epoch, loggersDict)

            # save model checkpoint
            is_best = valid_acc > best_prec1
            best_prec1 = max(valid_acc, best_prec1)
            save_checkpoint(self.trainFolderPath, model, epoch, best_prec1, is_best)

            ## train weights ##
            # create epoch train weights folder
            epochName = '{}_w'.format(epoch)
            epochFolderPath = '{}/{}'.format(self.trainFolderPath, epochName)
            # turn off alphas
            model.turnOffAlphas()
            # turn on weights gradients
            model.turnOnWeights()
            # init optimizer
            optimizer = SGD(model.parameters(), args.learning_rate,
                            momentum=args.momentum, weight_decay=args.weight_decay)
            # train weights with 1 epoch per stage
            wEpoch = 1
            switchStageFlag = True
            while switchStageFlag:
                # init epoch train logger
                trainLogger = initTrainLogger(str(wEpoch), epochFolderPath, args.propagate)
                # train stage weights
                trainWeights(self.train_queue, model, model.choosePathByAlphas, self.cross_entropy, optimizer,
                             args.grad_clip, wEpoch, dict(train=trainLogger))
                # switch stage
                switchStageFlag = model.switch_stage(trainLogger)
                # update epoch number
                wEpoch += 1

            # init epoch train logger for last epoch
            trainLogger = initTrainLogger(str(wEpoch), epochFolderPath, args.propagate)
            # set loggers dictionary
            loggersDict = dict(train=trainLogger, main=self.logger)
            # last weights training epoch we want to log also to main logger
            trainWeights(self.train_queue, model, model.choosePathByAlphas, self.cross_entropy, optimizer,
                         args.grad_clip, epochName, loggersDict)
            # validation on optimal model
            valid_acc = infer(self.valid_queue, model, model.evalMode, self.cross_entropy, epochName, loggersDict)
            # # calc validation accuracy & loss on uniform model
            # infer(self.valid_queue, model, model.uniformMode, self.cross_entropy, 'Uniform', dict(main=self.logger))

            # save model checkpoint
            is_best = valid_acc > best_prec1
            best_prec1 = max(valid_acc, best_prec1)
            save_checkpoint(self.trainFolderPath, model, epoch, best_prec1, is_best)


class TripleAlphaReplicator(ModelReplicator):
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
        # init loss samples list for ALL alphas
        allLossSamples = []
        # init how many samples per alpha
        nSamplesPerAlpha = cModel.nSamplesPerAlpha
        nSamplesPerAlpha1 = int(nSamplesPerAlpha / 2)
        nSamplesPerAlpha2 = nSamplesPerAlpha - nSamplesPerAlpha1
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
                # select the specific alpha in this layer
                layer.curr_alpha_idx = i

                # sample path based on alphas distribution
                alphaLossSamples1 = []
                for _ in range(nSamplesPerAlpha1):
                    logits = cModel.forward(input)
                    alphaLossSamples1.append(cModel._criterion(logits, target, cModel.countBops()).detach())

                # calc alpha average loss
                alphaAvgLoss1 = sum(alphaLossSamples1) / nSamplesPerAlpha1

                # set previous & following layer alpha as current layer alpha
                prevLayer = cModel.layersList[layerIdx - 1] if layerIdx - 1 >= 0 else None
                nextLayer = cModel.layersList[layerIdx + 1] if layerIdx + 1 < cModel.nLayers() else None
                for l in [prevLayer, nextLayer]:
                    if l:
                        # turn off coin toss for previous layer
                        l.alphas.requires_grad = False
                        # select the specific alpha in previous layer
                        l.curr_alpha_idx = i

                # sample paths
                alphaLossSamples2 = []
                for _ in range(nSamplesPerAlpha2):
                    logits = cModel.forward(input)
                    alphaLossSamples2.append(cModel._criterion(logits, target, cModel.countBops()).detach())

                # calc alpha average loss
                alphaAvgLoss2 = sum(alphaLossSamples2) / nSamplesPerAlpha2

                for l in [prevLayer, nextLayer]:
                    if l:
                        # multiply by layer alphas
                        lProbs = F.softmax(l.alphas, dim=-1)
                        alphaAvgLoss2 *= lProbs[i]
                        # turn on alphas grads in layer
                        l.alphas.requires_grad = True

                # calc merged avg loss of alphaAvgLoss1,alphaAvgLoss2
                # add current alpha loss samples to all loss samples list
                allLossSamples.extend(alphaLossSamples1)
                allLossSamples.extend(alphaLossSamples2)
                # calc alpha average loss
                alphaAvgLoss = (alphaAvgLoss1 + alphaAvgLoss2) * 0.5
                layerAlphasGrad[i] = alphaAvgLoss
                # add alpha loss to total loss
                totalLoss += (alphaAvgLoss * probs[i])

                # calc loss samples variance
                lossVariance = [((x - alphaAvgLoss) ** 2) for x in alphaLossSamples1] + \
                               [((x - alphaAvgLoss) ** 2) for x in alphaLossSamples2]
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
