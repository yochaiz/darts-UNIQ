from .regime import TrainRegime


class MinimalAlphaSamplesLoss(TrainRegime):
    def __init__(self, args, model, modelClass, logger):
        super(MinimalAlphaSamplesLoss, self).__init__(args, model, modelClass, logger)

        self.nSamplesPerAlpha = model.nSamplesPerAlpha

    def train(self):


        trainLogger = initTrainLogger(str(epoch), trainFolderPath, args.propagate)

        model.train()

        nBatches = len(search_queue)

        for step, (input, target) in enumerate(search_queue):
            startTime = time()

            input = Variable(input, requires_grad=False).cuda()
            target = Variable(target, requires_grad=False).cuda(async=True)

            model.trainMode()
            loss = model._loss(input, target)

            # add alphas data to statistics
            stats.addBatchData(model, nEpoch, step)
            # log dominant QuantizedOp in each layer
            logDominantQuantizedOp(model, k=3, logger=trainLogger)
            # save alphas to csv
            model.save_alphas_to_csv(data=[nEpoch, step])

            endTime = time()

            if trainLogger:
                trainLogger.info('train [{}/{}] arch_loss:[{:.5f}] time:[{:.5f}]'
                                 .format(step, nBatches, loss, endTime - startTime))

        # log accuracy, loss, etc.
        message = 'Epoch:[{}] , arch loss:[{:.3f}] , OptBopsRatio:[{:.3f}] , lr:[{:.5f}]' \
            .format(nEpoch, loss_container.avg, bopsRatio, architect.lr)

        for _, logger in loggers.items():
            logger.info(message)








        # init total loss
        totalLoss = 0.0
        # init loss samples list for ALL alphas
        allLossSamples = []
        for j, layer in enumerate(self.layersList):
            # turn off coin toss for this layer
            layer.alphas.requires_grad = False
            # init layer alphas gradient
            layerAlphasGrad = zeros(len(layer.alphas)).cuda()
            # calc layer alphas softmax
            probs = F.softmax(layer.alphas, dim=-1)

            for i, alpha in enumerate(layer.alphas):
                # select the specific alpha in this layer
                layer.curr_alpha_idx = i
                # init loss samples list
                alphaLossSamples = []
                for _ in range(nSamplesPerAlpha):
                    # forward through some path in model
                    logits = self.forward(input)
                    # alphaLoss += self._criterion(logits, target, self.countBops()).detach()
                    alphaLossSamples.append(self._criterion(logits, target, self.countBops()).detach())

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
                # add alpha loss average to statistics
                self.stats.containers[self.stats.alphaLossAvgKey][j][i].append(alphaAvgLoss.item())
                # add alpha loss variance to statistics
                self.stats.containers[self.stats.alphaLossVarianceKey][j][i].append(lossVariance.item())

            # turn in coin toss for this layer
            layer.alphas.requires_grad = True
            # set layer alphas gradient
            layer.alphas.grad = layerAlphasGrad

            # add gradNorm to statistics
            self.stats.containers[self.stats.gradNormKey][j].append(layerAlphasGrad.norm().item())