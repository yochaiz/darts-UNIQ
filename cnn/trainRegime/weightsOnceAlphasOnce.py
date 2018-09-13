from .regime import TrainRegime, trainAlphas, infer, trainWeights
from cnn.utils import initTrainLogger, save_checkpoint
from torch.optim import SGD
from cnn.architect import Architect


class WeightsOnceAlphasOnce(TrainRegime):
    def __init__(self, args, model, modelClass,logger):
        super(WeightsOnceAlphasOnce, self).__init__(args, model, modelClass, logger)

        # init architect
        self.architect = Architect(model, modelClass, args)


    def train(self):
        # init number of epochs
        nEpochs = self.model.nLayers()
        # init validation best precision value
        best_prec1 = 0.0

        for epoch in range(self.epoch + 1, self.epoch + nEpochs + 1):
            # turn on alphas
            self.model.turnOnAlphas()
            print('========== Epoch:[{}] =============='.format(epoch))
            # init epoch train logger
            trainLogger = initTrainLogger(str(epoch), self.trainFolderPath, self.args.propagate)
            # set loggers dictionary
            loggersDict = dict(train=trainLogger, main=self.logger)
            # train alphas
            trainAlphas(self.search_queue, self.model, self.architect, epoch, loggersDict)

            # validation on current optimal model
            valid_acc = infer(self.valid_queue, self.model, self.model.evalMode, self.cross_entropy, epoch, loggersDict)
            # update architecture learning rate
            if self.updateLR == 1:
                self.architect.lr = max(self.architect.lr / 10, 0.001)
            self.updateLR = (self.updateLR + 1) % 2

            # save model checkpoint
            is_best = valid_acc > best_prec1
            best_prec1 = max(valid_acc, best_prec1)
            save_checkpoint(self.trainFolderPath, self.model, epoch, best_prec1, is_best)

            ## train weights ##
            trainLogger.info('===== train weights =====')
            # turn off alphas
            self.model.turnOffAlphas()
            # turn on weights gradients
            self.model.turnOnWeights()
            # init optimizer
            optimizer = SGD(self.model.parameters(), self.args.learning_rate,
                            momentum=self.args.momentum, weight_decay=self.args.weight_decay)
            # train weights with 1 epoch per stage
            wEpoch = 1
            switchStageFlag = True
            while switchStageFlag:
                trainWeights(self.train_queue, self.model, self.model.choosePathByAlphas, self.cross_entropy, optimizer,
                             self.args.grad_clip, wEpoch, dict(train=trainLogger))
                # switch stage
                switchStageFlag = self.model.switch_stage(trainLogger)
                wEpoch += 1

            # set weights training epoch string
            wEpoch = '{}_w'.format(epoch)
            # last weights training epoch we want to log also to main logger
            trainWeights(self.train_queue, self.model, self.model.choosePathByAlphas, self.cross_entropy, optimizer,
                         self.args.grad_clip, wEpoch, loggersDict)
            # validation on optimal model
            infer(self.valid_queue, self.model, self.model.evalMode, self.cross_entropy, wEpoch, loggersDict)
            # calc validation accuracy & loss on uniform model
            infer(self.valid_queue, self.model, self.model.uniformMode, self.cross_entropy, 'Uniform', dict(main=self.logger))
