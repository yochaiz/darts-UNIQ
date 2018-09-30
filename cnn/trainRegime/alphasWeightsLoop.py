from torch.optim import SGD

from .regime import TrainRegime, infer, trainWeights, initTrainLogger, save_checkpoint
from cnn.architect import Architect
import cnn.gradEstimators as gradEstimators


class AlphasWeightsLoop(TrainRegime):
    def __init__(self, args, model, modelClass, logger):
        super(AlphasWeightsLoop, self).__init__(args, model, modelClass, logger)

        # init model replicator
        replicatorClass = gradEstimators.__dict__[args.grad_estimator]
        replicator = replicatorClass(model, modelClass, args)
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
            save_checkpoint(self.trainFolderPath, model, args, epoch, best_prec1, is_best)

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
                trainLogger = initTrainLogger('{}_{}'.format(epochName, wEpoch), epochFolderPath, args.propagate)
                # train stage weights
                trainWeights(self.train_queue, model, model.choosePathByAlphas, self.cross_entropy, optimizer,
                             args.grad_clip, wEpoch, dict(train=trainLogger))
                # switch stage
                switchStageFlag = model.switch_stage(trainLogger)
                # update epoch number
                wEpoch += 1

            # init epoch train logger for last epoch
            trainLogger = initTrainLogger('{}_{}'.format(epochName, wEpoch), epochFolderPath, args.propagate)
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
            save_checkpoint(self.trainFolderPath, model, args, epoch, best_prec1, is_best)

        # send final email
        self.sendEmail('Final', 0, 0)
