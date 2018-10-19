from torch.optim import SGD

from .regime import TrainRegime, save_checkpoint, HtmlLogger
from cnn.architect import Architect
import cnn.gradEstimators as gradEstimators


class AlphasWeightsLoop(TrainRegime):
    def __init__(self, args, logger):
        super(AlphasWeightsLoop, self).__init__(args, logger)

        # init model replicator
        replicatorClass = gradEstimators.__dict__[args.grad_estimator]
        replicator = replicatorClass(self.model, self.modelClass, args)
        # init architect
        self.architect = Architect(replicator, args)
        # set number of different partitions we want to draw from alphas multinomial distribution in order to estimate their validation accuracy
        self.nValidPartitions = 5

    # run on validation set and add validation data to main data row
    def __inferWithData(self, setModelPathFunc, epoch, loggersDict, dataRow, best_prec1):
        model = self.model
        args = self.args
        # run on validation set
        valid_acc, validData = self.infer(setModelPathFunc, epoch, loggersDict)

        # update epoch
        dataRow[self.epochNumKey] = epoch
        # merge dataRow with validData
        for k, v in validData.items():
            dataRow[k] = v

        # save model checkpoint
        is_best = valid_acc > best_prec1
        best_prec1 = max(valid_acc, best_prec1)
        save_checkpoint(self.trainFolderPath, model, args, epoch, best_prec1, is_best)

        return best_prec1

    def train(self):
        model = self.model
        args = self.args
        logger = self.logger
        # init number of epochs
        nEpochs = self.model.nLayers()
        # init validation best precision value
        best_prec1 = 0.0

        for epoch in range(1, nEpochs + 1):
            print('========== Epoch:[{}] =============='.format(epoch))
            # init epoch train logger
            trainLogger = HtmlLogger(self.trainFolderPath, str(epoch))
            # set loggers dictionary
            loggersDict = dict(train=trainLogger)
            # train alphas
            alphaData = self.trainAlphas(self.search_queue[epoch % args.alphas_data_parts], model, self.architect, epoch, loggersDict)
            # validation on fixed partition by alphas values
            best_prec1 = self.__inferWithData(model.setFiltersByAlphas, epoch, loggersDict, alphaData, best_prec1)
            # add data to main logger table
            logger.addDataRow(alphaData)

            ## train weights ##
            # create epoch train weights folder
            epochName = '{}_w'.format(epoch)
            epochFolderPath = '{}/{}'.format(self.trainFolderPath, epochName)
            # turn off alphas
            model.turnOffAlphas()
            # turn on weights gradients
            model.turnOnWeights()
            # init optimizer
            optimizer = SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
            # train weights with 1 epoch per stage
            wEpoch = 1
            switchStageFlag = True
            while switchStageFlag:
                # init epoch train logger
                trainLogger = HtmlLogger(epochFolderPath, '{}_{}'.format(epochName, wEpoch))
                # train stage weights
                self.trainWeights(model.choosePathByAlphas, optimizer, wEpoch, dict(train=trainLogger))
                # switch stage
                switchStageFlag = model.switch_stage([lambda msg: trainLogger.addInfoToDataTable(msg)])
                # update epoch number
                wEpoch += 1

            # init epoch train logger for last epoch
            trainLogger = HtmlLogger(epochFolderPath, '{}_{}'.format(epochName, wEpoch))
            # set loggers dictionary
            loggersDict = dict(train=trainLogger)
            # last weights training epoch we want to log also to main logger
            trainData = self.trainWeights(model.choosePathByAlphas, optimizer, wEpoch, loggersDict)
            # validation on fixed partition by alphas values
            best_prec1 = self.__inferWithData(model.setFiltersByAlphas, epoch, loggersDict, trainData, best_prec1)
            # add data to main logger table
            logger.addDataRow(trainData)
            # validation on different partitions from alphas multinomial distribution
            for _ in range(self.nValidPartitions):
                dataRow = {}
                best_prec1 = self.__inferWithData(model.choosePathByAlphas, epoch, loggersDict, dataRow, best_prec1)
                # add data to main logger table
                logger.addDataRow(dataRow)

        # send final email
        self.sendEmail('Final', 0, 0)
