from torch import save as saveModel

from .regime import TrainRegime, save_checkpoint


class OptimalModel(TrainRegime):
    def __init__(self, args, logger):
        args.init_weights_train = True
        super(OptimalModel, self).__init__(args, logger)

        # log bops ratio
        bopsRatioStr = '{:.3f}'.format(self.model.calcBopsRatio())
        logger.addInfoTable(title='Bops ratio', rows=[[bopsRatioStr]])

    def train(self):
        # make sure model is quantized
        assert (self.model.isQuantized() is True)
        # update statistics in current model, i.e. last checkpoint
        self.model.calcStatistics(self.statistics_queue)
        # save model checkpoint
        checkpoint, (lastPath, _) = save_checkpoint(self.trainFolderPath, self.model, self.args, self.epoch, self.args.best_prec1, is_best=False)
        checkpoint['updated_statistics'] = True
        saveModel(checkpoint, lastPath)
        self.logger.addInfoToDataTable('Updated statistics in last checkpoint: [{}]'.format(checkpoint['updated_statistics']))

        # update statistics in optimal model
        optCheckpoint, optPath = self.optimalModelCheckpoint
        statisticsWereUpdated = self.model.updateCheckpointStatistics(optCheckpoint, optPath, self.statistics_queue)
        self.logger.addInfoToDataTable('Updated statistics in optimal checkpoint: [{}]'.format(statisticsWereUpdated))
