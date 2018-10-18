from torch import save as saveModel

from .regime import TrainRegime, save_checkpoint


class OptimalModel(TrainRegime):
    def __init__(self, args, logger):
        args.init_weights_train = True
        super(OptimalModel, self).__init__(args, logger)

    def train(self):
        # make sure model is quantized
        for layer in self.model.layersList:
            assert (layer.quantized is True)
            assert (layer.added_noise is False)
        # update statistics in current model, i.e. last checkpoint
        self.model.calcStatistics()
        # save model checkpoint
        checkpoint, (lastPath, _) = save_checkpoint(self.trainFolderPath, self.model, self.args, self.epoch, self.args.best_prec1, is_best=False)
        checkpoint['updated_statistics'] = True
        saveModel(checkpoint, lastPath)
        self.logger.addInfoToDataTable('Updated statistics in last checkpoint: [{}]'.format(checkpoint['updated_statistics']))

        # update statistics in optimal model
        optCheckpoint, optPath = self.optimalModelCheckpoint
        statisticsWereUpdated = self.model.updateCheckpointStatistics(optCheckpoint, optPath)
        self.logger.addInfoToDataTable('Updated statistics in optimal checkpoint: [{}]'.format(statisticsWereUpdated))
