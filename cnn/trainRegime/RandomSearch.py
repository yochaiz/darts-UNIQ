from .regime import TrainRegime
from time import sleep


class RandomSearch(TrainRegime):
    def __init__(self, args, model, modelClass, logger):
        super(RandomSearch, self).__init__(args, model, modelClass, logger)

        # init number of random allocations we want to train
        self.nAllocations = 1
        # set bops ratio range limits
        self.minBopsRatio = 0.9
        self.maxBopsRatio = 1.1
        # set sleep time (minutes)
        self.sleepTime = 30

    def train(self):
        model = self.model
        # switch to eval mode
        model.evalMode()
        for i in range(self.nAllocations):
            isCaseValid = False
            # find allocation with bops ratio in required range
            while isCaseValid is False:
                # choose random bitwidth in each layer
                model.chooseRandomPath()
                # set 1st layer to maximal bitwidth (based on bops)
                layer = model.layersList[0]
                layer.curr_alpha_idx = layer.numOfOps() - 1
                layer2 = model.layersList[1]
                layer2.prev_alpha_idx = layer.curr_alpha_idx
                # calc bops ratio
                bopsRatio = model.calcBopsRatio()
                # set current allocation as optimal model
                bitwidthKey = self.setOptModelBitwidth()
                # check if case is valid
                isCaseValid = (bitwidthKey not in self.optModelBitwidthCounter) and \
                              ((bopsRatio >= self.minBopsRatio) and (bopsRatio <= self.maxBopsRatio))
                # (bopsRatio <= self.maxBopsRatio)

            # set counter, such that sendOptModel() will train it
            self.optModelBitwidthCounter[bitwidthKey] = self.nBatchesOptModel - 1
            # train selected allocation
            self.sendOptModel(bitwidthKey, i, 0)
            # log allocations
            self.logger.info('Bitwidth:{} , bopsRatio:[{:.5f}], validation accuracy:[]'.format(bitwidthKey, bopsRatio))

        # wait until all allocations sent successfully
        while len(self.optModelTrainingQueue) > 0:
            print('Queue length:[{}], going to sleep for [{}] mins.'
                  .format(len(self.optModelTrainingQueue), self.sleepTime))
            # wait 30 minutes
            sleep(self.sleepTime * 60)
            self.trySendQueuedJobs()
