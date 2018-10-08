from torch.nn.parallel.data_parallel import DataParallel

from .regime import TrainRegime


class OptimalModel(TrainRegime):
    def __init__(self, args, model, modelClass, logger):
        args.init_weights_train = True
        model = DataParallel(model, args.gpu)
        super(OptimalModel, self).__init__(args, model, modelClass, logger)

    def train(self):
        pass
