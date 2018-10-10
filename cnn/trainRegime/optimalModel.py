from .regime import TrainRegime


class OptimalModel(TrainRegime):
    def __init__(self, args, model, modelClass, logger):
        args.init_weights_train = True
        super(OptimalModel, self).__init__(args, model, modelClass, logger)

    def train(self):
        pass
