from argparse import ArgumentParser, Namespace
from json import loads
from os import path, remove

from cnn.trainRegime.regime import TrainRegime
from cnn.utils import initLogger, printModelToFile, models, create_exp_dir, modelsRefs, logUniformModel

parser = ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='JSON file path')
parser.add_argument('--epochs', type=str, default='5',
                    help='num of training epochs per layer, as list, e.g. 5,4,3,8,6.'
                         'If len(epochs)<len(layers) then last value is used for rest of the layers')

scriptArgs = parser.parse_args()
# convert epochs to list
scriptArgs.epochs = [int(i) for i in scriptArgs.epochs.split(',')]

# def G(scriptArgs):
with open(scriptArgs.data, 'r') as f:
    # read JSON data
    args = loads(f.read())
    args = Namespace(**args)
    # update values
    args.train_portion = 1.0
    args.batch_size = 250
    args.epochs = scriptArgs.epochs
    # extract args JSON folder path
    folderName = path.dirname(scriptArgs.data)
    # convert model bitwidths to string
    modelFolderName = ''
    for bitwidth in args.optModel_bitwidth:
        modelFolderName += '{},'.format(bitwidth)
    modelFolderName = modelFolderName[:-1]
    # set results folder path
    args.save = '{}/{}'.format(folderName, modelFolderName)
    if not path.exists(args.save):
        create_exp_dir(args.save)
        # init logger
        logger = initLogger(args.save, args.propagate)
        # select model constructor
        modelClass = models.__dict__.get(args.model)
        if modelClass:
            # sort bitwidths as list of tuples
            args.optModel_bitwidth = [[(v[0], v[1])] for v in args.optModel_bitwidth]
            args.MaxBopsBits = args.MaxBopsBits[0]
            args.MaxBopsBits = [(args.MaxBopsBits[0], args.MaxBopsBits[1])]
            # set bitwidth to optimal model bitwidth
            args.bitwidth = args.optModel_bitwidth
            # build optimal model
            model = modelClass(args)
            model = model.cuda()
            # select pre-trained key
            pre_trained_path = modelsRefs.get(args.model)
            if pre_trained_path:
                args.loadedOpsWithDiffWeights = model.loadPreTrained(pre_trained_path, logger, args.gpu[0])
                if args.loadedOpsWithDiffWeights is False:
                    # print some attributes
                    print(args)
                    printModelToFile(model, args.save)
                    logger.info('GPU:{}'.format(args.gpu))
                    logger.info("args = %s", args)
                    logger.info('Ops per layer:{}'.format([layer.numOfOps() for layer in model.layersList]))
                    logger.info('nPerms:[{}]'.format(model.nPerms))

                    # load uniform model
                    logUniformModel(args, logger)
                    # log bops ratio
                    logger.info('Bops ratio:[{:.5f}]'.format(model.calcBopsRatio()))

                    # build regime for alphas optimization, it performs initial weights training
                    TrainRegime(args, model, modelClass, logger)
                    logger.info('Done !')

# remove the JSON file
if path.exists(scriptArgs.data):
    remove(scriptArgs.data)
