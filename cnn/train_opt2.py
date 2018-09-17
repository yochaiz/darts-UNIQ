from argparse import ArgumentParser, Namespace
from json import loads
from os import path

import cnn.trainRegime as trainRegimes
from cnn.utils import load_pre_trained, initLogger, printModelToFile, models, create_exp_dir

# references to models pre-trained
modelsRefs = {
    'thin_resnet': '/home/yochaiz/DropDarts/cnn/pre_trained/thin_resnet/train/model_opt.pth.tar',
    'thin_resnet_[2]': '/home/yochaiz/DropDarts/cnn/uniform/thin_resnet_[2]/train/model_opt.pth.tar',
    'resnet': '/home/yochaiz/DropDarts/cnn/pre_trained/resnet_cifar10_trained_32_bit_deeper/model_best.pth.tar',
    'resnet_[2]': '/home/yochaiz/DropDarts/cnn/uniform/resnet_[2]/train/model_opt.pth.tar'
}

parser = ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='JSON file path')
parser.add_argument('--epochs', type=str, default='10',
                    help='num of training epochs per layer, as list, e.g. 5,4,3,8,6.'
                         'If len(epochs)<len(layers) then last value is used for rest of the layers')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')

scriptArgs = parser.parse_args()
# convert epochs to list
scriptArgs.epochs = [int(i) for i in scriptArgs.epochs.split(',')]

with open(scriptArgs.data, 'r') as f:
    # read JSON data
    args = loads(f.read())
    args = Namespace(**args)
    # update values
    args.train_portion = 1.0
    args.epochs = scriptArgs.epochs
    args.learning_rate = scriptArgs.learning_rate
    # extract args JSON folder path
    folderName = path.dirname(scriptArgs.data)
    # set results folder path
    args.save = '{}/{}'.format(folderName, args.folderName)
    create_exp_dir(args.save)
    # init logger
    logger = initLogger(args.save, args.propagate)
    # select model constructor
    modelClass = models.__dict__.get(args.model)
    if modelClass:
        # set bitwidth to optimal model bitwidth
        args.bitwidth = args.optModel_bitwidth
        # build optimal model
        model = modelClass(args)
        model = model.cuda()
        # select pre-trained key
        pre_trained_path = modelsRefs.get(args.model)
        if pre_trained_path:
            args.loadedOpsWithDiffWeights = load_pre_trained(pre_trained_path, model, logger, args.gpu[0])
            if args.loadedOpsWithDiffWeights is False:
                # print some attributes
                print(args)
                printModelToFile(model, args.save)
                logger.info('GPU:{}'.format(args.gpu))
                logger.info("args = %s", args)
                logger.info('Ops per layer:{}'.format([len(layer.ops) for layer in model.layersList]))
                logger.info('nPerms:[{}]'.format(model.nPerms))

                # build regime for alphas optimization
                alphasRegimeClass = trainRegimes.__dict__.get(args.alphas_regime)
                if alphasRegimeClass:
                    # create train regime instance, it performs initial weights training
                    alphasRegimeClass(args, model, modelClass, logger)

                    logger.info('Done !')
