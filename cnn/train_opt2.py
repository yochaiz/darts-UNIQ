from argparse import ArgumentParser, Namespace
from json import loads
from os import path, remove

from torch import load as loadModel

import cnn.trainRegime as trainRegimes
from cnn.utils import load_pre_trained, initLogger, printModelToFile, models, create_exp_dir

# references to models pre-trained
modelsRefs = {
    'thin_resnet': '/home/yochaiz/DropDarts/cnn/pre_trained/thin_resnet/train/model_opt.pth.tar',
    'thin_resnet_w:[2]_a:[32]': '/home/yochaiz/DropDarts/cnn/uniform/thin_resnet_w:[2]_a:[32]/train/model_opt.pth.tar',
    'resnet': '/home/yochaiz/DropDarts/cnn/pre_trained/resnet_cifar10_trained_32_bit_deeper/model_best.pth.tar',
    'resnet_w:[1]_a:[1]': '/home/yochaiz/DropDarts/cnn/uniform/resnet_w:[1]_a:[1]/train/model_opt.pth.tar',
    'resnet_w:[1]_a:[32]': '/home/yochaiz/DropDarts/cnn/uniform/resnet_w:[1]_a:[32]/train/model_opt.pth.tar',
    'resnet_w:[2]_a:[32]': '/home/yochaiz/DropDarts/cnn/uniform/resnet_w:[2]_a:[32]/train/model_opt.pth.tar',
    'resnet_w:[2]_a:[3]': '/home/yochaiz/DropDarts/cnn/uniform/resnet_w:[2]_a:[3]/train/model_opt.pth.tar',
    'resnet_w:[2]_a:[4]': '/home/yochaiz/DropDarts/cnn/uniform/resnet_w:[2]_a:[4]/train/model_opt.pth.tar',
    'resnet_w:[3]_a:[3]': '/home/yochaiz/DropDarts/cnn/uniform/resnet_w:[3]_a:[3]/train/model_opt.pth.tar',
    'resnet_w:[3]_a:[32]': '/home/yochaiz/DropDarts/cnn/uniform/resnet_w:[3]_a:[32]/train/model_opt.pth.tar',
    'resnet_w:[4]_a:[4]': '/home/yochaiz/DropDarts/cnn/uniform/resnet_w:[4]_a:[4]/train/model_opt.pth.tar',
    'resnet_w:[4]_a:[32]': '/home/yochaiz/DropDarts/cnn/uniform/resnet_w:[4]_a:[32]/train/model_opt.pth.tar',
    'resnet_w:[5]_a:[5]': '/home/yochaiz/DropDarts/cnn/uniform/resnet_w:[5]_a:[5]/train/model_opt.pth.tar',
    'resnet_w:[8]_a:[32]': '/home/yochaiz/DropDarts/cnn/uniform/resnet_w:[8]_a:[32]/train/model_checkpoint.pth.tar',
}

parser = ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='JSON file path')
parser.add_argument('--epochs', type=str, default='5',
                    help='num of training epochs per layer, as list, e.g. 5,4,3,8,6.'
                         'If len(epochs)<len(layers) then last value is used for rest of the layers')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')

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
    args.learning_rate = scriptArgs.learning_rate
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
            args.MaxBopsBits = (args.MaxBopsBits[0], args.MaxBopsBits[1])
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

                    # load uniform model
                    uniformKey = '{}_w:[{}]_a:[{}]'.format(args.model, args.MaxBopsBits[0], args.MaxBopsBits[-1])
                    uniformPath = modelsRefs.get(uniformKey)
                    best_prec1 = 'Not found'
                    if uniformPath and path.exists(uniformPath):
                        uniform_checkpoint = loadModel(uniformPath,
                                                       map_location=lambda storage, loc: storage.cuda(args.gpu[0]))
                        best_prec1 = uniform_checkpoint.get('best_prec1', best_prec1)
                    # print result
                    logger.info('Uniform {} validation accuracy:[{:.5f}]'.format(uniformKey, best_prec1))
                    # log bops ratio
                    logger.info('Bops ratio:[{:.5f}]'.format(model.calcBopsRatio()))

                    # build regime for alphas optimization
                    alphasRegimeClass = trainRegimes.__dict__.get(args.alphas_regime)
                    if alphasRegimeClass:
                        # create train regime instance, it performs initial weights training
                        alphasRegimeClass(args, model, modelClass, logger)

                        logger.info('Done !')

# remove the JSON file
if path.exists(scriptArgs.data):
    remove(scriptArgs.data)
