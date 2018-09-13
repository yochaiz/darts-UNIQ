from torch.multiprocessing import set_start_method
from sys import exit
from time import strftime
from json import dump
import numpy as np
import argparse
from inspect import isclass

import torch.backends.cudnn as cudnn
from torch.cuda import is_available, set_device
from torch.cuda import manual_seed as cuda_manual_seed
from torch import manual_seed as torch_manual_seed

import cnn.models as models
from cnn.utils import create_exp_dir, count_parameters_in_MB, load_pre_trained
from cnn.utils import initLogger, printModelToFile
from cnn.optimize import optimize
import cnn.trainRegime as trainRegime

# collect possible models names
def loadModelNames():
    return [name for (name, obj) in models.__dict__.items() if isclass(obj) and name.islower()]

#collect possible alphas optimization
def loadAlphasRegimeNames():
    return [name for (name, obj) in trainRegime.__dict__.items() if isclass(obj) and (name != 'TrainRegime')]


def parseArgs(lossFuncsLambda):
    modelNames = loadModelNames()
    alphasRegimeNames = loadAlphasRegimeNames()

    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--data', type=str, required=True, help='location of the data corpus')
    parser.add_argument('--dataset', metavar='DATASET', default='cifar10', help='dataset name')
    parser.add_argument('--model', '-a', metavar='MODEL', default='tinynet', choices=modelNames,
                        help='model architecture: ' + ' | '.join(modelNames) + ' (default: alexnet)')
    parser.add_argument('--alphas_reg',  metavar='ALPHAS_REG', default='WeightsOnceAlphasOnce', choices=alphasRegimeNames,
                        help='alphas optimization method: ' + ' | '.join(alphasRegimeNames) + ' (default: weights once alphas once)')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--nSamplesPerAlpha', type=int, default=50, help='How many alphas to sample to estimate the expectation value')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=1E-5, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=1, help='report frequency')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device id, e.g. 0,1,3')
    parser.add_argument('--nCopies', type=int, default=1, help='number of model copies per GPU')
    parser.add_argument('--epochs', type=str, default='5',
                        help='num of training epochs per layer, as list, e.g. 5,4,3,8,6.'
                             'If len(epochs)<len(layers) then last value is used for rest of the layers')
    parser.add_argument('--workers', type=int, default=1, choices=range(1, 32), help='num of workers')
    # parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
    # parser.add_argument('--layers', type=int, default=8, help='total number of layers')
    # parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
    parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
    parser.add_argument('--propagate', action='store_true', default=False, help='print to stdout')
    parser.add_argument('--arch_learning_rate', type=float, default=0.2, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

    parser.add_argument('--pre_trained', type=str,
                        default=None)
    # default='/home/yochaiz/darts/cnn/pre_trained_models/resnet_18_3_ops/model_opt.pth.tar')
    parser.add_argument('--nBitsMin', type=int, default=1, choices=range(1, 32 + 1), help='min number of bits')
    parser.add_argument('--nBitsMax', type=int, default=3, choices=range(1, 32 + 1), help='max number of bits')
    parser.add_argument('--bitwidth', type=str, default=None, help='list of bitwidth values, e.g. 1,4,16')
    parser.add_argument('--kernel', type=str, default='3', help='list of conv kernel sizes, e.g. 1,3,5')

    parser.add_argument('--loss', type=str, default='UniqLoss', choices=[key for key in lossFuncsLambda.keys()])
    parser.add_argument('--lmbda', type=float, default=1.0, help='Lambda value for UniqLoss')
    parser.add_argument('--MaxBopsBits', type=int, default=3, choices=range(1, 32),
                        help='maximum bits for uniform division')
    # select bops counter function
    bopsCounterKeys = list(models.BaseNet.countBopsFuncs.keys())
    parser.add_argument('--bopsCounter', type=str, default=bopsCounterKeys[0], choices=bopsCounterKeys)

    args = parser.parse_args()

    # manipulaye lambda value according to selected loss
    lossLambda = lossFuncsLambda[args.loss]
    args.lmbda *= lossLambda

    # convert epochs to list
    args.epochs = [int(i) for i in args.epochs.split(',')]

    # convert bitwidth to list or range
    if args.bitwidth:
        args.bitwidth = [int(i) for i in args.bitwidth.split(',')]
    else:
        args.bitwidth = list(range(args.nBitsMin, args.nBitsMax + 1))

    # convert kernel sizes to list, sorted ascending
    args.kernel = [int(i) for i in args.kernel.split(',')]
    args.kernel.sort()

    # update GPUs list
    if type(args.gpu) is str:
        args.gpu = [int(i) for i in args.gpu.split(',')]

    args.device = 'cuda:' + str(args.gpu[0])

    # set train folder name
    args.trainFolder = 'train'

    args.save = 'results/search-{}-{}'.format(args.save, strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(args.save)

    # save args to JSON
    with open('{}/args.json'.format(args.save), 'w') as f:
        dump(vars(args), f)

    return args


if __name__ == '__main__':
    # loss functions manipulate lambda value
    lossFuncsLambda = dict(UniqLoss=1.0, CrossEntropy=0.0)
    args = parseArgs(lossFuncsLambda)
    print(args)
    logger = initLogger(args.save, args.propagate)

    if not is_available():
        logger.info('no gpu device available')
        exit(1)

    np.random.seed(args.seed)
    set_device(args.gpu[0])
    cudnn.benchmark = True
    torch_manual_seed(args.seed)
    cudnn.enabled = True
    cuda_manual_seed(args.seed)

    # build model for uniform distribution of bits
    modelClass = models.__dict__[args.model]
    uniform_args = argparse.Namespace(**vars(args))
    uniform_args.bitwidth = [args.MaxBopsBits]
    uniform_model = modelClass(uniform_args)
    # init maxBops
    args.maxBops = uniform_model.countBops()

    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        raise ValueError('spawn failed')

    # init model
    model = modelClass(args)
    model = model.cuda()
    # load pre-trained full-precision model
    args.loadedOpsWithDiffWeights = load_pre_trained(args.pre_trained, model, logger, args.gpu[0])

    # print some attributes
    printModelToFile(model, args.save)
    logger.info('GPU:{}'.format(args.gpu))
    logger.info("args = %s", args)
    logger.info("param size = %fMB", count_parameters_in_MB(model))
    logger.info('Learnable params:[{}]'.format(len(model.learnable_params)))
    logger.info('Ops per layer:{}'.format([len(layer.ops) for layer in model.layersList]))
    logger.info('nPerms:[{}]'.format(model.nPerms))

    #optimize(args, model, modelClass, logger)

    # build regime for alphas optimization
    alphasRegClass = trainRegime.__dict__[args.alphas_reg]
    alphasReg = alphasRegClass(args, model, modelClass, logger)
    #train according to chosen regime
    alphasReg.train()


    logger.info('Done !')
