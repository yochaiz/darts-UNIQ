from torch.multiprocessing import set_start_method
from sys import exit, argv
from time import strftime
import numpy as np
import argparse
from traceback import format_exc
from os import getpid
from os.path import dirname, abspath
from inspect import getfile, currentframe, isclass

import torch.backends.cudnn as cudnn
from torch.cuda import is_available, set_device
from torch.cuda import manual_seed as cuda_manual_seed
from torch import manual_seed as torch_manual_seed

import cnn.trainRegime as trainRegimes
from cnn.HtmlLogger import HtmlLogger
from cnn.utils import create_exp_dir, saveArgsToJSON, loadGradEstimatorsNames, loadModelNames, models, sendEmail, loadDatasets


# collect possible alphas optimization
def loadAlphasRegimeNames():
    return [name for (name, obj) in trainRegimes.__dict__.items() if isclass(obj) and name.islower()]


def parseArgs(lossFuncsLambda):
    modelNames = loadModelNames()
    alphasRegimeNames = loadAlphasRegimeNames()
    gradEstimatorsNames = loadGradEstimatorsNames()
    datasets = loadDatasets()

    parser = argparse.ArgumentParser("F-BANNAS")
    parser.add_argument('--data', type=str, required=True, help='location of the data corpus')
    parser.add_argument('--dataset', metavar='DATASET', default='cifar100', choices=datasets.keys(), help='dataset name')
    parser.add_argument('--model', '-a', metavar='MODEL', default='tinynet', choices=modelNames,
                        help='model architecture: ' + ' | '.join(modelNames) + ' (default: alexnet)')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=1E-5, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=1, help='report frequency')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device id, e.g. 0,1,3')
    parser.add_argument('--nCopies', type=int, default=1, help='number of model copies per GPU')
    parser.add_argument('--epochs', type=str, default='5', help='num of training epochs per layer, as list, e.g. 5,4,3,8,6.'
                                                                'If len(epochs)<len(layers) then last value is used for rest of the layers')
    # parser.add_argument('--infer_epochs', type=int, default=10, help='number of epochs for training & inference after model is quantized')
    parser.add_argument('--optimal_epochs', type=int, default=30, help='stop training weights if there is no new optimum in last optimal_epochs')
    parser.add_argument('--workers', type=int, default=1, choices=range(1, 32), help='num of workers')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
    parser.add_argument('--propagate', action='store_true', default=False, help='print to stdout')
    parser.add_argument('--arch_learning_rate', type=float, default=0.01, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

    parser.add_argument('--pre_trained', type=str, default=None, help='pre-trained model to copy weights from')
    parser.add_argument('--init_weights_train', action='store_true', default=False,
                        help='initial train model weights (if required) before alphas optimization')

    parser.add_argument('--nBitsMin', type=int, default=1, choices=range(1, 32 + 1), help='min number of bits')
    parser.add_argument('--nBitsMax', type=int, default=3, choices=range(1, 32 + 1), help='max number of bits')
    parser.add_argument('--bitwidth', type=str, default=None, help='list of bitwidth values, e.g. 1,4,16')
    parser.add_argument('--kernel', type=str, default='3', help='list of conv kernel sizes, e.g. 1,3,5')

    parser.add_argument('--alphas', default=None, help='list of alphas tensors to update model layers alphas values')
    parser.add_argument('--alphas_regime', default='alphas_weights_loop', choices=alphasRegimeNames, help='alphas optimization method')
    parser.add_argument('--grad_estimator', default='layer_same_path', choices=gradEstimatorsNames, help='gradient estimation method')
    parser.add_argument('--nSamples', type=int, default=20, help='How many paths to sample in order to estimate gradient')
    parser.add_argument('--alphas_data_parts', type=int, default=4, help='split alphas training data to parts. each loop uses single part')
    parser.add_argument('--alpha_limit', type=float, default=0.8, help='if a layer opt alpha reached alpha_limit, then stop optimize layer alphas')
    parser.add_argument('--alpha_limit_counter', type=int, default=10,
                        help='how many consecutive steps the optimal alpha has to be over limit in order to stop layer alphas optimization')

    parser.add_argument('--partition', default=None, help='list of model layers partition')

    parser.add_argument('--loss', type=str, default='UniqLoss', choices=[key for key in lossFuncsLambda.keys()])
    parser.add_argument('--lmbda', type=float, default=1.0, help='Lambda value for UniqLoss')
    parser.add_argument('--baselineBits', type=str, default='3,3', help='bits budget')
    # select bops counter function
    bopsCounterKeys = list(models.BaseNet.countBopsFuncs.keys())
    parser.add_argument('--bopsCounter', type=str, default=bopsCounterKeys[0], choices=bopsCounterKeys)

    args = parser.parse_args()

    # manipulaye lambda value according to selected loss
    lossLambda = lossFuncsLambda[args.loss]
    args.lmbda *= lossLambda

    # convert epochs to list
    args.epochs = [int(i) for i in args.epochs.split(',')]

    # convert bitwidth to list
    if args.bitwidth:
        args.bitwidth = [(int(x[0]), int(x[-1])) for x in [y.split(',') for y in args.bitwidth.split('#')]]
    else:
        args.bitwidth = [(x, x) for x in list(range(args.nBitsMin, args.nBitsMax + 1))]

    # convert baselineBits to tuple
    args.baselineBits = args.baselineBits.split(',')
    args.baselineBits = [(int(args.baselineBits[0]), int(args.baselineBits[-1]))]

    # convert kernel sizes to list, sorted ascending
    args.kernel = [int(i) for i in args.kernel.split(',')]
    args.kernel.sort()

    # update GPUs list
    if type(args.gpu) is str:
        args.gpu = [int(i) for i in args.gpu.split(',')]

    args.device = 'cuda:' + str(args.gpu[0])

    # set number of model output classes
    args.nClasses = datasets[args.dataset]

    # set train folder name
    args.trainFolder = 'train'

    # init current file folder
    baseFolder = dirname(abspath(getfile(currentframe())))  # script directory
    # create folder
    # args.folderName = 'search-{}-{}'.format(args.save, strftime("%Y%m%d-%H%M%S"))
    args.time = strftime("%Y%m%d-%H%M%S")
    # args.folderName = '{}-{}-{}'.format(args.lmbda, args.dataset, args.time)
    args.folderName = '{},[{}],[{}],[{}]'.format(args.bitwidth, args.lmbda, args.dataset, args.time)
    args.save = '{}/results/{}'.format(baseFolder, args.folderName)
    args.codeZip = create_exp_dir(args.save)

    # init pre-trained checkpoints zip file
    preTrainedZipFilename = 'pre-trained.zip'
    preTrainedZipPath = '{}/../{}'.format(baseFolder, preTrainedZipFilename)
    args.preTrainedZip = (preTrainedZipPath, preTrainedZipFilename)

    # init emails recipients
    # args.recipients = ['evron.itay@gmail.com', 'chaimbaskin@cs.technion.ac.il', 'evgeniizh@campus.technion.ac.il', 'yochaiz.cs@gmail.com']
    args.recipients = ['yochaiz.cs@gmail.com']

    # copyBaselineKeys is a parameter whether to copy keys like epochs, learning rate from baseline to current model training regime
    # used mostly in OptimalModel subclass
    args.copyBaselineKeys = False

    # save args to JSON
    saveArgsToJSON(args)

    return args


if __name__ == '__main__':
    # loss functions manipulate lambda value
    lossFuncsLambda = dict(UniqLoss=1.0, CrossEntropy=0.0)
    args = parseArgs(lossFuncsLambda)
    logger = HtmlLogger(args.save, 'log')

    if not is_available():
        print('no gpu device available')
        exit(1)

    np.random.seed(args.seed)
    set_device(args.gpu[0])
    cudnn.benchmark = True
    torch_manual_seed(args.seed)
    cudnn.enabled = True
    cuda_manual_seed(args.seed)

    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        raise ValueError('spawn failed')

    try:
        # log command line
        logger.addInfoTable(title='Command line', rows=[[' '.join(argv)], ['PID:[{}]'.format(getpid())]])
        # build regime for alphas optimization
        alphasRegimeClass = trainRegimes.__dict__[args.alphas_regime]
        alphasRegime = alphasRegimeClass(args, logger)
        # train according to chosen regime
        alphasRegime.train()
        logger.addInfoToDataTable('Done !')

    except Exception as e:
        # create message content
        messageContent = '[{}] stopped due to error [{}] \n traceback:[{}]'. \
            format(args.folderName, str(e), format_exc())

        # create data table if exception happened before we create data table
        if logger.dataTableCols is None:
            logger.createDataTable('Exception', ['Error message'])
        # log to logger
        logger.addInfoToDataTable(messageContent, color='lightsalmon')
        # send e-mail with error details
        subject = '[{}] stopped'.format(args.folderName)
        sendEmail(['yochaiz.cs@gmail.com'], subject, messageContent)

        # forward exception
        raise e

## ======================================
# # set optimal model bitwidth per layer
# model.evalMode()
# args.optModel_bitwidth = [layer.getBitwidth() for layer in model.layersList]
# # save args to JSON
# saveArgsToJSON(args)
# # init args JSON destination path on server
# dstPath = '/home/vista/Desktop/Architecture_Search/DropDarts/cnn/optimal_models/{}/{}.json' \
#     .format(args.model, args.folderName)
# from shutil import copyfile
#
# copyfile(args.jsonPath, dstPath)
# from cnn.train_opt2 import G
#
# t = {'data': dstPath, 'epochs': [5], 'learning_rate': 0.1}
# from argparse import Namespace
#
# t = Namespace(**t)
# G(t)
## =====================================
