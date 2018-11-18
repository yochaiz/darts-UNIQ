from sys import argv
from os import path, getpid, rename
from numpy import random
from inspect import getfile, currentframe
from argparse import ArgumentParser
from datetime import datetime

from cnn.trainRegime.optimalModel import OptimalModel
from cnn.HtmlLogger import HtmlLogger
from cnn.utils import create_exp_dir

import torch.backends.cudnn as cudnn
from torch.cuda import is_available, set_device
from torch.cuda import manual_seed as cuda_manual_seed
from torch import manual_seed as torch_manual_seed
from torch import load as loadCheckpoint
from torch import save as saveCheckpoint

if not is_available():
    print('no gpu device available')
    exit(1)


def G(scriptArgs):
    # load args from file
    args = loadCheckpoint(scriptArgs.json, map_location=lambda storage, loc: storage.cuda())

    # # ========================== DEBUG ===============================
    # print(args)
    # # extract args JSON folder path
    # folderName = path.dirname(scriptArgs.json)
    # # results folder is JSON filename
    # jsonFileName = path.basename(scriptArgs.json)
    # # set results folder path
    # args.save = '{}/{}'.format(folderName, jsonFileName[:-5])
    # print('====')
    # print(args.save)
    # create_exp_dir(args.save)
    # # init logger
    # logger = HtmlLogger(args.save, 'log')
    # # init project base folder
    # baseFolder = path.dirname(path.abspath(getfile(currentframe())))  # script directory
    # # set pre-trained path
    # preTrainedKey = '[(32, 32)],[{}]'.format(args.model)
    # preTrainedFileName = 'model.updated_stats.pth.tar'
    # args.pre_trained = '{}/../pre_trained/{}/train_portion_1.0/{}/train/{}'.format(baseFolder, args.dataset, preTrainedKey, preTrainedFileName)
    # print(args.pre_trained)
    #
    # # build model for uniform distribution of bits
    # from cnn.utils import models
    # modelClass = models.__dict__[args.model]
    # # init model
    # model = modelClass(args)
    # model = model.cuda()
    # # load pre-trained full-precision model
    # args.loadedOpsWithDiffWeights = model.loadPreTrained(args.pre_trained, logger, args.gpu[0])
    #
    # print(args)
    # setattr(args, 'Validation acc', 33.4)
    # setattr(args, 'Validation loss', 1.347)
    #
    # saveCheckpoint(args, scriptArgs.json)
    # # move checkpoint to finished jobs folder
    # newDestination = scriptArgs.json
    # if scriptArgs.dstFolder is not None:
    #     newDestination = '{}/{}'.format(scriptArgs.dstFolder, jsonFileName)
    #     rename(scriptArgs.json, newDestination)
    #
    # print(args)
    # print('DDone !')
    # from time import sleep
    # sleep(60)
    # exit(0)
    # # ================================================================

    args.seed = datetime.now().microsecond
    # update cudnn parameters
    random.seed(args.seed)
    set_device(scriptArgs.gpu[0])
    cudnn.benchmark = True
    torch_manual_seed(args.seed)
    cudnn.enabled = True
    cuda_manual_seed(args.seed)
    # update values
    args.train_portion = 1.0
    args.batch_size = 250
    args.gpu = scriptArgs.gpu
    args.data = scriptArgs.data
    # extract args JSON folder path
    folderName = path.dirname(scriptArgs.json)
    # results folder is JSON filename
    jsonFileName = path.basename(scriptArgs.json)
    # set results folder path
    args.save = '{}/{}'.format(folderName, jsonFileName[:-5])
    if not path.exists(args.save):
        create_exp_dir(args.save)
        # init logger
        logger = HtmlLogger(args.save, 'log')
        # log command line
        logger.addInfoTable(title='Command line', rows=[[' '.join(argv)], ['PID:[{}]'.format(getpid())]])
        # init project base folder
        baseFolder = path.dirname(path.abspath(getfile(currentframe())))  # script directory
        # set pre-trained path
        preTrainedKey = '[(32, 32)],[{}]'.format(args.model)
        preTrainedFileName = 'model.updated_stats.pth.tar'
        args.pre_trained = '{}/../pre_trained/{}/train_portion_1.0/{}/train/{}'.format(baseFolder, args.dataset, preTrainedKey, preTrainedFileName)
        # check path exists
        if path.exists(args.pre_trained):
            alphasRegime = OptimalModel(args, logger)
            # train according to chosen regime
            alphasRegime.train()
            # best_prec1, best_valid_loss are now part of args, therefore we have to save args again, the sender will be able to read these values
            saveCheckpoint(args, scriptArgs.json)
            # move checkpoint to finished jobs folder
            newDestination = scriptArgs.json
            if scriptArgs.dstFolder is not None:
                newDestination = '{}/{}'.format(scriptArgs.dstFolder, jsonFileName)
                rename(scriptArgs.json, newDestination)

            print(args)
            logger.addInfoToDataTable('Saved args to [{}]'.format(newDestination))
            logger.addInfoToDataTable('Done !')


# ========================================================================================================================

parser = ArgumentParser()
parser.add_argument('--json', type=str, required=True, help='JSON file path')
parser.add_argument('--dstFolder', type=str, default=None, help='JSON file destination when training is over')
parser.add_argument('--data', type=str, default='datasets/', help='datasets folder path')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id, e.g. 0,1,3')

scriptArgs = parser.parse_args()
# update GPUs list
if type(scriptArgs.gpu) is str:
    scriptArgs.gpu = [int(i) for i in scriptArgs.gpu.split(',')]

G(scriptArgs)

#
# # select model constructor
# modelClass = models.__dict__.get(args.model)
# if modelClass:
#     # sort bitwidths as list of tuples
#     args.optModel_bitwidth = [[(v[0], v[1])] for v in args.optModel_bitwidth]
#     args.baselineBits = args.baselineBits[0]
#     args.baselineBits = [(args.baselineBits[0], args.baselineBits[1])]
#     # set bitwidth to optimal model bitwidth
#     args.bitwidth = args.optModel_bitwidth
#     # build optimal model
#     model = modelClass(args)
#     model = model.cuda()
#     # select pre-trained key
#     pre_trained_path = modelsRefs.get(args.model)
#     if pre_trained_path:
#         args.loadedOpsWithDiffWeights = model.loadPreTrained(pre_trained_path, logger, args.gpu[0])
#         if args.loadedOpsWithDiffWeights is False:
#             # log parameters & load uniform model
#             uniform_best_prec1, uniformKey = logParameters(logger, args, model)
#             # log bops ratio
#             bopsRatioStr = '{:.3f}'.format(model.calcBopsRatio())
#             logger.addInfoTable(title='Bops ratio', rows=[[bopsRatioStr]])
#
#             # build regime for alphas optimization, it performs initial weights training
#             OptimalModel(args, model, modelClass, logger)
#             # load model best_prec1
#             best_prec1 = getattr(args, 'best_prec1', None)
#             # send mail if model beats uniform model
#             if (best_prec1 is not None) and (uniform_best_prec1 is not None):
#                 if best_prec1 > uniform_best_prec1:
#                     subject = '[{}] - found better allocation'.format(args.folderName)
#                     content = 'The following allocation achieves better accuracy than uniform {}\n' \
#                         .format(uniformKey)
#                     content += 'Validation acc: current:[{}] > uniform:[{}]\n' \
#                         .format(best_prec1, uniform_best_prec1)
#                     content += 'Bops ratio:[{}]'.format(bopsRatioStr) + '\n\n'
#                     # add model bitwidth allocation
#                     for i, layerBitwidth in enumerate(args.bitwidth):
#                         content += 'Layer [{}]: {}\n'.format(i, layerBitwidth)
#                     # send email
#                     sendEmail(args.recipients, subject, content)
#                     # log to table
#                     logger.addInfoToDataTable(content)
