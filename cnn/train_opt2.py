from argparse import ArgumentParser, Namespace
from json import loads
from os import path, remove

from cnn.trainRegime.optimalModel import OptimalModel
from cnn.HtmlLogger import HtmlLogger
from cnn.utils import models, create_exp_dir, modelsRefs, sendEmail, logParameters

parser = ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='JSON file path')

scriptArgs = parser.parse_args()

# def G(scriptArgs):
with open(scriptArgs.data, 'r') as f:
    # read JSON data
    args = loads(f.read())
    args = Namespace(**args)
    # update values
    args.init_weights_train = True
    args.train_portion = 1.0
    args.batch_size = 250
    args.epochs = [5]
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
        logger = HtmlLogger(args.save, 'log')
        # select model constructor
        modelClass = models.__dict__.get(args.model)
        if modelClass:
            # sort bitwidths as list of tuples
            args.optModel_bitwidth = [[(v[0], v[1])] for v in args.optModel_bitwidth]
            args.baselineBits = args.baselineBits[0]
            args.baselineBits = [(args.baselineBits[0], args.baselineBits[1])]
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
                    # log parameters & load uniform model
                    uniform_best_prec1, uniformKey = logParameters(logger, args, model)
                    # log bops ratio
                    bopsRatioStr = '{:.3f}'.format(model.calcBopsRatio())
                    logger.addInfoTable(title='Bops ratio', rows=[[bopsRatioStr]])

                    # build regime for alphas optimization, it performs initial weights training
                    OptimalModel(args, model, modelClass, logger)
                    # load model best_prec1
                    best_prec1 = getattr(args, 'best_prec1', None)
                    # send mail if model beats uniform model
                    if (best_prec1 is not None) and (uniform_best_prec1 is not None):
                        if best_prec1 > uniform_best_prec1:
                            subject = '[{}] - found better allocation'.format(args.folderName)
                            content = 'The following allocation achieves better accuracy than uniform {}\n' \
                                .format(uniformKey)
                            content += 'Validation acc: current:[{}] > uniform:[{}]\n' \
                                .format(best_prec1, uniform_best_prec1)
                            content += 'Bops ratio:[{}]'.format(bopsRatioStr) + '\n\n'
                            # add model bitwidth allocation
                            for i, layerBitwidth in enumerate(args.bitwidth):
                                content += 'Layer [{}]: {}\n'.format(i, layerBitwidth)
                            # send email
                            sendEmail(args.recipients, subject, content)
                            # log to table
                            logger.addInfoToDataTable(content)

                    logger.addInfoToDataTable('Done !')

# remove the JSON file
if path.exists(scriptArgs.data):
    remove(scriptArgs.data)
