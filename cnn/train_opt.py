import argparse
from json import load
from numpy import random

from torch.cuda import is_available, set_device
from torch.cuda import manual_seed as cuda_manual_seed
from torch import manual_seed as torch_manual_seed
from torch import load as loadModel
import torch.backends.cudnn as cudnn

from cnn.utils import initLogger, load_pre_trained, stateOptModelPattern, stateFilenameDefault
from cnn.utils import printModelToFile
from cnn.uniq_loss import UniqLoss
from cnn.resnet_model_search import ResNet
from cnn.optimize import optimize


def parseArgs():
    parser = argparse.ArgumentParser("opt")
    parser.add_argument('--folder', type=str, required=True, help='folder when opt model is located')
    parser.add_argument('--pre_trained', type=str)
    parser.add_argument('--epochs', type=str,
                        help='num of training epochs per layer, as list, e.g. 5,4,3,8,6.'
                             'If len(epochs)<len(layers) then last value is used for rest of the layers')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size')

    args = parser.parse_args()
    # update folder location
    args.save = args.folder

    return args


# load folder name
args = parseArgs()
folderName = args.folder
# args as dict
argsDict = vars(args)
# save a copy of original values
argsOrg = dict(argsDict)

# load folder args
with open('{}/args.json'.format(folderName), 'r') as f:
    argsJSON = load(f)

# update values from args JSON
for k, v in argsJSON.items():
    argsDict[k] = v

# restore current args values
for k, v in argsOrg.items():
    if v:
        argsDict[k] = v

# load logger
logger = initLogger(args.save, args.propagate)
logger.info('Training opt model')

if not is_available():
    logger.info('no gpu device available')
    exit(1)

random.seed(args.seed)
set_device(args.gpu[0])
cudnn.benchmark = True
torch_manual_seed(args.seed)
cudnn.enabled = True
cuda_manual_seed(args.seed)

# init losses
crit = UniqLoss(lmdba=args.lmbda, MaxBopsBits=args.MaxBopsBits, kernel_sizes=args.kernel,
                bopsFuncKey=args.bopsCounter, folderName=args.save)
crit = crit.cuda()
model = ResNet(crit, args.bitwidth, args.kernel, args.bopsCounter)
model = model.cuda()
# load pre-trained full-precision model
load_pre_trained(args.pre_trained, model, logger, args.gpu[0])

# load optimal model checkpoint
optModelChkpnt = stateOptModelPattern.format(args.save, stateFilenameDefault)
optModelChkpnt = loadModel(optModelChkpnt, map_location=lambda storage, loc: storage.cuda(args.gpu[0]))
# load alphas from checkpoint
model.load_alphas_state(optModelChkpnt['alphas'])
# convert opt model to discrete
model.toDiscrete()

# print model to file
printModelToFile(model, args.save, fname='opt_model')
logger.info("args = %s", args)
logger.info('Learnable params:[{}]'.format(len(model.learnable_params)))
logger.info('alphas tensor size:[{}]'.format(model.arch_parameters()[0].size()))

# set train_portion to 1.0, we do not optimize architecture now, only model weights
args.train_portion = 1.0
# set train folder
args.trainFolder = 'train_opt'
# do the magic
optimize(args, model, logger)
