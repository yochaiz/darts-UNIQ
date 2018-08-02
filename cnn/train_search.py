import os
from sys import exit
from time import time, strftime
import glob
import numpy as np
import argparse

from torch.nn import CrossEntropyLoss
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.backends.cudnn as cudnn
from torch.cuda import is_available, set_device
from torch.cuda import manual_seed as cuda_manual_seed
from torch import manual_seed as torch_manual_seed
from torch import no_grad
from torch.optim import SGD
from torch.autograd.variable import Variable

from cnn.utils import create_exp_dir, count_parameters_in_MB, accuracy, AvgrageMeter, save
from cnn.utils import initLogger, printModelToFile, initTrainLogger, logDominantQuantizedOp, save_checkpoint
from cnn.utils import load_data, load_pre_trained
from cnn.resnet_model_search import ResNet
from cnn.architect import Architect
from cnn.uniq_loss import UniqLoss


def parseArgs(lossFuncs):
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--data', type=str, required=True, help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=1E-8, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=1, help='report frequency')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device id, e.g. 0,1,3')
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
    parser.add_argument('--arch_learning_rate', type=float, default=0.01, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

    parser.add_argument('--pre_trained', type=str,
                        default=None)
    # default='/home/yochaiz/darts/cnn/pre_trained_models/resnet_18_3_ops/model_opt.pth.tar')
    parser.add_argument('--nBitsMin', type=int, default=1, choices=range(1, 32 + 1), help='min number of bits')
    parser.add_argument('--nBitsMax', type=int, default=3, choices=range(1, 32 + 1), help='max number of bits')
    parser.add_argument('--bitwidth', type=str, default=None, help='list of bitwidth values, e.g. 1,4,16')
    parser.add_argument('--kernel', type=str, default='3', help='list of conv kernel sizes, e.g. 1,3,5')

    parser.add_argument('--loss', type=str, default='UniqLoss', choices=[key for key in lossFuncs.keys()])
    parser.add_argument('--lmbda', type=float, default=1.0, help='Lambda value for UniqLoss')
    parser.add_argument('--MaxBopsBits', type=int, default=3, choices=range(1, 32), help='maximum bits for uniform division')

    args = parser.parse_args()

    # manipulaye lambda value according to selected loss
    _, lossLambda = lossFuncs[args.loss]
    args.lmbda *= lossLambda

    # convert epochs to list
    args.epochs = [int(i) for i in args.epochs.split(',')]

    # convert bitwidth to list or range
    if args.bitwidth:
        args.bitwidth = [int(i) for i in args.bitwidth.split(',')]
    else:
        args.bitwidth = range(args.nBitsMin, args.nBitsMax + 1)

    # convert kernel sizes to list, sorted ascending
    args.kernel = [int(i) for i in args.kernel.split(',')]
    args.kernel.sort()

    # update GPUs list
    if type(args.gpu) is str:
        args.gpu = [int(i) for i in args.gpu.split(',')]

    args.device = 'cuda:' + str(args.gpu[0])

    args.save = 'results/search-{}-{}'.format(args.save, strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    return args


def train(train_queue, search_queue, args, model, architect, criterion, optimizer, lr, logger):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    grad = AvgrageMeter()

    model.train()
    nBatches = len(train_queue)

    for step, (input, target) in enumerate(train_queue):
        startTime = time()
        n = input.size(0)

        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda(async=True)

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(search_queue))
        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(target_search, requires_grad=False).cuda(async=True)

        arch_grad_norm = architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
        grad.update(arch_grad_norm)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        endTime = time()

        if step % args.report_freq == 0:
            logger.info('train [{}/{}] Loss:[{:.5f}] Accuracy:[{:.3f}] time:[{:.5f}]'
                        .format(step, nBatches, objs.avg, top1.avg, endTime - startTime))

    return top1.avg, objs.avg


def infer(valid_queue, args, model, criterion, logger):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    model.eval()
    nBatches = len(valid_queue)

    with no_grad():
        for step, (input, target) in enumerate(valid_queue):
            startTime = time()

            input = Variable(input, volatile=True).cuda()
            target = Variable(target, volatile=True).cuda(async=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            endTime = time()

            if step % args.report_freq == 0:
                logger.info('validation [{}/{}] Loss:[{:.5f}] Accuracy:[{:.3f}] time:[{:.5f}]'.
                            format(step, nBatches, objs.avg, top1.avg, endTime - startTime))

    return top1.avg, objs.avg


# loss functions can manipulate lambda value
lossFuncs = dict(UniqLoss=(UniqLoss, 1.0), CrossEntropy=(CrossEntropyLoss, 0.0))
args = parseArgs(lossFuncs)
print(args)
logger = initLogger(args.save, args.propagate)
CIFAR_CLASSES = 10

if not is_available():
    logger.info('no gpu device available')
    exit(1)

np.random.seed(args.seed)
set_device(args.gpu[0])
cudnn.benchmark = True
torch_manual_seed(args.seed)
cudnn.enabled = True
cuda_manual_seed(args.seed)

cross_entropy = CrossEntropyLoss().cuda()
criterion = UniqLoss(lmdba=args.lmbda, MaxBopsBits=args.MaxBopsBits, kernel_sizes=args.kernel)
criterion = criterion.cuda()
# criterion = criterion.to(args.device)
model = ResNet(criterion, args.bitwidth, args.kernel)
# model = DataParallel(model, args.gpu)
model = model.cuda()
# model = model.to(args.device)
# load pre-trained full-precision model
load_pre_trained(args.pre_trained, model, logger, args.gpu[0])

# print some attributes
printModelToFile(model, args.save)
logger.info('GPU:{}'.format(args.gpu))
logger.info("args = %s", args)
logger.info("param size = %fMB", count_parameters_in_MB(model))
logger.info('Learnable params:[{}]'.format(len(model.learnable_params)))
logger.info('alphas tensor size:[{}]'.format(model.arch_parameters()[0].size()))

optimizer = SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
# optimizer = Adam(model.parameters(), lr=args.learning_rate,
#                  betas=(0.5, 0.999), weight_decay=args.weight_decay)

# load data
train_queue, search_queue, valid_queue = load_data(args)

# extend epochs list as number of model layers
while len(args.epochs) < model.nLayers():
    args.epochs.append(args.epochs[-1])
# init epochs number where we have to switch stage in
epochsSwitchStage = [0]
for e in args.epochs:
    epochsSwitchStage.append(e + epochsSwitchStage[-1])
# total number of epochs is the last value in epochsSwitchStage
nEpochs = epochsSwitchStage[-1] + 1
# remove epoch 0 from list, we don't want to switch stage at the beginning
epochsSwitchStage = epochsSwitchStage[1:]

logger.info('nEpochs:[{}]'.format(nEpochs))
logger.info('epochsSwitchStage:{}'.format(epochsSwitchStage))

scheduler = CosineAnnealingLR(optimizer, float(nEpochs), eta_min=args.learning_rate_min)
architect = Architect(model, args)

best_prec1 = 0.0

for epoch in range(1, nEpochs + 1):
    trainLogger = initTrainLogger(str(epoch), args.save, args.propagate)

    scheduler.step()
    lr = scheduler.get_lr()[0]

    trainLogger.info('optimizer_lr:[{:.5f}], scheduler_lr:[{:.5f}]'.format(optimizer.defaults['lr'], lr))

    # print(F.softmax(model.alphas_normal, dim=-1))
    # print(F.softmax(model.alphas_reduce, dim=-1))

    # training
    train_acc, train_loss = train(train_queue, search_queue, args, model, architect, cross_entropy,
                                  optimizer, lr, trainLogger)

    # log accuracy, loss, etc.
    message = 'Epoch:[{}] , training accuracy:[{:.3f}] , training loss:[{:.3f}] , optimizer_lr:[{:.5f}], scheduler_lr:[{:.5f}]' \
        .format(epoch, train_acc, train_loss, optimizer.defaults['lr'], lr)
    logger.info(message)
    trainLogger.info(message)

    # log dominant QuantizedOp in each layer
    logDominantQuantizedOp(model, k=3, logger=trainLogger)

    # save model checkpoint
    save_checkpoint(args.save, model, epoch, best_prec1, is_best=False)

    # switch stage, i.e. freeze one more layer
    if (epoch in epochsSwitchStage) or (epoch == nEpochs):
        # validation
        valid_acc, valid_loss = infer(valid_queue, args, model, cross_entropy, trainLogger)
        message = 'Epoch:[{}] , validation accuracy:[{:.3f}] , validation loss:[{:.3f}]'.format(epoch, valid_acc,
                                                                                                valid_loss)
        logger.info(message)
        trainLogger.info(message)

        # save model checkpoint
        is_best = valid_acc > best_prec1
        best_prec1 = max(valid_acc, best_prec1)
        save_checkpoint(args.save, model, epoch, best_prec1, is_best)

        # switch stage
        model.switch_stage(trainLogger)
        # update optimizer & scheduler due to update in learnable params
        optimizer = SGD(model.parameters(), scheduler.get_lr()[0],
                        momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, float(nEpochs), eta_min=args.learning_rate_min)

save(model, os.path.join(args.save, 'weights.pt'))
logger.info('Done !')
