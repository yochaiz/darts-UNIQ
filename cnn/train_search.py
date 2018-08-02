# import torch.nn.functional as F
import os
from sys import stdout, exit
from time import time, strftime
import glob
import numpy as np
import logging
import argparse

from torch.nn import CrossEntropyLoss
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.datasets.cifar import CIFAR10
import torch.backends.cudnn as cudnn
from torch.cuda import is_available, set_device
from torch.cuda import manual_seed as cuda_manual_seed
from torch import manual_seed as torch_manual_seed
from torch import no_grad
from torch.optim import SGD, Adam
from torch.autograd.variable import Variable

from cnn.utils import create_exp_dir, count_parameters_in_MB, _data_transforms_cifar10, accuracy, AvgrageMeter, save
from cnn.model_search import Network
from cnn.resnet_model_search import ResNet
from cnn.architect import Architect
from cnn.operations import OPS
from cnn.build_discrete_model import DiscreteResNet
from cnn.uniq_loss import UniqLoss

def parseArgs():
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--data', type=str, default='../data/', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=3, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=1, help='report frequency')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device id, e.g. 0,1,3')
    parser.add_argument('--epochs', type=str, default='1',
                        help='num of training epochs per layer, as list, e.g. 5,4,3,8,6.'
                             'If len(epochs)<len(layers) then last value is used for rest of the layers')
    parser.add_argument('--workers', type=int, default=16, choices=range(1, 32), help='num of workers')
    parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
    parser.add_argument('--layers', type=int, default=8, help='total number of layers')
    parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
    parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

    parser.add_argument('--nBitsMin', type=int, default=1, choices=range(1, 32), help='min number of bits')
    parser.add_argument('--nBitsMax', type=int, default=3, choices=range(1, 32), help='max number of bits')
    parser.add_argument('--loss', type = str , default ='UniqLoss', choices = ['CrossEntropy', 'UniqLoss'])
    parser.add_argument('--MaxBopsBits', type=int, default=3, choices=range(1, 32), help='maximum bits for uniform division')
    args = parser.parse_args()
    # update epochs per layer list
    args.epochs = [int(i) for i in args.epochs.split(',')]
    while len(args.epochs) < args.layers:
        args.epochs.append(args.epochs[-1])

    # update GPUs list
    if type(args.gpu) is str:
        args.gpu = [int(i) for i in args.gpu.split(',')]

    args.device = 'cuda:' + str(args.gpu[0])

    args.save = 'search-{}-{}'.format(args.save, strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    return args


def train(train_queue, search_queue, args, model, architect, criterion, optimizer, lr):
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
        loss = cross_entropy(logits, target)

        loss.backward()
        clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        endTime = time()

        if step % args.report_freq == 0:
            logger.info('train [{}/{}] Loss:[{:.5f}] Accuracy:[{:.3f}] time:[{:.5f}]'
                        .format(step, nBatches, objs.avg, top1.avg, endTime - startTime))

    return top1.avg, objs.avg, grad.avg


def infer(valid_queue, args, model, criterion):
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
            loss = cross_entropy(logits, target)

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


args = parseArgs()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))

logger = logging.getLogger('darts')
logger.addHandler(fh)
# disable logging to stdout
logger.propagate = False

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

#criterion = CrossEntropyLoss()
#criterion = criterion.cuda()
cross_entropy = CrossEntropyLoss().cuda()
criterion = UniqLoss(lmdba=1, MaxBopsBits = args.MaxBopsBits, batch_size = args.batch_size) if args.loss == 'UniqLoss' else cross_entropy
# criterion = criterion.to(args.device)
# model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
model = ResNet(criterion, args.nBitsMin, args.nBitsMax,args.batch_size)
# model = DataParallel(model, args.gpu)
model = model.cuda()
# model = model.to(args.device)

# print some attributes
logger.info('{}'.format(model))
logger.info('GPU:{}'.format(args.gpu))
logger.info("args = %s", args)
logger.info("param size = %fMB", count_parameters_in_MB(model))
logger.info('Learnable params:[{}]'.format(len(model.learnable_params)))
logger.info('Number of operations:[{}]'.format(len(OPS)))
logger.info('OPS:{}'.format(OPS.keys()))
logger.info('alphas tensor size:[{}]'.format(model.arch_parameters()[0].size()))

optimizer = SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
# optimizer = Adam(model.parameters(), lr=args.learning_rate,
#                  betas=(0.5, 0.999), weight_decay=args.weight_decay)

train_transform, valid_transform = _data_transforms_cifar10(args)
train_data = CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
valid_data = CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

#### narrow data for debug purposes
train_data.train_data = train_data.train_data[0:2000]
train_data.train_labels = train_data.train_labels[0:2000]
valid_data.test_data = valid_data.test_data[0:1000]
valid_data.test_labels = valid_data.test_labels[0:1000]
####

num_train = len(train_data)
indices = list(range(num_train))
split = int(np.floor(args.train_portion * num_train))

train_queue = DataLoader(train_data, batch_size=args.batch_size,
                         sampler=SubsetRandomSampler(indices[:split]), pin_memory=True, num_workers=args.workers)

search_queue = DataLoader(train_data, batch_size=args.batch_size, sampler=SubsetRandomSampler(indices[split:num_train]),
                          pin_memory=True, num_workers=args.workers)

valid_queue = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False,
                         pin_memory=True, num_workers=args.workers)


# init epochs number we have to switch stage in
# epochsSwitchStage = [0]
# for e in args.epochs:
#     epochsSwitchStage.append(e + epochsSwitchStage[-1])
# # total number of epochs is the last value in epochsSwitchStage
# nEpochs = epochsSwitchStage[-1]
# # remove epoch 0 from list, and last switch, since after last switch there are no layers to quantize
# epochsSwitchStage = epochsSwitchStage[1:-1]

epochsSwitchStage = [0]
nLayers = model.nLayers() if hasattr(model, 'nLayers') and callable(getattr(model, 'nLayers')) else args.layers
for _ in range(nLayers):
    epochsSwitchStage.append(20 + epochsSwitchStage[-1])

nEpochs = epochsSwitchStage[-1]
epochsSwitchStage = epochsSwitchStage[1:-1]

logger.info('nEpochs:[{}]'.format(nEpochs))
logger.info('epochsSwitchStage:{}'.format(epochsSwitchStage))

scheduler = CosineAnnealingLR(optimizer, float(nEpochs), eta_min=args.learning_rate_min)

architect = Architect(model, args)

for epoch in range(nEpochs):
#for epoch in range(4):
    # switch stage, i.e. freeze one more layer
    if epoch in epochsSwitchStage:
        model.switch_stage(logger)
        # update optimizer & scheduler due to update in learnable params
        optimizer = SGD(model.parameters(), scheduler.get_lr()[0],
                        momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, float(nEpochs), eta_min=args.learning_rate_min)

    scheduler.step()
    lr = scheduler.get_lr()[0]
    logger.info('Epoch:[{}] , optimizer_lr:[{}], scheduler_lr:[{}]'.format(epoch, optimizer.defaults['lr'], lr))

    # genotype = model.genotype()
    # logger.info('genotype = %s', genotype)

    # print(F.softmax(model.alphas_normal, dim=-1))
    # print(F.softmax(model.alphas_reduce, dim=-1))

    # training
    logger.info('BEFORE:{}'.format(model.block1.ops._modules['0'].op._modules['0']._modules['0'].weight[0, 0]))
    train_acc, train_obj, arch_grad_norm = train(train_queue, search_queue, args, model, architect, criterion, optimizer, lr)
    logger.info('training accuracy:[{:.3f}]'.format(train_acc))
    logger.info('AFTER:{}'.format(model.block1.ops._modules['0'].op._modules['0']._modules['0'].weight[0, 0]))

    # validation
    valid_acc, valid_obj = infer(valid_queue, args, model, criterion)
    logger.info('validation accuracy:[{:.3f}]'.format(valid_acc))

    save(model, os.path.join(args.save, 'weights.pt'))

#discrete_model = DiscreteResNet(criterion, args.nBitsMin, args.nBitsMax, model)
