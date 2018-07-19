# import torch.nn.functional as F
import os
from sys import stdout, exit
import time
import glob
import numpy as np
import logging
import argparse
from torch.nn import CrossEntropyLoss
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.datasets.cifar import CIFAR10
import torch.backends.cudnn as cudnn
from torch.cuda import is_available, set_device
from torch.cuda import manual_seed as cuda_manual_seed
from torch import manual_seed as torch_manual_seed
from torch.optim import SGD

from torch.autograd import Variable
from cnn.utils import create_exp_dir, count_parameters_in_MB, _data_transforms_cifar10, accuracy, AvgrageMeter, save
from cnn.model_search import Network
from cnn.architect import Architect

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/home/yochaiz/UNIQ/results', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=1, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
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
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10


def main():
    if not is_available():
        logging.info('no gpu device available')
        exit(1)

    np.random.seed(args.seed)
    set_device(args.gpu)
    cudnn.benchmark = True
    torch_manual_seed(args.seed)
    cudnn.enabled = True
    cuda_manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    criterion = CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    model = model.cuda()
    logging.info("param size = %fMB", count_parameters_in_MB(model))

    optimizer = SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    train_transform, valid_transform = _data_transforms_cifar10(args)
    train_data = CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = DataLoader(train_data, batch_size=args.batch_size,
                             sampler=SubsetRandomSampler(indices[:split]), pin_memory=True, num_workers=2)

    search_queue = DataLoader(train_data, batch_size=args.batch_size,
                              sampler=SubsetRandomSampler(indices[split:num_train]), pin_memory=True, num_workers=2)

    valid_queue = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False,
                             pin_memory=True, num_workers=2)

    scheduler = CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        # print(F.softmax(model.alphas_normal, dim=-1))
        # print(F.softmax(model.alphas_reduce, dim=-1))

        # training
        train_acc, train_obj, arch_grad_norm = train(train_queue, search_queue, model, architect, criterion, optimizer, lr)
        logging.info('training accuracy:[{:.3f}]'.format(train_acc))

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('validation accuracy:[{:.3f}]'.format(valid_acc))

        save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, search_queue, model, architect, criterion, optimizer, lr):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    grad = AvgrageMeter()

    nBatches = len(train_queue)

    for step, (input, target) in enumerate(train_queue):
        model.train()
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

        if step % args.report_freq == 0:
            logging.info('train [{}/{}] Loss:[{:.5f}] Accuracy:[{:.3f}]'.format(step, nBatches, objs.avg, top1.avg))

    return top1.avg, objs.avg, grad.avg


def infer(valid_queue, model, criterion):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    model.eval()

    nBatches = len(valid_queue)

    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda(async=True)

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('validation [{}/{}] Loss:[{:.5f}] Accuracy:[{:.3f}]'.format(step, nBatches, objs.avg, top1.avg))

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
