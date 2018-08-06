import os
import numpy as np
import torch
from shutil import copyfile
import logging
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch import save as saveModel
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets.cifar import CIFAR10


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for v in model.parameters()) / 1e6


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)

    if scripts_to_save is not None:
        os.makedirs(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            copyfile(script, dst_file)


checkpointFileType = 'pth.tar'
stateFilenameDefault = 'model'
stateCheckpointPattern = '{}/{}_checkpoint.' + checkpointFileType
stateOptModelPattern = '{}/{}_opt.' + checkpointFileType


def save_state(state, is_best, path='.', filename=stateFilenameDefault):
    default_filename = stateCheckpointPattern.format(path, filename)
    saveModel(state, default_filename)
    if is_best:
        copyfile(default_filename, stateOptModelPattern.format(path, filename))


def save_checkpoint(path, model, epoch, best_prec1, is_best=False):
    state = dict(epoch=epoch + 1, state_dict=model.state_dict(), alphas=model.alphas_state(),
                 nLayersQuantCompleted=model.nLayersQuantCompleted, best_prec1=best_prec1)
    save_state(state, is_best, path=path)


def load_pre_trained(path, model, logger, gpu):
    if path is not None:
        if os.path.exists(path):
            # model.loadPreTrainedModel(path, logger, gpu)
            model.loadUNIQPre_trained(path, logger, gpu)
        else:
            logger.info('Failed to load pre-trained from [{}], path does not exists'.format(path))


def setup_logging(log_file, logger_name, propagate=False):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    # logging to stdout
    logger.propagate = propagate

    return logger


def initLogger(folderName, propagate=False):
    filePath = '{}/log.txt'.format(folderName)
    logger = setup_logging(filePath, 'darts', propagate)

    logger.info('Experiment dir: [{}]'.format(folderName))

    return logger


def initTrainLogger(logger_file_name, folder_path, propagate=False):
    # folder_path = '{}/train'.format(save_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    log_file_path = '{}/{}.txt'.format(folder_path, logger_file_name)
    logger = setup_logging(log_file_path, logger_file_name, propagate)

    return logger


def logDominantQuantizedOp(model, k, logger):
    top = model.topOps(k=k)
    logger.info('=============================================')
    logger.info('Top [{}] quantizations per layer:'.format(k))
    logger.info('=============================================')
    attributes = ['bitwidth', 'act_bitwidth']
    for i, layerTop in enumerate(top):
        message = 'Layer:[{}]  '.format(i)
        for idx, w, alpha, layer in layerTop:
            message += 'Idx:[{}]  w:[{:.3f}]  alpha:[{:.3f}]  '.format(idx, w, alpha)
            for attr in attributes:
                v = getattr(layer, attr, None)
                if v:
                    message += '{}:{}  '.format(attr, v)

            message += '||  '

        logger.info(message)

    logger.info('=============================================')


def printModelToFile(model, save_path, fname='model'):
    filePath = '{}/{}.txt'.format(save_path, fname)
    logger = setup_logging(filePath, 'modelLogger')
    logger.info('{}'.format(model))
    logDominantQuantizedOp(model, k=2, logger=logger)


def load_data(args):
    train_transform, valid_transform = _data_transforms_cifar10(args)
    train_data = CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    #### narrow data for debug purposes
    # train_data.train_data = train_data.train_data[0:640]
    # train_data.train_labels = train_data.train_labels[0:640]
    # valid_data.test_data = valid_data.test_data[0:320]
    # valid_data.test_labels = valid_data.test_labels[0:320]
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

    return train_queue, search_queue, valid_queue
