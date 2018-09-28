import os
import numpy as np
from numpy import argmax
import torch
from shutil import copyfile
import logging
from inspect import getfile, currentframe, isclass
from os import path, listdir, walk
from smtplib import SMTP
from email import encoders
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from base64 import b64decode
from zipfile import ZipFile, ZIP_DEFLATED
from json import dump

from torch.autograd import Variable
from torch import save as saveModel
from torch import load as loadModel
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from UNIQ.preprocess import get_transform
from UNIQ.data import get_dataset

import cnn.models as models
import cnn.gradEstimators as gradEstimators

# import torchvision.transforms as transforms
# import torchvision.transforms as transforms

# references to models pre-trained
modelsRefs = {
    'thin_resnet': '/home/yochaiz/DropDarts/cnn/pre_trained/thin_resnet/train/model_opt.pth.tar',
    'thin_resnet_w:[2]_a:[32]': '/home/yochaiz/DropDarts/cnn/uniform/thin_resnet_w:[2]_a:[32]/train/model_opt.pth.tar',
    'thin_resnet_w:[3]_a:[3]': '/home/yochaiz/DropDarts/cnn/uniform/thin_resnet_w:[3]_a:[3]/train/model_opt.pth.tar',
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


def logUniformModel(args, logger):
    uniformBops = args.MaxBopsBits[0]
    uniformKey = '{}_w:[{}]_a:[{}]'.format(args.model, uniformBops[0], uniformBops[-1])
    uniformPath = modelsRefs.get(uniformKey)
    best_prec1 = 'Not found'
    if uniformPath and path.exists(uniformPath):
        uniform_checkpoint = loadModel(uniformPath,
                                       map_location=lambda storage, loc: storage.cuda(args.gpu[0]))
        best_prec1 = uniform_checkpoint.get('best_prec1', best_prec1)
    # print result
    logger.info('Uniform {} validation accuracy:[{}]'.format(uniformKey, best_prec1))


# collect possible models names
def loadModelNames():
    return [name for (name, obj) in models.__dict__.items() if isclass(obj) and name.islower()]


# collect possible gradient estimators names
def loadGradEstimatorsNames():
    return [name for (name, obj) in gradEstimators.__dict__.items() if isclass(obj) and name.islower()]


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


def saveArgsToJSON(args):
    # save args to JSON
    args.jsonPath = '{}/args.json'.format(args.save)
    with open(args.jsonPath, 'w') as f:
        dump(vars(args), f)


def zipFolder(p, zipf):
    # get p folder relative path
    folderName = path.relpath(p)
    for base, dirs, files in walk(p):
        if base.endswith('__pycache__'):
            continue

        for file in files:
            if file.endswith('.tar'):
                continue

            fn = path.join(base, file)
            zipf.write(fn, fn[fn.index(folderName):])


def zipFiles(saveFolder, zipFname, attachPaths):
    zipPath = '{}/{}'.format(saveFolder, zipFname)
    zipf = ZipFile(zipPath, 'w', ZIP_DEFLATED)
    for p in attachPaths:
        if path.exists(p):
            if path.isdir(p):
                zipFolder(p, zipf)
            else:
                zipf.write(p)
    zipf.close()

    return zipPath


def sendEmail(toAddr, subject, content, zipPath=None, zipFname=None):
    # init email addresses
    fromAddr = "yochaiz@campus.technion.ac.il"
    # init connection
    server = SMTP('smtp.office365.com', 587)
    server.ehlo()
    server.starttls()
    server.ehlo()
    passwd = b'WXo4Nzk1NzE='
    server.login(fromAddr, b64decode(passwd).decode('utf-8'))
    # init message
    msg = MIMEMultipart()
    msg['From'] = fromAddr
    msg['Subject'] = subject
    msg.attach(MIMEText(content, 'plain'))

    # add zip file if required
    if (zipPath is not None) and (zipFname is not None):
        with open(zipPath, 'rb') as z:
            # attach zip file
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(z.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', "attachment; filename= %s" % zipFname)
            msg.attach(part)

    # send message
    for dst in toAddr:
        msg['To'] = dst
        text = msg.as_string()
        try:
            server.sendmail(fromAddr, dst, text)
        except Exception as e:
            print('Sending email failed, error:[{}]'.format(e))
    server.close()


def sendDataEmail(model, args, trainFolderPath, content):
    saveFolder = args.save
    # init files to zip
    attachPaths = [trainFolderPath, model.alphasCsvFileName, model.stats.saveFolder, args.jsonPath,
                   model._criterion.bopsLossImgPath]
    # zip files
    zipFname = 'attach.zip'
    zipPath = zipFiles(saveFolder, zipFname, attachPaths)
    # init email addresses
    toAddr = ['evron.itay@gmail.com', 'chaimbaskin@cs.technion.ac.il', 'evgeniizh@campus.technion.ac.il',
              'yochaiz.cs@gmail.com', ]
    # init subject
    subject = 'Results [{}] - Model:[{}] Bitwidth:{}'.format(args.folderName, args.model, args.bitwidth)
    # send email
    sendEmail(toAddr, subject, content, zipPath, zipFname)


def create_exp_dir(resultFolderPath):
    # create folders
    if not os.path.exists(resultFolderPath):
        os.makedirs(resultFolderPath)

    zipPath = '{}/code.zip'.format(resultFolderPath)
    zipf = ZipFile(zipPath, 'w', ZIP_DEFLATED)

    # init project base folder
    baseFolder = path.dirname(path.abspath(getfile(currentframe())))  # script directory
    baseFolder += '/../'
    # init folders we want to zip
    foldersToZip = ['cnn/models', 'cnn/trainRegime', 'UNIQ']
    # save folders files
    for folder in foldersToZip:
        folderFullPath = baseFolder + folder
        zipFolder(folderFullPath, zipf)

    # save cnn folder files
    foldersToZip = ['cnn']
    for folder in foldersToZip:
        folderFullPath = baseFolder + folder
        for file in listdir(folderFullPath):
            if path.isfile(file):
                zipf.write(file)

    # close zip file
    zipf.close()


checkpointFileType = 'pth.tar'
stateFilenameDefault = 'model'
stateCheckpointPattern = '{}/{}_checkpoint.' + checkpointFileType
stateOptModelPattern = '{}/{}_opt.' + checkpointFileType


def save_state(state, is_best, path, filename):
    default_filename = stateCheckpointPattern.format(path, filename)
    saveModel(state, default_filename)
    if is_best:
        copyfile(default_filename, stateOptModelPattern.format(path, filename))


def save_checkpoint(path, model, epoch, best_prec1, is_best=False, filename=None):
    # set state dictionary
    state = dict(epoch=epoch + 1, state_dict=model.state_dict(), alphas=model.alphas_state(),
                 nLayersQuantCompleted=model.nLayersQuantCompleted, best_prec1=best_prec1)
    # set state filename
    filename = filename or stateFilenameDefault
    # save state to file
    save_state(state, is_best, path=path, filename=filename)


def load_pre_trained(path, model, logger, gpu):
    # init bool flag whether we loaded ops in the same layer with equal or different weights
    loadOpsWithDifferentWeights = False
    if path is not None:
        if os.path.exists(path):
            # load checkpoint
            checkpoint = loadModel(path, map_location=lambda storage, loc: storage.cuda(gpu))
            chckpntStateDict = checkpoint['state_dict']

            # remove prev layers
            prevLayers = []
            for layer in model.layersList:
                prevLayers.append(layer.prevLayer)
                layer.prevLayer = None

            # load model state dict keys
            modelStateDictKeys = set(model.state_dict().keys())
            # compare dictionaries
            dictDiff = modelStateDictKeys.symmetric_difference(set(chckpntStateDict.keys()))
            # # update flag value
            # loadOpsWithDifferentWeights = True
            # for v in dictDiff:
            #     if v.endswith('.num_batches_tracked') is False:
            #         loadOpsWithDifferentWeights = False
            #         break
            #     else:
            #         chckpntStateDict.pop(v)
            # update flag value
            loadOpsWithDifferentWeights = len(dictDiff) == 0
            # decide how to load checkpoint state dict
            if loadOpsWithDifferentWeights:
                # load directly, keys are the same
                model.load_state_dict(chckpntStateDict)
            else:
                # use some function to map keys
                model.loadUNIQPre_trained(chckpntStateDict)

            # restore prev layers
            for pLayer, layer in zip(prevLayers, model.layersList):
                layer.prevLayer = pLayer

            logger.info('Loaded model from [{}]'.format(path))
            logger.info('checkpoint validation accuracy:[{:.5f}]'.format(checkpoint['best_prec1']))
        else:
            logger.info('Failed to load pre-trained from [{}], path does not exists'.format(path))

    return loadOpsWithDifferentWeights


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
    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    logger = setup_logging(log_file_path, logger_file_name, propagate)

    return logger


def logForwardCounters(model, trainLogger):
    if not trainLogger:
        return

    trainLogger.info('=============================================')
    trainLogger.info('=========== Ops forward counters ============')
    trainLogger.info('=============================================')
    for layerIdx, layer in enumerate(model.layersList):
        trainLogger.info('Layer:[{}]  '.format(layerIdx))
        # collect layer counters to 2 arrays:
        # counters holds the counters values
        # indices holds the corresponding counter value indices
        counters, indices = [], []
        for i, counterList in enumerate(layer.opsForwardCounters):
            for j, counter in enumerate(counterList):
                counters.append(counter)
                indices.append((i, j))

        # for each layer, sort counters in descending order
        msg = ''
        while len(counters) > 0:
            # find max counter and print it
            maxIdx = argmax(counters)
            i, j = indices[maxIdx]
            msg += '[{}][{}]: [{}] || '.format(i, j, counters[maxIdx])
            # remove max counter from lists
            del counters[maxIdx]
            del indices[maxIdx]

        trainLogger.info(msg)

        # reset layer counters
        layer.resetOpsForwardCounters()

    trainLogger.info('=============================================')


def logDominantQuantizedOp(model, k, logger):
    if not logger:
        return

    top = model.topOps(k=k)
    logger.info('=============================================')
    logger.info('Top [{}] quantizations per layer:'.format(k))
    logger.info('=============================================')
    attributes = ['bitwidth', 'act_bitwidth']
    for i, layerTop in enumerate(top):
        message = 'Layer:[{}]  '.format(i)
        for idx, w, alpha, layer in layerTop:
            message += 'Idx:[{}]  w:[{:.5f}]  alpha:[{:.5f}]  '.format(idx, w, alpha)
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


# def _data_transforms_cifar10(args):
#     CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
#     CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
#
#     train_transform = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
#     ])
#     if args.cutout:
#         train_transform.transforms.append(Cutout(args.cutout_length))
#
#     valid_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
#     ])
#     return train_transform, valid_transform

def load_data(args):
    # train_transform, valid_transform = _data_transforms_cifar10(args)
    # train_data = CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    # valid_data = CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    # init transforms
    transform = {
        'train': get_transform(args.dataset, augment=True),
        'eval': get_transform(args.dataset, augment=False)
    }

    train_data = get_dataset(args.dataset, train=True, transform=transform['train'], datasets_path=args.data)
    valid_data = get_dataset(args.dataset, train=False, transform=transform['eval'], datasets_path=args.data)

    ### narrow data for debug purposes
    # train_data.train_data = train_data.train_data[0:50]
    # train_data.train_labels = train_data.train_labels[0:50]
    # valid_data.test_data = valid_data.test_data[0:100]
    # valid_data.test_labels = valid_data.test_labels[0:100]
    ####

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = DataLoader(train_data, batch_size=args.batch_size,
                             sampler=SubsetRandomSampler(indices[:split]), pin_memory=True, num_workers=args.workers)

    search_queue = DataLoader(train_data, batch_size=args.batch_size,
                              sampler=SubsetRandomSampler(indices[split:num_train]),
                              pin_memory=True, num_workers=args.workers)

    valid_queue = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False,
                             pin_memory=True, num_workers=args.workers)

    return train_queue, search_queue, valid_queue
