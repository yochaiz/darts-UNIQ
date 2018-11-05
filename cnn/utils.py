import os
import numpy as np
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

from cnn.HtmlLogger import HtmlLogger
import cnn.gradEstimators as gradEstimators
import cnn.models as models


def loadCheckpoint(dataset, model, bitwidth, filename='model_opt.pth.tar'):
    # init project base folder
    baseFolder = path.dirname(path.abspath(getfile(currentframe())))  # script directory
    # init checkpoint key
    checkpointKey = '{}_{}'.format(model, bitwidth)
    # init checkpoint path
    checkpointPath = '{}/../pre_trained/{}/train_portion_1.0/{}/train/{}'.format(baseFolder, dataset, checkpointKey, filename)
    # load checkpoint
    checkpoint = None
    if path.exists(checkpointPath):
        checkpoint = loadModel(checkpointPath, map_location=lambda storage, loc: storage.cuda())

    return checkpoint, checkpointKey


def logBaselineModel(args, logger):
    # get baseline bitwidth
    bitwidth = args.baselineBits[0]
    # load baseline checkpoint
    checkpoint, checkpointKey = loadCheckpoint(args.dataset, args.model, bitwidth)
    # init logger rows
    loggerRows = []
    # init best_prec1 values
    best_prec1_str = 'Not found'

    if checkpoint is not None:
        keysFromUniform = ['epochs', 'learning_rate']
        # extract keys from uniform checkpoint
        for key in keysFromUniform:
            if key in checkpoint:
                value = checkpoint.get(key)
                if args.copyBaselineKeys:
                    setattr(args, key, value)
                    if logger:
                        loggerRows.append(['Loaded key [{}]'.format(key), value])
                elif logger:
                    loggerRows.append(['{}'.format(key), '{}'.format(value)])
        # extract best_prec1 from uniform checkpoint
        best_prec1 = checkpoint.get('best_prec1')
        if best_prec1:
            best_prec1_str = '{:.3f}'.format(best_prec1)

    # print result
    if logger:
        loggerRows.append(['Model', '{}'.format(checkpointKey)])
        loggerRows.append(['Validation accuracy', '{}'.format(best_prec1_str)])
        logger.addInfoTable('Baseline model', loggerRows)


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
    args.jsonPath = '{}/args.txt'.format(args.save)
    with open(args.jsonPath, 'w') as f:
        dump(vars(args), f, indent=4, sort_keys=True)


def logParameters(logger, args, model):
    if not logger:
        return

    # calc number of permutations
    permutationStr = model.nPerms
    for p in [12, 9, 6, 3]:
        v = model.nPerms / (10 ** p)
        if v > 1:
            permutationStr = '{:.3f} * 10<sup>{}</sup>'.format(v, p)
            break
    # log other parameters
    logger.addInfoTable('Parameters', HtmlLogger.dictToRows(
        {
            'Parameters size': '{:.3f} MB'.format(count_parameters_in_MB(model)),
            'Learnable params': len(model.learnable_params),
            'Ops per layer': [layer.numOfOps() for layer in model.layersList],
            'Permutations': permutationStr
        }, nElementPerRow=2))
    # log baseline model
    logBaselineModel(args, logger)
    # log args
    logger.addInfoTable('args', HtmlLogger.dictToRows(vars(args), nElementPerRow=3))
    # print args
    print(args)


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


# msg - email message, MIMEMultipart() object
def attachFiletoEmail(msg, fileFullPath):
    with open(fileFullPath, 'rb') as z:
        # attach file
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(z.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', "attachment; filename= %s" % path.basename(fileFullPath))
        msg.attach(part)


def sendEmail(toAddr, subject, content, attachments=None):
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

    if attachments:
        for att in attachments:
            if path.exists(att):
                if path.isdir(att):
                    for filename in listdir(att):
                        attachFiletoEmail(msg, '{}/{}'.format(att, filename))
                else:
                    attachFiletoEmail(msg, att)

    # send message
    for dst in toAddr:
        msg['To'] = dst
        text = msg.as_string()
        try:
            server.sendmail(fromAddr, dst, text)
        except Exception as e:
            print('Sending email failed, error:[{}]'.format(e))

    server.close()


def sendDataEmail(model, args, logger, content):
    # init files to send
    attachments = [model.alphasCsvFileName, model.stats.saveFolder, args.jsonPath, model._criterion.bopsLossImgPath, logger.fullPath]
    # init subject
    subject = 'Results [{}] - Model:[{}] Bitwidth:{} dataset:[{}] lambda:[{}]' \
        .format(args.folderName, args.model, args.bitwidth, args.dataset, args.lmbda)
    # send email
    sendEmail(args.recipients, subject, content, attachments)


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
    foldersToZip = ['cnn/models', 'cnn/trainRegime', 'cnn/gradEstimators', 'UNIQ', 'NICE']
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

    is_best_filename = None
    if is_best:
        is_best_filename = stateOptModelPattern.format(path, filename)
        copyfile(default_filename, is_best_filename)

    return default_filename, is_best_filename


def save_checkpoint(path, model, args, epoch, best_prec1, is_best=False, filename=None):
    print('*** save_checkpoint ***')
    # we want to save full-precision weights, we will quantize them after loading them
    model.removeQuantizationFromStagedLayers()
    # set state dictionary
    state = dict(nextEpoch=epoch + 1, state_dict=model.state_dict(), epochs=args.epochs, alphas=model.save_alphas_state(), updated_statistics=False,
                 nLayersQuantCompleted=model.nLayersQuantCompleted, best_prec1=best_prec1, learning_rate=args.learning_rate)
    # set state filename
    filename = filename or stateFilenameDefault
    # save state to file
    filePaths = save_state(state, is_best, path=path, filename=filename)

    # restore quantization in staged layers
    model.restoreQuantizationForStagedLayers()
    print('*** END save_checkpoint ***')

    return state, filePaths


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


def loadDatasets():
    return dict(cifar10=10, cifar100=100, imagenet=1000)


def load_data(args):
    # init transforms
    transform = {
        'train': get_transform(args.dataset, augment=True),
        'eval': get_transform(args.dataset, augment=False)
    }

    train_data = get_dataset(args.dataset, train=True, transform=transform['train'], datasets_path=args.data)
    valid_data = get_dataset(args.dataset, train=False, transform=transform['eval'], datasets_path=args.data)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = DataLoader(train_data, batch_size=args.batch_size,
                             sampler=SubsetRandomSampler(indices[:split]), pin_memory=True, num_workers=args.workers)

    statistics_queue = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=args.workers, pin_memory=True)

    # search_queue = DataLoader(train_data, batch_size=args.batch_size,
    #                           sampler=SubsetRandomSampler(indices[split:num_train]),
    #                           pin_memory=True, num_workers=args.workers)

    valid_queue = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    # split search_queue to parts
    nParts = args.alphas_data_parts
    nSamples = num_train - split
    nSamplesPerPart = int(nSamples / nParts)
    startIdx = split
    endIdx = startIdx + nSamplesPerPart
    search_queue = []
    for _ in range(nParts - 1):
        dl = DataLoader(train_data, batch_size=args.batch_size, sampler=SubsetRandomSampler(indices[startIdx:endIdx]),
                        pin_memory=True, num_workers=args.workers)
        search_queue.append(dl)
        startIdx = endIdx
        endIdx += nSamplesPerPart
    # last part takes what left
    dl = DataLoader(train_data, batch_size=args.batch_size, sampler=SubsetRandomSampler(indices[startIdx:num_train]),
                    pin_memory=True, num_workers=args.workers)
    search_queue.append(dl)

    return train_queue, search_queue, valid_queue, statistics_queue

# def logDominantQuantizedOpOLD(model, k, logger):
#     if not logger:
#         return
#
#     top = model.topOps(k=k)
#     logger.info('=============================================')
#     logger.info('Top [{}] quantizations per layer:'.format(k))
#     logger.info('=============================================')
#     attributes = ['bitwidth', 'act_bitwidth']
#     for i, layerTop in enumerate(top):
#         message = 'Layer:[{}]  '.format(i)
#         for idx, w, alpha, layer in layerTop:
#             message += 'Idx:[{}]  w:[{:.5f}]  alpha:[{:.5f}]  '.format(idx, w, alpha)
#             for attr in attributes:
#                 v = getattr(layer, attr, None)
#                 if v:
#                     message += '{}:{}  '.format(attr, v)
#
#             message += '||  '
#
#         logger.info(message)
#
#     logger.info('=============================================')

# def printModelToFile(model, save_path, fname='model'):
#     filePath = '{}/{}.txt'.format(save_path, fname)
#     logger = setup_logging(filePath, 'modelLogger')
#     logger.info('{}'.format(model))
#     logDominantQuantizedOpOLD(model, k=2, logger=logger)

# # references to UNIQ baseline models
# modelsRefsUNIQ = {
#     'thin_resnet': '/home/yochaiz/DropDarts/cnn/pre_trained/thin_resnet/train/model_opt.pth.tar',
#     'thin_resnet_w:[32]_a:[32]': '/home/yochaiz/DropDarts/cnn/pre_trained/thin_resnet/train/model_opt.pth.tar',
#     'thin_resnet_w:[2]_a:[32]': '/home/yochaiz/DropDarts/cnn/uniform/thin_resnet_w:[2]_a:[32]/train/model_opt.pth.tar',
#     'thin_resnet_w:[3]_a:[3]': '/home/yochaiz/DropDarts/cnn/uniform/thin_resnet_w:[3]_a:[3]/train/model_opt.pth.tar',
#     'thin_resnet_w:[6]_a:[6]': '/home/yochaiz/DropDarts/cnn/uniform/thin_resnet_w:[6]_a:[6]/train/model_checkpoint.pth.tar',
#     'resnet': '/home/yochaiz/DropDarts/cnn/pre_trained/resnet_cifar10_trained_32_bit_deeper/model_best.pth.tar',
#     'resnet_w:[32]_a:[32]': '/home/yochaiz/DropDarts/cnn/pre_trained/resnet_cifar10_trained_32_bit_deeper/model_best.pth.tar',
#     'resnet_w:[1]_a:[1]': '/home/yochaiz/DropDarts/cnn/uniform/resnet_w:[1]_a:[1]/train/model_opt.pth.tar',
#     'resnet_w:[1]_a:[32]': '/home/yochaiz/DropDarts/cnn/uniform/resnet_w:[1]_a:[32]/train/model_opt.pth.tar',
#     'resnet_w:[2]_a:[32]': '/home/yochaiz/DropDarts/cnn/uniform/resnet_w:[2]_a:[32]/train/model_opt.pth.tar',
#     'resnet_w:[2]_a:[3]': '/home/yochaiz/DropDarts/cnn/uniform/resnet_w:[2]_a:[3]/train/model_opt.pth.tar',
#     'resnet_w:[2]_a:[4]': '/home/yochaiz/DropDarts/cnn/uniform/resnet_w:[2]_a:[4]/train/model_opt.pth.tar',
#     'resnet_w:[3]_a:[3]': '/home/yochaiz/DropDarts/cnn/uniform/resnet_w:[3]_a:[3]/train/model_opt.pth.tar',
#     'resnet_w:[3]_a:[32]': '/home/yochaiz/DropDarts/cnn/uniform/resnet_w:[3]_a:[32]/train/model_opt.pth.tar',
#     'resnet_w:[4]_a:[4]': '/home/yochaiz/DropDarts/cnn/uniform/resnet_w:[4]_a:[4]/train/model_opt.pth.tar',
#     'resnet_w:[4]_a:[32]': '/home/yochaiz/DropDarts/cnn/uniform/resnet_w:[4]_a:[32]/train/model_opt.pth.tar',
#     'resnet_w:[5]_a:[5]': '/home/yochaiz/DropDarts/cnn/uniform/resnet_w:[5]_a:[5]/train/model_opt.pth.tar',
#     'resnet_w:[6]_a:[6]': '/home/yochaiz/DropDarts/cnn/uniform/resnet_w:[6]_a:[6]/train/model_opt.pth.tar',
#     'resnet_w:[8]_a:[32]': '/home/yochaiz/DropDarts/cnn/uniform/resnet_w:[8]_a:[32]/train/model_checkpoint.pth.tar',
# }

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

# class Cutout(object):
#     def __init__(self, length):
#         self.length = length
#
#     def __call__(self, img):
#         h, w = img.size(1), img.size(2)
#         mask = np.ones((h, w), np.float32)
#         y = np.random.randint(h)
#         x = np.random.randint(w)
#
#         y1 = np.clip(y - self.length // 2, 0, h)
#         y2 = np.clip(y + self.length // 2, 0, h)
#         x1 = np.clip(x - self.length // 2, 0, w)
#         x2 = np.clip(x + self.length // 2, 0, w)
#
#         mask[y1: y2, x1: x2] = 0.
#         mask = torch.from_numpy(mask)
#         mask = mask.expand_as(img)
#         img *= mask
#         return img
