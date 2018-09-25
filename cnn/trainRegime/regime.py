from time import time
from abc import abstractmethod
from os import makedirs, path, system

from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import no_grad
from torch.optim import SGD
from torch.autograd.variable import Variable
from torch.nn import CrossEntropyLoss

from cnn.utils import accuracy, AvgrageMeter, load_data, saveArgsToJSON
from cnn.utils import initTrainLogger, logDominantQuantizedOp, save_checkpoint
from cnn.utils import sendEmail as uSendEmail, logForwardCounters


def trainWeights(train_queue, model, modelChoosePathFunc, crit, optimizer, grad_clip, nEpoch, loggers):
    loss_container = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    trainLogger = loggers.get('train')

    model.train()

    nBatches = len(train_queue)

    for step, (input, target) in enumerate(train_queue):
        startTime = time()
        n = input.size(0)

        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda(async=True)

        # choose alpha per layer
        bopsRatio = modelChoosePathFunc()
        # optimize model weights
        optimizer.zero_grad()
        logits = model(input)
        # calc loss
        loss = crit(logits, target)
        # back propagate
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        # update weights
        optimizer.step()

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        loss_container.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        endTime = time()

        if trainLogger:
            trainLogger.info(
                'train [{}/{}] weight_loss:[{:.5f}] Accuracy:[{:.3f}] PathBopsRatio:[{:.3f}] time:[{:.5f}]'
                    .format(step, nBatches, loss_container.avg, top1.avg, bopsRatio, endTime - startTime))

    # log accuracy, loss, etc.
    message = 'Epoch:[{}] , training accuracy:[{:.3f}] , training loss:[{:.3f}] , optimizer_lr:[{:.5f}]' \
        .format(nEpoch, top1.avg, loss_container.avg, optimizer.param_groups[0]['lr'])

    for _, logger in loggers.items():
        logger.info(message)

    # log dominant QuantizedOp in each layer
    logDominantQuantizedOp(model, k=2, logger=trainLogger)

    # log forward counters
    logForwardCounters(model, trainLogger)


def infer(valid_queue, model, modelInferMode, crit, nEpoch, loggers):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    trainLogger = loggers.get('train')

    model.eval()
    bopsRatio = modelInferMode()
    # print eval layer index selection
    if trainLogger:
        trainLogger.info('Layers optimal indices:{}'.format([layer.curr_alpha_idx for layer in model.layersList]))

    nBatches = len(valid_queue)

    with no_grad():
        for step, (input, target) in enumerate(valid_queue):
            startTime = time()

            input = Variable(input).cuda()
            target = Variable(target).cuda(async=True)

            logits = model(input)
            loss = crit(logits, target)

            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            endTime = time()

            if trainLogger:
                trainLogger.info(
                    'validation [{}/{}] Loss:[{:.5f}] Accuracy:[{:.3f}] OptBopsRatio:[{:.3f}] time:[{:.5f}]'
                        .format(step, nBatches, objs.avg, top1.avg, bopsRatio, endTime - startTime))

    message = 'Epoch:[{}] , validation accuracy:[{:.3f}] , validation loss:[{:.3f}] , OptBopsRatio:[{:.3f}]' \
        .format(nEpoch, top1.avg, objs.avg, bopsRatio)

    for _, logger in loggers.items():
        logger.info(message)

    return top1.avg


class TrainRegime:
    def __init__(self, args, model, modelClass, logger):
        self.args = args
        self.model = model
        self.modelClass = modelClass
        self.logger = logger

        # init email time
        self.lastMailTime = time()
        self.secondsBetweenMails = 1 * 3600

        # init train optimal model counter.
        # each key is an allocation, and the map hold a counter per key, how many batches this allocation is optimal
        self.optModelBitwidthCounter = {}
        # init optimal model training queue, in case we send too many jobs to server
        self.optModelTrainingQueue = []

        self.trainFolderPath = '{}/{}'.format(args.save, args.trainFolder)

        # init cross entropy loss
        self.cross_entropy = CrossEntropyLoss().cuda()

        # load data
        self.train_queue, self.search_queue, self.valid_queue = load_data(args)

        # extend epochs list as number of model layers
        while len(args.epochs) < model.nLayers():
            args.epochs.append(args.epochs[-1])
        # init epochs number where we have to switch stage in
        epochsSwitchStage = [0]
        for e in args.epochs:
            epochsSwitchStage.append(e + epochsSwitchStage[-1])
        # total number of epochs is the last value in epochsSwitchStage
        nEpochs = epochsSwitchStage[-1] + args.epochs[-1]
        # remove epoch 0 from list, we don't want to switch stage at the beginning
        epochsSwitchStage = epochsSwitchStage[1:]

        logger.info('nEpochs:[{}]'.format(nEpochs))
        logger.info('epochsSwitchStage:{}'.format(epochsSwitchStage))

        # init epoch
        self.epoch = 0
        self.nEpochs = nEpochs
        self.epochsSwitchStage = epochsSwitchStage

        # if we loaded ops in the same layer with the same weights, then we loaded the optimal full precision model,
        # therefore we have to train the weights for each QuantizedOp
        if args.loadedOpsWithDiffWeights is False:
            self.epoch = self.gwow(model, args, trainFolderName='init_weights_train')
        else:
            # we loaded ops in the same layer with different weights, therefore we just have to switch_stage
            switchStageFlag = True
            while switchStageFlag:
                switchStageFlag = model.switch_stage(logger)

    @abstractmethod
    def train(self):
        raise NotImplementedError('subclasses must override train()!')

    def __getBitwidthKey(self, optModel_bitwidth):
        return '{}'.format(optModel_bitwidth)

    def setOptModelBitwidth(self):
        model = self.model
        args = self.args

        args.optModel_bitwidth = [layer.getBitwidth() for layer in model.layersList]
        # check if current bitwidth has already been sent for training
        bitwidthKey = self.__getBitwidthKey(args.optModel_bitwidth)

        return bitwidthKey

    # try to send more queued jobs to server
    def trySendQueuedJobs(self):
        args = self.args

        trySendJobs = len(self.optModelTrainingQueue) > 0
        while trySendJobs:
            # take last job in queue
            command, optModel_bitwidth = self.optModelTrainingQueue[-1]
            # update optModel_bitwidth in args
            args.optModel_bitwidth = optModel_bitwidth
            # get key
            bitwidthKey = self.__getBitwidthKey(optModel_bitwidth)
            # save args to JSON
            saveArgsToJSON(args)
            # send job
            retVal = system(command)

            if retVal == 0:
                # delete the job we sent from queue
                self.optModelTrainingQueue = self.optModelTrainingQueue[:-1]
                print('sent model with allocation:{}, queue size:[{}]'
                      .format(bitwidthKey, len(self.optModelTrainingQueue)))

            # update loop flag, keep sending if current job sent successfully & there are more jobs to send
            trySendJobs = (retVal == 0) and (len(self.optModelTrainingQueue) > 0)

    def sendOptModel(self, bitwidthKey, nEpoch, nBatch):
        args = self.args

        # check if this is the 1st time this allocation is optimal
        if bitwidthKey not in self.optModelBitwidthCounter:
            self.optModelBitwidthCounter[bitwidthKey] = 0

        # increase model allocation counter
        self.optModelBitwidthCounter[bitwidthKey] += 1

        # if this allocation has been optimal enough batches, let's check its accuracy
        if self.optModelBitwidthCounter[bitwidthKey] >= 20:
            # save args to JSON
            saveArgsToJSON(args)
            # init args JSON destination path on server
            dstPath = '/home/yochaiz/DropDarts/cnn/optimal_models/{}/{}-[{}-{}].json' \
                .format(args.model, args.folderName, nEpoch, nBatch)
            # init copy command & train command
            copyJSONcommand = 'scp {} yochaiz@132.68.39.32:{}'.format(args.jsonPath, dstPath)
            trainOptCommand = 'ssh yochaiz@132.68.39.32 sbatch /home/yochaiz/DropDarts/cnn/sbatch_opt.sh --data {}' \
                .format(dstPath)
            # perform commands
            print('%%%%%%%%%%%%%%')
            print('sent model with allocation:{}, queue size:[{}]'
                  .format(bitwidthKey, len(self.optModelTrainingQueue)))
            command = '{} && {}'.format(copyJSONcommand, trainOptCommand)
            retVal = system(command)

            if retVal == 0:
                self.trySendQueuedJobs()
            else:
                # server is full with jobs, add current job to queue
                self.optModelTrainingQueue.append((command, args.optModel_bitwidth))
                print('No available GPU, adding {} to queue, queue size:[{}]'
                      .format(bitwidthKey, len(self.optModelTrainingQueue)))
                # remove args JSON
                system('ssh yochaiz@132.68.39.32 rm {}'.format(dstPath))

    def trainOptimalModel(self, nEpoch, nBatch):
        model = self.model
        # set optimal model bitwidth per layer
        model.evalMode()
        bitwidthKey = self.setOptModelBitwidth()
        # train optimal model
        self.sendOptModel(bitwidthKey, nEpoch, nBatch)

    def gwow(self, model, args, trainFolderName, filename=None):
        nEpochs = self.nEpochs

        # create train folder
        folderPath = '{}/{}'.format(self.trainFolderPath, trainFolderName)
        if not path.exists(folderPath):
            makedirs(folderPath)

        # init optimizer
        optimizer = SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                        weight_decay=args.weight_decay)
        # init scheduler
        scheduler = CosineAnnealingLR(optimizer, float(nEpochs), eta_min=args.learning_rate_min)

        # init validation best precision value
        best_prec1 = 0.0

        for epoch in range(1, nEpochs + 1):
            trainLogger = initTrainLogger(str(epoch), folderPath, args.propagate)
            # set loggers dictionary
            loggersDict = dict(train=trainLogger, main=self.logger)

            scheduler.step()
            lr = scheduler.get_lr()[0]

            trainLogger.info(
                'optimizer_lr:[{:.5f}], scheduler_lr:[{:.5f}]'.format(optimizer.param_groups[0]['lr'], lr))

            # training
            print('========== Epoch:[{}] =============='.format(epoch))
            trainWeights(self.train_queue, model, model.chooseRandomPath, self.cross_entropy, optimizer, args.grad_clip,
                         epoch, loggersDict)

            # switch stage, i.e. freeze one more layer
            if (epoch in self.epochsSwitchStage) or (epoch == nEpochs):
                # validation
                valid_acc = infer(self.valid_queue, model, model.evalMode, self.cross_entropy, epoch, loggersDict)

                # switch stage
                model.switch_stage(trainLogger)
                # update optimizer & scheduler due to update in learnable params
                optimizer = SGD(model.parameters(), scheduler.get_lr()[0],
                                momentum=args.momentum, weight_decay=args.weight_decay)
                scheduler = CosineAnnealingLR(optimizer, float(nEpochs), eta_min=args.learning_rate_min)
                scheduler.step()

                # save model checkpoint
                is_best = valid_acc > best_prec1
                best_prec1 = max(valid_acc, best_prec1)
                save_checkpoint(self.trainFolderPath, model, epoch, best_prec1, is_best, filename)
            else:
                # save model checkpoint
                save_checkpoint(self.trainFolderPath, model, epoch, best_prec1, is_best=False, filename=filename)

        self.logger.info('Optimal validation accuracy: [{:.3f}]'.format(best_prec1))

        return epoch

    def sendEmail(self, nEpoch, batchNum, nBatches):
        body = ['Hi', 'Files are attached.', 'Epoch:[{}]  Batch:[{}/{}]'.format(nEpoch, batchNum, nBatches)]
        content = ''
        for line in body:
            content += line + '\n'

        uSendEmail(self.model, self.args, self.trainFolderPath, content)

    def trainAlphas(self, search_queue, model, architect, nEpoch, loggers):
        loss_container = AvgrageMeter()
        optBopsRatio = -1

        trainLogger = loggers.get('train')

        model.train()

        # update model replications weights
        architect.modelReplicator.updateModelWeights(model)
        trainLogger.info('Model replications weights have been updated')

        nBatches = len(search_queue)

        for step, (input, target) in enumerate(search_queue):
            startTime = time()
            n = input.size(0)

            input = Variable(input, requires_grad=False).cuda()
            target = Variable(target, requires_grad=False).cuda(async=True)

            loss = architect.step(model, input, target)

            # train optimal model
            self.trainOptimalModel(nEpoch, step)
            # add alphas data to statistics
            optBopsRatio = model.stats.addBatchData(model, nEpoch, step)
            # log dominant QuantizedOp in each layer
            logDominantQuantizedOp(model, k=2, logger=trainLogger)
            # log forward counters
            logForwardCounters(model, trainLogger)
            # save alphas to csv
            model.save_alphas_to_csv(data=[nEpoch, step])
            # save loss to container
            loss_container.update(loss, n)

            endTime = time()

            # send email
            if (endTime - self.lastMailTime > self.secondsBetweenMails) or ((step + 1) % int(nBatches / 2) == 0):
                self.sendEmail(nEpoch, step, nBatches)
                # update last email time
                self.lastMailTime = time()
                # from now on we send every 5 hours
                self.secondsBetweenMails = 5 * 3600

            if trainLogger:
                trainLogger.info('train [{}/{}] arch_loss:[{:.5f}] OptBopsRatio:[{:.3f}] time:[{:.5f}]'
                                 .format(step, nBatches, loss_container.avg, optBopsRatio, endTime - startTime))

        # log accuracy, loss, etc.
        message = 'Epoch:[{}] , arch loss:[{:.3f}] , OptBopsRatio:[{:.3f}] , lr:[{:.5f}]' \
            .format(nEpoch, loss_container.avg, optBopsRatio, architect.lr)

        for _, logger in loggers.items():
            logger.info(message)
