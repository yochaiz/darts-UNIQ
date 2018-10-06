from time import time, sleep
from abc import abstractmethod
from os import makedirs, path, system

from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import no_grad
from torch.optim import SGD
from torch.autograd.variable import Variable
from torch.nn import CrossEntropyLoss

from cnn.utils import accuracy, AvgrageMeter, load_data, saveArgsToJSON, logDominantQuantizedOp, save_checkpoint
from cnn.utils import sendDataEmail, logForwardCounters
from cnn.HtmlLogger import HtmlLogger


class TrainRegime:
    trainLossKey = 'Training loss'
    trainAccKey = 'Training acc'
    validLossKey = 'Validation loss'
    validAccKey = 'Validation acc'
    archLossKey = 'Arch loss'
    epochNumKey = 'Epoch #'
    batchNumKey = 'Batch #'
    pathBopsRatioKey = 'Path bops ratio'
    optBopsRatioKey = 'Optimal bops ratio'
    timeKey = 'Time'
    lrKey = 'Optimizer lr'

    # init formats for keys
    formats = {validLossKey: '{:.5f}', validAccKey: '{:.3f}', optBopsRatioKey: '{:.3f}', timeKey: '{:.3f}', archLossKey: '{:.5f}', lrKey: '{:.5f}',
               trainLossKey: '{:.5f}', trainAccKey: '{:.3f}', pathBopsRatioKey: '{:.3f}'}

    initWeightsTrainTableTitle = 'Initial weights training'
    alphasTableTitle = 'Alphas (top [{}])'
    forwardCountersTitle = 'Forward counters'

    colsTrainWeights = [batchNumKey, trainLossKey, trainAccKey, pathBopsRatioKey, timeKey]
    colsMainInitWeightsTrain = [epochNumKey, trainLossKey, trainAccKey, validLossKey, validAccKey, pathBopsRatioKey, lrKey]
    colsTrainAlphas = [batchNumKey, archLossKey, alphasTableTitle, forwardCountersTitle, optBopsRatioKey, timeKey]
    colsValidation = [batchNumKey, validLossKey, validAccKey, optBopsRatioKey, timeKey]
    colsMainLogger = [epochNumKey, archLossKey, optBopsRatioKey, trainLossKey, trainAccKey, validLossKey, validAccKey, lrKey]

    def __init__(self, args, model, modelClass, logger):
        self.args = args
        self.model = model
        self.modelClass = modelClass
        self.logger = logger

        # init email time
        self.lastMailTime = time()
        self.secondsBetweenMails = 1 * 3600

        # number of batches for allocation as optimal model in order to train it from full-precision
        self.nBatchesOptModel = 20

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

        logger.addInfoTable('Epochs', [['nEpochs', '{}'.format(nEpochs)], ['epochsSwitchStage', '{}'.format(epochsSwitchStage)]])

        # init epoch
        self.epoch = 0
        self.nEpochs = nEpochs
        self.epochsSwitchStage = epochsSwitchStage

        # if we loaded ops in the same layer with the same weights, then we loaded the optimal full precision model,
        # therefore we have to train the weights for each QuantizedOp
        if (args.loadedOpsWithDiffWeights is False) and args.init_weights_train:
            self.epoch = self.initialWeightsTraining(model, args, trainFolderName='init_weights_train')
        else:
            rows = [['Switching stage']]
            # we loaded ops in the same layer with different weights, therefore we just have to switch_stage
            switchStageFlag = True
            while switchStageFlag:
                switchStageFlag = model.switch_stage([lambda msg: rows.append([msg])])
            # create info table
            logger.addInfoTable(self.initWeightsTrainTableTitle, rows)

        # init logger data table
        logger.createDataTable('Summary', self.colsMainLogger)

    @abstractmethod
    def train(self):
        raise NotImplementedError('subclasses must override train()!')

    @staticmethod
    def __getBitwidthKey(optModel_bitwidth):
        return '{}'.format(optModel_bitwidth)

    def setOptModelBitwidth(self):
        model = self.model
        args = self.args

        args.optModel_bitwidth = [layer.getBitwidth() for layer in model.layersList]
        # check if current bitwidth has already been sent for training
        bitwidthKey = self.__getBitwidthKey(args.optModel_bitwidth)

        return bitwidthKey

    # wait for sending all queued jobs
    def waitForQueuedJobs(self):
        while len(self.optModelTrainingQueue) > 0:
            self.logger.addInfoToDataTable('Waiting for queued jobs, queue size:[{}]'.format(len(self.optModelTrainingQueue)))
            self.trySendQueuedJobs()
            sleep(60)

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

        # if this allocation has been optimal enough batches, let's train it
        if self.optModelBitwidthCounter[bitwidthKey] == self.nBatchesOptModel:
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

            if retVal != 0:
                # server is full with jobs, add current job to queue
                self.optModelTrainingQueue.append((command, args.optModel_bitwidth))
                print('No available GPU, adding {} to queue, queue size:[{}]'
                      .format(bitwidthKey, len(self.optModelTrainingQueue)))
                # remove args JSON
                system('ssh yochaiz@132.68.39.32 rm {}'.format(dstPath))

        # try to send queued jobs, regardless current optimal model
        self.trySendQueuedJobs()

    def trainOptimalModel(self, nEpoch, nBatch):
        model = self.model
        # set optimal model bitwidth per layer
        optBopsRatio = model.evalMode()
        bitwidthKey = self.setOptModelBitwidth()
        # train optimal model
        self.sendOptModel(bitwidthKey, nEpoch, nBatch)

        return optBopsRatio

    def initialWeightsTraining(self, model, args, trainFolderName, filename=None):
        nEpochs = self.nEpochs
        logger = self.logger

        # create train folder
        folderPath = '{}/{}'.format(self.trainFolderPath, trainFolderName)
        if not path.exists(folderPath):
            makedirs(folderPath)

        # init optimizer
        optimizer = SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        # init scheduler
        scheduler = CosineAnnealingLR(optimizer, float(nEpochs), eta_min=args.learning_rate_min)

        # init validation best precision value
        best_prec1 = 0.0

        epoch = 0
        # init table in main logger
        logger.createDataTable(self.initWeightsTrainTableTitle, self.colsMainInitWeightsTrain)

        for epoch in range(1, nEpochs + 1):
            scheduler.step()
            lr = scheduler.get_lr()[0]

            trainLogger = HtmlLogger(folderPath, str(epoch))
            trainLogger.addInfoTable('Learning rates', [
                ['optimizer_lr', self.formats[self.lrKey].format(optimizer.param_groups[0]['lr'])],
                ['scheduler_lr', self.formats[self.lrKey].format(lr)]
            ])

            # set loggers dictionary
            loggersDict = dict(train=trainLogger)

            # training
            print('========== Epoch:[{}] =============='.format(epoch))
            trainData = self.trainWeights(model.chooseRandomPath, optimizer, epoch, loggersDict)

            # add epoch number
            trainData[self.epochNumKey] = epoch
            # add learning rate
            trainData[self.lrKey] = self.formats[self.lrKey].format(optimizer.param_groups[0]['lr'])

            # switch stage, i.e. freeze one more layer
            if (epoch in self.epochsSwitchStage) or (epoch == nEpochs):
                # validation
                valid_acc, validData = self.infer(epoch, loggersDict)

                # merge trainData with validData
                for k, v in validData.items():
                    trainData[k] = v

                # switch stage
                model.switch_stage(loggerFuncs=[lambda msg: trainLogger.addInfoTable(title='Switching stage', rows=[[msg]])])
                # update optimizer & scheduler due to update in learnable params
                optimizer = SGD(model.parameters(), scheduler.get_lr()[0], momentum=args.momentum, weight_decay=args.weight_decay)
                scheduler = CosineAnnealingLR(optimizer, float(nEpochs), eta_min=args.learning_rate_min)
                scheduler.step()

                # save model checkpoint
                is_best = valid_acc > best_prec1
                best_prec1 = max(valid_acc, best_prec1)
                save_checkpoint(self.trainFolderPath, model, args, epoch, best_prec1, is_best, filename)
            else:
                # save model checkpoint
                save_checkpoint(self.trainFolderPath, model, args, epoch, best_prec1, is_best=False, filename=filename)

            # add data to main logger table
            logger.addDataRow(trainData)
            # add columns row from time to time
            if epoch % 10 == 0:
                logger.addColumnsRowToDataTable()

        # add optimal accuracy
        logger.addSummaryDataRow({self.epochNumKey: 'Optimal', self.validAccKey: '{:.3f}'.format(best_prec1)})

        # save pre-trained checkpoint
        save_checkpoint(self.trainFolderPath, model, args, epoch, best_prec1, is_best=False, filename='pre_trained')

        args.best_prec1 = best_prec1

        return epoch

    def sendEmail(self, nEpoch, batchNum, nBatches):
        body = ['Hi', 'Files are attached.', 'Epoch:[{}]  Batch:[{}/{}]'.format(nEpoch, batchNum, nBatches)]
        content = ''
        for line in body:
            content += line + '\n'

        sendDataEmail(self.model, self.args, content)

    # apply defined formats on dict values by keys
    def __applyFormats(self, dict):
        for k in dict.keys():
            if k in self.formats:
                dict[k] = self.formats[k].format(dict[k])

    def trainAlphas(self, search_queue, model, architect, nEpoch, loggers):
        loss_container = AvgrageMeter()

        model.train()

        trainLogger = loggers.get('train')
        # init updateModelWeights() logger func
        loggerFunc = []
        if trainLogger:
            trainLogger.createDataTable('Epoch:[{}] - Alphas'.format(nEpoch), self.colsTrainAlphas)
            loggerFunc = [lambda msg: trainLogger.addInfoToDataTable(msg)]

        # update model replications weights
        architect.modelReplicator.updateModelWeights(model, loggerFuncs=loggerFunc)
        # quantize all ops
        architect.modelReplicator.quantize()

        nBatches = len(search_queue)

        # init logger functions
        def createInfoTable(dict, key, logger, rows):
            dict[key] = logger.createInfoTable('Show', rows)

        def alphasFunc(k, rows):
            createInfoTable(dataRow, self.alphasTableTitle, trainLogger, rows)

        def forwardCountersFunc(rows):
            createInfoTable(dataRow, self.forwardCountersTitle, trainLogger, rows)

        for step, (input, target) in enumerate(search_queue):
            startTime = time()
            n = input.size(0)
            dataRow = {}

            input = Variable(input, requires_grad=False).cuda()
            target = Variable(target, requires_grad=False).cuda(async=True)

            loss = architect.step(model, input, target)

            # train optimal model
            optBopsRatio = self.trainOptimalModel(nEpoch, step)
            # add alphas data to statistics
            model.stats.addBatchData(model, optBopsRatio, nEpoch, step)

            func = []
            if trainLogger:
                # log dominant QuantizedOp in each layer
                logDominantQuantizedOp(model, k=2, loggerFuncs=[alphasFunc])
                func = [forwardCountersFunc]
            # log forward counters. if loggerFuncs==[] then it is just resets counters
            logForwardCounters(model, loggerFuncs=func)
            # save alphas to csv
            model.save_alphas_to_csv(data=[nEpoch, step])
            # log allocations
            self.logAllocations()
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
                # collect missing keys
                dataRow[self.batchNumKey] = '{}/{}'.format(step, nBatches)
                dataRow[self.optBopsRatioKey] = optBopsRatio
                dataRow[self.timeKey] = endTime - startTime
                dataRow[self.archLossKey] = loss
                # apply formats
                self.__applyFormats(dataRow)
                # add row to data table
                trainLogger.addDataRow(dataRow)
                # add columns row
                if (step + 1) % 10 == 0:
                    trainLogger.addColumnsRowToDataTable()


        # restore quantization for all replications ops
        architect.modelReplicator.restore_quantize()

        # log accuracy, loss, etc.
        summaryData = {self.epochNumKey: nEpoch, self.lrKey: architect.lr, self.batchNumKey: 'Summary', self.archLossKey: loss_container.avg,
                       self.optBopsRatioKey: optBopsRatio}
        self.__applyFormats(summaryData)

        for _, logger in loggers.items():
            logger.addSummaryDataRow(summaryData)

        return summaryData

    def trainWeights(self, modelChoosePathFunc, optimizer, epoch, loggers):
        loss_container = AvgrageMeter()
        top1 = AvgrageMeter()
        top5 = AvgrageMeter()

        trainLogger = loggers.get('train')
        if trainLogger:
            trainLogger.createDataTable('Epoch:[{}] - Training weights'.format(epoch), self.colsTrainWeights)

        model = self.model
        crit = self.cross_entropy
        train_queue = self.train_queue
        grad_clip = self.args.grad_clip

        model.train()

        nBatches = len(train_queue)

        for step, (input, target) in enumerate(train_queue):
            startTime = time()
            n = input.size(0)

            input = Variable(input, requires_grad=False).cuda()
            target = Variable(target, requires_grad=False).cuda(async=True)

            # choose alpha per layer
            modelChoosePathFunc()
            bopsRatio = model.calcBopsRatio()
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
                dataRow = {
                    self.batchNumKey: '{}/{}'.format(step, nBatches), self.trainLossKey: loss, self.trainAccKey: prec1,
                    self.pathBopsRatioKey: bopsRatio, self.timeKey: (endTime - startTime)
                }
                # apply formats
                self.__applyFormats(dataRow)
                # add row to data table
                trainLogger.addDataRow(dataRow)
                # add columns row
                if (step + 1) % 10 == 0:
                    trainLogger.addColumnsRowToDataTable()

        # log accuracy, loss, etc.
        summaryData = {self.trainLossKey: loss_container.avg, self.trainAccKey: top1.avg, self.batchNumKey: 'Summary'}
        # apply formats
        self.__applyFormats(summaryData)

        for _, logger in loggers.items():
            logger.addSummaryDataRow(summaryData)

        # log dominant QuantizedOp in each layer
        if trainLogger:
            logDominantQuantizedOp(model, k=2,
                                   loggerFuncs=[lambda k, rows: trainLogger.addInfoTable(title=self.alphasTableTitle.format(k), rows=rows)])

        # log forward counters. if loggerFuncs==[] then it is just resets counters
        func = [lambda rows: trainLogger.addInfoTable(title=self.forwardCountersTitle, rows=rows)] if trainLogger else []
        logForwardCounters(model, loggerFuncs=func)

        return summaryData

    def infer(self, nEpoch, loggers):
        objs = AvgrageMeter()
        top1 = AvgrageMeter()
        top5 = AvgrageMeter()

        model = self.model
        modelInferMode = model.evalMode
        valid_queue = self.valid_queue
        crit = self.cross_entropy

        model.eval()
        bopsRatio = modelInferMode()
        # print eval layer index selection
        trainLogger = loggers.get('train')
        if trainLogger:
            trainLogger.createDataTable('Epoch:[{}] - Validation'.format(nEpoch), self.colsValidation)
            trainLogger.addInfoToDataTable('Layers optimal indices:{}'.format([layer.curr_alpha_idx for layer in model.layersList]))

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
                    dataRow = {
                        self.batchNumKey: '{}/{}'.format(step, nBatches), self.validLossKey: loss, self.validAccKey: prec1,
                        self.optBopsRatioKey: bopsRatio, self.timeKey: endTime - startTime
                    }
                    # apply formats
                    self.__applyFormats(dataRow)
                    # add row to data table
                    trainLogger.addDataRow(dataRow)
                    # add columns row
                    if (step + 1) % 10 == 0:
                        trainLogger.addColumnsRowToDataTable()

        # create summary row
        summaryRow = {self.batchNumKey: 'Summary', self.validLossKey: objs.avg, self.validAccKey: top1.avg, self.pathBopsRatioKey: bopsRatio}
        # apply formats
        self.__applyFormats(summaryRow)

        for _, logger in loggers.items():
            logger.addSummaryDataRow(summaryRow)

        # log forward counters. if loggerFuncs==[] then it is just resets counters
        func = []
        if trainLogger:
            colName = 'Values'
            trainLogger.createDataTable('Validation forward counters', [colName])
            func = [lambda rows: trainLogger.addDataRow({colName: trainLogger.createInfoTable('Show', rows)})]

        logForwardCounters(model, loggerFuncs=func)

        return top1.avg, summaryRow

    def logAllocations(self):
        logger = HtmlLogger(self.args.save, 'allocations', overwrite=True)
        allocationKey = 'Allocation'
        nBatchesKey = 'Number of batches'
        logger.createDataTable('Allocations', [allocationKey, nBatchesKey])

        for bitwidth, nBatches in self.optModelBitwidthCounter.items():
            logger.addDataRow({allocationKey: bitwidth, nBatchesKey: nBatches})

            # bitwidthList = bitwidth[1:-1].replace('),', ');')
            # bitwidthList = bitwidthList.split('; ')
            # bitwidthStr = ''
            # for layerIdx, b in enumerate(bitwidthList):
            #     bitwidthStr += 'Layer [{}]: {}\n'.format(layerIdx, b)
            #
            # logger.addDataRow({allocationKey: bitwidthStr, nBatchesKey: nBatches})
