import sys
from os import makedirs, rename
from os.path import exists
from multiprocessing import Process
from time import sleep

from torch.optim import SGD
from torch import save as saveCheckpoint
from torch import load as loadCheckpoint

from .regime import TrainRegime, save_checkpoint
from cnn.architect import Architect
from cnn.HtmlLogger import HtmlLogger
from cnn.managers.aws import AWS_Manager
import cnn.gradEstimators as gradEstimators


def manageJobs(args, jobsPathLocal, jobsDownloadedPathLocal, noMoreJobsFilename):
    # redirect stdout, stderr
    sys.stdout = open('{}/AWS.out'.format(args.save), 'w')
    sys.stderr = open('{}/AWS.err'.format(args.save), 'w')
    manager = AWS_Manager(args, jobsPathLocal, jobsDownloadedPathLocal, noMoreJobsFilename)
    manager.manage()


class TrainingJob:
    def __init__(self, dict):
        for k, v in dict.items():
            setattr(self, k, v)

    def generateTempValue(self, key):
        return '{}_{}'.format(self.jsonFileName, key)


class AlphasWeightsLoop(TrainRegime):
    jobNameKey = 'Job name'
    jobUpdateTimeKey = 'Update time'
    lastCheckKey = 'Last check'
    jobsLoggerColumns = [jobNameKey, jobUpdateTimeKey]

    def __init__(self, args, logger):
        super(AlphasWeightsLoop, self).__init__(args, logger)

        # set number of different partitions we want to draw from alphas multinomial distribution in order to estimate their validation accuracy
        self.nValidPartitions = 5

        # create folder for jobs JSONs
        self.jobsPath = '{}/jobs'.format(args.save)
        self.jobsDownloadedPath = '{}/Downloaded'.format(self.jobsPath)
        self.jobsUpdatedPath = '{}/Updated'.format(self.jobsPath)
        # when we are done creating jobs, create this file and upload it to remote server to let it know there are no more jobs
        self.noMoreJobsFilename = 'DONE'

        # create path folders
        for p in [self.jobsPath, self.jobsDownloadedPath, self.jobsUpdatedPath]:
            if not exists(p):
                makedirs(p)

        # init dictionary of list of training jobs we yet have to get their values
        # each key is epoch number
        self.jobsList = {}
        # init data table row keys to replace
        self.rowKeysToReplace = [self.validLossKey, self.validAccKey]
        # init validation best precision value from all training jobs
        self.best_prec1 = 0.0

        # init jobs logger
        self.jobsLogger = HtmlLogger(args.save, 'jobsLogger')
        self.jobsLogger.addInfoTable(self.lastCheckKey, [[self.jobsLogger.getTimeStr()]])
        self.jobsLogger.createDataTable('Jobs', self.jobsLoggerColumns + self.rowKeysToReplace)

        # # create process to manage epoch jobs
        # job_process = Process(target=manageJobs, args=(args, self.jobsPath, self.jobsDownloadedPath, self.noMoreJobsFilename,))
        # job_process.start()
        # ========================== DEBUG ===============================
        # return
        # ================================================================

        # init model replicator
        replicatorClass = gradEstimators.__dict__[args.grad_estimator]
        replicator = replicatorClass(self.model, self.modelClass, args, logger)
        # init architect
        self.architect = Architect(replicator, args)

    # run on validation set and add validation data to main data row
    def __inferWithData(self, setModelPathFunc, epoch, nEpochs, loggersDict, dataRow):
        # run on validation set
        _, _, validData = self.infer(setModelPathFunc, epoch, loggersDict)

        # update epoch
        dataRow[self.epochNumKey] = '{}/{}'.format(epoch, nEpochs)
        # merge dataRow with validData
        for k, v in validData.items():
            dataRow[k] = v

    # creates job of training model partition
    def __createTrainingJob(self, setModelPartitionFunc, nEpoch, id):
        args = self.args
        model = self.model
        # set model partition
        setModelPartitionFunc()
        # set JSON file name
        jsonFileName = '{}-{}-{}.json'.format(args.time, nEpoch, id)
        # create training job instance
        trainingJob = TrainingJob(dict(bopsRatio=model.calcBopsRatio(), bops=model.countBops(), epochID=nEpoch, ID=id,
                                       bitwidthInfoTable=self.createBitwidthsTable(model, self.logger, self.bitwidthKey),
                                       jsonFileName=jsonFileName, jsonPath='{}/{}'.format(self.jobsPath, jsonFileName)))

        # save model layers partition
        args.partition = model.getCurrentFiltersPartition()
        # save model bops
        args.bops = trainingJob.bops
        # save job epoch
        args.epoch = nEpoch
        # save job ID
        args.jobID = id
        # reset accuracy & loss values, in case they are set in args
        for key in self.rowKeysToReplace:
            setattr(args, key, None)
        # save args to checkpoint
        saveCheckpoint(args, trainingJob.jsonPath)
        # reset args.partition
        args.partition = None
        # delete args.bops
        del args.bops
        del args.epoch
        del args.jobID

        # add job to jobsLogger ##############
        jobsLogger = self.jobsLogger
        # create job data row
        dataRow = {k: trainingJob.generateTempValue(k) for k in jobsLogger.dataTableCols}
        dataRow[self.jobNameKey] = trainingJob.jsonFileName
        jobsLogger.addDataRow(dataRow)

        return trainingJob

    # create all training jobs for a single epoch
    def __createEpochJobs(self, epoch):
        model = self.model
        # train from scratch (from 32-bit pre-trained) model partitions based on alphas distribution
        epochJobs = []

        # single training on setFiltersByAlphas
        epochJobs.append(self.__createTrainingJob(model.setFiltersByAlphas, epoch, 0))
        # nValidPartitions trainings on choosePathByAlphas
        for id in range(1, self.nValidPartitions + 1):
            epochJobs.append(self.__createTrainingJob(model.choosePathByAlphas, epoch, id))

        return epochJobs

    @staticmethod
    def __getEpochRange(nEpochs):
        return range(1, nEpochs + 1)

    def train(self):
        model = self.model
        modelParallel = self.modelParallel
        args = self.args
        logger = self.logger
        # init number of epochs
        nEpochs = 40
        epochRange = self.__getEpochRange(nEpochs)

        # # ========================== DEBUG ===============================
        # # create epoch jobs
        # for epoch in range(1, 4):
        #     epochJobsList = self.__createEpochJobs(epoch)
        #     self.jobsList[epoch] = epochJobsList
        #     # add data rows for epoch JSONs
        #     self.__addEpochJSONsDataRows(epochJobsList, epoch, nEpochs)
        #     self.__updateDataTableAndBopsPlot()
        #
        # open('{}/{}'.format(self.jobsPath, self.noMoreJobsFilename), 'w+')
        # # wait until all jobs have finished
        # while self.isDictEmpty(self.jobsList) is False:
        #     self.__updateDataTableAndBopsPlot()
        #     # wait 10 mins
        #     sleep(1 * 60)
        # return
        # # ================================================================

        epoch = 'Init'
        # create initial alphas distribution job
        epochJobsList = [self.__createTrainingJob(model.setFiltersByAlphas, epoch, 0)]
        self.jobsList[epoch] = epochJobsList
        # add data rows for epoch JSONs
        self.__addEpochJSONsDataRows(epochJobsList, epoch, nEpochs)

        for epoch in epochRange:
            print('========== Epoch:[{}] =============='.format(epoch))
            # calc alpha trainset loss on baselines
            self.calcAlphaTrainsetLossOnBaselines(self.trainFolderPath, '{}_{}'.format(epoch, self.archLossKey), logger)

            # init epoch train logger
            trainLogger = HtmlLogger(self.trainFolderPath, str(epoch))
            # set loggers dictionary
            loggersDict = dict(train=trainLogger)

            # train alphas
            dataRow = self.trainAlphas(self.search_queue[epoch % args.alphas_data_parts], self.architect, epoch, loggersDict)

            # create epoch jobs
            epochJobsList = self.__createEpochJobs(epoch)
            # add current epoch JSONs with the rest of JSONs
            self.jobsList[epoch] = epochJobsList

            # validation on fixed partition by alphas values
            self.__inferWithData(model.setFiltersByAlphas, epoch, nEpochs, loggersDict, dataRow)
            # add data to main logger table
            logger.addDataRow(dataRow)

            # train weights ###########
            # create epoch train weights folder
            epochName = '{}_w'.format(epoch)
            epochFolderPath = '{}/{}'.format(self.trainFolderPath, epochName)
            # turn off alphas
            model.turnOffAlphas()
            # turn on weights gradients
            model.turnOnWeights()
            # init optimizer
            optimizer = SGD(modelParallel.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
            # train weights with 1 epoch per stage
            wEpoch = 1
            switchStageFlag = True
            while switchStageFlag:
                # init epoch train logger
                trainLogger = HtmlLogger(epochFolderPath, '{}_{}'.format(epochName, wEpoch))
                # train stage weights
                self.trainWeights(optimizer, wEpoch, dict(train=trainLogger))
                # switch stage
                switchStageFlag = model.switch_stage([lambda msg: trainLogger.addInfoToDataTable(msg)])
                # update epoch number
                wEpoch += 1

            # init epoch train logger for last epoch
            trainLogger = HtmlLogger(epochFolderPath, '{}_{}'.format(epochName, wEpoch))
            # set loggers dictionary
            loggersDict = dict(train=trainLogger)
            # last weights training epoch we want to log also to main logger
            dataRow = self.trainWeights(optimizer, wEpoch, loggersDict)

            # validation on fixed partition by alphas values
            self.__inferWithData(model.setFiltersByAlphas, epoch, nEpochs, loggersDict, dataRow)
            # add data to main logger table
            logger.addDataRow(dataRow)

            # add data rows for epoch JSONs
            self.__addEpochJSONsDataRows(epochJobsList, epoch, nEpochs)

            # update temp values in data table + update bops plot
            self.__updateDataTableAndBopsPlot()

            # save checkpoint
            save_checkpoint(self.trainFolderPath, model, args, epoch, self.best_prec1)

        logger.addInfoToDataTable('Finished training, waiting for jobs to finish')
        # create file to let remote server know we are done creating jobs
        open('{}/{}'.format(self.jobsPath, self.noMoreJobsFilename), 'w+')
        # wait until all jobs have finished
        while self.isDictEmpty(self.jobsList) is False:
            self.__updateDataTableAndBopsPlot()
            # wait 10 mins
            sleep(10 * 60)

        # save checkpoint
        save_checkpoint(self.trainFolderPath, model, args, epoch, self.best_prec1)
        # send final email
        # self.sendEmail('Final', 0, 0)

    @staticmethod
    def isDictEmpty(dict):
        for k, v in dict.items():
            if len(v) > 0:
                return False

        return True

    def __updateDataTableAndBopsPlot(self):
        jobsLogger = self.jobsLogger
        # update time in last update time
        jobsLogger.addInfoTable(self.lastCheckKey, [[self.jobsLogger.getTimeStr()]])
        # init plot data list
        bopsPlotData = {}
        # init updated jobs dictionary, jobs we haven't got their values yet
        updatedJobsList = {}
        # init epochs as keys in bopsPlotData, empty list per key
        for epoch in self.jobsList.keys():
            updatedJobsList[epoch] = []

        # copy files back from server and check if best_prec1, best_valid_loss exists
        for epoch, epochJobsList in self.jobsList.items():
            for job in epochJobsList:
                # init job path in downloaded folder
                jobDownloadedPath = '{}/{}'.format(self.jobsDownloadedPath, job.jsonFileName)
                jobExists = exists(jobDownloadedPath)
                # load checkpoint
                if jobExists:
                    # load checkpoint
                    checkpoint = loadCheckpoint(jobDownloadedPath, map_location=lambda storage, loc: storage.cuda())
                    # init list of existing keys we found in checkpoint
                    existingKeys = []
                    # update keys if they exist
                    for key in self.rowKeysToReplace:
                        v = getattr(checkpoint, key, None)
                        # if v is not None, then update value in tables
                        if v is not None:
                            # update key exists
                            existingKeys.append(key)
                            # load key temp value
                            keyTempValue = job.generateTempValue(key)
                            # replace value in main table & jobsLogger
                            for l in [self.logger, jobsLogger]:
                                l.replaceValueInDataTable(keyTempValue, self.formats[key].format(v))
                            # add tuple of (bitwidth, bops, accuracy) to plotData if we have accuracy value
                            if key == self.validAccKey:
                                # add key to bopsPlotData dictionary, if doesn't exist
                                if epoch not in bopsPlotData:
                                    bopsPlotData[epoch] = []
                                # add point data
                                title = epoch if isinstance(epoch, str) else None
                                bopsPlotData[epoch].append((title, job.bops, v))
                                # update best_prec1 of all training jobs we have trained
                                self.best_prec1 = max(self.best_prec1, v)

                    # add job to updatedJobsList if we haven't got all keys, otherwise we are done with this job
                    if len(existingKeys) < len(self.rowKeysToReplace):
                        updatedJobsList[epoch].append(job)
                    else:
                        # if we are done with the job, move the JSON from downloaded to updated
                        jobUpdatedDst = '{}/{}'.format(self.jobsUpdatedPath, job.jsonFileName)
                        rename(jobDownloadedPath, jobUpdatedDst)
                        jobsLogger.replaceValueInDataTable(job.generateTempValue(self.jobUpdateTimeKey), jobsLogger.getTimeStr())
                else:
                    # add job to updatedJobsList if it hasn't been downloaded yet from remote server
                    updatedJobsList[epoch].append(job)

        # update self.jobsList to updatedJobsList
        self.jobsList = updatedJobsList

        # send new bops plot values to plot
        self.model.stats.addBopsData(bopsPlotData)

    def __addEpochJSONsDataRows(self, epochJobsList, epoch, nEpochs):
        logger = self.logger

        for job in epochJobsList:
            # set data row unique values for json, in order to overwrite them in the future when we will have values
            dataRow = {self.epochNumKey: '{}/{}'.format(epoch, nEpochs), self.validBopsRatioKey: job.bopsRatio,
                       self.bitwidthKey: job.bitwidthInfoTable}
            # apply formats
            self._applyFormats(dataRow)
            for key in self.rowKeysToReplace:
                dataRow[key] = job.generateTempValue(key)
            # add data row
            logger.addDataRow(dataRow, trType='<tr bgcolor="#2CBDD6">')

# ====================  gaon server code ==================================================
# # copy JSON from server back here
# retVal = system(job.cmdCopyFromServer)
# self.jobsLogger.addRow([['Action', 'Copy from server'], ['Epoch', epoch], ['File', job.jsonFileName],
#                         ['Status', returnSuccessOrFail(retVal)]])

# # create dir on remove server
# self.remoteDirPath = '/home/yochaiz/F-BANNAS/cnn/trained_models/{}/{}/{}'.format(args.model, args.dataset, args.folderName)
# command = 'ssh yochaiz@132.68.39.32 mkdir "{}"'.format(escape(self.remoteDirPath))
# system(command)

# def escape(txt):
#     translation = str.maketrans({'[': '\[', ']': '\]', '(': '\(', ')': '\)', ',': '\,', ' ': '\ '})
#     return txt.translate(translation)

# def returnSuccessOrFail(retVal, msg):
#     res = 'Success' if retVal == 0 else 'Fail'
#     return msg.format(res)

# def returnSuccessOrFail(retVal):
#     return 'Success' if retVal == 0 else 'Fail'
#
# def __buildCommand(jobTitle, nGPUs, nCPUs, server, data):
#     # --mail-user=yochaiz@cs.technion.ac.il --mail-type=ALL
#     return 'ssh yochaiz@132.68.39.32 srun -o {}.out -I10 --gres=gpu:{} -c {} -w {} -t 01-00:00:00 -p gip,all ' \
#            '-J "{}" ' \
#            '/home/yochaiz/F-BANNAS/cnn/sbatch_opt.sh --data "{}"'.format(jobTitle, nGPUs, nCPUs, server, jobTitle, data)
#
# def manageJobs(epochJobs, epoch, folderPath):
#     pid = getpid()
#     # create logger for manager
#     logger = SimpleLogger(folderPath, '[{}]-manager'.format(epoch))
#     logger.addInfoTable('Details', [['Epoch', epoch], ['nJobs', len(epochJobs)], ['Folder', folderPath], ['PID', pid]])
#     logger.setMaxTableCellLength(250)
#
#     # copy jobs JSON to server
#     for job in epochJobs:
#         # perform command
#         retVal = system(job.cmdCopyToServer)
#         logger.addRow([['Action', 'Copy to server'], ['File', job.jsonFileName], ['Status', returnSuccessOrFail(retVal)]])
#
#     # init server names
#     servers = ['gaon6', 'gaon4', 'gaon2', 'gaon5']
#     # init number of maximal GPUs we can run in single sbatch
#     nMaxGPUs = 2
#     # # init number of maximal CPUs
#     nMaxCPUs = 4
#     # init number of minutes to wait
#     nMinsWaiting = 10
#     # try sending as much jobs as possible under single sbatch
#     # try sending as much sbatch commands as possible
#     while len(epochJobs) > 0:
#         logger.addRow([['Jobs Waiting', len(epochJobs)]])
#         # set number of jobs we want to run in a single SBATCH command
#         nJobs = min(nMaxGPUs, len(epochJobs))
#         # try to send sbatch command to server, stop when successful
#         retVal = -1
#         while (nJobs > 0) and (retVal != 0):
#             # concatenate JSON files for jobs
#             files = ''
#             for job in epochJobs[:nJobs]:
#                 files += escape(job.dstPath) + '?'
#             # remove last comma
#             files = files[:-1]
#             # set number of CPUs
#             nCPUs = min(nMaxCPUs, 3 * nJobs)
#             # try to perform command on one of the servers
#             for serv in servers:
#                 # create command
#                 jobTitle = 'PID_[{}]_Epoch_[{}]_nJobs_[{}]_jobsLeft_[{}]'.format(pid, epoch, nJobs, len(epochJobs) - nJobs)
#                 trainCommand = __buildCommand(jobTitle, nJobs, nCPUs, serv, files)
#                 # send command to server, we added the -I flag, so if it won't be able to run immediately, it fails, no more pending
#                 retVal = system(trainCommand)
#                 # add data row with information
#                 dataRow = [['#Trainings', nJobs], ['Server', serv], ['#Waiting', len(epochJobs)], ['retVal', retVal], ['Command', trainCommand]]
#                 logger.addRow(dataRow)
#                 # clear successfully sent jobs
#                 if retVal == 0:
#                     epochJobs = epochJobs[nJobs:]
#                     logger.addSummaryRow([['#Trainings', nJobs], ['Server', serv], ['#Waiting', len(epochJobs)], ['Status', 'Success']])
#                     break
#
#             # check if jobs not sent, try sending less jobs, i.e. use less GPUs
#             # we don't really need to check retVal here, but anyway ...
#             if retVal != 0:
#                 nJobs -= 1
#             # wait a bit ... mostly for debugging purposes
#             sleep(10)
#
#         # if didn't manage to send any job, wait 10 mins
#         if retVal != 0:
#             logger.addRow([['Send status', 'Failed'], ['Waiting time (mins)', nMinsWaiting]])
#             sleep(nMinsWaiting * 60)
#
#     logger.addSummaryRow('Sent all jobs successfully')
#     logger.addSummaryRow('Done !')

# class TrainingJob:
#     def __init__(self, dict):
#         for k, v in dict.items():
#             setattr(self, k, v)
#
#         # set dstPath
#         self.dstPath = '{}/{}'.format(self.remoteDirPath, self.jsonFileName)
#         # init copy to server command
#         self.cmdCopyToServer = 'scp {} "yochaiz@132.68.39.32:{}"'.format(escape(self.jsonPath), escape(self.dstPath))
#         # init copy from server command
#         self.cmdCopyFromServer = 'scp "yochaiz@132.68.39.32:{}" "{}"'.format(escape(self.dstPath), self.jsonPath)

# =========================================================================================================

# def train(self):
#     # init logger data table
#     self.logger.createDataTable('Summary', self.colsMainLogger)
#     # init number of epochs
#     nEpochs = self.model.nLayers()
#
#     for epoch in range(1, nEpochs + 1):
#         print('========== Epoch:[{}] =============='.format(epoch))
#         # create epoch jobs
#         epochJobsList = self.__createEpochJobs(epoch)
#         # merge current epoch JSONs with the rest of JSONs
#         self.jobsList.extend(epochJobsList)
#
#         # add data rows for epoch JSONs
#         self.__addEpochJSONsDataRows(epochJobsList, epoch)
#
#         # update temp values in data table + update bops plot
#         self.__updateDataTableAndBopsPlot()

# =========================== very old code ====================================================================================
# def manageJobs(epochJobs, epoch):
#     # copy jobs JSON to server
#     for job in epochJobs:
#         # perform command
#         retVal = system(job.cmdCopyToServer)
#         printSuccessOrFail(retVal, 'Copied +{}+ to server'.format(job.jsonFileName) + ':[{}]')
#     # init server names
#     servers = ['gaon6', 'gaon4', 'gaon2', 'gaon5']
#     # init maximal SBATCH jobs we can run concurrently
#     nMaxSBATCH = 4
#     # init number of maximal GPUs we can run in single sbatch
#     nMaxGPUs = 2
#     # init number of maximal CPUs
#     nMaxCPUs = 8
#     # init number of minutes to wait
#     nMinsWaiting = 10
#     # try sending as much jobs as possible under single sbatch
#     # try sending as much sbatch commands as possible
#     while len(epochJobs) > 0:
#         print('Epoch:[{}] Jobs waiting:[{}]'.format(epoch, len(epochJobs)))
#         # set number of jobs we want to run in a single SBATCH command
#         nJobs = min(nMaxGPUs, len(epochJobs))
#         # try to send sbatch command to server, stop when successful
#         retVal = -1
#         while (nJobs > 0) and (retVal != 0):
#             # concatenate JSON files for jobs
#             files = ''
#             for job in epochJobs[:nJobs]:
#                 files += escape(job.dstPath) + '?'
#             # remove last comma
#             files = files[:-1]
#             # set number of CPUs
#             nCPUs = min(nMaxCPUs, 3 * nJobs)
#             # count how many running SBATCH jobs we have on server
#             nRunningJobs = system('ssh yochaiz@132.68.39.32 ./countRunning.sh') >> 8
#             nPendingJobs = system('ssh yochaiz@132.68.39.32 ./countPending.sh') >> 8
#             print('Epoch:[{}] - nRunningJobs:[{}] nPendingJobs:[{}]'.format(epoch, nRunningJobs, nPendingJobs))
#             # if there is room left for running a job, try to run it
#             if nRunningJobs < nMaxSBATCH:
#                 # change servers order if there are pending jobs
#                 if nPendingJobs > 0:
#                     s = servers[0]
#                     servers = servers[1:]
#                     servers.append(s)
#                 # try to perform command on one of the servers
#                 for serv in servers:
#                     print('Epoch:[{}] - trying to send [{}] trainings to [{}], jobs still waiting:[{}]'.format(epoch, nJobs, serv, len(epochJobs)))
#                     # create command
#              trainCommand = 'ssh yochaiz@132.68.39.32 sbatch -I --gres=gpu:{} -c {} -w {} /home/yochaiz/F-BANNAS/cnn/sbatch_opt.sh --data "{}"' \
#                         .format(nJobs, nCPUs, serv, files)
#                     retVal = system(trainCommand)
#                     # clear successfully sent jobs
#                     if retVal == 0:
#                         epochJobs = epochJobs[nJobs:]
#                         print('Epoch:[{}] - sent [{}] trainings successfully to [{}], jobs still waiting:[{}]'
#                               .format(epoch, nJobs, serv, len(epochJobs)))
#                         break
#
#             # check if jobs not sent, try sending less jobs, i.e. use less GPUs
#             # we don't really need to check retVal here, but anyway ...
#             if retVal != 0:
#                 nJobs -= 1
#
#             sleep(30)
#
#         # if didn't manage to send any job, wait 10 mins
#         if retVal != 0:
#             print('Epoch:[{}] - did not manage to send trainings,  waiting [{}] mins'.format(epoch, nMinsWaiting))
#             sleep(nMinsWaiting * 60)
#
#     print('Epoch:[{}] - sent all jobs successfully'.format(epoch))
#     print('Epoch:[{}] - Done !'.format(epoch))

# # create all training jobs for a single epoch
# def __createEpochJobs(self, epoch):
#     model = self.model
#     # train from scratch (from 32-bit pre-trained) model partitions based on alphas distribution
#     epochJobs = []
#     # init json files names list
#     jsonFilesList = []
#     # single training on setFiltersByAlphas
#     epochJobs.append(self.__createTrainingJob(model.setFiltersByAlphas, epoch, 0))
#     # epochJobsData[jsonFileName] = [bopsRatio, bitwidthInfoTable]
#     # jsonFilesList.append([jsonFileName, jsonPath])
#     # nValidPartitions trainings on setFiltersByAlphas
#     for id in range(1, self.nValidPartitions + 1):
#         epochJobs.append(self.__createTrainingJob(model.choosePathByAlphas, epoch, id))
#         # jsonFileName, jsonPath, bopsRatio, bitwidthInfoTable = self.__createTrainingJob(model.choosePathByAlphas, epoch, id)
#         # epochJobsData[jsonFileName] = [bopsRatio, bitwidthInfoTable]
#         # jsonFilesList.append([jsonFileName, jsonPath])
#
#     # init JSON path on server
#     dstPathList = []
#     # init list of commands to copy JSON from server back to here, in order to read its values back to data table
#     copyJSONcmdList = []
#     # copy JSON files to server
#     for jsonFileName, jsonPath in jsonFilesList:
#         # init args checkpoint destination path on server
#         dstPath = '{}/{}'.format(self.remoteDirPath, jsonFileName)
#         dstPathList.append(dstPath)
#         # init copy command
#         cmd = 'scp {} "yochaiz@132.68.39.32:{}"'.format(escape(jsonPath), escape(dstPath))
#         # init reverse copy command, from server back here
#         reverseCommand = 'scp "yochaiz@132.68.39.32:{}" "{}"'.format(escape(dstPath), jsonPath)
#         copyJSONcmdList.append(reverseCommand)
#         # perform command
#         retVal = system(cmd)
#         printSuccessOrFail(retVal, 'Copied +{}+ to server'.format(jsonFileName) + ':[{}]')
#
#     # add reverse copy JSON commands to jsonFilesList
#     for i, cmd in enumerate(copyJSONcmdList):
#         jsonFilesList[i].append(cmd)
#
#     # create process to manage epoch jobs
#     pool = Pool(processes=1)
#     pool.apply_async(manageJobs, args=(epochJobs, epoch,))
#
#     return epochJobsData, jsonFilesList
