from os import makedirs, listdir, rename, getpid
from os.path import exists, isfile
from time import sleep
from json import loads

from cnn.HtmlLogger import SimpleLogger


class AWS_Manager:
    def __init__(self, args, jobsPathLocal, jobsDownloadedPathLocal, noMoreJobsFilename):
        self.args = args
        # set jobs local paths
        self.jobsPathLocal = jobsPathLocal
        self.jobsDownloadedPathLocal = jobsDownloadedPathLocal
        self.jobsUploadedPathLocal = '{}/Uploaded'.format(jobsPathLocal)
        # create missing folders
        for folder in [self.jobsPathLocal, self.jobsDownloadedPathLocal, self.jobsUploadedPathLocal]:
            if not exists(folder):
                makedirs(folder)
        # init folders path on remote (AWS)
        self.projectFolderPathRemote = 'F-BANNAS'
        self.jobsPathRemote = '{}'.format(args.time)
        self.jobsFinishedPathRemote = '{}/Finished'.format(self.jobsPathRemote)
        self.jobsDownloadedPathRemote = '{}/Downloaded'.format(self.jobsPathRemote)
        # noMoreJobsFilename is a file created on local, to tell remote it finished creating jobs
        self.noMoreJobsFilename = noMoreJobsFilename
        # init number of machine GPUs
        self.nGPUs = 8
        # init general window id
        self.generalWindowID = self.nGPUs + 1
        # init number of mins to wait
        self.nWaitingMins = 1

        # init jobs waiting list, elements are (local path, remote path)
        self.jobsWaitingList = []

        # we create aws instance only after we have enough jobs
        # we don't want to turn on the machine too early
        self.aws = None

        self.pid = getpid()
        # create logger
        self.logger = SimpleLogger(args.save, 'aws_manager')
        self.logger.setMaxTableCellLength(250)
        # log self variables
        self.logger.addInfoTable('Parameters', SimpleLogger.dictToRows(vars(self), nElementPerRow=3))

    def manage(self):
        logger = self.logger
        logger.addRow('Waiting until we have [{}] jobs'.format(self.nGPUs))

        # # wait until we have enough jobs
        # while self.__countFilesInLocalFolder(self.jobsPathLocal) < self.nGPUs:
        #     sleep(self.nWaitingMins * 60)

        self.aws = self.__createTask()
        self.__createProjectSourceCodeFolderRemote()
        self.__createRemoteJSONsFolder()
        self.__setEnvironment()

        # 1st loop waits until we finish uploading all jobs from local
        while self.__didUploadAllJobsFromLocal() is False:
            self.__uploadJobsFromLocal()
            self.__assignJobsToGPUs()
            self.__downloadJobsToLocal()
            # wait some time ...
            sleep(self.nWaitingMins * 60)

        # 2nd loop waits for all jobs on remote to finish, i.e. no more files in jobsPathRemote
        while self.__countFilesInRemoteFolder(self.jobsPathRemote, 'remoteJobsFolderFilesCounter.out') > 0:
            self.__assignJobsToGPUs()
            self.__downloadJobsToLocal()
            # wait some time ...
            sleep(self.nWaitingMins * 60)

        # 3rd loop waits for all finished jobs on remote to be downloaded to local
        while self.__countFilesInRemoteFolder(self.jobsFinishedPathRemote, 'remoteFinishedJobsFolderFilesCounter.out') > 0:
            self.__downloadJobsToLocal()
            # wait some time ...
            sleep(self.nWaitingMins * 60)

        self.logger.addSummaryRow('All jobs have been finished and downloaded successfully')
        # # turn off machine
        # task.run('sudo shutdown -h -P 1')  # shutdown the instance in 1 min

    def __countFilesInLocalFolder(self, localFolderPath):
        counter = 0
        for file in listdir(localFolderPath):
            filePath = '{}/{}'.format(localFolderPath, file)
            if isfile(filePath):
                counter += 1

        self.logger.addRow([['Folder', localFolderPath], ['Files#', counter]])
        return counter

    # checks if all jobs have finished on remote
    def __countFilesInRemoteFolder(self, remoteFolderPath, fName):
        aws = self.aws
        aws.switch_window(self.generalWindowID)
        # list files in remote folder
        aws.run('find {} -mindepth 1 -maxdepth 1 -type f | wc -l > {}'.format(remoteFolderPath, fName))
        localFname = self.__setFnamePath(fName)
        aws.download(fName, localFname)
        # read value in file
        with open(localFname, 'r') as f:
            values = f.read().splitlines()
            v = int(values[0])
            self.logger.addRow([['Remote folder', remoteFolderPath], ['Files#', v]])
            return v

    # download JSONs from remote finished to local downloaded
    # move JSONS from remote finished to remote downloaded folder
    def __downloadJobsToLocal(self):
        aws = self.aws
        aws.switch_window(self.generalWindowID)
        # list files in remote finished folder
        fName = 'remoteFinishedFolderFilesList.out'
        aws.run('ls {} > {}'.format(self.jobsFinishedPathRemote, fName))
        localFname = self.__setFnamePath(fName)
        aws.download(fName, localFname)
        with open(localFname, 'r') as f:
            filesList = f.read().splitlines()
            for file in filesList:
                # download to local downloaded folder
                filePath = '{}/{}'.format(self.jobsFinishedPathRemote, file)
                localPath = '{}/{}'.format(self.jobsDownloadedPathLocal, file)
                aws.download(filePath, localPath)
                # move it on remote server from finished folder to downloaded folder
                newRemoteDst = '{}/{}'.format(self.jobsDownloadedPathRemote, file)
                aws.run('mv {} {}'.format(filePath, newRemoteDst))
                self.logger.addRow([['File', filePath], ['Local destination', self.jobsDownloadedPathLocal], ['Remote destination', newRemoteDst]])

    # returns a list of available GPUs
    def __listAvailableGPUs(self):
        aws = self.aws
        fName = 'stat.out'
        aws.switch_window(self.generalWindowID)
        aws.run('gpustat --json > {}'.format(fName))
        localFname = self.__setFnamePath(fName)
        aws.download(fName, localFname)

        jsonData = open(localFname).read()

        data = loads(jsonData)
        gpus = data['gpus']
        availableGPUs = [gpuID for gpuID, gpuData in enumerate(gpus) if len(gpuData['processes']) == 0]
        return availableGPUs

    # assigns waiting jobs to available GPUs
    def __assignJobsToGPUs(self):
        aws = self.aws
        logger = self.logger
        # get list of available GPUs
        availableGPUs = self.__listAvailableGPUs()
        # assign jobs to availble GPUs
        while (len(availableGPUs) > 0) and (len(self.jobsWaitingList) > 0):
            logger.addRow([['Available GPUs', availableGPUs], ['jobsWaitingList', [[v] for v in self.jobsWaitingList]]])
            gpu = availableGPUs[0]
            aws.switch_window(gpu)
            # get job remote path
            jobRemotePath, jobName = self.jobsWaitingList[0]
            # build command line
            command = 'CUDA_VISIBLE_DEVICES={} PYTHONPATH={} python3 {}/cnn/train_opt2.py --json {} --dstFolder {} > {}.out' \
                .format(gpu, self.projectFolderPathRemote, self.projectFolderPathRemote, jobRemotePath, self.jobsFinishedPathRemote, jobName)
            # run command
            aws.run(command, non_blocking=True)
            logger.addSummaryRow([['Job name', jobName], ['Job remote path', jobRemotePath], ['Assigned GPU#', gpu], ['Command', command]])
            # update lists
            availableGPUs = availableGPUs[1:]
            self.jobsWaitingList = self.jobsWaitingList[1:]
            sleep(10)

    def __didUploadAllJobsFromLocal(self):
        jobsPathLocal = self.jobsPathLocal

        otherFilesExists = False
        noMoreJobsFileExists = False
        # look for file that is not self.noMoreJobsFilename
        # if there is one like that, then we haven't uploaded all jobs from local
        for file in listdir(jobsPathLocal):
            filePath = '{}/{}'.format(jobsPathLocal, file)
            if isfile(filePath):
                # if there are other files, then we didn't upload all jobs
                if file != self.noMoreJobsFilename:
                    otherFilesExists = True
                else:
                    noMoreJobsFileExists = True

        self.logger.addRow([['Folder', jobsPathLocal], ['Jobs exist', otherFilesExists],
                            ['[{}] exists'.format(self.noMoreJobsFilename), noMoreJobsFileExists]])
        return (otherFilesExists is False) and (noMoreJobsFileExists is True)

    # upload files from local and move them to uploaded folder on local
    def __uploadJobsFromLocal(self):
        aws = self.aws
        aws.switch_window(self.generalWindowID)
        jobsPathLocal = self.jobsPathLocal
        logger = self.logger

        for file in listdir(jobsPathLocal):
            filePath = '{}/{}'.format(jobsPathLocal, file)
            # make sure we don't upload noMoreJobsFilename
            if isfile(filePath) and file != self.noMoreJobsFilename:
                # upload to aws
                aws.upload(filePath, self.jobsPathRemote)
                logger.addRow([['Local job', filePath], ['Uploaded to', self.jobsPathRemote]])
                # move folder on local machine from main to Uploaded
                newDst = '{}/{}'.format(self.jobsUploadedPathLocal, file)
                rename(filePath, newDst)
                logger.addRow([['Local job', filePath], ['Moved to', newDst]])
                # add job to waiting list
                jobRemotePath = '{}/{}'.format(self.jobsPathRemote, file)
                self.jobsWaitingList.append((jobRemotePath, file))
                logger.addRow([['jobsWaitingList#', len(self.jobsWaitingList)]])

    def __createRemoteJSONsFolder(self):
        aws = self.aws
        aws.switch_window(self.generalWindowID)
        # create folders
        for folderPath in [self.jobsPathRemote, self.jobsFinishedPathRemote, self.jobsDownloadedPathRemote]:
            self.logger.addRow([['Folder', folderPath], ['Status', 'Created']])
            aws.run('mkdir {}'.format(folderPath))

    # create project source code folder
    def __createProjectSourceCodeFolderRemote(self):
        aws = self.aws
        aws.switch_window(self.generalWindowID)
        args = self.args
        projectFolderPathRemote = self.projectFolderPathRemote
        # remove project folder if exists for some reason ...
        aws.run('rm -rf {}'.format(projectFolderPathRemote))
        # create project folder
        aws.run('mkdir {}'.format(projectFolderPathRemote))
        # upload code zip file
        aws.upload(args.codePath, projectFolderPathRemote)
        # unzip code to folder
        aws.run('unzip {}/{} -d {}'.format(projectFolderPathRemote, args.codeFilename, projectFolderPathRemote))
        # upload pre-trained models
        # ????????????????????????
        # unzip pre-trained models
        # ?????????????????????????
        self.logger.addRow('Creating project source code folder on remote server')
        self.logger.addRow('Uploading pre-trained model to remote server')

    # install environment
    def __setEnvironment(self):
        aws = self.aws
        aws.switch_window(self.generalWindowID)
        self.logger.addRow('Setting environment')
        # install gpustat
        aws.run('sudo pip3 install gpustat')
        # activate pytorch virtual environment
        for gpu in range(self.nGPUs):
            aws.switch_window(gpu)
            aws.run('source activate pytorch_p36')
        # # downgrade to pytorch 0.4.0
        # aws.run('conda install --yes pytorch=0.4.0')

    # create task, i.e. start a machine
    def __createTask(self):
        # moved import here, otherwise it asks to install ncluster on AWS machine
        from ncluster import use_aws, make_task
        use_aws()
        taskName = self.args.time
        self.logger.addRow([['Task name', taskName]])
        aws = make_task(instance_type='p3.16xlarge', name=taskName, image_name='Deep Learning AMI (Ubuntu) Version 16.0')

        return aws

    def __setFnamePath(self, fName):
        return '{}/../{}'.format(self.jobsPathLocal, fName)
