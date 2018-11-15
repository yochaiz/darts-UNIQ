from cnn.statistics import Statistics
from torch import load
from os.path import dirname, isdir, exists
from os import listdir


def plotFromFile(plotPath):
    baseFolder = dirname(plotPath)  # script directory
    plotsData = load(plotPath)
    Statistics.plotBops(plotsData, 'bops', 'Baseline', baseFolder)


def plotFromFolder(folderPath, plotsDataPath):
    plotsData = load(plotsDataPath)
    for folderName in listdir(folderPath):
        fPath = '{}/{}'.format(folderPath, folderName)
        if isdir(fPath):
            checkpointPath = '{}/train/model_opt.pth.tar'.format(fPath)
            if exists(checkpointPath):
                checkpoint = load(checkpointPath)


plotPath = '/home/vista/Desktop/Architecture_Search/FF/plots.data'
# plotFromFile(plotPath)

folderPath = '/home/vista/Desktop/Architecture_Search/FF'
plotFromFolder(folderPath, plotPath)
