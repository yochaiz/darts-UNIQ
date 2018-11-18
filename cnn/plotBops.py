from torch import load, save
from os.path import dirname, isfile
from os import listdir

from cnn.statistics import Statistics
from cnn.trainRegime.regime import TrainRegime


def plotFromFile(plotPath):
    baseFolder = dirname(plotPath)  # script directory
    plotsData = load(plotPath)
    Statistics.plotBops(plotsData, 'bops', 'Baseline', baseFolder)


def plotFromFolder(folderPath, plotsDataPath, epoch=None):
    epochKey = 'epoch'
    bopsKey = 'bops'
    rowKeysToReplace = [TrainRegime.validAccKey, bopsKey]
    if epoch is None:
        rowKeysToReplace.append(epochKey)

    plotsData = load(plotsDataPath)
    bopsPlotData = plotsData[bopsKey]
    for folderName in listdir(folderPath):
        fPath = '{}/{}'.format(folderPath, folderName)
        if isfile(fPath):
            checkpoint = load(fPath)
            # init list of existing keys we found in checkpoint
            existingKeys = []
            # update keys if they exist
            for key in rowKeysToReplace:
                v = getattr(checkpoint, key, None)
                # if v is not None, then update value in tables
                if v is not None:
                    # update key exists
                    existingKeys.append(key)

            if len(existingKeys) == len(rowKeysToReplace):
                epoch = getattr(checkpoint, epochKey, epoch)
                # add key to bopsPlotData dictionary, if doesn't exist
                if epoch not in bopsPlotData:
                    bopsPlotData[epoch] = []

                # add point data
                title = epoch if isinstance(epoch, str) else None
                bopsPlotData[epoch].append((title, getattr(checkpoint, bopsKey), getattr(checkpoint, TrainRegime.validAccKey)))

    # save updated plots data
    save(plotsData, plotsDataPath)
    # plot
    Statistics.plotBops(plotsData, bopsKey, 'Baseline', folderPath)


plotPath = '/home/vista/Desktop/Architecture_Search/FF/plots.data'
# plotFromFile(plotPath)

folderPath = '/home/vista/Desktop/Architecture_Search/FF'
plotFromFolder(folderPath, plotPath)
