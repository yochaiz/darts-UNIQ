from torch import load, save
from os.path import dirname, isfile
from os import listdir, remove
from ast import literal_eval

from cnn.statistics import Statistics
from cnn.trainRegime.regime import TrainRegime

bopsKey = 'bops'
baselineKey = 'Baseline'


def plotFromFile(plotPath):
    baseFolder = dirname(plotPath)  # script directory
    plotsData = load(plotPath)
    Statistics.plotBops(plotsData, bopsKey, baselineKey, baseFolder)


def plotEpochsFromFolder(folderPath, plotsDataPath, epoch=None):
    epochKey = 'epoch'
    rowKeysToReplace = [TrainRegime.validAccKey, bopsKey]
    if epoch is None:
        rowKeysToReplace.append(epochKey)

    plotsData = load(plotsDataPath)
    bopsPlotData = plotsData[bopsKey]
    for folderName in sorted(listdir(folderPath)):
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
    Statistics.plotBops(plotsData, bopsKey, baselineKey, folderPath)


def plotPartitionsFromFolder(folderPath, plotsDataPath):
    rowKeysToReplace = [TrainRegime.validAccKey, bopsKey]

    counterDict = {}
    counter = 0

    plotsData = load(plotsDataPath)
    bopsPlotData = plotsData[bopsKey]
    for folderName in sorted(listdir(folderPath)):
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
                epoch = str(checkpoint.partition[0].tolist())
                # add key to bopsPlotData dictionary, if doesn't exist
                if epoch not in counterDict:
                    counterDict[epoch] = counter
                    counter += 1
                    epochKey = '{}-[{}]'.format(epoch, counterDict[epoch])
                    bopsPlotData[epochKey] = []

                # add point data
                title = '[{}]'.format(counterDict[epoch]) if isinstance(epoch, str) else None
                epochKey = '{}-[{}]'.format(epoch, counterDict[epoch])
                bopsPlotData[epochKey].append((title, getattr(checkpoint, bopsKey), getattr(checkpoint, TrainRegime.validAccKey)))

    # save updated plots data
    save(plotsData, plotsDataPath)
    # plot
    Statistics.plotBops(plotsData, bopsKey, baselineKey, folderPath)


def generateCSV(folderPath):
    data = {}
    partitionKeys = []
    resultsKey = 'results'
    for file in sorted(listdir(folderPath)):
        fPath = '{}/{}'.format(folderPath, file)
        if isfile(fPath):
            checkpoint = load(fPath)
            try:
                partition = getattr(checkpoint, 'partition')
                if not isinstance(partition[0], list):
                    partition = partition[0].tolist()
                bops = getattr(checkpoint, bopsKey)
                repeatNum = int(file[file.rfind('-') + 1:file.rfind('.')])
                validAcc = getattr(checkpoint, TrainRegime.validAccKey)
            except Exception as e:
                remove(fPath)
                continue

            partitionStr = str(partition)
            if partitionStr not in data:
                data[partitionStr] = {bopsKey: bops, resultsKey: [None] * 5}
                partitionKeys.append(partition)

            results = data[partitionStr][resultsKey]
            results[repeatNum - 1] = validAcc

    for partition in reversed(sorted(partitionKeys)):
        partitionStr = str(partition)
        partitionData = data[partitionStr]

        bops = partitionData[bopsKey]
        results = partitionData[resultsKey]
        resultsStr = ''
        for r in results:
            resultsStr += ',{:.3f}'.format(r) if r else ','
        print('"{}",{}{}'.format(partition, bops, resultsStr))

    print('Partition,Bops,1,2,3,4,5')


plotPath = '/home/vista/Desktop/Architecture_Search/FF/plots.data'
# plotFromFile(plotPath)

folderPath = '/home/vista/Desktop/Architecture_Search/FF-2'
# plotPartitionsFromFolder(folderPath, plotPath)
generateCSV(folderPath)

# # ====== average key values =============
# from torch import load, save
# from cnn.statistics import Statistics
#
# bopsKey = 'bops'
# folderPath = '/home/vista/Desktop/Architecture_Search/FF'
# plotsDataPath = '/home/vista/Desktop/Architecture_Search/FF/plots.data.avg'
# plotsData = load(plotsDataPath)
# bopsPlotData = plotsData[bopsKey]
#
# for k in bopsPlotData.keys():
#     results = bopsPlotData[k]
#     avg = 0.0
#     for title, bops, r in results:
#         avg += r
#     avg /= len(results)
#     bopsPlotData[k] = [(title, bops, avg)]
#
# Statistics.plotBops(plotsData, bopsKey, 'Baseline', folderPath)
# # save updated plots data
# save(plotsData, plotsDataPath)

# # ======= L1 distance =================
# from torch import load, save
# from ast import literal_eval
# from cnn.statistics import Statistics
#
# bopsKey = 'bops'
# folderPath = '/home/vista/Desktop/Architecture_Search/FF'
# plotsDataPath = '/home/vista/Desktop/Architecture_Search/FF/plots.data.avg'
# plotsData = load(plotsDataPath)
# bopsPlotData = plotsData[bopsKey]
#
# pivot = [8, 2, 6, 0]
#
# for k, results in bopsPlotData.items():
#     # extract 1st layer partition
#     if isinstance(k, str):
#         partition = literal_eval(k[:k.find('-')])
#     else:
#         if k == (3, 3):
#             partition = [0, 16, 0, 0]
#         elif k == (4, 4):
#             partition = [0, 0, 16, 0]
#         else:
#             continue
#
#     # calc L1 partition distance from pivot
#     partitionDist = sum([abs(x - y) for x, y in zip(partition, pivot)])
#
#     newResults = []
#     for title, bops, r in results:
#         newResults.append((title, partitionDist, r))
#
#     bopsPlotData[k] = newResults
#
# Statistics.plotBops(plotsData, bopsKey, 'Baseline', folderPath)
