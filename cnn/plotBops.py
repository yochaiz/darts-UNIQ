from cnn.statistics import Statistics
from torch import load
from os.path import dirname

plotPath = '/home/vista/Desktop/Architecture_Search/plots.data.new'
baseFolder = dirname(plotPath)  # script directory
plotsData = load(plotPath)
Statistics.plotBops(plotsData, 'bops', baseFolder)
