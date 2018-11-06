from torch.nn.modules.module import Module
from abc import abstractmethod


class Block(Module):
    @abstractmethod
    def getBops(self, input_bitwidth):
        raise NotImplementedError('subclasses must override getBops()!')

    @abstractmethod
    def getCurrentOutputBitwidth(self):
        raise NotImplementedError('subclasses must override getCurrentOutputBitwidth()!')

    @abstractmethod
    def choosePathByAlphas(self):
        raise NotImplementedError('subclasses must override choosePathByAlphas()!')

    @abstractmethod
    def numOfOps(self):
        raise NotImplementedError('subclasses must override numOfOps()!')

    @abstractmethod
    def getLayers(self):
        raise NotImplementedError('subclasses must override getLayers()!')

    @abstractmethod
    def outputLayer(self):
        raise NotImplementedError('subclasses must override outputLayer()!')

# @abstractmethod
# def evalMode(self):
#     raise NotImplementedError('subclasses must override evalMode()!')

# @abstractmethod
# def getOutputBitwidthList(self):
#     raise NotImplementedError('subclasses must override getOutputBitwidthList()!')

# @abstractmethod
# def chooseRandomPath(self):
#     raise NotImplementedError('subclasses must override chooseRandomPath()!')