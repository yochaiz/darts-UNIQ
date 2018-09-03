from abc import abstractmethod

from itertools import product

from torch import tensor
from torch.nn import Module
from torch.nn import functional as F

from cnn.MixedOp import MixedOp


class BaseNet(Module):
    # counts the entire model bops in continuous mode
    def countBopsContinuous(self):
        totalBops = 0
        for layer in self.layersList:
            weights = F.softmax(layer.alphas, dim=-1)
            for w, b in zip(weights, layer.bops):
                totalBops += (w * b)

        return totalBops

    # counts the entire model bops in discrete mode
    def countBopsDiscrete(self):
        totalBops = 0
        for layer in self.layersList:
            totalBops += layer.bops[layer.curr_alpha_idx]

        return totalBops

    def countBops(self):
        # wrapper is needed because countBopsFuncs is defined outside __init__()
        return self.countBopsFunc(self)

    countBopsFuncs = dict(continuous=countBopsContinuous, discrete=countBopsDiscrete)

    def __init__(self, criterion, initLayersParams, bopsFuncKey):
        super(BaseNet, self).__init__()
        # init criterion
        self._criterion = criterion
        # init layers
        self.initLayers(initLayersParams)
        # build mixture layers list
        self.layersList = [m for m in self.modules() if isinstance(m, MixedOp)]
        # set bops counter function
        self.countBopsFunc = self.countBopsFuncs[bopsFuncKey]
        # collect learnable params (weights)
        self.learnable_params = [param for param in self.parameters() if param.requires_grad]
        # init learnable alphas
        self.learnable_alphas = []
        # init number of layers we have completed its quantization
        self.nLayersQuantCompleted = 0

        # init layers permutation list
        self.layersPerm = []
        # init number of permutations counter
        self.nPerms = 1
        for layer in self.layersList:
            # add layer numOps range to permutation list
            self.layersPerm.append(list(range(len(layer.alphas))))
            self.nPerms *= len(layer.alphas)

    @abstractmethod
    def initLayers(self, params):
        raise NotImplementedError('subclasses must override initLayers()!')

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError('subclasses must override forward()!')

    @abstractmethod
    def switch_stage(self, logger=None):
        raise NotImplementedError('subclasses must override switch_stage()!')

    @abstractmethod
    def loadUNIQPre_trained(self, path, logger, gpu):
        raise NotImplementedError('subclasses must override loadUNIQPre_trained()!')

    def nLayers(self):
        return len(self.layersList)

    def arch_parameters(self):
        return self.learnable_alphas

    def _loss(self, input, target):
        # sum all paths losses * the path alphas multiplication
        totalLoss = 0.0
        for perm in product(*self.layersPerm):
            alphasProd = 1.0
            # set perm index in each layer
            for i, p in enumerate(perm):
                layer = self.layersList[i]
                layer.curr_alpha_idx = p
                probs = F.softmax(layer.alphas)
                alphasProd *= probs[p]

            logits = self.forward(input)
            # only the alphas are changing...
            totalLoss += (alphasProd * self._criterion(logits, target, self.countBops()))
        # TODO: do we need to average the totalLoss ???

        # print('totalLoss:[{:.5f}]'.format(totalLoss))
        return totalLoss

        # logits = self.forward(input)
        # return self._criterion(logits, target, self.countBops())

    def trainMode(self):
        for l in self.layersList:
            l.trainMode()
        # calc bops ratio
        bopsRatio = self._criterion.calcBopsRatio(self.countBops())
        # bopsLoss = self._criterion.calcBopsLoss(bopsRatio)
        return bopsRatio

    def evalMode(self):
        for l in self.layersList:
            l.evalMode()
        # calc bops ratio
        bopsRatio = self._criterion.calcBopsRatio(self.countBops())
        # bopsLoss = self._criterion.calcBopsLoss(bopsRatio)
        return bopsRatio

    # return top k operations per layer
    def topOps(self, k):
        top = []
        for layer in self.layersList:
            # calc weights from alphas and sort them
            weights = F.softmax(layer.alphas, dim=-1)
            # weights = layer.alphas
            wSorted, wIndices = weights.sort(descending=True)
            # keep only top-k
            wSorted = wSorted[:k]
            wIndices = wIndices[:k]
            # add to top
            top.append([(i, w.item(), layer.alphas[i], layer.ops[i]) for w, i in zip(wSorted, wIndices)])

        return top

    # create list of lists of alpha with its corresponding operation
    def alphas_state(self):
        res = []
        for layer in self.layersList:
            layerAlphas = [(a, op) for a, op in zip(layer.alphas, layer.ops)]
            res.append(layerAlphas)

        return res

    def load_alphas_state(self, state):
        for layer, layerAlphas in zip(self.layersList, state):
            for i, elem in enumerate(layerAlphas):
                a, _ = elem
                layer.alphas[i] = a

    # convert current model to discrete, i.e. keep nOpsPerLayer optimal operations per layer
    def toDiscrete(self, nOpsPerLayer=1):
        for layer in self.layersList:
            # calc weights from alphas and sort them
            weights = F.softmax(layer.alphas, dim=-1)
            _, wIndices = weights.sort(descending=True)
            # update layer alphas
            layer.alphas = layer.alphas[wIndices[:nOpsPerLayer]]
            #        layer.alphas = tensor(tensor(layer.alphas.tolist()).cuda(), requires_grad=True)
            layer.alphas = tensor(tensor(layer.alphas.tolist()).cuda())
            # take indices of ops we want to remove from layer
            wIndices = wIndices[nOpsPerLayer:]
            # convert to list
            wIndices = wIndices.tolist()
            # sort indices ascending
            wIndices.sort()
            # remove ops and corresponding bops from layer
            for w in reversed(wIndices):
                del layer.ops[w]
                del layer.bops[w]
