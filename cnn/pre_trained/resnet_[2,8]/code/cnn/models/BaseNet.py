from abc import abstractmethod

from itertools import product
from random import randint
from pandas import DataFrame

from torch import tensor, zeros
from torch.nn import Module
from torch.nn import functional as F

from cnn.MixedOp import MixedOp
from cnn.uniq_loss import UniqLoss

from UNIQ.actquant import ActQuant
from UNIQ.quantize import quantize, restore_weights, backup_weights
from collections import OrderedDict


def save_quant_state(self, _):
    assert (self.noise is False)
    if self.quant and not self.noise and self.training:
        self.full_parameters = {}
        layers_list = self.get_layers_list()
        layers_steps = self.get_layers_steps(layers_list)
        assert (len(layers_steps) == 1)

        self.full_parameters = backup_weights(layers_steps[0], self.full_parameters)
        quantize(layers_steps[0], bitwidth=self.bitwidth[0])


def restore_quant_state(self, _, __):
    assert (self.noise is False)
    if self.quant and not self.noise and self.training:
        layers_list = self.get_layers_list()
        layers_steps = self.get_layers_steps(layers_list)
        assert (len(layers_steps) == 1)

        restore_weights(layers_steps[0], self.full_parameters)  # Restore the quantized layers


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

    alphasCsvFileName = 'alphas.csv'

    def __init__(self, lmbda, maxBops, initLayersParams, bopsFuncKey, saveFolder):
        super(BaseNet, self).__init__()
        # init layers
        self.initLayers(initLayersParams)
        # build mixture layers list
        self.layersList = [m for m in self.modules() if isinstance(m, MixedOp)]
        # set bops counter function
        self.countBopsFunc = self.countBopsFuncs[bopsFuncKey]
        # init criterion
        self._criterion = UniqLoss(lmdba=lmbda, maxBops=maxBops or self.countBops(), folderName=saveFolder)
        self._criterion = self._criterion.cuda()
        # collect learnable params (weights)
        self.learnable_params = [param for param in self.parameters() if param.requires_grad]
        # init learnable alphas
        self.learnable_alphas = []
        # init number of layers we have completed its quantization
        self.nLayersQuantCompleted = 0
        # init all batch samples loss variance holder
        self.allLossSamplesVariance = 0

        # init layers permutation list
        self.layersPerm = []
        # init number of permutations counter
        self.nPerms = 1
        for layer in self.layersList:
            # add layer numOps range to permutation list
            self.layersPerm.append(list(range(len(layer.alphas))))
            self.nPerms *= len(layer.alphas)

        # init alphas DataFrame
        self.alphas_df = None
        if saveFolder:
            # update save path if saveFolder exists
            self.alphasCsvFileName = '{}/{}'.format(saveFolder, self.alphasCsvFileName)
            # init DataFrame cols
            cols = ['Epoch', 'Batch']
            cols += ['Layer_{}'.format(i) for i in range(self.nLayers())]
            self.cols = cols
            # init DataFrame
            self.alphas_df = DataFrame([], columns=self.cols)
            # set init data
            data = ['init', 'init']
            # save alphas data
            self.save_alphas_to_csv(data)

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
    def loadUNIQPre_trained(self, chckpntDict):
        raise NotImplementedError('subclasses must override loadUNIQPre_trained()!')

    @abstractmethod
    def turnOnWeights(self):
        raise NotImplementedError('subclasses must override turnOnWeights()!')

    def nLayers(self):
        return len(self.layersList)

    def arch_parameters(self):
        return self.learnable_alphas

    def turnOffAlphas(self):
        for layer in self.layersList:
            # turn off alphas gradients
            layer.alphas.requires_grad = False

        self.learnable_alphas = []

    def turnOnAlphas(self):
        self.learnable_alphas = []
        for layer in self.layersList:
            # turn on alphas gradients
            layer.alphas.requires_grad = True
            self.learnable_alphas.append(layer.alphas)

            for op in layer.ops:
                # turn off noise in op
                op.noise = False

                ## ==== for tinyNet ====
                # # set pre & post quantization hooks, from now on we want to quantize these ops
                # op.register_forward_pre_hook(save_quant_state)
                # op.register_forward_hook(restore_quant_state)

    def loadBitwidthWeigths(self, stateDict, MaxBopsBits, bitwidth):
        # TODO: get model and assure the selected index bitwidth
        # check idx of MaxBopsBits inside bitwidths
        maxBopsBitsIdx = bitwidth.index(MaxBopsBits)
        maxBopsStateDict = OrderedDict()
        opsKey = 'ops.'
        for key in stateDict.keys():
            # if operation is for max bops bits idx
            if opsKey in key:
                keyOp_num = key.split(opsKey)[1][0]
                if int(keyOp_num) == maxBopsBitsIdx:
                    maxBopsKey = key.replace(opsKey + keyOp_num, opsKey + '0')
                    maxBopsStateDict[maxBopsKey] = stateDict[key]
            else:
                maxBopsStateDict[key] = stateDict[key]

        self.load_state_dict(maxBopsStateDict)

    # def _loss(self, input, target):
    #     totalLoss = 0.0
    #     nIter = min(self.nPerms, 1000)
    #     for _ in range(nIter):
    #         logits = self.forward(input)
    #
    #         # calc alphas product
    #         alphasProduct = 1.0
    #         for layer in self.layersList:
    #             probs = F.softmax(layer.alphas)
    #             alphasProduct *= probs[layer.curr_alpha_idx]
    #
    #         permLoss = alphasProduct * self._criterion(logits, target, self.countBops())
    #         # permLoss = self._criterion(logits, target, self.countBops()) / nIter
    #         permLoss.backward(retain_graph=True)
    #
    #         totalLoss += permLoss.item()
    #
    #     return totalLoss

    def _loss(self, input, target):
        # init how many samples per alpha
        nSamplesPerAlpha = 50
        # init total loss
        totalLoss = 0.0
        # init loss samples list for ALL alphas
        allLossSamples = []
        for layer in self.layersList:
            # turn off coin toss for this layer
            layer.alphas.requires_grad = False
            # init layer alphas gradient
            layerAlphasGrad = zeros(len(layer.alphas)).cuda()
            # calc layer alphas softmax
            probs = F.softmax(layer.alphas, dim=-1)

            for i, alpha in enumerate(layer.alphas):
                # select the specific alpha in this layer
                layer.curr_alpha_idx = i
                # init alpha loss
                alphaLoss = 0.0
                # init loss samples list
                alphaLossSamples = []
                for _ in range(nSamplesPerAlpha):
                    # forward through some path in model
                    logits = self.forward(input)
                    # alphaLoss += self._criterion(logits, target, self.countBops()).detach()
                    alphaLossSamples.append(self._criterion(logits, target, self.countBops()).detach())

                # add current alpha loss samples to all loss samples list
                allLossSamples.extend(alphaLossSamples)
                # update alpha average loss
                # alphaLoss /= nSamplesPerAlpha
                # calc alpha average loss
                alphaAvgLoss = sum(alphaLossSamples) / nSamplesPerAlpha
                layerAlphasGrad[i] = alphaAvgLoss
                # add alpha loss to total loss
                totalLoss += (alphaAvgLoss * probs[i])

                # calc loss samples variance
                lossVariance = [((x - alphaAvgLoss) ** 2) for x in alphaLossSamples]
                layer.alphasLossVariance.data[i] = sum(lossVariance) / (nSamplesPerAlpha - 1)

            # turn in coin toss for this layer
            layer.alphas.requires_grad = True
            # set layer alphas gradient
            layer.alphas.grad = layerAlphasGrad

        # average total loss
        totalLoss /= self.nLayers()
        # calc all loss samples average
        nTotalSamples = len(allLossSamples)
        allLossSamplesAvg = sum(allLossSamples) / nTotalSamples
        # calc all loss samples variance
        allLossSamples = [((x - allLossSamplesAvg) ** 2) for x in allLossSamples]
        self.allLossSamplesVariance = (sum(allLossSamples) / (nTotalSamples - 1)).item()

        # subtract average total loss from every alpha gradient
        for layer in self.layersList:
            layer.alphas.grad -= totalLoss
            # calc layer alphas softmax
            probs = F.softmax(layer.alphas, dim=-1)
            # multiply each grad by its probability
            layer.alphas.grad *= probs

        return totalLoss

    # def _loss(self, input, target):
    #     # sum all paths losses * the path alphas multiplication
    #     totalLoss = 0.0
    #     nIter = min(self.nPerms, 1000)
    #     for _ in range(nIter):
    #         # for perm in product(*self.layersPerm):
    #         perm = [randint(0, len(layer.alphas) - 1) for layer in self.layersList]
    #         alphasProduct = 1.0
    #         # set perm index in each layer
    #         for i, p in enumerate(perm):
    #             layer = self.layersList[i]
    #             layer.curr_alpha_idx = p
    #             probs = F.softmax(layer.alphas)
    #             alphasProduct *= probs[p]
    #
    #         logits = self.forward(input)
    #         # only the alphas are changing...
    #         permLoss = (alphasProduct * self._criterion(logits, target, self.countBops()))
    #         permLoss.backward(retain_graph=True)
    #         totalLoss += permLoss.item()
    #     # TODO: do we need to average the totalLoss ???
    #
    #     # print('totalLoss:[{:.5f}]'.format(totalLoss))
    #     return totalLoss

    #
    #     # logits = self.forward(input)
    #     # return self._criterion(logits, target, self.countBops())

    def calcBopsRatio(self):
        return self._criterion.calcBopsRatio(self.countBops())

    # select random alpha
    def chooseRandomPath(self):
        for l in self.layersList:
            l.chooseRandomPath()

        # calc bops ratio
        return self.calcBopsRatio()

    def choosePathByAlphas(self):
        for l in self.layersList:
            l.choosePathByAlphas()

        # calc bops ratio
        return self.calcBopsRatio()

    def trainMode(self):
        for l in self.layersList:
            l.trainMode()

    def evalMode(self):
        for l in self.layersList:
            l.evalMode()

        # calc bops ratio
        return self.calcBopsRatio()

    # return top k operations per layer
    def topOps(self, k):
        top = []
        for layer in self.layersList:
            # calc weights from alphas and sort them
            weights = F.softmax(layer.alphas, dim=-1)
            wSorted, wIndices = weights.sort(descending=True)
            # keep only top-k
            wSorted = wSorted[:k]
            wIndices = wIndices[:k]
            # add to top
            top.append([(i, w.item(), layer.alphas[i], layer.ops[i]) for w, i in zip(wSorted, wIndices)])

        return top

    # save alphas values to csv
    def save_alphas_to_csv(self, data):
        if self.alphas_df is not None:
            data += [[round(e.item(), 5) for e in layer.alphas] for layer in self.layersList]
            # create new row
            d = DataFrame([data], columns=self.cols)
            # add row
            self.alphas_df = self.alphas_df.append(d)
            # save DataFrame
            self.alphas_df.to_csv(self.alphasCsvFileName)

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
