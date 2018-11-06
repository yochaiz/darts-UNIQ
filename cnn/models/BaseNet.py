from abc import abstractmethod
from pandas import DataFrame
from os.path import exists
from numpy import argmax

from torch.nn import Module, Conv2d
from torch.nn import functional as F
from torch import load as loadModel

from cnn.MixedLayer import MixedLayerNoBN as MixedLayer
from cnn.MixedFilter import MixedConvBNWithReLU as MixedConvWithReLU
from cnn.uniq_loss import UniqLoss
import cnn.statistics
from cnn.HtmlLogger import HtmlLogger

from UNIQ.quantize import check_quantization


# from torch import save as saveModel
# from torch import ones, zeros, no_grad, cat, tensor
# from torch.nn import  CrossEntropyLoss


# preForward hook for training weights phase.
# when we train weights, we need to quantize staged layers before forward, and remove quantization after forward in order to update by gradient
# same for noise, we need to add noise before forward, and remove noise after forward, in order to update by gradient
def preForward(self, input):
    print('BaseNet preForward')
    deviceID = input[0].device.index
    assert (deviceID not in self.hookDevices)
    self.hookDevices.append(deviceID)

    assert (self.training is True)
    # update layers list to new DataParallel layers copies
    self.layersList = self.buildLayersList()
    # quantize staged layers
    self.restoreQuantizationForStagedLayers()

    # add noise to next to be staged layer
    if self.nLayersQuantCompleted < self.nLayers():
        layer = self.layersList[self.nLayersQuantCompleted]
        assert (layer.added_noise is True)
        for op in layer.opsList():
            assert (op.noise is True)
            op.add_noise()


def postForward(self, input, __):
    print('BaseNet postForward')
    deviceID = input[0].device.index
    assert (deviceID in self.hookDevices)
    self.hookDevices.remove(deviceID)

    assert (self.training is True)
    # remove quantization from staged layers
    self.removeQuantizationFromStagedLayers()

    # remove noise from next to be staged layer
    if self.nLayersQuantCompleted < self.nLayers():
        layer = self.layersList[self.nLayersQuantCompleted]
        assert (layer.added_noise is True)
        for op in layer.opsList():
            assert (op.noise is True)
            op.restore_state()


class BaseNet(Module):
    # init bitwidth of input to model
    modelInputBitwidth = 8
    modelInputnFeatureMaps = 3

    # counts the entire model bops in discrete mode
    def countBopsDiscrete(self):
        totalBops = 0
        # input_bitwidth is a list of bitwidth per feature map
        input_bitwidth = [self.modelInputBitwidth] * self.modelInputnFeatureMaps

        for layer in self.layers:
            totalBops += layer.getBops(input_bitwidth)
            input_bitwidth = layer.getCurrentOutputBitwidth()

        totalBops /= 1E9
        return totalBops

    def countBops(self):
        # wrapper is needed because countBopsFuncs is defined outside __init__()
        return self.countBopsFunc(self)

    countBopsFuncs = dict(discrete=countBopsDiscrete)

    alphasCsvFileName = 'alphas.csv'

    def buildLayersList(self):
        layersList = []
        for layer in self.layers:
            layersList.extend(layer.getLayers())

        return layersList

    def __init__(self, args, initLayersParams):
        super(BaseNet, self).__init__()
        # init save folder
        saveFolder = args.save
        # init layers
        self.layers = self.initLayers(initLayersParams)
        # build mixture layers list
        self.layersList = self.buildLayersList()
        # set bops counter function
        self.countBopsFunc = self.countBopsFuncs[args.bopsCounter]
        # init statistics
        self.stats = cnn.statistics.Statistics(self.layersList, saveFolder)
        # collect learnable params (weights)
        self.learnable_params = [param for param in self.parameters() if param.requires_grad]
        # init learnable alphas
        self.learnable_alphas = self.getLearnableAlphas()
        # init number of layers we have completed its quantization
        self.nLayersQuantCompleted = 0
        # calc init baseline bops
        baselineBops = self.calcBaselineBops()
        args.baselineBops = baselineBops[args.baselineBits[0]]
        # plot baselines bops
        self.stats.addBaselineBopsData(args, baselineBops)
        # init criterion
        self._criterion = UniqLoss(args)
        self._criterion = self._criterion.cuda()

        # init hooks handlers list
        self.hooksList = []
        # set hook flag, to make sure hook happens
        # # turn it on on pre-forward hook, turn it off on post-forward hook
        # self.hookFlag = False
        self.hookDevices = []

        self.printToFile(saveFolder)
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
        self.__initAlphasDataFrame(saveFolder)

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
    def loadUNIQPreTrained(self, checkpoint):
        raise NotImplementedError('subclasses must override loadUNIQPreTrained()!')

    @abstractmethod
    def loadSingleOpPreTrained(self, checkpoint):
        raise NotImplementedError('subclasses must override loadSingleOpPreTrained()!')

    @abstractmethod
    def turnOnWeights(self):
        raise NotImplementedError('subclasses must override turnOnWeights()!')

    def nLayers(self):
        return len(self.layersList)

    def getLearnableAlphas(self):
        return [layer.alphas for layer in self.layersList if layer.alphas.requires_grad is True]

    def updateLearnableAlphas(self):
        self.learnable_alphas = self.getLearnableAlphas()

    def arch_parameters(self):
        return self.learnable_alphas

    # def calcStatistics(self, statistics_queue):
    #     # prepare for collecting statistics, reset register_buffers values
    #     for layer in self.layersList:
    #         for op in layer.opsList:
    #             conv = op.getConv()
    #             # reset conv register_buffer values
    #             conv.layer_b = ones(1).cuda()
    #             conv.layer_basis = ones(1).cuda()
    #             conv.initial_clamp_value = ones(1).cuda()
    #             # get actquant
    #             actQuant = op.getReLU()
    #             if actQuant:
    #                 # reset actquant register_buffer values
    #                 actQuant.running_mean = zeros(1).cuda()
    #                 actQuant.running_std = zeros(1).cuda()
    #                 actQuant.clamp_val.data = zeros(1).cuda()
    #                 # set actquant to statistics forward
    #                 actQuant.forward = actQuant.statisticsForward
    #
    #     # train for statistics
    #     criterion = CrossEntropyLoss().cuda()
    #     nBatches = 80
    #     self.eval()
    #     with no_grad():
    #         for step, (input, target) in enumerate(statistics_queue):
    #             if step >= nBatches:
    #                 break
    #
    #             output = self(input.cuda())
    #             criterion(output, target.cuda())
    #
    #     # apply quantize class statistics functions
    #     for layerIdx, layer in enumerate(self.layersList):
    #         # concat layer feature maps together, in order to get initial_clamp_value identical to NICE
    #         # because initial_clamp_value is calculated based on feature maps weights values
    #         x = tensor([]).cuda()
    #         for op in layer.opsList:
    #             x = cat((x, op.getConv().weight), dim=0)
    #
    #         for op in layer.opsList:
    #             clamp_value = op.quantize.basic_clamp(x)
    #             conv = op.getConv()
    #             conv.initial_clamp_value = clamp_value
    #             # restore actquant forward function
    #             actQuant = op.getReLU()
    #             # set actquant to standard forward
    #             if actQuant:
    #                 op.quantize.get_act_max_value_from_pre_calc_stats([actQuant])
    #                 actQuant.forward = actQuant.standardForward
    #
    #         print('Layer [{}] - initial_clamp_value:[{}]'.format(layerIdx, conv.initial_clamp_value.item()))
    #
    #     # for op in layer.opsList:
    #     #     opModulesList = list(op.modules())
    #     #     op.quantize.get_act_max_value_from_pre_calc_stats(opModulesList)
    #     #     op.quantize.set_weight_basis(opModulesList, None)
    #     #
    #     #     conv = op.getConv()
    #     #     print(conv.initial_clamp_value)
    #     #
    #
    # # updates statistics in checkpoint, in order to avoid calculating statistics when loading model from checkpoint
    # def updateCheckpointStatistics(self, checkpoint, path, statistics_queue):
    #     needToUpdate = ('updated_statistics' not in checkpoint) or (checkpoint['updated_statistics'] is not True)
    #     if needToUpdate:
    #         # quantize model
    #         self.quantizeUnstagedLayers()
    #         # change self.nLayersQuantCompleted so calcStatistics() won't quantize again
    #         nLayersQuantCompletedOrg = self.nLayersQuantCompleted
    #         self.nLayersQuantCompleted = self.nLayers()
    #         # load checkpoint weights
    #         self.load_state_dict(checkpoint['state_dict'])
    #         # calc weights statistics
    #         self.calcStatistics(statistics_queue)
    #         # update checkpoint
    #         checkpoint['state_dict'] = self.state_dict()
    #         checkpoint['updated_statistics'] = True
    #         # save updated checkpoint
    #         saveModel(checkpoint, path)
    #         # restore nLayersQuantCompleted
    #         self.nLayersQuantCompleted = nLayersQuantCompletedOrg
    #
    #     return needToUpdate

    # layer_basis is a function of filter quantization,
    # therefore we have to update its value bases on weight_max_int, which is a function of weights bitwidth
    def __updateStatistics(self, loggerFuncs=[]):
        for layer in self.layersList:
            for op in layer.opsList():
                conv = op.getModule(Conv2d)
                # update layer_basis value based on weights bitwidth
                conv.layer_basis = conv.initial_clamp_value / op.quantize.weight_max_int

        for f in loggerFuncs:
            f('Updated layer_basis according to bitwidth (weight_max_int)')

    def loadPreTrained(self, path, logger, gpu):
        # init bool flag whether we loaded ops in the same layer with equal or different weights
        loadOpsWithDifferentWeights = False
        loggerRows = []
        loadSuccess = None
        if path is not None:
            if exists(path):
                # load checkpoint
                checkpoint = loadModel(path, map_location=lambda storage, loc: storage.cuda(gpu))
                assert (checkpoint['updated_statistics'] is True)
                chckpntStateDict = checkpoint['state_dict']
                # load model state dict keys
                modelStateDictKeys = set(self.state_dict().keys())
                # compare dictionaries
                dictDiff = modelStateDictKeys.symmetric_difference(set(chckpntStateDict.keys()))
                # update flag value
                loadOpsWithDifferentWeights = len(dictDiff) == 0
                # decide how to load checkpoint state dict
                if loadOpsWithDifferentWeights:
                    # load directly, keys are the same
                    self.load_state_dict(chckpntStateDict)
                else:
                    # use some function to map keys
                    loadFuncs = [self.loadUNIQPreTrained, self.loadSingleOpPreTrained]
                    for func in loadFuncs:
                        loadSuccess = func(chckpntStateDict)
                        if loadSuccess is not False:
                            # update statistics if we don't load ops with different statistics
                            self.__updateStatistics(loggerFuncs=[lambda msg: loggerRows.append(['Statistics update', msg])])
                            break

                if loadSuccess is not False:
                    # add info rows about checkpoint
                    loggerRows.append(['Path', '{}'.format(path)])
                    loggerRows.append(['Validation accuracy', '{:.5f}'.format(checkpoint['best_prec1'])])
                    loggerRows.append(['checkpoint[updated_statistics]', checkpoint['updated_statistics']])
                    # check if model includes stats
                    modelIncludesStats = False
                    for key in chckpntStateDict.keys():
                        if key.endswith('.layer_basis'):
                            modelIncludesStats = True
                            break
                    loggerRows.append(['Includes stats', '{}'.format(modelIncludesStats)])
                else:
                    loggerRows.append(['Path', 'Failed to load pre-trained from [{}], state_dict does not fit'.format(path)])
            else:
                loggerRows.append(['Path', 'Failed to load pre-trained from [{}], path does not exists'.format(path)])

            # load pre-trained model if we tried to load pre-trained
            logger.addInfoTable('Pre-trained model', loggerRows)

        return loadOpsWithDifferentWeights

    # load weights for each filter from its uniform model, i.e. load 2-bits filter weights from uniform 2-bits model
    # weights in uniform models are full-precision, i.e. before quantization
    def loadUniformPreTrained(self, args, logger):
        from collections import OrderedDict
        from cnn.MixedFilter import QuantizedOp
        from cnn.utils import loadCheckpoint
        from torch.nn.modules import Linear, BatchNorm2d

        def b(op, prefix):
            keysList = []

            for name, param in op._parameters.items():
                if param is not None:
                    keysList.append(prefix + name)
            for name, buf in op._buffers.items():
                if buf is not None:
                    keysList.append(prefix + name)
            for name, module in op._modules.items():
                if module is not None:
                    keysList.extend(b(module, prefix + name + '.'))

            return keysList

        def a(model, dict, prefix=''):
            for name, module in model._modules.items():
                key = None
                if isinstance(module, QuantizedOp):
                    key = module.getBitwidth()
                # elif isinstance(module, BatchNorm2d) or isinstance(module, Linear):
                elif isinstance(module, Linear):
                    key = (32, 32)

                if key is not None:
                    if key not in dict.keys():
                        dict[key] = []

                    dict[key].extend(b(module, prefix + name + '.'))

                else:
                    a(module, dict, prefix + name + '.')

        modelDict = OrderedDict()
        a(self, modelDict)

        # transform downsamples keys
        transformMap = [[(2, None), (2, 2)], [(3, None), (3, 3)], [(4, None), (4, 4)], [(8, None), (8, 8)]]
        for srcBitwidth, dstBitwidth in transformMap:
            if srcBitwidth in modelDict.keys():
                modelDict[dstBitwidth].extend(modelDict[srcBitwidth])
                del modelDict[srcBitwidth]

        keysList = []
        for bitwidth, bitwidthKeysList in modelDict.items():
            keysList.extend(bitwidthKeysList)

        modelStateDictKeys = set(self.state_dict().keys())
        dictDiff = modelStateDictKeys.symmetric_difference(set(keysList))
        assert (len(dictDiff) == 0)

        stateDict = OrderedDict()
        token1 = '.ops.'
        token2 = '.op.'
        for bitwidth, bitwidthKeysList in modelDict.items():
            if bitwidth == (32, 32):
                continue

            checkpoint, _ = loadCheckpoint(args.dataset, args.model, bitwidth)
            assert (checkpoint is not None)
            chckpntStateDict = checkpoint['state_dict']
            for key in bitwidthKeysList:
                prefix = key[:key.index(token1)]
                suffix = key[key.rindex(token2):]
                # convert model key to checkpoint key
                chckpntKey = prefix + token1 + '0.0' + suffix
                # add value to dict
                stateDict[key] = chckpntStateDict[chckpntKey]

        # # load keys from (32, 32) checkpoint, no need to transform keys
        bitwidth = (8, 8)
        checkpoint, _ = loadCheckpoint(args.dataset, args.model, bitwidth)  # , filename='model.updated_stats.pth.tar')
        # checkpoint = loadModel("/home/vista/Desktop/Architecture_Search/ZZ/cifar100/resnet_[2#2,4#3#4#8]/pre_trained_checkpoint.pth.tar")
        assert (checkpoint is not None)
        chckpntStateDict = checkpoint['state_dict']
        # map = self.buildStateDictMap(chckpntStateDict)
        # invMap = {v: k for k, v in map.items()}
        bitwidth = (32, 32)
        for key in modelDict[bitwidth]:
            stateDict[key] = chckpntStateDict[key]

            # prefix = key[:key.rindex('.')]
            # suffix = key[key.rindex('.'):]
            # newKey = invMap[prefix]
            # stateDict[key] = chckpntStateDict[newKey + suffix]

        dictDiff = modelStateDictKeys.symmetric_difference(set(stateDict.keys()))
        assert (len(dictDiff) == 0)

        self.load_state_dict(stateDict)
        logger.addInfoTable('Pre-trained model', [['Loaded each filter with filter from the corresponding bitwidth uniform model']])

    def loss(self, logits, target):
        return self._criterion(logits, target, self.countBops())

    def turnOffAlphas(self):
        for layer in self.layersList:
            layer.alphas.grad = None

    def calcBopsRatio(self):
        return self._criterion.calcBopsRatio(self.countBops())

    def choosePathByAlphas(self, loggerFuncs=[]):
        for l in self.layers:
            l.choosePathByAlphas()

        logMsg = 'Model layers filters partition has been updated by alphas distribution'
        for f in loggerFuncs:
            f(logMsg)

    # set curr_alpha_idx to each filter by alphas values
    def setFiltersByAlphas(self, loggerFuncs=[]):
        for layer in self.layersList:
            layer.setFiltersPartitionByAlphas()

        logMsg = 'Model layers filters partition has been updated by alphas values'
        for f in loggerFuncs:
            f(logMsg)

    # returns list of layers filters partition
    def getCurrentFiltersPartition(self):
        return [layer.getCurrentFiltersPartition() for layer in self.layersList]

    # partition is list of int tensors
    # given a partition, set model filters accordingly
    def setFiltersByPartition(self, partition, loggerFuncs=[]):
        for layer, p in zip(self.layersList, partition):
            layer.setFiltersPartition(p)

        logMsg = 'Model layers filters partition has been updated by given partition'
        for f in loggerFuncs:
            f(logMsg)

    def isQuantized(self):
        for layerIdx, layer in enumerate(self.layersList):
            assert (layer.quantized is True)
            assert (layer.added_noise is False)
            for opIdx, op in enumerate(layer.opsList()):
                assert (check_quantization(op.getModule(Conv2d).weight) <= (2 ** op.bitwidth[0]))

        return True

    def setWeightsTrainingHooks(self):
        assert (len(self.hooksList) == 0)
        # assign pre & post forward hooks
        self.hooksList = [self.register_forward_pre_hook(preForward), self.register_forward_hook(postForward)]

    def removeWeightsTrainingHooks(self):
        for handler in self.hooksList:
            handler.remove()
        # clear hooks handlers list
        self.hooksList.clear()

    # remove quantization from staged layers before training weights
    # quantization will be set through pre-forward hook
    # we keep ActQaunt.qunatize_during_training == True
    def removeQuantizationFromStagedLayers(self):
        for layerIdx in range(self.nLayersQuantCompleted):
            layer = self.layersList[layerIdx]
            assert (layer.quantized is True)
            # remove quantization from layer ops
            for op in layer.opsList():
                op.restore_state()

    # restore quantization for staged layers after training weights
    # quantization will be set through pre-forward hook
    # we keep ActQaunt.qunatize_during_training == True
    def restoreQuantizationForStagedLayers(self):
        for layerIdx in range(self.nLayersQuantCompleted):
            layer = self.layersList[layerIdx]
            assert (layer.quantized is True)
            # refresh layer ops list. we want ops list to contain the ops DataParallel GPU copies
            # quantize layer ops
            for op in layer.opsList():
                op.quantizeFunc()
                assert (check_quantization(op.getModule(Conv2d).weight) <= (2 ** op.bitwidth[0]))

    def quantizeUnstagedLayers(self):
        # quantize model layers that haven't switched stage yet
        # no need to turn gradients off, since with no_grad() does it
        if self.nLayersQuantCompleted < self.nLayers():
            # turn off noise if 1st unstaged layer
            layer = self.layersList[self.nLayersQuantCompleted]
            layer.turnOffNoise(self.nLayersQuantCompleted)
            # quantize all unstaged layers
            for layerIdx, layer in enumerate(self.layersList[self.nLayersQuantCompleted:]):
                # quantize
                layer.quantize(self.nLayersQuantCompleted + layerIdx)

        assert (self.isQuantized() is True)

    def unQuantizeUnstagedLayers(self):
        # restore weights (remove quantization) of model layers that haven't switched stage yet
        if self.nLayersQuantCompleted < self.nLayers():
            for layerIdx, layer in enumerate(self.layersList[self.nLayersQuantCompleted:]):
                # remove quantization
                layer.unQuantize(self.nLayersQuantCompleted + layerIdx)
            # add noise back to 1st unstaged layer
            layer = self.layersList[self.nLayersQuantCompleted]
            layer.turnOnNoise(self.nLayersQuantCompleted)

    def resetForwardCounters(self):
        for layer in self.layersList:
            for filter in layer.filters:
                # reset filter counters
                filter.resetOpsForwardCounters()

    # apply some function on baseline models
    # baseline models are per each filter bitwidth
    # this function create a map from baseline bitwidth to func() result on baseline model
    def applyOnBaseline(self, func, applyOnAlphasDistribution=False):
        baselineBops = {}
        # save current model filters curr_alpha_idx
        modelFiltersIdx = [[filter.curr_alpha_idx for filter in layer.filters] for layer in self.layersList]
        # iterate over model layers
        for layer in self.layersList:
            # we want to iterate only over MixedConvWithReLU filters layer
            if isinstance(layer.filters[0], MixedConvWithReLU):
                # get layer filters bitwidth list
                layerBitwidths = layer.getAllBitwidths()
                # iterate over bitwidth and calc bops for their uniform model
                for idx, bitwidth in enumerate(layerBitwidths):
                    # calc only for bitwidths that are not in baselineBops dictionary
                    if bitwidth not in baselineBops:
                        # if we need to calc bops for bitwidth uniform model, then we have to set filters curr_alpha_idx
                        for layer2 in self.layersList:
                            # get layer bitwidth list
                            layerBitwidths2 = layer2.getAllBitwidths()
                            # find target bitwidth in bitwidth list
                            if bitwidth in layerBitwidths2:
                                idx = layerBitwidths2.index(bitwidth)
                            else:
                                # if it is a MixedConv layer, then modify the bitwidth we are looking for
                                modifiedBitwidth = (bitwidth[0], None)
                                idx = layerBitwidths2.index(modifiedBitwidth)
                            # set layers curr_alpha_idx to target bitwidth index
                            for filter in layer2.filters:
                                filter.curr_alpha_idx = idx
                        # update bops value in dictionary
                        baselineBops[bitwidth] = func()

        # apply on current alphas distribution
        if applyOnAlphasDistribution:
            self.setFiltersByAlphas()
            # &#945; is greek alpha symbol in HTML
            baselineBops['&#945;'] = func()

        # restore filters curr_alpha_idx
        for layer, layerFiltersIdx in zip(self.layersList, modelFiltersIdx):
            for filter, filterIdx in zip(layer.filters, layerFiltersIdx):
                filter.curr_alpha_idx = filterIdx

        return baselineBops

    # calc bops of uniform models, based on filters ops bitwidth
    def calcBaselineBops(self):
        return self.applyOnBaseline(self.countBops)

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
            # get layer bitwidths
            bitwidths = layer.getAllBitwidths()
            # add to top
            top.append([(i, w.item(), layer.alphas[i], bitwidths[i]) for w, i in zip(wSorted, wIndices)])

        return top

    # create list of tuples (layer index, layer alphas)
    def save_alphas_state(self):
        return [(i, layer.alphas) for i, layer in enumerate(self.layersList)]

    def load_alphas_state(self, state, loggerFuncs=[]):
        for layerIdx, alphas in state:
            layerAlphas = self.layersList[layerIdx].alphas
            device = layerAlphas.device
            layerAlphas.data = alphas.data.to(device)

        logMsg = 'Loaded alphas from checkpoint'
        # log message to all loggers
        for f in loggerFuncs:
            f(logMsg)

    def __initAlphasDataFrame(self, saveFolder):
        if saveFolder:
            # update save path if saveFolder exists
            self.alphasCsvFileName = '{}/{}'.format(saveFolder, self.alphasCsvFileName)
            # init DataFrame cols
            cols = ['Epoch', 'Batch']
            cols += ['Layer_{}'.format(i) for i in range(self.nLayers())]
            self.cols = cols
            # init DataFrame
            self.alphas_df = DataFrame([], columns=cols)
            # set init data
            data = ['init', 'init']
            # save alphas data
            self.save_alphas_to_csv(data)

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

    def logDominantQuantizedOp(self, k, loggerFuncs=[]):
        if (not loggerFuncs) or (len(loggerFuncs) == 0):
            return

        rows = [['Layer #', 'Alphas']]
        alphaCols = ['Index', 'Ratio', 'Value', 'Bitwidth']

        top = self.topOps(k=k)
        for i, layerTop in enumerate(top):
            layerRow = [alphaCols]
            for idx, w, alpha, bitwidth in layerTop:
                alphaRow = [idx, '{:.5f}'.format(w), '{:.5f}'.format(alpha), bitwidth]
                # add alpha data row to layer data table
                layerRow.append(alphaRow)
            # add layer data table to model table as row
            rows.append([i, layerRow])

        # apply loggers functions
        for f in loggerFuncs:
            f(k, rows)

    def printToFile(self, saveFolder):
        logger = HtmlLogger(saveFolder, 'model')

        layerIdxKey = 'Layer#'
        nFiltersKey = 'Filters#'
        bitwidthsKey = 'Bitwidths'
        filterArchKey = 'Filter Architecture'
        alphasKey = 'Alphas distribution'

        logger.createDataTable('Model architecture', [layerIdxKey, nFiltersKey, bitwidthsKey, filterArchKey])
        for layerIdx, layer in enumerate(self.layersList):
            bitwidths = layer.getAllBitwidths()

            dataRow = {layerIdxKey: layerIdx, nFiltersKey: layer.nFilters(), bitwidthsKey: bitwidths, filterArchKey: next(layer.opsList())}
            logger.addDataRow(dataRow)

        # log layers alphas distribution
        self.logDominantQuantizedOp(len(bitwidths), loggerFuncs=[lambda k, rows: logger.addInfoTable(alphasKey, rows)])

    def logForwardCounters(self, loggerFuncs):
        if (not loggerFuncs) or (len(loggerFuncs) == 0):
            self.resetForwardCounters()
            return

        rows = [['Layer #', 'Counters']]
        counterCols = ['Prev idx', 'bitwidth', 'Counter']

        for layerIdx, layer in enumerate(self.layersList):
            filter = layer.filters[0]
            # sum counters of all filters by indices
            countersByIndices = [[0] * len(filter.opsForwardCounters[0]) for _ in range(len(filter.opsForwardCounters))]
            for filter in layer.filters:
                for i, counterList in enumerate(filter.opsForwardCounters):
                    for j, counter in enumerate(counterList):
                        countersByIndices[i][j] += counter
                # reset filter counters
                filter.resetOpsForwardCounters()

            # collect layer counters to 2 arrays:
            # counters holds the counters values
            # indices holds the corresponding counter value indices
            counters, indices = [], []
            for i in range(len(countersByIndices)):
                for j in range(len(countersByIndices[0])):
                    counters.append(countersByIndices[i][j])
                    indices.append((i, j))

            # get layer bitwidths
            bitwidths = layer.getAllBitwidths()
            # for each layer, sort counters in descending order
            layerRows = [counterCols]
            countersTotal = 0
            while len(counters) > 0:
                # find max counter and print it
                maxIdx = argmax(counters)
                i, j = indices[maxIdx]

                # add counter as new row
                layerRows.append([i, bitwidths[j], counters[maxIdx]])

                # update countersTotal
                countersTotal += counters[maxIdx]
                # remove max counter from lists
                del counters[maxIdx]
                del indices[maxIdx]

            # add counters total row
            layerRows.append(['Total', '', countersTotal])
            # add layer row to model table
            rows.append([layerIdx, layerRows])

        # apply loggers functions
        for f in loggerFuncs:
            f(rows)

# def __loadStatistics(self, filename):
#     if exists(filename):
#         # stats is a list of dicts per layer
#         stats = loadModel(filename)
#         print('Loading statistics')
#
#         for i, layer in enumerate(self.layersList):
#             # get layer dict
#             layerStats = stats[i]
#             # iterate over layer filters
#             for filter in layer.filters:
#                 # iterate over filter modules
#                 for m in filter.modules():
#                     # create module type as string
#                     moduleType = '{}'.format(type(m))
#                     NICEprefix = "'NICE."
#                     if NICEprefix in moduleType:
#                         moduleType = moduleType.replace(NICEprefix, "'")
#
#                     # check if type string is in dict
#                     if moduleType in layerStats:
#                         # go over dict keys, which is the module variables
#                         for varName in layerStats[moduleType].keys():
#                             v = getattr(m, varName)
#                             # if variable has value in dict, assign it
#                             if v is not None:
#                                 v.data = layerStats[moduleType][varName].data

# # select random alpha
# def chooseRandomPath(self):
#     for l in self.layers:
#         l.chooseRandomPath()

# # layerIdx, alphaIdx meaning: self.layersList[layerIdx].curr_alpha_idx = alphaIdx
# # def choosePathByAlphas(self, layerIdx=None, alphaIdx=None):
# def choosePathByAlphas(self):
#     for l in self.layers:
#         l.choosePathByAlphas()
#
#     if (layerIdx is not None) and (alphaIdx is not None):
#         layer = self.layersList[layerIdx]
#         layer.curr_alpha_idx = alphaIdx

# def evalMode(self):
#     for l in self.layers:
#         l.evalMode()
#
#     # calc bops ratio
#     return self.calcBopsRatio()

# def uniformMode(self):
#     for l in self.layersList:
#         l.uniformMode(self._criterion.baselineBits)
#
#     # calc bops ratio
#     return self.calcBopsRatio()

# def turnOffAlphas(self):
#     for layer in self.layersList:
#         # turn off alphas gradients
#         layer.alphas.requires_grad = False
#
#     self.learnable_alphas = []

# def turnOnAlphas(self):
#     self.learnable_alphas = []
#     for layer in self.layersList:
#         # turn on alphas gradients
#         layer.alphas.requires_grad = True
#         self.learnable_alphas.append(layer.alphas)
#
#         for op in layer.getOps():
#             # turn off noise in op
#             op.noise = False
#
#             ## ==== for tinyNet ====
#             # # set pre & post quantization hooks, from now on we want to quantize these ops
#             # op.register_forward_pre_hook(save_quant_state)
#             # op.register_forward_hook(restore_quant_state)

# # convert current model to discrete, i.e. keep nOpsPerLayer optimal operations per layer
# def toDiscrete(self, nOpsPerLayer=1):
#     for layer in self.layersList:
#         # calc weights from alphas and sort them
#         weights = F.softmax(layer.alphas, dim=-1)
#         _, wIndices = weights.sort(descending=True)
#         # update layer alphas
#         layer.alphas = layer.alphas[wIndices[:nOpsPerLayer]]
#         #        layer.alphas = tensor(tensor(layer.alphas.tolist()).cuda(), requires_grad=True)
#         layer.alphas = tensor(tensor(layer.alphas.tolist()).cuda())
#         # take indices of ops we want to remove from layer
#         wIndices = wIndices[nOpsPerLayer:]
#         # convert to list
#         wIndices = wIndices.tolist()
#         # sort indices ascending
#         wIndices.sort()
#         # remove ops and corresponding bops from layer
#         for w in reversed(wIndices):
#             del layer.ops[w]
#             del layer.bops[w]

# def loadBitwidthWeigths(self, stateDict, MaxBopsBits, bitwidth):
#     # check idx of MaxBopsBits inside bitwidths
#     maxBopsBitsIdx = bitwidth.index(MaxBopsBits)
#     maxBopsStateDict = OrderedDict()
#     opsKey = 'ops.'
#     for key in stateDict.keys():
#         # if operation is for max bops bits idx
#         if opsKey in key:
#             keyOp_num = key.split(opsKey)[1][0]
#             if int(keyOp_num) == maxBopsBitsIdx:
#                 maxBopsKey = key.replace(opsKey + keyOp_num, opsKey + '0')
#                 maxBopsStateDict[maxBopsKey] = stateDict[key]
#         else:
#             maxBopsStateDict[key] = stateDict[key]
#
#     self.load_state_dict(maxBopsStateDict)

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
#
#     # print('totalLoss:[{:.5f}]'.format(totalLoss))
#     return totalLoss

#
#     # logits = self.forward(input)
#     # return self._criterion(logits, target, self.countBops())

# def _loss(self, input, target):
#     # init how many samples per alpha
#     nSamples = self.nSamples
#     # init total loss
#     totalLoss = 0.0
#     # init loss samples list for ALL alphas
#     allLossSamples = []
#     for j, layer in enumerate(self.layersList):
#         # turn off coin toss for this layer
#         layer.alphas.requires_grad = False
#         # init layer alphas gradient
#         layerAlphasGrad = zeros(len(layer.alphas)).cuda()
#         # calc layer alphas softmax
#         probs = F.softmax(layer.alphas, dim=-1)
#
#         for i, alpha in enumerate(layer.alphas):
#             # select the specific alpha in this layer
#             layer.curr_alpha_idx = i
#             # init loss samples list
#             alphaLossSamples = []
#             for _ in range(nSamples):
#                 # forward through some path in model
#                 logits = self(input)
#                 # alphaLoss += self._criterion(logits, target, self.countBops()).detach()
#                 alphaLossSamples.append(self._criterion(logits, target, self.countBops()).detach())
#
#             # add current alpha loss samples to all loss samples list
#             allLossSamples.extend(alphaLossSamples)
#             # calc alpha average loss
#             alphaAvgLoss = sum(alphaLossSamples) / nSamples
#             layerAlphasGrad[i] = alphaAvgLoss
#             # add alpha loss to total loss
#             totalLoss += (alphaAvgLoss * probs[i])
#
#             # calc loss samples variance
#             lossVariance = [((x - alphaAvgLoss) ** 2) for x in alphaLossSamples]
#             lossVariance = sum(lossVariance) / (nSamples - 1)
#             # add alpha loss average to statistics
#             self.stats.containers[self.stats.alphaLossAvgKey][j][i].append(alphaAvgLoss.item())
#             # add alpha loss variance to statistics
#             self.stats.containers[self.stats.alphaLossVarianceKey][j][i].append(lossVariance.item())
#
#         # turn in coin toss for this layer
#         layer.alphas.requires_grad = True
#         # set layer alphas gradient
#         layer.alphas.grad = layerAlphasGrad
#
#         # add gradNorm to statistics
#         self.stats.containers[self.stats.gradNormKey][j].append(layerAlphasGrad.norm().item())
#
#     # average total loss
#     totalLoss /= self.nLayers()
#     # calc all loss samples average
#     nTotalSamples = len(allLossSamples)
#     allLossSamplesAvg = sum(allLossSamples) / nTotalSamples
#     # calc all loss samples variance
#     allLossSamples = [((x - allLossSamplesAvg) ** 2) for x in allLossSamples]
#     allLossSamplesVariance = (sum(allLossSamples) / (nTotalSamples - 1)).item()
#     # add all samples average & loss variance to statistics
#     self.stats.containers[self.stats.allSamplesLossAvgKey][0].append(allLossSamplesAvg)
#     self.stats.containers[self.stats.allSamplesLossVarianceKey][0].append(allLossSamplesVariance)
#
#     # subtract average total loss from every alpha gradient
#     for layer in self.layersList:
#         layer.alphas.grad -= totalLoss
#         # calc layer alphas softmax
#         probs = F.softmax(layer.alphas, dim=-1)
#         # multiply each grad by its probability
#         layer.alphas.grad *= probs
#
#     return totalLoss
