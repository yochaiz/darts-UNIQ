from multiprocessing import Pool
from abc import abstractmethod
from math import floor

from torch.cuda import set_device
from torch.nn import functional as F


class ModelReplicator:
    def __init__(self, model, modelClass, args):
        self.gpuIDs = args.gpu
        self.alphaLimit = args.alpha_limit
        self.alphaLimitCounter = args.alpha_limit_counter
        # save number of samples
        self.nSamples = args.nSamples
        # init replications list
        self.replications = []

        # create replications
        for gpu in self.gpuIDs:
            for _ in range(args.nCopies):
                # set device to required gpu
                set_device(gpu)
                # create model new instance
                cModel = modelClass(args)
                # set model to cuda on specific GPU
                cModel = cModel.cuda()
                # turn off noise in 1st layer
                layerIdx = 0
                cModel.layersList[layerIdx].turnOffNoise(layerIdx)
                # set model criterion to its GPU
                cModel._criterion.cuda()
                # set mode to eval mode
                cModel.eval()
                # add model to replications
                self.replications.append((cModel, gpu))

        # update replications weights + quantize weights
        self.updateModelWeights(model)
        # restore original gpu
        set_device(args.gpu[0])

    # build args for pool.map
    @abstractmethod
    def buildArgs(self, inputPerGPU, targetPerGPU, nSamplesPerModel):
        raise NotImplementedError('subclasses must override buildArgs()!')

    # get model from args tuple
    @abstractmethod
    def getModel(self, args):
        raise NotImplementedError('subclasses must override getModel()!')

    # calc loss distributed, i.e. for each model replication
    @abstractmethod
    def lossPerReplication(self, args):
        raise NotImplementedError('subclasses must override lossPerReplication()!')

    # process results from all pool processes
    @abstractmethod
    def processResults(self, model, results):
        raise NotImplementedError('subclasses must override processResults()!')

    # quantize all replications ops
    def quantize(self):
        print('model replicator quantize()')
        for cModel, _ in self.replications:
            for layerIdx, layer in enumerate(cModel.layersList):
                layer.quantize(layerIdx)
                # for op in layer.getOps():
                #     save_state(op, None)

    def restore_quantize(self):
        print('model replicator restore_quantize()')
        for cModel, _ in self.replications:
            for layerIdx, layer in enumerate(cModel.layersList):
                layer.unQuantize(layerIdx)
                # for op in layer.getOps():
                #     restore_state(op, None, None)

    # Wrapper function per process, i.e. per replication
    def replicationFunc(self, args):
        # calc loss per replication
        result = self.lossPerReplication(args)
        # get model in order to extract the forward counters
        cModel = self.getModel(args)
        # extract forward counters
        counters = [[filter.opsForwardCounters.copy() for filter in layer.filters] for layer in cModel.layersList]

        return result, counters

    def updateModelWeights(self, model, loggerFuncs=[]):
        assert (model.isQuantized() is True)
        # load model state dict
        modelStateDict = model.state_dict()

        # load model weights
        for cModel, _ in self.replications:
            assert (cModel.layersList[0].quantized is False)
            cModel.load_state_dict(modelStateDict)

        # apply loggers funcs
        for f in loggerFuncs:
            f('Model replications weights have been updated')

    def updateLayersAlphaOptimization(self, model):
        # init list of layers indices we still have to optimize their alphas
        optimizeLayerIdx = []
        # update layers alphas training status, if optimal alpha reached training limit then stop training
        for idx, layer in enumerate(model.layersList):
            if layer.alphas.requires_grad is True:
                # calc layer alphas softmax
                probs = F.softmax(layer.alphas, dim=-1)
                optProb = probs.max().item()

                # update layer limit counter or reset it
                if optProb >= self.alphaLimit:
                    layer.optLimitCounter += 1
                else:
                    layer.optLimitCounter = 0

                # check if counter is enough or not
                if layer.optLimitCounter >= self.alphaLimitCounter:
                    # reset gradient
                    layer.alphas.grad = None
                    # then turn off requires_grad
                    layer.alphas.requires_grad = False
                    # set optimal alpha probability to 1 and the rest to zero
                    optIdx = probs.argmax().item()
                    print('Stopped training alphas in layer [{}]: idx:[{}], prob:[{:.3f}]'.format(idx, optIdx, optProb))
                    layer.alphas.data.fill_(0.0)
                    layer.alphas.data[optIdx] = 1000.0
                else:
                    optimizeLayerIdx.append(idx)

        # update list of learnable alphas
        model.updateLearnableAlphas()

        return optimizeLayerIdx

    @staticmethod
    def splitSamples(nSamples, nCopies):
        # split number of samples between model replications
        nSamplesPerCopy = [floor(nSamples / nCopies)] * nCopies
        # last copy takes the difference due to floor
        nSamplesPerCopy[-1] = nSamples - sum(nSamplesPerCopy[:-1])
        # it might be that last copy took more than one sample compared to other copies, therefore balance the difference between all copies
        for i in range(len(nSamplesPerCopy) - 1):
            if nSamplesPerCopy[-1] - nSamplesPerCopy[i] > 1:
                # move some sample from last copy to some other copy
                nSamplesPerCopy[i] += 1
                nSamplesPerCopy[-1] -= 1

        return nSamplesPerCopy

    def loss(self, model, input, target):
        nCopies = len(self.replications)
        if nCopies > 0:
            # clone input & target to all GPUs
            inputPerGPU = {}
            targetPerGPU = {}
            for id in self.gpuIDs:
                inputPerGPU[id] = input if (id == input.device.index) else input.clone().cuda(id)
                targetPerGPU[id] = target if (id == target.device.index) else target.clone().cuda(id)

            # # update model layers alphas optimization status
            # optimizeLayerIdx = self.updateLayersAlphaOptimization(model)
            # # split layers indices between models
            # layersIndicesPerModel = array_split(optimizeLayerIdx, nCopies)

            # split samples between model copies
            nSamplesPerModel = self.splitSamples(self.nSamples, nCopies)

            # copy model alphas
            for cModel, _ in self.replications:
                for cLayer, mLayer in zip(cModel.layersList, model.layersList):
                    cLayer.alphas.data.copy_(mLayer.alphas.data)
                    cLayer.alphas.requires_grad = mLayer.alphas.requires_grad

            args = self.buildArgs(inputPerGPU, targetPerGPU, nSamplesPerModel)

            with Pool(processes=nCopies, maxtasksperchild=1) as pool:
                results = pool.map(self.replicationFunc, args)

            # separate cModel forward counters from results
            counters = []
            for i, result in enumerate(results):
                counters.append(result[-1])
                results[i] = results[i][0]

            res = self.processResults(model, results)

            # reset model layers forward counters
            model.resetForwardCounters()
            # sum forward counters
            for replicationCounter in counters:
                for layerIdx, layer in enumerate(model.layersList):
                    for filterIdx, filter in enumerate(layer.filters):
                        filterCounter = replicationCounter[layerIdx][filterIdx]
                        for prev_alpha in range(filter.nOpsCopies()):
                            for curr_alpha in range(filter.numOfOps()):
                                filter.opsForwardCounters[prev_alpha][curr_alpha] += filterCounter[prev_alpha][curr_alpha]

            return res
