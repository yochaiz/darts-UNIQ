from numpy import array_split
from multiprocessing import Pool
from abc import abstractmethod

from torch.cuda import set_device
from torch.nn import functional as F
from UNIQ.uniq import save_state, restore_state


class ModelReplicator:
    def __init__(self, model, modelClass, args):
        self.gpuIDs = args.gpu
        self.alphaLimit = args.alpha_limit
        self.alphaLimitCounter = args.alpha_limit_counter
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
                # set model criterion to its GPU
                cModel._criterion.cuda()
                # set mode to eval mode
                cModel.eval()
                # remove UNIQ save_state(), restore_state() hooks for all model ops
                for layer in cModel.layersList:
                    for op in layer.getOps():
                        # remove hooks
                        for hook in op.hooks:
                            hook.remove()
                        # clear hooks list
                        op.hooks.clear()
                # add model to replications
                self.replications.append((cModel, gpu))

        # update replications weights + quantize weights
        self.updateModelWeights(model)
        # restore original gpu
        set_device(args.gpu[0])

    # build args for pool.map
    @abstractmethod
    def buildArgs(self, inputPerGPU, targetPerGPU, layersIndicesPerModel):
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
            for layer in cModel.layersList:
                for op in layer.getOps():
                    save_state(op, None)

    def restore_quantize(self):
        print('model replicator restore_quantize()')
        for cModel, _ in self.replications:
            for layer in cModel.layersList:
                for op in layer.getOps():
                    restore_state(op, None, None)

    # Wrapper function per process, i.e. per replication
    def replicationFunc(self, args):
        # calc loss per replication
        result = self.lossPerReplication(args)
        # get model in order to extract the forward counters
        cModel = self.getModel(args)
        # extract forward counters
        counters = [layer.opsForwardCounters.copy() for layer in cModel.layersList]

        return result, counters

    def updateModelWeights(self, model, loggerFuncs=[]):
        # load model state dict
        modelStateDict = model.state_dict()

        # load model weights
        for cModel, _ in self.replications:
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

    def loss(self, model, input, target):
        nCopies = len(self.replications)
        if nCopies > 0:
            # clone input & target to all GPUs
            inputPerGPU = {}
            targetPerGPU = {}
            for id in self.gpuIDs:
                inputPerGPU[id] = input if (id == input.device.index) else input.clone().cuda(id)
                targetPerGPU[id] = target if (id == target.device.index) else target.clone().cuda(id)

            # update model layers alphas optimization status
            optimizeLayerIdx = self.updateLayersAlphaOptimization(model)
            # split layers indices between models
            layersIndicesPerModel = array_split(optimizeLayerIdx, nCopies)

            # copy model alphas
            for cModel, _ in self.replications:
                for cLayer, mLayer in zip(cModel.layersList, model.layersList):
                    cLayer.alphas.data.copy_(mLayer.alphas.data)
                    cLayer.alphas.requires_grad = mLayer.alphas.requires_grad

            args = self.buildArgs(inputPerGPU, targetPerGPU, layersIndicesPerModel)

            with Pool(processes=nCopies, maxtasksperchild=1) as pool:
                results = pool.map(self.replicationFunc, args)

            # separate cModel forward counters from results
            counters = []
            for i, result in enumerate(results):
                counters.append(result[-1])
                results[i] = results[i][0]

            res = self.processResults(model, results)

            # reset model layers forward counters
            for layer in model.layersList:
                layer.resetOpsForwardCounters()
            # sum forward counters
            for modelCounters in counters:
                for layerCounters, mLayer in zip(modelCounters, model.layersList):
                    for i in range(len(mLayer.opsForwardCounters)):
                        for j in range(len(mLayer.opsForwardCounters[i])):
                            mLayer.opsForwardCounters[i][j] += layerCounters[i][j]

            return res
