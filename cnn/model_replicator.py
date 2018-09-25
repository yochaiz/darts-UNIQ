from numpy import array_split
from multiprocessing import Pool
from abc import abstractmethod


class ModelReplicator:
    def __init__(self, model, modelClass, args):
        self.gpuIDs = args.gpu
        # init replications list
        self.replications = []
        # count number of replications and assign each of them to a GPU
        gpus = [gpu for gpu in args.gpu for _ in range(args.nCopies)]
        # load model state dict
        modelStateDict = model.state_dict()
        # create replications
        for gpu in gpus:
            # create model new instance
            cModel = modelClass(args)
            # set model to cuda on specific GPU
            cModel = cModel.cuda(gpu)
            # set model criterion to its GPU
            cModel._criterion.cuda(gpu)
            cModel._criterion.search_loss.cuda(gpu)
            # load model weights
            cModel.load_state_dict(modelStateDict)
            # add model to replications
            self.replications.append((cModel, gpu))

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

    # Wrapper function per process, i.e. per replication
    def replicationFunc(self, args):
        # calc loss per replication
        result = self.lossPerReplication(args)
        # get model in order to extract the forward counters
        cModel = self.getModel(args)
        # extract forward counters
        counters = [layer.opsForwardCounters for layer in cModel.layersList]

        return result, counters

    def updateModelWeights(self, model):
        # load model state dict
        modelStateDict = model.state_dict()

        # load model weights
        for cModel, _ in self.replications:
            cModel.load_state_dict(modelStateDict)

    def loss(self, model, input, target):
        nCopies = len(self.replications)
        if nCopies > 0:
            print('model replication loss()')
            # clone input & target to all GPUs
            inputPerGPU = {}
            targetPerGPU = {}
            for id in self.gpuIDs:
                inputPerGPU[id] = input if (id == input.device.index) else input.clone().cuda(id)
                targetPerGPU[id] = target if (id == target.device.index) else target.clone().cuda(id)

            # split layers indices between models
            layersIndicesPerModel = array_split(range(model.nLayers()), nCopies)

            # copy model alphas
            for cModel, _ in self.replications:
                for cLayer, mLayer in zip(cModel.layersList, model.layersList):
                    cLayer.alphas.data.copy_(mLayer.alphas.data)
                    cLayer.alphas.requires_grad = mLayer.alphas.requires_grad

            args = self.buildArgs(inputPerGPU, targetPerGPU, layersIndicesPerModel)
            print('built args')

            with Pool(processes=nCopies, maxtasksperchild=1) as pool:
                # results = pool.map(self.replicationFunc, args)
                results = pool.map(self.lossPerReplication, args)

            # # separate cModel forward counters from results
            # counters = []
            # for i, result in enumerate(results):
            #     counters.append(result[-1])
            #     results[i] = results[i][0]

            print('pool ended, pre processResults()')
            res = self.processResults(model, results)
            print('post processResults()')
            print ('===================')

            # # reset model layers forward counters
            # for layer in model.layersList:
            #     layer.resetOpsForwardCounters()
            # # sum forward counters
            # for modelCounters in counters:
            #     for layerCounters, mLayer in zip(modelCounters, model.layersList):
            #         for i in range(len(mLayer.opsForwardCounters)):
            #             for j in range(len(mLayer.opsForwardCounters[i])):
            #                 mLayer.opsForwardCounters[i][j] += layerCounters[i][j]

            return res
