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

    # calc loss distributed, i.e. for each model replication
    @abstractmethod
    def lossPerReplication(self, args):
        raise NotImplementedError('subclasses must override lossPerReplication()!')

    # process results from all pool processes
    @abstractmethod
    def processResults(self, model, results):
        raise NotImplementedError('subclasses must override processResults()!')

    def loss(self, model, input, target):
        nCopies = len(self.replications)
        if nCopies > 0:
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

            with Pool(processes=nCopies, maxtasksperchild=1) as pool:
                results = pool.map(self.lossPerReplication, args)

            res = self.processResults(model, results)
            return res
