from collections import OrderedDict

from torch.nn import Sequential
from torch import load as loadModel

from cnn.MixedOp import MixedConvWithReLU, MixedLinear
from cnn.models import BaseNet


class TinyNet(BaseNet):
    def __init__(self, crit, bitwidths, kernel_sizes, bopsFuncKey):
        super(TinyNet, self).__init__(criterion=crit, initLayersParams=(bitwidths, kernel_sizes),
                                      bopsFuncKey=bopsFuncKey)

        # init layers permutation list
        self.layersPerm = []
        # init number of permutations counter
        self.nPerms = 1

        for layer in self.layersList:
            # turn on noise
            for op in layer.ops:
                op.noise = op.quant

            # turn on alphas gradients
            layer.alphas.requires_grad = True
            self.learnable_alphas.append(layer.alphas)

            # add layer numOps range to permutation list
            self.layersPerm.append(list(range(len(layer.alphas))))
            self.nPerms *= len(layer.alphas)

        self.nLayersQuantCompleted = 0

    def initLayers(self, params):
        bitwidths, kernel_sizes = params

        # self.features = nn.Sequential(
        #     MixedConv(bitwidths, 3, 16, kernel_sizes, stride=2), nn.ReLU(inplace=True),
        #     MixedConv(bitwidths, 16, 32, kernel_sizes, stride=2), nn.ReLU(inplace=True),
        #     MixedConv(bitwidths, 32, 64, kernel_sizes, stride=2), nn.ReLU(inplace=True),
        #     MixedConv(bitwidths, 64, 128, kernel_sizes, stride=2), nn.ReLU(inplace=True)
        # )
        self.features = Sequential(
            MixedConvWithReLU(bitwidths, 3, 16, kernel_sizes, stride=2),
            MixedConvWithReLU(bitwidths, 16, 32, kernel_sizes, stride=2),
            MixedConvWithReLU(bitwidths, 32, 64, kernel_sizes, stride=2),
            MixedConvWithReLU(bitwidths, 64, 128, kernel_sizes, stride=2),
        )
        self.fc = MixedLinear(bitwidths, 512, 10)

        # self.features = nn.Sequential(
        #     nn.Conv2d(3, 16, 3, 2, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 128, 3, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        # )
        # self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)

    def switch_stage(self, logger=None):
        pass

    # load original pre_trained model of UNIQ
    def loadUNIQPre_trained(self, path, logger, gpu):
        map = {}
        map['features.0'] = 'features.0.ops.0.op.0.0'
        map['features.1'] = 'features.0.ops.0.op.0.1'

        map['features.3'] = 'features.1.ops.0.op.0.0'
        map['features.4'] = 'features.1.ops.0.op.0.1'

        map['features.6'] = 'features.2.ops.0.op.0.0'
        map['features.7'] = 'features.2.ops.0.op.0.1'

        map['features.9'] = 'features.3.ops.0.op.0.0'
        map['features.10'] = 'features.3.ops.0.op.0.1'

        map['fc'] = 'fc.ops.0.op'

        checkpoint = loadModel(path, map_location=lambda storage, loc: storage.cuda(gpu))
        chckpntDict = checkpoint['state_dict']
        newStateDict = OrderedDict()

        # d = self.state_dict()

        token = '.ops.'
        for key in chckpntDict.keys():
            if 'num_batches_tracked' in key:
                continue

            prefix = key[:key.rindex('.')]
            suffix = key[key.rindex('.'):]
            newKey = map[prefix]
            # find new key layer
            newKeyOp = newKey[:newKey.index(token)]
            # init path to layer
            layerPath = [p for p in newKeyOp.split('.')]
            # get layer by walking through path
            layer = self
            for p in layerPath:
                layer = getattr(layer, p)
            # update layer ops
            for i in range(len(layer.ops)):
                newStateDict[newKey + suffix] = chckpntDict[key]
                newKey = newKey.replace(newKeyOp + token + '{}.'.format(i), newKeyOp + token + '{}.'.format(i + 1))

        # load model weights
        self.load_state_dict(newStateDict)

        logger.info('Loaded model from [{}]'.format(path))
        logger.info('checkpoint validation accuracy:[{:.5f}]'.format(checkpoint['best_prec1']))
