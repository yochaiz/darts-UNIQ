
import torch
import torch.nn as nn
from operations import *
from cnn.operations import OPS
import torch.nn.functional as F
from utils import drop_path
from collections import namedtuple


class Build_discrete_model(nn.Module):
    def __init__(self, model_search,auxiliary):
        super(Build_discrete_model, self).__init__()
        self._layers = model_search._layers
        self._auxiliary = auxiliary

        stem_multiplier = 3
        C_curr = stem_multiplier * model_search._C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, model_search._C

        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(self._layers):
            if i in [self._layers // 3, 2 * self._layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = DiscreteCell(model_search, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * self._layers // 3:
                C_to_auxiliary = C_prev

        self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, model_search._num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, model_search._num_classes)



    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux





class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x

class DiscreteCell(nn.Module):
    def __init__(self, model_search, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(DiscreteCell, self).__init__()
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        genotype = self._build_discrete_genotype(model_search)
        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices


    def _build_discrete_genotype(self, model_search):
        def _parse(steps, weights):
            gene = []
            n = 2
            start = 0
            for i in range(steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != list(OPS.keys()).index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != list(OPS.keys()).index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((list(OPS.keys())[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(model_search._steps, F.softmax(model_search.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(model_search._steps, F.softmax(model_search.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + model_search._steps - model_search._multiplier, model_search._steps + 2)
        Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype


    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)








