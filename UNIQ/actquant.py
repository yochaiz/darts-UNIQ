import torch.nn as nn
from UNIQ.quantize import act_quantize, act_noise
import torch.nn.functional as F


class ActQuant(nn.Module):

    def __init__(self, quatize_during_training=False, noise_during_training=False, quant=False, noise=False, bitwidth=32):
        super(ActQuant, self).__init__()
        self.quant = quant
        self.noise = noise
        assert (isinstance(bitwidth, int))
        self.bitwidth = bitwidth
        self.quatize_during_training = quatize_during_training
        self.noise_during_training = noise_during_training

    def update_stage(self, quatize_during_training=False, noise_during_training=False):
        self.quatize_during_training = quatize_during_training
        self.noise_during_training = noise_during_training

    def forward(self, input):

        if self.quant and (not self.training or (self.training and self.quatize_during_training)):
            x = act_quantize(input, bitwidth=self.bitwidth)
        elif self.noise and self.training and self.noise_during_training:
            x = act_noise(input, bitwidth=self.bitwidth, training=self.training)
        else:
            x = F.relu(input)

        return x
