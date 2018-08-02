import math
import torch
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F

sqrt_of_2 = math.sqrt(2)
eps = 1e-5


def my_mean(x):
    size = x.size()
    size_tensor = torch.Tensor([s for s in size])
    elements = torch.prod(size_tensor)
    return torch.sum(x) / elements


def my_std(x):
    size = x.size()
    size_tensor = torch.Tensor([s for s in size])
    elements = torch.FloatTensor([torch.prod(size_tensor)])
    x_min_mean_sq = (x - my_mean(x)) * (x - my_mean(x))
    std = torch.sqrt(torch.sum(x_min_mean_sq) / (elements - 1))
    return std[0]


def norm_cdf(x, mean, std):
    return 1. / 2 * (1. + torch.erf((x - mean) / (std * sqrt_of_2)))


def norm_icdf(x, mean, std):
    return mean + std * sqrt_of_2 * torch.erfinv(2. * x - 1)


def add_noise(modules, training=False, bitwidth=32, noise=True, high_noise=False):
    for m in modules:
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.LSTM):
            for p in m._parameters:
                if m._parameters[p] is not None:
                    d = m._parameters[p].device

                    mean_p = torch.mean(m._parameters[p].data)

                    std_p = max(torch.std(m._parameters[p].data), eps)

                    y_p = norm_cdf(m._parameters[p].data, mean_p, std_p)

                    noise_step = 1. / (2 ** (bitwidth + 1)) if not high_noise else 1. / (2 ** (bitwidth))
                    noise = y_p.clone().uniform_(-noise_step, noise_step)
                    y_out_p = norm_icdf(torch.clamp(y_p + noise, eps, 1 - eps), mean_p, std_p)

                    m._parameters[p].data = y_out_p.to(d)


def quantize(modules, bitwidth=32):
    if bitwidth > 16:
        return  # No quantization for high bitwidths
    for m in modules:
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.LSTM):
            for p in m._parameters:
                if m._parameters[p] is not None:
                    d = m._parameters[p].device
                    mean_p = torch.mean(m._parameters[p].data)
                    std_p = max(torch.std(m._parameters[p].data), eps)
                    y_p = norm_cdf(m._parameters[p].data, mean_p, std_p)
                    m._parameters[p].data = norm_icdf(
                        torch.clamp((torch.round(y_p * 2 ** bitwidth - 0.5) + 0.5) / 2 ** bitwidth,
                                    1. / 2 ** (bitwidth + 1), 1. - 1. / 2 ** (bitwidth + 1)),
                        mean_p, std_p).to(d)


def act_quantize(x, bitwidth=32):
    if bitwidth > 16:
        return F.relu(x)  # No quantization for high bitwidths

    mean = torch.mean(x.data)
    std = torch.std(x.data)
    cdf = norm_cdf(x.data, mean, std)
    low_bound = norm_cdf(torch.cuda.FloatTensor([0]), mean, std)[0]
    indexes_of_zero = cdf < low_bound
    cdf[indexes_of_zero] = low_bound  # ReLU
    num_of_bins = 2 ** bitwidth - 1  # one for 0

    mapped_interval = (cdf - low_bound) / (1. - low_bound)

    quantized_values = (torch.round(mapped_interval * num_of_bins - 0.5) + 0.5) / num_of_bins
    back_mapped = quantized_values * (1. - low_bound) + low_bound
    back_mapped[indexes_of_zero] = low_bound  # ReLU
    clipped_values = torch.clamp(back_mapped, 1. / 2 ** (bitwidth + 1), 1. - 1. / 2 ** (bitwidth + 1))
    norm = norm_icdf(clipped_values, mean, std)

    return Variable(norm, requires_grad=False).to(x.device)


def act_noise(x, training=False, bitwidth=32, noise=True, high_noise=False):
    act_mean = torch.mean(x.data)
    act_std = max(torch.std(x.data), eps)
    act_cdf = norm_cdf(x.data, act_mean, act_std)

    low_bound = norm_cdf(torch.cuda.FloatTensor([0]), act_mean, act_std)[0]
    indexes_of_zero = act_cdf < low_bound
    act_cdf[indexes_of_zero] = low_bound  # ReLU
    num_of_bins = 2 ** bitwidth - 1  # one for 0

    noise_step = (1. - low_bound) / (2 * num_of_bins) if not high_noise else (1. - low_bound) / num_of_bins
    noise = act_cdf.clone().uniform_(-noise_step, noise_step)
    noise[indexes_of_zero] = 0.
    clipped_values = torch.clamp(act_cdf + noise, 1. / 2 ** (bitwidth + 1), 1. - 1. / 2 ** (bitwidth + 1))
    res = Variable(norm_icdf(clipped_values, act_mean, act_std), requires_grad=False)
    return res.to(x.device)


def backup_weights(modules, bk):
    for m in modules:
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.LSTM):
            for p in m._parameters:
                if m._parameters[p] is not None:
                    bk[(m, p)] = m._parameters[p].data.clone()

    return bk


def restore_weights(modules, bk):
    for m in modules:
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.LSTM):
            for p in m._parameters:
                if m._parameters[p] is not None:
                    m._parameters[p].data = bk[(m, p)].clone()


def init_params(layers):
    for layer in layers:
        for param in layer.parameters():
            if param.data.dim() >= 2:
                init.kaiming_normal(param.data)
            else:
                param.data = param.data.clone().uniform_(-0.1, 0.1)
