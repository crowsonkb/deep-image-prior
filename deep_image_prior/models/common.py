import numpy as np
import torch
import torch.nn as nn
from torchvision import ops

from .updown import Downsample2d, Upsample2d, _kernels

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)
    
torch.nn.Module.add = add_module

class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]        

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs: 
                diff2 = (inp.size(2) - target_shape2) // 2 
                diff3 = (inp.size(3) - target_shape3) // 2 
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


class GenNoise(nn.Module):
    def __init__(self, dim2):
        super(GenNoise, self).__init__()
        self.dim2 = dim2

    def forward(self, input):
        a = list(input.size())
        a[1] = self.dim2
        # print (input.data.type())

        b = torch.zeros(a).type_as(input.data)
        b.normal_()

        x = torch.autograd.Variable(b)

        return x


class Swish(nn.Module):
    """
        https://arxiv.org/abs/1710.05941
        The hype was so huge that I could not help but try it
    """
    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)


class DeformConvWrapper(nn.Module):
    """Wraps a deformable convolution layer and the branch that produces its offset map."""
    def __init__(self, main_layer, offset_branch, input_branch=None):
        super().__init__()
        self.main_layer = main_layer
        self.offset_branch = offset_branch
        self.input_branch = input_branch if input_branch is not None else nn.Identity()

    def forward(self, input):
        offsets = self.offset_branch(input)
        input = self.input_branch(input)
        return self.main_layer(input, offsets)


class DecorrelatedColorsToRGB(nn.Module):
    """Converts from a decorrelated color space to RGB. See
    https://github.com/eps696/aphantasia/blob/master/aphantasia/image.py. Usually intended
    to be followed by a sigmoid.
    """
    def __init__(self, inv_color_scale=1.6):
        super().__init__()
        color_correlation_svd_sqrt = torch.tensor([[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]])
        color_correlation_svd_sqrt /= torch.tensor([inv_color_scale, 1., 1.])  # saturate, empirical
        max_norm_svd_sqrt = color_correlation_svd_sqrt.norm(dim=0).max()
        color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt
        self.register_buffer('colcorr_t', color_correlation_normalized.T)

    def inverse(self, image):
        colcorr_t_inv = torch.linalg.inv(self.colcorr_t)
        return torch.einsum('nchw,cd->ndhw', image, colcorr_t_inv)

    def forward(self, image):
        return torch.einsum('nchw,cd->ndhw', image, self.colcorr_t)


def act(act_fun = 'LeakyReLU'):
    '''
        Either string defining an activation function or module (e.g. nn.ReLU)
    '''
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'Swish':
            return Swish()
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        else:
            assert False
    else:
        return act_fun()


def bn(num_features):
    return nn.BatchNorm2d(num_features)


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride', offset_groups=0, offset_type='1x1'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode in _kernels and stride == 2:
            downsampler = Downsample2d(downsample_mode)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    if kernel_size == 1 or not offset_groups or not offset_type or offset_type == 'none':
        convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)
    else:
        while offset_groups > 1 and in_f % offset_groups != 0:
            offset_groups -= 1
        main_layer = ops.DeformConv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)
        if offset_type == '1x1':
            offset_layer = nn.Conv2d(in_f, 2 * offset_groups * kernel_size * kernel_size, 1, stride, padding=to_pad)
            convolver = DeformConvWrapper(main_layer, offset_layer, padder)
            padder = None
        elif offset_type == 'full':
            offset_layer = nn.Conv2d(in_f, 2 * offset_groups * kernel_size * kernel_size, kernel_size, stride, padding=to_pad)
            convolver = DeformConvWrapper(main_layer, offset_layer)
        else:
            assert False

    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)


def up(mode):
    if mode == 'nearest':
        return nn.Upsample(scale_factor=2, mode=mode)
    if mode == 'bilinear':
        return nn.Upsample(scale_factor=2, mode=mode, align_corners=False)
    if mode in _kernels:
        return Upsample2d(mode)
    assert False
