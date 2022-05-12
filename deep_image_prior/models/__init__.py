from .skip import skip
from .texture_nets import get_texture_nets
from .resnet import ResNet
from .unet import UNet

import torch.nn as nn

def get_net(input_depth, NET_TYPE, pad, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
    if NET_TYPE == 'ResNet':
        # TODO
        net = ResNet(input_depth, 3, 10, 16, 1, nn.BatchNorm2d, False)
    elif NET_TYPE == 'skip':
        net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

    elif NET_TYPE == 'texture_nets':
        net = get_texture_nets(inp=input_depth, ratios = [32, 16, 8, 4, 2, 1], fill_noise=False,pad=pad)

    elif NET_TYPE =='UNet':
        net = UNet(num_input_channels=input_depth, num_output_channels=3, 
                   feature_scale=4, more_layers=0, concat_x=False,
                   upsample_mode=upsample_mode, pad=pad, norm_layer=nn.BatchNorm2d, need_sigmoid=True, need_bias=True)
    elif NET_TYPE == 'identity':
        assert input_depth == 3
        net = nn.Sequential()
    else:
        assert False

    return net


def get_hq_skip_net(input_depth, pad='reflection', upsample_mode='cubic', n_channels=3, act_fun='LeakyReLU', skip_n33d=192, skip_n33u=192, skip_n11=4, num_scales=6, downsample_mode='cubic', decorr_rgb=True, offset_groups=4, offset_type='1x1'):
    """Constructs and returns a skip network with higher quality default settings, including
    deformable convolutions (can be slow, disable with offset_groups=0). Further
    improvements can be seen by setting offset_type to 'full', but then you may have to
    reduce the learning rate of the offset layers to ~1/10 of the rest of the layers. See
    the get_offset_params() and get_non_offset_params() functions to construct the
    parameter groups."""
    net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                        num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                        num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                        upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                        need_sigmoid=True, need_bias=True, decorr_rgb=decorr_rgb, pad=pad, act_fun=act_fun,
                                        offset_groups=offset_groups, offset_type=offset_type)
    return net


def get_offset_params(net):
    """Returns an iterable of parameters of layers that output offsets for deformable
    convolutions (for setting their learning rate lower than the rest).

    Example:
        >>> params = [{'params': get_non_offset_params(net), 'lr': lr},
        >>>           {'params': get_offset_params(net), 'lr': lr / 10}]
        >>> opt = optim.Adam(params)
    """
    return [p for n, p in net.named_parameters() if 'offset_branch' in n]


def get_non_offset_params(net):
    """Returns an iterable of parameters of layers that do not output offsets for
    deformable convolutions.

    Example:
        >>> params = [{'params': get_non_offset_params(net), 'lr': lr},
        >>>           {'params': get_offset_params(net), 'lr': lr / 10}]
        >>> opt = optim.Adam(params)
    """
    return [p for n, p in net.named_parameters() if 'offset_branch' not in n]
