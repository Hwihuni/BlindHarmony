import os
import time

import numpy as np
import scipy.misc
import sacred

import torch
from torch import nn

from sacred import Experiment, observers
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments import autils
from experiments.autils import Conv2dSameSize, LogProbWrapper
from experiments.images_data import get_data, Preprocess

from data import load_num_batches
from torchvision.utils import make_grid, save_image

from nde import distributions, transforms, flows
import utils
import optim
import nn as nn_

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

class ConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.net = nn.Sequential(
            Conv2dSameSize(in_channels, hidden_channels, kernel_size=3),
            nn.ReLU(),
            Conv2dSameSize(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            Conv2dSameSize(hidden_channels, out_channels, kernel_size=3),
        )

    def forward(self, inputs, context=None):
        return self.net.forward(inputs)

def create_transform_step(num_channels,hidden_channels, config):
    actnorm = config.actnorm
    coupling_layer_type = config.coupling_layer_type
    spline_params = config.spline_params
    use_resnet = config.use_resnet
    num_res_blocks = config.num_res_blocks
    resnet_batchnorm = config.resnet_batchnorm
    dropout_prob = config.dropout_prob
    if use_resnet:
        def create_convnet(in_channels, out_channels):
            net = nn_.ConvResidualNet(in_channels=in_channels,
                                      out_channels=out_channels,
                                      hidden_channels=hidden_channels,
                                      num_blocks=num_res_blocks,
                                      use_batch_norm=resnet_batchnorm,
                                      dropout_probability=dropout_prob)
            return net
    else:
        if dropout_prob != 0.:
            raise ValueError()
        def create_convnet(in_channels, out_channels):
            return ConvNet(in_channels, hidden_channels, out_channels)

    mask = utils.create_mid_split_binary_mask(num_channels)

    if coupling_layer_type == 'cubic_spline':
        coupling_layer = transforms.PiecewiseCubicCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet,
            tails='linear',
            tail_bound=spline_params['tail_bound'],
            num_bins=spline_params['num_bins'],
            apply_unconditional_transform=spline_params['apply_unconditional_transform'],
            min_bin_width=spline_params['min_bin_width'],
            min_bin_height=spline_params['min_bin_height']
        )
    elif coupling_layer_type == 'quadratic_spline':
        coupling_layer = transforms.PiecewiseQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet,
            tails='linear',
            tail_bound=spline_params['tail_bound'],
            num_bins=spline_params['num_bins'],
            apply_unconditional_transform=spline_params['apply_unconditional_transform'],
            min_bin_width=spline_params['min_bin_width'],
            min_bin_height=spline_params['min_bin_height']
        )
    elif coupling_layer_type == 'rational_quadratic_spline':
        coupling_layer = transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet,
            tails='linear',
            tail_bound=spline_params['tail_bound'],
            num_bins=spline_params['num_bins'],
            apply_unconditional_transform=spline_params['apply_unconditional_transform'],
            min_bin_width=spline_params['min_bin_width'],
            min_bin_height=spline_params['min_bin_height'],
            min_derivative=spline_params['min_derivative']
        )
    elif coupling_layer_type == 'affine':
        coupling_layer = transforms.AffineCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet
        )
    elif coupling_layer_type == 'additive':
        coupling_layer = transforms.AdditiveCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet
        )
    else:
        raise RuntimeError('Unknown coupling_layer_type')

    step_transforms = []

    if actnorm:
        step_transforms.append(transforms.ActNorm(num_channels))

    step_transforms.extend([
        transforms.OneByOneConvolution(num_channels),
        coupling_layer
    ])

    return transforms.CompositeTransform(step_transforms)


def create_transform(c, h, w, config):
    levels = config.levels
    hidden_channels = config.hidden_channels
    steps_per_level = config.steps_per_level
    alpha = config.alpha
    num_bits = config.num_bits
    preprocessing = config.preprocessing
    multi_scale = config.multi_scale
    if not isinstance(hidden_channels, list):
        hidden_channels = [hidden_channels] * levels

    if multi_scale:
        mct = transforms.MultiscaleCompositeTransform(num_transforms=levels)
        for level, level_hidden_channels in zip(range(levels), hidden_channels):
            squeeze_transform = transforms.SqueezeTransform()
            c, h, w = squeeze_transform.get_output_shape(c, h, w)

            transform_level = transforms.CompositeTransform(
                [squeeze_transform]
                + [create_transform_step(c, level_hidden_channels, config) for _ in range(steps_per_level)]
                + [transforms.OneByOneConvolution(c)] # End each level with a linear transformation.
            )

            new_shape = mct.add_transform(transform_level, (c, h, w))
            if new_shape:  # If not last layer
                c, h, w = new_shape
    else:
        all_transforms = []

        for level, level_hidden_channels in zip(range(levels), hidden_channels):
            squeeze_transform = transforms.SqueezeTransform()
            c, h, w = squeeze_transform.get_output_shape(c, h, w)

            transform_level = transforms.CompositeTransform(
                [squeeze_transform]
                + [create_transform_step(c, level_hidden_channels, config) for _ in range(steps_per_level)]
                + [transforms.OneByOneConvolution(c)] # End each level with a linear transformation.
            )
            all_transforms.append(transform_level)

        all_transforms.append(transforms.ReshapeTransform(
            input_shape=(c,h,w),
            output_shape=(c*h*w,)
        ))
        mct = transforms.CompositeTransform(all_transforms)

    # Inputs to the model in [0, 2 ** num_bits]

    if preprocessing == 'glow':
        # Map to [-0.5,0.5]
        preprocess_transform = transforms.AffineScalarTransform(scale=(1. / 2 ** num_bits),
                                                                shift=-0.5)
    elif preprocessing == 'realnvp':
        preprocess_transform = transforms.CompositeTransform([
            # Map to [0,1]
            transforms.AffineScalarTransform(scale=(1. / 2 ** num_bits)),
            # Map into unconstrained space as done in RealNVP
            transforms.AffineScalarTransform(shift=alpha,
                                             scale=(1 - alpha)),
            transforms.Logit()
        ])

    elif preprocessing == 'realnvp_2alpha':
        preprocess_transform = transforms.CompositeTransform([
            transforms.AffineScalarTransform(scale=(1. / 2 ** num_bits)),
            transforms.AffineScalarTransform(shift=alpha,
                                             scale=(1 - 2. * alpha)),
            transforms.Logit()
        ])
    else:
        raise RuntimeError('Unknown preprocessing type: {}'.format(preprocessing))

    return transforms.CompositeTransform([preprocess_transform, mct])

def create_flow(c, h, w,flow_checkpoint, config):
    distribution = distributions.StandardNormal((c * h * w,))
    transform = create_transform(c, h, w,config)

    flow = flows.Flow(transform, distribution)

    print('There are {} trainable parameters in this model.'.format(
        utils.get_num_parameters(flow)))

    if flow_checkpoint is not None:
        flow.load_state_dict(torch.load(flow_checkpoint))
        print('Flow state loaded from {}'.format(flow_checkpoint))
    return flow