import os
import time
import copy
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from PIL import Image
import cv2
import torchvision.transforms as transforms
import torchvision
from scipy.ndimage import zoom
from scipy.special import logsumexp
from collections import OrderedDict
import torchvision
from torchvision import transforms
from torchsummary import summary
import torch.nn.functional as F
import functools
import math


class FeatureExtractor(torch.nn.Module):
    """Defines the backbone architecture of the model"""
    def __init__(self, features, targets):
        """Features refers to the bacbone while the targets are the layers of interest"""
        super().__init__()
        self.features = features
        self.targets = targets
        self.outputs = {}

        for target in targets:
            layer = dict([*self.features.named_modules()])[target]
            layer.register_forward_hook(self.save_outputs_hook(target))

    def save_outputs_hook(self, layer_id: str):
        def fn(_, __, output):
            self.outputs[layer_id] = output.clone()
        return fn

    def forward(self, x):

        self.outputs.clear()
        self.features(x)
        return [self.outputs[target] for target in self.targets]
    

def upscale(tensor, size):
    """
    Function that upsamples the map from the readout layer back to the size of the input
    """
    tensor_size = torch.tensor(tensor.shape[2:]).type(torch.float32)
    target_size = torch.tensor(size).type(torch.float32)
    factors = torch.ceil(target_size / tensor_size)
    factor = torch.max(factors).type(torch.int64).to(tensor.device)
    assert factor >= 1

    tensor = torch.repeat_interleave(tensor, factor, dim=2)
    tensor = torch.repeat_interleave(tensor, factor, dim=3)

    tensor = tensor[:, :, :size[0], :size[1]]

    return tensor


def gaussian_filter_1d(tensor, dim, sigma, truncate=4, kernel_size=None, padding_mode='nearest', padding_value=0.0):
    """
    Gaussian filter that is to be added to the the output of the finalizer network
    """
    sigma = torch.as_tensor(sigma, device=tensor.device, dtype=tensor.dtype)

    if kernel_size is not None:
        kernel_size = torch.as_tensor(kernel_size, device=tensor.device, dtype=torch.int64)
    else:
        kernel_size = torch.as_tensor(2 * torch.ceil(truncate * sigma) + 1, device=tensor.device, dtype=torch.int64)

    kernel_size = kernel_size.detach()

    kernel_size_int = kernel_size.detach().cpu().numpy()

    mean = (torch.as_tensor(kernel_size, dtype=tensor.dtype) - 1) / 2

    grid = torch.arange(kernel_size, device=tensor.device) - mean

    kernel_shape = (1, 1, kernel_size)
    grid = grid.view(kernel_shape)

    grid = grid.detach()

    source_shape = tensor.shape

    tensor = torch.movedim(tensor, dim, len(source_shape)-1)
    dim_last_shape = tensor.shape
    assert tensor.shape[-1] == source_shape[dim]

    # we need reshape instead of view for batches like B x C x H x W
    tensor = tensor.reshape(-1, 1, source_shape[dim])

    padding = (math.ceil((kernel_size_int - 1) / 2), math.ceil((kernel_size_int - 1) / 2))
    tensor_ = F.pad(tensor, padding, padding_mode, padding_value)

    # create gaussian kernel from grid using current sigma
    kernel = torch.exp(-0.5 * (grid / sigma) ** 2)
    kernel = kernel / kernel.sum()

    # convolve input with gaussian kernel
    tensor_ = F.conv1d(tensor_, kernel)
    tensor_ = tensor_.view(dim_last_shape)
    tensor_ = torch.movedim(tensor_, len(source_shape)-1, dim)

    assert tensor_.shape == source_shape

    return tensor_


class GaussianFilterNd(nn.Module):
    """A differentiable gaussian filter"""

    def __init__(self, dims, sigma, truncate=4, kernel_size=None, padding_mode='nearest', padding_value=0.0,
                 trainable=False):
        """
        The gaussain id filter is added to the model
        Creates a 1d gaussian filter
        Args:
            dims ([int]): the dimensions to which the gaussian filter is applied. Negative values won't work
            sigma (float): standard deviation of the gaussian filter (blur size)
            input_dims (int, optional): number of input dimensions ignoring batch and channel dimension,
                i.e. use input_dims=2 for images (default: 2).
            truncate (float, optional): truncate the filter at this many standard deviations (default: 4.0).
                This has no effect if the `kernel_size` is explicitely set
            kernel_size (int): size of the gaussian kernel convolved with the input
            padding_mode (string, optional): Padding mode implemented by `torch.nn.functional.pad`.
            padding_value (string, optional): Value used for constant padding.
        """
        # IDEA determine input_dims dynamically for every input
        super(GaussianFilterNd, self).__init__()

        self.dims = dims
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32), requires_grad=trainable)  # default: no optimization
        self.truncate = truncate
        self.kernel_size = kernel_size

        # setup padding
        self.padding_mode = padding_mode
        self.padding_value = padding_value

    def forward(self, tensor):
        """Apply the gaussian filter to the given tensor"""
        for dim in self.dims:
            tensor = gaussian_filter_1d(
                tensor,
                dim=dim,
                sigma=self.sigma,
                truncate=self.truncate,
                kernel_size=self.kernel_size,
                padding_mode=self.padding_mode,
                padding_value=self.padding_value,
            )

        return tensor


class Finalizer(nn.Module):
    """Transforms a readout into a gaze prediction
    A readout network returns a single, spatial map of probable gaze locations.
    This module bundles the common processing steps necessary to transform this into
    the predicted gaze distribution:
     - resizing to the stimulus size
     - smoothing of the prediction using a gaussian filter
     - removing of channel and time dimension
     - weighted addition of the center bias
     - normalization
    """

    def __init__(
        self,
        sigma,
        kernel_size=None,
        learn_sigma=False,
        center_bias_weight=1.0,
        learn_center_bias_weight=True,
        saliency_map_factor=4,
    ):
        """Creates a new finalizer
        Args:
            size (tuple): target size for the predictions
            sigma (float): standard deviation of the gaussian kernel used for smoothing
            kernel_size (int, optional): size of the gaussian kernel
            learn_sigma (bool, optional): If True, the standard deviation of the gaussian kernel will
                be learned (default: False)
            center_bias (string or tensor): the center bias
            center_bias_weight (float, optional): initial weight of the center bias
            learn_center_bias_weight (bool, optional): If True, the center bias weight will be
                learned (default: True)
        """
        super(Finalizer, self).__init__()

        self.saliency_map_factor = saliency_map_factor

        self.gauss = GaussianFilterNd([2, 3], sigma, truncate=3, trainable=learn_sigma)
        self.center_bias_weight = nn.Parameter(torch.Tensor([center_bias_weight]), requires_grad=learn_center_bias_weight)

    def forward(self, readout, centerbias):
        """Applies the finalization steps to the given readout"""

        downscaled_centerbias = F.interpolate(
            centerbias.view(centerbias.shape[0], 1, centerbias.shape[1], centerbias.shape[2]),
            scale_factor=1 / self.saliency_map_factor,
            recompute_scale_factor=False,
        )[:, 0, :, :]

        out = F.interpolate(
            readout,
            size=[downscaled_centerbias.shape[1], downscaled_centerbias.shape[2]]
        )

        # apply gaussian filter
        out = self.gauss(out)

        # remove channel dimension
        out = out[:, 0, :, :]

        # add to center bias
        out = out + self.center_bias_weight * downscaled_centerbias

        out = F.interpolate(out[:, np.newaxis, :, :], size=[centerbias.shape[1], centerbias.shape[2]])[:, 0, :, :]

        # normalize
        out = out - out.logsumexp(dim=(1, 2), keepdim=True)

        return out
    
    
    
    class LayerNorm(nn.Module):
        """Applies Layer Normalization over a mini-batch of inputs as described in
        the paper `Layer Normalization`_ .
        .. math::
            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
        The mean and standard-deviation are calculated separately over the last
        certain number dimensions which have to be of the shape specified by
        :attr:`normalized_shape`.
        :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
        :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
        .. note::
            Unlike Batch Normalization and Instance Normalization, which appliesDEEPGAZE
            
            scalar scale and bias for each entire channel/plane with the
            :attr:`affine` option, Layer Normalization applies per-element scale and
            bias with :attr:`elementwise_affine`.
        This layer uses statistics computed from input data in both training and
        evaluation modes.
        Args:
            normalized_shape (int or list or torch.Size): input shape from an expected input
                of size
                .. math::
                    [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                        \times \ldots \times \text{normalized\_shape}[-1]]
                If a single integer is used, it is treated as a singleton list, and this module will
                normalize over the last dimension which is expected to be of that specific size.
            eps: a value added to the denominator for numerical stability. Default: 1e-5
            elementwise_affine: a boolean value that when set to ``True``, this module
                has learnable per-element affine parameters initialized to ones (for weights)
                and zeros (for biases). Default: ``True``.
        Shape:
            - Input: :math:`(N, *)`
            - Output: :math:`(N, *)` (same shape as input)
        Examples::
            >>> input = torch.randn(20, 5, 10, 10)
            >>> # With Learnable Parameters
            >>> m = nn.LayerNorm(input.size()[1:])
            >>> # Without Learnable Parameters
            >>> m = nn.LayerNorm(input.size()[1:], elementwise_affine=False)
            >>> # Normalize over last two dimensions
            >>> m = nn.LayerNorm([10, 10])
            >>> # Normalize over last dimension of size 10
            >>> m = nn.LayerNorm(10)
            >>> # Activating the module
            >>> output = m(input)
        Layer Normalization`: https://arxiv.org/abs/1607.06450"""
    
    __constants__ = ['features', 'weight', 'bias', 'eps', 'center', 'scale']

    def __init__(self, features, eps=1e-12, center=True, scale=True):
        super(LayerNorm, self).__init__()
        self.features = features
        self.eps = eps
        self.center = center
        self.scale = scale

        if self.scale:
            self.weight = nn.Parameter(torch.Tensor(self.features))
        else:
            self.register_parameter('weight', None)

        if self.center:
            self.bias = nn.Parameter(torch.Tensor(self.features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.scale:
            nn.init.ones_(self.weight)

        if self.center:
            nn.init.zeros_(self.bias)

    def adjust_parameter(self, tensor, parameter):
        return torch.repeat_interleave(
            torch.repeat_interleave(
                parameter.view(-1, 1, 1),
                repeats=tensor.shape[2],
                dim=1),
            repeats=tensor.shape[3],
            dim=2
        )

    def forward(self, input):
        normalized_shape = (self.features, input.shape[2], input.shape[3])
        weight = self.adjust_parameter(input, self.weight)
        bias = self.adjust_parameter(input, self.bias)
        return F.layer_norm(
            input, normalized_shape, weight, bias, self.eps)

    def extra_repr(self):
        return '{features}, eps={eps}, ' \
            'center={center}, scale={scale}'.format(**self.__dict__)
            

class Bias(nn.Module):
    """Bias to be added to the blurred image to cater for the bias towards looking at the center of the image"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, tensor):
        return tensor + self.bias[np.newaxis, :, np.newaxis, np.newaxis]

    def extra_repr(self):
        return f'channels={self.channels}'
    
# Create a readout network to be used by the model
read101_readout = nn.Sequential(OrderedDict([
    ('layernorm0', LayerNorm(2048)),
    ('conv0', nn.Conv2d(2048, 8, (1, 1), bias=False)),
    ('bias0', Bias(8)),
    ('softplus0', nn.Softplus()),

    ('layernorm1', LayerNorm(8)),
    ('conv1', nn.Conv2d(8, 16, (1, 1), bias=False)),
    ('bias1', Bias(16)),
    ('softplus1', nn.Softplus()),

    ('layernorm2', LayerNorm(16)),
    ('conv2', nn.Conv2d(16, 1, (1, 1), bias=False)),
    ('bias2', Bias(1)),
    ('softplus2', nn.Softplus()),
]))


class DeepGazeII(torch.nn.Module):
    def __init__(self, features, readout_network, downsample=2, readout_factor=16, saliency_map_factor=2, initial_sigma=8.0):
        super().__init__()

        self.readout_factor = readout_factor
        self.saliency_map_factor = saliency_map_factor

        self.features = features

        for param in self.features.parameters():
            param.requires_grad = False
        self.features.eval()

        self.readout_network = readout_network
        self.finalizer = Finalizer(
            sigma=initial_sigma,
            learn_sigma=True,
            saliency_map_factor=self.saliency_map_factor,
        )
        self.downsample = downsample

    def forward(self, x, centerbias):
        orig_shape = x.shape
        x = F.interpolate(
            x,
            scale_factor=1 / self.downsample,
            recompute_scale_factor=False,
        )
        x = self.features(x)

        readout_shape = [math.ceil(orig_shape[2] / self.downsample / self.readout_factor), math.ceil(orig_shape[3] / self.downsample / self.readout_factor)]
        x = [F.interpolate(item, readout_shape) for item in x]

        x = torch.cat(x, dim=1)
        x = self.readout_network(x)
        x = self.finalizer(x, centerbias)

        return x

    def train(self, mode=True):
        self.features.eval()
        self.readout_network.train(mode=mode)
        self.finalizer.train(mode=mode)
        
#  Classes defining the ResNet models to be used       
class RGBResNet34(nn.Sequential):
    def __init__(self):
        super(RGBResNet34, self).__init__()
        self.resnet = torchvision.models.resnet34(pretrained=True)
        self.normalizer = Normalizer()
        super(RGBResNet34, self).__init__(self.normalizer, self.resnet)


class RGBResNet50(nn.Sequential):
    def __init__(self):
        super(RGBResNet50, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.normalizer = Normalizer()
        super(RGBResNet50, self).__init__(self.normalizer, self.resnet)


class RGBResNet50_alt(nn.Sequential):
    def __init__(self):
        super(RGBResNet50, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.normalizer = Normalizer()
        state_dict = torch.load("Resnet-AlternativePreTrain.pth")
        model.load_state_dict(state_dict)
        super(RGBResNet50, self).__init__(self.normalizer, self.resnet)



class RGBResNet101(nn.Sequential):
    def __init__(self):
        super(RGBResNet101, self).__init__()
        self.resnet = torchvision.models.resnet101(pretrained=True)
        self.normalizer = Normalizer()
        super(RGBResNet101, self).__init__(self.normalizer, self.resnet)