# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, Linear, ShapeSpec, get_norm
from detectron2.utils.registry import Registry

ROI_BOX_HEAD_REGISTRY = Registry("ROI_BOX_HEAD")
ROI_BOX_HEAD_REGISTRY.__doc__ = """
Registry for box heads, which make box predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""
 # from facebook detectron2
class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None

def Max(x):
    """
    A wrapper around torch.max in Spatial Attention Module (SAM) to support empty inputs and more features.
    """
    if x.numel() == 0:
        output_shape = [x.shape[0], 1, x.shape[2], x.shape[3]]
        empty = _NewEmptyTensorOp.apply(x, output_shape)
        return empty
    return torch.max(x, dim=1, keepdim=True)[0]

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        weight_init.c2_msra_fill(self.conv)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = Max(x)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv(scale)
        return x * self.sigmoid(scale)

@ROI_BOX_HEAD_REGISTRY.register()
class FastRCNNConvFCHeadWithAttention(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and then
    several fc layers (each followed by relu).
    """

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        num_conv: int,
        conv_dim: int,
        num_fc: int,
        fc_dim: int,
        conv_norm="",
    ):
        """
        Args:
            input_shape (ShapeSpec): shape of the input feature.
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the output dimension of the conv/fc layers
            conv_norm: normalization for the conv layers. See :func:`detectron2.layers.get_norm`
                for supported types.
        """
        super().__init__()
        assert num_conv + num_fc > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.spatialAtt = SpatialAttention()
        self.fcs = []
        for k in range(num_fc):
            fc = Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "num_conv": cfg.MODEL.ROI_BOX_HEAD.NUM_CONV,
            "conv_dim": cfg.MODEL.ROI_BOX_HEAD.CONV_DIM,
            "num_fc": cfg.MODEL.ROI_BOX_HEAD.NUM_FC,
            "fc_dim": cfg.MODEL.ROI_BOX_HEAD.FC_DIM,
            "conv_norm": cfg.MODEL.ROI_BOX_HEAD.NORM,
            "input_shape": input_shape,
        }

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        x = self.spatialAtt(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x))
        return x

    @property
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])

@ROI_BOX_HEAD_REGISTRY.register()
class FastRCNNConvFCHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and then
    several fc layers (each followed by relu).
    """

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        num_conv: int,
        conv_dim: int,
        num_fc: int,
        fc_dim: int,
        conv_norm="",
    ):
        """
        Args:
            input_shape (ShapeSpec): shape of the input feature.
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the output dimension of the conv/fc layers
            conv_norm: normalization for the conv layers. See :func:`detectron2.layers.get_norm`
                for supported types.
        """
        super().__init__()
        assert num_conv + num_fc > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k in range(num_fc):
            fc = Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "num_conv": cfg.MODEL.ROI_BOX_HEAD.NUM_CONV,
            "conv_dim": cfg.MODEL.ROI_BOX_HEAD.CONV_DIM,
            "num_fc": cfg.MODEL.ROI_BOX_HEAD.NUM_FC,
            "fc_dim": cfg.MODEL.ROI_BOX_HEAD.FC_DIM,
            "conv_norm": cfg.MODEL.ROI_BOX_HEAD.NORM,
            "input_shape": input_shape,
        }

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x))
        return x

    @property
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])


def build_box_head(cfg, input_shape):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_BOX_HEAD.NAME
    return ROI_BOX_HEAD_REGISTRY.get(name)(cfg, input_shape)
