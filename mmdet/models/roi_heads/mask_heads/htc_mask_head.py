from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from .fcn_mask_head import FCNMaskHead

import torch
import torch.nn as nn
import fvcore.nn.weight_init as weight_init

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
        self.conv = ConvModule(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = Max(x)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv(scale)
        return x * self.sigmoid(scale)

@HEADS.register_module()
class HTCMaskHead(FCNMaskHead):

    def __init__(self, with_conv_res=True, *args, **kwargs):
        super(HTCMaskHead, self).__init__(*args, **kwargs)
        self.with_conv_res = with_conv_res
        if self.with_conv_res:
            self.conv_res = ConvModule(
                self.conv_out_channels,
                self.conv_out_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        self.spatialAtt = SpatialAttention()

    def init_weights(self):
        super(HTCMaskHead, self).init_weights()
        if self.with_conv_res:
            self.conv_res.init_weights()

    def forward(self, x, res_feat=None, return_logits=True, return_feat=True):
        if res_feat is not None:
            assert self.with_conv_res
            res_feat = self.conv_res(res_feat)
            x = x + res_feat
        for conv in self.convs:
            x = conv(x)
        x = self.spatialAtt(x)
        res_feat = x
        outs = []
        if return_logits:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
            mask_pred = self.conv_logits(x)
            outs.append(mask_pred)
        if return_feat:
            outs.append(res_feat)
        return outs if len(outs) > 1 else outs[0]
