#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import numpy as np

def swish(x):
    return x*torch.sigmoid(x)

def get_actfn(use_act):
    if use_act == 1:
        actfn = F.elu
    elif use_act == 2:
        actfn = swish
    else:
        raise
    return actfn

def get_padding_size(use_causal, kernel_size, dilation):
    if use_causal == 0:
        if kernel_size%2 != 1: raise
        padding = (kernel_size//2)*dilation
    elif use_causal == 1:
        padding = (kernel_size-1)*dilation
    else:
        raise
    return padding

def get_norm(use_batchnorm, num_features, ndims=4):
    if ndims == 4:
        if use_batchnorm == 1:
            norm = nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.001)
        elif use_batchnorm == 2:
            norm = nn.GroupNorm(1, num_features, eps=1e-5, affine=True)
        elif use_batchnorm == 3:
            norm = nn.GroupNorm(num_features, num_features, eps=1e-5, affine=True)
        else:
            raise
    elif ndims == 3:
        if use_batchnorm == 1:
            norm = nn.BatchNorm1d(num_features, eps=1e-5, momentum=0.001)
        elif use_batchnorm == 2:
            norm = nn.GroupNorm(1, num_features, eps=1e-5, affine=True)
        elif use_batchnorm == 3:
            norm = nn.GroupNorm(num_features, num_features, eps=1e-5, affine=True)
        else:
            raise
    else:
        raise
    return norm

class ConvBNReLUBlock(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)
    def __init__(self, in_channels, out_channels, kernel_size,
            stride=1, padding=0, dilation=1,
            use_batchnorm=2, use_deconv=0, use_act=1,
            use_convbnrelu=1, memory_efficient=0, ndims=4):
        super(ConvBNReLUBlock, self).__init__()
        if ndims == 4:
            module_name = nn.ConvTranspose2d if use_deconv == 1 else nn.Conv2d
        elif ndims == 3:
            module_name = nn.ConvTranspose1d if use_deconv == 1 else nn.Conv1d
        else:
            raise
        conv = module_name(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation)
        if use_convbnrelu == 0:
            #bnreluconv
            norm = get_norm(use_batchnorm, in_channels, ndims=ndims)
        else:
            #convbnrelu or convrelubn
            norm = get_norm(use_batchnorm, out_channels, ndims=ndims)
        self.add_module('conv', conv)
        self.add_module('norm', norm)
        self.actfn = get_actfn(use_act)
        self.use_convbnrelu = use_convbnrelu
        self.memory_efficient = memory_efficient

        self.init_weights()

    def init_weights(self):
        self.conv.bias.data[:] = 0.0

    def forward(self, h):
        def f_(h_):
            if self.use_convbnrelu == 2:
                #convrelubn
                return self['norm'](self.actfn(self['conv'](h_)))
            elif self.use_convbnrelu == 1:
                #convbnrelu
                return self.actfn(self['norm'](self['conv'](h_)))
            else:
                #bnreluconv
                return self['conv'](self.actfn(self['norm'](h_)))
        return cp.checkpoint(f_, h) if self.memory_efficient else f_(h)

class Conv1dBNReLUBlock(ConvBNReLUBlock):
    def __init__(self, in_channels, out_channels, kernel_size,
            stride=1, padding=0, dilation=1,
            use_batchnorm=2, use_deconv=0, use_act=1,
            use_convbnrelu=1, memory_efficient=0):
        super(Conv1dBNReLUBlock, self).__init__(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation,
                use_batchnorm=use_batchnorm, use_deconv=use_deconv, use_act=use_act,
                use_convbnrelu=use_convbnrelu, memory_efficient=memory_efficient, ndims=3)
        pass
 
class Conv2dBNReLUBlock(ConvBNReLUBlock):
    def __init__(self, in_channels, out_channels, kernel_size,
            stride=1, padding=0, dilation=1,
            use_batchnorm=2, use_deconv=0, use_act=1,
            use_convbnrelu=1, memory_efficient=0):
        super(Conv2dBNReLUBlock, self).__init__(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation,
                use_batchnorm=use_batchnorm, use_deconv=use_deconv, use_act=use_act,
                use_convbnrelu=use_convbnrelu, memory_efficient=memory_efficient, ndims=4)
        pass

class DenseBlockNoOutCat(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)
    def __init__(self, in_channels, growth_rate, kernel_size,
            n_layers=4, use_dilation=0,
            use_batchnorm=2, use_act=1,
            use_convbnrelu=1, memory_efficient=0):
        super(DenseBlockNoOutCat, self).__init__()
        self._layers = []
        sum_channels = in_channels
        for ll in range(n_layers):
            dilation = (2**ll) if use_dilation else 1
            padding = (get_padding_size(False, kernel_size[0], dilation), kernel_size[1]//2)
            convbnrelu = Conv2dBNReLUBlock(sum_channels, growth_rate, kernel_size, 
                    stride=(1,1), padding=padding, dilation=(dilation,1),
                    use_batchnorm=use_batchnorm, use_deconv=0, use_act=use_act,
                    use_convbnrelu=use_convbnrelu, memory_efficient=0)
            self.add_module("convbnrelu%d"%ll, convbnrelu)
            self._layers.append(convbnrelu)
            sum_channels += growth_rate
        self.n_layers = n_layers
        self.memory_efficient = memory_efficient

    def forward(self, x):
        x = [x]
        for idx, nn_module in enumerate(self._layers):
            def g_(nn_module_=nn_module):
                def f_(*x_):
                    x_ = torch.cat(x_, dim=1)
                    return nn_module_(x_)
                return f_
            if self.memory_efficient:
                h = cp.checkpoint(g_(), *x)
            else:
                h = g_()(*x)
            x.append(h)
        return h

class FreqMapping(nn.Module):
    def __init__(self, in_channels, growth_rate, n_freqs,
            use_batchnorm=2, use_act=1,
            use_convbnrelu=1, memory_efficient=0):
        super(FreqMapping, self).__init__()

        self.convbnrelu1 = Conv2dBNReLUBlock(in_channels, growth_rate, kernel_size=(1,1),
                stride=(1,1), padding=(0,0), dilation=(1,1),
                use_batchnorm=use_batchnorm, use_deconv=0, use_act=use_act,
                use_convbnrelu=use_convbnrelu, memory_efficient=0)

        self.convbnrelu2 = Conv2dBNReLUBlock(n_freqs, n_freqs, kernel_size=(1,1),
                stride=(1,1), padding=(0,0), dilation=(1,1),
                use_batchnorm=use_batchnorm, use_deconv=0, use_act=use_act,
                use_convbnrelu=use_convbnrelu, memory_efficient=0)

        self.memory_efficient = memory_efficient

    def forward(self, x):
        def f_(x_):
            x_ = self.convbnrelu1(x_) #[B,C,T,F]
            x_ = x_.transpose(1,3) #[B,F,T,C]        
            x_ = self.convbnrelu2(x_) #[B,F,T,C]
            x_ = x_.transpose(1,3) #[B,C,T,F]
            return x_
        return cp.checkpoint(f_, x) if self.memory_efficient else f_(x)

class DenseBlockNoOutCatFM(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)
    def __init__(self, in_channels, growth_rate, kernel_size, n_freqs, 
            n_layers=5, use_dilation=0,
            use_batchnorm=2, use_act=1,
            use_convbnrelu=1, memory_efficient=0):
        super(DenseBlockNoOutCatFM, self).__init__()
        self._layers = []
        sum_channels = in_channels
        middle_layer = n_layers // 2
        assert n_layers % 2 == 1 # The middle one does frequency mapping.
        for ll in range(n_layers):
            if ll == middle_layer:
                convbnrelu = FreqMapping(sum_channels, growth_rate, n_freqs, 
                        use_batchnorm=use_batchnorm, use_act=use_act,
                        use_convbnrelu=use_convbnrelu, memory_efficient=0)
            else:
                dilation = 2**(ll-(1 if ll > middle_layer else 0)) if use_dilation else 1
                padding = (get_padding_size(False, kernel_size[0], dilation), kernel_size[1]//2)
                convbnrelu = Conv2dBNReLUBlock(sum_channels, growth_rate, kernel_size, 
                        stride=(1,1), padding=padding, dilation=(dilation,1),
                        use_batchnorm=use_batchnorm, use_deconv=0, use_act=use_act,
                        use_convbnrelu=use_convbnrelu, memory_efficient=0)
            self.add_module("convbnrelu%d"%ll, convbnrelu)
            self._layers.append(convbnrelu)
            sum_channels += growth_rate
        self.n_layers = n_layers
        self.memory_efficient = memory_efficient

    def forward(self, x):
        x = [x]
        for idx, nn_module in enumerate(self._layers):
            def g_(nn_module_=nn_module):
                def f_(*x_):
                    x_ = torch.cat(x_, dim=1)
                    return nn_module_(x_)
                return f_
            h = cp.checkpoint(g_(), *x) if self.memory_efficient else g_()(*x)
            x.append(h)
        return h

class TCNDepthWise(nn.Module):
    def __init__(self, input_channels, out_channels, kernel_size=3,
            use_batchnorm=1, use_act=1, dilation=1, use_causal=0):
        super(TCNDepthWise, self).__init__()

        assert use_causal == 0

        padding = get_padding_size(use_causal, kernel_size, dilation)

        self.norm0 = get_norm(use_batchnorm,input_channels,ndims=3)
        self.conv00 = nn.Conv1d(input_channels, input_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=input_channels)
        self.conv01 = nn.Conv1d(input_channels, out_channels, kernel_size=1)

        self.norm1 = get_norm(use_batchnorm,out_channels,ndims=3)
        self.conv10 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=out_channels)
        self.conv11 = nn.Conv1d(out_channels, out_channels, kernel_size=1)

        self.actfn = get_actfn(use_act)

        self.init_weights()

    def init_weights(self):
        self.conv00.weight.data.normal_(0, 0.01)
        self.conv01.weight.data.normal_(0, 0.01)
        self.conv10.weight.data.normal_(0, 0.01)
        self.conv11.weight.data.normal_(0, 0.01)
        self.conv00.bias.data[:] = 0.0
        self.conv01.bias.data[:] = 0.0
        self.conv10.bias.data[:] = 0.0
        self.conv11.bias.data[:] = 0.0

    def forward(self, x, hidden_dropout_rate=0.0, dilation_dropout_rate=0.0):

        h = self.actfn(self.norm0(x))

        if hidden_dropout_rate > 0.0:
            h = F.dropout(h, p=hidden_dropout_rate, training=True, inplace=False)
        
        h = self.conv00(h)
        h = self.conv01(h)

        h = self.actfn(self.norm1(h))

        if hidden_dropout_rate > 0.0:
            h = F.dropout(h, p=hidden_dropout_rate, training=True, inplace=False)
        
        h = self.conv10(h)
        h = self.conv11(h)

        return h+x
