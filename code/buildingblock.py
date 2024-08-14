# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 08:17:41 2023

@author: Administrator
"""

from functools import partial 
import torch 
import torch.nn as nn 
from torch.nn import functional as F
from seblock import ChannelSELayer3D, ChannelSpatialSELayer3D, SpatialSELayer3D

def create_conv(in_channels,out_channels,kernel_size,order,num_groups,padding,is3d):
    
    assert 'c' in order,"Conv layer Must be present"
    assert order[0] not in 'rle' , "nonlinear cannot be the first operation in the layer"
    
    modules = []
    for i , char in enumerate(order):
        if char =='r':
            modules.append(('ReLU',nn.ReLU(inplace=True)))
        elif char =='l':
            modules.append(('LeakReLU',nn.LeakyReLU(inplace=True)))
        elif char =='e':
            modules.append(('ELU',nn.ELU(inplace=True)))
        elif char =='c':
            bias = not('g' in order or 'b' in order)
            if is3d:
                conv = nn.Conv3d(in_channels,out_channels,kernel_size,padding=padding,bias=bias)
            else:
                conv = nn.Conv2d(in_channels,out_channels,kernel_size,padding=padding,bias=bias)
                
            modules.append(('conv',conv))
            
        elif char =='g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels 
            
            if num_channels < num_groups:
                num_groups = 1 
                
            assert num_channels % num_groups == 0 ,f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm',nn.GroupNorm(num_groups=num_groups,num_channels=num_channels)))
            
        elif char =='b':
            is_before_conv = i < order.index('c')
            if is3d:
                bn = nn.BatchNorm3d 
            else:
                bn = nn.BatchNorm2d 
                
            if is_before_conv:
                modules.append(('BatchNorm',bn(in_channels)))
            else:
                modules.append(('BatchNorm',bn(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}' must be one of['b','g','r','l','e','c']")
            
    return modules 

class SingleConv(nn.Sequential):
    def __init__(self,in_channels,out_channels,kernel_size=3,order='gcr',num_groups=8,padding=1,is3d=True):
        super(SingleConv,self).__init__()
        for name,module in create_conv(in_channels,out_channels,kernel_size,order,num_groups,padding,is3d):
            self.add_module(name, module)
            
class DoubleConv(nn.Sequential):
    def __init__(self,in_channels,out_channels,encoder,kernel_size=3,order='gcr',num_groups=8,padding=1,is3d=True):
        super(DoubleConv,self).__init__()
        if encoder:
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2 
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            conv1_in_channels,conv1_out_channels = in_channels,out_channels 
            conv2_in_channels,conv2_out_channels = out_channels,out_channels 
            
        self.add_module('SingleConv1',SingleConv(conv1_in_channels,conv1_out_channels,kernel_size,order,num_groups,
                                                 padding=padding,is3d=is3d))
        self.add_module('SingleConv2', SingleConv(conv2_in_channels,conv2_out_channels,kernel_size,order,num_groups,
                                                  padding=padding,is3d=is3d))
        
class ResNetBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,order='cge',num_groups=8,is3d=True,**kwargs):
        super(ResNetBlock,self).__init__()
        if in_channels != out_channels:
            #conv 1x1 for increasing the number of channels 
            if is3d:
                self.conv1 = nn.Conv3d(in_channels,out_channels,kernel_size=1)
            else:
                self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=1)
                
        else:
            self.conv1 = nn.Identity()
            
        self.conv2 = SingleConv(out_channels, out_channels,kernel_size=kernel_size,
                                order=order,num_groups=num_groups,is3d=is3d)
        n_order = order 
        for c in 'rel':
            n_order = n_order.replace(c,'')
        self.conv3 = SingleConv(out_channels,out_channels,kernel_size=kernel_size,order=n_order,
                                num_groups=num_groups,is3d=is3d)
        
        if 'l' in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1,inplace=True)
        elif 'e' in order:
            self.non_linearity = nn.ELU(inplace=True)
        else:
            self.non_linearity = nn.ReLU(inplace=True)
            
    def forward(self,x):
        residual = self.conv1(x)
        out = self.conv2(residual)
        out = self.conv3(out)
        out += residual 
        out = self.non_linearity(out)
        return out 
    
class ResNetBlockSE(ResNetBlock):
    def __init__(self,in_channels,out_channels,kernel_size=3,order='cge',num_groups=8,se_module='scse',**kwargs):
        super(ResNetBlockSE,self).__init__(in_channels,out_channels,kernel_size=kernel_size, order=order,
            num_groups=num_groups, **kwargs)
        assert se_module in ['scse','cse','sse']
        if se_module =='scse':
            self.se_module = ChannelSpatialSELayer3D(num_channels=out_channels,reduction_ratio=1)
        elif se_module =='cse':
            self.se_module = ChannelSELayer3D(num_channels=out_channels,reduction_ratio=1)
        elif se_module =='sse':
            self.se_module = SpatialSELayer3D(num_channels=out_channels)
    
    def forward(self,x):
        out = super().forward(x)
        out = self.se_module(out)
        
        return out 
    
class Encoder(nn.Module):
    def __init__(self,in_channels,out_channels,conv_kernel_size=3,apply_pool=True,
                 pool_kernel_size=2,pool_type='max',basic_module=DoubleConv,conv_layer_order='gcr',
                 num_groups=8,padding=1,is3d=True):
        super(Encoder,self).__init__()
        assert pool_type in ['max','avg']
        self.apply_pool = apply_pool
        if self.apply_pool:
            if pool_type == 'max':
                if is3d:
                    self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
                else:
                    self.pooling = nn.MaxPool2d(kernel_size=pool_kernel_size)
            else:
                if is3d:
                    self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
                else:
                    self.pooling = nn.AvgPool2d(kernel_size=pool_kernel_size)
                    
        else:
            self.pooling = None 
            
        self.basic_module = basic_module(in_channels, out_channels, encoder=True,
                                         kernel_size = conv_kernel_size,
                                         order = conv_layer_order,
                                         num_groups = num_groups,
                                         padding=padding,is3d=is3d)
        
    def forward(self,x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x 
    
class Decoder(nn.Module):
    def __init__(self,in_channels,out_channels,conv_kernel_size=3,scale_factor=(2,2,2),basic_module=DoubleConv,
                 conv_layer_order='gcr',num_groups=8,mode='nearest',padding=1,upsample=True,is3d=True):
        super(Decoder,self).__init__()
        
        if upsample:
            if basic_module == DoubleConv:
                self.upsampling = InterpolateUpsampling(mode=mode)
                self.joining = partial(self._joining,concat=True)
            else:
                self.upsampling = TransposeConvUpsampling(in_channels=in_channels,out_channels=out_channels,
                                                        kernel_size=conv_kernel_size,scale_factor=scale_factor)
                self.joining = partial(self._joining,concat=False)
                in_channels = out_channels 
        else:
            self.upsampling = NoUpsampling()
            self.joining = partial(self._joining,concat=True)
        
        self.basic_module = basic_module(in_channels, out_channels, encoder=False,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding,
                                         is3d=is3d)
        
    def forward(self,encoder_features,x):
        x = self.upsampling(encoder_features=encoder_features,x=x)
        x = self.joining(encoder_features,x)
        x = self.basic_module(x)
        return x 
    
    @staticmethod 
    def _joining(encoder_features,x,concat):
        if concat:
            return torch.cat((encoder_features,x),dim=1)
        else:
            return encoder_features + x 
        
def create_encoders(in_channels,f_maps,basic_module,conv_kernel_size,conv_padding,layer_order,num_groups,
                   pool_kernel_size,is3d):
    encoders = []
    for i, out_feature_num in enumerate(f_maps):
        if i==0:
            encoder = Encoder(in_channels,out_feature_num,
                              apply_pool=False,#第一个encoder没有下采样
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              padding=conv_padding,
                              is3d=is3d)
        else:
            encoder = Encoder(f_maps[i - 1],out_feature_num,
                              apply_pool=True,
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              pool_kernel_size=pool_kernel_size,
                              padding=conv_padding,
                              is3d=is3d)
        encoders.append(encoder)
        
    return nn.ModuleList(encoders)

def create_decoders(f_maps,basic_module,conv_kernel_size,conv_padding,layer_order,num_groups,is3d):
    #create decoder path consisting of the Decoder modules. The length of the decoder list is equal to `len(f_maps) - 1`
    decoders = []
    reversed_f_maps = list(reversed(f_maps))
    for i in range(len(reversed_f_maps)-1):
        if basic_module == DoubleConv:
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
        else:
            in_feature_num = reversed_f_maps[i]
        
        out_feature_num = reversed_f_maps[i + 1]
        
        decoder = Decoder(in_feature_num,out_feature_num,
                          basic_module=basic_module,
                          conv_layer_order=layer_order,
                          conv_kernel_size=conv_kernel_size,
                          num_groups=num_groups,
                          padding=conv_padding,
                          is3d=is3d)
        
        decoders.append(decoder)
        
    return nn.ModuleList(decoders)


"""
Abstract class for upsampling. A given implementation should upsample a given 5D input tensor using either
interpolation or learned transposed convolution.
"""
class AbstractUpsampling(nn.Module):
    def __init__(self,upsample):
        super(AbstractUpsampling,self).__init__()
        self.upsample = upsample 
        
    def forward(self,encoder_features,x):
        output_size = encoder_features.size()[2:]
        
        return self.upsample(x,output_size)
    
class InterpolateUpsampling(AbstractUpsampling):
    def __init__(self,mode='nearest'):
        upsample = partial(self._interpolate,mode=mode)
        super().__init__(upsample)
        
    @staticmethod 
    def _interpolate(x,size,mode):
        return F.interpolate(x, size=size,mode=mode)
    
class TransposeConvUpsampling(AbstractUpsampling):
    """
    Args:
        in_channels (int): number of input channels for transposed conv
            used only if transposed_conv is True
        out_channels (int): number of output channels for transpose conv
            used only if transposed_conv is True
        kernel_size (int or tuple): size of the convolving kernel
            used only if transposed_conv is True
        scale_factor (int or tuple): stride of the convolution
            used only if transposed_conv is True

    """
    def __init__(self,in_channels=None,out_channels=None,kernel_size=3,scale_factor=(2,2,2)):
        upsample = nn.ConvTranspose3d(in_channels,out_channels,kernel_size=kernel_size,stride=scale_factor,
                                      padding=1)
        super().__init__(upsample)
        
class NoUpsampling(AbstractUpsampling):
    def __init__(self):
        super().__init__(self._no_upsampling)

    @staticmethod 
    def _no_upsampling(x,size):
        return x 
                
        

    
            