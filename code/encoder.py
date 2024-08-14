# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 09:18:52 2023

@author: Administrator
"""

import torch 
import torch.nn as nn
from buildingblock import DoubleConv, ResNetBlock, ResNetBlockSE, create_decoders, create_encoders

from utils import get_class, number_of_features_per_level 

class AbstractNet(nn.Module):
    
    def __init__(self,in_channels,out_channels,final_sigmoid,basic_module,f_maps=32,
                 layer_order='gcr',num_groups=8,num_levels=4,is_segmentation=True,conv_kernel_size=3,
                 pool_kernel_size=2,conv_padding=1,is3d=True):
        super(AbstractNet,self).__init__()
        
        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps,num_levels=num_levels)
            
        assert isinstance(f_maps, list) or isinstance(f_maps,tuple)
        assert len(f_maps) > 1 ,"Required at least 2 levels in the U-Net"
        
        if 'g' in layer_order:
            assert num_groups is not None,"num_groups must be specified if GroupNorm is used"
            
        self.encoders = create_encoders(in_channels,f_maps,basic_module,conv_kernel_size,
                                        conv_padding,layer_order,num_groups,pool_kernel_size,is3d)
        
        # self.decoders = create_decoders(f_maps,basic_module,conv_kernel_size,conv_padding,layer_order,
        #                                 num_groups,is3d)
        
        if is3d:
            self.final_conv = nn.Conv3d(f_maps[-1],out_channels,kernel_size=1)
        else:
            self.final_conv = nn.Conv2d(f_maps[-1],out_channels,kernel_size=1)
            
        if is_segmentation:
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            self.final_activation = None
            
        self.avg = nn.AdaptiveAvgPool3d((1,1,1))
            
        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                # nn.init.xavier_normal_(m.weight,gain=nn.init.calculate_gain('tanh'))
                nn.init.kaiming_normal_(m.weight,mode='fan_out')
                
            

    def forward(self, x1, x2, x3):
        x = torch.cat([x1, x2, x3],dim=1)
        for encoder in self.encoders:
            x = encoder(x)
            
        x = self.final_conv(x)
        x = self.avg(x)
        # x = x.view(x.size(0),-1)
        
        # if not self.training and self.final_activation is not None:
        #     x = self.final_activation(x)
        return x 
    
        
class UNet3D(AbstractNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=8, layer_order='gcr',
                 num_groups=4, num_levels=4, is_segmentation=True, conv_padding=1, **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     is3d=True)
        
# p = torch.rand(2,1,96,96,96).cuda()
# v = torch.rand(2,1,96,96,96).cuda()
# vp = torch.rand(2,1,96,96,96).cuda()
# model = UNet3D(in_channels=3,out_channels=2,final_sigmoid=False).cuda()
# out = model(p,v,vp)
