# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 15:46:24 2023

@author: Administrator
"""

import torch 
import torch.nn as nn
from buildingblock import DoubleConv, ResNetBlock, ResNetBlockSE, create_decoders, create_encoders

from utils import get_class, number_of_features_per_level 

class AbstractUNet(nn.Module):
    
    def __init__(self,in_channels,out_channels,final_sigmoid,basic_module,f_maps=32,
                 layer_order='gcr',num_groups=8,num_levels=4,is_segmentation=True,conv_kernel_size=3,
                 pool_kernel_size=2,conv_padding=1,is3d=True):
        super(AbstractUNet,self).__init__()
        
        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps,num_levels=num_levels)
            
        assert isinstance(f_maps, list) or isinstance(f_maps,tuple)
        assert len(f_maps) > 1 ,"Required at least 2 levels in the U-Net"
        
        if 'g' in layer_order:
            assert num_groups is not None,"num_groups must be specified if GroupNorm is used"
            
        self.encoders = create_encoders(in_channels,f_maps,basic_module,conv_kernel_size,
                                        conv_padding,layer_order,num_groups,pool_kernel_size,is3d)
        
        self.decoders = create_decoders(f_maps,basic_module,conv_kernel_size,conv_padding,layer_order,
                                        num_groups,is3d)
        
        if is3d:
            self.final_conv = nn.Conv3d(f_maps[0],out_channels,kernel_size=1)
        else:
            self.final_conv = nn.Conv2d(f_maps[0],out_channels,kernel_size=1)
            
        if is_segmentation:
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            self.final_activation = None
            
        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                # nn.init.xavier_normal_(m.weight,gain=nn.init.calculate_gain('tanh'))
                nn.init.kaiming_normal_(m.weight,mode='fan_out')
                
            

    def forward(self,x):
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0,x)
            
        encoders_features = encoders_features[1:]
        
        for decoder,encoder_features in zip(self.decoders,encoders_features):
            x = decoder(encoder_features,x)
            
        x = self.final_conv(x)
        
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)
        
        return x 
          
        
class UNet3D(AbstractUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """  

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=32, layer_order='gcr',
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


class ResidualUNet3D(AbstractUNet):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ResNetBlock as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
    Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=16, layer_order='gcr',
                 num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1, **kwargs):
        super(ResidualUNet3D, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             final_sigmoid=final_sigmoid,
                                             basic_module=ResNetBlock,
                                             f_maps=f_maps,
                                             layer_order=layer_order,
                                             num_groups=num_groups,
                                             num_levels=num_levels,
                                             is_segmentation=is_segmentation,
                                             conv_padding=conv_padding,
                                             is3d=True)


class ResidualUNetSE3D(AbstractUNet):
    """_summary_
    Residual 3DUnet model implementation with squeeze and excitation based on 
    https://arxiv.org/pdf/1706.00120.pdf.
    Uses ResNetBlockSE as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch
    out for block artifacts). Since the model effectively becomes a residual
    net, in theory it allows for deeper UNet.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1, **kwargs):
        super(ResidualUNetSE3D, self).__init__(in_channels=in_channels,
                                               out_channels=out_channels,
                                               final_sigmoid=final_sigmoid,
                                               basic_module=ResNetBlockSE,
                                               f_maps=f_maps,
                                               layer_order=layer_order,
                                               num_groups=num_groups,
                                               num_levels=num_levels,
                                               is_segmentation=is_segmentation,
                                               conv_padding=conv_padding,
                                               is3d=True)


class UNet2D(AbstractUNet):
    """
    2DUnet model from
    `"U-Net: Convolutional Networks for Biomedical Image Segmentation" <https://arxiv.org/abs/1505.04597>`
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, **kwargs):
        super(UNet2D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     is3d=False)


def get_model(model_config):
    model_class = get_class(model_config['name'], modules=['model'])
    return model_class(**model_config)
        
        
# model = ResidualUNetSE3D(in_channels=1, out_channels=1)
# device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# model.train()
# input=torch.rand(2,1,96,96,96).cuda()
# output=model(input) 
# print(output.shape)    



                    