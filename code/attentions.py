# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 14:40:07 2023

@author: Administrator
"""

import torch
import torch.nn as nn
from torch.nn import einsum
from einops import rearrange
from torch.nn import functional as F
import math

class ChannelSElayer(nn.Module):
    def __init__(self,num_channels,reduction_ratio):
        super(ChannelSElayer,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((1,1,1))
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels,num_channels_reduced,bias=False)
        self.fc2 = nn.Linear(num_channels_reduced,num_channels,bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        batchSize, numChannels,D,W,H = x.size()
        squeezeTensor = self.avg_pool(x)
        
        fcOut1 = self.relu(self.fc1(squeezeTensor.view(batchSize,numChannels)))
        fcOut2 = self.sigmoid(self.fc2(fcOut1))
        
        outputTensor = torch.mul(x,fcOut2.view(batchSize,numChannels,1,1,1))
        
        return outputTensor
    
class SpatialSElayer(nn.Module):
    def __init__(self,num_channels):
        super(SpatialSElayer,self).__init__()
        self.conv = nn.Conv3d(num_channels,1,kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        batchSize, channel, D, H, W = x.size()
        out = self.conv(x)
        squeezeTensor = self.sigmoid(out)
        outputTensor = torch.mul(x,squeezeTensor.view(batchSize,1,D,H,W))
        return outputTensor
    
class ChannelSpatialSElayer(nn.Module):
    def __init__(self,num_channels,reduction_ratio):
        super(ChannelSpatialSElayer,self).__init__()
        self.cSE = ChannelSElayer(num_channels, reduction_ratio)
        self.sSE = SpatialSElayer(num_channels)
    def forward(self,x):
        outputTensor = self.cSE(x) + self.sSE(x)
        return outputTensor
    

        
class FeatureSelect(nn.Module):
    def __init__(self,num_channels):
        super(FeatureSelect,self).__init__()
        
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.maxpool = nn.AdaptiveMaxPool3d((1,1,1))
        self.spectfilter = nn.Conv3d()
        self.relu = nn.ReLU()
        
    def forward(self,fmulti,fca,ft):
        
        fconcat = torch.cat((fca,ft),dim=1)
        favg = self.avgpool(fconcat)
        fmax = self.maxpool(fconcat)
        fAvgDiff = abs(self.avgpool(fmulti) - favg)
        fMaxDiff = abs(self.maxpool(fmulti) - fmax)
        
        fFS = torch.cat((favg + 0.1*fAvgDiff,fmax + 0.1*fMaxDiff),dim=1)
        
        return fFS
    
class BasicConv(nn.Module):
    def __init__(self,in_plane,out_plane,kernel_size,stride=1,padding=0,
                 dilation=1,groups=1,relu=True,bn=True,bias=False):
        super(BasicConv,self).__init__()
        
        self.conv = nn.Conv2d(in_plane,out_plane,kernel_size=kernel_size,stride=stride,
                              padding=padding,dilation=dilation,groups=groups,bias=bias)
        self.bn = nn.BatchNorm2d(out_plane,momentum=0.01)
        self.relu = nn.ReLU() if relu else None
        
    def forward(self,x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x 

class ZPool(nn.Module):
    def forward(self,x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1),torch.mean(x,1).unsqueeze(1)),dim=1)

class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate,self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(in_plane=2,out_plane=1,kernel_size=kernel_size,
                              stride=1,padding=(kernel_size-1)//2,relu=False)
    
    def forward(self,x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x*scale
    
class TripleAttention(nn.Module):
    def __init__(self,no_spatial=False):
        super(TripleAttention,self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()
    
    def forward(self,x):
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1/3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1/2 * (x_out11 + x_out21)
        return x_out

"""self attention"""
        
class SelfAttention(nn.Module):
    def __init__(self,in_dim):
        super(SelfAttention,self)
        self.query_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim,kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self,x):
        batchSize, C, H, W = x.size()
        proj_query = self.query_conv(x).view(batchSize,-1,H*W).permute(0,2,1)
        proj_key = self.key_conv(x).view(batchSize,-1,H*W)
        proj_value = self.value_conv(x).view(batchSize,-1,H*W)
        energy = torch.bmm(proj_query,proj_key)
        attention = self.softmax(energy)
        out = torch.bmm(proj_value,attention.permute(0,2,1))
        out = self.gamma*out + x
        return out

""" multi-head attention"""

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super(MultiHeadSelfAttention,self)
        self.num_head = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.query = nn.Linear(embed_dim,embed_dim)
        self.key = nn.Linear(embed_dim,embed_dim)
        self.value = nn.Linear(embed_dim,embed_dim)
        self.fc = nn.Linear(embed_dim,embed_dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self,x):
        batchSize, num_head, H, W = x.shape
        proj_query = self.query(x).view(batchSize,-1,H*W).view(batchSize,self.num_head,self.head_dim,H*W)
        proj_query = proj_query.permute(0,1,3,2).contiguous()
        proj_key = self.key(x).view(batchSize,-1,H*W).view(batchSize,self.num_head,self.head_dim,H*W)
        proj_value = self.value(batchSize,-1,H*W).view(batchSize,self.num_head,self.head_dim,H*W)
        
        attention = torch.matmul(proj_query,proj_key) / torch.sqrt(torch.tensor(self.head_dim,dtype=torch.float))
        attention = torch.softmax(attention, dim=-1)
        out = torch.matmul(proj_value,attention.permute(0,1,3,2)).view(batchSize,-1,H,W)
        out = self.gamma*out + x
        return out
        
"""Criss Cross Attention"""

# def INF(B,H,W):
#     return -torch.diag(torch.tensor(float("inf")).repeat(H),0).unsqueeze(0).repeat(B*W,1,1)
#     """ return B*W 个对角线有H个inf的0矩阵"""
    
# class CrissCrossAttention(nn.Module):
#     def __init__(self,in_dim):
#         super(CrissCrossAttention,self).__init__()
        
#         self.query_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=1)
#         self.val_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim,kernel_size=1)
        
#         self.softmax = torch.nn.Softmax(dim=3)
#         self.INF = INF
#         self.gamma = nn.Parameter(torch.zeros(1))
        
#     def forward(self,x):
#         batchSize, C, H, W = x.size()
#         proj_query = self.query_conv(x)
#         # proj_query_H  'b,c,h,w'>'b,w,c,h'>'b*w,c,h'>'(b*w,h,c)'
#         proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(batchSize*W,-1,H).permute(0,2,1)
#         # proj_query_W 'b,c,h,w'>'b,h,c,w'>'b*h,c,w'>'(b*h,w,c)'
#         proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(batchSize*H,-1,W).permute(0,2,1)
        
#         proj_key = self.key_conv(x)
#         #proj_key_H 'b,c,h,w'>'b,w,c,h'>'b*w,c,h'
#         proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(batchSize*W,-1,H)
#         #proj_key_W 'b,c,h,w'>'b,h,c,w'>'b*h,c,w'
#         proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(batchSize*H,-1,W)
        
#         proj_value = self.val_conv(x)
#         #'b*w,c,h'
#         proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(batchSize*W,-1,H)
#         #'b*h,c,w'
#         proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(batchSize*H,-1,W)
#         # 'b*w,h,h' > 'b,w,h,h' > 'b,h,w,h'
#         energy_H = (torch.bmm(proj_query_H,proj_key_H)+self.INF(batchSize,H,W)).view(batchSize,W,H,H).permute(0,2,1,3)
#         # 'b*h,w,w' > 'b,h,w,w'
#         energy_W = torch.bmm(proj_query_W,proj_key_W).view(batchSize,H,W,W)
#         #' b,h,w,h + b,h,w,w' > 'b,h,w,h+w'
#         concate = self.softmax(torch.cat((energy_H,energy_W),dim=3))
        
#         # 'b,h,w,h' > 'b,w,h,h' >'b*w,h,h'
#         atten_H = concate[:,:,:,0:H].permute(0,2,1,3).contiguous().view(batchSize*W,H,H)
#         # 'b,h,w,w' > 'b*h,w,w'
#         atten_W = concate[:,:,:,H:H+W].contiguous().view(batchSize*H,W,W)
#         # 'b*w,c,h' x 'b*w,h,h' > 'b*w,c,h' > 'b,w,c,h' > 'b,c,h,w'
#         out_H = torch.bmm(proj_value_H,atten_H.permute(0,2,1)).view(batchSize,W,-1,H).permute(0,2,3,1)
#         out_W = torch.bmm(proj_value_W,atten_W.permute(0,2,1)).view(batchSize,H,-1,W).permute(0,2,1,3)
        
#         output = self.gamma*(out_H + out_W) + x
                              
#         return output
    
def INF(B,H,W):
    return -torch.diag(torch.tensor(float('inf')).repeat(H),0).unsqueeze(0).repeat(B*W,1,1)


class CrissCrossAttention(nn.Module):
    
    def __init__(self,in_dim):
        super().__init__()
        
        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8,kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.inf = INF
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self,x):
        b, C, H, W = x.shape
        
        proj_q = self.query_conv(x)
        proj_q_H = rearrange(proj_q, 'b c h w -> (b w) h c')
        proj_q_W = rearrange(proj_q, 'b c h w -> (b h) w c')
        
        proj_k = self.key_conv(x)
        proj_k_H = rearrange(proj_k, 'b c h w -> (b w) c h')
        proj_k_W = rearrange(proj_k, 'b c h w -> (b h) c w')
        
        proj_v = self.value_conv(x)
        proj_v_H = rearrange(proj_v, 'b c h w -> (b w) c h')
        proj_v_W = rearrange(proj_v, 'b c h w -> (b h) c w')
        
        atten_H = rearrange((torch.bmm(proj_q_H,proj_k_H) + self.inf(b,H,W)), '(b w) h h -> b h w h')
        atten_W = rearrange((torch.bmm(proj_q_W,proj_k_W) + self.inf(b,H,W)), '(b h) w w -> b h w w')
        attentionMap = self.softmax(torch.cat([atten_H,atten_W],dim=3))
        atten_H = rearrange(attentionMap[:,:,:,0:H], 'b h w h -> (b w) h h')
        atten_W = rearrange(attentionMap[:,:,:,H:H+W], 'b h w h -> (b h) w w')
        
        out_H = einsum('(b w) c h2, (b w h1 h2) -> b c h1 w', proj_v_H, atten_H)
        out_W = einsum('(b h) c w2, (b h) w1 w2 -> b c h w2', proj_v_W, atten_W)
        output = self.gamma*(out_H + out_W) + x
        return output
    
input_test = torch.randn(3,64,94,94,94).cuda()
model = CrissCrossAttention(in_dim=64).cuda()
out = model(input_test)
print(out.size())
# xx = torch.rand(3,96,128,128)

# maxxx = torch.max(xx,1)

""" SKNet """
class SKConv(nn.Module):
    def __init__(self,in_channel,branch=3,ratio=4,MinDim=16):
        super(SKConv,self).__init__()
        
        self.in_channel = in_channel
        self.branch = branch
        self.ratio = ratio
        self.hiddenDim = max(in_channel//ratio,MinDim)
        self.convs = nn.ModuleList([])
        for i in range(self.branch):
            self.convs.append(
                nn.Sequential(
                    nn.Conv3d(in_channel,in_channel,kernel_size=3+i*2,padding=1+i),
                    nn.BatchNorm3d(in_channel,momentum=0.9),
                    nn.ReLU()
                    )
                )
        self.gap = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Conv3d(in_channel,self.hiddenDim,kernel_size=(1,1,1),padding=0)
        self.fcs = nn.ModuleList([])
        for j in range(self.branch):
            self.fcs.append(
                nn.Sequential(
                    nn.Conv3d(self.hiddenDim,in_channel,kernel_size=(1,1,1),padding=0),
                    nn.BatchNorm3d(in_channel,momentum=0.9),
                    nn.ReLU()
                    )
                )
            
    def forward(self,x):
        
        batchSize, C, D, H, W = x.shape
        features = [conv(x) for conv in self.convs]
        features = torch.cat(features,dim=1)
        features = features.view(batchSize,self.branch,self.in_channel,D,H,W)
        merged_features = torch.sum(features,dim=1)
        squeezed_features = self.gap(merged_features)
        exicted_in_features = self.fc(squeezed_features)
        exicted_out_features = [fc(exicted_in_features) for fc in self.fcs]
        exicted_out_features = torch.cat(exicted_out_features,dim=1)
        exicted_out_features = exicted_out_features.view(batchSize,self.branch,self.in_channel,1,1,1)
        attention_maps = torch.softmax(exicted_out_features,dim=1)
        attention_features = torch.mul(features,attention_maps)
        attention_block = torch.sum(attention_features,dim=1)
        return attention_block



