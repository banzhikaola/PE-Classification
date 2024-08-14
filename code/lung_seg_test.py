# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 22:16:09 2023

@author: Administrator
"""

import monai
import numpy as np
import torch
import os
import glob

from monai import transforms
from monai.data import Dataset, DataLoader, create_test_image_3d, decollate_batch
from monai.inferers import sliding_window_inference

from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    SaveImage,
    ScaleIntensityd,
    FillHoles
)
from model import UNet3D
import matplotlib.pyplot as plt

seed = 14
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)



def showResult(val_input,val_output):
    val_images = val_input[0,0,:,:,:].cpu().detach().numpy()
    val_outputs = val_output[0,0,:,:,:].cpu().detach().numpy()
    for i in range(20,70):
        # plt.subplot(1,2,1)
        plt.imshow(val_images[i], cmap='gray')
        # plt.subplot(1,2,1)
        plt.imshow(val_outputs[i], cmap='blues', alpha=0.5)
        plt.show()

initialPath = ''
imagePath = os.listdir(initialPath)
normalPath = os.path.join(initialPath,'Normal')
patientPath = os.path.join(initialPath,'Patient')
normalimgs = [os.sep.join([normalPath, img]) for img in os.listdir(normalPath)]
patientimgs = [os.sep.join([patientPath,img]) for img in os.listdir(patientPath)]
allImages = normalimgs + patientimgs
originDatas = []
maskDatas = []
for singleFile in allImages:
    for Modalities in os.listdir(singleFile):
        originData = glob.glob(os.sep.join([singleFile, Modalities])+'\\[0-9]*.nii')
        originDatas.append(originData[0])
        
        maskData = glob.glob(os.sep.join([singleFile, Modalities])+'\\*.nrrd')
        maskDatas.append(maskData[0])
        
files = [{"img": img, "mask": mask} for img, mask in zip(originDatas, maskDatas)]

val_transforms=transforms.Compose([
    transforms.LoadImaged(keys=["img","mask"]),
    transforms.ToTensor(),
    transforms.EnsureChannelFirstd(keys=["img","mask"]),
    transforms.Orientationd(keys=["img","mask"],axcodes='IPL'),
    transforms.ScaleIntensityRangePercentilesD(keys="img", lower=0.1, upper=99.9, b_min=0, b_max=1),     
    transforms.NormalizeIntensityd(keys="img"),
    transforms.CenterSpatialCropd(keys=["img","mask"],roi_size=(96,96,96)),
    ])


batch_size = 1
val_ds=monai.data.Dataset(data=files,transform=val_transforms)
val_loader=DataLoader(val_ds,batch_size=batch_size,shuffle=False,pin_memory=True,drop_last=True)

dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
post_trans = Compose([Activations(sigmoid=False), AsDiscrete(threshold=0.5), FillHoles()])
# saver = SaveImage(output_ext=".nii.gz")
    # create UNet, DiceLoss and Adam optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet3D(in_channels=1,out_channels=1)
model = model.to(device)
model.load_state_dict(torch.load(''))


model.eval()
resultrecord = {'predMask', 'label'}
with torch.no_grad():
    val_images = None
    val_labels = None
    val_outputs = None
    val_losses = 0
    for val_data in val_loader:
        val_images, val_labels = val_data["img"].to(device), val_data["mask"].to(device)
        roi_size = (96, 96, 96)
        sw_batch_size = 1
        # val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
        val_outputs = model(val_images)
        val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
        val_labels = decollate_batch(val_labels)
        outputdir = os.path.dirname(val_images.meta['filename_or_obj'][0])
        resultrecord['predMask'].append(val_outputs[0][0,:,:,:])
        resultrecord['label'].append(val_labels)
        # saver_crop = SaveImage(output_dir=outputdir,output_postfix='centercrop',output_ext=".nii",separate_folder=False)
        # saver_crop(val_images[0,0,:,:,:])
        # saver_mask = SaveImage(output_dir=outputdir,output_postfix='mask',output_ext='.nii',separate_folder=False)
        # saver_mask(val_outputs[0][0,:,:,:])
        # with autocast():
        # val_outputs = model(val_images)
        # val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
        # compute metric for current iteration
        dice_metric(y_pred=val_outputs, y=val_labels)
        # showResult(val_images, val_outputs)
        
    # aggregate the final mean dice result
    metric = dice_metric.aggregate().item()
    print(f"current mean dice: {metric:.4f}")
    dice_metric.reset()
