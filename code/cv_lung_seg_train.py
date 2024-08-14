# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 16:53:39 2024

@author: Administrator
"""

import os
import pandas as pd
import shutil
import glob
import random
import torch
import numpy as np
from monai.utils import set_determinism
import monai
from monai import transforms
from monai.data import DataLoader, decollate_batch
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    RemoveSmallObjects,
    FillHoles
)
from monai.inferers import sliding_window_inference

from model import UNet3D, ResidualUNetSE3D, ResidualUNet3D

seed = 3407
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic=False
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.enabled=True
set_determinism(seed=seed)
os.environ['PYTHONHASHSEED'] = str(seed)



class Args():
    def __init__(self):
        self.model = UNet3D(in_channels=1, out_channels=1)
        self.batch_size = 8
        self.learning_rate = 1e-3
        self.loss_function = monai.losses.DiceLoss(sigmoid=True)
        
argument = Args()


dataPath = ''
dataList = ''
targetPath = ''

def dataset(dataPath, dataList):
    datas = os.listdir(dataPath)
    info_all = pd.read_excel(dataList)
    
    missingData = []
    AllData = []
    for data in datas:
        PerfusionNumber = os.path.basename(data)
        if not('-' in PerfusionNumber):
            PerfusionNumber = int(PerfusionNumber)
        PersonInfo = info_all.loc[info_all['exam'] == PerfusionNumber]
        if len(PersonInfo) == 0:
            missingData.append(data)
            continue

        perfusion = glob.glob(os.path.join(dataPath, data, 'Perfusion', '[0-9]*[0-9].nii'))
        ventilation = glob.glob(os.path.join(dataPath, data, 'Ventilation', '[0-9]*[0-9].nii'))
    
        pmask = glob.glob(os.path.join(dataPath, data, 'Perfusion', '*label*.nrrd'))
        vmask = glob.glob(os.path.join(dataPath, data, 'Ventilation', '*label*.nrrd'))
        AllData.append({'img':perfusion, 'mask':pmask})
        AllData.append({'img':ventilation, 'mask':vmask})
    random.shuffle(AllData)
        
    splitdatalist = np.array_split(AllData, 5)
    return splitdatalist, AllData
    
splitdataset, AllData = dataset(dataPath, dataList)

    
def CVData(datalist, section, folds):
    
    splitdatalist = datalist
    
    if section == "training":
        if len(folds) <= 1:
            raise ValueError("For Training Data, Fold Number Must More Than One !")
        returnlist = []
        for f in folds:
            for data in splitdatalist[f]:
                returnlist.append(data)
    else:
        if not(isinstance(folds, int)):
            raise ValueError("For Validation Data, Fold Number Must be 1!")
        returnlist = list(splitdatalist[folds])
        
    return returnlist

dataset_fold0_train = CVData(splitdataset, section="training", folds=[1, 2, 3, 4])
dataset_fold0_val = CVData(splitdataset, section="val", folds=0)

dataset_fold1_train = CVData(splitdataset, section="training",folds=[0, 2, 3, 4])
dataset_fold1_val = CVData(splitdataset, section="val", folds=1)

dataset_fold2_train = CVData(splitdataset, section="training",folds=[0, 1, 3, 4])
dataset_fold2_val = CVData(splitdataset, section="val",folds=2)

dataset_fold3_train = CVData(splitdataset, section="training",folds=[0, 1, 2, 4])
dataset_fold3_val = CVData(splitdataset, section="val",folds=3)

dataset_fold4_train = CVData(splitdataset, section="training",folds=[0, 1, 2, 3])
dataset_fold4_val = CVData(splitdataset, section="val",folds=4)

train_dataset = {'0':dataset_fold0_train, '1':dataset_fold1_train, '2':dataset_fold2_train,
            '3':dataset_fold3_train, '4':dataset_fold4_train}
val_dataset = {'0':dataset_fold0_val, '1':dataset_fold1_val, '2':dataset_fold2_val,
          '3':dataset_fold3_val, '4':dataset_fold4_val}


train_transform=transforms.Compose([
    transforms.LoadImaged(keys=["img", "mask"], image_only=True),
    transforms.ToTensor(),
    transforms.EnsureChannelFirstd(keys=["img","mask"]),
    transforms.Orientationd(keys=["img","mask"],axcodes='IPL'),
    transforms.ScaleIntensityRangePercentilesD(keys="img", lower=0.1, upper=99.9, b_min=0, b_max=1),
    transforms.NormalizeIntensityd(keys="img"),    
    transforms.CenterSpatialCropd(keys=["img","mask"],roi_size=(96,96,96)),
    transforms.RandRotated(keys=["img","mask"], prob=0.4, range_x=[0.4, 0.4], mode=['bilinear','nearest']),
    transforms.RandZoomd(keys=["img","mask"],prob=0.1,min_zoom=0.9,max_zoom=1.1,mode=['area', 'nearest'])
        ])

val_transforms=transforms.Compose([
    transforms.LoadImaged(keys=["img","mask"],image_only=True),
    transforms.ToTensor(),
    transforms.EnsureChannelFirstd(keys=["img","mask"]),
    transforms.Orientationd(keys=["img","mask"],axcodes='IPL'),
    transforms.ScaleIntensityRangePercentilesD(keys="img", lower=0.1, upper=99.9, b_min=0, b_max=1),
    transforms.NormalizeIntensityd(keys="img"),        
    transforms.CenterSpatialCropd(keys=["img","mask"],roi_size=(96,96,96)),
    ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = argument.model
model = model.to(device)


dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
post_trans = Compose([Activations(sigmoid=False), AsDiscrete(threshold=0.5), RemoveSmallObjects(),FillHoles()])

optimizer = torch.optim.Adam(model.parameters(), argument.learning_rate)
loss_function = argument.loss_function
cosine_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=200)


for fold in range(5):
    
    train_ds = list(train_dataset[str(fold)])
    val_ds = list(val_dataset[str(fold)])
    
    train_ds=monai.data.Dataset(data=train_ds,transform=train_transform)
    val_ds=monai.data.Dataset(data=val_ds,transform=val_transforms)
    
    train_loader=DataLoader(train_ds,batch_size=argument.batch_size,shuffle=True,pin_memory=True)
    val_loader=DataLoader(val_ds,batch_size=argument.batch_size,shuffle=False,pin_memory=True)
    
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    metric_values = list()
    patience = 30
    es = 0
    for epoch in range(200):
        print("-" * 30)
        print(f"epoch {epoch + 1}/{200}")
        model.train()
        epoch_loss = 0
        step = 0
        for  batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["mask"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_loader)
            if step % 20 ==0:
                print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
        model.eval()
        with torch.no_grad():
            val_images = None
            val_labels = None
            val_outputs = None
            val_losses = 0
            for val_data in val_loader:
                val_images, val_labels = val_data["img"].to(device), val_data["mask"].to(device)
                roi_size = (96, 96, 96)
                sw_batch_size = argument.batch_size
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                # val_outputs = model(val_images)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)
                
            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            # reset the status for next validation round
            dice_metric.reset()
            
            metric_values.append(metric)
            print(f"current epoch: {epoch+1}, current mean dice: {metric:.4f}")
            if metric > best_metric:
                es=0
                best_metric = metric
                best_metric_epoch = epoch + 1
                if metric > 0.93:
                    best_metric = round(best_metric,4)
                    torch.save(model.state_dict(), f"best_metric_modelfold{fold}_" +str(best_metric)+"_.pth")
                    print("saved new best metric model")
                    print(f"current epoch: {epoch+1} current mean dice: {metric} best mean dice: {best_metric} at epoch {best_metric_epoch}")
            else:
                es += 1
            if es == patience:
                print('Early stopping at epoch {}...'.format(epoch))
               
                break
        cosine_schedule.step()