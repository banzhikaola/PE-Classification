# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 10:37:36 2023

@author: Administrator
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from monai import transforms
from monai.transforms import Activations,Compose,AsDiscrete
import monai
from monai.data import DataLoader,CSVSaver
from monai.data import DataLoader,decollate_batch
from monai.metrics import ROCAUCMetric
import torch.nn as nn
import torch
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_curve
from sklearn.utils.extmath import stable_cumsum
import pandas as pd
from monai.visualize.occlusion_sensitivity import OcclusionSensitivity
from monai.visualize.class_activation_maps import GradCAMpp
from monai.visualize.utils import blend_images, matshow3d

from twomodalityvit import ViT

DataPath = ''
InfoPath = ''


def dataset(DataPath, InfoPath):
    
    AllData = []
    
    datas = [os.path.join(DataPath, subject) for subject in os.listdir(DataPath)]
    info_all = pd.read_excel(InfoPath)
    
    for data in datas:
        PerfusionNumber = os.path.basename(data)
        if not('-' in PerfusionNumber):
            PerfusionNumber = int(PerfusionNumber)
        PersonInfo = info_all.loc[info_all['exam'] == PerfusionNumber]

        label = list(PersonInfo['group'])[0]
        if np.isnan(label):
            continue

        perfusion = glob.glob(os.path.join(data, 'Perfusion', '[0-9]*[0-9].nii'))
        ventilation = glob.glob(os.path.join(data, 'Ventilation', 'rnmi*_masked.nii'))
        vp = glob.glob(os.path.join(data, 'VP_quotient', '[0-9]*.nii'))
        pmask = glob.glob(os.path.join(data, 'Perfusion', '*mask.nii'))
        vmask = glob.glob(os.path.join(data, 'Ventilation', 'rnmi*mask.nii'))
        AllData.append({'p':perfusion, 'v':ventilation, 'vp':vp, 'pmask':pmask,'vmask':vmask, 'label':int(label)})

    return AllData

alldatas = dataset(DataPath, InfoPath)


test_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["p","v","vp","pmask","vmask"],image_only=False),
    transforms.ToTensor(),
    transforms.EnsureChannelFirstd(keys=["p","v","vp","pmask","vmask"]),
    transforms.CenterSpatialCropd(keys=["p","v","vp","pmask","vmask"], roi_size=(96,96,96)),
    transforms.Orientationd(keys=["p","v","vp","pmask","vmask"],axcodes='IPL'),
    transforms.ScaleIntensityRangePercentilesd(keys=["v","p"], lower=0.1, upper=99.9,b_max=1,b_min=0),
    ])

# # create a validation data loaderval
test_ds = monai.data.Dataset(data=alldatas, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=1,pin_memory=True)


post_pred = Compose([Activations(softmax=True)])
post_label = Compose([AsDiscrete(to_onehot=2)])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sumAccuracy = []
sumRecall = []
sumSpecificity = []
sumF1Score = []
sumAUC = []
sumnet_benefit = []
sumall_benefit = []
sumthreshold = []
sumpred = []
sumlabel = []
sumfeatures = []
sumlabels = []
def LoadModel(para_path,trained):
    
    model = ViT(in_channels=2, img_size=(96, 96, 96), patch_size=[8, 8, 8], classification=True, dropout_rate=0.0)
    if trained:
        net_dict = model.state_dict()
        pretrainmodel = torch.load(para_path)
        pretrain = pretrainmodel.state_dict()
        pretrain_dict = {k: v for k, v in pretrain.items() if k in net_dict.keys()}
        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)
    return model

modelpath = ""

def show_perfusion(p):
    meta = os.path.basename(p.meta['filename_or_obj'][0])
    p = p[0,0,:,:,:].cpu().numpy()
    for i in range(24,72):
        plt.title(meta)
        plt.imshow(p[i],cmap='gray')
        plt.show()


""" Occlusion Sensitivity Analysis """
def occlusion_analysis(model,p,v,pmask,vmask):
    occ_sens = OcclusionSensitivity(nn_module=model,n_batch=1,mask_size=12,overlap=0.6,activate=True)
    occ_map, most_probable_class = occ_sens(x=p,x2=v,pmask=pmask,vmask=vmask)
    most_probable_class = most_probable_class[0,0,:,:,:].cpu().numpy()
    most_probable_class1 = np.expand_dims(most_probable_class, axis=0)
    blend_cam = blend_images(image=p[0,:,:,:,:], label=occ_map[0,1:2,:,:,:].cpu().numpy(),alpha=0.5,transparent_background=True)
    blend_cam = np.transpose(blend_cam, (1,2,3,0))
    meta = os.path.basename(p.meta['filename_or_obj'][0])
    for i in range(24,72):
        plt.title(meta)
        plt.imshow(blend_cam[i])
        plt.show()
    return occ_map
        
for fold in range(1):
    
    auc_metric=ROCAUCMetric()
    modeldict = glob.glob(os.path.join(modelpath, 'alltrain_vit_.pth'))[0]
    model = torch.load(modeldict)
    # model = LoadModel(modeldict,trained=True)
    model.to(device)
    model.eval()
    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y = torch.tensor([], dtype=torch.long, device=device)
        # saver = CSVSaver(output_dir="./output")
        foldfeature = []
        for number, test_data in enumerate(test_loader):
            
            p,v,vp, test_labels = test_data["p"].to(device), test_data["v"].to(device),test_data["vp"].to(device),test_data["label"].to(device)
            pmask, vmask = test_data["pmask"].to(device), test_data["vmask"].to(device)
            # if os.path.basename(p.meta['filename_or_obj'][0]) == '151569-3.nii':
                # occ_map,most_probable_class = occlusion_analysis(model, p, v, pmask, vmask)
            test_outputs = model(p,v,pmask,vmask)
            y_pred = torch.cat([y_pred, test_outputs], dim=0)       
            y = torch.cat([y, test_labels], dim=0)
            # foldfeature.extend(test_feature.cpu().numpy())
        y_pred_output = y_pred.argmax(dim=1)
    
        acc_value = torch.eq(y_pred_output,y)
        acc_metric = acc_value.sum().item() / len(acc_value)
    
        tn, fp, fn, tp = confusion_matrix(y.cpu().numpy(), y_pred_output.detach().cpu().numpy()).ravel() 
        precision = precision_score(y.cpu().numpy(),  y_pred_output.detach().cpu().numpy())
        recall = recall_score(y.cpu().numpy(),  y_pred_output.detach().cpu().numpy())
        f1 = f1_score(y.cpu().numpy(),  y_pred_output.detach().cpu().numpy())
        specificity = tn / (tn + fp)
        
        y_pred_proba = torch.softmax(y_pred, dim=1)[:,1].cpu().numpy()
        sumpred.append(y_pred_proba)
        sumlabel.append(y.cpu().numpy())
        
        
        y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
        y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
        auc_metric(y_pred_act, y_onehot)
        auc_result = auc_metric.aggregate()
        
        auc_metric.reset()
        # del y_pred_act, y_onehot
        
        sumAccuracy.append(acc_metric)
        sumRecall.append(recall)
        sumSpecificity.append(specificity)
        sumF1Score.append(f1)
        sumAUC.append(auc_result)
        
    # sumfeatures.append(foldfeature)
    sumlabels.append(y.cpu().numpy())
        
def MeanAndStd(sum_metric):
    averagemetric = np.mean(sum_metric)
    stdmetric = np.std(sum_metric)
    return averagemetric, stdmetric

averageAccuracy, stdAccuracy = MeanAndStd(sumAccuracy)

averageRecall, stdRecall = MeanAndStd(sumRecall)

averageSpec, stdSpec = MeanAndStd(sumSpecificity)

averageF1, stdF1 = MeanAndStd(sumF1Score)

averageAUC, stdAUC = MeanAndStd(sumAUC)



print("Following Result: ")
print(f'Accuracy: {averageAccuracy:.3f} ± {stdAccuracy:.3f}')
print(f"Recall is: {averageRecall:.3f} ± {stdRecall:.3f}")
print(f"Specitfity is :{averageSpec:.3f} ± {stdSpec:.3f}")
print(f"F1 Score is: {averageF1:.3f} ± {stdF1:.3f}")
print(f"AUC Result is  :{averageAUC:.3f} ± {stdAUC:.3f}")
