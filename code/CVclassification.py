# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:27:27 2024

@author: Administrator
"""

import os
import random
import numpy as np
import glob

import torch.optim as optim
import torch
import torch.nn as nn

import pandas as pd

import monai
from monai.data import DataLoader,decollate_batch
from monai import transforms
from monai.metrics import ROCAUCMetric
from monai.transforms import Activations,Compose,AsDiscrete
from monai.utils import set_determinism
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,roc_curve
from sklearn.model_selection import KFold

from twomodalityvit import ViT
import visdom 
viz = visdom.Visdom()


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

class Args:
    def __init__(self):
        self.batch_size=4
        self.lr=3e-5
        self.epochs=20
        self.patience=30
        self.model = ViT(in_channels=2, img_size=(96, 96, 96), patch_size=[8, 8, 8], classification=True, dropout_rate=0.0)

        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
args=Args()

def dataset():
    DataPath = 'datapath'
    InfoPath = 'datainformation'
    AllData = []
    
    datas = [os.path.join(DataPath, subject) for subject in os.listdir(DataPath)]
    info_all = pd.read_excel(InfoPath)
    
    for data in datas:
        PerfusionNumber = os.path.basename(data)
        if not('-' in PerfusionNumber):
            PerfusionNumber = int(PerfusionNumber)
        PersonInfo = info_all.loc[info_all['检查号'] == PerfusionNumber]

        label = list(PersonInfo['组别'])[0]
        if np.isnan(label):
            continue
        perfusion = glob.glob(os.path.join(data, 'Perfusion', '[0-9]*[0-9].nii'))
        ventilation = glob.glob(os.path.join(data, 'Ventilation', 'rnmi*_masked.nii'))
        vp = glob.glob(os.path.join(data, 'VP_quotient', '[0-9]*.nii'))
        pmask = glob.glob(os.path.join(data, 'Perfusion', '*mask.nii'))
        vmask = glob.glob(os.path.join(data, 'Ventilation', 'rnmi*mask.nii'))
        AllData.append({'p':perfusion, 'v':ventilation, 'vp':vp, 'pmask':pmask,'vmask':vmask, 'label':int(label)})

    return AllData
AllData = dataset()


train_transform=transforms.Compose([
    transforms.LoadImaged(keys=["p","v","vp","pmask","vmask"],image_only=False),
    transforms.ToTensor(),
    transforms.EnsureChannelFirstd(keys=["p","v","vp","pmask","vmask"]),
    transforms.CenterSpatialCropd(keys=["p","v","vp","pmask","vmask"], roi_size=(96,96,96)),

    transforms.Orientationd(keys=["p","v","vp","pmask","vmask"],axcodes='IPL'),
    transforms.ScaleIntensityRangePercentilesd(keys=["v","p"], lower=0.1, upper=99.9,b_max=1,b_min=0),

    transforms.RandRotated(keys=["p","v","vp","pmask","vmask"],range_x=[0.4,0.4],prob=0.2),
    transforms.RandZoomd(keys=["p","v","vp","pmask","vmask"],prob=0.2),
    transforms.RandAxisFlipd(keys=["p","v","vp","pmask","vmask"]),
  ])


val_transforms=transforms.Compose([
    transforms.LoadImaged(keys=["p","v","vp","pmask","vmask"],image_only=False),
    transforms.ToTensor(),
    transforms.EnsureChannelFirstd(keys=["p","v","vp","pmask","vmask"]),
    transforms.CenterSpatialCropd(keys=["p","v","vp","pmask","vmask"], roi_size=(96,96,96)),

    transforms.Orientationd(keys=["p","v","vp","pmask","vmask"],axcodes='IPL'),
    transforms.ScaleIntensityRangePercentilesd(keys=["v","p"], lower=0.1, upper=99.9,b_max=1,b_min=0),
    ])





kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
train_data = []
val_data = []
for fold, (train_idx, val_idx) in enumerate(kfold.split(AllData)):
    train_paths = np.array(AllData)[train_idx].tolist()
    val_paths = np.array(AllData)[val_idx].tolist()
    train_data.append({'fold':fold, 'data':train_paths})
    val_data.append({'fold':fold, 'data':val_paths})



trainfold = 0

loss_opts = dict(title="Train and Validation Loss",showlegend=True, legend=['Train Loss', 'Validation Loss'],
            xlabel='Epoch', ylabel='Loss', markercolor=np.array([[0,0,255], [255,0,0]]),
            linecolor=np.array([[0,0,255], [255,0,0]]))
win_loss = viz.line(X=np.column_stack((np.array([0]),np.array([0]))), Y=np.column_stack((np.array([0]),np.array([0]))), opts=loss_opts)
accuracy_opts = dict(title="Train and Validation Accuracy",showlegend=True, legend=['Train Accuarcy', 'Validation Accuracy'],
                  xlabel='Epoch',ylabel='Accuracy',markercolor=np.array([[0,0,255], [255,0,0]]),
                  linecolor=np.array([[0,0,255],[255,0,0]]))
win_acc = viz.line(X=np.column_stack((np.array([0]),np.array([0]))), Y=np.column_stack((np.array([0]),np.array([0]))),opts=accuracy_opts)

win_precision = viz.line(X=np.array([0]), Y=np.array([0]), opts=dict(title='Validation Precision'))
win_recall = viz.line(X=np.array([0]), Y=np.array([0]), opts=dict(title='Validation Recall'))
win_specificity = viz.line(X=np.array([0]), Y=np.array([0]), opts=dict(title='Validation Specificity'))
win_f1 = viz.line(X=np.array([0]), Y=np.array([0]), opts=dict(title='Validation f1-score'))


train_ds=monai.data.Dataset(data=train_data[trainfold]['data'],transform=train_transform)
val_ds=monai.data.Dataset(data=val_data[trainfold]['data'],transform=val_transforms)

train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,pin_memory=True)
val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,pin_memory=True)

num_traindata = len(train_loader.dataset)
post_pred = Compose([Activations(softmax=True)])
post_label = Compose([AsDiscrete(to_onehot=2)])
auc_metric=ROCAUCMetric()

model=args.model
model.to(args.device)


entropy_function1 = nn.CrossEntropyLoss().to(args.device)
optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-3,betas=[0.9,0.99],eps=1e-8)
cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=200)

best_metric = -1
best_metric_epoch = -1
auc_result = -1
best_auc = -1

es=0
for epoch in range(0, args.epochs ):

    """ Start Training"""    
    print("-" * 50)
    print(f"Epoch: {epoch + 1 }/{args.epochs }")
    train_loss = 0 
    step = 0
    correct = 0
    b = 0.02
    print('Train Data:{}, Epoch_len:{}'.format(num_traindata, len(train_loader)))
    model.train()
    out_logit = torch.tensor([], dtype=torch.float32, device=args.device)
    train_label = torch.tensor([], dtype=torch.long, device=args.device)
    for batch_idx,batch_data in enumerate(train_loader) : 
        step +=1 
        p, v, vp, labels = batch_data["p"].to(args.device), batch_data["v"].to(args.device),batch_data["vp"].to(args.device),batch_data["label"].to(args.device)
        pmask, vmask = batch_data["pmask"].to(args.device), batch_data["vmask"].to(args.device)
        optimizer.zero_grad()
        train_output, _ = model(p, v, pmask, vmask)
        cls_loss = entropy_function1(train_output, labels)
        loss = cls_loss
        loss.backward()
        optimizer.step()
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        out_logit = torch.cat([out_logit, train_output], dim=0)
        train_label = torch.cat([train_label, labels], dim=0)
        train_loss += loss.item()
        if (batch_idx + 1) % 8 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLearning_Rate:{:.9f}'.format(
            epoch+1, (batch_idx + 1) * args.batch_size, len(train_loader.dataset),
            100. * (batch_idx + 1) / len(train_loader), loss.item(),lr))
    train_loss /= step 
    print(f"Epoch {epoch +1 }\tAverage Loss: {train_loss:.4f}")
    acc_value = torch.eq(out_logit.argmax(dim=1),train_label)
    train_acc = acc_value.sum().item() / len(acc_value)
    print('Epoch {}\tTrain Accuracy:{}'.format(epoch+1,train_acc))
    print("\n")
    
    """ Start Validation """
    val_loss = 0
    print('Validation Data:{}, Epoch_len:{}'.format(len(val_loader.dataset), len(val_loader)))
    model.eval()
    with torch.no_grad():
        y_pred = torch.tensor([], dtype=torch.float32, device=args.device)
        y = torch.tensor([], dtype=torch.long, device=args.device)
        for val_data in val_loader:
            p,v,vp, val_labels = val_data["p"].to(args.device), val_data["v"].to(args.device),val_data["vp"].to(args.device),val_data["label"].to(args.device)
            pmask, vmask = val_data["pmask"].to(args.device), val_data["vmask"].to(args.device)
            val_output, _ = model(p, v, pmask, vmask)
            y_pred = torch.cat([y_pred, val_output], dim=0)       
            y = torch.cat([y, val_labels], dim=0)
            loss_cls = entropy_function1(val_output,val_labels)
            loss = loss_cls

            val_loss += loss.item()

        y_pred_output = y_pred.argmax(dim=1)
        average_loss = val_loss / len(val_loader)
        acc_value = torch.eq(y_pred_output,y)
        acc_metric = acc_value.sum().item() / len(acc_value)

        tn, fp, fn, tp = confusion_matrix(y.cpu().numpy(), y_pred_output.detach().cpu().numpy()).ravel()
        precision = precision_score(y.cpu().numpy(),  y_pred_output.detach().cpu().numpy(),zero_division=0.0)
        recall = recall_score(y.cpu().numpy(),  y_pred_output.detach().cpu().numpy())
        f1 = f1_score(y.cpu().numpy(),  y_pred_output.detach().cpu().numpy())
        specificity = tn / (tn + fp)
        print('Val set: Average loss: {:.4f}, Accuracy: {} ({:.0f}%)\n'.format(
            average_loss, acc_metric, 100 * acc_metric))
        y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
        y_pred_onehot = [post_label(i) for i in decollate_batch(y_pred_output, detach=False)]
        y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
        auc_metric(y_pred_act, y_onehot)
        auc_result = auc_metric.aggregate()
        auc_metric.reset()
        del y_pred_act, y_onehot
        if acc_metric >= best_metric:
            es=0
            best_loss = average_loss
            best_metric = acc_metric
            best_auc = auc_result
            best_metric_epoch = epoch + 1
            
            print(
                "current epoch: {} current accuracy: {:.4f} current AUC: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                    epoch + 1, acc_metric, best_auc, best_metric, best_metric_epoch
                )
            )
        if acc_metric >=0.8:
            acc_metric = round(acc_metric,4)
            modelPath = "_vit_"+str(acc_metric)+'_'+str(trainfold)+"fold.pth"
            # torch.save(model, modelPath)
            print("saved new best metric model")

        es = es + 1
      
    if es == args.patience:
        print(f'Early stopping in Fold :{trainfold}, at epoch: {epoch + 1}...')

        break   
    cosine_schedule.step()
    
    viz.line(Y=np.column_stack((np.array([train_loss]),np.array([average_loss]))), X=np.column_stack((np.array([epoch+1]),np.array([epoch+1]))), win=win_loss, update='append')
    viz.line(Y=np.column_stack((np.array([train_acc]),np.array([acc_metric]))), X=np.column_stack((np.array([epoch+1]),np.array([epoch+1]))), win=win_acc, update='append')
    viz.line(Y=np.array([precision]), X=np.array([epoch+1]), win=win_precision, update='append')
    viz.line(Y=np.array([recall]), X=np.array([epoch+1]), win=win_recall, update='append')
    viz.line(Y=np.array([specificity]), X=np.array([epoch+1]), win=win_specificity, update='append')
    viz.line(Y=np.array([f1]), X=np.array([epoch+1]), win=win_f1, update='append')
