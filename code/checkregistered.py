# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 21:37:49 2023

@author: hemiao
检查配准图像的配准效果
"""

import os 
import SimpleITK as sitk
import matplotlib.pyplot as plt 
import glob
import pandas as pd

# #选择图像文件
# patientList = pd.read_excel('../register_record.xlsx')
# patientList = patientList['inference_Patient']

testData = "D:\\biaozhushuju\\InternalTestData"

def LoadData(path):
    registedFiles = []
    files = os.listdir(path)
    for i in range(len(files)):
        filesPath = os.path.join(path,files[i])
        if os.path.isdir(filesPath):
            registedFiles.append(filesPath)
    return registedFiles

registedFilestest = LoadData(testData)
# registedFilestrain = LoadData(trainData)
registedFiles =  registedFilestest
# for i in range(len(patientList)):
#     filesPath = os.path.join(testData,str(patientList[i]))
#     if os.path.isdir(filesPath):
#         registedFiles.append(filesPath)

                
def GetArray(path):
    obj = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(obj)
    print(f"dataPath:{path}")
    return array

from matplotlib.colors import ListedColormap 
import numpy as np
def apply_colormap(data):
    z=np.zeros((255,3))

    a=[i for i in range(0,255,2)]
    a.append(255)
    z[64:193,0]=a
    z[193:,0]=255
    b=[i for i in range(0,126,2)]
    
    z[:63,1]=b
    z[63:124,1]=[i for i in range(122,0,-2)]
    
    c=[i for i in range(0,255,2)]
    c.append(255)
    c.append(255)
    c.append(255)
    z[124:,1]=c
    
    
    z[:128,2]=[i for i in range(0,255,2)]
    z[128:192,2]=[i for i in range(255,0,-4)]
    z[192:,2]=[i for i in range(0,252,4)]
    
    cmap = ListedColormap(z/255., name='my_colormap', N=255)
    

    img_norm = (data - data.min()) / (data.max() - data.min())
    img_rgba = cmap(img_norm)
    img_rgba = img_rgba[:,:,:,:3]
    img_rgb = img_rgba

    img_rgb = np.uint16(255 * img_rgb)
    return img_rgb   
    
def LoadImage(num):
    images = os.listdir(registedFiles[num])
    perfusionPath = os.path.join(registedFiles[num],images[0])
    perfusionImage = glob.glob(perfusionPath + '\\*.nii')
    print("******"+str(num))
    perfusionArray = GetArray(perfusionImage[0])
    
    ventilationPath = os.path.join(registedFiles[num],images[1])
    vImage = glob.glob(ventilationPath + '\\*[0-9].nii')
    vArray = GetArray(vImage[0])
    # perfusionArray_nonzero = np.where(perfusionArray != 0, perfusionArray, 1)  
    
    perfusionArray = apply_colormap(perfusionArray)
    vArray = apply_colormap(vArray)
    
    regImages = glob.glob(ventilationPath + '\\rnmi_tril*.nii')
    regArray = GetArray(regImages[0])
    print("\n")
    # vp = regArray / perfusionArray_nonzero
    # vp = apply_colormap(vp)
    # regArray = apply_colormap(regArray)
                
    return perfusionArray, vArray, regArray,num


# for j in range(400,552):
perfusionArray, vArray, regArray, num = LoadImage(6)
# vp = vArray / perfusionArray


method = {'r':regArray}

for i in range(16,113):
    plt.subplot(221)
    plt.title("Perfusion Image" + str(num) )
    plt.tick_params(axis='both', which='both', labelsize=7)
    plt.imshow(perfusionArray[i])
    # plt.imshow(perfusionArray[i],cmap='gray')
    plt.subplot(222)
    plt.title("Registered Ventilation")
    plt.tick_params(axis='both', which='both', labelsize=7)
    # plt.imshow(regArray[i])
    plt.imshow(method['r'][i],cmap='gray')
    # plt.subplot(223)
    # plt.imshow(vp[i])
    plt.subplot(224)
    plt.title("Ventilation Image",fontsize=10)
    plt.tick_params(axis='both', which='both', labelsize=7)
    # plt.imshow(vArray[i],cmap='gray')
    plt.imshow(vArray[i])
    plt.show()

# #把配准数据的顺序编号和灌注图像检查号关联起来存到DataSearch表格中
# def extract_digit(column):
#     return column.str.extract(r'(\d+)', expand=False)

# recordFile = pd.read_excel('D:\\patient_descirbe\\DataSearch.xlsx')
# recordFile['digit_part'] = extract_digit(recordFile['测试数据配准检查'])
# manualRegisterList = list(recordFile['digit_part'])
# # manualRegisterList = manualRegisterList[:22]
# x = 0
# for i in range(len(registedFiles)):
#     data = registedFiles[i]
#     ExamNumber = os.path.basename(data)
#     if str(i) in manualRegisterList:
#         if '-' not in ExamNumber:
#             ExamNumber = int(ExamNumber)
#             #在dataframe中，检查号没有-的是字符，按照字符类型进行检索。纯数字的是整数型，按照整数类型检索
#         patient = recordFile.loc[recordFile['检查号']==ExamNumber]
#         recordFile.loc[patient.index, '配准方法'] = 1

