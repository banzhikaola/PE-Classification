# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:40:25 2023

@author: Administrator
"""

import os
import glob
import SimpleITK as sitk
import itk
root = ""
registerFiles = []
files = os.listdir(root)
for i in range(len(files)):
    registerFiles.append(os.path.join(root, files[i]))
from tqdm import tqdm

unprocess = 0
# people = registerFiles[462]
for people in tqdm(registerFiles):
    images = os.listdir(people)
    perfusion = glob.glob(os.path.join(people,'Perfusion')+'\\[0-9]*[0-9].nii')[0]
    ventilation = glob.glob(os.path.join(people,'Ventilation')+'\\[0-9]*[0-9].nii')[0]
    outputdir = glob.glob(os.path.join(people, 'Ventilation')+'\\rnmi*.nii')
    if len(outputdir) != 0:
        unprocess = 1 + unprocess
        continue
    filename = os.path.basename(ventilation)
    ventilation_number = filename.split('.')[0]
    outputdir = os.path.join(people, 'Ventilation') + '\\rnmi_tril' + ventilation_number + '_centercrop_masked.nii'
    
    fixedObject = sitk.ReadImage(perfusion)
    fixed_image = sitk.GetArrayFromImage(fixedObject)
    
    movingObject = sitk.ReadImage(ventilation)
    moving_image = sitk.GetArrayFromImage(movingObject)
    
    
    # Import Default Parameter Map
    parameter_object = itk.ParameterObject.New()
    default_rigid_parameter_map = parameter_object.GetDefaultParameterMap('rigid')
    parameter_object.AddParameterMap(default_rigid_parameter_map)

    
    result_image,result_transform_parameters = itk.elastix_registration_method(
        fixed_image, moving_image,
        parameter_object=parameter_object,
        log_to_console=True
        )
    
    outputImage = sitk.GetImageFromArray(result_image)
    outputImage.SetSpacing(movingObject.GetSpacing())
    outputImage.SetOrigin(movingObject.GetOrigin())
    outputImage.SetDirection(movingObject.GetDirection())
    sitk.WriteImage(outputImage,outputdir)
    print('save image:' + outputdir)