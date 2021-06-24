# -*- coding: utf-8 -*-

import os
import warnings
from time import time

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from scipy.ndimage import label
from torch.autograd import Variable
from tqdm import tqdm

from dataset.dataset import Dataset
from net import Unet
from utilities.utils import count_params

#import ttach as tta

test_ct_path = 'LITS2017/CT'   #需要预测的CT图像
seg_result_path = 'LITS2017/seg' #需要预测的CT图像标签，如果要在线提交codelab，需要先得到预测过的70例肝脏标签


def dice(pred, truth):
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy().copy()
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy().copy()
    smooth = 1e-5
    intersection = (pred * truth)
    dice = (2. * intersection.sum() + smooth) / (pred.sum() + truth.sum() + smooth)
    return dice

def main():

    model = Unet.U_Net()

    model = torch.nn.DataParallel(model).cuda()

    model.load_state_dict(torch.load('models/Unet/2021-06-19-22-29-03/epoch154-0.9404-0.6316_model.pth'))
    model.eval()
    # model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
    
    log = pd.DataFrame(index=[], columns=[
        'file','slice', 'dice_liver', 'dice_tumor'
    ])

    for file_index, file in enumerate(range(100,131)):

        # if file.replace('volume', 'segmentation').replace('nii','nii.gz') in os.listdir(pred_path):
        #     print('already predict {}'.format(file))
        #     continue
        # 将CT读入内存
        file_name = 'volume-{}.nii'.format(file)
        ct = sitk.ReadImage(os.path.join(test_ct_path, file_name), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)

        mask = sitk.ReadImage(os.path.join(seg_result_path, file_name.replace('volume', 'segmentation')), sitk.sitkUInt8)
        mask_array = sitk.GetArrayFromImage(mask)

        print('start predict file:',file_name)

        ct_array[ct_array > 200] = 200
        ct_array[ct_array < -200] = -200

        ct_array = ct_array.astype(np.float32)
        ct_array = ct_array / 200

        # 找到肝脏区域开始和结束的slice，并各向外扩张slice
        z = np.any(mask_array, axis=(1, 2))
        start_slice, end_slice = np.where(z)[0][[0, -1]]

        start_slice = max(0, start_slice - 10)
        end_slice = min(mask_array.shape[0]-1, end_slice + 10)

        ct_crop = ct_array[start_slice:end_slice+1,32:480,32:480]
        mask_crop = mask_array[start_slice:end_slice+1, 32:480, 32:480]

        slice_predictions = np.zeros((ct_array.shape[0],512,512),dtype=np.int16)
        
        dice_liver = 0
        dice_tumor = 0

        with torch.no_grad():
            for n_slice in range(ct_crop.shape[0]-3):
                ct_tensor = torch.FloatTensor(ct_crop[n_slice: n_slice + 3]).cuda()
                ct_tensor = ct_tensor.unsqueeze(dim=0)
                # print('ct_tensor',ct_tensor.shape,n_slice)
                output = model(ct_tensor)
                output = torch.sigmoid(output).data.cpu().numpy()
                probability_map = np.zeros([1, 448, 448], dtype=np.uint8)
                #预测值拼接回去
                # i = 0
                for idz in range(output.shape[1]):
                    for idx in range(output.shape[2]):
                        for idy in range(output.shape[3]):
                            if (output[0,0, idx, idy] > 0.65):
                                probability_map[0, idx, idy] = 1        
                            if (output[0,1, idx, idy] > 0.5):
                                probability_map[0, idx, idy] = 2

                slice_predictions[n_slice+start_slice+1,32:480,32:480] = probability_map        

                liver_truth = mask_crop[n_slice].copy()
                liver_pred = probability_map[0].copy()
                liver_truth[liver_truth==2] = 1
                liver_pred[liver_pred==2] = 1
                dice_liver += dice(liver_pred, liver_truth)

                tumor_truth = mask_crop[n_slice].copy()
                tumor_pred = probability_map[0].copy()
                tumor_truth[tumor_truth==1]=0
                tumor_truth[tumor_truth==2]=1
                tumor_pred[tumor_pred==1]=0
                tumor_pred[tumor_pred==2]=1
                dice_tumor += dice(tumor_pred, tumor_truth)

            torch.cuda.empty_cache()
        
        num = ct_crop.shape[0]-3
        tmp = pd.Series([
            file_name,
            num,
            dice_liver / num,
            dice_tumor / num
        ], index=['file','slice', 'dice_liver', 'dice_tumor'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('test.csv', index=False)
                        
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    main()
                
