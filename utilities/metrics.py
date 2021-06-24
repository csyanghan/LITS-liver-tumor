import numpy as np
import torch
import torch.nn.functional as F


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy().copy()

    output_ = output > 0.5
    target_ = target > 0.5

    pred = output_[:,0,:,:] + output_[:,1,:,:]
    total_target = target_[:,0,:,:] + target_[:,1,:,:]
    intersection = (pred & total_target).sum(axis=(1,2))
    union = (pred | total_target).sum(axis=(1,2))

    return np.mean((intersection + smooth) / (union + smooth))


def dice_coef(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy().copy()

    batch_size = target.shape[0]
    input_1 = output[:,0,:,:]
    input_1 = input_1 > 0.5
    input_1 = input_1.reshape(batch_size,-1)
    
    input_2 = output[:,1,:,:]
    input_2 = input_2 > 0.5
    input_2 = input_2.reshape(batch_size,-1)

    target_1 = target[:,0,:,:]
    target_1 = target_1.reshape(batch_size,-1)
    target_2 = target[:,1,:,:]
    target_2 = target_2.reshape(batch_size,-1)

    intersection_1 = (input_1 * target_1)
    intersection_2 = (input_2 * target_2)

    dice_1 = (2. * intersection_1.sum(1) + smooth) / (input_1.sum(1) + target_1.sum(1) + smooth)
    dice_2 = (2. * intersection_2.sum(1) + smooth) / (input_2.sum(1) + target_2.sum(1) + smooth)

    return dice_1.sum() / batch_size ,dice_2.sum() / batch_size

