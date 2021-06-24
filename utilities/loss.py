import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MulticlassDiceLoss(nn.Module):
    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target):
        # input = torch.sigmoid(input)
        input = input.clamp (min=-10, max=10)
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)

        input_1 = input[:,0,:,:]
        input_2 = input[:,1,:,:]
        target_1 = target[:,0,:,:]
        target_2 = target[:,1,:,:]

        input_1 = input_1.view(num, -1)
        input_2 = input_2.view(num, -1)

        target_1 = target_1.view(num, -1)
        target_2 = target_2.view(num, -1)

        intersection_1 = (input_1 * target_1)
        intersection_2 = (input_2 * target_2)

        dice_1 = (2. * intersection_1.sum(1) + smooth) / (input_1.sum(1) + target_1.sum(1) + smooth)
        dice_2 = (2. * intersection_2.sum(1) + smooth) / (input_2.sum(1) + target_2.sum(1) + smooth)

        dice_1 = 1 - dice_1.sum() / num
        dice_2 = 1 - dice_2.sum() / num

        dice = (dice_1+dice_2)/2.0
        if (np.isnan(bce.detach().cpu())):
            raise Exception("!!!Nan!!!")
        return 0.5 * bce + dice
