# -*- coding: utf-8 -*-

import datetime
import os
import random
import time
from collections import OrderedDict
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.dataset import Dataset
from net import Unet
from utilities.loss import MulticlassDiceLoss
from utilities.metrics import dice_coef, iou_score
from utilities.utils import count_params


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    ious = AverageMeter()
    dices_1s = AverageMeter()
    dices_2s = AverageMeter()
    model.train()

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.cuda()
        target = target.cuda()

        if len(np.unique(target.cpu())) == 1:
            continue
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iou = iou_score(output, target)
        dice_1 = dice_coef(output, target)[0]
        dice_2 = dice_coef(output, target)[1]

        writer.add_scalar('Epoch{}/ Loss'.format(epoch), loss.item(), i)
        writer.add_scalar('Epoch{}/IoU'.format(epoch), iou, i)
        writer.add_scalar('Epoch{}/Dice Liver'.format(epoch), dice_1, i)
        writer.add_scalar('Epoch{}/Dice Tumor'.format(epoch), dice_2, i)
    
        losses.update(loss.item())
        ious.update(iou)
        dices_1s.update(dice_1)
        dices_2s.update(dice_2)


    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
        ('dice_1', dices_1s.avg),
        ('dice_2', dices_2s.avg)
    ])

    return log


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    ious = AverageMeter()
    dices_1s = AverageMeter()
    dices_2s = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()


            if len(np.unique(target.cpu())) == 1:
                continue
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)
            dice_1 = dice_coef(output, target)[0]
            dice_2 = dice_coef(output, target)[1]

            print("Loss: {:.4f}, IoU: {:.4f}, Dice_Liver: {:.4f}, Dice_tumor: {:.4f}".format(loss.item(), iou, dice_1, dice_2))
            losses.update(loss.item())
            ious.update(iou)
            dices_1s.update(dice_1)
            dices_2s.update(dice_2)

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
        ('dice_1', dices_1s.avg),
        ('dice_2', dices_2s.avg)
    ])

    return log


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式 
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

timestamp  = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
writer = SummaryWriter('./experiments/{}'.format(timestamp))

def main():

    if not os.path.exists('models/{}/{}'.format('Unet',timestamp)):
        os.makedirs('models/{}/{}'.format('Unet',timestamp))

    criterion = MulticlassDiceLoss().cuda()

    cudnn.benchmark = True

    # Data loading code
    img_paths = glob('./data/trainImage_k1_1217/*')
    mask_paths = glob('./data/trainMask_k1_1217/*')

    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
        train_test_split(img_paths, mask_paths, test_size=0.23, random_state=39)
    
    print("train_num:%s"%str(len(train_img_paths)))
    print("val_num:%s"%str(len(val_img_paths)))

    # create model
    #换模型需要修改的地方
    model = Unet.U_Net()
    # 参数初始化 https://zhuanlan.zhihu.com/p/188701989
    model.apply(weight_init)

    model = torch.nn.DataParallel(model).cuda()

    print('Model Parameters: ', count_params(model))

    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

    train_dataset = Dataset(train_img_paths, train_mask_paths)
    val_dataset = Dataset(val_img_paths, val_mask_paths)

    batch_size = 20
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'iou','dice_1', 'dice_2', 'val_loss', 'val_iou','val_dice_1', 'val_dice_2'
    ])

    best_loss = 100
    trigger = 0
    epochs = 200
    first_time = time.time()
    for epoch in range(epochs):
        print('Epoch [%d/%d]' %(epoch+1, epochs))
        # train for one epoch
        
        train_log = train(train_loader, model, criterion, optimizer, epoch)
        scheduler.step()
        
        # evaluate on validation set
        val_log = validate(val_loader, model, criterion)

        print('loss %.4f - iou %.4f - dice_1 %.4f - dice_2 %.4f - val_loss %.4f - val_iou %.4f - val_dice_1 %.4f - val_dice_2 %.4f'
                  %(train_log['loss'], train_log['iou'], train_log['dice_1'], train_log['dice_2'], val_log['loss'], val_log['iou'], val_log['dice_1'], val_log['dice_2']))

        end_time = time.time()
        print("time:", (end_time - first_time) / 60)

        tmp = pd.Series([
            epoch,
            1e-4,
            train_log['loss'],
            train_log['iou'],
            train_log['dice_1'],
            train_log['dice_2'],
            val_log['loss'],
            val_log['iou'],
            val_log['dice_1'],
            val_log['dice_2'],
        ], index=['epoch', 'lr', 'loss', 'iou', 'dice_1' ,'dice_2' ,'val_loss', 'val_iou', 'val_dice_1' ,'val_dice_2'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/{}/{}/log.csv'.format('Unet',timestamp), index=False)

        trigger += 1

        val_loss = val_log['loss']
        if val_loss < best_loss:
            torch.save(model.state_dict(), 'models/{}/{}/epoch{}-{:.4f}-{:.4f}_model.pth'.format('Unet',timestamp,epoch,val_log['dice_1'],val_log['dice_2']))
            best_loss = val_loss
            print("=> saved best model")
            trigger = 0

        torch.cuda.empty_cache()

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    main()
