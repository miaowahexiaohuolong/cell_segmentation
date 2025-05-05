from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
import os
from tqdm import tqdm
from time import time

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

from cenet import CE_Net_
from framework import MyFrame
from loss import dice_bce_loss
from dataset_handle import ImageFolder
from Visualizer import Visualizer

import Constants
import image_utils

# Please specify the ID of graphics cards that you want to use
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def CE_Net_Train():
    # NAME = 'CE-Net' + Constants.ROOT.split('/')[-1]
    NAME = 'cell_segmentation'
    # run the Visdom
    # viz = Visualizer(env=NAME)
    solver = MyFrame(CE_Net_, dice_bce_loss, 1e-6)# 学习率 loss 网络名称
    batchsize = torch.cuda.device_count() * Constants.BATCHSIZE_PER_CARD
    scheduler=torch.optim.lr_scheduler.ExponentialLR(solver.optimizer, 0.9)
    
    #运行一次后才可以使用权重，不然的话是用不了
    #solver.load('E:/Github/ce_unet/weights/200.th')
    # For different 2D medical image segmentation tasks, please specify the dataset which you use
    # for examples: you could specify "dataset = 'DRIVE' " for retinal vessel detection.

    dataset = ImageFolder(root_path='', mode='train')
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=12)

    # start the logging files
    mylog = open('logs/' + NAME + '.log', 'w')
    # tic = time()

    no_optim = 0
    total_epoch = Constants.TOTAL_EPOCH
    train_epoch_best_loss = Constants.INITAL_EPOCH_LOSS
    for epoch in range(1, total_epoch + 1):
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        index = 0
        # scheduler.step()
        for img, mask in tqdm(data_loader_iter):
            solver.set_input(img, mask)
            train_loss, pred = solver.optimize()
            train_epoch_loss += train_loss
            index = index + 1

        train_epoch_loss = train_epoch_loss/len(data_loader_iter)
        mylog.write('epoch: '+ str(epoch) + ' ' + ' train_loss: ' + str(train_epoch_loss.cpu().numpy()) + '\n')
        print('epoch:', epoch, 'train_loss:', train_epoch_loss.cpu().numpy(), 'lr: ' + format(scheduler.get_lr()[0]))
        # solver.save('./weights/' + NAME + '.th')
        solver.save('./weights/'+str(epoch)+'.th')
        mylog.flush()

    # print(mylog, 'Finish!')
    print('Finish!')
    mylog.close()


if __name__ == '__main__':
    print(torch.__version__)
    CE_Net_Train()



