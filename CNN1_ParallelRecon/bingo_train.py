# -*- coding: UTF-8 -*-
'''
Created on Thur Aug 19 14:30:00 2021

@author: Qinqin
'''
import os
import csv
import argparse
import numpy as np
import time

import torch
from tensorboardX import SummaryWriter
from network.ResUNet import Inference,loss_fun_ifft
from tools.modelsummary import summary

from tools.data_loader import get_loader,get_loader_test
from tools.misc import mkexperiment,save_torch_result
from tools.transforms import ssos

def main(config):
    #-----GPU-----#
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU_NUM
    torch.backends.cudnn.benchmark = True

    #-----random seed-----#
    np.random.seed(1)
    torch.manual_seed(1)
    time2 = 0

    # -----experiment-----#
    experiment_path = mkexperiment(config, cover=True)
    save_inter_result = os.path.join(experiment_path, 'inter_result')
    model_path = os.path.join(config.model_path,config.name)

    #-----dataloader-----#
    data_dir = config.data_dir
    train_batch = get_loader(data_dir, config,num_workers=1, shuffle=True, mode='train')
    val_batch = get_loader(data_dir, config, num_workers=1, shuffle=True, mode='test')
    brain_batch = get_loader_test(data_dir, config,num_workers=1, shuffle=False, mode='brain')

    #-----model-----#
    net = Inference(config)
    summary(net, (config.INPUT_C, config.INPUT_H, config.INPUT_W))

    # -----lossfunc-----#
    criterion = loss_fun_ifft()

    if torch.cuda.is_available():
        net.cuda()
        criterion.cuda()

    # -----optim-----#
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr,betas=(config.beta1, config.beta2))

    #-----Setup device-----#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----Tensorboard-----#
    writer_train = SummaryWriter(log_dir = os.path.join(os.path.join(experiment_path, 'tensorboard'),'train'))
    writer_val = SummaryWriter(log_dir= os.path.join(os.path.join(experiment_path, 'tensorboard'),'test'))

    # ----csv----- #
    f = open(os.path.join(experiment_path, 'result.csv'), 'a', encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow(['train loss', 'val loss', 'lr', 'total_iters', 'epochs'])

    if config.mode =='train':
        total_iters = 0
        train_loss = 0
        train_length = 0
        val_loss = 0
        val_length = 0

        for epoch in range(1,config.num_epochs+1):
            # ********************************************train*****************************************************#
            for i,(images, GT, mask) in enumerate(train_batch):
                images, GT, mask = images.type(torch.FloatTensor), GT.type(torch.FloatTensor), mask.type(torch.FloatTensor)
                images, GT, mask = images.to(device), GT.to(device), mask.to(device)

                SR = net(images,mask)  # forward

                loss = criterion(SR, GT)
                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_length += images.size(0)
                total_iters += 1

                # learing rate decay
                if (total_iters % config.lr_updata) == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.8

                if total_iters%config.step == 0:
                    lr = optimizer.param_groups[0]['lr']
                    # # *************************************validation**********************************************#
                    with torch.no_grad():
                        net.eval()
                        for i,(images_val, GT_val, mask_val) in enumerate(val_batch):
                            images_val, GT_val, mask_val = images_val.type(torch.FloatTensor), GT_val.type(torch.FloatTensor), mask_val.type(torch.FloatTensor)
                            images_val, GT_val, mask_val = images_val.to(device), GT_val.to(device), mask_val.to(device)

                            SR_val = net(images_val,mask_val)
                            loss_val = criterion(SR_val, GT_val)

                            val_loss += loss_val.item()
                            val_length += images_val.size(0)

                        # Print the log info
                        time1 = time.clock()
                        temp_time = time1-time2
                        # Print the log info
                        print(
                            'Epoch [%d/%d], Total_iters [%d], Train Loss: %.5f, Val Loss: %.5f, lr: %.8f,time: %.3f' % (
                                epoch, config.num_epochs, total_iters,
                                train_loss / train_length, val_loss / val_length, lr, temp_time))
                        time2 = time.clock()
                        writer_train.add_scalar('data/loss', train_loss / train_length, total_iters)
                        wr.writerow([train_loss / train_length, val_loss / val_length, lr, total_iters, epoch])

                        train_loss = 0
                        train_length = 0
                        val_loss = 0
                        val_length = 0

                        ## ********************************************test***********************************************#
                        for i, (brain_images, mask_images) in enumerate(brain_batch):
                            brain_images = brain_images.type(torch.FloatTensor)
                            mask_images = mask_images.type(torch.FloatTensor)

                            brain_images = brain_images.to(device)
                            mask_images = mask_images.to(device)

                            SR_brain = net(brain_images, mask_images)
                            SR_result = ssos(SR_brain,k=False)

                            # save result in fold
                            save_dir = os.path.join(save_inter_result,'inter_odd_'+str(total_iters)+'_brain')
                            save_torch_result(SR_result, save_dir,
                                              format='png', cmap='gray', norm=False, crange=[0, 0.5])
                        net.train()

            # -----save_model-----#
            if (epoch) % config.model_save_step == 0 and epoch > config.model_save_start:
                if not os.path.exists(model_path):
                    os.mkdir(model_path)
                torch.save(net.state_dict(), model_path + '/' + config.name + '_epoch_' +str(epoch) + '.pth')

        f.close()
        writer_train.close()
        writer_val.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # experiment name
    parser.add_argument('--name', type=str, default='experiment')
    parser.add_argument('--experiment_path', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='./dataset/')
    parser.add_argument('--GPU_NUM', type=str, default='0')

    # model hyper-parameters
    parser.add_argument('--INPUT_H', type=int, default=128)
    parser.add_argument('--INPUT_W', type=int, default=128)
    parser.add_argument('--INPUT_C', type=int, default=32)
    parser.add_argument('--OUTPUT_C', type=int, default=32)

    # training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--BATCH_SIZE', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_updata', type=int, default=8000)  # epoch num for lr updata
    parser.add_argument('--beta1', type=float, default=0.9)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
    parser.add_argument('--DClambda', type=float, default=1)

    parser.add_argument('--step', type=int, default=50)
    parser.add_argument('--model_save_start', type=int, default=1)
    parser.add_argument('--model_save_step', type=int, default=50)

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_path', type=str, default='./models/')
    parser.add_argument('--result_path', type=str, default='./result/')

    config = parser.parse_args()

    config.name = 'MOLED_parallelRec'
    main(config)