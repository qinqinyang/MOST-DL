# -*- coding: UTF-8 -*-
'''
Created on Wed Oct 9 20:15:00 2019

@author: Qinqin
'''
import argparse
import time
import numpy as np
import torch
from tools.data_loader import get_loader

def dataload(config):
    #-----random seed-----#
    np.random.seed(1)
    torch.manual_seed(1)

    #-----dataloader-----#
    data_dir = config.data_dir
    data_batch = get_loader(data_dir, config, crop_key=True,num_workers=config.NUM_WORKERS, shuffle=True, mode='train')

    #-----Setup device-----#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(1,config.num_epochs+1):
        for i,(images, GT) in enumerate(data_batch):
            t0 = time.clock()
            images, GT = images.type(torch.FloatTensor), GT.type(torch.FloatTensor)
            images, GT = images.to(device), GT.to(device)
            t1 = time.clock()
            print('%.9f secodes read time'%(t1-t0))
            time.sleep(0.5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # experiment name
    parser.add_argument('--data_dir', type=str, default='./dataset/')

    # model hyper-parameters
    parser.add_argument('--INPUT_H', type=int, default=256)
    parser.add_argument('--INPUT_W', type=int, default=256)
    parser.add_argument('--INPUT_C', type=int, default=2)
    parser.add_argument('--OUTPUT_C', type=int, default=1)
    parser.add_argument('--LABEL_C', type=int, default=3)
    parser.add_argument('--DATA_C', type=int, default=5)

    parser.add_argument('--CROP_SIZE', type=int, default=96)

    # training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=700)
    parser.add_argument('--BATCH_SIZE', type=int, default=8)
    parser.add_argument('--NUM_WORKERS', type=int, default=3)

    config = parser.parse_args()

    dataload(config)