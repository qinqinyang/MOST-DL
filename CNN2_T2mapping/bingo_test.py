# -*- coding: UTF-8 -*-
'''
Created on Wed Oct 9 20:15:00 2019
Modified on Thur Aug 19 14:30:00 2021

@author: Qinqin Yang
'''
import os
import argparse
import scipy.io as matio

import torch
import numpy as np
from network.UNet import Inference

from torch.utils import data
from torchvision import transforms as T

def load_data_Charles_test(image_path, config):
    print(image_path)
    data_in = np.fromfile(image_path, dtype=np.float32)
    data_pairs = data_in.reshape(config.INPUT_H, config.INPUT_W, config.DATA_C)
    input_sets = data_pairs[:, :, :config.INPUT_C]
    label_sets = data_pairs[:, :, :config.INPUT_C]

    return input_sets,label_sets

def load_data_mat(image_path):
    print(image_path)
    data_in = matio.loadmat(image_path)
    input_sets = data_in['input']
    label_sets = data_in['t2']

    return input_sets, label_sets

class ImageFolder(data.Dataset):
    """Load Variaty Chinese Fonts for Iterator. """

    def __init__(self, root, config, mode='train'):
        """Initializes image paths and preprocessing module."""
        self.config = config
        self.root = root
        self.mode = mode
        self.image_dir = os.path.join(root, mode)

        self.image_paths = list(map(lambda x: os.path.join(self.image_dir, x), os.listdir(self.image_dir)))
        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))
        self.image_paths.sort(reverse=True)

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        #image, GT = load_data_Charles_test(image_path, self.config)
        image, GT = load_data_mat(image_path)

        # -----To Tensor------#
        Transform = T.ToTensor()
        image = Transform(image)
        GT = Transform(GT)

        return image, GT

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)

def get_loader(image_path, config, crop_key, num_workers, shuffle=True,mode='train'):
    """Builds and returns Dataloader."""

    dataset = ImageFolder(root=image_path, config=config, mode=mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.BATCH_SIZE,
                                  shuffle=shuffle,
                                  num_workers=num_workers)
    return data_loader

def test(config):
    #-----GPU-----#
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU_NUM

    #-----random seed-----#
    np.random.seed(1)
    torch.manual_seed(1)

    # -----path-----#
    model_dir = os.path.join(config.model_path, config.name+'/'+ config.name+ '_epoch_' +config.model_num + '.pth')
    if not os.path.exists(model_dir):
        print('Model not found, please check you path to model')
        print(model_dir)
        os._exit(0)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    #-----dataloader-----#
    test_batch = get_loader(config.data_dir, config, crop_key=False, num_workers=1, shuffle=False, mode=config.test_dir)

    #-----model-----#
    net = Inference(config.INPUT_C,config.OUTPUT_C)

    if torch.cuda.is_available():
        net.cuda()

    #-----modelloader-----#
    net.load_state_dict(torch.load(model_dir))
    print('Model parameters loaded!')

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ********************************************test*****************************************************#
    net.eval()
    for i,(images, GT) in enumerate(test_batch):
        images, GT = images.type(torch.FloatTensor), GT.type(torch.FloatTensor)
        images, GT = images.to(device), GT.to(device)

        SR = net(images)  # forward

        if i == 0:
            OUT_test = SR.permute(0, 2, 3, 1).cpu().detach().numpy()
        else:
            OUT_test = np.concatenate((SR.permute(0, 2, 3, 1).cpu().detach().numpy(),OUT_test),axis=0)

    #-----save mat-----#
    print('.' * 30)
    print('OUT_test:', OUT_test.shape)
    print('.' * 30)
    matio.savemat(
        os.path.join(config.result_path, config.name + '_result_' + config.test_dir + '.mat'),
        {
            'output': OUT_test
        })
    print('Save result in ',config.name + '_result_' + config.test_dir + '.mat')
    print('.' * 30)
    print('Finished!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # experiment name
    parser.add_argument('--name', type=str, default='experiment')
    parser.add_argument('--data_dir', type=str, default='./dataset/')
    parser.add_argument('--GPU_NUM', type=str, default='0')

    # model hyper-parameters
    parser.add_argument('--INPUT_H', type=int, default=256)
    parser.add_argument('--INPUT_W', type=int, default=256)
    parser.add_argument('--INPUT_C', type=int, default=2)
    parser.add_argument('--OUTPUT_C', type=int, default=1)
    parser.add_argument('--LABEL_C', type=int, default=3)
    parser.add_argument('--DATA_C', type=int, default=7)

    # test hyper-parameters
    parser.add_argument('--BATCH_SIZE', type=int, default=1)

    parser.add_argument('--model_path', type=str, default='./models/')
    parser.add_argument('--model_num', type=str, default='')
    parser.add_argument('--result_path', type=str, default='./test_result/')
    parser.add_argument('--test_dir', type=str, default='')

    config = parser.parse_args()

    config.model_num = '600'
    config.test_dir = 'brain_subject_01'

    config.name = 'MOLED_T2map_MoCo'
    #config.name = 'MOLED_T2map_noMoCo'

    test(config)