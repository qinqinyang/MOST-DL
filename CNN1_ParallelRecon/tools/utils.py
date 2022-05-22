# -*- coding: UTF-8 -*-
import os
import torch
import numpy as np
from PIL import Image
from os.path import join
import matplotlib.pyplot as plt
irange = range

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".tif", ".bmp"])

def listdir_abspath(dir):
    '''
        return abs path of all files from a direction

        :return [file1, file2, file3, ...]

    '''
    dirlist = []
    for (dirpath, dirnames, filenames) in os.walk(dir):
        for filename in filenames:
            dirlist += [os.path.join(dirpath, filename)]
    return dirlist

def listdir_abspath_mulitch(dir):
    '''
        return abs path of all files from a multi-channels direction

       :return [[file1_ch1, file1_ch2, ...],[file2_ch1, file2_ch2, ...], ...]

    '''
    filename_list = os.listdir(join(dir, '1'))
    channels_num =len(os.listdir(dir))
    temp_list = []
    data_list = []
    for filename in filename_list:
        for channel in range(1, channels_num+1):
            temp_list.append(join(join(dir, str(channel)), filename))
        data_list.append(temp_list)
        temp_list = []
    return data_list

def lossmap(img1, img2):
    return torch.abs(img1.cpu()-img2.cpu())

def save_tensor_to_mat(result, label, target, filename, fmat='.jpg'):
    imgg = np.concatenate((result, label), axis=1)
    im = Image.fromarray(imgg*256)
    im = im.convert('L')
    im.save(join(target, filename + fmat))

def save_single_image_result(tensor, save_fn):
    w = tensor.size(2)
    h = tensor.size(3)
    img = np.zeros((h, w))
    tensor = tensor.permute(0, 2, 3, 1).cpu().detach().numpy()
    img[:,:] = tensor[0,:,:,0]
    plt.imsave(save_fn, img, cmap='gray')
