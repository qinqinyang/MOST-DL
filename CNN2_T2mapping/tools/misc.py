import os
import math
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt

irange = range

def save_torch_result_multi(tensor, save_dir,
                      nrow=1, padding=0, pad_value=0,
                      format='jpg', cmap='gray', norm=False, crange=[0, 1]):
    """Save a given Tensor into an image file.
    """
    nmaps = tensor.size(0)
    tensor = torch.cat((tensor[:, 0:1, :, :], tensor[:, 1:2, :, :], tensor[:, 2:3, :, :], tensor[:, 3:4, :, :]), dim=3)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(tensor.size(1), height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding) \
                .narrow(2, x * width + padding, width - padding) \
                .copy_(tensor[k])
            k = k + 1
    merge_img = np.squeeze(grid.permute(1, 2, 0).cpu().detach().numpy())
    if norm:
        plt.imsave(save_dir + '.' + format, merge_img, cmap=cmap)
    else:
        plt.imsave(save_dir + '.' + format, merge_img, cmap=cmap, vmin=crange[0], vmax=crange[1])

def save_torch_result(tensor, save_dir,
                      nrow=4, padding=0, pad_value=0,
                      format='jpg', cmap='gray', norm=False, crange=[0, 1]):
    """Save a given Tensor into an image file.
    """
    nmaps = tensor.size(0)

    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(tensor.size(1), height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding) \
                .narrow(2, x * width + padding, width - padding) \
                .copy_(tensor[k])
            k = k + 1
    merge_img = np.squeeze(grid.permute(1, 2, 0).cpu().detach().numpy())
    if norm:
        plt.imsave(save_dir + '.' + format, merge_img, cmap=cmap)
    else:
        plt.imsave(save_dir + '.' + format, merge_img, cmap=cmap, vmin=crange[0], vmax=crange[1])


def save_torch_result_with_label(tensor, label, save_dir, loss=False,
                                 nrow=4, padding=0, pad_value=0,
                                 format='jpg', cmap='gray', norm=False, crange=[0, 1]):
    nmaps = tensor.size(0)
    if loss:
        tensor = torch.cat((tensor, label, torch.abs(tensor-label)), dim=3)
    else:
        tensor = torch.cat((tensor, label), dim=3)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(tensor.size(1), height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding) \
                .narrow(2, x * width + padding, width - padding) \
                .copy_(tensor[k])
            k = k + 1
    merge_img = np.squeeze(grid.permute(1, 2, 0).cpu().detach().numpy())

    if norm:
        plt.imsave(save_dir + '.' + format, merge_img, cmap=cmap)
    else:
        plt.imsave(save_dir + '.' + format, merge_img, cmap=cmap, vmin=crange[0], vmax=crange[1])

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

def mkexperiment(config,cover=False):

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    experiment_path = os.path.join(config.result_path,config.name)
    if os.path.exists(experiment_path):
        if cover:
            shutil.rmtree(experiment_path)
            os.makedirs(experiment_path)
            os.makedirs(os.path.join(experiment_path, 'tensorboard'))
            os.makedirs(os.path.join(experiment_path, 'inter_result'))
        else:
            raise ValueError("Experiment '{}' already exists. Please modify the experiment name!"
                             .format(config.name))
    else:
        os.makedirs(experiment_path)
        os.makedirs(os.path.join(experiment_path, 'tensorboard'))
        os.makedirs(os.path.join(experiment_path, 'inter_result'))
    return experiment_path