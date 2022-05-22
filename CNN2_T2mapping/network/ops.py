import torch
import torch.nn as nn
from torch.nn import init

def net_analysis(net):
    tol_conv = 0
    print('.'*70)
    for layer in net.named_modules():
        if isinstance(layer[1], nn.Conv2d):
            print(layer[1])
            tol_conv += 1
    print('.' * 70)
    print('# Model contains %d Conv layers.'%(tol_conv))
    print('# Model parameters:', sum(param.numel() for param in net.parameters()))
    print('.' * 70)

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class TotalVariation(nn.Module):
    def __init__(self):
        super(TotalVariation, self).__init__()

    def _dx(self,tensor):
        return tensor[..., 1:] - tensor[..., :-1]

    def _dy(self,tensor):
        return tensor[..., 1:, :] - tensor[..., :-1, :]

    def forward(self,tensor):
        return torch.add(torch.abs(self._dx(tensor)).mean(), torch.abs(self._dy(tensor)).mean())
