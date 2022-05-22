# -*- coding: UTF-8 -*-
'''
Created on Thur Aug 19 14:30:00 2021

@author: Qinqin Yang
'''
import torch
import torch.nn as nn
from tools.fft_lib import ifftshift,fftshift

class IFFT_layer(nn.Module):
    """ Create data consistency operator
    Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of the input.
    This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
    and applies FFT2 to the (nx, ny) axis.
    """

    def __init__(self):
        super(IFFT_layer, self).__init__()

    def tensor_to_complex(self,x):
        real = x[:, 0::2, :, :]
        imag = x[:, 1::2, :, :]
        complex = torch.stack((real, imag), 4)
        return complex

    def forward(self, *input, **kwargs):
        return self.IFFT2C(*input)

    def IFFT2C(self,input_image):
        inp = self.tensor_to_complex(input_image)  # [batch, 32, 128, 128]

        inp = ifftshift(inp, dim=(-3, -2))
        out = torch.ifft(inp, 2, normalized=True)
        out = fftshift(out, dim=(-3, -2))

        real = out[:, :, :, :, 0]
        imag = out[:, :, :, :, 1]
        tensor = torch.zeros_like(input_image)
        tensor[:, 0::2, :, :] = real
        tensor[:, 1::2, :, :] = imag
        return tensor

class DClayer(nn.Module):
    """ Create data consistency operator
    Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of the input.
    This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
    and applies FFT2 to the (nx, ny) axis.
    """

    def __init__(self, DClambda):
        super(DClayer, self).__init__()
        self.DClambda = DClambda

    def tensor_to_complex(self,x):
        real = x[:, 0::2, :, :]
        imag = x[:, 1::2, :, :]
        complex = torch.stack((real, imag), 4)
        return complex

    def forward(self, inp, oup, mask):
        return self.DataConsistency(inp, oup, mask)

    def DataConsistency(self,input_image, output_image, mask):
        inp = self.tensor_to_complex(input_image)  # [batch, 32, 128, 128]
        oup = self.tensor_to_complex(output_image)  # [batch, 32, 128, 128]

        mask = torch.unsqueeze(mask[:,0:16,:,:],dim=4)
        mask = torch.cat((mask,mask),dim=4)
        #mask = mask.byte()

        inp_k = torch.fft(inp, 2)
        oup_k = torch.fft(oup, 2)

        dc_out_k = torch.where(mask > 0, (inp_k+self.DClambda*oup_k)/(1+self.DClambda), oup_k)
        dc_out = torch.ifft(dc_out_k, 2)  # [2, 16, 128, 128, 2]

        real = dc_out[:, :, :, :, 0]
        imag = dc_out[:, :, :, :, 1]
        tensor = torch.zeros_like(input_image)
        tensor[:, 0::2, :, :] = real
        tensor[:, 1::2, :, :] = imag

        return tensor

class ResBlockAllConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.LeakyReLU(inplace=True),
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.conv(x)
        x = self.Conv_1x1(x)
        x = x + x1
        return x

class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, padding=1):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=1, padding=padding, bias=True),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Inference(nn.Module):
    def __init__(self, config):
        super().__init__()
        img_ch = config.INPUT_C
        output_ch = config.OUTPUT_C
        DClambda = config.DClambda

        self.DC = DClayer(DClambda)
        self.IFFT = IFFT_layer()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.RRCNN1 = ResBlockAllConv(ch_in=img_ch, ch_out=128)
        self.RRCNN2 = ResBlockAllConv(ch_in=128, ch_out=128)
        self.RRCNN3 = ResBlockAllConv(ch_in=128, ch_out=256)
        self.RRCNN4 = ResBlockAllConv(ch_in=256, ch_out=512)
        self.RRCNN5 = ResBlockAllConv(ch_in=512, ch_out=1024)

        self.Up5 = UpConv(ch_in=1024, ch_out=512)
        self.Up_RRCNN5 = ResBlockAllConv(ch_in=1024, ch_out=512)

        self.Up4 = UpConv(ch_in=512, ch_out=256)
        self.Up_RRCNN4 = ResBlockAllConv(ch_in=512, ch_out=256)

        self.Up3 = UpConv(ch_in=256, ch_out=128)
        self.Up_RRCNN3 = ResBlockAllConv(ch_in=256, ch_out=128)

        self.Up2 = UpConv(ch_in=128, ch_out=128)
        self.Up_RRCNN2 = ResBlockAllConv(ch_in=256, ch_out=128)

        self.Conv = OutConv(128, output_ch)

    def forward(self, x, mask):
        # encoding path
        x = self.IFFT(x)

        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)
        out = self.DC(x, out, mask)

        return out

class loss_fun_ifft(nn.Module):
    def __init__(self):
        super().__init__()
        self.ifft2c = IFFT_layer()
        self.loss = torch.nn.L1Loss()

    def forward(self, x, y):
        y = self.ifft2c(y)
        loss = self.loss(x,y)
        return loss