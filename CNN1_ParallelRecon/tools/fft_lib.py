# -*- coding: UTF-8 -*-
import torch

def tensor_to_complex(x):
    real = x[:, 0::2, :, :]
    imag = x[:, 1::2, :, :]
    complex = torch.stack((real, imag), 4)
    return complex

def complex_to_tensor(complex):
    real = complex[:, :, :, :, 0]
    imag = complex[:, :, :, :, 1]
    tensor_size = real.shape
    tensor = torch.zeros((tensor_size[0], tensor_size[1] * 2, tensor_size[2], tensor_size[3]))
    tensor[:, 0::2, :, :] = real
    tensor[:, 1::2, :, :] = imag
    return tensor.cuda()

def IFFT2C(input_image):
    inp = tensor_to_complex(input_image)  # [batch, 32, 128, 128]

    inp = ifftshift(inp, dim=(-3, -2))
    out = torch.ifft(inp, 2)
    out = fftshift(out, dim=(-3, -2))

    out = complex_to_tensor(out)
    return out

def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)

def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)

def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)

def ifft2(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-2, -1))
    data = torch.ifft(data, 2, normalized=True)
    data = fftshift(data, dim=(-2, -1))
    return data