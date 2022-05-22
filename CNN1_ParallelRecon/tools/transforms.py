"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.

    Args:
        data (np.array): Input numpy array

    Returns:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    data = data.astype(np.float32)
    return torch.from_numpy(data)


def apply_mask(data, mask_func, seed=None):
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data (torch.Tensor): The input k-space data. This should have at least 3 dimensions, where
            dimensions -3 and -2 are the spatial dimensions, and the final dimension has size
            2 (for complex values).
        mask_func (callable): A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed (int or 1-d array_like, optional): Seed for the random number generator.

    Returns:
        (tuple): tuple containing:
            masked data (torch.Tensor): Subsampled k-space data
            mask (torch.Tensor): The generated mask
    """
    shape = np.array(data.shape)
    shape[:-3] = 1
    mask = mask_func(shape, seed)
    return torch.where(mask == 0, torch.Tensor([0]), data), mask


def phase_adjust(data, RO_L, PE_L, RO_A, PE_A, RO_chirp = 2.0, PE_chirp = 1.0):
    """
    Apply Spen sequence to adjust phase in sampling space (two dimensions chirp)

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
        RO_L, PE_L (float): FOV of Read out and Phase encoding dimensions about spen sequence
        RO_A, PE_A (float): deconvolution coe about spen sequence (reference deconvolution paper from ccb)
        RO_chirp, PE_chirp (float): Specify param about spen sequence (PE: 90 chirp RO: 180 equal PE_chirp=1.0, RO_chirp=2.0)

    Returns:
        torch.Tensor: The phase adjust of the input
    """
    assert data.size(-1) == 2
    N, M, _ = data.size()
    cz_x = torch.linspace(-RO_L/2.0, RO_L/2.0, steps=M)
    cz_y = torch.linspace(-PE_L/2.0, PE_L/2.0, steps=N)
    CZ_Y, CZ_X = torch.meshgrid(cz_x, cz_y)

    PE_phase = -0.5*PE_A*PE_chirp*CZ_Y*CZ_Y
    RO_phase = 0.5*RO_A*RO_chirp*CZ_X*CZ_X

    PE_adjust_r = torch.cos(PE_phase).unsqueeze(-1)
    PE_adjust_i = torch.sin(PE_phase).unsqueeze(-1)

    RO_adjust_r = torch.cos(RO_phase).unsqueeze(-1)
    RO_adjust_i = torch.sin(RO_phase).unsqueeze(-1)


    PE_adjust   = torch.cat((PE_adjust_r, PE_adjust_i), dim=-1)
    RO_adjust   = torch.cat((RO_adjust_r, RO_adjust_i), dim=-1)

    data        = complex_dotmul(data, PE_adjust)
    data        = complex_dotmul(data, RO_adjust)

    return data

def Pmatrix2d(RO_Num, PE_Num, RO_L, PE_L, RO_A, PE_A, RO_chirp = 2.0, PE_chirp = 1.0):
    """
    Apply Encode matrix for seperate channel

    Args:
        RO_Num, PE_Num (float): matrix size of Read out and Phase encoding dimensions about spen sequence
        RO_L, PE_L (float): FOV of Read out and Phase encoding dimensions about spen sequence
        RO_A, PE_A (float): deconvolution coe about spen sequence (reference deconvolution paper from ccb)
        RO_chirp, PE_chirp (float): Specify param about spen sequence (PE: 90 chirp RO: 180 equal PE_chirp=1.0, RO_chirp=2.0)

    Returns:
        PE_P_matrix (torch.Tensor):
            (..., PE_Num, PE_Num, 2)
        RO_P_matrix (torch.Tensor):
            (..., RO_Num, RO_Num, 2)

    """
    recon_x = torch.linspace(-RO_L/2.0, RO_L/2.0, steps=RO_Num)
    recon_y = torch.linspace(-PE_L/2.0, PE_L/2.0, steps=PE_Num)

    RX_1, RX_2 = torch.meshgrid(recon_x, recon_x)
    RY_1, RY_2 = torch.meshgrid(recon_y, recon_y)

    RO_theta = - 0.5 * RO_A * RO_chirp * (RX_1-RX_2)**2
    PE_theta = - 0.5 * PE_A * PE_chirp * (RY_1-RY_2)**2

    RO_P_matrix_r = torch.cos(RO_theta)
    RO_P_matrix_i = torch.sin(RO_theta)

    PE_P_matrix_r = torch.cos(PE_theta)
    PE_P_matrix_i = torch.sin(PE_theta)

    RO_P_matrix   = torch.cat((RO_P_matrix_r.unsqueeze(-1), RO_P_matrix_i.unsqueeze(-1)), dim=-1)
    PE_P_matrix   = torch.cat((PE_P_matrix_r.unsqueeze(-1), PE_P_matrix_i.unsqueeze(-1)), dim=-1)


    RO_P_matrix_np = RO_P_matrix_r.numpy() +1j* RO_P_matrix_i.numpy()
    PE_P_matrix_np = PE_P_matrix_r.numpy() +1j* PE_P_matrix_i.numpy()

    inv_RO_P_matrix_np = np.linalg.pinv(RO_P_matrix_np)
    inv_PE_P_matrix_np = np.linalg.pinv(PE_P_matrix_np)


    inv_RO_P_matrix_r    = torch.from_numpy(inv_RO_P_matrix_np.real)
    inv_RO_P_matrix_i    = torch.from_numpy(inv_RO_P_matrix_np.imag)

    inv_PE_P_matrix_r    = torch.from_numpy(inv_PE_P_matrix_np.real)
    inv_PE_P_matrix_i    = torch.from_numpy(inv_PE_P_matrix_np.imag)

    inv_RO_P_matrix   = torch.cat((inv_RO_P_matrix_r.unsqueeze(-1), inv_RO_P_matrix_i.unsqueeze(-1)), dim=-1)
    inv_PE_P_matrix   = torch.cat((inv_PE_P_matrix_r.unsqueeze(-1), inv_PE_P_matrix_i.unsqueeze(-1)), dim=-1)
    return PE_P_matrix, RO_P_matrix, inv_PE_P_matrix, inv_RO_P_matrix

def complex_dotmul(A, B):
    """
    Apply complex dot mul for seperate channel

    Args:
        A (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
            (..., n_input, m_input, 2)
        B (torch.Tensor):
            (..., n_input, m_input, 2)

    Returns:
        C (torch.Tensor): 
            (..., n_input, m_input, 2)
    """
    assert A.size(-1)==2
    assert B.size(-1)==2
    A_r, A_i = torch.split(A, 1, dim=-1)
    B_r, B_i = torch.split(B, 1, dim=-1)

    C_r      = A_r*B_r-A_i*B_i
    C_i      = A_r*B_i+A_i*B_r

    C        = torch.cat((C_r, C_i), dim=-1)
    return C

def complex_matmul(A, B):
    """
    Apply complex mat mul for seperate channel

    Args:
        A (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
            (..., n_target, n_input, 2)
        B (torch.Tensor):
            (..., n_input, m_input, 2)

    Returns:
        C (torch.Tensor): 
            (..., n_target, m_input, 2)
    """
    assert A.size(-1)==2
    assert B.size(-1)==2
    assert A.size(-2)==B.size(-3)
    A_r, A_i = torch.split(A, 1, dim=-1)
    B_r, B_i = torch.split(B, 1, dim=-1)

    C_r = (torch.matmul(A_r.squeeze(-1), B_r.squeeze(-1))
        -torch.matmul(A_i.squeeze(-1), B_i.squeeze(-1))).unsqueeze(-1)
    C_i = (torch.matmul(A_r.squeeze(-1), B_i.squeeze(-1))
        +torch.matmul(A_i.squeeze(-1), B_r.squeeze(-1))).unsqueeze(-1)

    C   = torch.cat((C_r, C_i), dim=-1)
    return C

def complex_ctranspose(A):
    """
    Apply complex transpose for seperate channel

    Args:
        A (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
            (..., n, m, 2)

    Returns:
        At (torch.Tensor): 
            (..., n, m, 2)
    """
    assert A.size(-1)==2
    A_r, A_i = torch.split(A, 1, dim=-1)

    At_r    = torch.transpose(A_r, -3, -2)
    At_i    = torch.transpose(-A_i, -3, -2)
    At      = torch.cat((At_r, At_i), dim=-1)
    return At

def fft2(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The FFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.fft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data

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
    data = ifftshift(data, dim=(-3, -2))
    data = torch.ifft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data


def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1).sqrt()


def root_sum_of_squares(data, dim=0):
    """
    Compute the Root Sum of Squares (RSS) transform along a given dimension of a tensor.

    Args:
        data (torch.Tensor): The input tensor
        dim (int): The dimensions along which to apply the RSS transform

    Returns:
        torch.Tensor: The RSS value
    """
    return torch.sqrt((data ** 2).sum(dim))


def center_crop(data, shape):
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data (torch.Tensor): The input tensor to be center cropped. It should have at
            least 2 dimensions and the cropping is applied along the last two dimensions.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]


def complex_center_crop(data, shape):
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data (torch.Tensor): The complex input tensor to be center cropped. It should
            have at least 3 dimensions and the cropping is applied along dimensions
            -3 and -2 and the last dimensions should have a size of 2.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-3]
    assert 0 < shape[1] <= data.shape[-2]
    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to, :]


def normalize(data, mean, stddev, eps=0.):
    """
    Normalize the given tensor using:
        (data - mean) / (stddev + eps)

    Args:
        data (torch.Tensor): Input data to be normalized
        mean (float): Mean value
        stddev (float): Standard deviation
        eps (float): Added to stddev to prevent dividing by zero

    Returns:
        torch.Tensor: Normalized tensor
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(data, eps=0.):
    """
        Normalize the given tensor using:
            (data - mean) / (stddev + eps)
        where mean and stddev are computed from the data itself.

        Args:
            data (torch.Tensor): Input data to be normalized
            eps (float): Added to stddev to prevent dividing by zero

        Returns:
            torch.Tensor: Normalized tensor
        """
    mean = data.mean()
    std = data.std()
    return normalize(data, mean, std, eps), mean, std

def normalize_max(data, max_val):
    """
    Normalize the given tensor using:
        (data - mean) / (stddev + eps)

    Args:
        data (torch.Tensor): Input data to be normalized
        mean (float): Mean value
        stddev (float): Standard deviation
        eps (float): Added to stddev to prevent dividing by zero

    Returns:
        torch.Tensor: Normalized tensor
    """
    return data / max_val

def normalize_instance_max(data):
    """
        Normalize the given tensor using:
            (data - mean) / (stddev + eps)
        where mean and stddev are computed from the data itself.

        Args:
            data (torch.Tensor): Input data to be normalized
            eps (float): Added to stddev to prevent dividing by zero

        Returns:
            torch.Tensor: Normalized tensor
        """
    abs_data = complex_abs(data)
    max_val  = abs_data.max()
    
    return normalize_max(data, max_val), max_val

# Helper functions

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

def tensor_to_complex(x):
    real = x[:, 0::2, :, :]
    imag = x[:, 1::2, :, :]
    complex = torch.stack((real, imag), 4)
    return complex

def ssos(data, k=True):
    xp = tensor_to_complex(data) # [2,16,128,128,2]
    if k:
        xp = ifft2(xp)
    xp = complex_abs(xp) # [2,16,128,128]
    xp = root_sum_of_squares(xp, dim=1) # [2,128,128]
    xp = torch.unsqueeze(xp,dim=1)
    return xp