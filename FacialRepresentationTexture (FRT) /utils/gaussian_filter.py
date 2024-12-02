import numpy as np

def gaussian_filter(kernel_size, sigma):
    """
    Generate a Gaussian filter kernel.
    Args:
        kernel_size (int): Size of the kernel (should be odd).
        sigma (float): Standard deviation of the Gaussian distribution.
    Returns:
        kernel (ndarray): Gaussian filter kernel.
    """
    k = (kernel_size - 1) // 2
    i, j = np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1))
    g = (1 / (2 * np.pi * sigma**2)) * np.exp(-(i**2 + j**2) / (2 * sigma**2))
    return g / g.sum()
