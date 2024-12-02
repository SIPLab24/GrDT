import numpy as np
from utils.gaussian_filter import gaussian_filter

def generate_mask(image_shape, keypoints):
    """
    Generate binary mask from keypoints.
    Args:
        image_shape (tuple): Shape of the image (H, W).
        keypoints (list): List of (x, y) keypoints.
    Returns:
        mask (ndarray): Binary mask.
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for x, y in keypoints:
        cv2.circle(mask, (x, y), radius=10, color=1, thickness=-1)  # ROI regions
    return mask

def smooth_mask(mask, sigma=2):
    """
    Smooth the binary mask using a Gaussian filter.
    Args:
        mask (ndarray): Binary mask.
        sigma (float): Standard deviation of Gaussian filter.
    Returns:
        smoothed_mask (ndarray): Smoothed mask.
    """
    kernel_size = int(np.ceil(3 * sigma) * 2 + 1)
    gaussian_kernel = gaussian_filter(kernel_size, sigma)
    smoothed_mask = cv2.filter2D(mask, -1, gaussian_kernel)
    return smoothed_mask
