from src.mask_generation import generate_mask, smooth_mask
import numpy as np

def test_mask_generation():
    image_shape = (256, 256)
    keypoints = [(50, 50), (100, 100), (150, 150)]
    mask = generate_mask(image_shape, keypoints)
    assert mask.sum() > 0, "Mask generation failed!"
    
    smoothed_mask = smooth_mask(mask, sigma=2)
    assert np.max(smoothed_mask) <= 1, "Mask smoothing failed!"
    print("Mask generation and smoothing test passed!")
