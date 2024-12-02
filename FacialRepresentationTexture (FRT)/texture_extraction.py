import numpy as np
from utils.glcm import compute_glcm

def extract_texture_features(masked_image, distances=[1], angles=[0, 45, 90, 135]):
    """
    Extract texture features (Contrast, Entropy, Homogeneity, Energy) from masked image.
    Args:
        masked_image (ndarray): Image with masked regions.
        distances (list): Pixel distances for GLCM computation.
        angles (list): Angles (in degrees) for GLCM computation.
    Returns:
        features (ndarray): Flattened feature vector.
    """
    H, W = masked_image.shape
    features = []

    # Binarize the image
    binary_image = (masked_image > 127).astype(np.uint8)

    for d in distances:
        for angle in angles:
            glcm = compute_glcm(binary_image, d, angle)

            # Compute features
            contrast = np.sum((np.arange(2)[:, None] - np.arange(2)) ** 2 * glcm)
            entropy = -np.sum(glcm * np.log2(glcm + 1e-12))
            homogeneity = np.sum(glcm / (1 + np.abs(np.arange(2)[:, None] - np.arange(2))))
            energy = np.sum(glcm ** 2)

            features.extend([contrast, entropy, homogeneity, energy])

    return np.array(features)
