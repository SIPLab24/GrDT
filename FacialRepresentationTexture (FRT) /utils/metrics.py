import numpy as np

def compute_contrast(glcm):
    """
    Compute contrast from GLCM.
    Args:
        glcm (ndarray): Input GLCM.
    Returns:
        contrast (float): Contrast value.
    """
    indices = np.arange(glcm.shape[0])
    i, j = np.meshgrid(indices, indices)
    contrast = np.sum((i - j)**2 * glcm)
    return contrast

def compute_entropy(glcm):
    """
    Compute entropy from GLCM.
    Args:
        glcm (ndarray): Input GLCM.
    Returns:
        entropy (float): Entropy value.
    """
    entropy = -np.sum(glcm * np.log2(glcm + 1e-12))  # Avoid log(0)
    return entropy

def compute_homogeneity(glcm):
    """
    Compute homogeneity from GLCM.
    Args:
        glcm (ndarray): Input GLCM.
    Returns:
        homogeneity (float): Homogeneity value.
    """
    indices = np.arange(glcm.shape[0])
    i, j = np.meshgrid(indices, indices)
    homogeneity = np.sum(glcm / (1 + np.abs(i - j)))
    return homogeneity

def compute_energy(glcm):
    """
    Compute energy from GLCM.
    Args:
        glcm (ndarray): Input GLCM.
    Returns:
        energy (float): Energy value.
    """
    energy = np.sum(glcm**2)
    return energy
