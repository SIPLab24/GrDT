import matplotlib.pyplot as plt

def visualize_keypoints(image, keypoints):
    """
    Visualize facial keypoints on an image.
    Args:
        image (ndarray): Input image.
        keypoints (list): List of (x, y) coordinates for keypoints.
    """
    plt.imshow(image, cmap='gray')
    for x, y in keypoints:
        plt.scatter(x, y, c='red', s=10)
    plt.title("Facial Keypoints")
    plt.show()

def visualize_mask(image, mask):
    """
    Visualize the binary mask over the original image.
    Args:
        image (ndarray): Input image.
        mask (ndarray): Binary mask.
    """
    plt.imshow(image, cmap='gray')
    plt.imshow(mask, cmap='jet', alpha=0.5)
    plt.title("Mask Overlay")
    plt.show()

def plot_texture_features(features, feature_names):
    """
    Plot extracted texture features.
    Args:
        features (list): List of feature values.
        feature_names (list): Names of the features.
    """
    plt.bar(feature_names, features)
    plt.title("Texture Features")
    plt.xlabel("Features")
    plt.ylabel("Values")
    plt.show()
