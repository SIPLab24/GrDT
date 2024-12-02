import numpy as np

def compute_glcm(image, distance, angle):
    H, W = image.shape
    glcm = np.zeros((2, 2))  # For binary images
    dx, dy = {
        0: (0, distance),
        45: (-distance, distance),
        90: (-distance, 0),
        135: (-distance, -distance),
    }[angle]

    for u in range(H):
        for v in range(W):
            neighbor_x, neighbor_y = u + dx, v + dy
            if 0 <= neighbor_x < H and 0 <= neighbor_y < W:
                i, j = image[u, v], image[neighbor_x, neighbor_y]
                glcm[i, j] += 1

    return glcm / glcm.sum()