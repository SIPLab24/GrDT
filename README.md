# GrDT
Towards Robust Deepfake Detection using Geometric Representation Distribution and Texture

# Facial Representation Texture (FRT) Analysis

This repository implements the Facial Representation Texture (FRT) algorithm for texture-based face region analysis. The pipeline involves:

1. **Facial Keypoint Detection**: Detect keypoints corresponding to facial regions of interest.
2. **Mask Generation**: Create and smooth binary masks for regions of interest (ROIs).
3. **Texture Feature Extraction**: Compute texture features using Grey Level Co-occurrence Matrix (GLCM).
4. **Classification**: Use a Random Forest classifier for texture feature classification.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/FRT-Texture-Analysis.git
cd FRT-Texture-Analysis
pip install -r requirements.txt

