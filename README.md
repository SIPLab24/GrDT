# 🌟 GrDT Network Architecture 🌟

GrDT (**Geometric Representation and Facial Texture**) Network is a multi-path classification framework that integrates **Facial Representation Texture (FRT)** and **Geometric Representation Distribution (GRD)** through **Adaptive Weight Fusion**. It is designed for robust classification in scenarios where input data may contain varying levels of quality, occlusions, or missing information, such as in deepfake detection or facial analysis tasks.

---

## 🚀 Key Features

### 1. **Facial Representation Texture (FRT) Path** 🖼️
- Extracts features from masked textures using the **Grey Level Co-occurrence Matrix (GLCM)**.
- Captures texture properties like:
  - 🎨 **Contrast**
  - 🧩 **Entropy**
  - 🧾 **Homogeneity**
  - ⚡ **Energy**
- Classifies the extracted texture features using a texture classifier.

---

### 2. **Geometric Representation Distribution (GRD) Path** 🧬
- Processes input images using a **CNN-based architecture** to extract geometric features.
- Constructs an **Adjacency Matrix** for feature points.
- Utilizes **Self-Attention** to enhance feature relationships.
- Outputs classification results from the geometric classifier.

---

### 3. **Adaptive Weight Fusion** ⚖️
- Combines predictions from FRT and GRD paths using **adaptive weights**.
- Mitigates the impact of incomplete or noisy data by dynamically optimizing the weight of each path.
- Computes the final classification result.

---

## 📜 Architecture Workflow

1. **Input**: A base image \( I \).
2. **Landmark Detection**: Extract facial key points for masking.
3. **Path (a) - FRT**:
   - Extract masked texture and compute GLCM features.
   - Classify features using the texture classifier.
4. **Path (b) - GRD**:
   - Process the image with a CNN.
   - Apply self-attention and classify using the geometric classifier.
5. **Adaptive Weight Fusion**:
   - Dynamically combine for final output.

---

## 🛠️ Installation

### Prerequisites
- 🐍 Python >= 3.7
- 🔥 PyTorch >= 1.10
- 🎥 OpenCV >= 4.5

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/GrDT-Network-Architecture.git
   cd GrDT-Network-Architecture
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📂 Directory Structure

```
GrDT-Network-Architecture/
FacialRepresentationTexture/
├── tests/
│   ├── test_keypoint_detection.py   # Unit test for keypoint detection
│   ├── test_mask_generation.py      # Unit test for mask generation and smoothing
│   ├── test_texture_extraction.py   # Unit test for texture feature extraction
├── utils/
│   ├── gaussian_filter.py           # Gaussian filter implementation
│   ├── glcm.py                      # GLCM computation for texture analysis
│   ├── metrics.py                   # Metrics for contrast, entropy, etc.
│   ├── visualization.py             # Tools for visualizing keypoints, masks, features
├── feature_classification.py        # Classification model for texture features
├── keypoint_detection.py            # Facial keypoint detection algorithm
├── mask_generation.py               # ROI mask creation and smoothing
├── texture_extraction.py            # Texture feature extraction logic
├── src.py                           # Main script for combining modules

GNNDeepfakeClassification/
├── data/                            # Directory for input data
├── lib/
│   ├── GRD.py                       # Geometric representation distribution logic
│   ├── util.py                      # Helper functions for GRD path
│   ├── mlp_dropout.py               # Multi-layer perceptron with dropout
│   ├── run_GRD.sh                   # Shell script for running GRD tasks
├── LICENSE                          # Project license
├── main.py                          # Main script for integrating FRT and GRD paths
├── README.md                        # Documentation and instructions
├── requirements.txt                 # Python dependencies
├── geometric_classifier.py          # Classification model for geometric features
├── texture_classifier.py            # Classification model for texture features
├── train.py                         # Training pipeline for FRT and GRD models

```

---

## ⚡ Usage

Here’s the **Training** section of the README for the `GrDT-Network-Architecture` project, based on the provided directory structure:

---

## 🚀 Training

The GrDT Network combines **Facial Representation Texture (FRT)** and **Geometric Representation Distribution (GRD)** paths to train a robust classification model. Follow the steps below to train the network:

---

### Prerequisites

1. Ensure that all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare your dataset:
   - Place input data (e.g., images) in the `GNNDeepfakeClassification/data/` directory.
   - Organize data into appropriate training and testing splits (e.g., `train/` and `test/` subdirectories).

---

### Training the Model

1. Train the **Facial Representation Texture (FRT)** path:
   ```bash
   python FacialRepresentationTexture/src.py
   ```
   This will:
   - Perform keypoint detection using `keypoint_detection.py`.
   - Generate and smooth ROI masks via `mask_generation.py`.
   - Extract texture features using `texture_extraction.py`.
   - Train the texture classifier using the extracted features.

2. Train the **Geometric Representation Distribution (GRD)** path:
   ```bash
   python GNNDeepfakeClassification/train.py
   ```
   This will:
   - Process the images with a CNN to extract geometric features.
   - Construct adjacency matrices and apply self-attention with `GRD.py`.
   - Train the geometric classifier.

3. Combine and train both paths with **Adaptive Weight Fusion**:
   ```bash
   python GNNDeepfakeClassification/main.py
   ```
   This will:
   - Fuse the outputs of the FRT and GRD classifiers.
   - Optimize the adaptive weights for robust classification.

---

### Configuration

Modify the training configurations as needed:
- **FRT Path**:
  - Edit the parameters in `FacialRepresentationTexture/src.py` (e.g., number of keypoints, GLCM settings).
- **GRD Path**:
  - Adjust hyperparameters in `GNNDeepfakeClassification/train.py` (e.g., learning rate, batch size).
- **Fusion**:
  - Update fusion weights and loss function parameters in `GNNDeepfakeClassification/main.py`.

---

### Output

- The trained models will be saved in the `GNNDeepfakeClassification/lib/` directory.
- Logs and metrics (e.g., accuracy, loss) will be saved in the `data/output/` directory.

---

### Example Commands

To train both paths and perform fusion in sequence:
```bash
# Train FRT path
python FacialRepresentationTexture/src.py

# Train GRD path
python GNNDeepfakeClassification/train.py

# Perform adaptive weight fusion
python GNNDeepfakeClassification/main.py
```

---

## 📈 Future Work
- Add support for additional feature extraction paths.
- Explore alternative weighting mechanisms, such as attention-based fusion.
- Integrate larger datasets for enhanced robustness.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

### 🌟 If you find this project helpful, give it a ⭐ on GitHub! 🌟
