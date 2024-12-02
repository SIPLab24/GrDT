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
   - Dynamically combine \( T_\text{texture} \) and \( T_\text{geometric} \) for final output.

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
├── README.md                    # Project documentation
├── requirements.txt             # Required dependencies
├── FRT/
│   ├── main.py                  # Main script for training and inference
│   ├── weight_fusion.py         # Adaptive weight fusion implementation
│   ├── texture_classifier.py    # Texture-based classifier
│   ├── geometric_classifier.py  # Geometric-based classifier
│   ├── loss.py                  # Cross-entropy loss calculation
│   ├── train.py                 # Training pipeline
│   ├── inference.py             # Inference pipeline
├── GNN/
│   ├── sample_data/             # Input example data
│   ├── models/                  # Folder for pre-trained models
│   ├── output/                  # Folder for saving output results
└── tests/
    ├── test_fusion.py           # Test adaptive weight fusion
    ├── test_classifiers.py      # Test classifiers
```

---

## ⚡ Usage

### Training 🏋️
Train the GrDT Network on your dataset:
```bash
python train.py
```
**Optional**: Customize paths and hyperparameters in the script.

---


---

## 🧩 Example Workflow

1. **Input an image**: Add your input image to `data/sample_data/`.
2. **Training**: Train the FRT and GRD paths using `src/train.py`.
3. **Adaptive Fusion**: Combine the predictions using the fusion module during training or inference.
4. **Output**: Check results in `data/output/`.

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
