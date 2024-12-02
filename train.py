import torch
from torch.optim import Adam
from weight_fusion import AdaptiveWeightFusion
from texture_classifier import TextureClassifier
from geometric_classifier import GeometricClassifier
from loss import cross_entropy_loss

import torch.nn.functional as F

def cross_entropy_loss(predictions, labels):
    """
    Compute cross-entropy loss.
    Args:
        predictions (Tensor): Model predictions.
        labels (Tensor): Ground truth labels.
    Returns:
        Tensor: Cross-entropy loss value.
    """
    return F.cross_entropy(predictions, labels)

# Initialize classifiers and fusion module
texture_model = TextureClassifier(input_dim=256, output_dim=2)
geometric_model = GeometricClassifier(input_dim=256, output_dim=2)
fusion_model = AdaptiveWeightFusion()

# Optimizer
optimizer = Adam(list(texture_model.parameters()) + 
                 list(geometric_model.parameters()) + 
                 list(fusion_model.parameters()), lr=0.001)

# Dummy data
X_texture = torch.rand(32, 256)  # Batch of 32 texture features
X_geometric = torch.rand(32, 256)  # Batch of 32 geometric features
y = torch.randint(0, 2, (32,))  # Ground truth labels

# Training loop
for epoch in range(10):
    # Forward pass
    T_G = texture_model(X_texture)
    T_F = geometric_model(X_geometric)
    T_total = fusion_model.forward(T_G, T_F)

    # Compute loss
    loss = cross_entropy_loss(T_total, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
