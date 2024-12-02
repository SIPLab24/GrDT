import torch.nn as nn

class GeometricClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GeometricClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)
