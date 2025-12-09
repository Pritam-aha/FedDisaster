import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Federated CNN Feature Extractor (NO classifier head).
    Used only to extract features for the centralized Random Forest.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # 64x64 -> 32x32 -> 16x16
        self.feature_dim = 32 * 16 * 16  # 8192

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 64 -> 32
        x = self.pool(F.relu(self.conv2(x)))  # 32 -> 16
        x = x.view(x.size(0), -1)
        return x   # ✅ Features only


# Optional: Local client training head (clients only, NOT shared)
class LocalHead(nn.Module):
    """
    Used only on clients for supervised training.
    NEVER sent to the server.
    """

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
