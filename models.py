import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """A small CNN suitable for 64x64 RGB images.

    Adjust channels and layers as needed; this is intentionally lightweight
    for CPU training in federated rounds.
    """

    def __init__(self, num_classes: int, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        # After two pools on 64x64 -> 16x16 feature map with 32 channels
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 64->32
        x = self.pool(F.relu(self.conv2(x)))  # 32->16
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
