from typing import Tuple
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMG_SIZE = 64  # Small input size to keep the CNN lightweight


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Return train and test transforms for images.

    - Resize to a fixed size and convert to tensor.
    - Basic normalization (0.5 mean/std for all channels).
    """
    common = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    train_tfms = transforms.Compose(common)
    test_tfms = transforms.Compose(common)
    return train_tfms, test_tfms


def load_imagefolder_dataloaders(
    train_dir: str,
    test_dir: str,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader, int]:
    """Create ImageFolder-based train/test DataLoaders and return num_classes.

    Expects directory layout:
      train_dir/<class_name>/*.{jpg,png,...}
      test_dir/<class_name>/*.{jpg,png,...}
    """
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    train_tfms, test_tfms = build_transforms()
    train_ds = datasets.ImageFolder(root=train_dir, transform=train_tfms)
    test_ds = datasets.ImageFolder(root=test_dir, transform=test_tfms)

    num_classes = len(train_ds.classes)
    if num_classes <= 1:
        raise ValueError(
            "Detected <=1 class. Ensure data is organized in subfolders per class."
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader, num_classes


def load_global_test_loader(global_test_dir: str, batch_size: int = 32) -> Tuple[DataLoader, int]:
    """Create a DataLoader for the global held-out test set and return (loader, num_classes)."""
    if not os.path.isdir(global_test_dir):
        raise FileNotFoundError(f"Global test directory not found: {global_test_dir}")

    _, test_tfms = build_transforms()
    test_ds = datasets.ImageFolder(root=global_test_dir, transform=test_tfms)
    num_classes = len(test_ds.classes)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return test_loader, num_classes


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device, criterion=None) -> Tuple[float, float]:
    """Return (loss, accuracy) with optional criterion for loss."""
    model.eval()
    correct, total = 0, 0
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total if total > 0 else 0.0
    avg_loss = (total_loss / total) if (criterion is not None and total > 0) else 0.0
    return avg_loss, acc
