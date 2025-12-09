#!/usr/bin/env python3
"""
Quick test to verify SimpleCNN can train on the merged dataset.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from dataset_loader import load_imagefolder_dataloaders, train_one_epoch, evaluate
from models import SimpleCNN
from utils import get_device

def main():
    print("=" * 60)
    print("Testing SimpleCNN training on merged dataset")
    print("=" * 60)
    print()
    
    # Load client 1 data as a test
    train_dir = "data/client_1/train"
    test_dir = "data/client_1/test"
    batch_size = 32
    
    print(f"Loading data from {train_dir} and {test_dir}...")
    train_loader, test_loader, num_classes = load_imagefolder_dataloaders(
        train_dir, test_dir, batch_size=batch_size
    )
    
    print(f"✓ Data loaded successfully")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Test batches: {len(test_loader)}")
    print()
    
    # Initialize model
    device = get_device()
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"✓ Model initialized on {device}")
    print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Train for 1 epoch
    print("Training for 1 epoch...")
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"✓ Training completed")
    print(f"  - Training loss: {train_loss:.4f}")
    print()
    
    # Evaluate
    print("Evaluating on test set...")
    test_loss, test_acc = evaluate(model, test_loader, device, criterion=criterion)
    print(f"✓ Evaluation completed")
    print(f"  - Test loss: {test_loss:.4f}")
    print(f"  - Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print()
    
    print("=" * 60)
    print("SUCCESS! Model can train on merged dataset.")
    print("=" * 60)

if __name__ == "__main__":
    main()
