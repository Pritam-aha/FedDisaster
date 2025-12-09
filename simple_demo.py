#!/usr/bin/env python3
"""
Federated Learning Demo with Global Random Forest Classifier
CNN is trained via FedAvg
Random Forest is trained centrally on extracted CNN features
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import joblib

from models import SimpleCNN
from dataset_loader import (
    load_imagefolder_dataloaders,
    load_global_test_loader,
    train_one_epoch,
)
from utils import get_device, get_parameters_from_model, set_parameters_to_model
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

class LocalHead(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# ---------------------------------------------------------
# ✅ GLOBAL RF EVALUATION FUNCTION
# ---------------------------------------------------------
def evaluate_with_rf(cnn_model, test_loader, device):
    cnn_model.eval()
    X, y = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            features = cnn_model(images)
            X.append(features.cpu().numpy())
            y.append(labels.numpy())
    X = np.vstack(X)
    y = np.hstack(y)

    # Adaptive PCA
    n_samples, n_features = X.shape
    n_components = min(128, n_samples, n_features)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_pca, y)
    acc = rf.score(X_pca, y)
    print(f"🎯 GLOBAL RF ACCURACY: {acc:.4f}")
    return acc



# ---------------------------------------------------------
# ✅ FEDERATED ROUND
# ---------------------------------------------------------
def simulate_federated_round(
    client_models, client_loaders, global_model, device, criterion, round_num
):
    print(f"\n🔄 FEDERATED ROUND {round_num}")
    print("=" * 50)

    global_params = get_parameters_from_model(global_model)
    client_updates = []
    client_sizes = []

        # ---- CLIENT SIDE TRAINING ----
    for cid, (model, (train_loader, test_loader)) in enumerate(
        zip(client_models, client_loaders), 1
    ):
        print(f"\n📱 CLIENT {cid} LOCAL TRAINING:")

        set_parameters_to_model(model, global_params)

        # ✅ Create a LOCAL classification head (NOT federated)
        local_head = LocalHead(model.feature_dim, 2).to(device)

        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(local_head.parameters()), lr=1e-3
        )

        model.train()
        local_head.train()

        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # ✅ Forward: CNN → Features → Local Head
            features = model(images)
            outputs = local_head(features)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        print(f"   Local train loss: {train_loss:.4f}")

        client_updates.append(get_parameters_from_model(model))
        client_sizes.append(len(train_loader.dataset))



# ---------------------------------------------------------
# ✅ MAIN
# ---------------------------------------------------------
def main():
    print("🌊 FEDERATED LEARNING + RANDOM FOREST (GLOBAL)")
    print("=" * 60)

    device = get_device()

    # ✅ FEATURE LEARNING ONLY (NO CLASSIFIER LOSS)
    criterion = nn.CrossEntropyLoss()  # ✅ correct for classification


    # ---- LOAD CLIENT DATA ----
    print("📁 LOADING CLIENT DATA:")
    client_loaders = []

    for cid in [1, 2, 3]:
        train_loader, test_loader, _ = load_imagefolder_dataloaders(
            f"data/client_{cid}/train",
            f"data/client_{cid}/test",
            batch_size=16,
        )
        client_loaders.append((train_loader, test_loader))
        print(
            f"   Client {cid}: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test"
        )

    # ---- LOAD GLOBAL TEST ----
    global_test_loader, num_classes = load_global_test_loader(
        "data/global_test", batch_size=16
    )
    print(f"   Global test: {len(global_test_loader.dataset)} images")

    # ---- INIT MODELS ----
    print(f"\n🤖 INITIALIZING MODELS:")
    global_model = SimpleCNN().to(device)
    client_models = [SimpleCNN().to(device) for _ in client_loaders]

    # ---- STREAMLIT METRICS ----
    def update_streamlit_metrics(accuracies, training_complete=False):
        metrics = {
            "accuracies": accuracies,
            "training_complete": training_complete,
            "last_updated": datetime.now().isoformat(),
            "rounds_expected": 3,
        }
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    accuracies = []

    # ---- FEDERATED ROUNDS ----
    num_rounds = 3
    for round_num in range(1, num_rounds + 1):
        simulate_federated_round(
            client_models,
            client_loaders,
            global_model,
            device,
            criterion,
            round_num,
        )

        # ✅ TRAIN + EVALUATE RANDOM FOREST
        global_acc = evaluate_with_rf(global_model, global_test_loader, device)
        accuracies.append(global_acc)

        print(
            f"\n🎯 GLOBAL RF ACCURACY AFTER ROUND {round_num}: {global_acc:.4f}"
        )

        update_streamlit_metrics(
            accuracies, training_complete=(round_num == num_rounds)
        )

        print(f"   ⏸️  Pause for dashboard update...")
        time.sleep(5)

    print(f"\n🏆 FINAL GLOBAL RF ACCURACY: {accuracies[-1]:.4f}")
    print(f"✅ Global CNN saved: global_cnn.pt")
    print(f"✅ Global RF saved: global_rf.pkl")
    print(f"✅ Global PCA saved: global_pca.pkl")
    print(f"✅ Streamlit ready")


if __name__ == "__main__":
    main()
