import argparse
import json
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common import ndarrays_to_parameters

from dataset_loader import load_global_test_loader, load_imagefolder_dataloaders
from models import EfficientNetB0Extractor, SimpleCNN
from utils import get_device, set_parameters_to_model, get_parameters_from_model


round_accuracies = []  # RF accuracy collected after each federated round


def _save_metrics(accuracies, out_path: str = "metrics.json", extra: Optional[Dict] = None):
    """Persist metrics for external UIs (e.g., Streamlit)."""
    payload = {"accuracies": accuracies}
    if extra:
        payload.update(extra)

    try:
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        print(f"[Server] Failed to write {out_path}: {e}")


def _build_backbone(backbone: str) -> torch.nn.Module:
    backbone = (backbone or "simplecnn").lower()
    if backbone in {"efficientnet", "efficientnet_b0", "effnet_b0"}:
        return EfficientNetB0Extractor(pretrained=True)
    return SimpleCNN()


def _preset_for_backbone(backbone: str) -> str:
    backbone = (backbone or "simplecnn").lower()
    if backbone in {"efficientnet", "efficientnet_b0", "effnet_b0"}:
        return "efficientnet_b0"
    return "simplecnn"


def _train_and_evaluate_global_rf(
    backbone_model: torch.nn.Module,
    client_loaders: List[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]],
    global_test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    rf_out_path: str = "global_rf.pkl",
    pca_out_path: str = "global_pca.pkl",
) -> float:
    """Train a centralized PCA+RandomForest on backbone features and evaluate on global test."""
    try:
        import joblib
        from sklearn.decomposition import PCA
        from sklearn.ensemble import RandomForestClassifier
    except Exception as e:
        raise ImportError(
            "Missing sklearn/joblib dependencies for RandomForest evaluation. "
            "Install with: pip install scikit-learn joblib"
        ) from e

    backbone_model.eval()

    X_train, y_train = [], []
    with torch.no_grad():
        for train_loader, _ in client_loaders:
            for images, labels in train_loader:
                images = images.to(device)
                feats = backbone_model(images)
                X_train.append(feats.cpu().numpy())
                y_train.append(labels.numpy())

    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)

    # Feature noise (matches simple_demo behavior)
    X_train = X_train + 0.01 * np.random.randn(*X_train.shape)

    pca = PCA(n_components=0.90, whiten=True)
    X_train_pca = pca.fit_transform(X_train)

    rf = RandomForestClassifier(
        n_estimators=80,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
    )
    rf.fit(X_train_pca, y_train)

    # Evaluate on global test
    X_test, y_test = [], []
    with torch.no_grad():
        for images, labels in global_test_loader:
            images = images.to(device)
            feats = backbone_model(images)
            X_test.append(feats.cpu().numpy())
            y_test.append(labels.numpy())

    X_test = np.vstack(X_test)
    y_test = np.hstack(y_test)
    X_test_pca = pca.transform(X_test)

    acc = float(rf.score(X_test_pca, y_test))

    # Persist models for downstream inference
    joblib.dump(rf, rf_out_path)
    joblib.dump(pca, pca_out_path)

    return acc


def get_evaluate_fn(
    model: torch.nn.Module,
    global_test_loader,
    device: torch.device,
    backbone: str,
    num_clients: int,
    batch_size: int,
):
    preset = _preset_for_backbone(backbone)

    def evaluate(server_round, parameters, config):
        # Load aggregated backbone weights
        if parameters is not None:
            if hasattr(parameters, "tensors"):
                ndarrays = fl.common.parameters_to_ndarrays(parameters)
            else:
                ndarrays = parameters
            set_parameters_to_model(model, ndarrays)

        # Centralized RF on backbone features (demo-style, assumes server can read client folders)
        client_loaders = []
        for cid in range(1, int(num_clients) + 1):
            train_dir = f"data/client_{cid}/train"
            test_dir = f"data/client_{cid}/test"
            train_loader, test_loader, _ = load_imagefolder_dataloaders(
                train_dir,
                test_dir,
                batch_size=batch_size,
                preset=preset,
            )
            client_loaders.append((train_loader, test_loader))

        rf_acc = _train_and_evaluate_global_rf(
            model,
            client_loaders=client_loaders,
            global_test_loader=global_test_loader,
            device=device,
        )

        round_accuracies.append(rf_acc)
        _save_metrics(
            round_accuracies,
            extra={
                "last_round": int(server_round),
                "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "backbone": preset,
            },
        )

        print(f"[Server] Round {server_round}: GLOBAL RF acc (on global_test) = {rf_acc:.4f}")
        # Flower expects (loss, metrics). We use 1-acc as a pseudo-loss.
        return float(1.0 - rf_acc), {"rf_accuracy": float(rf_acc)}

    return evaluate


def get_on_fit_config_fn(epochs: int, batch_size: int):
    def on_fit_config_fn(server_round: int):
        return {"epochs": epochs, "batch_size": batch_size}

    return on_fit_config_fn


def plot_accuracies(accuracies, out_path: str = "accuracy_curve.png"):
    if not accuracies:
        return
    rounds = list(range(1, len(accuracies) + 1))
    plt.figure(figsize=(6, 4))
    plt.plot(rounds, accuracies, marker="o")
    plt.title("Global RF Accuracy vs Federated Round")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[Server] Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="simplecnn", choices=["simplecnn", "efficientnet_b0"], help="Feature extractor backbone")
    parser.add_argument("--num_clients", type=int, default=3, help="Number of clients (expects data/client_1..data/client_N)")
    parser.add_argument("--num_rounds", type=int, default=5, help="Number of federated rounds")
    parser.add_argument("--epochs", type=int, default=1, help="Local epochs per client per round")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for clients and server RF eval")
    parser.add_argument("--address", type=str, default="127.0.0.1:8080", help="gRPC server address")
    args = parser.parse_args()

    device = get_device()
    preset = _preset_for_backbone(args.backbone)

    # Global test loader for RF evaluation (uses same preset as backbone)
    global_test_loader, _num_classes = load_global_test_loader(
        "data/global_test",
        batch_size=args.batch_size,
        preset=preset,
    )

    # Backbone model (for evaluation and initial parameters)
    model = _build_backbone(args.backbone).to(device)

    # Initial parameters so all clients start from same weights
    initial_ndarrays = get_parameters_from_model(model)
    initial_parameters = ndarrays_to_parameters(initial_ndarrays)

    strategy = fl.server.strategy.FedAvg(
        evaluate_fn=get_evaluate_fn(
            model,
            global_test_loader=global_test_loader,
            device=device,
            backbone=args.backbone,
            num_clients=args.num_clients,
            batch_size=args.batch_size,
        ),
        on_fit_config_fn=get_on_fit_config_fn(args.epochs, args.batch_size),
        initial_parameters=initial_parameters,
    )

    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

    plot_accuracies(round_accuracies, out_path="accuracy_curve.png")


if __name__ == "__main__":
    main()
