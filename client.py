import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl

from dataset_loader import load_imagefolder_dataloaders
from models import EfficientNetB0Extractor, SimpleCNN, LocalHead
from utils import get_device, get_parameters_from_model, set_parameters_to_model


def _preset_for_backbone(backbone: str) -> str:
    backbone = (backbone or "simplecnn").lower()
    if backbone in {"efficientnet", "efficientnet_b0", "effnet_b0"}:
        return "efficientnet_b0"
    return "simplecnn"


def _build_backbone(backbone: str) -> torch.nn.Module:
    backbone = (backbone or "simplecnn").lower()
    if backbone in {"efficientnet", "efficientnet_b0", "effnet_b0"}:
        return EfficientNetB0Extractor(pretrained=True)
    return SimpleCNN()


def get_loaders_for_client(cid: int, batch_size: int, preset: str):
    train_dir = f"data/client_{cid}/train"
    test_dir = f"data/client_{cid}/test"
    return load_imagefolder_dataloaders(train_dir, test_dir, batch_size=batch_size, preset=preset)


class FlowerClient(fl.client.NumPyClient):
    """
    Federated client:
    - Receives global CNN feature extractor
    - Trains a LOCAL classification head
    - Sends back ONLY the CNN parameters (FedAvg)
    """

    def __init__(self, cid: int, batch_size: int = 32, lr: float = 1e-3, backbone: str = "simplecnn"):
        self.cid = cid
        self.batch_size = batch_size
        self.lr = lr
        self.backbone = backbone
        self.preset = _preset_for_backbone(backbone)

        # ---- Data ----
        self.train_loader, self.test_loader, self.num_classes = get_loaders_for_client(cid, batch_size, preset=self.preset)

        # ---- Device ----
        self.device = get_device()

        # ---- Global Federated Model (Feature Extractor) ----
        self.model = _build_backbone(backbone).to(self.device)

        # ---- Local Head (NOT SHARED) ----
        self.local_head = LocalHead(self.model.feature_dim, self.num_classes).to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        # Optimizer: train only the local head by default (fast & stable)
        self.optimizer = optim.Adam(self.local_head.parameters(), lr=self.lr)

    # Flower will call this to get current local weights (CNN ONLY)
    def get_parameters(self, config):
        return get_parameters_from_model(self.model)

    def fit(self, parameters, config):
        # ---- Load global CNN parameters ----
        if parameters is not None and len(parameters) > 0:
            set_parameters_to_model(self.model, parameters)

        self.model.train()
        self.local_head.train()

        # Read training config from server
        epochs = int(config.get("epochs", 1))
        batch_size = int(config.get("batch_size", self.batch_size))

        if batch_size != self.batch_size:
            self.train_loader, self.test_loader, _ = get_loaders_for_client(self.cid, batch_size, preset=self.preset)
            self.batch_size = batch_size

        # ---- Local Training (CNN frozen, Head trained) ----
        for epoch in range(epochs):
            running_loss = 0.0
            total = 0

            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                # CNN feature extraction
                with torch.no_grad():  # ✅ Freeze CNN
                    features = self.model(images)

                # Classification via local head
                logits = self.local_head(features)
                loss = self.criterion(logits, labels)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)
                total += images.size(0)

            avg_loss = running_loss / (total + 1e-12)
            print(f"[Client {self.cid}] Epoch {epoch+1}/{epochs} train loss: {avg_loss:.4f}")

        # ---- Local Evaluation (for monitoring only) ----
        self.model.eval()
        self.local_head.eval()

        correct = 0
        total = 0
        test_loss = 0.0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                features = self.model(images)
                logits = self.local_head(features)

                loss = self.criterion(logits, labels)
                preds = logits.argmax(dim=1)

                test_loss += loss.item() * images.size(0)
                correct += (preds == labels).sum().item()
                total += images.size(0)

        test_loss /= (total + 1e-12)
        test_acc = correct / (total + 1e-12)

        print(f"[Client {self.cid}] Local eval -> loss: {test_loss:.4f}, acc: {test_acc:.4f}")

        # ✅ Return ONLY CNN weights to server
        return get_parameters_from_model(self.model), len(self.train_loader.dataset), {"accuracy": test_acc}

    def evaluate(self, parameters, config):
        # Evaluate current global CNN + local head for reporting only
        if parameters is not None and len(parameters) > 0:
            set_parameters_to_model(self.model, parameters)

        self.model.eval()
        self.local_head.eval()

        correct = 0
        total = 0
        test_loss = 0.0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                features = self.model(images)
                logits = self.local_head(features)

                loss = self.criterion(logits, labels)
                preds = logits.argmax(dim=1)

                test_loss += loss.item() * images.size(0)
                correct += (preds == labels).sum().item()
                total += images.size(0)

        test_loss /= (total + 1e-12)
        test_acc = correct / (total + 1e-12)

        return float(test_loss), len(self.test_loader.dataset), {"accuracy": float(test_acc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True, help="Client ID, e.g., 1")
    parser.add_argument("--backbone", type=str, default="simplecnn", choices=["simplecnn", "efficientnet_b0"], help="Feature extractor backbone")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--address", type=str, default="127.0.0.1:8080", help="gRPC server address")
    args = parser.parse_args()

    client = FlowerClient(cid=args.cid, batch_size=args.batch_size, lr=args.lr, backbone=args.backbone)
    fl.client.start_client(server_address=args.address, client=client.to_client())


if __name__ == "__main__":
    main()
