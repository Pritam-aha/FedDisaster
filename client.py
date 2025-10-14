import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from dataset_loader import load_imagefolder_dataloaders, train_one_epoch, evaluate
from models import SimpleCNN
from utils import get_device, get_parameters_from_model, set_parameters_to_model


def get_loaders_for_client(cid: int, batch_size: int):
    train_dir = f"data/client_{cid}/train"
    test_dir = f"data/client_{cid}/test"
    return load_imagefolder_dataloaders(train_dir, test_dir, batch_size=batch_size)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: int, batch_size: int = 32, lr: float = 1e-3):
        self.cid = cid
        self.batch_size = batch_size
        self.lr = lr

        # Data
        self.train_loader, self.test_loader, num_classes = get_loaders_for_client(cid, batch_size)

        # Model
        self.device = get_device()
        self.model = SimpleCNN(num_classes=num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def get_parameters(self, config):  # Flower will call this to get current local weights
        return get_parameters_from_model(self.model)

    def fit(self, parameters, config):
        # Load global parameters
        if parameters is not None and len(parameters) > 0:
            set_parameters_to_model(self.model, parameters)

        # Read training config from server
        epochs = int(config.get("epochs", 1))
        batch_size = int(config.get("batch_size", self.batch_size))
        if batch_size != self.batch_size:
            # If the server changed batch size, reload loaders
            self.train_loader, self.test_loader, _ = get_loaders_for_client(self.cid, batch_size)
            self.batch_size = batch_size

        # Train locally
        for _ in range(epochs):
            train_one_epoch(self.model, self.train_loader, self.criterion, self.optimizer, self.device)

        # Evaluate locally after training
        test_loss, test_acc = evaluate(self.model, self.test_loader, self.device, criterion=self.criterion)
        print(f"[Client {self.cid}] Local eval -> loss: {test_loss:.4f}, acc: {test_acc:.4f}")

        # Return updated weights and metrics
        return get_parameters_from_model(self.model), len(self.train_loader.dataset), {"accuracy": test_acc}

    def evaluate(self, parameters, config):
        # Evaluate current (already updated) model on local test set
        if parameters is not None and len(parameters) > 0:
            set_parameters_to_model(self.model, parameters)
        test_loss, test_acc = evaluate(self.model, self.test_loader, self.device, criterion=self.criterion)
        return float(test_loss), len(self.test_loader.dataset), {"accuracy": float(test_acc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True, help="Client ID, e.g., 1")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    client = FlowerClient(cid=args.cid, batch_size=args.batch_size, lr=args.lr)
    fl.client.start_client(server_address="127.0.0.1:8080", client=client.to_client())


if __name__ == "__main__":
    main()
