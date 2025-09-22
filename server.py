import argparse
import matplotlib.pyplot as plt
import flwr as fl
import torch
import torch.nn as nn
from flwr.common import ndarrays_to_parameters

from dataset_loader import load_global_test_loader
from models import SimpleCNN
from utils import get_device, set_parameters_to_model, get_parameters_from_model


round_accuracies = []  # Collected after each federated round on global test


def get_evaluate_fn(model, test_loader, device):
    criterion = nn.CrossEntropyLoss()

    def evaluate(server_round: int, parameters, config):
        # Update model with the aggregated parameters
        if parameters is not None:
            # Flower passes Parameters object; convert to ndarrays
            # In recent flwr versions, parameters.tensors may already be bytes
            # We can use flwr.common.parameters_to_ndarrays for safety
            ndarrays = fl.common.parameters_to_ndarrays(parameters)
            set_parameters_to_model(model, ndarrays)

        loss, acc = 0.0, 0.0
        model.eval()
        with torch.no_grad():
            import dataset_loader as dl  # local import to reuse evaluate helper
            loss, acc = dl.evaluate(model, test_loader, device, criterion=criterion)

        round_accuracies.append(acc)
        print(f"[Server] Round {server_round}: global_test acc = {acc:.4f}")
        # Return a loss and a dictionary of metrics
        return float(loss), {"accuracy": float(acc)}

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
    plt.title("Global Test Accuracy vs Federated Round")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[Server] Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_rounds", type=int, default=5, help="Number of federated rounds")
    parser.add_argument("--epochs", type=int, default=1, help="Local epochs per client per round")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for clients and server eval")
    parser.add_argument("--address", type=str, default="127.0.0.1:8080", help="gRPC server address")
    args = parser.parse_args()

    device = get_device()

    # Prepare global test loader to evaluate aggregated model
    global_test_loader, num_classes = load_global_test_loader("data/global_test", batch_size=args.batch_size)

    # Build model (for evaluation and initial parameters)
    model = SimpleCNN(num_classes=num_classes).to(device)

    # Provide initial parameters to ensure all clients start from the same weights
    initial_ndarrays = get_parameters_from_model(model)
    initial_parameters = ndarrays_to_parameters(initial_ndarrays)

    # Strategy with evaluation and fit config
    strategy = fl.server.strategy.FedAvg(
        evaluate_fn=get_evaluate_fn(model, global_test_loader, device),
        on_fit_config_fn=get_on_fit_config_fn(args.epochs, args.batch_size),
        initial_parameters=initial_parameters,
    )

    # Start the Flower server
    fl.server.start_server(server_address=args.address, config=fl.server.ServerConfig(num_rounds=args.num_rounds), strategy=strategy)

    # After training completes, plot the global test accuracy over rounds
    plot_accuracies(round_accuracies, out_path="accuracy_curve.png")


if __name__ == "__main__":
    main()
