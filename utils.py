import torch
from typing import List


def get_device() -> torch.device:
    """Return CPU device (prototype runs on CPU only)."""
    return torch.device("cpu")


def get_parameters_from_model(model: torch.nn.Module) -> List:
    """Extract model parameters as a list of numpy arrays.

    The order of parameters follows state_dict() item order and must be
    consistent between client and server.
    """
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters_to_model(model: torch.nn.Module, parameters: List) -> None:
    """Load a list of numpy arrays into model.state_dict() order."""
    state_dict = model.state_dict()
    new_state_dict = {}
    for (k, v), p in zip(state_dict.items(), parameters):
        new_state_dict[k] = torch.tensor(p)
    model.load_state_dict(new_state_dict, strict=True)
