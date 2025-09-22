# Federated Learning Prototype: Flood-Damage Detection (Flower + PyTorch)

This is a minimal, working federated learning prototype that uses Flower (flwr) and PyTorch for a flood-damage detection task.
It supports offline local datasets per client and a held-out global test set for server-side evaluation.

Project structure:
- data/
  - client_1/train/ ... images per class (e.g., damaged/, not_damaged/)
  - client_1/test/  ... images per class
  - client_2/train/ ...
  - client_2/test/  ...
  - global_test/    ... images per class
- client.py         ... Flower NumPyClient implementation (local train + evaluation)
- server.py         ... FedAvg server with server-side eval on global_test each round; plots accuracy vs round
- dataset_loader.py ... Data loading utilities for image folders (ImageFolder)
- models.py         ... Small CNN for images
- utils.py          ... Shared utilities (parameter conversion, device selection)
- requirements.txt  ... flwr, torch, torchvision, matplotlib, numpy

Assumptions:
- This prototype uses image data via torchvision.datasets.ImageFolder (recommended).
- For flood-damage detection, place images into per-class subfolders (e.g., damaged/ and not_damaged/).
- CPU only; no GPU needed.

1) Prepare offline data
- Put your client-specific datasets here:
  data/client_1/train/<class_name>/*.jpg|png
  data/client_1/test/<class_name>/*.jpg|png
  data/client_2/train/<class_name>/*.jpg|png
  data/client_2/test/<class_name>/*.jpg|png
  ...
- Put the held-out global test set here:
  data/global_test/<class_name>/*.jpg|png

2) Install dependencies (recommended in a virtual environment)
- python -m venv .venv
- .venv\\Scripts\\activate   # PowerShell on Windows
- pip install -r requirements.txt

3) Run the server (in one terminal)
- python server.py --num_rounds 5 --epochs 1 --batch_size 32
  - After every round, the server evaluates the aggregated model on data/global_test and logs accuracy.
  - At the end, it saves accuracy_curve.png.

4) Run three client instances (in separate terminals)
- python client.py --cid 1
- python client.py --cid 2
- python client.py --cid 3

Notes
- You can choose any number of clients; the server will wait for available clients each round.
- All accuracy numbers are computed from actual data; there is no dummy/random evaluation.
- Code includes comments to help beginners modify the model and training settings.
