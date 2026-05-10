# proxclient.py  —  FedProx + ChaCha20-Poly1305 + zlib
# Install: pip install pycryptodome flower torch torchvision
# Run:     python proxclient.py client1
#
# BEFORE RUNNING:
#   Set the same FL_SHARED_KEY that the server uses:
#     Windows PowerShell:  $env:FL_SHARED_KEY = "<hex_output>"
#     Linux / macOS:       export FL_SHARED_KEY="<hex_output>"
#
# FedProx CLIENT NOTE:
#   L_fedprox(w) = L(w) + (mu/2) * ||w - w_global||²
#   When mu=0 this is identical to FedAvg.

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import flwr as fl

from proxcrypto import ChaCha20Cipher, pack_parameters, unpack_parameters, is_packed


# ── THE ONLY CHANGE FROM THE ORIGINAL ───────────────────────────────
# DATA_BASE_DIR was previously read at module import time:
#   DATA_BASE_DIR = os.environ.get("FL_DATA_DIR", <fallback>)
# That ran before pipeline_cli.py could set FL_DATA_DIR in the env,
# so the env var was always missed and the fallback path was used.
#
# Fix: read it inside load_data() at call time so it is always fresh.
#   • Manual run  (python proxclient.py client1) → uses fallback path  ✔
#   • Pipeline run (launched by pipeline_cli.py) → uses FL_DATA_DIR    ✔
# ────────────────────────────────────────────────────────────────────
_FALLBACK_DATA_DIR = os.getcwd()


def load_data(client_name: str):
    DATA_BASE_DIR = os.environ.get("FL_DATA_DIR", _FALLBACK_DATA_DIR)
    data_dir      = os.path.join(DATA_BASE_DIR, client_name)

    print(f"📂 FL_DATA_DIR : {DATA_BASE_DIR}", flush=True)
    print(f"📂 data_dir    : {data_dir}", flush=True)

    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "Set FL_DATA_DIR environment variable to the correct base path."
        )
    if not os.path.exists(os.path.join(data_dir, "train")):
        raise FileNotFoundError(f"train/ folder not found inside: {data_dir}")
    if not os.path.exists(os.path.join(data_dir, "test")):
        raise FileNotFoundError(f"test/  folder not found inside: {data_dir}")

    train_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tfms)
    test_ds  = datasets.ImageFolder(os.path.join(data_dir, "test"),  transform=test_tfms)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=0)

    return train_loader, test_loader, len(train_ds), len(test_ds), train_ds.classes


def get_model(num_classes: int):
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = None
        self.early_stop = False

    def __call__(self, val_loss: float):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_fedprox(model, global_params, train_loader, criterion, optimizer, mu, epochs=1):
    model.train()
    for epoch in range(epochs):
        epoch_task_loss = epoch_prox_loss = num_batches = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs   = model(imgs)
            task_loss = criterion(outputs, labels)
            prox_loss = torch.tensor(0.0, device=device)
            if mu > 0.0:
                for w, w_global in zip(model.parameters(), global_params):
                    prox_loss = prox_loss + torch.sum((w - w_global) ** 2)
                prox_loss = (mu / 2.0) * prox_loss
            (task_loss + prox_loss).backward()
            optimizer.step()
            epoch_task_loss += task_loss.item()
            epoch_prox_loss += prox_loss.item() if mu > 0.0 else 0.0
            num_batches     += 1
        avg_task = epoch_task_loss / max(num_batches, 1)
        avg_prox = epoch_prox_loss / max(num_batches, 1)
        print(f"   Epoch {epoch+1}/{epochs} – "
              f"task_loss={avg_task:.4f}  prox_loss={avg_prox:.4f}  "
              f"total={avg_task+avg_prox:.4f}", flush=True)


def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs     = model(imgs)
            total_loss += criterion(outputs, labels).item() * imgs.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
    n = len(test_loader.dataset)
    return total_loss / n, correct / n


class FedProxClient(fl.client.NumPyClient):

    def __init__(self, client_name: str):
        self.train_loader, self.test_loader, self.train_size, \
            self.test_size, self.classes = load_data(client_name)
        self.model         = get_model(num_classes=len(self.classes)).to(device)
        self.criterion     = nn.CrossEntropyLoss()
        self.optimizer     = optim.Adam(self.model.parameters(), lr=1e-4)
        self.early_stop    = EarlyStopping(patience=3, min_delta=0.001)
        self.cipher        = ChaCha20Cipher()
        self.round_num     = 0
        self.global_params: list = []
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"🔐 ChaCha20-Poly1305 + zlib compression initialised", flush=True)
        print(f"📐 FedProx client ready", flush=True)
        print(f"✅ Client '{client_name}'  |  parameters: {total_params:,}", flush=True)
        print(f"   Train: {self.train_size}  |  Test: {self.test_size}", flush=True)
        print(f"   Classes: {self.classes}", flush=True)

    def get_parameters(self, config) -> list:
        params = [v.cpu().numpy() for v in self.model.state_dict().values()]
        print("🔒 Compressing & encrypting parameters...", flush=True)
        packed = pack_parameters(params, self.cipher)
        original_bytes   = sum(p.nbytes for p in params)
        compressed_bytes = packed.nbytes
        ratio = (1 - compressed_bytes / original_bytes) * 100
        print(f"✅ Packed: {original_bytes/1e6:.2f} MB → {compressed_bytes/1e6:.2f} MB "
              f"({ratio:.1f}% reduction)", flush=True)
        return [packed]

    def set_parameters(self, parameters: list):
        if is_packed(parameters):
            print("🔓 Decrypting & decompressing received parameters...", flush=True)
            try:
                params = unpack_parameters(parameters[0], self.cipher)
            except ValueError as e:
                raise RuntimeError(
                    f"Integrity check failed on received parameters: {e}\n"
                    "Payload may be tampered. Aborting round."
                ) from e
            print(f"✅ Unpacked {len(params)} parameter tensors", flush=True)
        else:
            params = parameters
        state_dict = self.model.state_dict()
        new_state  = {}
        for k, v in zip(state_dict.keys(), params):
            t = torch.tensor(v)
            if state_dict[k].size() != t.size():
                raise ValueError(
                    f"Size mismatch for layer '{k}': "
                    f"expected {state_dict[k].size()}, got {t.size()}."
                )
            new_state[k] = t.to(device)
        self.model.load_state_dict(new_state, strict=True)
        self.global_params = [p.detach().clone() for p in self.model.parameters()]

    def fit(self, parameters, config):
        self.round_num += 1
        mu = float(config.get("mu", 0.01))
        print(f"\n{'='*60}", flush=True)
        print(f"🔄 Round {self.round_num} – FedProx Fit  (μ={mu})", flush=True)
        print(f"{'='*60}", flush=True)
        self.set_parameters(parameters)
        if self.early_stop.early_stop:
            print("⏹  Early stopping active – skipping training this round", flush=True)
            return self.get_parameters({}), 0, {"early_stopped": True}
        train_fedprox(self.model, self.global_params, self.train_loader,
                      self.criterion, self.optimizer, mu, epochs=1)
        print(f"✅ FedProx training complete  (samples: {self.train_size}, μ={mu})",
              flush=True)
        return self.get_parameters({}), self.train_size, {"mu": mu}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = evaluate_model(self.model, self.test_loader, self.criterion)
        self.early_stop(loss)
        print(f"🔍 Round {self.round_num} – Loss={loss:.4f}, Accuracy={acc:.4f}"
              + ("  [early stop triggered]" if self.early_stop.early_stop else ""),
              flush=True)
        return float(loss), self.test_size, {"accuracy": float(acc)}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python proxclient.py <client_name>  (e.g. client1)", flush=True)
        sys.exit(1)

    client_name = sys.argv[1]
    print(f"🚀 Starting FedProx + ChaCha20-Poly1305 client for '{client_name}'...",
          flush=True)
    print(f"💻 Device: {device}", flush=True)

    try:
        railway_server = os.environ.get(
        'RAILWAY_SERVER_ADDRESS',
        'your-railway-service.up.railway.app:8081'
    )
        fl.client.start_numpy_client(
        server_address=railway_server,
        client=FedProxClient(client_name),
    )
        
    except Exception as e:
        print(f"\n❌ FL client failed: {e}", flush=True)
        print("   → Is the FL server running on localhost:8081?", flush=True)
        raise