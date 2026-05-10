import os
import time
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import (
    Parameters, Scalar, FitRes,
    parameters_to_ndarrays, ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import SimpleClientManager

from proxcrypto import ChaCha20Cipher, pack_parameters, unpack_parameters, is_packed


FEDPROX_MU: float = float(os.environ.get("FEDPROX_MU", "0.01"))

MODEL_DIR  = os.path.join(os.getcwd(), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "resnet18_fedprox_model_best.pth")

# Server no longer needs FL_DATA_DIR — clients evaluate on their own local data.
# The only server-side I/O is checkpoint saving (MODEL_PATH above).

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REGISTRATION_WINDOW_SECS = 60
MIN_CLIENTS              = 1

PER_CLIENT_TIMEOUT_SECS  = 300


def create_global_model(num_classes: int = 8):
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ── No load_server_test_data() — the server holds NO client data. ──────────
# All evaluation happens locally on each client (FedProxClient.evaluate()),
# and per-round accuracy is aggregated by Flower's built-in metric aggregation.
# This is true federated learning: raw data never leaves the client.
# ───────────────────────────────────────────────────────────────────────────


model = create_global_model()

if os.path.exists(MODEL_PATH):
    print("Loading existing global model from checkpoint...")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(" Loaded from checkpoint (model_state_dict key)")
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
        print("Loaded from state_dict")
    else:
        print(" Unknown checkpoint format – starting fresh.")
else:
    print(" No checkpoint found – starting from ImageNet weights.")

model = model.to(device)

cipher = ChaCha20Cipher()


class TimedClientManager(SimpleClientManager):
    def __init__(self, registration_window: float, min_clients: int):
        super().__init__()
        self.registration_window = registration_window
        self.min_clients         = min_clients
        self._first_round_done   = False
        self._window_start       = None

    def sample(self, num_clients, min_num_clients=None, criterion=None):
        if not self._first_round_done:
            if self._window_start is None:
                self._window_start = time.time()
                print(f"\n⏳ Registration window open – waiting {self.registration_window}s "
                      f"for clients to connect (minimum required: {self.min_clients})")

            elapsed   = time.time() - self._window_start
            remaining = self.registration_window - elapsed

            while remaining > 0:
                connected = len(self.clients)
                print(f"   ⏱  {remaining:5.1f}s remaining | {connected} client(s) connected")
                time.sleep(min(5.0, remaining))
                elapsed   = time.time() - self._window_start
                remaining = self.registration_window - elapsed

            connected = len(self.clients)
            print(f"\n Registration window closed – {connected} client(s) connected")

            if connected < self.min_clients:
                raise RuntimeError(
                    f" Only {connected} client(s) connected after the registration window. "
                    f"Need at least {self.min_clients}. Aborting."
                )
            self._first_round_done = True

        available = len(self.clients)
        n         = min(num_clients, available)
        return super().sample(n, min_num_clients=min(self.min_clients, available), criterion=criterion)


def fit_config(server_round: int) -> Dict[str, Scalar]:
    return {
        "mu":           FEDPROX_MU,
        "server_round": server_round,
    }


def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    """
    Aggregate client-reported evaluation metrics by weighted average.

    Flower calls this after collecting evaluate() results from all clients.
    Each client returns (num_examples, {"accuracy": float, ...}).
    We compute a sample-count-weighted mean so larger clients contribute more.
    """
    total_examples = sum(n for n, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated: Dict[str, float] = {}
    for num_examples, client_metrics in metrics:
        weight = num_examples / total_examples
        for key, value in client_metrics.items():
            aggregated[key] = aggregated.get(key, 0.0) + weight * float(value)

    # Pretty-print so progress is visible in server logs
    metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in aggregated.items())
    print(f"📊 Aggregated client metrics → {metrics_str}")
    return aggregated


class ChaChaFedProx(fl.server.strategy.FedAvg):

    def __init__(self, shared_cipher: ChaCha20Cipher, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cipher = shared_cipher

    def _update_min_clients(self, n: int):
        """Keep Flower's internal thresholds in sync with actual client count."""
        self.min_fit_clients       = max(MIN_CLIENTS, n)
        self.min_evaluate_clients  = max(MIN_CLIENTS, n)
        self.min_available_clients = max(MIN_CLIENTS, n)

    def configure_fit(self, server_round, parameters, client_manager):
        client_instructions = super().configure_fit(
            server_round, parameters, client_manager
        )
        if not client_instructions:
            return client_instructions

        num_clients = len(client_instructions)
        print(f"\n Assigning GPU slots to {num_clients} client(s) "
              f"({PER_CLIENT_TIMEOUT_SECS}s per slot)  |  μ={FEDPROX_MU}:")

        updated = []
        for slot_idx, (client, fit_ins) in enumerate(client_instructions):
            new_config = dict(fit_ins.config)
            new_config["gpu_slot"]      = slot_idx
            new_config["per_slot_secs"] = PER_CLIENT_TIMEOUT_SECS
            new_fit_ins = fl.common.FitIns(
                parameters=fit_ins.parameters,
                config=new_config,
            )
            updated.append((client, new_fit_ins))
            print(f"   Slot {slot_idx} → starts at T+{slot_idx * PER_CLIENT_TIMEOUT_SECS}s")

        return updated

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

        self._update_min_clients(len(results))

        print(f"\n{'='*60}")
        print(f"🔄 Round {server_round} – FedProx Aggregation  (μ={FEDPROX_MU})")
        print(f"{'='*60}")
        print(f"📥 Received updates from {len(results)} client(s)")

        print("🔓 Decrypting & decompressing client parameters...")
        clean_results = []
        for client, fit_res in results:
            params_list = parameters_to_ndarrays(fit_res.parameters)
            if is_packed(params_list):
                try:
                    unpacked = unpack_parameters(params_list[0], self.cipher)
                    new_fit_res = FitRes(
                        status=fit_res.status,
                        parameters=ndarrays_to_parameters(unpacked),
                        num_examples=fit_res.num_examples,
                        metrics=fit_res.metrics,
                    )
                    clean_results.append((client, new_fit_res))
                except ValueError as e:
                    print(f" MAC check FAILED — discarding client update: {e}")
                except Exception as e:
                    print(f"  Unpack error — discarding client update: {e}")
            else:
                clean_results.append((client, fit_res))

        if not clean_results:
            print(" No valid updates after unpacking – skipping aggregation.")
            return None, {}

        print(f" Unpacked {len(clean_results)} client update(s)")

        aggregatable = [(c, r) for c, r in clean_results if r.num_examples > 0]
        if not aggregatable:
            print("⚠  All clients early-stopped this round – skipping aggregation.")
            return None, {}
        if len(aggregatable) < len(clean_results):
            skipped = len(clean_results) - len(aggregatable)
            print(f"⚠  {skipped} client(s) early-stopped and excluded from aggregation.")

        print("🔄 Aggregating (weighted FedAvg)...")
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, aggregatable, failures
        )
        if aggregated_parameters is None:
            return None, metrics
        print(" Aggregation complete")

        print("🔒 Compressing & encrypting aggregated parameters...")
        agg_arrays       = parameters_to_ndarrays(aggregated_parameters)
        packed           = pack_parameters(agg_arrays, self.cipher)
        enc_params       = ndarrays_to_parameters([packed])

        original_bytes   = sum(a.nbytes for a in agg_arrays)
        compressed_bytes = packed.nbytes
        ratio = (1 - compressed_bytes / original_bytes) * 100
        print(f" Packed: {original_bytes/1e6:.2f} MB → {compressed_bytes/1e6:.2f} MB "
              f"({ratio:.1f}% reduction)")
        print(f"{'='*60}\n")

        _save_model(agg_arrays)
        return enc_params, metrics


def _save_model(agg_arrays: list):
    state_dict = {
        k: torch.tensor(v)
        for k, v in zip(model.state_dict().keys(), agg_arrays)
    }
    model.load_state_dict(state_dict, strict=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"💾 Checkpoint saved: {MODEL_PATH}")


client_manager = TimedClientManager(
    registration_window=REGISTRATION_WINDOW_SECS,
    min_clients=MIN_CLIENTS,
)

strategy = ChaChaFedProx(
    shared_cipher=cipher,
    on_fit_config_fn=fit_config,
    # evaluate_fn=None  ← default; no server-side evaluation.
    # Clients call evaluate() locally and report metrics back via Flower's
    # built-in client evaluation protocol (configure_evaluate / aggregate_evaluate).
    evaluate_fn=None,
    evaluate_metrics_aggregation_fn=weighted_average,  # aggregates client accuracy reports
    min_available_clients=MIN_CLIENTS,
    min_fit_clients=MIN_CLIENTS,
    min_evaluate_clients=MIN_CLIENTS,
)

if __name__ == "__main__":
    print(" Starting FedProx + ChaCha20-Poly1305 + zlib Flower Server...")
    print(f" Authenticated encryption (ChaCha20-Poly1305) + zlib compression enabled")
    print(f" FedProx μ = {FEDPROX_MU}  (override: $env:FEDPROX_MU = '0.1')")
    print(f" Device: {device}")
    print(f" Registration window:      {REGISTRATION_WINDOW_SECS}s")
    print(f" Minimum clients required: {MIN_CLIENTS}")
    print(f" GPU slot duration:        {PER_CLIENT_TIMEOUT_SECS}s per client")
    print(f" Evaluation: CLIENT-LOCAL ONLY (no server test data)")

    fl.server.start_server(
        server_address="localhost:8081",
        config=fl.server.ServerConfig(
            num_rounds=10,
            round_timeout=PER_CLIENT_TIMEOUT_SECS * 10 + 120,
        ),
        strategy=strategy,
        client_manager=client_manager,
    )

    print("\n Federated training complete.")