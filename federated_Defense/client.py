# =============================================================================
# FIXED client.py - Add poison_local_data() method
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from attacks.factory import AttackFactory
import random


class DecentralizedClient:
    """Decentralized FL client with peer-to-peer communication and attack simulation"""

    def __init__(self, client_id, X, y, model, attack_config=None, neighbor_ids=None):
        """
        Args:
            client_id (int): Unique client ID
            X, y: Local dataset
            model: Local model
            attack_config: Optional malicious attack configuration
            neighbor_ids (list[int]): List of neighboring clients for gossip exchange
        """
        self.client_id = client_id
        self.model = model
        self.attack_config = attack_config
        self.is_malicious = attack_config is not None
        self.attack_instance = None
        self.poison_mask = np.zeros(len(y), dtype=bool)
        self.neighbor_ids = neighbor_ids or []  # P2P connections

        # Store CLEAN data (never modified) — zero-copy views of the original arrays
        self.X_clean = np.asarray(X)
        self.y_clean = np.asarray(y)

        # Working copies (will be poisoned if malicious)
        self.X = self.X_clean.copy()
        self.y = self.y_clean.copy()

        # GPU tensor cache — avoids repeated CPU→GPU copies every round
        self._gpu_X = None
        self._gpu_y = None
        self._gpu_dataset = None
        self._gpu_data_id = None  # hash to detect data changes

        # Training cache — avoids re-creating optimizer/criterion/device every round
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._criterion = nn.CrossEntropyLoss()
        self._cached_lr = None
        self._optimizer = None

        # Create attack instance if malicious
        if self.is_malicious:
            self.attack_instance = AttackFactory.create_attack(attack_config)
            try:
                self.poison_local_data()
                # Individual client initialization messages removed for cleaner output
                # Summary will be printed by coordinator
            except Exception as e:
                print(f"[WARN] Client {client_id} initial poisoning failed: {e}")

        self.local_accuracy_history = []
        self.received_updates = []

    # -------------------------------------------------------------------------
    # Attack setup
    # -------------------------------------------------------------------------
    def poison_local_data(self):
        """
        Prepare attack state.

        Called once at __init__ for ALL attacks (poisons data or snapshots model).
        Called again each round by coordinator ONLY for POISONS_MODEL attacks
        (Byzantine / model-poisoning) to re-snapshot _initial_params from
        the latest global model.

        Data-only attacks (SLF, DLF, triggers) poison data once at init —
        self.X/self.y stay poisoned across rounds, no per-round call needed.
        """
        if not self.is_malicious or self.attack_instance is None:
            return

        attack = self.attack_instance

        # Model-only attacks (Byzantine): snapshot current model params only.
        # Data is never modified by these attacks.
        if not attack.POISONS_DATA and attack.POISONS_MODEL:
            try:
                attack.poison_dataset(self.X, self.y, self.client_id, self.model)
            except Exception:
                pass
            return

        # Data-poisoning attacks (SLF, DLF, triggers, hybrid):
        # Poison data from clean copy. Called once at init.
        self.X = self.X_clean.copy()
        self.y = self.y_clean.copy()

        try:
            self.X, self.y, self.poison_mask = attack.poison_dataset(
                self.X, self.y, self.client_id, self.model
            )
            num_poisoned = np.sum(self.poison_mask)
            if attack.POISONS_DATA and num_poisoned > 0:
                print(f"[POISON] Client {self.client_id}: Poisoned {num_poisoned}/{len(self.y)} samples")
        except Exception as e:
            print(f"[ERROR] Client {self.client_id}: Poisoning failed - {e}")
            self.X = self.X_clean.copy()
            self.y = self.y_clean.copy()
            self.poison_mask = np.zeros(len(self.y), dtype=bool)

        # Invalidate GPU cache since data changed
        self._gpu_data_id = None

    # -------------------------------------------------------------------------
    # Snapshot / Reset — enables reuse across ablation runs without deep copy
    # -------------------------------------------------------------------------
    def snapshot_state(self):
        """Save current state so the client can be cheaply reset for a new run.
        Call ONCE after create_attack_scenario() has finished setting up poisoned data."""
        self._snapshot_model_state = {
            k: v.cpu().clone() for k, v in self.model.state_dict().items()
        }
        self._snapshot_X = self.X.copy()
        self._snapshot_y = self.y.copy()
        self._snapshot_poison_mask = self.poison_mask.copy()
        # Save Byzantine attack _initial_params if present
        if self.attack_instance is not None and hasattr(self.attack_instance, '_initial_params'):
            ip = self.attack_instance._initial_params
            if ip is not None:
                self._snapshot_initial_params = {
                    k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
                    for k, v in ip.items()
                }
            else:
                self._snapshot_initial_params = None
        else:
            self._snapshot_initial_params = None

    def reset_for_new_run(self):
        """Restore client to its post-poisoning initial state (no deep copy needed).
        Call at the start of each ablation run instead of copy.deepcopy()."""
        # Restore model weights
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self._snapshot_model_state:
                    param.data.copy_(self._snapshot_model_state[name])
            # Also restore non-parameter buffers (e.g. batch norm running stats)
            for name, buf in self.model.named_buffers():
                if name in self._snapshot_model_state:
                    buf.data.copy_(self._snapshot_model_state[name])

        # Restore data
        self.X = self._snapshot_X.copy()
        self.y = self._snapshot_y.copy()
        self.poison_mask = self._snapshot_poison_mask.copy()

        # Restore attack _initial_params
        if self.attack_instance is not None and hasattr(self.attack_instance, '_initial_params'):
            if self._snapshot_initial_params is not None:
                self.attack_instance._initial_params = {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in self._snapshot_initial_params.items()
                }
            else:
                self.attack_instance._initial_params = None

        # Clear per-run state
        self.local_accuracy_history = []
        self.received_updates = []
        self.neighbor_ids = []

        # Invalidate optimizer cache (model weights changed)
        self._optimizer = None
        self._cached_lr = None
        # Invalidate tensor cache (data may have changed)
        self._gpu_data_id = None

        # Invalidate GPU cache (data was restored from snapshot)
        self._gpu_data_id = None

    # -------------------------------------------------------------------------
    # Local training (uses self.X and self.y which are poisoned for malicious clients)
    # -------------------------------------------------------------------------
    def local_training(self, epochs=5, lr=0.01, batch_size=None):
        """Perform local training and return model parameters.

        Optimisations:
        - Device, criterion, optimizer cached across rounds (not recreated).
        - GPU/CPU tensors cached (data rarely changes).
        - Redundant local-accuracy forward pass removed.
        """
        device = self._device
        self.model.to(device)
        self.model.train()

        # Cache optimizer — only recreate if lr changes
        if self._optimizer is None or self._cached_lr != lr:
            self._optimizer = optim.SGD(self.model.parameters(), lr=lr)
            self._cached_lr = lr

        criterion = self._criterion

        # --- Tensor cache — avoids repeated CPU→GPU copies every round ---
        data_id = (id(self.X), id(self.y), len(self.X))
        if self._gpu_data_id != data_id:
            self._gpu_X = torch.FloatTensor(self.X).to(device)
            self._gpu_y = torch.LongTensor(self.y).to(device)
            self._gpu_dataset = TensorDataset(self._gpu_X, self._gpu_y)
            self._gpu_data_id = data_id

        # When batch_size is None, use full dataset as one batch
        effective_bs = batch_size if batch_size is not None else len(self._gpu_dataset)

        # Full-batch fast path: skip DataLoader overhead entirely
        if effective_bs >= len(self._gpu_dataset):
            epoch_losses = []
            for ep in range(epochs):
                self._optimizer.zero_grad()
                outputs = self.model(self._gpu_X)
                loss = criterion(outputs, self._gpu_y)
                loss.backward()
                self._optimizer.step()
                epoch_losses.append(loss.item())
        else:
            dataloader = DataLoader(self._gpu_dataset, batch_size=effective_bs, shuffle=True)
            epoch_losses = []
            for ep in range(epochs):
                epoch_loss = 0.0
                for X_batch, y_batch in dataloader:
                    self._optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    self._optimizer.step()
                    epoch_loss += loss.item()
                epoch_losses.append(epoch_loss / len(dataloader))

        # NOTE: Local accuracy evaluation REMOVED for speed.  The per-client
        # accuracy was only used for debug printing and is not part of the FL
        # algorithm.  Returning 0.0 keeps the return signature unchanged.
        avg_loss = np.mean(epoch_losses)

        # Apply model-level poisoning AFTER training — only for attacks that need it
        if self.is_malicious and self.attack_instance is not None and self.attack_instance.POISONS_MODEL:
            if hasattr(self.attack_instance, 'manipulate_update'):
                try:
                    self.model = self.attack_instance.manipulate_update(self.model)
                except Exception as e:
                    print(f"  [ERROR] Client {self.client_id}: Update manipulation failed - {e}")
            elif hasattr(self.attack_instance, 'poison_model'):
                try:
                    self.model = self.attack_instance.poison_model(self.model)
                except Exception as e:
                    print(f"  [ERROR] Client {self.client_id}: Model poisoning failed - {e}")

        params = {name: param.data.cpu().clone() for name, param in self.model.named_parameters()}
        return params, avg_loss, 0.0

    # -------------------------------------------------------------------------
    # Peer update exchange and aggregation (UNCHANGED)
    # -------------------------------------------------------------------------
    def receive_peer_updates(self, peer_updates):
        """Receive and store updates from peers"""
        self.received_updates = peer_updates

    def select_peers(self, all_clients, num_peers=2):
        """Randomly select a subset of peers for gossip communication"""
        available_peers = [c for c in all_clients if c.client_id != self.client_id]
        selected = random.sample(available_peers, min(num_peers, len(available_peers)))
        self.neighbor_ids = [p.client_id for p in selected]
        return selected

    def send_model_to_peers(self):
        """Prepare local parameters for sending to peers"""
        return {name: p.data.cpu().clone() for name, p in self.model.named_parameters()}

    def aggregate_with_peers(self, aggregation_method="mean", weights=None):
        """Aggregate local model with peer updates"""
        if not self.received_updates:
            return  # No peer updates received

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Get current parameters
        current_params = {n: p.data.cpu().clone() for n, p in self.model.named_parameters()}
        all_params = [current_params] + self.received_updates

        # Apply weights if provided (e.g., from topology matrix)
        if weights is None:
            weights = [1.0 / len(all_params)] * len(all_params)
        else:
            total = sum(weights)
            weights = [w / total for w in weights]

        aggregated_params = {}
        for name in current_params.keys():
            stacked = torch.stack([params[name] * weights[i] for i, params in enumerate(all_params)])
            if aggregation_method == "mean":
                aggregated_params[name] = stacked.sum(dim=0)
            elif aggregation_method == "median":
                aggregated_params[name] = torch.median(torch.stack([params[name] for params in all_params]), dim=0).values

        # Update model parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data.copy_(aggregated_params[name])

    # -------------------------------------------------------------------------
    # Attack evaluation (UNCHANGED)
    # -------------------------------------------------------------------------
    def evaluate_attack_success(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Compute attack success rate (ASR)"""
        if not self.is_malicious or self.attack_instance is None:
            return 0.0
        try:
            return self.attack_instance.evaluate_attack_success(self.model, X_test, y_test)
        except Exception as e:
            print(f"[WARN] Client {self.client_id} ASR evaluation failed: {e}")
            return 0.0


