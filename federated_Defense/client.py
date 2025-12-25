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

        # Store CLEAN data (never modified)
        self.X_clean = np.array(X)
        self.y_clean = np.array(y)
        
        # Working copies (will be poisoned if malicious)
        self.X = np.array(X)
        self.y = np.array(y)

        # Create attack instance if malicious
        if self.is_malicious:
            self.attack_instance = AttackFactory.create_attack(attack_config)
            try:
                self.poison_local_data()
                print(f"[INFO] Client {client_id} initialized as MALICIOUS with '{attack_config.attack_type}' attack")
            except Exception as e:
                print(f"[WARN] Client {client_id} initial poisoning failed: {e}")
        else:
            print(f"[INFO] Client {client_id} initialized as HONEST")

        self.local_accuracy_history = []
        self.received_updates = []

    # -------------------------------------------------------------------------
    # NEW METHOD: Re-poison data (called each round)
    # -------------------------------------------------------------------------
    def poison_local_data(self):
        """
        Apply attack to local training data.
        This method should be called at the START of each training round.
        """
        if not self.is_malicious or self.attack_instance is None:
            return
        
        # Reset to clean data first
        self.X = self.X_clean.copy()
        self.y = self.y_clean.copy()
        
        # Apply poisoning
        try:
            self.X, self.y, self.poison_mask = self.attack_instance.poison_dataset(
                self.X, self.y, self.client_id, self.model
            )
            num_poisoned = np.sum(self.poison_mask)
            print(f"[POISON] Client {self.client_id}: Poisoned {num_poisoned}/{len(self.y)} samples")
        except Exception as e:
            print(f"[ERROR] Client {self.client_id}: Poisoning failed - {e}")
            # Fallback to clean data
            self.X = self.X_clean.copy()
            self.y = self.y_clean.copy()
            self.poison_mask = np.zeros(len(self.y), dtype=bool)

    # -------------------------------------------------------------------------
    # Local training (uses self.X and self.y which are poisoned for malicious clients)
    # -------------------------------------------------------------------------
    def local_training(self, epochs=10, lr=0.01, batch_size=32):
        """Perform local training and return model parameters"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.train()

        # FIXED: Malicious clients train moderately more to strengthen poison signal
        # Reduced amplification for balanced attack-defense dynamics
        if self.is_malicious:
            epochs = int(epochs * 1.5)  # REDUCED: 1.5x instead of 2.5x
            lr = lr * 1.2  # REDUCED: 1.2x instead of 2.0x
            print(f"  [MALICIOUS] Client {self.client_id}: Training with {epochs} epochs, lr={lr:.3f} (moderate boost)")

        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        # Use current X and y (poisoned if malicious)
        dataset = TensorDataset(torch.FloatTensor(self.X), torch.LongTensor(self.y))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        epoch_losses = []

        for ep in range(epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_losses.append(epoch_loss / len(dataloader))

        # Evaluate local accuracy
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(self.X).to(device)
            y_tensor = torch.LongTensor(self.y).to(device)
            outputs = self.model(X_tensor)
            preds = torch.argmax(outputs, dim=1)
            local_accuracy = (preds == y_tensor).float().mean().item()
            self.local_accuracy_history.append(local_accuracy)

        # CRITICAL: Apply model poisoning attacks AFTER training (for model poisoning attacks)
        if self.is_malicious and self.attack_instance is not None:
            if hasattr(self.attack_instance, 'poison_model'):
                try:
                    print(f"  [MODEL POISON] Client {self.client_id}: Applying model-level poisoning")
                    self.model = self.attack_instance.poison_model(self.model)
                except Exception as e:
                    print(f"  [ERROR] Client {self.client_id}: Model poisoning failed - {e}")
            elif hasattr(self.attack_instance, 'manipulate_update'):
                try:
                    print(f"  [UPDATE MANIPULATION] Client {self.client_id}: Manipulating model update")
                    self.model = self.attack_instance.manipulate_update(self.model)
                except Exception as e:
                    print(f"  [ERROR] Client {self.client_id}: Update manipulation failed - {e}")

        params = {name: param.data.cpu().clone() for name, param in self.model.named_parameters()}
        return params, np.mean(epoch_losses), local_accuracy

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


