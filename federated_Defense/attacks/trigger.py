from attacks.base import AttackConfig, BaseAttack
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from typing import Optional, Tuple, Dict


# ==================================================================
# TRIGGER GENERATION UTILITIES (ENHANCED)
# ==================================================================

class TriggerGenerator:
    """Utility class for generating different types of triggers (robust to small images)."""

    @staticmethod
    def pixel_pattern_trigger(image_shape: Tuple[int, ...], pattern_size: int = 2,
                              position: str = "bottom_right", intensity: Optional[float] = 1.0) -> np.ndarray:
        """Generate pixel pattern trigger (works for (C,H,W) and (H,W))."""
        if intensity is None:
            intensity = 1.0
        trigger = np.zeros(image_shape, dtype=np.float32)

        # Normalize pattern_size so it doesn't exceed image dims
        if len(image_shape) == 3:
            _, h, w = image_shape
        else:
            h, w = image_shape

        ps = max(1, min(pattern_size, min(h, w)))

        if len(image_shape) == 3:
            _, h, w = image_shape
            if position == "bottom_right":
                trigger[:, h-ps:h, w-ps:w] = intensity
            elif position == "top_left":
                trigger[:, :ps, :ps] = intensity
            elif position == "center":
                ch, cw = h // 2, w // 2
                h0 = max(0, ch - ps//2)
                w0 = max(0, cw - ps//2)
                trigger[:, h0:h0+ps, w0:w0+ps] = intensity
        else:
            if position == "bottom_right":
                trigger[h-ps:h, w-ps:w] = intensity
            elif position == "top_left":
                trigger[:ps, :ps] = intensity
            elif position == "center":
                ch, cw = h // 2, w // 2
                h0 = max(0, ch - ps//2)
                w0 = max(0, cw - ps//2)
                trigger[h0:h0+ps, w0:w0+ps] = intensity

        return trigger

    @staticmethod
    def checkerboard_trigger(image_shape: Tuple[int, ...], square_size: int = 1,
                              position: str = "bottom_right") -> np.ndarray:
        """Generate checkerboard pattern trigger; safe for small images."""
        trigger = np.zeros(image_shape, dtype=np.float32)

        # Determine H,W
        if len(image_shape) == 3:
            _, h, w = image_shape
        else:
            h, w = image_shape

        # minimal pattern size
        square_size = max(1, int(square_size))
        pattern_size = min(h, w, square_size * 4)
        if pattern_size <= 0:
            return trigger

        # choose start positions safely
        if position == "bottom_right":
            start_h, start_w = max(0, h - pattern_size), max(0, w - pattern_size)
        elif position == "top_left":
            start_h, start_w = 0, 0
        else:
            start_h, start_w = max(0, (h - pattern_size)//2), max(0, (w - pattern_size)//2)

        for i in range(pattern_size):
            for j in range(pattern_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    pos_h, pos_w = start_h + i, start_w + j
                    if 0 <= pos_h < h and 0 <= pos_w < w:
                        if len(image_shape) == 3:
                            trigger[:, pos_h, pos_w] = 1.0
                        else:
                            trigger[pos_h, pos_w] = 1.0

        return trigger

    @staticmethod
    def random_patch_trigger(image_shape: Tuple[int, ...], patch_size: Tuple[int,int]=(2,2),
                             intensity: Optional[float] = 1.0, seed: Optional[int] = None) -> np.ndarray:
        """Generate a small random patch trigger (robust: patch clipped to image)."""
        if seed is not None:
            rndstate = np.random.get_state()
            np.random.seed(seed)

        if intensity is None:
            intensity = 1.0

        trigger = np.zeros(image_shape, dtype=np.float32)

        # Get H, W and clamp patch size
        if len(image_shape) == 3:
            _, h, w = image_shape
        else:
            h, w = image_shape

        ph, pw = max(1, int(patch_size[0])), max(1, int(patch_size[1]))
        ph = min(ph, h)
        pw = min(pw, w)

        # Ensure randint high is at least 1
        h_high = max(1, h - ph + 1)
        w_high = max(1, w - pw + 1)
        start_h = np.random.randint(0, h_high)
        start_w = np.random.randint(0, w_high)

        pattern = np.random.uniform(0, float(intensity), (ph, pw)).astype(np.float32)
        if len(image_shape) == 3:
            trigger[:, start_h:start_h+ph, start_w:start_w+pw] = pattern
        else:
            trigger[start_h:start_h+ph, start_w:start_w+pw] = pattern

        if seed is not None:
            np.random.set_state(rndstate)

        return trigger

    @staticmethod
    def combine_triggers(image_shape: Tuple[int, ...], intensity: Optional[float] = 0.3,  # CHANGED: increased from 0.05
                         seed: Optional[int] = None) -> np.ndarray:
        """Combine pixel + checkerboard + random patch into a single base trigger, robust defaults."""
        if intensity is None:
            intensity = 0.3  # CHANGED: increased default intensity
        if seed is not None:
            rndstate = np.random.get_state()
            np.random.seed(seed)

        # CHANGED: Increased pattern sizes for stronger triggers
        t_pixel = TriggerGenerator.pixel_pattern_trigger(image_shape, pattern_size=4,  # CHANGED: from 2
                                                         position="bottom_right", intensity=float(intensity))
        t_check = TriggerGenerator.checkerboard_trigger(image_shape, square_size=2,  # CHANGED: from 1
                                                        position="bottom_right")
        t_rand = TriggerGenerator.random_patch_trigger(image_shape, patch_size=(4,4),  # CHANGED: from (2,2)
                                                       intensity=float(intensity))

        # CHANGED: Stronger combination with higher weights
        base = t_pixel * 1.5 + (t_check * float(intensity) * 2.0) + (t_rand * float(intensity) * 1.5)
        base = np.clip(base, -1.0, 1.0)

        if seed is not None:
            np.random.set_state(rndstate)

        return base

    # ==============================
    # ALIGN TRIGGER WITH GRADIENT
    # ==============================
    @staticmethod
    def align_trigger_with_gradient(model: torch.nn.Module, x: np.ndarray, y_target: int,
                                    base_trigger: Optional[np.ndarray] = None,
                                    optimization_steps: int = 50, lr: float = 0.15,  # CHANGED: from 15 steps, 0.05 lr
                                    epsilon: Optional[float] = None,
                                    loss_fn: Optional[nn.Module] = None) -> np.ndarray:
        """
        Optimize a (small) trigger so that, when added to x, the model predicts y_target.
        Returns the optimized trigger numpy array (same shape as x).
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        if epsilon is None:
            epsilon = 0.3  # CHANGED: increased from 0.1

        # x can be (C,H,W) or (H,W) - we add batch dim
        x_tensor = torch.FloatTensor(x).unsqueeze(0).to(device)  # shape (1,C,H,W) or (1,H,W)
        y_tensor = torch.LongTensor([int(y_target)]).to(device)
        loss_fn = loss_fn or nn.CrossEntropyLoss()

        if base_trigger is not None:
            trigger_t = torch.FloatTensor(base_trigger).unsqueeze(0).to(device)
        else:
            trigger_t = torch.zeros_like(x_tensor).to(device)

        trigger_t = trigger_t.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([trigger_t], lr=lr)

        for step in range(int(optimization_steps)):
            optimizer.zero_grad()
            x_tr = torch.clamp(x_tensor + trigger_t, 0.0, 1.0)
            out = model(x_tr)
            # Ensure target is scalar index (CrossEntropy expects scalar class indices)
            if y_tensor.dim() > 1:
                y_tensor = torch.argmax(y_tensor, dim=1)
            loss = loss_fn(out, y_tensor)
            loss.backward()
            optimizer.step()

            # Project trigger to epsilon-ball (L2) - CHANGED: stronger projection
            with torch.no_grad():
                trigger_t.data = trigger_t.data * 1.2  # CHANGED: boost trigger strength
                flat = trigger_t.view(trigger_t.size(0), -1)
                norms = flat.norm(p=2, dim=1, keepdim=True)
                scale = torch.clamp(epsilon / (norms + 1e-12), max=2.0)  # CHANGED: increased max scale from 1.0
                trigger_t.data = (flat * scale).view_as(trigger_t)

        trigger_np = trigger_t.detach().squeeze(0).cpu().numpy()
        return trigger_np


# =============================================================================
# TRIGGER-BASED ATTACKS (CENTRALIZED, COORDINATED, RANDOM, MODEL-DEPENDENT)
# =============================================================================

class CentralizedTriggerAttack(BaseAttack):
    """Centralized Trigger-Based Backdoor Attack"""

    def __init__(self, config: AttackConfig):
        super().__init__(config)
        self.trigger_pattern = None
        self.target_class = config.target_class

    def generate_trigger(self, data_shape: Tuple[int, ...]) -> np.ndarray:
        """Generate fixed combined base trigger"""
        # CHANGED: Use much stronger intensity
        base = TriggerGenerator.combine_triggers(data_shape,
                                                 intensity=0.4,  # CHANGED: override config for stronger attack
                                                 seed=None)
        return base

    def apply_trigger(self, x: np.ndarray, trigger: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply trigger to input sample"""
        if trigger is None:
            if self.trigger_pattern is None:
                self.trigger_pattern = self.generate_trigger(x.shape)
            trigger = self.trigger_pattern
        triggered_x = x + trigger
        return np.clip(triggered_x, 0.0, 1.0)

    def poison_dataset(self, X: np.ndarray, y: np.ndarray,
                       client_id: int = 0, model: Optional[nn.Module] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Poison dataset with triggers"""
        X_poisoned = X.copy()
        y_poisoned = y.copy()

        n_samples = len(X)
        n_poison = int(self.config.poisoning_rate * n_samples)
        if n_poison <= 0:
            poison_indices = []
        else:
            poison_indices = np.random.choice(n_samples, n_poison, replace=False)

        for idx in poison_indices:
            if model is not None:
                base_trigger = TriggerGenerator.combine_triggers(
                    X[idx].shape,
                    intensity=0.4,  # CHANGED: stronger intensity
                    seed=client_id + idx
                )
                optimized = TriggerGenerator.align_trigger_with_gradient(
                    model, X[idx], self.target_class,
                    base_trigger=base_trigger,
                    optimization_steps=50,  # CHANGED: from 15
                    lr=0.15,  # CHANGED: from 0.05
                    epsilon=0.4  # CHANGED: from 0.1
                )
                X_poisoned[idx] = np.clip(X[idx] + optimized, 0.0, 1.0)
            else:
                X_poisoned[idx] = self.apply_trigger(X[idx])
            y_poisoned[idx] = self.target_class

        poison_mask = np.zeros(n_samples, dtype=bool)
        if len(poison_indices) > 0:
            poison_mask[poison_indices] = True

        print(f"  [Centralized] Client {client_id}: Poisoned {n_poison}/{n_samples} samples")
        return X_poisoned, y_poisoned, poison_mask

    def evaluate_attack_success(self, model: nn.Module, X_test: np.ndarray,
                                y_test: np.ndarray) -> float:
        """Evaluate backdoor success rate"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        base_trigger = self.trigger_pattern if self.trigger_pattern is not None else self.generate_trigger(X_test[0].shape)
        X_triggered = np.array([self.apply_trigger(x, base_trigger) for x in X_test])

        X_tensor = torch.FloatTensor(X_triggered).to(device)
        with torch.no_grad():
            outputs = model(X_tensor)
            y_pred = torch.argmax(outputs, dim=1)
        # compute_asr expects (y_true, y_pred); keep types consistent
        return self.compute_asr(y_true=torch.tensor(y_test), y_pred=y_pred)

    def attack_summary(self) -> Dict:
        base_summary = super().attack_summary()
        base_summary.update({
            "target_class": self.target_class,
            "trigger_type": "centralized_combined"
        })
        return base_summary


class CoordinatedTriggerAttack(BaseAttack):
    """Coordinated Trigger-Based Backdoor Attack"""

    def __init__(self, config: AttackConfig):
        super().__init__(config)
        self.client_triggers: Dict[int, np.ndarray] = {}
        self.target_class = config.target_class

    def generate_client_trigger(self, client_id: int, data_shape: Tuple[int, ...]) -> np.ndarray:
        """Generate per-client combined trigger"""
        # CHANGED: stronger intensity
        base = TriggerGenerator.combine_triggers(data_shape,
                                                 intensity=0.35,  # CHANGED: from config value
                                                 seed=client_id + 1000)
        return base

    def get_client_trigger(self, client_id: int, data_shape: Tuple[int, ...],
                          model: Optional[nn.Module] = None) -> np.ndarray:
        """Get or create optimized trigger for specific client"""
        if client_id not in self.client_triggers:
            base = self.generate_client_trigger(client_id, data_shape)
            if model is not None:
                representative_dummy = np.zeros(data_shape, dtype=np.float32)
                optimized = TriggerGenerator.align_trigger_with_gradient(
                    model, representative_dummy, self.target_class,
                    base_trigger=base,
                    optimization_steps=45,  # CHANGED: from 12
                    lr=0.12,  # CHANGED: from 0.05
                    epsilon=0.35  # CHANGED: from 0.1
                )
                self.client_triggers[client_id] = optimized
            else:
                self.client_triggers[client_id] = base
        return self.client_triggers[client_id]

    def apply_trigger(self, x: np.ndarray, trigger: np.ndarray) -> np.ndarray:
        triggered_x = x + trigger
        return np.clip(triggered_x, 0.0, 1.0)

    def poison_dataset(self, X: np.ndarray, y: np.ndarray,
                       client_id: int = 0, model: Optional[nn.Module] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_poisoned = X.copy()
        y_poisoned = y.copy()

        data_shape = X[0].shape if len(X) > 0 else X.shape[1:]
        client_trigger = self.get_client_trigger(client_id, data_shape, model=model)

        n_samples = len(X)
        n_poison = int(self.config.poisoning_rate * n_samples)
        if n_poison <= 0:
            poison_indices = []
        else:
            poison_indices = np.random.choice(n_samples, n_poison, replace=False)

        for idx in poison_indices:
            X_poisoned[idx] = self.apply_trigger(X[idx], client_trigger)
            y_poisoned[idx] = self.target_class

        poison_mask = np.zeros(n_samples, dtype=bool)
        if len(poison_indices) > 0:
            poison_mask[poison_indices] = True

        print(f"  [Coordinated] Client {client_id}: Poisoned {n_poison}/{n_samples} samples")
        return X_poisoned, y_poisoned, poison_mask

    def evaluate_attack_success(self, model: nn.Module, X_test: np.ndarray,
                                y_test: np.ndarray) -> float:
        """Evaluate backdoor success using the first client's trigger"""
        if not self.client_triggers:
            return 0.0
        first_client = list(self.client_triggers.keys())[0]
        trigger = self.client_triggers[first_client]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        X_triggered = np.array([self.apply_trigger(x, trigger) for x in X_test])
        X_tensor = torch.FloatTensor(X_triggered).to(device)
        with torch.no_grad():
            outputs = model(X_tensor)
            y_pred = torch.argmax(outputs, dim=1)
        return self.compute_asr(y_true=torch.tensor(y_test), y_pred=y_pred)

    def attack_summary(self) -> Dict:
        base_summary = super().attack_summary()
        base_summary.update({
            "target_class": self.target_class,
            "num_clients_with_triggers": len(self.client_triggers)
        })
        return base_summary


class RandomTriggerAttack(BaseAttack):
    """Random Trigger-Based Backdoor Attack"""

    def __init__(self, config: AttackConfig):
        super().__init__(config)
        self.target_class = config.target_class
        self.trigger_history = []

    def generate_random_trigger(self, data_shape: Tuple[int, ...],
                                seed: Optional[int] = None) -> np.ndarray:
        """Generate random combined trigger"""
        # CHANGED: stronger intensity
        trigger = TriggerGenerator.combine_triggers(data_shape,
                                                    intensity=0.35,  # CHANGED: from config value
                                                    seed=seed)
        self.trigger_history.append(trigger.copy())
        return trigger

    def apply_trigger(self, x: np.ndarray, trigger: np.ndarray) -> np.ndarray:
        triggered_x = x + trigger
        return np.clip(triggered_x, 0.0, 1.0)

    def poison_dataset(self, X: np.ndarray, y: np.ndarray,
                       client_id: int = 0, model: Optional[nn.Module] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_poisoned = X.copy()
        y_poisoned = y.copy()

        n_samples = len(X)
        n_poison = int(self.config.poisoning_rate * n_samples)
        if n_poison <= 0:
            poison_indices = []
        else:
            poison_indices = np.random.choice(n_samples, n_poison, replace=False)

        for i, idx in enumerate(poison_indices):
            seed = client_id * 10000 + idx + i
            base = self.generate_random_trigger(X[idx].shape, seed=seed)
            if model is not None:
                optimized = TriggerGenerator.align_trigger_with_gradient(
                    model, X[idx], self.target_class,
                    base_trigger=base,
                    optimization_steps=40,  # CHANGED: from 12
                    lr=0.12,  # CHANGED: from 0.05
                    epsilon=0.35  # CHANGED: from 0.1
                )
                X_poisoned[idx] = np.clip(X[idx] + optimized, 0.0, 1.0)
            else:
                X_poisoned[idx] = self.apply_trigger(X[idx], base)
            y_poisoned[idx] = self.target_class

        poison_mask = np.zeros(n_samples, dtype=bool)
        if len(poison_indices) > 0:
            poison_mask[poison_indices] = True

        print(f"  [Random] Client {client_id}: Poisoned {n_poison}/{n_samples} samples")
        return X_poisoned, y_poisoned, poison_mask

    def evaluate_attack_success(self, model: nn.Module, X_test: np.ndarray,
                                y_test: np.ndarray, n_trigger_samples: int = 10) -> float:
        """Evaluate backdoor success with multiple random triggers"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        success_rates = []
        data_shape = X_test[0].shape
        test_subset = X_test[:min(100, len(X_test))]
        for i in range(n_trigger_samples):
            seed = 50000 + i
            random_trigger = self.generate_random_trigger(data_shape, seed=seed)
            X_triggered = np.array([self.apply_trigger(x, random_trigger) for x in test_subset])
            X_tensor = torch.FloatTensor(X_triggered).to(device)
            with torch.no_grad():
                outputs = model(X_tensor)
                y_pred = torch.argmax(outputs, dim=1)
                success_rate = (y_pred == self.target_class).float().mean().item()
                success_rates.append(success_rate)
        # Return ASR for the last computed y_pred (used by compute_asr). Alternatively you could average.
        # Ensure same length between y_true and y_pred
        y_true_t = torch.tensor(y_test)
        if len(y_pred) != len(y_true_t):
            min_len = min(len(y_pred), len(y_true_t))
            y_true_t = y_true_t[:min_len]
            y_pred = y_pred[:min_len]

        return self.compute_asr(y_true=y_true_t, y_pred=y_pred)


    def attack_summary(self) -> Dict:
        base_summary = super().attack_summary()
        base_summary.update({
            "target_class": self.target_class,
            "num_unique_triggers": len(self.trigger_history)
        })
        return base_summary


class ModelDependentTriggerAttack(BaseAttack):
    """Model-Dependent Trigger-Based Backdoor Attack"""

    def __init__(self, config: AttackConfig):
        super().__init__(config)
        if config.source_class is None:
            raise ValueError("Model-dependent attacks require a source class")
        self.source_class = config.source_class
        self.target_class = config.target_class
        self.class_centroids = {}
        self.optimized_triggers: Dict[Tuple[int,int], np.ndarray] = {}

    def compute_class_centroids(self, model: nn.Module, X: np.ndarray,
                                y: np.ndarray) -> Dict[int, torch.Tensor]:
        """Compute class centroids in feature space"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        centroids: Dict[int, torch.Tensor] = {}
        unique_classes = np.unique(y)
        with torch.no_grad():
            for class_id in unique_classes:
                class_mask = (y == class_id)
                if np.sum(class_mask) > 0:
                    class_samples = X[class_mask]
                    X_tensor = torch.FloatTensor(class_samples).to(device)
                    features = model.get_features(X_tensor)
                    centroid = torch.mean(features, dim=0)
                    centroids[int(class_id)] = centroid
        return centroids

    def optimize_trigger(self, model: nn.Module, sample: np.ndarray,
                         target_centroid: torch.Tensor,
                         optimization_steps: int = 75) -> np.ndarray:  # CHANGED: from 50
        """Optimize combined base trigger"""
        base = TriggerGenerator.combine_triggers(sample.shape,
                                                 intensity=0.4,  # CHANGED: stronger intensity
                                                 seed=None)
        optimized = TriggerGenerator.align_trigger_with_gradient(
            model, sample, self.target_class,
            base_trigger=base,
            optimization_steps=optimization_steps,
            lr=0.2,  # CHANGED: from 0.05
            epsilon=0.35  # CHANGED: from 0.1
        )
        return optimized

    def poison_dataset(self, X: np.ndarray, y: np.ndarray,
                       client_id: int = 0, model: Optional[nn.Module] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Poison dataset with model-dependent triggers"""
        if model is None:
            raise ValueError("Model-dependent attacks require a model for trigger optimization")

        X_poisoned = X.copy()
        y_poisoned = y.copy()

        self.class_centroids = self.compute_class_centroids(model, X, y)
        if self.target_class not in self.class_centroids:
            print(f"  [ModelDep] Warning: Target class {self.target_class} not in client {client_id}")
            return X_poisoned, y_poisoned, np.zeros(len(X), dtype=bool)
        target_centroid = self.class_centroids[self.target_class]

        source_indices = np.where(y == self.source_class)[0]
        if len(source_indices) == 0:
            print(f"  [ModelDep] Warning: Source class {self.source_class} not in client {client_id}")
            return X_poisoned, y_poisoned, np.zeros(len(X), dtype=bool)

        if self.config.poisoning_rate >= 1.0:
            poison_indices = source_indices
        else:
            n_poison = max(1, int(self.config.poisoning_rate * len(source_indices)))
            if n_poison > len(source_indices):
                n_poison = len(source_indices)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            model.eval()
            losses = []
            with torch.no_grad():
                for idx in source_indices:
                    x_tensor = torch.FloatTensor(X[idx]).unsqueeze(0).to(device)
                    y_tensor = torch.LongTensor([int(y[idx])]).to(device)
                    loss = nn.CrossEntropyLoss()(model(x_tensor), y_tensor).item()
                    losses.append(loss)
            losses = np.array(losses)
            # Select top-loss samples, fill remaining with random if needed
            sorted_idx = np.argsort(losses)[::-1]  # descending
            top_indices = sorted_idx[:n_poison]
            if len(top_indices) < n_poison:
                remaining = list(set(range(len(source_indices))) - set(top_indices))
                additional = np.random.choice(remaining, size=n_poison - len(top_indices), replace=False)
                poison_indices = source_indices[np.concatenate([top_indices, additional])]
            else:
                poison_indices = source_indices[top_indices]
        for idx in poison_indices:
            sample_key = (client_id, int(idx))
            if sample_key in self.optimized_triggers:
                optimized_trigger = self.optimized_triggers[sample_key]
            else:
                optimized_trigger = self.optimize_trigger(
                    model, X[idx], target_centroid,
                    optimization_steps=75  # CHANGED: using the updated default
                )
                self.optimized_triggers[sample_key] = optimized_trigger
            X_poisoned[idx] = np.clip(X[idx] + optimized_trigger, 0.0, 1.0)
            y_poisoned[idx] = self.target_class

        poison_mask = np.zeros(len(X), dtype=bool)
        if len(poison_indices) > 0:
            poison_mask[poison_indices] = True

        print(f"  [ModelDep] Client {client_id}: Poisoned {len(poison_indices)}/{len(X)} samples")
        return X_poisoned, y_poisoned, poison_mask

    def evaluate_attack_success(self, model: nn.Module, X_test: np.ndarray,
                                y_test: np.ndarray) -> float:
        """Evaluate backdoor success rate"""
        source_mask = (y_test == self.source_class)
        if not np.any(source_mask):
            return 0.0
        X_source = X_test[source_mask]

        centroids = self.compute_class_centroids(model, X_test, y_test)
        if self.target_class not in centroids:
            return 0.0
        target_centroid = centroids[self.target_class]

        test_samples = X_source[:min(50, len(X_source))]
        success_rates = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        for sample in test_samples:
            optimized_trigger = self.optimize_trigger(
                model, sample, target_centroid,
                optimization_steps=getattr(self.config, "trigger_optimization_steps_eval", 10)
            )
            triggered_sample = np.clip(sample + optimized_trigger, 0.0, 1.0)
            x_tensor = torch.FloatTensor(triggered_sample).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(x_tensor)
                y_pred = torch.argmax(outputs, dim=1)
                success_rate = (y_pred == self.target_class).float().mean().item()
                success_rates.append(success_rate)
        # Return ASR based on last y_pred (consistent with other evaluate_* implementations)
        return self.compute_asr(y_true=torch.tensor(y_test), y_pred=y_pred)

    def attack_summary(self) -> Dict:
        base_summary = super().attack_summary()
        base_summary.update({
            "source_class": self.source_class,
            "target_class": self.target_class,
            "num_optimized_triggers": len(self.optimized_triggers)
        })
        return base_summary
