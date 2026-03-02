from attacks.base import AttackConfig, BaseAttack
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from typing import Optional, Tuple, Dict, Union


def _is_text_data(shape: Tuple[int, ...]) -> bool:
    """1D data = text sequence (Shakespeare / Sentiment140), 2D/3D = image."""
    return len(shape) == 1


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
    def combine_triggers(image_shape: Tuple[int, ...], intensity: Optional[float] = 0.8,  # CHANGED: Very high intensity for visible impact
                         seed: Optional[int] = None) -> np.ndarray:
        """Combine pixel + checkerboard + random patch into a single base trigger, robust defaults."""
        if intensity is None:
            intensity = 0.8  # CHANGED: High default intensity for aggressive attacks
        if seed is not None:
            rndstate = np.random.get_state()
            np.random.seed(seed)

        # CHANGED: Much larger pattern sizes for highly visible triggers
        t_pixel = TriggerGenerator.pixel_pattern_trigger(image_shape, pattern_size=6,  # CHANGED: larger pattern
                                                         position="bottom_right", intensity=float(intensity))
        t_check = TriggerGenerator.checkerboard_trigger(image_shape, square_size=3,  # CHANGED: larger squares
                                                        position="bottom_right")
        t_rand = TriggerGenerator.random_patch_trigger(image_shape, patch_size=(6,6),  # CHANGED: larger patch
                                                       intensity=float(intensity))

        # CHANGED: Much stronger combination for aggressive, visible attacks
        base = t_pixel * 2.5 + (t_check * float(intensity) * 3.0) + (t_rand * float(intensity) * 2.5)
        base = np.clip(base, -1.0, 1.0)

        if seed is not None:
            np.random.set_state(rndstate)

        return base

    # ==============================
    # TEXT (1-D) TRIGGER UTILITIES
    # ==============================
    @staticmethod
    def text_trigger(seq_len: int, trigger_size: int = 4,
                     position: str = 'end',
                     seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate trigger tokens for a 1-D text sequence.

        Args:
            seq_len:      Length of the input sequence (e.g. 80 for Shakespeare).
            trigger_size: Number of tokens to replace (capped at seq_len // 5).
            position:     'end' (default), 'start', or 'random'.
            seed:         Deterministic seed for reproducibility.

        Returns:
            (positions, values) — both int64 numpy arrays.
            ``positions`` are indices into the sequence; ``values`` are the
            replacement token ids (rare / high-index tokens).
        """
        trigger_size = max(1, min(trigger_size, max(1, seq_len // 5)))

        rng = np.random.RandomState(seed if seed is not None else 42)

        # --- positions ---
        if position == 'start':
            positions = np.arange(trigger_size, dtype=np.int64)
        elif position == 'random':
            positions = rng.choice(seq_len, size=trigger_size, replace=False).astype(np.int64)
            positions.sort()
        else:  # 'end' (default)
            positions = np.arange(seq_len - trigger_size, seq_len, dtype=np.int64)

        # --- values: rare high-index tokens, deterministic from seed ---
        # Shakespeare vocab 0-79, Sentiment140 vocab 0-~30000.
        # We pick from the top 10 % of the index range so the tokens are rare.
        low = max(1, int(seq_len * 0.9))
        high = max(low + 1, seq_len)
        values = rng.randint(low, high, size=trigger_size).astype(np.int64)

        return positions, values

    @staticmethod
    def apply_text_trigger(x: np.ndarray,
                           positions: np.ndarray,
                           values: np.ndarray) -> np.ndarray:
        """Replace tokens at *positions* with *values* in a 1-D sequence ``x``."""
        x_out = x.copy()
        for pos, val in zip(positions, values):
            if 0 <= pos < len(x_out):
                x_out[pos] = val
        return x_out

    # ==============================
    # ALIGN TRIGGER WITH GRADIENT
    # ==============================
    @staticmethod
    def align_trigger_with_gradient(model: torch.nn.Module, x: np.ndarray, y_target: int,
                                    base_trigger: Optional[np.ndarray] = None,
                                    optimization_steps: int = 75, lr: float = 0.25,  # CHANGED: more steps, higher lr for stronger attacks
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
            epsilon = 0.6  # CHANGED: much larger epsilon for aggressive attacks

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

            # Project trigger to epsilon-ball (L2) - FIXED: Stable projection without multiplier
            with torch.no_grad():
                flat = trigger_t.view(trigger_t.size(0), -1)
                norms = flat.norm(p=2, dim=1, keepdim=True)
                # Proper L2 projection: clip norm to epsilon (stable and correct)
                scale = torch.clamp(epsilon / (norms + 1e-12), max=1.0)
                trigger_t.data = (flat * scale).view_as(trigger_t)

        trigger_np = trigger_t.detach().squeeze(0).cpu().numpy()
        return trigger_np


# =============================================================================
# TRIGGER-BASED ATTACKS (CENTRALIZED, COORDINATED, RANDOM, MODEL-DEPENDENT)
# =============================================================================

class CentralizedTriggerAttack(BaseAttack):
    """Centralized Trigger-Based Backdoor Attack (FIXED: source class targeting)"""

    def __init__(self, config: AttackConfig):
        super().__init__(config)
        self.trigger_pattern = None

        # FIXED: Support multiple source classes for backdoor injection
        if config.source_class is not None:
            if isinstance(config.source_class, (list, tuple)):
                self.source_classes = list(config.source_class)
            else:
                self.source_classes = [config.source_class]
        else:
            # Default: Use even digits as source classes for MNIST-like datasets
            self.source_classes = [0, 2, 4, 6, 8]

        # Ensure target_class is a single integer (not a list)
        if isinstance(config.target_class, (list, tuple)):
            self.target_class = config.target_class[0]
        else:
            self.target_class = config.target_class

    def generate_trigger(self, data_shape: Tuple[int, ...]):
        """Generate fixed combined base trigger for centralized coordination.
        Returns (positions, values) tuple for text data, np.ndarray for images."""
        if _is_text_data(data_shape):
            return TriggerGenerator.text_trigger(data_shape[0], seed=42)
        # Use FIXED seed=42 so all clients generate identical triggers
        base = TriggerGenerator.combine_triggers(data_shape,
                                                 intensity=0.9,
                                                 seed=42)  # Fixed seed for centralized attack
        return base

    def apply_trigger(self, x: np.ndarray, trigger=None) -> np.ndarray:
        """Apply trigger to input sample. Trigger is (positions, values) for text, ndarray for images."""
        if trigger is None:
            if self.trigger_pattern is None:
                self.trigger_pattern = self.generate_trigger(x.shape)
            trigger = self.trigger_pattern
        if isinstance(trigger, tuple):
            positions, values = trigger
            return TriggerGenerator.apply_text_trigger(x, positions, values)
        triggered_x = x + trigger
        return np.clip(triggered_x, 0.0, 1.0)

    def poison_dataset(self, X: np.ndarray, y: np.ndarray,
                       client_id: int = 0, model: Optional[nn.Module] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        FIXED: Poison SOURCE CLASS samples only (proper backdoor attack).
        Injects centralized trigger (same for all clients) into source class samples.
        """
        X_poisoned = X.copy()
        y_poisoned = y.copy()

        # FIXED: Select only SOURCE CLASS samples for poisoning
        source_mask = np.isin(y, self.source_classes)
        source_indices = np.where(source_mask)[0]

        if len(source_indices) == 0:
            print(f"  [Centralized] Client {client_id}: No source class samples found")
            return X_poisoned, y_poisoned, np.zeros(len(X), dtype=bool)

        # Apply data_poisoning_rate to SOURCE class samples only
        n_poison = int(self.config.data_poisoning_rate * len(source_indices))
        if n_poison <= 0:
            poison_indices = []
        else:
            poison_indices = np.random.choice(source_indices, n_poison, replace=False)

        # FIXED: Generate a SINGLE consistent trigger pattern for all poisoned samples
        # For Centralized attack, ALL clients must use the SAME trigger (fixed seed)
        text_mode = _is_text_data(X[0].shape)
        if len(poison_indices) > 0:
            if self.trigger_pattern is None:
                if text_mode:
                    self.trigger_pattern = TriggerGenerator.text_trigger(
                        X[0].shape[0], seed=42)
                else:
                    self.trigger_pattern = TriggerGenerator.combine_triggers(
                        X[0].shape,
                        intensity=0.9,
                        seed=42  # FIXED seed for centralized coordination
                    )

        # Apply the SAME trigger to all poisoned samples
        for idx in poison_indices:
            if model is not None and not text_mode:
                # Optimize the consistent trigger for better effectiveness
                # (gradient optimization is skipped for text — discrete tokens)
                optimized = TriggerGenerator.align_trigger_with_gradient(
                    model, X[idx], self.target_class,
                    base_trigger=self.trigger_pattern,  # Use consistent trigger
                    optimization_steps=75,
                    lr=0.25,
                    epsilon=0.7
                )
                X_poisoned[idx] = np.clip(X[idx] + optimized, 0.0, 1.0)
            else:
                X_poisoned[idx] = self.apply_trigger(X[idx], self.trigger_pattern)
            y_poisoned[idx] = self.target_class

        poison_mask = np.zeros(len(X), dtype=bool)
        if len(poison_indices) > 0:
            poison_mask[poison_indices] = True

        print(f"  [Centralized] Client {client_id}: Poisoned {n_poison}/{len(source_indices)} source class samples ({n_poison/len(source_indices):.1%})")
        return X_poisoned, y_poisoned, poison_mask

    def evaluate_attack_success(self, model: nn.Module, X_test: np.ndarray,
                                y_test: np.ndarray) -> float:
        """Evaluate backdoor success rate"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        # FIXED: Only evaluate on non-target samples (standard backdoor ASR definition)
        non_target_mask = (y_test != self.target_class)
        if not np.any(non_target_mask):
            return 0.0

        X_non_target = X_test[non_target_mask]

        # Apply trigger to non-target samples
        trigger = self.trigger_pattern if self.trigger_pattern is not None else self.generate_trigger(X_non_target[0].shape)
        X_triggered = np.array([self.apply_trigger(x, trigger) for x in X_non_target])

        X_tensor = torch.FloatTensor(X_triggered).to(device)
        with torch.no_grad():
            outputs = model(X_tensor)
            y_pred = torch.argmax(outputs, dim=1)

        success_mask = (y_pred.cpu() == self.target_class)
        asr = success_mask.float().mean().item()
        return asr

    def attack_summary(self) -> Dict:
        base_summary = super().attack_summary()
        base_summary.update({
            "target_class": self.target_class,
            "trigger_type": "centralized_combined"
        })
        return base_summary


class CoordinatedTriggerAttack(BaseAttack):
    """Coordinated Trigger-Based Backdoor Attack (FIXED: source class targeting)"""

    def __init__(self, config: AttackConfig):
        super().__init__(config)
        self.client_triggers: Dict[int, object] = {}  # ndarray (image) or tuple (text)

        # FIXED: Support multiple source classes
        if config.source_class is not None:
            if isinstance(config.source_class, (list, tuple)):
                self.source_classes = list(config.source_class)
            else:
                self.source_classes = [config.source_class]
        else:
            self.source_classes = [0, 2, 4, 6, 8]  # Default for MNIST

        # Ensure target_class is a single integer (not a list)
        if isinstance(config.target_class, (list, tuple)):
            self.target_class = config.target_class[0]
        else:
            self.target_class = config.target_class

    def generate_client_trigger(self, client_id: int, data_shape: Tuple[int, ...]):
        """Generate per-client trigger. Returns tuple for text, ndarray for images."""
        if _is_text_data(data_shape):
            return TriggerGenerator.text_trigger(data_shape[0], seed=client_id + 1000)
        base = TriggerGenerator.combine_triggers(data_shape,
                                                 intensity=0.85,
                                                 seed=client_id + 1000)
        return base

    def get_client_trigger(self, client_id: int, data_shape: Tuple[int, ...],
                          model: Optional[nn.Module] = None):
        """Get or create optimized trigger for specific client"""
        if client_id not in self.client_triggers:
            base = self.generate_client_trigger(client_id, data_shape)
            if isinstance(base, tuple):
                # Text trigger — no gradient optimization for discrete tokens
                self.client_triggers[client_id] = base
            elif model is not None:
                representative_dummy = np.zeros(data_shape, dtype=np.float32)
                optimized = TriggerGenerator.align_trigger_with_gradient(
                    model, representative_dummy, self.target_class,
                    base_trigger=base,
                    optimization_steps=75,
                    lr=0.25,
                    epsilon=0.65
                )
                self.client_triggers[client_id] = optimized
            else:
                self.client_triggers[client_id] = base
        return self.client_triggers[client_id]

    def apply_trigger(self, x: np.ndarray, trigger) -> np.ndarray:
        """Apply trigger. Trigger is (positions, values) for text, ndarray for images."""
        if isinstance(trigger, tuple):
            positions, values = trigger
            return TriggerGenerator.apply_text_trigger(x, positions, values)
        triggered_x = x + trigger
        return np.clip(triggered_x, 0.0, 1.0)

    def poison_dataset(self, X: np.ndarray, y: np.ndarray,
                       client_id: int = 0, model: Optional[nn.Module] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """FIXED: Poison SOURCE CLASS samples only (proper backdoor attack)."""
        X_poisoned = X.copy()
        y_poisoned = y.copy()

        # FIXED: Select only SOURCE CLASS samples
        source_mask = np.isin(y, self.source_classes)
        source_indices = np.where(source_mask)[0]

        if len(source_indices) == 0:
            print(f"  [Coordinated] Client {client_id}: No source class samples found")
            return X_poisoned, y_poisoned, np.zeros(len(X), dtype=bool)

        data_shape = X[0].shape if len(X) > 0 else X.shape[1:]
        client_trigger = self.get_client_trigger(client_id, data_shape, model=model)

        # Apply data_poisoning_rate to SOURCE class samples only
        n_poison = int(self.config.data_poisoning_rate * len(source_indices))
        if n_poison <= 0:
            poison_indices = []
        else:
            poison_indices = np.random.choice(source_indices, n_poison, replace=False)

        for idx in poison_indices:
            X_poisoned[idx] = self.apply_trigger(X[idx], client_trigger)
            y_poisoned[idx] = self.target_class

        poison_mask = np.zeros(len(X), dtype=bool)
        if len(poison_indices) > 0:
            poison_mask[poison_indices] = True

        print(f"  [Coordinated] Client {client_id}: Poisoned {n_poison}/{len(source_indices)} source class samples ({n_poison/len(source_indices):.1%})")
        return X_poisoned, y_poisoned, poison_mask

    def evaluate_attack_success(self, model: nn.Module, X_test: np.ndarray,
                                y_test: np.ndarray) -> float:
        """Evaluate backdoor success using the first client's trigger"""
        if not self.client_triggers:
            return 0.0

        # FIXED: Only evaluate on non-target samples
        non_target_mask = (y_test != self.target_class)
        if not np.any(non_target_mask):
            return 0.0

        X_non_target = X_test[non_target_mask]

        first_client = list(self.client_triggers.keys())[0]
        trigger = self.client_triggers[first_client]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        X_triggered = np.array([self.apply_trigger(x, trigger) for x in X_non_target])
        X_tensor = torch.FloatTensor(X_triggered).to(device)
        with torch.no_grad():
            outputs = model(X_tensor)
            y_pred = torch.argmax(outputs, dim=1)

        # ASR = proportion of non-target samples misclassified as target
        success_mask = (y_pred.cpu() == self.target_class)
        asr = success_mask.float().mean().item()
        return asr

    def attack_summary(self) -> Dict:
        base_summary = super().attack_summary()
        base_summary.update({
            "target_class": self.target_class,
            "num_clients_with_triggers": len(self.client_triggers)
        })
        return base_summary


class RandomTriggerAttack(BaseAttack):
    """Random Trigger-Based Backdoor Attack (FIXED: source class targeting)"""

    def __init__(self, config: AttackConfig):
        super().__init__(config)

        # FIXED: Support multiple source classes
        if config.source_class is not None:
            if isinstance(config.source_class, (list, tuple)):
                self.source_classes = list(config.source_class)
            else:
                self.source_classes = [config.source_class]
        else:
            self.source_classes = [0, 2, 4, 6, 8]  # Default for MNIST

        # Ensure target_class is a single integer (not a list)
        if isinstance(config.target_class, (list, tuple)):
            self.target_class = config.target_class[0]
        else:
            self.target_class = config.target_class
        self.trigger_history = []

    def generate_random_trigger(self, data_shape: Tuple[int, ...],
                                seed: Optional[int] = None):
        """Generate random trigger. Returns tuple for text, ndarray for images."""
        if _is_text_data(data_shape):
            trig = TriggerGenerator.text_trigger(data_shape[0], seed=seed)
            self.trigger_history.append(trig)
            return trig
        trigger = TriggerGenerator.combine_triggers(data_shape,
                                                    intensity=0.85,
                                                    seed=seed)
        self.trigger_history.append(trigger.copy())
        return trigger

    def apply_trigger(self, x: np.ndarray, trigger) -> np.ndarray:
        """Apply trigger. Trigger is (positions, values) for text, ndarray for images."""
        if isinstance(trigger, tuple):
            positions, values = trigger
            return TriggerGenerator.apply_text_trigger(x, positions, values)
        triggered_x = x + trigger
        return np.clip(triggered_x, 0.0, 1.0)

    def poison_dataset(self, X: np.ndarray, y: np.ndarray,
                       client_id: int = 0, model: Optional[nn.Module] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """FIXED: Poison SOURCE CLASS samples only (proper backdoor attack)."""
        X_poisoned = X.copy()
        y_poisoned = y.copy()

        # FIXED: Select only SOURCE CLASS samples
        source_mask = np.isin(y, self.source_classes)
        source_indices = np.where(source_mask)[0]

        if len(source_indices) == 0:
            print(f"  [Random] Client {client_id}: No source class samples found")
            return X_poisoned, y_poisoned, np.zeros(len(X), dtype=bool)

        # Apply data_poisoning_rate to SOURCE class samples only
        n_poison = int(self.config.data_poisoning_rate * len(source_indices))
        if n_poison <= 0:
            poison_indices = []
        else:
            poison_indices = np.random.choice(source_indices, n_poison, replace=False)

        text_mode = _is_text_data(X[0].shape)
        for i, idx in enumerate(poison_indices):
            seed = client_id * 10000 + idx + i
            base = self.generate_random_trigger(X[idx].shape, seed=seed)
            if model is not None and not text_mode:
                # Gradient optimization is skipped for text (discrete tokens)
                optimized = TriggerGenerator.align_trigger_with_gradient(
                    model, X[idx], self.target_class,
                    base_trigger=base,
                    optimization_steps=75,
                    lr=0.25,
                    epsilon=0.65
                )
                X_poisoned[idx] = np.clip(X[idx] + optimized, 0.0, 1.0)
            else:
                X_poisoned[idx] = self.apply_trigger(X[idx], base)
            y_poisoned[idx] = self.target_class

        poison_mask = np.zeros(len(X), dtype=bool)
        if len(poison_indices) > 0:
            poison_mask[poison_indices] = True

        print(f"  [Random] Client {client_id}: Poisoned {n_poison}/{len(source_indices)} source class samples ({n_poison/len(source_indices):.1%})")
        return X_poisoned, y_poisoned, poison_mask

    def evaluate_attack_success(self, model: nn.Module, X_test: np.ndarray,
                                y_test: np.ndarray, n_trigger_samples: int = 10) -> float:
        """Evaluate backdoor success with multiple random triggers"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        # FIXED: Only evaluate on non-target samples
        non_target_mask = (y_test != self.target_class)
        if not np.any(non_target_mask):
            return 0.0

        X_non_target = X_test[non_target_mask]
        test_subset = X_non_target[:min(100, len(X_non_target))]

        success_rates = []
        data_shape = test_subset[0].shape

        # Properly compute ASR by averaging success rates across multiple random triggers
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

        # Return the average success rate
        return float(np.mean(success_rates)) if success_rates else 0.0


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
        # Ensure source_class and target_class are single integers (not lists)
        if isinstance(config.source_class, (list, tuple)):
            self.source_class = config.source_class[0]
        else:
            self.source_class = config.source_class
        if isinstance(config.target_class, (list, tuple)):
            self.target_class = config.target_class[0]
        else:
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
                         optimization_steps: int = 75):
        """Generate/optimize trigger. Returns (positions, values) for text, ndarray for images."""
        if _is_text_data(sample.shape):
            # Text data: discrete tokens, no gradient optimization possible
            return TriggerGenerator.text_trigger(sample.shape[0], seed=42)
        base = TriggerGenerator.combine_triggers(sample.shape,
                                                 intensity=0.9,
                                                 seed=None)
        optimized = TriggerGenerator.align_trigger_with_gradient(
            model, sample, self.target_class,
            base_trigger=base,
            optimization_steps=optimization_steps,
            lr=0.3,
            epsilon=0.7
        )
        return optimized

    def poison_dataset(self, X: np.ndarray, y: np.ndarray,
                       client_id: int = 0, model: Optional[nn.Module] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        FIXED: Poison SOURCE CLASS samples only (proper backdoor attack).
        Injects trigger into source class samples and flips to target class.
        """
        if model is None:
            raise ValueError("Model-dependent attacks require a model for trigger optimization")

        X_poisoned = X.copy()
        y_poisoned = y.copy()

        self.class_centroids = self.compute_class_centroids(model, X, y)
        if self.target_class not in self.class_centroids:
            print(f"  [ModelDep] Warning: Target class {self.target_class} not in client {client_id}")
            return X_poisoned, y_poisoned, np.zeros(len(X), dtype=bool)
        target_centroid = self.class_centroids[self.target_class]

        # FIXED: Poison only SOURCE CLASS samples (proper backdoor behavior)
        source_indices = np.where(y == self.source_class)[0]

        if len(source_indices) == 0:
            print(f"  [ModelDep] Warning: No source class {self.source_class} samples in client {client_id}")
            return X_poisoned, y_poisoned, np.zeros(len(X), dtype=bool)

        # Apply data_poisoning_rate to SOURCE class samples only
        n_poison = int(self.config.data_poisoning_rate * len(source_indices))

        if n_poison <= 0:
            print(f"  [ModelDep] Warning: No samples to poison (data_poisoning_rate={self.config.data_poisoning_rate})")
            return X_poisoned, y_poisoned, np.zeros(len(X), dtype=bool)

        # Select subset of source class samples to poison
        poison_indices = np.random.choice(source_indices, min(n_poison, len(source_indices)), replace=False)

        print(f"  [ModelDep] Client {client_id}: Poisoning {len(poison_indices)}/{len(source_indices)} source class {self.source_class} samples")

        # FIXED: Use a SINGLE consistent optimized trigger for all poisoned samples
        # This ensures the model learns ONE backdoor pattern instead of many different ones
        client_key = (client_id, 'shared')
        if client_key not in self.optimized_triggers:
            # Optimize trigger on a representative sample
            representative_sample = X[poison_indices[0]]
            optimized_trigger = self.optimize_trigger(
                model, representative_sample, target_centroid,
                optimization_steps=75
            )
            self.optimized_triggers[client_key] = optimized_trigger

        # Apply the SAME optimized trigger to ALL poisoned samples
        shared_trigger = self.optimized_triggers[client_key]
        for idx in poison_indices:
            if isinstance(shared_trigger, tuple):
                positions, values = shared_trigger
                X_poisoned[idx] = TriggerGenerator.apply_text_trigger(X[idx], positions, values)
            else:
                X_poisoned[idx] = np.clip(X[idx] + shared_trigger, 0.0, 1.0)
            y_poisoned[idx] = self.target_class

        poison_mask = np.zeros(len(X), dtype=bool)
        if len(poison_indices) > 0:
            poison_mask[poison_indices] = True

        print(f"  [ModelDep] Client {client_id}: Total poisoned {len(poison_indices)}/{len(X)} samples ({len(poison_indices)/len(X):.1%})")
        return X_poisoned, y_poisoned, poison_mask

    def evaluate_attack_success(self, model: nn.Module, X_test: np.ndarray,
                                y_test: np.ndarray) -> float:
        """Evaluate backdoor success rate on non-target class samples"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        # FIXED: Only evaluate on non-target samples
        non_target_mask = (y_test != self.target_class)
        if not np.any(non_target_mask):
            return 0.0

        X_non_target = X_test[non_target_mask]
        test_samples = X_non_target[:min(100, len(X_non_target))]

        # FIXED: Use the shared trigger from training (if available)
        # Otherwise, optimize a single trigger for evaluation
        shared_trigger = None
        for key in self.optimized_triggers:
            if key[1] == 'shared':  # Find the shared trigger
                shared_trigger = self.optimized_triggers[key]
                break

        if shared_trigger is None:
            # If no shared trigger exists, optimize one
            centroids = self.compute_class_centroids(model, X_test, y_test)
            if self.target_class not in centroids:
                return 0.0
            target_centroid = centroids[self.target_class]
            shared_trigger = self.optimize_trigger(
                model, test_samples[0], target_centroid,
                optimization_steps=75
            )

        # Apply the SAME trigger to all test samples
        if isinstance(shared_trigger, tuple):
            positions, values = shared_trigger
            X_triggered = np.array([TriggerGenerator.apply_text_trigger(s, positions, values) for s in test_samples])
        else:
            X_triggered = np.array([np.clip(sample + shared_trigger, 0.0, 1.0) for sample in test_samples])
        X_tensor = torch.FloatTensor(X_triggered).to(device)

        with torch.no_grad():
            outputs = model(X_tensor)
            y_pred = torch.argmax(outputs, dim=1)
            asr = (y_pred == self.target_class).float().mean().item()

        return asr

    def attack_summary(self) -> Dict:
        base_summary = super().attack_summary()
        base_summary.update({
            "source_class": self.source_class,
            "target_class": self.target_class,
            "num_optimized_triggers": len(self.optimized_triggers)
        })
        return base_summary

