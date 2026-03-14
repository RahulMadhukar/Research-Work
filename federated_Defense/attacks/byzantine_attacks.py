"""
Byzantine Gradient Manipulation Attacks

These attacks manipulate the model update (gradient) directly before aggregation,
following the classical Byzantine attack models from the federated learning literature.

All three attacks share the same mechanism:
    1. Capture the model state before local training  (in poison_dataset)
    2. After training, compute  update = trained_params - initial_params
    3. Manipulate the update according to the attack strategy
    4. Return  initial_params + manipulated_update  as the submitted model

Attacks:
    GradientScalingAttack : scales each element of the update by random λ ∈ [a, 1)
    SameValueAttack       : replaces the update with zeros (no-op submission)
    BackGradientAttack    : negates the update direction
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict
from .base import BaseAttack, AttackConfig


class _ByzantineBase(BaseAttack):
    """
    Shared base for Byzantine gradient manipulation attacks.

    poison_dataset()  — returns data untouched; snapshots the current model
                        (= the global model at round start) into _initial_params.
    poison_model()    — called post-training by client.py; computes the update,
                        delegates manipulation to _craft_update(), and rebuilds
                        the model parameters.
    """
    POISONS_DATA  = False   # data returned unchanged (snapshot only)
    POISONS_MODEL = True    # real attack lives in poison_model()

    def __init__(self, config: AttackConfig):
        super().__init__(config)
        self.poison_mask = np.array([], dtype=bool)
        self._initial_params: Optional[Dict[str, torch.Tensor]] = None

    # ------------------------------------------------------------------
    # Data poisoning hook  (no-op; only used to snapshot initial model)
    # ------------------------------------------------------------------
    def poison_dataset(self, X: np.ndarray, y: np.ndarray,
                       client_id: int = 0,
                       model: Optional[nn.Module] = None):
        self.poison_mask = np.zeros(len(y), dtype=bool)

        if model is not None:
            self._initial_params = {
                name: param.data.cpu().clone()
                for name, param in model.named_parameters()
            }

        return X, y, self.poison_mask

    # ------------------------------------------------------------------
    # Update computation helpers
    # ------------------------------------------------------------------
    def _get_update(self, trained_model: nn.Module) -> Dict[str, torch.Tensor]:
        """update = trained_params - initial_params  (per parameter tensor)."""
        update = {}
        for name, param in trained_model.named_parameters():
            if self._initial_params is not None and name in self._initial_params:
                update[name] = param.data.cpu() - self._initial_params[name]
            else:
                # Fallback: treat full params as the update
                update[name] = param.data.cpu().clone()
        return update

    def _apply_update(self, model: nn.Module,
                      manipulated_update: Dict[str, torch.Tensor]) -> nn.Module:
        """Set  model_params = initial_params + manipulated_update."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if self._initial_params is not None and name in self._initial_params:
                    param.data.copy_(
                        self._initial_params[name] + manipulated_update[name]
                    )
                else:
                    param.data.copy_(manipulated_update[name])
        return model

    # ------------------------------------------------------------------
    # Main poisoning entry-point (called by client.py after local training)
    # ------------------------------------------------------------------
    def poison_model(self, model: nn.Module) -> nn.Module:
        update = self._get_update(model)
        manipulated = self._craft_update(update)
        return self._apply_update(model, manipulated)

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------
    def _craft_update(self, update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Evaluation  (untargeted attack → success = accuracy degradation)
    # ------------------------------------------------------------------
    def evaluate_attack_success(self, model: nn.Module,
                                X_test: np.ndarray,
                                y_test: np.ndarray) -> float:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(device)
            y_tensor = torch.LongTensor(y_test)
            outputs = model(X_tensor)
            preds = torch.argmax(outputs.cpu(), dim=1)
            accuracy = (preds == y_tensor).float().mean().item()

        return 1.0 - accuracy


# ======================================================================
# Concrete attacks
# ======================================================================

class GradientScalingAttack(_ByzantineBase):
    """
    Gradient Scaling Attack  (CMFL paper Section 6.3.1, enhanced).

    Multi-mode attack that randomly selects between three strategies:
      Mode 1 (40%): Negative scaling — partially reverses gradient direction
      Mode 2 (40%): Amplification — overwhelms honest gradients in FedAvg
      Mode 3 (20%): Original paper attenuation — weakened but correct direction

    Config:
        epsilon  – scaling parameter *a* (default 0.5).
    """

    def __init__(self, config: AttackConfig):
        super().__init__(config)
        self.a = config.epsilon if config.epsilon is not None else 0.5

    def _craft_update(self, update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mode = np.random.random()
        if mode < 0.4:
            # Mode 1: Negative scaling — partially reverses gradient direction
            lam = np.random.uniform(-1.0, -self.a)
        elif mode < 0.8:
            # Mode 2: Amplification — overwhelms honest gradients in FedAvg
            lam = np.random.uniform(5.0, 10.0)
        else:
            # Mode 3: Original paper attenuation
            lam = np.random.uniform(self.a, 1.0)
        return {name: tensor * lam for name, tensor in update.items()}


class SameValueAttack(_ByzantineBase):
    """
    Same-Value (Zero) Attack  (CMFL paper Section 6.3.1, Ref [55]).

    Replaces the entire update with zeros.  The malicious client effectively
    sends back the unchanged global model — contributing nothing to the
    aggregated update.
    """

    def _craft_update(self, update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {name: torch.zeros_like(tensor) for name, tensor in update.items()}


class BackGradientAttack(_ByzantineBase):
    """
    Back-Gradient Attack  (CMFL paper Section 6.3.1, Ref [56]).

    Replaces the update with a vector in the *opposite direction*: the
    malicious client sends  -1 × honest_update.  Simple negation, no
    amplification, matching the original paper definition.
    """

    def _craft_update(self, update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {name: -tensor for name, tensor in update.items()}
