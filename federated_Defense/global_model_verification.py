"""
Global Model Verification Module

Runs AFTER consensus and BEFORE model dissemination to verify
the new global model is reasonable compared to the previous one.

Checks:
  1. Accuracy drop: previous_accuracy - new_accuracy
  2. Parameter deviation: ||W_{t+1} - W_t||_2

If deviations exceed thresholds, the model is marked SUSPICIOUS.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional


class GlobalModelVerifier:
    """Post-consensus, pre-dissemination global model verification."""

    def __init__(self,
                 accuracy_drop_threshold: float = 0.10,
                 param_deviation_threshold: Optional[float] = None,
                 adaptive_warmup_rounds: int = 5,
                 adaptive_multiplier: float = 3.0):
        """
        Args:
            accuracy_drop_threshold: Max acceptable accuracy drop between rounds.
                E.g. 0.10 means a drop of more than 10% flags the model.
            param_deviation_threshold: Static threshold for ||W_{t+1} - W_t||.
                If None, uses adaptive threshold based on historical deviations.
            adaptive_warmup_rounds: Number of rounds before adaptive thresholding
                activates (always BENIGN during warmup).
            adaptive_multiplier: Multiplier for MAD-based adaptive threshold
                (median + multiplier * MAD).
        """
        self.accuracy_drop_threshold = accuracy_drop_threshold
        self.param_deviation_threshold = param_deviation_threshold
        self.adaptive_warmup_rounds = adaptive_warmup_rounds
        self.adaptive_multiplier = adaptive_multiplier

        self.previous_accuracy = None
        self.previous_params = None
        self.verification_history = []
        self._deviation_history = []

    def verify(self, new_params: Dict[str, torch.Tensor],
               new_accuracy: float,
               round_id: int) -> Tuple[str, Dict]:
        """
        Verify the new global model against the previous one.

        Args:
            new_params: New aggregated model parameters (state_dict-like)
            new_accuracy: Test accuracy of the new model
            round_id: Current round number

        Returns:
            (status, details) where status is 'BENIGN' or 'SUSPICIOUS'
        """
        details = {
            'round_id': round_id,
            'accuracy_before': self.previous_accuracy if self.previous_accuracy is not None else 0.0,
            'accuracy_after': new_accuracy,
            'accuracy_drop': 0.0,
            'parameter_deviation': 0.0,
            'status': 'BENIGN',
            'flags': [],
        }

        # First round: no previous model to compare against
        if self.previous_params is None or self.previous_accuracy is None:
            self.update_baseline(new_params, new_accuracy)
            self.verification_history.append(details)
            return 'BENIGN', details

        # Compute accuracy drop
        accuracy_drop = self.previous_accuracy - new_accuracy
        details['accuracy_drop'] = accuracy_drop

        # Compute parameter deviation ||W_{t+1} - W_t||_2
        param_deviation = self._compute_parameter_deviation(self.previous_params, new_params)
        details['parameter_deviation'] = param_deviation
        self._deviation_history.append(param_deviation)

        # Check accuracy drop
        if accuracy_drop > self.accuracy_drop_threshold:
            details['flags'].append(f'accuracy_drop={accuracy_drop:.4f}>{self.accuracy_drop_threshold}')

        # Check parameter deviation
        threshold = self._get_deviation_threshold()
        if threshold is not None and param_deviation > threshold:
            details['flags'].append(f'param_deviation={param_deviation:.4f}>{threshold:.4f}')

        # Mark status
        if details['flags'] and round_id > self.adaptive_warmup_rounds:
            details['status'] = 'SUSPICIOUS'
        else:
            details['status'] = 'BENIGN'

        self.verification_history.append(details)
        self.update_baseline(new_params, new_accuracy)

        return details['status'], details

    def update_baseline(self, params: Dict[str, torch.Tensor], accuracy: float):
        """Update stored previous model for next round's comparison."""
        self.previous_accuracy = accuracy
        self.previous_params = {
            name: val.cpu().clone() if isinstance(val, torch.Tensor) else val
            for name, val in params.items()
        }

    def _compute_parameter_deviation(self, old_params: Dict[str, torch.Tensor],
                                      new_params: Dict[str, torch.Tensor]) -> float:
        """Compute ||W_{t+1} - W_t||_2."""
        diff_vec = []
        for name in old_params:
            if name in new_params:
                old_t = old_params[name].float()
                new_t = new_params[name].float()
                diff_vec.append((new_t - old_t).flatten())
        if not diff_vec:
            return 0.0
        full_diff = torch.cat(diff_vec)
        return torch.norm(full_diff, p=2).item()

    def _get_deviation_threshold(self) -> Optional[float]:
        """Get the parameter deviation threshold (static or adaptive)."""
        if self.param_deviation_threshold is not None:
            return self.param_deviation_threshold

        # Adaptive: need enough history for a meaningful threshold
        if len(self._deviation_history) < self.adaptive_warmup_rounds:
            return None

        dev_array = np.array(self._deviation_history)
        median_dev = float(np.median(dev_array))
        mad = float(np.median(np.abs(dev_array - median_dev)))
        return median_dev + self.adaptive_multiplier * max(mad, 1e-8)
