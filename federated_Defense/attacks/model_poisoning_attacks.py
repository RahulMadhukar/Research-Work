"""
Model Poisoning Attacks for Federated Learning

This module implements various model poisoning attacks that target the model parameters
rather than the training data:

1. Local Model Poisoning Attacks:
   - Local Model Replacement Attack: Replace local model with malicious parameters
   - Local Model Poisoning via Noise: Add carefully crafted noise to model parameters

2. Global Model Poisoning Attacks:
   - Global Model Replacement Attack: Attempt to replace the global model
   - Aggregation Process Modification: Manipulate the aggregation process
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from .base import BaseAttack, AttackConfig


class LocalModelReplacementAttack(BaseAttack):
    """
    Local Model Noise Injection Attack (FIXED: formerly replacement)

    The attacker adds carefully scaled noise to their trained local model parameters
    to degrade global model performance while remaining somewhat undetectable.

    Strategy (FIXED):
    - ADD scaled Gaussian noise to trained model weights (not replacement)
    - Or flip and scale down gradients for moderate attack
    - Or scale parameters toward zero (model degradation)

    Note: Uses ADDITIVE perturbations for realistic FL attack (10-20% impact)
    """
    POISONS_DATA  = False   # poison_dataset only sets a tracking mask; data unchanged
    POISONS_MODEL = True    # real attack in poison_model()

    def __init__(self, config: AttackConfig):
        super().__init__(config)
        self.poison_mask = None
        self.replacement_scale = config.epsilon if config.epsilon else 0.05  # FIXED: Use small perturbation scale
        self.strategy = 'additive_noise'  # FIXED: Use additive instead of replacement

    def poison_dataset(self, X: np.ndarray, y: np.ndarray, client_id: int = 0, model: Optional[nn.Module] = None):
        """
        For model replacement attack, we don't poison the dataset.
        The poisoning happens at the model parameter level.
        """
        # Create poison mask (mark all samples as poisoned for tracking)
        n_samples = len(y)
        n_poison = int(n_samples * self.config.data_poisoning_rate)

        self.poison_mask = np.zeros(n_samples, dtype=bool)
        poison_indices = np.random.choice(n_samples, n_poison, replace=False)
        self.poison_mask[poison_indices] = True

        # Return original data unchanged (poisoning happens at model level)
        return X, y, self.poison_mask

    def poison_model(self, model: nn.Module) -> nn.Module:
        """
        Add noise perturbations to model parameters (FIXED: additive instead of replacement).
        This should be called after local training.
        """
        with torch.no_grad():
            if self.strategy == 'additive_noise':
                # FIXED: ADD noise to trained model instead of replacing
                for param in model.parameters():
                    noise = torch.randn_like(param) * self.replacement_scale
                    param.data += noise  # Additive perturbation

            elif self.strategy == 'gradient_flip':
                # Flip the sign and scale down (moderate attack)
                for param in model.parameters():
                    param.data = -param.data * min(self.replacement_scale, 0.1)

            elif self.strategy == 'zero_model':
                # Add zero-centering noise instead of complete zeroing
                for param in model.parameters():
                    param.data *= (1.0 - self.replacement_scale)

        return model

    def evaluate_attack_success(self, model: nn.Module, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Measure attack success by model accuracy degradation.
        Lower accuracy = higher attack success
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(device)
            y_tensor = torch.LongTensor(y_test)

            outputs = model(X_tensor)
            predictions = torch.argmax(outputs.cpu(), dim=1)

            correct = (predictions == y_tensor).sum().item()
            total = len(y_tensor)

        accuracy = correct / total if total > 0 else 0.0
        # Attack success is inverse of accuracy (higher ASR = lower accuracy)
        attack_success_rate = 1.0 - accuracy

        return attack_success_rate


class LocalModelNoiseAttack(BaseAttack):
    """
    Local Model Poisoning through Adding Noise

    Add carefully crafted Gaussian or adversarial noise to model parameters
    to degrade global model performance while maintaining plausibility.

    This attack is more subtle than full replacement and harder to detect.
    """
    POISONS_DATA  = False   # poison_dataset only sets a tracking mask; data unchanged
    POISONS_MODEL = True    # real attack in poison_model()

    def __init__(self, config: AttackConfig):
        super().__init__(config)
        self.poison_mask = None
        self.noise_scale = config.epsilon if config.epsilon else 0.1
        self.noise_type = 'gaussian'  # Options: 'gaussian', 'uniform', 'targeted'

    def poison_dataset(self, X: np.ndarray, y: np.ndarray, client_id: int = 0, model: Optional[nn.Module] = None):
        """
        For noise-based model poisoning, we don't modify the dataset.
        """
        n_samples = len(y)
        n_poison = int(n_samples * self.config.data_poisoning_rate)

        self.poison_mask = np.zeros(n_samples, dtype=bool)
        poison_indices = np.random.choice(n_samples, n_poison, replace=False)
        self.poison_mask[poison_indices] = True

        return X, y, self.poison_mask

    def poison_model(self, model: nn.Module) -> nn.Module:
        """
        Add noise to model parameters.
        """
        with torch.no_grad():
            if self.noise_type == 'gaussian':
                # Add Gaussian noise to each parameter
                for param in model.parameters():
                    noise = torch.randn_like(param) * self.noise_scale
                    param.data += noise

            elif self.noise_type == 'uniform':
                # Add uniform noise
                for param in model.parameters():
                    noise = (torch.rand_like(param) - 0.5) * 2 * self.noise_scale
                    param.data += noise

            elif self.noise_type == 'targeted':
                # Add noise proportional to parameter magnitude (targets large weights)
                for param in model.parameters():
                    noise = torch.randn_like(param) * param.abs() * self.noise_scale
                    param.data += noise

        return model

    def evaluate_attack_success(self, model: nn.Module, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Measure attack success by accuracy degradation.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(device)
            y_tensor = torch.LongTensor(y_test)

            outputs = model(X_tensor)
            predictions = torch.argmax(outputs.cpu(), dim=1)

            correct = (predictions == y_tensor).sum().item()
            total = len(y_tensor)

        accuracy = correct / total if total > 0 else 0.0
        attack_success_rate = 1.0 - accuracy

        return attack_success_rate


class GlobalModelReplacementAttack(BaseAttack):
    """
    Global Model Poisoning Attack (FIXED: removed Byzantine boosting)

    Attacker trains on poisoned data (label flipping) to degrade global model.
    FIXED: No longer uses parameter boosting (Byzantine strategy removed).

    Strategy:
    - Poison training data by flipping labels to target class
    - Train model normally on poisoned data
    - Submit trained model WITHOUT boosting (realistic FL attack)

    Note: Relies on data poisoning alone for 15-30% accuracy impact
    """
    POISONS_DATA  = True    # flips labels in poison_dataset()
    POISONS_MODEL = False   # poison_model() is a documented no-op

    def __init__(self, config: AttackConfig):
        super().__init__(config)
        self.poison_mask = None
        # REMOVED: boost_factor (Byzantine attack strategy removed)

    def poison_dataset(self, X: np.ndarray, y: np.ndarray, client_id: int = 0, model: Optional[nn.Module] = None):
        """
        Poison a subset of the training data with target labels.
        """
        n_samples = len(y)
        n_poison = int(n_samples * self.config.data_poisoning_rate)

        self.poison_mask = np.zeros(n_samples, dtype=bool)

        if n_poison == 0:
            return X, y, self.poison_mask

        # Randomly select samples to poison
        poison_indices = np.random.choice(n_samples, n_poison, replace=False)
        self.poison_mask[poison_indices] = True

        # Flip labels to target class
        y_poisoned = y.copy()
        target = self.config.target_class

        if isinstance(target, (list, tuple)):
            # If multiple targets, randomly assign
            for idx in poison_indices:
                y_poisoned[idx] = np.random.choice(target)
        else:
            y_poisoned[poison_indices] = target

        return X, y_poisoned, self.poison_mask

    def poison_model(self, model: nn.Module, boost_factor: Optional[float] = None) -> nn.Module:
        """
        FIXED: No model-level poisoning needed - data poisoning is sufficient.
        Removed Byzantine boosting strategy for realistic FL attack.
        """
        # FIXED: Return model unchanged - attack relies on data poisoning alone
        # This makes it a realistic model poisoning attack (15-30% impact)
        # instead of Byzantine attack (50-70% impact)
        return model

    def evaluate_attack_success(self, model: nn.Module, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Measure attack success based on target class prediction.
        """
        if self.poison_mask is None or len(self.poison_mask) == 0:
            return 0.0

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(device)
            outputs = model(X_tensor)
            preds = torch.argmax(outputs, dim=1).cpu()

        y_tensor = torch.LongTensor(y_test).cpu()

        # Use the base class compute_asr method
        asr = self.compute_asr(
            y_true=y_tensor,
            y_pred=preds,
            poison_mask=self.poison_mask[:len(y_tensor)] if len(self.poison_mask) >= len(y_tensor) else None,
            source_class=self.config.source_class,
            target_class=self.config.target_class
        )

        return asr


class AggregationModificationAttack(BaseAttack):
    """
    Aggregation Process Modification Attack

    Attacker sends updates designed to manipulate the aggregation process.
    This can include:
    - Sending outlier updates to poison robust aggregation
    - Coordinated attacks with other malicious clients
    - Adaptive attacks that estimate benign updates

    Note: This attack requires coordination at the aggregation level,
    so it works in conjunction with model poisoning.
    """
    POISONS_DATA  = True    # flips labels in poison_dataset()
    POISONS_MODEL = True    # manipulate_update() perturbs the submitted update

    def __init__(self, config: AttackConfig):
        super().__init__(config)
        self.poison_mask = None
        self.manipulation_type = 'outlier'  # Options: 'outlier', 'coordinated', 'adaptive'
        self.perturbation_scale = config.epsilon if config.epsilon else 5.0

    def poison_dataset(self, X: np.ndarray, y: np.ndarray, client_id: int = 0, model: Optional[nn.Module] = None):
        """
        Poison training data for aggregation manipulation.
        """
        n_samples = len(y)
        n_poison = int(n_samples * self.config.data_poisoning_rate)

        self.poison_mask = np.zeros(n_samples, dtype=bool)

        if n_poison == 0:
            return X, y, self.poison_mask

        poison_indices = np.random.choice(n_samples, n_poison, replace=False)
        self.poison_mask[poison_indices] = True

        # Flip labels
        y_poisoned = y.copy()
        target = self.config.target_class

        if isinstance(target, (list, tuple)):
            for idx in poison_indices:
                y_poisoned[idx] = np.random.choice(target)
        else:
            y_poisoned[poison_indices] = target

        return X, y_poisoned, self.poison_mask

    def manipulate_update(self, model: nn.Module, benign_updates: Optional[list] = None) -> nn.Module:
        """
        Manipulate model update to evade aggregation defenses.
        """
        with torch.no_grad():
            if self.manipulation_type == 'outlier':
                # Create outlier updates by adding large perturbations
                for param in model.parameters():
                    perturbation = torch.randn_like(param) * self.perturbation_scale
                    param.data += perturbation

            elif self.manipulation_type == 'coordinated':
                # All malicious clients send identical updates (amplifies effect)
                # This would require coordination between clients
                # For now, just scale up the update
                for param in model.parameters():
                    param.data *= 2.0

            elif self.manipulation_type == 'adaptive':
                # Estimate benign updates and craft adversarial update
                if benign_updates is not None and len(benign_updates) > 0:
                    # Calculate mean of benign updates
                    for i, param in enumerate(model.parameters()):
                        benign_mean = torch.stack([u[i].data for u in benign_updates]).mean(dim=0)
                        # Move in opposite direction
                        param.data = benign_mean - (param.data - benign_mean) * self.perturbation_scale
                else:
                    # Fallback to outlier strategy
                    for param in model.parameters():
                        perturbation = torch.randn_like(param) * self.perturbation_scale
                        param.data += perturbation

        return model

    def evaluate_attack_success(self, model: nn.Module, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Measure attack success.
        """
        if self.poison_mask is None or len(self.poison_mask) == 0:
            return 0.0

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(device)
            outputs = model(X_tensor)
            preds = torch.argmax(outputs, dim=1).cpu()

        y_tensor = torch.LongTensor(y_test).cpu()

        asr = self.compute_asr(
            y_true=y_tensor,
            y_pred=preds,
            poison_mask=self.poison_mask[:len(y_tensor)] if len(self.poison_mask) >= len(y_tensor) else None,
            source_class=self.config.source_class,
            target_class=self.config.target_class
        )

        return asr
