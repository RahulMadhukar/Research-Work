from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

@dataclass
class AttackConfig:
    def __init__(self, attack_type, poisoning_rate=0.0, poison_percentage=None, source_class=None, target_class=None,
                 num_malicious_clients=1, trigger_size=None, trigger_intensity=None, epsilon=None):
        self.attack_type = attack_type
        # poison_percentage: percentage of data to poison (20%, 40%, 60%, 80%)
        # poisoning_rate: kept for backwards compatibility (same as poison_percentage)
        self.poison_percentage = poison_percentage if poison_percentage is not None else poisoning_rate
        self.poisoning_rate = self.poison_percentage  # For backwards compatibility
        self.source_class = source_class
        self.target_class = target_class
        self.num_malicious_clients = num_malicious_clients
        self.trigger_size = trigger_size
        self.trigger_intensity = trigger_intensity
        self.epsilon = epsilon

class BaseAttack(ABC):
    def __init__(self, config: AttackConfig):
        self.config = config

    @abstractmethod
    def poison_dataset(self, X: np.ndarray, y: np.ndarray, client_id: int = 0, model: Optional[nn.Module] = None):
        pass

    def attack_summary(self) -> Dict:
        """Generic summary for all attacks"""
        return {
            "attack_type": self.__class__.__name__,
            "poisoning_rate": getattr(self.config, "poisoning_rate", None),
        }

    @abstractmethod
    def evaluate_attack_success(self, model: nn.Module, X_test: np.ndarray, y_test: np.ndarray) -> float:
        pass

    def compute_asr(self,
                    y_true: torch.Tensor,
                    y_pred: torch.Tensor,
                    *,
                    poison_mask: Optional[torch.Tensor] = None,
                    source_class: Optional[object] = None,
                    target_class: Optional[object] = None) -> float:
        """
        Compute Attack Success Rate (ASR).

        Returns
        -------
        float
            ASR in [0,1]
        """
        import torch
        import numpy as np

        # Convert numpy to tensor if needed
        if isinstance(y_true, np.ndarray):
            y_true = torch.from_numpy(y_true)
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.from_numpy(y_pred)

        y_true = y_true.detach().cpu().long()
        y_pred = y_pred.detach().cpu().long()

        # Determine mask
        if poison_mask is not None:
            if not isinstance(poison_mask, torch.Tensor):
                poison_mask = torch.tensor(poison_mask, dtype=torch.bool)
            mask = poison_mask.detach().cpu().bool()

            if mask.numel() != y_true.numel():
                raise ValueError("poison_mask length must equal number of samples")
        elif source_class is not None:
            if isinstance(source_class, (list, tuple, np.ndarray)):
                sc_tensor = torch.tensor(list(source_class), dtype=torch.long)
                mask = (y_true.unsqueeze(1) == sc_tensor.unsqueeze(0)).any(dim=1)
            else:
                mask = (y_true == int(source_class))
        else:
            mask = torch.ones_like(y_true, dtype=torch.bool)

        # Determine target(s)
        if target_class is None:
            target_attr = getattr(self, "target_class", None)
            if target_attr is None:
                raise ValueError("target_class must be provided (either as arg or self.target_class)")
            target_class = target_attr

        # support multi-target
        if isinstance(target_class, (list, tuple, np.ndarray)):
            tgt_tensor = torch.tensor(list(target_class), dtype=torch.long)
            success = (y_pred.unsqueeze(1) == tgt_tensor.unsqueeze(0)).any(dim=1)
        else:
            success = (y_pred == int(target_class))

        # Convert boolean mask & success to float for computation
        mask_float = mask.to(torch.float)
        success_float = success.to(torch.float)

        # Only consider masked positions
        masked_success = success_float * mask_float
        num_masked = mask_float.sum().item()
        if num_masked == 0:
            return 0.0

        asr = masked_success.sum().item() / num_masked
        return asr


    def evaluate_attack_success(self, model: nn.Module, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Default evaluation using compute_asr"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(device)
            outputs = model(X_tensor)
            preds = torch.argmax(outputs, dim=1).cpu()

        y_tensor = torch.LongTensor(y_test).cpu()
        asr = self.compute_asr(y_true=y_tensor, 
                        y_pred=preds,
                        poison_mask=self.poison_mask[:len(y_tensor)],
                        source_class=self.config.source_class,
                        target_class=self.config.target_class)
        return asr
