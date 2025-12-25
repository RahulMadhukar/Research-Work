from attacks.base import AttackConfig, BaseAttack
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from typing import Optional, Tuple, Dict
import torch.nn.functional as F    

# =============================================================================
# LABEL-FLIPPING ATTACKS
# =============================================================================

class StaticLabelFlipping(BaseAttack):
    """
    Multi-Class Static Label Flipping (SLF) Attack Implementation
    
    Mathematical formulation:
    - PDR = |D^t_{A_i}| / |D^t_i|
    - Poisoned set: D^t_{A_i} = {(x, ỹ=c_T) : x ∈ S}
    - Local objective: L^t_i = arg min [α F_i(L^t_i, D^t_i) + (1-α) F_{A_i}(L^t_i, D^t_{A_i})]
    
    Supports flipping multiple source classes into multiple target classes.
    Example mapping:
        source_classes = [0, 2, 4]
        target_classes = [1, 3, 5]
    Each source[i] → target[i]
    """

    def __init__(self, config: AttackConfig):
        super().__init__(config)
        
        # Allow both single and multiple classes
        if isinstance(config.source_class, (list, tuple)):
            self.source_classes = list(config.source_class)
        else:
            self.source_classes = [config.source_class if config.source_class is not None else 1]
        
        if isinstance(config.target_class, (list, tuple)):
            self.target_classes = list(config.target_class)
        else:
            self.target_classes = [config.target_class]
        
        # Safety check: ensure lengths match
        if len(self.target_classes) != len(self.source_classes):
            raise ValueError("Length of source_classes and target_classes must match.")
        
        # Important: assign for compute_asr
        self.source_class = self.source_classes
        self.target_class = self.target_classes
    
    def _select_by_feature_similarity(self, model, X, y, source_indices, target_class, n_poison, device):
        """Select samples closest to target class centroid in feature space"""
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X[source_indices]).to(device)
            
            # Get features using various methods
            if hasattr(model, "get_features"):
                feats = model.get_features(X_tensor).cpu().numpy()
            elif hasattr(model, "forward_features"):
                feats = model.forward_features(X_tensor).cpu().numpy()
            else:
                # Fallback: use logits as features
                feats = model(X_tensor).cpu().numpy()

        # Compute target centroid from available target-class examples
        target_idxs_in_client = np.where(y == target_class)[0]
        if len(target_idxs_in_client) == 0:
            # Fallback: random selection if no target samples present locally
            return np.random.choice(source_indices, n_poison, replace=False)

        # Compute centroid using embeddings of target samples
        with torch.no_grad():
            X_t = torch.FloatTensor(X[target_idxs_in_client]).to(device)
            if hasattr(model, "get_features"):
                tfeats = model.get_features(X_t).cpu().numpy()
            elif hasattr(model, "forward_features"):
                tfeats = model.forward_features(X_t).cpu().numpy()
            else:
                tfeats = model(X_t).cpu().numpy()

        centroid = tfeats.mean(axis=0)
        
        # Compute distances and select closest samples
        dists = np.linalg.norm(feats - centroid[None, :], axis=1)
        order = np.argsort(dists)
        selected = np.array(source_indices)[order[:n_poison]]
        return selected

    def _select_by_loss_if_relabelled(self, model, X, y, source_indices, target_class, n_poison, device):
        """Select samples with lowest loss when relabeled to target class"""
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X[source_indices]).to(device)
            outputs = model(X_tensor)
            
            # Create fake labels = target_class
            labels = torch.tensor([target_class] * len(source_indices)).to(device)
            losses = F.cross_entropy(outputs, labels, reduction='none').cpu().numpy()

        # Choose samples with smallest loss (easiest to absorb as target)
        order = np.argsort(losses)
        selected = np.array(source_indices)[order[:n_poison]]
        return selected

    def poison_dataset(self, X: np.ndarray, y: np.ndarray, 
                      client_id: int = 0, model: Optional[nn.Module] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Poison dataset by flipping labels from source to target classes.
        
        Args:
            X: Input features
            y: Labels
            client_id: Client identifier
            model: Optional model for intelligent sample selection
            
        Returns:
            Tuple of (poisoned_X, poisoned_y, poison_mask)
        """
        X_poisoned = X.copy()
        y_poisoned = y.copy()
        poison_mask = np.zeros(len(X), dtype=bool)
        total_samples = len(X)
        total_poisoned = 0
        
        # Get device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model is not None:
            model.to(device)

        for source_class, target_class in zip(self.source_classes, self.target_classes):
            source_indices = np.where(y == source_class)[0]

            if len(source_indices) == 0:
                print(f"  [SLF] No samples of class {source_class} in client {client_id}")
                continue

            # FIXED: Poison X% of EACH source class (not divided by number of classes)
            # This makes the attack much more aggressive
            #n_poison = int(self.config.poisoning_rate * len(source_indices))

            n_poison = int(self.config.poisoning_rate * total_samples / len(self.source_classes))
            n_poison = min(n_poison, len(source_indices))

            # Choose strategy: 'random', 'feature', 'loss'
            strategy = getattr(self.config, "slf_selection_strategy", "feature")

            if n_poison > 0:
                if strategy == "random" or model is None:
                    poison_indices = np.random.choice(source_indices, n_poison, replace=False)
                elif strategy == "feature":
                    poison_indices = self._select_by_feature_similarity(
                        model, X, y, source_indices, target_class, n_poison, device
                    )
                elif strategy == "loss":
                    poison_indices = self._select_by_loss_if_relabelled(
                        model, X, y, source_indices, target_class, n_poison, device
                    )
                else:
                    poison_indices = np.random.choice(source_indices, n_poison, replace=False)
                
                # Flip labels
                y_poisoned[poison_indices] = target_class
                poison_mask[poison_indices] = True
                total_poisoned += n_poison

                print(f"  [SLF] Client {client_id}: Poisoned {n_poison}/{len(source_indices)} "
                      f"({n_poison/len(source_indices):.2%}) from class {source_class} → {target_class}")

        if total_poisoned == 0:
            print(f"  [SLF] Client {client_id}: No samples poisoned")
        else:
            print(f"  [SLF] Client {client_id}: Total corrupted = {total_poisoned}/{total_samples} ({total_poisoned/total_samples:.1%})")

        return X_poisoned, y_poisoned, poison_mask

    def evaluate_attack_success(self, model: nn.Module, X_test: np.ndarray, 
                               y_test: np.ndarray) -> float:
        """Evaluate average attack success rate across all source-target pairs"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        success_rates = []

        for source_class, target_class in zip(self.source_classes, self.target_classes):
            source_mask = y_test == source_class
            if not np.any(source_mask):
                continue

            X_source = X_test[source_mask]
            X_tensor = torch.FloatTensor(X_source).to(device)

            with torch.no_grad():
                outputs = model(X_tensor)
                y_pred = torch.argmax(outputs, dim=1)

                # Ensure y_pred is a tensor
                if not isinstance(y_pred, torch.Tensor):
                    y_pred = torch.tensor([y_pred], device=device)

                # Ensure target_class is in the correct format
                if isinstance(target_class, (list, tuple, np.ndarray)):
                    # Multi-class target — check if predicted label in any target
                    target_tensor = torch.tensor(target_class, device=y_pred.device, dtype=torch.long)
                    success_mask = torch.isin(y_pred, target_tensor)
                else:
                    # Single target class
                    target_tensor = torch.full_like(y_pred, fill_value=int(target_class), dtype=torch.long)
                    success_mask = (y_pred == target_tensor)

                # Ensure success_mask is tensor (not bool)
                if not isinstance(success_mask, torch.Tensor):
                    success_mask = torch.tensor(success_mask, device=device, dtype=torch.bool)

                # Compute success rate safely
                success_rate = success_mask.float().mean().item()
                success_rates.append(success_rate)

        # Return mean success rate
        if success_rates:
            return float(np.mean(success_rates))
        else:
            return 0.0
        
    def attack_summary(self) -> Dict:
        """Return summary of attack configuration"""
        base_summary = super().attack_summary()
        base_summary.update({
            "source_classes": self.source_classes,
            "target_classes": self.target_classes,
            "num_class_mappings": len(self.source_classes)
        })
        return base_summary


class DynamicLabelFlipping(BaseAttack):
    """
    Dynamic Label Flipping (DLF) Attack Implementation (Enhanced)

    Combines multiple signals for more intensive attacks:
    - Feature centroids: μ_c = (1/|D_{mal}(c)|) Σ φ(x) for x ∈ D_{mal}(c)
    - Attack distance: AD(c,c') = ||μ_c - μ_c'||²₂
    - Sample selection: combines
        1) Feature distance to target centroid
        2) Loss-if-relabelled
        3) Optional gradient alignment
    """
    
    def __init__(self, config: AttackConfig):
        super().__init__(config)
        self.class_centroids = {}
        self.target_mapping = {}
        self.attack_distances = {}

        # FIXED: Don't set source_class/target_class here - they're empty!
        # These will be set in poison_dataset() after computing target_mapping
        self.source_class = []
        self.target_class = []


    def compute_class_centroids(self, model: nn.Module, X: np.ndarray, y: np.ndarray) -> Dict[int, np.ndarray]:
        """Compute feature centroids for each class"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        centroids = {}
        for class_id in np.unique(y):
            class_mask = y == class_id
            if np.sum(class_mask) > 0:
                X_tensor = torch.FloatTensor(X[class_mask]).to(device)
                
                # Get features
                with torch.no_grad():
                    if hasattr(model, 'get_features'):
                        features = model.get_features(X_tensor).cpu().numpy()
                    else:
                        features = model(X_tensor).cpu().numpy()
                
                centroids[int(class_id)] = np.mean(features, axis=0)
        
        return centroids

    def compute_attack_distances(self, centroids: Dict[int, np.ndarray]) -> Dict[Tuple[int,int], float]:
        """Compute pairwise attack distances between class centroids"""
        attack_distances = {}
        classes = list(centroids.keys())
        
        for c1 in classes:
            for c2 in classes:
                if c1 != c2:
                    attack_distances[(c1, c2)] = np.sum((centroids[c1] - centroids[c2])**2)
        
        return attack_distances

    def select_dynamic_targets(self, centroids: Dict[int, np.ndarray]) -> Dict[int, int]:
        """Select target class for each source class based on minimum attack distance"""
        attack_distances = self.compute_attack_distances(centroids)
        target_mapping = {}
        
        for source_class in centroids.keys():
            min_dist = float('inf')
            best_target = None
            
            for target_class in centroids.keys():
                if source_class != target_class:
                    dist = attack_distances[(source_class, target_class)]
                    if dist < min_dist:
                        min_dist = dist
                        best_target = target_class
            
            if best_target is not None:
                target_mapping[source_class] = best_target
                print(f"  [DLF] Class {source_class} → Class {best_target} (AD: {min_dist:.4f})")
        
        return target_mapping
    
   
    def _normalize_vector(self, arr):
        """Min-max normalize array to [0, 1]"""
        arr = np.array(arr, dtype=float)
        mi, ma = arr.min(), arr.max()
        if ma - mi < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mi) / (ma - mi)

    def _compute_logits_and_features(self, model, X, device, batch_size=256):
        """Compute logits and features in batches"""
        model.eval()
        logits_list, feats_list = [], []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.FloatTensor(X[i:i+batch_size]).to(device)
                out = model(batch)
                logits_list.append(out.cpu())
                
                if hasattr(model, "get_features"):
                    feats = model.get_features(batch).cpu()
                else:
                    feats = out.cpu()  # Fallback
                feats_list.append(feats)
        
        return torch.cat(logits_list, dim=0), torch.cat(feats_list, dim=0)

    def _select_combined(self, model, X, y, source_indices, target_class, n_poison, device, use_grad=False):
        """Select samples using combined scoring (loss + feature + optional gradient)"""
        if len(source_indices) == 0 or n_poison <= 0:
            return np.array([], dtype=int)

        # Compute logits & features
        logits_all, feats_all = self._compute_logits_and_features(model, X, device)
        logits_src = logits_all[source_indices].numpy()
        feats_src = feats_all[source_indices].numpy()
        target_centroid = self.class_centroids[target_class]

        # 1) Loss-if-relabelled score
        losses = []
        with torch.no_grad():
            for i_local in range(len(source_indices)):
                logit = torch.FloatTensor(logits_src[i_local:i_local+1]).to(device)
                label = torch.tensor([target_class]).to(device)
                loss = F.cross_entropy(logit, label, reduction='none').item()
                losses.append(loss)
        score_loss = 1.0 - self._normalize_vector(losses)

        # 2) Feature distance score
        dists = np.sum((feats_src - target_centroid)**2, axis=1)
        score_feat = 1.0 - self._normalize_vector(dists)

        # 3) Gradient alignment score (optional)
        score_grad = np.zeros_like(score_loss)
        if use_grad:
            model.eval()
            grad_scores = []
            for idx in source_indices:
                x_tensor = torch.FloatTensor(X[idx:idx+1]).to(device)
                label = torch.tensor([target_class]).to(device)
                model.zero_grad()
                out = model(x_tensor)
                loss = F.cross_entropy(out, label)
                loss.backward()
                
                grads = []
                for p in model.parameters():
                    if p.grad is not None:
                        grads.append(p.grad.detach().cpu().numpy().ravel())
                grad_scores.append(-np.linalg.norm(np.concatenate(grads)) if grads else 0.0)
            score_grad = self._normalize_vector(np.array(grad_scores))

        # Combine scores with configurable weights
        weights = getattr(self.config, "dlf_weights", {"loss": 0.45, "feature": 0.45, "grad": 0.1})
        composite = (weights.get("loss", 0.45) * score_loss + 
                    weights.get("feature", 0.45) * score_feat + 
                    weights.get("grad", 0.1) * score_grad)

        # Select top-k by composite score
        order = np.argsort(-composite)
        return np.array(source_indices)[order[:n_poison]]

    def poison_dataset(self, X: np.ndarray, y: np.ndarray, 
                      client_id: int = 0, model: Optional[nn.Module] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Poison dataset using dynamic label flipping with intelligent sample selection.
        
        Args:
            X: Input features
            y: Labels
            client_id: Client identifier
            model: Model for feature extraction (required)
            
        Returns:
            Tuple of (poisoned_X, poisoned_y, poison_mask)
        """
        if model is None:
            raise ValueError("DLF requires a model for feature extraction")

        X_poisoned = X.copy()
        y_poisoned = y.copy()
        
        # Compute centroids and select dynamic targets
        self.class_centroids = self.compute_class_centroids(model, X, y)
        self.target_mapping = self.select_dynamic_targets(self.class_centroids)

        # FIXED: Set source_class and target_class AFTER computing target_mapping
        self.source_class = list(self.target_mapping.keys())
        self.target_class = list(self.target_mapping.values())

        poison_mask = np.zeros(len(X), dtype=bool)
        total_poisoned = 0

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        for source_class, target_class in self.target_mapping.items():
            source_indices = np.where(y == source_class)[0]
            n_poison = int(self.config.poisoning_rate * len(source_indices))

            if n_poison <= 0:
                continue

            use_grad = getattr(self.config, "dlf_use_grad", False)
            poison_indices = self._select_combined(
                model, X, y, source_indices, target_class, n_poison, device, use_grad
            )

            # Apply label flips
            y_poisoned[poison_indices] = target_class
            poison_mask[poison_indices] = True
            total_poisoned += len(poison_indices)

            print(f"  [DLF] Client {client_id}: Poisoned {len(poison_indices)}/{len(source_indices)} "
                  f"({len(poison_indices)/len(source_indices):.2%}) from class {source_class} → {target_class}")

        print(f"  [DLF] Client {client_id}: Total poisoned = {total_poisoned}/{len(X)} ({total_poisoned/len(X):.1%}) using dynamic targeting")

        return X_poisoned, y_poisoned, poison_mask

    def evaluate_attack_success(self, model: nn.Module, X_test: np.ndarray, 
                               y_test: np.ndarray) -> float:
        """Evaluate average attack success rate across all source-target pairs"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        success_rates = []

        for source_class, target_class in zip(self.source_class, self.target_class):
            source_mask = y_test == source_class
            if not np.any(source_mask):
                continue

            X_source = X_test[source_mask]
            X_tensor = torch.FloatTensor(X_source).to(device)

            with torch.no_grad():
                outputs = model(X_tensor)
                y_pred = torch.argmax(outputs, dim=1)

                # Ensure y_pred is a tensor
                if not isinstance(y_pred, torch.Tensor):
                    y_pred = torch.tensor([y_pred], device=device)

                # Ensure target_class is in the correct format
                if isinstance(target_class, (list, tuple, np.ndarray)):
                    # Multi-class target — check if predicted label in any target
                    target_tensor = torch.tensor(target_class, device=y_pred.device, dtype=torch.long)
                    success_mask = torch.isin(y_pred, target_tensor)
                else:
                    # Single target class
                    target_tensor = torch.full_like(y_pred, fill_value=int(target_class), dtype=torch.long)
                    success_mask = (y_pred == target_tensor)

                # Ensure success_mask is tensor (not bool)
                if not isinstance(success_mask, torch.Tensor):
                    success_mask = torch.tensor(success_mask, device=device, dtype=torch.bool)

                # Compute success rate safely
                success_rate = success_mask.float().mean().item()
                success_rates.append(success_rate)

        # Return mean success rate
        if success_rates:
            return float(np.mean(success_rates))
        else:
            return 0.0
    
    def attack_summary(self) -> Dict:
        """Return summary of attack configuration"""
        base_summary = super().attack_summary()
        base_summary.update({
            "target_mapping": self.target_mapping,
            "num_class_mappings": len(self.target_mapping),
            "attack_distances": {str(k): float(v) for k, v in self.attack_distances.items()} if self.attack_distances else {}
        })
        return base_summary
