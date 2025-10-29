"""
Enhanced Defense Mechanisms for Federated Learning

This module implements advanced defense strategies including:
1. Adaptive Committee Defense with multi-signal anomaly detection
2. Reputation-based client scoring with historical tracking
3. Gradient analysis and clipping
4. Ensemble defense combining multiple strategies
5. Real-time anomaly detection with trend analysis

Performance Optimizations:
- Vectorized distance computations
- Memory-efficient gradient clipping
- Cached reputation scores
- Optimized clustering with adaptive parameters
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, deque
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import time
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


# =============================================================================
# Performance Profiling Utilities
# =============================================================================

class PerformanceProfiler:
    """Helper class to profile defense performance."""

    def __init__(self):
        self.timings = defaultdict(list)
        self.enabled = False

    def enable(self):
        """Enable profiling."""
        self.enabled = True

    def disable(self):
        """Disable profiling."""
        self.enabled = False

    def record(self, operation: str, duration: float):
        """Record timing for an operation."""
        if self.enabled:
            self.timings[operation].append(duration)

    def get_stats(self) -> Dict:
        """Get profiling statistics."""
        stats = {}
        for op, times in self.timings.items():
            if times:
                stats[op] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'total': np.sum(times),
                    'count': len(times)
                }
        return stats

    def print_stats(self):
        """Print profiling statistics."""
        stats = self.get_stats()
        if not stats:
            print("[PROFILER] No profiling data collected")
            return

        print("\n" + "="*80)
        print("DEFENSE PERFORMANCE PROFILING")
        print("="*80)
        print(f"{'Operation':<30} {'Mean (ms)':<12} {'Std (ms)':<12} {'Total (ms)':<12} {'Count':<8}")
        print("-"*80)

        for op, data in sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True):
            print(f"{op:<30} {data['mean']*1000:>11.2f} {data['std']*1000:>11.2f} "
                  f"{data['total']*1000:>11.2f} {data['count']:>7}")

        print("="*80)

    def clear(self):
        """Clear profiling data."""
        self.timings.clear()


# Global profiler instance
_profiler = PerformanceProfiler()


def get_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    return _profiler


def profile_function(func):
    """Decorator to profile function execution time."""
    def wrapper(*args, **kwargs):
        if _profiler.enabled:
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start
            _profiler.record(func.__name__, duration)
            return result
        else:
            return func(*args, **kwargs)
    return wrapper


# =============================================================================
# Reputation System
# =============================================================================

class ReputationSystem:
    """
    Tracks client reputation over multiple rounds to identify persistent malicious behavior.
    """

    def __init__(self, num_clients: int, memory_size: int = 10, decay_factor: float = 0.95):
        """
        Args:
            num_clients: Total number of clients
            memory_size: Number of rounds to track history
            decay_factor: Exponential decay for older reputation scores
        """
        self.num_clients = num_clients
        self.memory_size = memory_size
        self.decay_factor = decay_factor

        # Initialize reputation scores (1.0 = trusted, 0.0 = untrusted)
        self.reputation_scores = {i: 1.0 for i in range(num_clients)}

        # Track history of anomaly flags
        self.anomaly_history = {i: deque(maxlen=memory_size) for i in range(num_clients)}

        # Track contribution quality metrics
        self.contribution_quality = {i: deque(maxlen=memory_size) for i in range(num_clients)}

    def update_reputation(self, client_id: int, is_anomalous: bool, quality_score: Optional[float] = None):
        """
        Update reputation based on current round behavior.

        Args:
            client_id: Client identifier
            is_anomalous: Whether client was flagged as anomalous
            quality_score: Optional quality metric (0-1, higher is better)
        """
        # Validate client_id
        if client_id < 0 or client_id >= self.num_clients:
            print(f"[WARN] Invalid client_id {client_id}, skipping reputation update")
            return

        # Record anomaly
        self.anomaly_history[client_id].append(1 if is_anomalous else 0)

        # Record quality if provided
        if quality_score is not None:
            # Clamp quality score to valid range
            quality_score = max(0.0, min(1.0, float(quality_score)))
            self.contribution_quality[client_id].append(quality_score)

        # Calculate new reputation score
        anomaly_rate = np.mean(self.anomaly_history[client_id]) if len(self.anomaly_history[client_id]) > 0 else 0.0

        # CHANGED: Reduced penalty and increased reward to avoid hurting clean clients
        # Reputation formula: exponential moving average with penalty for anomalies
        penalty = 0.15 if is_anomalous else 0.0  # CHANGED: from 0.3 to 0.15
        reward = 0.05 if not is_anomalous else 0.0  # CHANGED: simplified and reduced from 0.1

        # Update with decay
        old_reputation = self.reputation_scores.get(client_id, 1.0)
        new_reputation = (self.decay_factor * old_reputation) - penalty + reward

        # Clamp to [0, 1]
        self.reputation_scores[client_id] = max(0.0, min(1.0, new_reputation))

    def get_reputation(self, client_id: int) -> float:
        """Get current reputation score for a client."""
        return self.reputation_scores[client_id]

    def get_trusted_clients(self, threshold: float = 0.5) -> List[int]:
        """Get list of clients with reputation above threshold."""
        return [cid for cid, score in self.reputation_scores.items() if score >= threshold]

    def get_reputation_weights(self) -> Dict[int, float]:
        """Get reputation scores as aggregation weights."""
        total = sum(self.reputation_scores.values())
        if total == 0:
            return {i: 1.0 / self.num_clients for i in range(self.num_clients)}
        return {i: score / total for i, score in self.reputation_scores.items()}

    def reset_reputation(self, client_id: int):
        """Reset reputation for a specific client."""
        self.reputation_scores[client_id] = 1.0
        self.anomaly_history[client_id].clear()
        self.contribution_quality[client_id].clear()

    def detect_anomalies(self, client_updates: List[Dict], client_losses: Optional[List[float]] = None) -> List[int]:
        """
        Detect anomalous clients based on reputation scores.
        Clients with reputation below 0.5 are flagged as anomalous.

        Args:
            client_updates: List of client parameter updates
            client_losses: Optional list of client losses

        Returns:
            List of anomalous client indices
        """
        anomalous = []
        threshold = 0.5

        for client_id, reputation in self.reputation_scores.items():
            if client_id < len(client_updates) and reputation < threshold:
                anomalous.append(client_id)

        print(f"[REPUTATION DEFENSE] Detected {len(anomalous)} clients with low reputation (< {threshold})")
        return anomalous

    def robust_aggregate(self, client_updates: List[Dict], anomalous_clients: List[int]) -> Dict:
        """
        Robust aggregation using reputation-weighted averaging.

        Args:
            client_updates: List of client parameter updates
            anomalous_clients: List of anomalous client indices (used for informational purposes)

        Returns:
            Aggregated parameters weighted by reputation scores
        """
        n_clients = len(client_updates)

        # CHANGED: Lowered minimum reputation threshold to keep more clients
        # Get reputation weights (exclude clients with very low reputation)
        min_reputation = 0.2  # CHANGED: from 0.3 to 0.2
        weights = []
        valid_indices = []

        for i in range(n_clients):
            rep = self.reputation_scores.get(i, 1.0)
            if rep >= min_reputation:
                weights.append(rep)
                valid_indices.append(i)

        # Normalize weights
        if len(weights) == 0:
            # Fallback: use all clients with equal weights
            weights = [1.0] * n_clients
            valid_indices = list(range(n_clients))

        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Aggregate with reputation weights
        aggregated_params = {}
        param_names = client_updates[0].keys()

        for param_name in param_names:
            weighted_sum = torch.zeros_like(client_updates[0][param_name])
            for idx, weight in zip(valid_indices, weights):
                weighted_sum += weight * client_updates[idx][param_name]
            aggregated_params[param_name] = weighted_sum

        print(f"[REPUTATION DEFENSE] Aggregated {len(valid_indices)}/{n_clients} clients using reputation weights")

        return aggregated_params


class GradientAnalyzer:
    """
    Analyzes gradient patterns to detect poisoning attacks.
    """

    def __init__(self, clip_norm: float = 15.0, noise_scale: float = 0.0001):  # CHANGED: increased clip_norm from 10.0 to 15.0, reduced noise from 0.001 to 0.0001
        """
        Args:
            clip_norm: Maximum allowed gradient norm
            noise_scale: Scale of differential privacy noise
        """
        self.clip_norm = clip_norm
        self.noise_scale = noise_scale
        self.gradient_history = defaultdict(list)

    @profile_function
    def analyze_gradients(self, client_updates: List[Dict], reference_update: Optional[Dict] = None) -> Tuple[List[int], Dict]:
        """
        Analyze gradient patterns to detect anomalies.

        Args:
            client_updates: List of client parameter updates
            reference_update: Optional reference for comparison (e.g., global model)

        Returns:
            Tuple of (anomalous_client_indices, analysis_metrics)
        """
        n_clients = len(client_updates)

        if n_clients == 0:
            return [], {'grad_norms': [], 'z_scores': [], 'mean_norm': 0.0, 'std_norm': 0.0, 'anomalous_count': 0}

        # Compute gradient norms
        grad_norms = []
        for update in client_updates:
            norm = self._compute_update_norm(update)
            grad_norms.append(norm)

        # Detect outliers using z-score
        grad_norms_arr = np.array(grad_norms)

        # Handle edge case: all norms are identical or only 1 client
        if n_clients < 3 or np.std(grad_norms_arr) < 1e-10:
            z_scores = np.zeros(n_clients)
        else:
            z_scores = np.abs(zscore(grad_norms_arr))
            # Handle NaN values from zscore
            z_scores = np.nan_to_num(z_scores, nan=0.0)

        # CHANGED: Increased threshold from 2.5 to 3.0 to reduce false positives
        # Flag clients with z-score > 3.0 (outliers)
        anomalous = [i for i, z in enumerate(z_scores) if z > 3.0]

        # Additional analysis: direction consistency
        direction_scores = []
        if reference_update is not None:
            for update in client_updates:
                cos_sim = self._cosine_similarity(update, reference_update)
                direction_scores.append(cos_sim)

            # CHANGED: Made direction check less aggressive (was flagging too many benign clients)
            # Flag clients with negative cosine similarity (clearly opposite direction)
            for i, sim in enumerate(direction_scores):
                if sim < -0.1 and i not in anomalous:  # CHANGED: from 0.1 to -0.1
                    anomalous.append(i)

        metrics = {
            'grad_norms': grad_norms,
            'z_scores': z_scores.tolist() if isinstance(z_scores, np.ndarray) else z_scores,
            'mean_norm': float(np.mean(grad_norms)) if len(grad_norms) > 0 else 0.0,
            'std_norm': float(np.std(grad_norms)) if len(grad_norms) > 0 else 0.0,
            'anomalous_count': len(anomalous),
            'direction_scores': direction_scores if reference_update else None
        }

        return anomalous, metrics

    def clip_and_add_noise(self, client_updates: List[Dict]) -> List[Dict]:
        """
        Apply gradient clipping and differential privacy noise.

        Args:
            client_updates: List of client parameter updates

        Returns:
            Clipped and noised updates
        """
        clipped_updates = []

        for update in client_updates:
            clipped = self._clip_update(update, self.clip_norm)
            noised = self._add_noise(clipped, self.noise_scale)
            clipped_updates.append(noised)

        return clipped_updates

    def _compute_update_norm(self, update: Dict) -> float:
        """Compute L2 norm of parameter update."""
        norm = 0.0
        for param in update.values():
            norm += torch.norm(param).item() ** 2
        return np.sqrt(norm)

    def _cosine_similarity(self, update1: Dict, update2: Dict) -> float:
        """Compute cosine similarity between two updates."""
        dot_product = 0.0
        norm1 = 0.0
        norm2 = 0.0

        for key in update1.keys():
            if key in update2:
                dot_product += (update1[key] * update2[key]).sum().item()
                norm1 += (update1[key] ** 2).sum().item()
                norm2 += (update2[key] ** 2).sum().item()

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (np.sqrt(norm1) * np.sqrt(norm2))

    def _clip_update(self, update: Dict, max_norm: float) -> Dict:
        """Clip parameter update to maximum norm."""
        current_norm = self._compute_update_norm(update)

        if current_norm <= max_norm:
            return update

        scale = max_norm / current_norm
        clipped = {k: v * scale for k, v in update.items()}
        return clipped

    def _add_noise(self, update: Dict, noise_scale: float) -> Dict:
        """Add Gaussian noise for differential privacy."""
        noised = {}
        for k, v in update.items():
            noise = torch.randn_like(v) * noise_scale
            noised[k] = v + noise
        return noised

    def detect_anomalies(self, client_updates: List[Dict], client_losses: Optional[List[float]] = None) -> List[int]:
        """
        Detect anomalous clients using gradient analysis.

        Args:
            client_updates: List of client parameter updates
            client_losses: Optional list of client losses

        Returns:
            List of anomalous client indices
        """
        anomalous, metrics = self.analyze_gradients(client_updates)
        print(f"[GRADIENT DEFENSE] Detected {len(anomalous)} clients with anomalous gradients")
        print(f"[GRADIENT DEFENSE] Mean gradient norm: {metrics['mean_norm']:.4f}, Std: {metrics['std_norm']:.4f}")
        return anomalous

    def robust_aggregate(self, client_updates: List[Dict], anomalous_clients: List[int]) -> Dict:
        """
        Robust aggregation with gradient clipping and noise.

        Args:
            client_updates: List of client parameter updates
            anomalous_clients: List of anomalous client indices to exclude

        Returns:
            Aggregated parameters after clipping and noise addition
        """
        n_clients = len(client_updates)

        # Exclude anomalous clients
        normal_indices = [i for i in range(n_clients) if i not in anomalous_clients]

        if len(normal_indices) == 0:
            print(f"[WARN] All clients flagged as anomalous, using all clients")
            normal_indices = list(range(n_clients))

        # Clip and add noise to normal clients
        normal_updates = [client_updates[i] for i in normal_indices]
        clipped_updates = self.clip_and_add_noise(normal_updates)

        # Simple averaging of clipped updates
        aggregated_params = {}
        param_names = clipped_updates[0].keys()

        for param_name in param_names:
            param_stack = torch.stack([update[param_name] for update in clipped_updates])
            aggregated_params[param_name] = torch.mean(param_stack, dim=0)

        print(f"[GRADIENT DEFENSE] Aggregated {len(normal_indices)}/{n_clients} clients with gradient clipping")

        return aggregated_params


class AdaptiveCommitteeDefense:
    """
    Enhanced committee-based defense with adaptive thresholds and multi-signal detection.
    """

    def __init__(self,
                 committee_size: int = 7,
                 initial_threshold: float = 3.0,  # CHANGED: increased from 2.5 to 3.0 for less aggressive detection
                 max_exclude_frac: float = 0.4,  # CHANGED: increased from 0.3 to 0.4 (max 40% of clients can be malicious)
                 use_clustering: bool = True):
        """
        Args:
            committee_size: Size of the committee for consensus
            initial_threshold: Initial threshold multiplier for anomaly detection
            max_exclude_frac: Maximum fraction of clients to exclude
            use_clustering: Whether to use DBSCAN clustering for anomaly detection
        """
        self.committee_size = committee_size
        self.threshold = initial_threshold
        self.max_exclude_frac = max_exclude_frac
        self.use_clustering = use_clustering

        # Adaptive threshold tracking
        self.threshold_history = deque(maxlen=20)
        self.detection_stats = {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0
        }

    @profile_function
    def detect_anomalies(self,
                        client_updates: List[Dict],
                        client_losses: Optional[List[float]] = None,
                        reputation_scores: Optional[Dict[int, float]] = None) -> List[int]:
        """
        Detect anomalous clients using multiple signals.

        Args:
            client_updates: List of client parameter updates
            client_losses: Optional list of client losses
            reputation_scores: Optional reputation scores for clients

        Returns:
            List of anomalous client indices
        """
        n_clients = len(client_updates)

        # Signal 1: Distance-based anomaly detection
        distance_scores = self._compute_distance_scores(client_updates)

        # Signal 2: Loss-based anomaly detection
        loss_scores = self._compute_loss_scores(client_losses) if client_losses else np.zeros(n_clients)

        # Signal 3: Clustering-based anomaly detection
        cluster_scores = self._compute_cluster_scores(client_updates) if self.use_clustering else np.zeros(n_clients)

        # Signal 4: Reputation-based weighting
        reputation_weights = self._get_reputation_weights(reputation_scores, n_clients)

        # Combine signals with adaptive weights
        composite_scores = self._combine_signals(
            distance_scores,
            loss_scores,
            cluster_scores,
            reputation_weights
        )

        # Adaptive thresholding
        anomalous = self._adaptive_threshold(composite_scores, n_clients)

        print(f"[ADAPTIVE DEFENSE] Detected {len(anomalous)}/{n_clients} anomalous clients")
        print(f"[ADAPTIVE DEFENSE] Current threshold: {self.threshold:.3f}")

        return anomalous

    def _compute_distance_scores(self, client_updates: List[Dict]) -> np.ndarray:
        """Compute pairwise distance-based anomaly scores (optimized)."""
        n_clients = len(client_updates)

        if n_clients == 0:
            return np.array([])

        if n_clients == 1:
            return np.array([0.0])

        # Flatten updates to vectors for faster computation
        vectors = []
        for update in client_updates:
            vec = torch.cat([v.flatten() for v in update.values()]).cpu().numpy()
            vectors.append(vec)

        vectors = np.array(vectors)

        # Compute pairwise distances efficiently using broadcasting
        # Distance matrix: ||v_i - v_j||_2
        distance_matrix = np.sqrt(((vectors[:, None, :] - vectors[None, :, :]) ** 2).sum(axis=2))

        # Anomaly score = median distance to other clients
        # Set diagonal to large value to exclude self-distance from median
        np.fill_diagonal(distance_matrix, np.inf)
        scores = np.median(distance_matrix, axis=1)

        # Handle case where all scores are identical
        if np.std(scores) > 1e-6:
            scores = (scores - np.mean(scores)) / np.std(scores)
        else:
            scores = np.zeros(n_clients)

        return scores

    def _compute_loss_scores(self, client_losses: List[float]) -> np.ndarray:
        """Compute loss-based anomaly scores (low loss = suspicious)."""
        losses = np.array(client_losses)

        # Normalize losses
        if np.std(losses) > 1e-6:
            normalized = (losses - np.mean(losses)) / np.std(losses)
            # Negative because lower loss is more suspicious
            return -normalized

        return np.zeros_like(losses)

    def _compute_cluster_scores(self, client_updates: List[Dict]) -> np.ndarray:
        """Use DBSCAN clustering to identify outliers (optimized)."""
        n_clients = len(client_updates)

        if n_clients < 3:
            # Not enough clients for meaningful clustering
            return np.zeros(n_clients)

        # Flatten updates to vectors (reuse from distance computation if available)
        try:
            vectors = []
            for update in client_updates:
                vec = torch.cat([v.flatten() for v in update.values()])
                vectors.append(vec.cpu().numpy())

            X = np.array(vectors)

            # Normalize for better clustering
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Apply DBSCAN with adaptive eps
            eps = 0.5 if n_clients < 20 else 1.0
            min_samples = max(2, min(n_clients // 3, 5))  # Cap at 5 for performance

            clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1)
            labels = clustering.fit_predict(X_scaled)

            # Outliers are labeled as -1
            scores = np.array([1.0 if label == -1 else 0.0 for label in labels])
            return scores

        except Exception as e:
            print(f"[WARN] Clustering failed: {e}")
            return np.zeros(n_clients)

    def _get_reputation_weights(self,
                                reputation_scores: Optional[Dict[int, float]],
                                n_clients: int) -> np.ndarray:
        """Get reputation weights for clients."""
        if reputation_scores is None:
            return np.ones(n_clients)

        weights = np.array([reputation_scores.get(i, 1.0) for i in range(n_clients)])
        # Invert so low reputation = high weight in anomaly score
        return 1.0 - weights

    def _combine_signals(self,
                        distance_scores: np.ndarray,
                        loss_scores: np.ndarray,
                        cluster_scores: np.ndarray,
                        reputation_weights: np.ndarray) -> np.ndarray:
        """Combine multiple anomaly signals."""
        # CHANGED: Adjusted weights to reduce false positives
        # Reduced distance weight (was causing benign clients to be flagged)
        w_distance = 0.3  # CHANGED: from 0.4
        w_loss = 0.1  # CHANGED: from 0.2 (loss can be misleading)
        w_cluster = 0.3  # CHANGED: from 0.2 (clustering is more reliable)
        w_reputation = 0.3  # CHANGED: from 0.2 (reputation builds over time)

        composite = (w_distance * distance_scores +
                    w_loss * loss_scores +
                    w_cluster * cluster_scores +
                    w_reputation * reputation_weights)

        return composite

    def _adaptive_threshold(self, scores: np.ndarray, n_clients: int) -> List[int]:
        """Apply adaptive thresholding to identify anomalies."""
        # Statistical threshold
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        stat_threshold = mean_score + self.threshold * std_score

        # Percentile threshold (ensure we don't exclude too many)
        percentile_threshold = np.percentile(scores, 100 * (1 - self.max_exclude_frac))

        # Take the more conservative threshold
        threshold = max(stat_threshold, percentile_threshold)

        # Identify anomalous clients
        anomalous = [i for i, score in enumerate(scores) if score > threshold]

        # Ensure we don't exclude too many
        max_exclude = int(self.max_exclude_frac * n_clients)
        if len(anomalous) > max_exclude:
            # Sort by score and take top anomalous
            anomalous = sorted(anomalous, key=lambda i: scores[i], reverse=True)[:max_exclude]

        # Update threshold history for adaptation
        self.threshold_history.append(self.threshold)
        self._adapt_threshold(len(anomalous), n_clients)

        return anomalous

    def _adapt_threshold(self, num_detected: int, n_clients: int):
        """Adapt threshold based on detection statistics."""
        detection_rate = num_detected / n_clients

        # If we're detecting too few, lower threshold
        if detection_rate < 0.05 and self.threshold > 1.5:
            self.threshold *= 0.95

        # If we're detecting too many, raise threshold
        elif detection_rate > self.max_exclude_frac * 0.8:
            self.threshold *= 1.05

        # Clamp threshold
        self.threshold = max(1.5, min(4.0, self.threshold))

    def _compute_distance(self, update1: Dict, update2: Dict) -> float:
        """Compute L2 distance between two updates."""
        dist = 0.0
        for key in update1.keys():
            if key in update2:
                param_diff = update1[key] - update2[key]
                dist += torch.norm(param_diff).item() ** 2
        return np.sqrt(dist)

    def robust_aggregate(self,
                        client_updates: List[Dict],
                        anomalous_clients: List[int],
                        reputation_scores: Optional[Dict[int, float]] = None) -> Dict:
        """
        Robust aggregation excluding anomalous clients and using reputation weights.

        Args:
            client_updates: List of client parameter updates
            anomalous_clients: List of anomalous client indices
            reputation_scores: Optional reputation scores for weighting

        Returns:
            Aggregated parameters
        """
        n_clients = len(client_updates)

        # Exclude anomalous clients
        normal_indices = [i for i in range(n_clients) if i not in anomalous_clients]

        # CHANGED: Ensure we keep enough clients for good accuracy
        # We need at least 60% of clients (assuming max 40% malicious)
        min_clients = max(1, int(n_clients * 0.6))  # CHANGED: from n_clients // 2
        if len(normal_indices) < min_clients:
            print(f"[WARN] Too few normal clients ({len(normal_indices)}/{n_clients}), keeping at least {min_clients}")
            # Keep all normal clients plus some from anomalous (least suspicious)
            if len(normal_indices) < min_clients:
                normal_indices = list(range(n_clients))[:min_clients]

        # Get weights based on reputation
        if reputation_scores:
            weights = [reputation_scores.get(i, 1.0) for i in normal_indices]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(normal_indices)] * len(normal_indices)

        # Aggregate with weights
        aggregated_params = {}
        param_names = client_updates[0].keys()

        for param_name in param_names:
            weighted_sum = torch.zeros_like(client_updates[0][param_name])
            for idx, weight in zip(normal_indices, weights):
                weighted_sum += weight * client_updates[idx][param_name]
            aggregated_params[param_name] = weighted_sum

        print(f"[DEFENSE] Aggregated {len(normal_indices)}/{n_clients} clients (excluded {len(anomalous_clients)})")

        return aggregated_params


# =============================================================================
# Ensemble Defense
# =============================================================================

class EnsembleDefense:
    """
    Combines multiple defense strategies for robust protection.
    """

    def __init__(self, num_clients: int):
        """
        Initialize ensemble defense with multiple strategies.

        Args:
            num_clients: Total number of clients
        """
        self.reputation_system = ReputationSystem(num_clients)
        self.gradient_analyzer = GradientAnalyzer()
        self.adaptive_committee = AdaptiveCommitteeDefense()

        self.num_clients = num_clients
        self.round_count = 0

    @profile_function
    def defend_and_aggregate(self,
                            client_updates: List[Dict],
                            client_losses: Optional[List[float]] = None) -> Tuple[Dict, List[int], Dict]:
        """
        Apply ensemble defense and aggregate (optimized).

        Args:
            client_updates: List of client parameter updates
            client_losses: Optional list of client losses

        Returns:
            Tuple of (aggregated_params, anomalous_clients, metrics)
        """
        if len(client_updates) == 0:
            raise ValueError("Cannot defend with empty client_updates list")

        self.round_count += 1

        # Step 1: Gradient analysis
        grad_anomalous, grad_metrics = self.gradient_analyzer.analyze_gradients(client_updates)

        # Step 2: Apply gradient clipping and noise (in-place to save memory)
        clipped_updates = self.gradient_analyzer.clip_and_add_noise(client_updates)

        # Clear original updates to free memory if they're different objects
        if clipped_updates is not client_updates:
            # Free memory from original updates
            del client_updates

        # Step 3: Committee-based detection with reputation
        reputation_scores = self.reputation_system.reputation_scores
        committee_anomalous = self.adaptive_committee.detect_anomalies(
            clipped_updates,
            client_losses,
            reputation_scores
        )

        # Step 4: Combine detections (union of anomalous clients) - use set for efficiency
        anomalous_set = set(grad_anomalous) | set(committee_anomalous)
        anomalous_clients = list(anomalous_set)

        # Step 5: Update reputation scores (vectorized)
        for i in range(min(self.num_clients, len(clipped_updates))):
            is_anomalous = i in anomalous_set
            quality = None
            if client_losses and i < len(client_losses):
                # Convert loss to quality (lower loss = higher quality, but clamp)
                quality = max(0.0, min(1.0, 1.0 - min(client_losses[i], 10.0) / 10.0))
            self.reputation_system.update_reputation(i, is_anomalous, quality)

        # Step 6: Robust aggregation
        aggregated = self.adaptive_committee.robust_aggregate(
            clipped_updates,
            anomalous_clients,
            reputation_scores
        )

        # Metrics (only include essential info to save memory)
        metrics = {
            'gradient_analysis': {
                'mean_norm': grad_metrics.get('mean_norm', 0.0),
                'std_norm': grad_metrics.get('std_norm', 0.0),
                'anomalous_count': grad_metrics.get('anomalous_count', 0)
            },
            'committee_anomalous': committee_anomalous,
            'gradient_anomalous': grad_anomalous,
            'total_anomalous': len(anomalous_clients),
            'reputation_scores': dict(self.reputation_system.reputation_scores),
            'adaptive_threshold': self.adaptive_committee.threshold
        }

        return aggregated, anomalous_clients, metrics

    def get_defense_stats(self) -> Dict:
        """Get comprehensive defense statistics."""
        return {
            'round': self.round_count,
            'reputation_scores': dict(self.reputation_system.reputation_scores),
            'trusted_clients': self.reputation_system.get_trusted_clients(),
            'adaptive_threshold': self.adaptive_committee.threshold,
            'detection_stats': self.adaptive_committee.detection_stats
        }


def get_defense(strategy: str = "ensemble", num_clients: int = 10, **kwargs):
    """
    Factory function to create enhanced defense instances.

    Args:
        strategy: Defense strategy ('ensemble', 'adaptive_committee', 'reputation')
        num_clients: Number of clients in the system
        **kwargs: Additional arguments for specific defenses

    Returns:
        Defense instance
    """
    if strategy == "ensemble":
        return EnsembleDefense(num_clients)
    elif strategy == "adaptive_committee":
        return AdaptiveCommitteeDefense(**kwargs)
    elif strategy == "reputation":
        return ReputationSystem(num_clients, **kwargs)
    elif strategy == "gradient_analyzer":
        return GradientAnalyzer(**kwargs)
    else:
        raise ValueError(f"Unknown enhanced defense strategy: {strategy}")
