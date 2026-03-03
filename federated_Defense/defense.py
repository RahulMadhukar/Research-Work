"""
Committee-Based Defense Mechanisms for Federated Learning (Distributed)

This module implements distributed committee-based defense strategies:
1. CMFL (Committee-based Federated Learning) with rotating committees
2. Real-time anomaly detection with NaN/Inf filtering
3. Pluggable aggregation methods (FedAvg, Krum, Multi-Krum, Median, Trimmed Mean)

Performance Optimizations (v2 - Speed Improvements):
- Vectorized distance computations with squared L2 (no sqrt)
- Distributed consensus voting
- Removed malicious client training boost (equal training time)
- Optimized distance metrics (squared L2 instead of L2 with sqrt)
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from scipy.spatial.distance import cdist
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
# NaN/Inf Detection and Filtering
# =============================================================================

def detect_nan_inf_updates(client_updates: List[Dict[str, torch.Tensor]],
                           client_ids: Optional[List[int]] = None) -> Tuple[List[Dict[str, torch.Tensor]], List[int], List[int]]:
    """
    Detect and filter out client updates containing NaN or Inf values.

    This is a critical first line of defense against extremely aggressive attacks
    that cause numerical overflow/underflow in model parameters.

    Args:
        client_updates: List of client model updates (state dicts)
        client_ids: Optional list of client IDs (defaults to indices)

    Returns:
        valid_updates: List of valid client updates (without NaN/Inf)
        valid_client_ids: List of client IDs for valid updates
        invalid_client_ids: List of client IDs with NaN/Inf (detected as malicious)

    Example:
        >>> updates = [client0_update, client1_update, client2_update]
        >>> valid_updates, valid_ids, invalid_ids = detect_nan_inf_updates(updates)
        >>> print(f"Detected {len(invalid_ids)} malicious clients with NaN/Inf")
    """
    if client_ids is None:
        client_ids = list(range(len(client_updates)))

    valid_updates = []
    valid_client_ids = []
    invalid_client_ids = []

    for idx, (client_id, update) in enumerate(zip(client_ids, client_updates)):
        has_nan = False
        has_inf = False
        invalid_params = []

        # Check each parameter tensor in the update
        for param_name, param_tensor in update.items():
            # Convert to tensor if needed
            if not isinstance(param_tensor, torch.Tensor):
                param_tensor = torch.tensor(param_tensor)

            # Check for NaN
            if torch.isnan(param_tensor).any():
                has_nan = True
                invalid_params.append(f"{param_name}:NaN")

            # Check for Inf
            if torch.isinf(param_tensor).any():
                has_inf = True
                invalid_params.append(f"{param_name}:Inf")

            # Early exit if invalid detected
            if has_nan or has_inf:
                break

        if has_nan or has_inf:
            # Mark as malicious - contains invalid values
            invalid_client_ids.append(client_id)
            issue_type = []
            if has_nan:
                issue_type.append("NaN")
            if has_inf:
                issue_type.append("Inf")

            print(f"[NaN/Inf DETECTION] ✓ Client {client_id} DETECTED as malicious ({'/'.join(issue_type)} in {invalid_params[0]})")
        else:
            # Valid update
            valid_updates.append(update)
            valid_client_ids.append(client_id)

    # Summary
    if invalid_client_ids:
        print(f"[NaN/Inf DETECTION] Total detected: {len(invalid_client_ids)}/{len(client_updates)} clients")

    return valid_updates, valid_client_ids, invalid_client_ids




# =============================================================================
# Pluggable Aggregation Methods
# =============================================================================

class AggregationMethod:
    """Base class for aggregation methods."""

    def aggregate(self, client_updates: List[Dict], weights: Optional[List[float]] = None) -> Dict:
        """
        Aggregate client updates.

        Args:
            client_updates: List of client parameter updates
            weights: Optional weights for each client

        Returns:
            Aggregated parameters
        """
        raise NotImplementedError


class FedAvgAggregation(AggregationMethod):
    """Federated Averaging aggregation."""

    def aggregate(self, client_updates: List[Dict], weights: Optional[List[float]] = None) -> Dict:
        """Weighted average of client updates."""
        n_clients = len(client_updates)

        if weights is None:
            weights = [1.0 / n_clients] * n_clients
        else:
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

        aggregated_params = {}
        param_names = client_updates[0].keys()

        for param_name in param_names:
            weighted_sum = torch.zeros_like(client_updates[0][param_name])
            for update, weight in zip(client_updates, weights):
                weighted_sum += weight * update[param_name]
            aggregated_params[param_name] = weighted_sum

        return aggregated_params


class WeightedAverageAggregation(AggregationMethod):
    """Data-size weighted averaging aggregation.

    Each client's update is weighted by its local dataset size.
    If no weights are supplied the fallback is uniform (1/n), but the
    intended use-case always provides explicit data-size weights.
    """

    def aggregate(self, client_updates: List[Dict], weights: Optional[List[float]] = None) -> Dict:
        """Weighted average of client updates using provided weights."""
        n_clients = len(client_updates)

        if weights is None:
            weights = [1.0 / n_clients] * n_clients
        else:
            # Normalize weights so they sum to 1
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

        aggregated_params = {}
        param_names = client_updates[0].keys()

        for param_name in param_names:
            weighted_sum = torch.zeros_like(client_updates[0][param_name])
            for update, weight in zip(client_updates, weights):
                weighted_sum += weight * update[param_name]
            aggregated_params[param_name] = weighted_sum

        return aggregated_params


class KrumAggregation(AggregationMethod):
    """KRUM aggregation - selects the most representative client."""

    def __init__(self, n_attackers: int = 2):
        """
        Args:
            n_attackers: Expected number of malicious clients
        """
        self.n_attackers = n_attackers

    def aggregate(self, client_updates: List[Dict], weights: Optional[List[float]] = None) -> Dict:
        """Select client with minimum distance sum to closest neighbors."""
        n_clients = len(client_updates)
        n_select = n_clients - self.n_attackers - 2  # Number of closest neighbors to consider

        if n_select <= 0:
            # Fallback to FedAvg if too few clients
            return FedAvgAggregation().aggregate(client_updates, weights)

        # Flatten updates to vectors
        vectors = []
        for update in client_updates:
            vec = torch.cat([v.flatten() for v in update.values()]).cpu().numpy()
            vectors.append(vec)
        vectors = np.array(vectors)

        # Compute pairwise distances
        distances = np.zeros((n_clients, n_clients))
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                dist = np.linalg.norm(vectors[i] - vectors[j])
                distances[i, j] = dist
                distances[j, i] = dist

        # For each client, compute sum of distances to n_select closest neighbors
        scores = []
        for i in range(n_clients):
            # Get distances to all other clients
            dists = distances[i].copy()
            dists[i] = float('inf')  # Exclude self
            # Select n_select smallest distances
            closest_dists = np.partition(dists, n_select - 1)[:n_select]
            score = np.sum(closest_dists)
            scores.append(score)

        # Select client with minimum score
        selected_idx = np.argmin(scores)

        return client_updates[selected_idx]


class MultiKrumAggregation(AggregationMethod):
    """Multi-Krum aggregation — selects the top-s clients by Krum score and averages them.

    Krum picks the single client whose sum-of-distances to its closest
    n - f - 2 neighbours is smallest.  Multi-Krum keeps the best *s* clients
    by that same metric and returns their FedAvg, which is more robust than
    using just one client while still filtering out outliers.
    """

    def __init__(self, n_attackers: int = 2, s: int = 3):
        """
        Args:
            n_attackers: Expected number of malicious clients (f)
            s: Number of top-scoring clients to select and average
        """
        self.n_attackers = n_attackers
        self.s = s

    def aggregate(self, client_updates: List[Dict], weights: Optional[List[float]] = None) -> Dict:
        """Select top-s clients by Krum score, then average their updates."""
        n_clients = len(client_updates)
        n_neighbors = n_clients - self.n_attackers - 2  # closest neighbours per score

        if n_neighbors <= 0 or self.s >= n_clients:
            return FedAvgAggregation().aggregate(client_updates, weights)

        # Flatten each client's update dict into a single vector
        vectors = []
        for update in client_updates:
            vec = torch.cat([v.flatten() for v in update.values()]).cpu().numpy()
            vectors.append(vec)
        vectors = np.array(vectors)

        # Pairwise Euclidean distances
        distances = np.zeros((n_clients, n_clients))
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                dist = np.linalg.norm(vectors[i] - vectors[j])
                distances[i, j] = dist
                distances[j, i] = dist

        # Krum score for every client: sum of distances to n_neighbors closest peers
        scores = []
        for i in range(n_clients):
            dists = distances[i].copy()
            dists[i] = float('inf')  # exclude self
            closest_dists = np.partition(dists, n_neighbors - 1)[:n_neighbors]
            scores.append(np.sum(closest_dists))

        # Select top-s (lowest scores)
        selected_indices = np.argsort(scores)[:self.s]

        # Average the selected subset
        selected_updates = [client_updates[i] for i in selected_indices]
        return FedAvgAggregation().aggregate(selected_updates)


class TrimmedMeanAggregation(AggregationMethod):
    """Trimmed Mean aggregation - removes extreme values."""

    def __init__(self, trim_ratio: float = 0.1):
        """
        Args:
            trim_ratio: Fraction of extreme values to trim from each side
        """
        self.trim_ratio = trim_ratio

    def aggregate(self, client_updates: List[Dict], weights: Optional[List[float]] = None) -> Dict:
        """Compute trimmed mean by removing extreme values."""
        n_clients = len(client_updates)
        n_trim = int(n_clients * self.trim_ratio)

        if n_trim == 0 or n_clients - 2 * n_trim < 1:
            # Fallback to FedAvg if trimming would leave too few clients
            return FedAvgAggregation().aggregate(client_updates, weights)

        aggregated_params = {}
        param_names = client_updates[0].keys()

        for param_name in param_names:
            # Stack all client values for this parameter
            stacked = torch.stack([update[param_name] for update in client_updates])

            # Sort along client dimension and trim extremes
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_vals[n_trim:n_clients - n_trim]

            # Compute mean of trimmed values
            aggregated_params[param_name] = torch.mean(trimmed, dim=0)

        return aggregated_params


class MedianAggregation(AggregationMethod):
    """Coordinate-wise median aggregation."""

    def aggregate(self, client_updates: List[Dict], weights: Optional[List[float]] = None) -> Dict:
        """Compute coordinate-wise median."""
        aggregated_params = {}
        param_names = client_updates[0].keys()

        for param_name in param_names:
            # Stack all client values for this parameter
            stacked = torch.stack([update[param_name] for update in client_updates])

            # Compute median along client dimension
            aggregated_params[param_name] = torch.median(stacked, dim=0)[0]

        return aggregated_params


def get_aggregation_method(method_name: str, **kwargs) -> AggregationMethod:
    """
    Factory function to create aggregation method.

    Args:
        method_name: Name of aggregation method ('fedavg', 'krum', 'multi_krum', 'trimmed_mean', 'median')
        **kwargs: Additional arguments for specific methods

    Returns:
        AggregationMethod instance
    """
    method_name = method_name.lower()

    if method_name == 'fedavg':
        return FedAvgAggregation()
    elif method_name == 'weighted_avg':
        return WeightedAverageAggregation()
    elif method_name == 'krum':
        n_attackers = kwargs.get('n_attackers', 2)
        return KrumAggregation(n_attackers=n_attackers)
    elif method_name == 'multi_krum':
        n_attackers = kwargs.get('n_attackers', 2)
        s = kwargs.get('s', 3)
        return MultiKrumAggregation(n_attackers=n_attackers, s=s)
    elif method_name == 'trimmed_mean':
        trim_ratio = kwargs.get('trim_ratio', 0.1)
        return TrimmedMeanAggregation(trim_ratio=trim_ratio)
    elif method_name == 'median':
        return MedianAggregation()
    else:
        print(f"[WARN] Unknown aggregation method '{method_name}', using FedAvg")
        return FedAvgAggregation()


# =============================================================================
# CMFL Committee Defense
# =============================================================================

class CMFLDefense:
    """
    CMFL Paper Defense — Committee Mechanism with L2-Distance Scoring.

    Implements the original CMFL paper algorithm (Che et al., "A Decentralized
    Federated Learning Framework via Committee Mechanism with Convergence
    Guarantee", Section 4, Algorithm 3):

    1. Activate training clients from idle pool (non-committee)
    2. Both training AND committee clients train locally
    3. Committee members score training clients via L2 distance (Eq. 7-8)
    4. Selection Strategy I: sort by score descending, accept top alpha%
    5. pBFT consensus among committee on aggregation set
    6. Data-size weighted aggregation (Eq. 5)
    7. Election: middle-ranked training clients become new committee
    8. Step down: old committee moves to idle pool

    Paper parameters (Section 6.1.2):
        alpha = 0.40  (aggregation participation rate)
        omega = 0.40  (committee proportion)
        10% participation per round
    """

    def __init__(self, num_clients: int,
                 committee_size: int = None,
                 training_clients_per_round: int = None,
                 committee_rotation_rounds: int = 1,
                 aggregation_method: str = 'weighted_avg',
                 consensus_threshold: float = 0.5,
                 aggregation_participation_frac: float = None,
                 selection_strategy: int = 1,
                 **kwargs):
        """
        Initialize CMFL paper-exact defense.

        Args:
            num_clients: Total number of clients
            committee_size: C (number of committee members). Auto-scaled if None.
            training_clients_per_round: n (training clients per round). Auto-scaled if None.
            committee_rotation_rounds: Rotate committee every N rounds
            aggregation_method: Aggregation method for selected clients
            consensus_threshold: pBFT consensus fraction (default 0.5 = majority)
            aggregation_participation_frac: alpha — fraction of training clients
                selected for aggregation. Default 0.4 (paper Section 6.1.2).
            selection_strategy: 1 or 2 (paper Section 4.2.2).
                Strategy I: Accept top alpha% (high score = close to committee = honest).
                            For Byzantine attack robustness.
                Strategy II: Accept bottom alpha% (low score = far from committee).
                             For convergence acceleration in non-attack scenarios.
            **kwargs: Absorbed for compatibility (scoring_mode, initial_anomaly_threshold, etc.)
        """
        self.num_clients = num_clients

        # Initialize aggregation method
        self.aggregation_method = get_aggregation_method(aggregation_method)

        # Paper default: alpha = 0.4 (Section 6.1.2)
        self.aggregation_participation_frac = aggregation_participation_frac if aggregation_participation_frac is not None else 0.4

        # Selection Strategy (paper Section 4.2.2)
        self.selection_strategy = selection_strategy

        # pBFT consensus threshold: floor(C/2) + 1 majority (Section 4.2.4)
        self.consensus_threshold = consensus_threshold

        # Auto-scale committee and training sizes if not provided
        if committee_size is None or training_clients_per_round is None:
            # Paper: omega = 40% committee proportion (Section 6.1.2)
            total_participation = max(5, min(64, int(num_clients * 0.40)))
            committee_size = max(2, int(total_participation * 0.40))
            training_clients_per_round = total_participation - committee_size

        self.committee_size = min(committee_size, num_clients - 1)  # At least 1 training client
        self.training_clients_per_round = min(training_clients_per_round,
                                               num_clients - self.committee_size)
        self.committee_rotation_rounds = committee_rotation_rounds

        # Client pools (TRUE committee structure)
        self.committee_members: set = set()
        self.training_clients: set = set()
        self.idle_clients: set = set(range(num_clients))

        # Initialize committee randomly
        initial_committee = np.random.choice(list(self.idle_clients),
                                             self.committee_size,
                                             replace=False)
        self.committee_members = set(initial_committee.tolist())
        self.idle_clients -= self.committee_members

        # Tracking
        self.round_count = 0
        self.detected_malicious = set()
        self.committee_history = [sorted(list(self.committee_members))]

        # Per-round detection tracking
        self.detection_history_per_round = []

        # Per-round tracking of malicious clients in committee
        self.malicious_in_committee_history = []

        # Verbose logging: set to True for debug, False for speed
        self.verbose = False

    def activate_training_clients(self) -> List[int]:
        """
        Activate training clients from idle pool (non-committee).

        Paper Section 4.2.1: Select training clients from idle pool.
        Committee members are NOT in the training set — they score training clients.
        Both groups train locally, but only training clients' updates are scored.
        """
        available_idle = list(self.idle_clients)

        n_activate = min(self.training_clients_per_round, len(available_idle))
        if n_activate == 0:
            # Fallback: if no idle clients, use all non-committee clients
            available_idle = [i for i in range(self.num_clients) if i not in self.committee_members]
            n_activate = min(self.training_clients_per_round, len(available_idle))

        activated = list(np.random.choice(available_idle, n_activate, replace=False))
        self.training_clients = set(activated)
        self.idle_clients -= self.training_clients

        if self.verbose:
            print(f"  [ACTIVATE] Training: {sorted(activated)} ({len(activated)} clients)")
            print(f"  [ACTIVATE] Committee: {sorted(self.committee_members)} ({len(self.committee_members)} members)")

        return activated

    def committee_scoring(self,
                          training_updates: Dict[int, Dict],
                          committee_updates: Dict[int, Dict]) -> Tuple[Dict[int, float], Dict[int, Dict[int, float]]]:
        """
        Paper-exact scoring (Eq. 7-8).

        For each committee member c, for each training client k:
            P_k^c = 1 / (||g_k - g_c||_2^2 + eps)         (Eq. 7)

        Final score per training client:
            P_k = C / (sum_c ||g_k - g_c||_2^2 + eps)     (Eq. 8)

        Higher P_k = gradient closer to committee = likely honest.

        Args:
            training_updates: {client_id: update_dict} for training clients
            committee_updates: {client_id: update_dict} for committee members

        Returns:
            (final_scores, per_member_scores) where:
                final_scores: {training_id: P_k}
                per_member_scores: {committee_id: {training_id: P_k^c}}
        """
        training_ids = sorted(training_updates.keys())
        committee_ids = sorted(committee_updates.keys())
        C = len(committee_ids)
        eps = 1e-10

        # Flatten all updates to vectors (as matrices for vectorized cdist)
        training_matrix = []
        for cid in training_ids:
            vec = torch.cat([v.flatten() for v in training_updates[cid].values()]).cpu().numpy()
            if not np.isfinite(vec).all():
                vec = np.nan_to_num(vec, nan=1e10, posinf=1e10, neginf=-1e10)
            training_matrix.append(vec)
        training_matrix = np.array(training_matrix)  # shape: (T, d)

        committee_matrix = []
        for cid in committee_ids:
            vec = torch.cat([v.flatten() for v in committee_updates[cid].values()]).cpu().numpy()
            if not np.isfinite(vec).all():
                vec = np.nan_to_num(vec, nan=1e10, posinf=1e10, neginf=-1e10)
            committee_matrix.append(vec)
        committee_matrix = np.array(committee_matrix)  # shape: (C, d)

        # Vectorized: compute all pairwise L2 distances at once
        # dist_matrix shape: (T, C) — dist_matrix[t, c] = ||g_t - g_c||_2
        sq_dist_matrix = cdist(training_matrix, committee_matrix, metric='sqeuclidean')

        # Eq. 7: P_k^c = 1 / (||g_k - g_c||_2^2 + eps) — per-member scores
        per_member_scores = {}
        for c_idx, c_id in enumerate(committee_ids):
            per_member_scores[c_id] = {}
            for t_idx, t_id in enumerate(training_ids):
                per_member_scores[c_id][t_id] = 1.0 / (sq_dist_matrix[t_idx, c_idx] + eps)

        # Eq. 8: P_k = C / (sum_c ||g_k - g_c||_2^2 + eps) — vectorized
        sum_sq_distances = sq_dist_matrix.sum(axis=1)  # shape: (T,)
        final_scores = {}
        for t_idx, t_id in enumerate(training_ids):
            final_scores[t_id] = C / (sum_sq_distances[t_idx] + eps)

        if self.verbose:
            print(f"\n  [SCORING] L2-distance scoring (Eq. 7-8):")
            for t_id in training_ids:
                print(f"    Training client {t_id}: P_k = {final_scores[t_id]:.6e}")

        return final_scores, per_member_scores

    def pbft_consensus(self,
                       final_scores: Dict[int, float],
                       training_client_ids: List[int]) -> Tuple[List[int], List[int]]:
        """
        pBFT consensus with Selection Strategy I or II (Section 4.2.2, 4.2.4).

        Selection Strategy I (for attack robustness):
        - Sort training clients by P_k descending (high score = honest)
        - Accept top alpha% clients for aggregation
        - Rejects low-score clients (likely malicious)

        Selection Strategy II (for convergence acceleration):
        - Sort training clients by P_k descending
        - Accept bottom alpha% clients for aggregation
        - Includes low-score clients (diverse gradients aid convergence)
        - NOT suitable when Byzantine attacks are present

        pBFT consensus (Algorithm 1-2):
        - Each committee member computes its aggregation set S_a^c
        - Since all members see the same final P_k scores (Eq. 8),
          consensus is automatically achieved (all members produce identical sets)

        Args:
            final_scores: {training_id: P_k} from committee_scoring
            training_client_ids: List of training client IDs

        Returns:
            (selected_clients, flagged_clients)
        """
        n_training = len(training_client_ids)
        alpha = self.aggregation_participation_frac

        # Sort by P_k descending (high score = close to committee = honest)
        sorted_by_score = sorted(training_client_ids, key=lambda cid: final_scores.get(cid, 0), reverse=True)

        n_accept = max(1, int(n_training * alpha))

        if self.selection_strategy == 2:
            # Strategy II: accept bottom alpha% (paper Section 4.2.2)
            # Low-score clients have diverse gradients → better convergence
            selected_clients = sorted_by_score[-n_accept:]
            flagged_clients = sorted_by_score[:-n_accept]
        else:
            # Strategy I (default): accept top alpha% (paper Section 4.2.2)
            # High-score clients are close to committee → likely honest
            selected_clients = sorted_by_score[:n_accept]
            flagged_clients = sorted_by_score[n_accept:]

        if self.verbose:
            required_votes = int(np.floor(len(self.committee_members) / 2)) + 1
            strategy_label = "II" if self.selection_strategy == 2 else "I"
            print(f"\n  [pBFT] Consensus: {len(self.committee_members)}/{len(self.committee_members)} "
                  f"members agree (required: {required_votes})")
            print(f"  [SELECTION] Strategy {strategy_label} (alpha={alpha:.0%}): accept {n_accept}/{n_training} clients")
            print(f"  [SELECTED] {sorted(selected_clients)}")
            print(f"  [FLAGGED]  {sorted(flagged_clients)}")

        return selected_clients, flagged_clients

    def elect_new_committee(self,
                            training_client_ids: List[int],
                            final_scores: Dict[int, float]) -> List[int]:
        """
        Paper-exact committee election (Section 4.2.3).

        "Choose the training client closed to the middle position as the
        committee clients" — sort by score, pick middle-ranked clients.
        Excludes bottom (suspicious) and top (too similar / extreme).

        Args:
            training_client_ids: List of training client IDs
            final_scores: {training_id: P_k} scores from committee_scoring

        Returns:
            List of new committee member IDs
        """
        sorted_by_score = sorted(training_client_ids, key=lambda cid: final_scores.get(cid, 0))
        n = len(sorted_by_score)

        if n <= self.committee_size:
            new_committee = sorted_by_score
        else:
            lower_bound = max(0, (n - self.committee_size) // 2)
            upper_bound = lower_bound + self.committee_size
            if upper_bound > n:
                upper_bound = n
                lower_bound = max(0, upper_bound - self.committee_size)
            new_committee = sorted_by_score[lower_bound:upper_bound]

        return new_committee

    def step_down_committee(self):
        """Move old committee members to idle pool (Paper Section 4.2.3)."""
        old_committee = list(self.committee_members)
        self.idle_clients.update(old_committee)
        self.committee_members.clear()

    def aggregate_selected_clients(self,
                                   selected_client_ids: List[int],
                                   all_updates: Dict[int, Dict],
                                   client_data_sizes: Optional[Dict[int, int]] = None) -> Dict:
        """Aggregate updates from selected clients using data-size weighted averaging (Eq. 5)."""
        if len(selected_client_ids) == 0:
            raise ValueError("Cannot aggregate with no selected clients")

        selected_updates = [all_updates[cid] for cid in selected_client_ids]

        if client_data_sizes:
            weights = [client_data_sizes.get(cid, 1) for cid in selected_client_ids]
        else:
            weights = [1.0] * len(selected_client_ids)

        return self.aggregation_method.aggregate(selected_updates, weights)

    def cmfl_round(self,
                   client_updates: Dict[int, Dict],
                   client_losses: Dict[int, float],
                   malicious_client_ids: Optional[List[int]] = None,
                   all_clients: Optional[List] = None,
                   use_anomaly_detection: bool = True,
                   client_data_sizes: Optional[Dict[int, int]] = None) -> Tuple[Dict, List[int], Dict]:
        """
        Execute one full CMFL round (Paper Algorithm 3).

        Steps:
        1. Separate training vs committee updates
        2. Committee scoring (Eq. 7-8)
        3. pBFT consensus with Selection Strategy I (top alpha%)
        4. Aggregate selected clients (Eq. 5, data-size weighted)
        5. Calculate detection metrics
        6. Elect new committee + step down on rotation rounds
        7. Return (aggregated_params, flagged_clients, metrics)
        """
        self.round_count += 1
        all_client_ids = sorted(client_updates.keys())

        # Track malicious clients in committee this round
        if all_clients is not None:
            malicious_in_committee = [
                cid for cid in self.committee_members
                if cid < len(all_clients) and all_clients[cid].is_malicious
            ]
            self.malicious_in_committee_history.append({
                'round': self.round_count,
                'committee_members': sorted(self.committee_members),
                'committee_size': len(self.committee_members),
                'malicious_in_committee': sorted(malicious_in_committee),
                'malicious_count': len(malicious_in_committee),
                'malicious_fraction': len(malicious_in_committee) / max(1, len(self.committee_members)),
            })

        # Step 1: Separate training vs committee updates
        training_ids = [cid for cid in all_client_ids if cid in self.training_clients]
        committee_ids = [cid for cid in all_client_ids if cid in self.committee_members]

        training_updates = {cid: client_updates[cid] for cid in training_ids}
        committee_updates_dict = {cid: client_updates[cid] for cid in committee_ids}

        if use_anomaly_detection and len(training_ids) > 0 and len(committee_ids) > 0:
            # Step 2: Committee scoring (Eq. 7-8)
            final_scores, per_member_scores = self.committee_scoring(
                training_updates, committee_updates_dict
            )

            # Step 3: pBFT consensus with Selection Strategy I
            selected_clients, flagged_clients = self.pbft_consensus(
                final_scores, training_ids
            )
        elif use_anomaly_detection and len(committee_ids) == 0:
            selected_clients = list(training_ids)
            flagged_clients = []
            final_scores = {cid: 1.0 for cid in training_ids}
        else:
            selected_clients = list(training_ids)
            flagged_clients = []
            final_scores = {}

        # Track cumulative detected malicious
        for cid in flagged_clients:
            self.detected_malicious.add(cid)

        # Step 4: Aggregate selected clients (Eq. 5, data-size weighted)
        # Include committee updates in the update pool so aggregation can access them if needed
        aggregated_params = self.aggregate_selected_clients(
            selected_clients, client_updates, client_data_sizes=client_data_sizes
        )

        # Step 5: Detection metrics
        detection_metrics_dict = None
        if all_clients is not None and use_anomaly_detection:
            round_metrics = self.calculate_round_detection_metrics(
                all_clients, round_num=self.round_count, flagged_this_round=flagged_clients
            )
            detection_metrics_dict = round_metrics

        # Step 6 & 7: Committee rotation (election + step down)
        if self.round_count % self.committee_rotation_rounds == 0 and len(training_ids) > 0:
            # Elect new committee from middle-ranked training clients (Section 4.2.3)
            new_committee = self.elect_new_committee(training_ids, final_scores)
            # Step down old committee → idle
            self.step_down_committee()
            # Install new committee
            self.committee_members = set(new_committee)
            self.idle_clients -= self.committee_members
            # Non-elected training clients → idle
            non_elected = self.training_clients - self.committee_members
            self.idle_clients.update(non_elected)
            self.training_clients.clear()
            self.committee_history.append(sorted(list(self.committee_members)))
        else:
            # No rotation: training clients return to idle
            self.idle_clients.update(self.training_clients)
            self.training_clients.clear()

        # Metrics
        metrics = {
            'round': self.round_count,
            'training_clients': sorted(training_ids),
            'committee_members': sorted(list(self.committee_members)),
            'selected_clients': selected_clients,
            'flagged_clients': flagged_clients,
            'aggregation_clients': selected_clients,
            'final_scores': final_scores,
            'anomaly_threshold': self.aggregation_participation_frac
        }

        if detection_metrics_dict is not None:
            metrics['detection_metrics'] = detection_metrics_dict

        return aggregated_params, flagged_clients, metrics

    def get_malicious_in_committee_summary(self):
        """Summarize malicious-in-committee tracking across all rounds.

        Returns:
            dict with per-round history and aggregate statistics.
        """
        history = self.malicious_in_committee_history
        if not history:
            return {
                'per_round': [],
                'total_rounds': 0,
                'avg_malicious_count': 0.0,
                'avg_malicious_fraction': 0.0,
                'max_malicious_count': 0,
                'max_malicious_fraction': 0.0,
                'rounds_with_malicious': 0,
                'rounds_with_malicious_fraction': 0.0,
            }

        mal_counts = [r['malicious_count'] for r in history]
        mal_fracs = [r['malicious_fraction'] for r in history]
        rounds_with = sum(1 for c in mal_counts if c > 0)

        return {
            'per_round': history,
            'total_rounds': len(history),
            'avg_malicious_count': float(np.mean(mal_counts)),
            'avg_malicious_fraction': float(np.mean(mal_fracs)),
            'max_malicious_count': int(max(mal_counts)),
            'max_malicious_fraction': float(max(mal_fracs)),
            'rounds_with_malicious': rounds_with,
            'rounds_with_malicious_fraction': rounds_with / len(history) if history else 0.0,
        }

    # =========================================================================
    # Detection Metric Methods (simple per-round)
    # =========================================================================

    def calculate_round_detection_metrics(self, all_clients: List, round_num: int = 0,
                                          flagged_this_round: Optional[List[int]] = None):
        """
        Calculate and store detection metrics for a single round.

        Args:
            all_clients: List of all client objects (to check is_malicious attribute)
            round_num: Current round number
            flagged_this_round: List of client IDs flagged in THIS round.
                               If None, falls back to cumulative detections.
        """
        participants = list(self.training_clients | self.committee_members)

        malicious_participants = [
            cid for cid in participants
            if cid < len(all_clients) and getattr(all_clients[cid], 'is_malicious', False)
        ]
        benign_participants = [
            cid for cid in participants
            if cid < len(all_clients) and not getattr(all_clients[cid], 'is_malicious', False)
        ]

        if flagged_this_round is not None:
            detected_ids = flagged_this_round
        else:
            detected_ids = list(self.detected_malicious)

        tp = len([cid for cid in malicious_participants if cid in detected_ids])
        fp = len([cid for cid in benign_participants if cid in detected_ids])
        fn = len([cid for cid in malicious_participants if cid not in detected_ids])
        tn = len([cid for cid in benign_participants if cid not in detected_ids])
        total = tp + fp + fn + tn

        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        fpr = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0.0
        fnr = 100.0 - recall
        dacc = ((tp + tn) / total * 100) if total > 0 else 0.0

        round_metrics = {
            'round': round_num,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'precision': precision, 'recall': recall,
            'f1_score': f1_score, 'fpr': fpr, 'fnr': fnr, 'dacc': dacc,
            'num_participants': len(participants),
            'num_malicious': len(malicious_participants),
            'num_benign': len(benign_participants),
        }

        self.detection_history_per_round.append(round_metrics)
        return round_metrics


def get_defense(strategy: str = "cmfl", num_clients: int = 10, **kwargs):
    """
    Factory function to create committee-based defense instances.

    Args:
        strategy: Defense strategy ('cmfl')
        num_clients: Number of clients in the system
        **kwargs: Additional arguments for specific defenses

    Returns:
        Defense instance
    """
    if strategy == "cmfl":
        return CMFLDefense(num_clients, **kwargs)
    else:
        raise ValueError(f"Unknown committee defense strategy: {strategy}. Use 'cmfl'.")
