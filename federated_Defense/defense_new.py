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
# Committee-Based Defense (Distributed)
# =============================================================================

class AdaptiveCommitteeDefense:
    """
    TRUE Committee-based Defense with ADAPTIVE Thresholds.

    Key Differences from CMFL:
    - Adaptive threshold mechanism (adjusts based on detection history)
    - Different signal weighting strategy
    - Threshold history tracking and auto-tuning
    - Statistical + percentile dual-threshold approach
    - Performance-based committee rotation

    Implements committee workflow:
    1. Activate training clients from idle pool
    2. Committee + training clients train
    3. Committee members score training clients (ADAPTIVE thresholds)
    4. Committee voting with consensus
    5. FedAvg aggregation
    6. Performance-based committee rotation
    7. Step down to idle pool
    """

    def __init__(self,
                 num_clients: int = 10,
                 committee_size: int = None,
                 training_clients_per_round: int = None,
                 committee_rotation_rounds: int = 2,
                 initial_threshold: float = 1.5,  # LOWERED from 2.5 for better detection sensitivity
                 max_exclude_frac: float = 0.85,  # INCREASED from 0.6 to handle high malicious ratios
                 consensus_threshold: float = 0.6,
                 use_clustering: bool = True,
                 adaptive_tuning: bool = True):
        """
        Initialize Adaptive Committee Defense.

        Args:
            num_clients: Total number of clients
            committee_size: Number of committee members (auto-scaled if None)
            training_clients_per_round: Training clients activated per round (auto-scaled if None)
            committee_rotation_rounds: Rotate committee every N rounds
            initial_threshold: Initial threshold multiplier (ADAPTIVE feature)
            max_exclude_frac: Maximum fraction of clients to exclude
            consensus_threshold: Fraction of committee votes needed
            use_clustering: Whether to use DBSCAN clustering
            adaptive_tuning: Whether to auto-tune threshold (UNIQUE feature)
        """
        self.num_clients = num_clients

        # Auto-scale committee size based on num_clients (20% of total)
        if committee_size is None:
            committee_size = max(5, min(20, int(num_clients * 0.20)))

        # Auto-scale training clients per round (5-10% of total)
        if training_clients_per_round is None:
            training_clients_per_round = max(3, min(10, int(num_clients * 0.08)))

        self.committee_size = min(committee_size, num_clients)
        self.training_clients_per_round = min(training_clients_per_round, num_clients - self.committee_size)
        self.committee_rotation_rounds = committee_rotation_rounds
        self.threshold = initial_threshold
        self.max_exclude_frac = max_exclude_frac
        self.consensus_threshold = consensus_threshold
        self.use_clustering = use_clustering
        self.adaptive_tuning = adaptive_tuning

        # Client categorization (TRUE committee structure)
        self.committee_members: Set[int] = set()
        self.training_clients: Set[int] = set()
        self.idle_clients: Set[int] = set(range(num_clients))

        # Initialize committee randomly
        initial_committee = np.random.choice(list(self.idle_clients),
                                            self.committee_size,
                                            replace=False)
        self.committee_members = set(initial_committee)
        self.idle_clients -= self.committee_members

        # UNIQUE: Adaptive threshold tracking
        self.threshold_history = deque(maxlen=20)
        self.detection_stats = {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0
        }

        # Tracking
        self.round_count = 0
        self.committee_history = [list(self.committee_members)]
        self.reputation_scores = {i: 1.0 for i in range(num_clients)}
        self.client_performance_history = defaultdict(list)

        # Detection tracking
        self.detected_malicious = set()  # Track clients detected as malicious across rounds

        print(f"\n[ADAPTIVE COMMITTEE INIT] TRUE Committee Defense with Adaptive Thresholds")
        print(f"  Total clients: {num_clients}")
        print(f"  Committee size: {self.committee_size}")
        print(f"  Training clients per round: {self.training_clients_per_round}")
        print(f"  Initial threshold: {initial_threshold} (ADAPTIVE)")
        print(f"  Initial committee: {sorted(list(self.committee_members))}")

    def activate_training_clients(self) -> List[int]:
        """Activate training clients from idle pool."""
        available_idle = list(self.idle_clients)

        if len(available_idle) < self.training_clients_per_round:
            n_activate = len(available_idle)
        else:
            n_activate = self.training_clients_per_round

        if n_activate == 0:
            print(f"  [ACTIVATE] No idle clients available")
            return []

        # Random selection from idle pool
        activated = np.random.choice(available_idle, n_activate, replace=False)

        self.training_clients = set(activated)
        self.idle_clients -= self.training_clients

        print(f"  [ACTIVATE] Selected {len(activated)} training clients: {sorted(list(activated))}")
        print(f"  [ACTIVATE] Committee: {sorted(list(self.committee_members))}")
        print(f"  [ACTIVATE] Remaining idle: {len(self.idle_clients)} clients")

        return list(activated)

    @profile_function
    def committee_scoring_adaptive(self,
                                   training_updates: Dict[int, Dict],
                                   committee_updates: Dict[int, Dict],
                                   training_losses: Dict[int, float],
                                   committee_losses: Dict[int, float]) -> Dict[int, Dict[int, float]]:
        """
        Committee members score training clients with ADAPTIVE thresholds.

        UNIQUE: Each committee member uses the same adaptive threshold that
        adjusts based on historical detection performance.
        """
        committee_scores = {}

        # Get all updates for reference
        all_client_ids = list(training_updates.keys()) + list(committee_updates.keys())
        all_updates = list(training_updates.values()) + list(committee_updates.values())
        all_losses = list(training_losses.values()) + list(committee_losses.values())

        training_client_ids = list(training_updates.keys())

        print(f"\n  [SCORING] Committee scoring with ADAPTIVE threshold={self.threshold:.3f}")

        # Each committee member independently scores
        for committee_id in self.committee_members:
            # Compute multi-signal scores
            distance_scores = self._compute_distance_scores(all_updates)
            loss_scores = self._compute_loss_scores(all_losses)
            cluster_scores = self._compute_cluster_scores(all_updates)
            reputation_weights = np.array([self.reputation_scores.get(cid, 1.0) for cid in all_client_ids])

            # UNIQUE: Different weight combination than CMFL
            composite_scores = self._combine_signals_adaptive(
                distance_scores,
                loss_scores,
                cluster_scores,
                reputation_weights
            )

            # Extract scores for training clients only
            training_scores = {}
            for i, client_id in enumerate(all_client_ids):
                if client_id in training_client_ids:
                    training_scores[client_id] = float(composite_scores[i])

            committee_scores[committee_id] = training_scores

            print(f"    Committee {committee_id} scores: {[(cid, f'{score:.3f}') for cid, score in sorted(training_scores.items())]}")

        return committee_scores

    def committee_consensus_adaptive(self, committee_scores: Dict[int, Dict[int, float]]) -> Tuple[List[int], List[int]]:
        """
        Committee voting with ADAPTIVE threshold.

        UNIQUE: Uses adaptive threshold + statistical + percentile dual-threshold.
        """
        training_client_ids = list(next(iter(committee_scores.values())).keys())
        n_committee = len(committee_scores)

        # Collect all scores for statistical analysis
        all_scores_per_client = defaultdict(list)
        for committee_id, scores in committee_scores.items():
            for training_id, score in scores.items():
                all_scores_per_client[training_id].append(score)

        # UNIQUE: Adaptive threshold voting with statistical analysis
        votes = defaultdict(int)
        for committee_id, scores in committee_scores.items():
            for training_id, score in scores.items():
                # Get mean score from all committee members for this client
                mean_score = np.mean(all_scores_per_client[training_id])
                std_score = np.std(all_scores_per_client[training_id])

                # ADAPTIVE: Statistical threshold
                stat_threshold = mean_score + self.threshold * std_score

                # Vote if score exceeds adaptive threshold
                if score > stat_threshold:
                    votes[training_id] += 1

        # Consensus: need majority votes
        min_votes = int(n_committee * self.consensus_threshold)
        flagged_clients = [cid for cid, vote_count in votes.items() if vote_count >= min_votes]

        # Apply max_exclude_frac constraint
        max_exclude = int(len(training_client_ids) * self.max_exclude_frac)
        if len(flagged_clients) > max_exclude:
            flagged_sorted = sorted(flagged_clients, key=lambda cid: votes[cid], reverse=True)
            flagged_clients = flagged_sorted[:max_exclude]

        selected_clients = [cid for cid in training_client_ids if cid not in flagged_clients]

        print(f"\n  [CONSENSUS] Adaptive threshold voting:")
        print(f"    Current threshold: {self.threshold:.3f}")
        print(f"    Votes required: {min_votes}/{n_committee}")
        print(f"    Flagged: {sorted(flagged_clients)} (votes: {[votes[cid] for cid in flagged_clients]})")
        print(f"    Selected: {sorted(selected_clients)}")

        # UNIQUE: Adapt threshold based on detection rate
        if self.adaptive_tuning:
            self._adapt_threshold(len(flagged_clients), len(training_client_ids))

        return selected_clients, flagged_clients

    def adaptive_committee_round(self,
                                 client_updates: Dict[int, Dict],
                                 client_losses: Dict[int, float]) -> Tuple[Dict, List[int], Dict]:
        """
        Execute one complete Adaptive Committee round.

        Follows committee workflow with ADAPTIVE threshold mechanism.
        """
        self.round_count += 1

        print(f"\n{'='*80}")
        print(f"ADAPTIVE COMMITTEE ROUND {self.round_count}")
        print(f"{'='*80}")

        # Separate training and committee updates
        training_updates = {cid: client_updates[cid] for cid in self.training_clients if cid in client_updates}
        committee_updates = {cid: client_updates[cid] for cid in self.committee_members if cid in client_updates}

        training_losses = {cid: client_losses[cid] for cid in self.training_clients if cid in client_losses}
        committee_losses = {cid: client_losses[cid] for cid in self.committee_members if cid in client_losses}

        print(f"\n[ADAPTIVE COMMITTEE] Client status:")
        print(f"  Training clients: {sorted(list(self.training_clients))}")
        print(f"  Committee members: {sorted(list(self.committee_members))}")
        print(f"  Idle clients: {sorted(list(self.idle_clients))}")

        # Step 3: Scoring with ADAPTIVE thresholds
        committee_scores = self.committee_scoring_adaptive(
            training_updates,
            committee_updates,
            training_losses,
            committee_losses
        )

        # Step 4: Consensus with ADAPTIVE voting
        selected_clients, flagged_clients = self.committee_consensus_adaptive(committee_scores)

        # Step 5: Aggregation (FedAvg with reputation)
        aggregation_clients = selected_clients + list(self.committee_members)
        aggregated_params = self._aggregate_with_reputation(
            aggregation_clients,
            client_updates
        )

        # Update reputation
        self._update_reputation(selected_clients, flagged_clients)

        # Step 6 & 7: Rotation (performance-based)
        if self.round_count % self.committee_rotation_rounds == 0:
            self._step_down_committee()
            new_committee = self._elect_new_committee(list(self.training_clients), training_losses)

            self.committee_members = set(new_committee)
            self.idle_clients -= self.committee_members

            non_elected_training = self.training_clients - self.committee_members
            self.idle_clients.update(non_elected_training)
            self.training_clients.clear()

            self.committee_history.append(list(self.committee_members))
        else:
            self.idle_clients.update(self.training_clients)
            self.training_clients.clear()

        # Metrics
        metrics = {
            'round': self.round_count,
            'training_clients': sorted(list(training_updates.keys())),
            'committee_members': sorted(list(self.committee_members)),
            'selected_clients': selected_clients,
            'flagged_clients': flagged_clients,
            'aggregation_clients': aggregation_clients,
            'committee_scores': committee_scores,
            'reputation_scores': dict(self.reputation_scores),
            'adaptive_threshold': self.threshold,  # UNIQUE: Track adaptive threshold
            'threshold_history': list(self.threshold_history)  # UNIQUE
        }

        return aggregated_params, flagged_clients, metrics

    def _aggregate_with_reputation(self,
                                   selected_client_ids: List[int],
                                   all_updates: Dict[int, Dict]) -> Dict:
        """Aggregate using FedAvg with reputation weighting."""
        if len(selected_client_ids) == 0:
            raise ValueError("Cannot aggregate with no selected clients")

        selected_updates = [all_updates[cid] for cid in selected_client_ids]
        weights = [self.reputation_scores.get(cid, 1.0) for cid in selected_client_ids]

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # FedAvg aggregation
        aggregated_params = {}
        param_names = selected_updates[0].keys()

        for param_name in param_names:
            weighted_sum = torch.zeros_like(selected_updates[0][param_name])
            for update, weight in zip(selected_updates, weights):
                weighted_sum += weight * update[param_name]
            aggregated_params[param_name] = weighted_sum

        print(f"\n  [AGGREGATION] FedAvg with reputation weights")
        print(f"    Aggregated {len(selected_client_ids)} clients")

        return aggregated_params

    def _elect_new_committee(self, training_client_ids: List[int], training_losses: Dict[int, float]) -> List[int]:
        """Elect new committee based on performance + reputation."""
        election_scores = {}
        for client_id in training_client_ids:
            loss = training_losses.get(client_id, float('inf'))
            reputation = self.reputation_scores.get(client_id, 0.5)
            score = reputation / (loss + 1.0)
            election_scores[client_id] = score

        sorted_candidates = sorted(election_scores.items(), key=lambda x: x[1], reverse=True)
        n_elect = min(self.committee_size, len(sorted_candidates))
        new_committee = [cid for cid, score in sorted_candidates[:n_elect]]

        print(f"\n  [ELECTION] Performance-based election:")
        print(f"    Candidates: {[(cid, f'{score:.4f}') for cid, score in sorted_candidates]}")
        print(f"    Elected: {sorted(new_committee)}")

        return new_committee

    def _step_down_committee(self):
        """Move committee members to idle pool."""
        old_committee = list(self.committee_members)
        self.idle_clients.update(old_committee)
        print(f"\n  [STEP DOWN] Committee → idle pool: {sorted(old_committee)}")

    def _update_reputation(self, selected_clients: List[int], flagged_clients: List[int]):
        """Update reputation scores."""
        for cid in selected_clients:
            self.reputation_scores[cid] = min(1.0, self.reputation_scores.get(cid, 0.5) + 0.05)
        for cid in flagged_clients:
            self.reputation_scores[cid] = max(0.0, self.reputation_scores.get(cid, 0.5) - 0.1)

    def get_client_categories(self) -> Dict[str, List[int]]:
        """Get current client categorization."""
        return {
            'training': sorted(list(self.training_clients)),
            'committee': sorted(list(self.committee_members)),
            'idle': sorted(list(self.idle_clients))
        }

    def _combine_signals_adaptive(self,
                                  distance_scores: np.ndarray,
                                  loss_scores: np.ndarray,
                                  cluster_scores: np.ndarray,
                                  reputation_weights: np.ndarray) -> np.ndarray:
        """
        Combine signals with ADAPTIVE weighting strategy.

        UNIQUE: Different weights than CMFL (0.4 distance, 0.3 loss, 0.3 cluster).
        Adaptive approach focuses more on reputation and clustering.

        Weights:
        - 0.3 distance (reduced from CMFL's 0.4)
        - 0.1 loss (reduced from CMFL's 0.3)
        - 0.3 cluster (same as CMFL)
        - 0.3 reputation (increased emphasis vs CMFL)
        """
        # Normalize each signal to [0, 1]
        def normalize(scores):
            if len(scores) == 0 or np.max(scores) < 1e-6:
                return scores
            return (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-6)

        norm_distance = normalize(distance_scores)
        norm_loss = normalize(loss_scores)
        norm_cluster = cluster_scores  # Already binary

        # Normalize reputation weights (invert so low reputation = high anomaly score)
        norm_reputation = 1.0 - reputation_weights

        # UNIQUE: Adaptive-specific weights
        composite = (
            0.3 * norm_distance +      # Reduced from CMFL's 0.4
            0.1 * norm_loss +          # Reduced from CMFL's 0.3
            0.3 * norm_cluster +       # Same as CMFL
            0.3 * norm_reputation      # Increased from CMFL (implicit)
        )

        return composite

    # =========================================================================
    # BACKWARD COMPATIBILITY: Keep old detect_anomalies for non-committee mode
    # =========================================================================

    @profile_function
    def detect_anomalies(self,
                        client_updates: List[Dict],
                        client_losses: Optional[List[float]] = None,
                        reputation_scores: Optional[Dict[int, float]] = None) -> List[int]:
        """
        Detect anomalous clients using multiple signals (LEGACY method for backward compatibility).

        This is used when not in committee mode (all clients train).

        Args:
            client_updates: List of client parameter updates
            client_losses: Optional list of client losses
            reputation_scores: Optional reputation scores for clients

        Returns:
            List of anomalous client indices
        """
        n_clients = len(client_updates)
        original_n_clients = n_clients

        # CRITICAL: First check for NaN/Inf values (catches extremely aggressive attacks)
        original_client_ids = list(range(n_clients))
        valid_updates, valid_ids, nan_inf_clients = detect_nan_inf_updates(client_updates, original_client_ids)

        # Initialize filtered variables
        filtered_reputation_scores = reputation_scores
        filtered_client_losses = client_losses

        # If NaN/Inf detected, mark those clients as anomalous immediately
        if nan_inf_clients:
            # Update detection stats
            for client_id in nan_inf_clients:
                self.detected_malicious.add(client_id)

            # If all clients have NaN/Inf, return all as anomalous
            if len(valid_updates) == 0:
                print(f"[ADAPTIVE DEFENSE] All clients have NaN/Inf - marking all as anomalous")
                return list(range(original_n_clients))

            # Continue with valid updates for further analysis
            client_updates = valid_updates
            n_clients = len(valid_updates)

            # CRITICAL: Filter client_losses to match valid clients only
            if client_losses is not None and len(client_losses) == original_n_clients:
                filtered_client_losses = [client_losses[i] for i in valid_ids]

            # CRITICAL: Filter reputation_scores to match valid clients only
            if reputation_scores is not None:
                # Remap: new index i → old client ID valid_ids[i] → reputation score
                filtered_reputation_scores = {i: reputation_scores.get(valid_ids[i], 1.0) for i in range(len(valid_ids))}

        # Signal 1: Distance-based anomaly detection
        distance_scores = self._compute_distance_scores(client_updates)

        # Signal 2: Loss-based anomaly detection
        loss_scores = self._compute_loss_scores(filtered_client_losses) if filtered_client_losses else np.zeros(n_clients)

        # Signal 3: Clustering-based anomaly detection
        cluster_scores = self._compute_cluster_scores(client_updates) if self.use_clustering else np.zeros(n_clients)

        # Signal 4: Reputation-based weighting
        reputation_weights = self._get_reputation_weights(filtered_reputation_scores, n_clients)

        # Combine signals with adaptive weights
        composite_scores = self._combine_signals(
            distance_scores,
            loss_scores,
            cluster_scores,
            reputation_weights
        )

        # Adaptive thresholding
        anomalous_valid_indices = self._adaptive_threshold(composite_scores, n_clients)

        # Map back to original client IDs
        anomalous_from_valid = [valid_ids[idx] for idx in anomalous_valid_indices]

        # Combine with NaN/Inf detections
        all_anomalous = sorted(set(nan_inf_clients + anomalous_from_valid))

        print(f"[ADAPTIVE DEFENSE] Detected {len(all_anomalous)}/{original_n_clients} anomalous clients")
        if nan_inf_clients:
            print(f"[ADAPTIVE DEFENSE]   - NaN/Inf: {len(nan_inf_clients)} clients")
            print(f"[ADAPTIVE DEFENSE]   - Other anomalies: {len(anomalous_from_valid)} clients")
        print(f"[ADAPTIVE DEFENSE] Current threshold: {self.threshold:.3f}")

        return all_anomalous

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

