"""
Committee-Based Defense Mechanisms for Federated Learning (Distributed)

This module implements distributed committee-based defense strategies:
1. Adaptive Committee Defense with multi-signal anomaly detection
2. CMFL (Committee-based Federated Learning) with rotating committees
3. Real-time anomaly detection with NaN/Inf filtering
4. Performance-based committee rotation
5. Reputation scoring within committees

Performance Optimizations (v2 - Speed Improvements):
- Vectorized distance computations with squared L2 (no sqrt)
- Cached reputation scores and distance vectors
- K-means clustering (10x faster than DBSCAN)
- Distributed consensus voting
- Removed malicious client training boost (equal training time)
- Optimized distance metrics (squared L2 instead of L2 with sqrt)
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, deque
from sklearn.cluster import KMeans  # OPTIMIZATION: K-means is much faster than DBSCAN
from sklearn.preprocessing import StandardScaler
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
                 initial_threshold: float = 2.0,  # Z-score multiplier: flag if > 2.0 std above global mean
                 max_exclude_frac: float = 0.50,  # Balanced to allow attacks some impact while still defending
                 consensus_threshold: float = 0.5,  # Lowered from 0.6 to require only majority (50%) instead of 60%
                 use_clustering: bool = True,
                 adaptive_tuning: bool = True,
                 min_committee_ratio: float = 0.30,  # NEW: Minimum 30% of participants
                 max_committee_ratio: float = 0.40,  # NEW: Maximum 40% of participants
                 aggregation_participation_frac: float = None,  # Fixed % of training clients for aggregation (None = pBFT)
                 aggregation_method: str = 'weighted_avg',  # Data-size weighted averaging
                 scoring_mode: str = 'fast',  # NEW: 'full', 'fast', or 'distance_only'
                 **aggregation_kwargs):  # FIXED: Added for aggregation parameters
        """
        Initialize Adaptive Committee Defense with dynamic committee sizing.

        Args:
            num_clients: Total number of clients
            committee_size: Number of committee members (auto-scaled if None)
            training_clients_per_round: Training clients activated per round (auto-scaled if None)
            committee_rotation_rounds: Rotate committee every N rounds
            initial_threshold: Initial threshold multiplier (ADAPTIVE feature)
            max_exclude_frac: Maximum fraction of clients to exclude
            consensus_threshold: Fraction of committee votes needed
            use_clustering: Whether to use DBSCAN clustering (deprecated - use scoring_mode instead)
            adaptive_tuning: Whether to auto-tune threshold (UNIQUE feature)
            min_committee_ratio: Minimum committee size as ratio of participants (default: 0.30 = 30%)
            max_committee_ratio: Maximum committee size as ratio of participants (default: 0.40 = 40%)
            scoring_mode: Performance mode ('full'=all signals, 'fast'=skip clustering [recommended], 'distance_only'=distance only)
        """
        self.num_clients = num_clients
        self.min_committee_ratio = min_committee_ratio
        self.max_committee_ratio = max_committee_ratio

        # Auto-scale based on PARTICIPATION model:
        # - Committee: ADAPTIVE between 20%-30% of participating clients
        # - Training: Remainder of participating clients
        if committee_size is None or training_clients_per_round is None:
            # CMFL paper: 40% participation per round (Section 6.3)
            total_participation = max(5, min(64, int(num_clients * 0.40)))

            # Committee: 40% of participating clients
            committee_size = max(2, int(total_participation * 0.40))

            # Training is the remainder
            training_clients_per_round = total_participation - committee_size

        self.committee_size = min(committee_size, num_clients)
        self.training_clients_per_round = min(training_clients_per_round, num_clients - self.committee_size)
        self.committee_rotation_rounds = committee_rotation_rounds
        self.threshold = initial_threshold
        self.max_exclude_frac = max_exclude_frac
        self.consensus_threshold = consensus_threshold
        self.use_clustering = use_clustering
        self.adaptive_tuning = adaptive_tuning
        self.aggregation_participation_frac = aggregation_participation_frac
        self.scoring_mode = scoring_mode  # NEW: Performance mode configuration

        # Initialize aggregation method (same as CMFL)
        self.aggregation_method = get_aggregation_method(aggregation_method, **aggregation_kwargs)

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

        # Tracking
        self.round_count = 0
        self.committee_history = [list(self.committee_members)]
        self.reputation_scores = {i: 1.0 for i in range(num_clients)}

        # Multi-Round Reputation Tracking (reduces False Positive Rate)
        # Two-tier detection system:
        # - STRONG signals (high anomaly score > 0.8) → Flag immediately (maintains Recall)
        # - WEAK signals (moderate score 0.5-0.8) → Require 3 out of 5 rounds (reduces FPR)
        self.suspicious_history = {i: [] for i in range(num_clients)}  # Track last 5 rounds per client
        self.SUSPICIOUS_THRESHOLD = 3  # Flag after 3 suspicious rounds out of last 5
        self.HISTORY_WINDOW = 5  # Track last 5 rounds
        self.STRONG_SIGNAL_THRESHOLD = 0.8  # Immediate flag if anomaly score > 0.8
        self.WEAK_SIGNAL_THRESHOLD = 0.5  # Multi-round if score > 0.5

        # Detection tracking
        self.detected_malicious = set()  # CUMULATIVE: Track clients detected as malicious across rounds

        # Per-round detection tracking for detailed analysis
        self.detection_history_per_round = []  # List of dicts with per-round metrics
        self.participants_per_round = []  # List of participating client IDs per round
        self.detected_per_round = []  # List of detected client IDs per round

        # Verbose logging: set to True for debug, False for speed
        self.verbose = False

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

        # FIXED: Ensure we rotate through ALL clients eventually
        # Use round-robin style to ensure every client gets evaluated
        activated = np.random.choice(available_idle, n_activate, replace=False)

        self.training_clients = set(activated)
        self.idle_clients -= self.training_clients

        if self.verbose:
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

        # CRITICAL FIX: Evaluate ALL active clients (training + committee), not just training
        # This prevents malicious clients in committee from avoiding detection
        clients_to_evaluate = list(training_updates.keys()) + list(committee_updates.keys())

        # OPTIMIZATION: Flatten updates ONCE, reuse for distance and cluster scoring
        precomputed = self._flatten_updates(all_updates)

        # OPTIMIZATION: Compute scores ONCE (inputs are identical for all committee members)
        distance_scores = self._compute_distance_scores(all_updates, precomputed=precomputed)

        # Conditional scoring based on performance mode
        if self.scoring_mode == 'distance_only':
            loss_scores = np.zeros(len(all_updates))
            cluster_scores = np.zeros(len(all_updates))
        elif self.scoring_mode == 'fast':
            loss_scores = self._compute_loss_scores(all_losses)
            cluster_scores = np.zeros(len(all_updates))
        else:  # 'full' mode
            loss_scores = self._compute_loss_scores(all_losses)
            cluster_scores = self._compute_cluster_scores(all_updates, precomputed=precomputed)

        reputation_weights = np.array([self.reputation_scores.get(cid, 1.0) for cid in all_client_ids])

        # UNIQUE: Different weight combination than CMFL
        composite_scores = self._combine_signals_adaptive(
            distance_scores,
            loss_scores,
            cluster_scores,
            reputation_weights
        )

        # Build the shared client score dict once
        shared_client_scores = {}
        for i, client_id in enumerate(all_client_ids):
            if client_id in clients_to_evaluate:
                shared_client_scores[client_id] = float(composite_scores[i])

        # Each committee member gets the same scores (they all see identical input data)
        for committee_id in self.committee_members:
            committee_scores[committee_id] = dict(shared_client_scores)  # shallow copy

        return committee_scores

    def pbft_consensus_adaptive(self,
                               committee_scores: Dict[int, Dict[int, float]],
                               operation: str = 'selection') -> Tuple[List[int], List[int]]:
        """
        pBFT-inspired consensus protocol for Adaptive Committee (with adaptive thresholds).

        Same as CMFLDefense.pbft_consensus but uses self.threshold (adaptive) instead of self.anomaly_threshold.

        Args:
            committee_scores: Dict mapping committee_id -> {client_id: score}
            operation: 'selection' for aggregation set or 'election' for new committee

        Returns:
            (selected_clients, flagged_clients) tuple
        """
        evaluated_client_ids = list(next(iter(committee_scores.values())).keys())
        committee_members = list(committee_scores.keys())
        n_committee = len(committee_members)

        # Consensus threshold: ≥⌊C/2⌋+1 responses needed
        min_consensus = (n_committee // 2) + 1

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            # Step 2: Select random primary committee client
            primary_id = np.random.choice(committee_members)
            replica_ids = [c for c in committee_members if c != primary_id]


            # Step 3: Each committee client decides its selection set independently
            committee_decisions = {}

            # Compute per-client mean scores from committee evaluations
            all_scores_per_client = defaultdict(list)
            for committee_id, scores in committee_scores.items():
                for client_id, score in scores.items():
                    all_scores_per_client[client_id].append(score)

            mean_scores = {cid: np.mean(scores) for cid, scores in all_scores_per_client.items()}
            all_mean_values = np.array(list(mean_scores.values()))

            # FIXED: Use GLOBAL score distribution for outlier detection
            # instead of per-client std across identical committee scores (which is always 0)
            global_mean = np.mean(all_mean_values) if len(all_mean_values) > 0 else 0.0
            global_std = np.std(all_mean_values) if len(all_mean_values) > 0 else 0.0
            global_median = np.median(all_mean_values) if len(all_mean_values) > 0 else 0.0
            global_mad = np.median(np.abs(all_mean_values - global_median)) if len(all_mean_values) > 0 else 0.0

            # Each committee member independently determines flagged clients
            for committee_id in committee_members:
                votes = []
                for client_id in evaluated_client_ids:
                    client_mean = mean_scores[client_id]

                    # Threshold 1: Z-score based (flag if > threshold std above global mean)
                    exceeds_zscore = False
                    if global_std > 1e-6:
                        z = (client_mean - global_mean) / global_std
                        exceeds_zscore = z > self.threshold  # default 2.0

                    # Threshold 2: MAD-based robust outlier detection
                    exceeds_mad = False
                    if global_mad > 1e-6:
                        mad_z = (client_mean - global_median) / (global_mad * 1.4826)
                        exceeds_mad = mad_z > 3.0

                    # Flag if EITHER meaningful threshold is exceeded
                    if exceeds_zscore or exceeds_mad:
                        votes.append(client_id)

                committee_decisions[committee_id] = set(votes)

            # Step 4: Primary creates its selection set
            primary_flagged = committee_decisions[primary_id]
            primary_selected = [cid for cid in evaluated_client_ids if cid not in primary_flagged]

            # Apply max_exclude_frac to primary's decision
            max_exclude = int(len(evaluated_client_ids) * self.max_exclude_frac)
            if len(primary_flagged) > max_exclude:
                # Sort by how many committee members flagged them
                vote_counts = defaultdict(int)
                for cid in evaluated_client_ids:
                    for committee_id in committee_members:
                        if cid in committee_decisions[committee_id]:
                            vote_counts[cid] += 1

                flagged_sorted = sorted(primary_flagged, key=lambda cid: vote_counts[cid], reverse=True)
                primary_flagged = set(flagged_sorted[:max_exclude])
                primary_selected = [cid for cid in evaluated_client_ids if cid not in primary_flagged]

            # Step 4: Primary broadcasts request to replicas
            request = {
                'type': 'Request',
                'primary_id': primary_id,
                'selected_set': set(primary_selected),
                'flagged_set': primary_flagged,
                'operation': operation,
                'timestamp': self.round_count
            }


            # Step 5: Replicas verify and reply
            replies = []
            for replica_id in replica_ids:
                # Replica checks if its decision matches primary's decision
                replica_flagged = committee_decisions[replica_id]
                replica_selected = set([cid for cid in evaluated_client_ids if cid not in replica_flagged])

                # Apply same max_exclude_frac constraint to replica
                if len(replica_flagged) > max_exclude:
                    vote_counts = defaultdict(int)
                    for cid in evaluated_client_ids:
                        for committee_id in committee_members:
                            if cid in committee_decisions[committee_id]:
                                vote_counts[cid] += 1

                    flagged_sorted = sorted(replica_flagged, key=lambda cid: vote_counts[cid], reverse=True)
                    replica_flagged = set(flagged_sorted[:max_exclude])
                    replica_selected = set([cid for cid in evaluated_client_ids if cid not in replica_flagged])

                # Check if replica's decision matches primary's
                matches = (replica_selected == request['selected_set'])

                if matches:
                    reply = {
                        'type': 'Reply',
                        'replica_id': replica_id,
                        'selected_set': replica_selected,
                        'flagged_set': replica_flagged,
                        'timestamp': self.round_count,
                        'matches': True
                    }
                    replies.append(reply)

            # Step 6: Primary checks for consensus (≥⌊C/2⌋+1 matching responses)
            matching_replies = len([r for r in replies if r['matches']])
            total_matching = matching_replies + 1  # +1 for primary itself

            if total_matching >= min_consensus:
                selected_clients = sorted(list(request['selected_set']))
                flagged_clients = sorted(list(request['flagged_set']))
                return selected_clients, flagged_clients
            else:
                retry_count += 1

        # Fallback: majority voting
        return self._fallback_majority_voting_adaptive(committee_decisions, evaluated_client_ids, max_exclude)

    def _fallback_majority_voting_adaptive(self,
                                          committee_decisions: Dict[int, set],
                                          evaluated_client_ids: List[int],
                                          max_exclude: int) -> Tuple[List[int], List[int]]:
        """Fallback to majority voting if pBFT consensus fails (adaptive version)."""
        n_committee = len(committee_decisions)
        min_votes = (n_committee // 2) + 1

        vote_counts = defaultdict(int)
        for client_id in evaluated_client_ids:
            for committee_id, flagged_set in committee_decisions.items():
                if client_id in flagged_set:
                    vote_counts[client_id] += 1

        flagged_clients = [cid for cid, votes in vote_counts.items() if votes >= min_votes]

        # Apply max_exclude_frac
        if len(flagged_clients) > max_exclude:
            flagged_sorted = sorted(flagged_clients, key=lambda cid: vote_counts[cid], reverse=True)
            flagged_clients = flagged_sorted[:max_exclude]

        selected_clients = [cid for cid in evaluated_client_ids if cid not in flagged_clients]

        return selected_clients, flagged_clients

    def adaptive_committee_round(self,
                                 client_updates: Dict[int, Dict],
                                 client_losses: Dict[int, float],
                                 malicious_client_ids: Optional[List[int]] = None,
                                 all_clients: Optional[List] = None,
                                 use_anomaly_detection: bool = True,
                                 client_data_sizes: Optional[Dict[int, int]] = None) -> Tuple[Dict, List[int], Dict]:
        """
        Execute one complete Adaptive Committee round.

        Follows committee workflow with ADAPTIVE threshold mechanism.

        Args:
            client_updates: Dict mapping client_id to model updates
            client_losses: Dict mapping client_id to losses
            malicious_client_ids: Optional list of known malicious client IDs (for diagnostics)
            all_clients: Optional list of all client objects (for detection metrics calculation)
            use_anomaly_detection: If False, skip detection and aggregate all updates (baseline/attack scenario)
        """
        self.round_count += 1

        # Separate training and committee updates
        training_updates = {cid: client_updates[cid] for cid in self.training_clients if cid in client_updates}
        committee_updates = {cid: client_updates[cid] for cid in self.committee_members if cid in client_updates}

        training_losses = {cid: client_losses[cid] for cid in self.training_clients if cid in client_losses}
        committee_losses = {cid: client_losses[cid] for cid in self.committee_members if cid in client_losses}

        # OPTION A: Conditional anomaly detection
        if use_anomaly_detection:
            # Step 3: Scoring with ADAPTIVE thresholds
            committee_scores = self.committee_scoring_adaptive(
                training_updates,
                committee_updates,
                training_losses,
                committee_losses
            )

            # Store mean anomaly scores for election (HYBRID strategy)
            self._last_training_anomaly_scores = {}
            # Get scores for ALL evaluated clients (training + committee)
            all_evaluated_ids = list(next(iter(committee_scores.values())).keys()) if committee_scores else []
            for client_id in all_evaluated_ids:
                # Average score across all committee members
                scores = [committee_scores[cid][client_id] for cid in committee_scores.keys() if client_id in committee_scores[cid]]
                if scores:
                    self._last_training_anomaly_scores[client_id] = np.mean(scores)

            # Step 4: Client selection — fixed fraction OR pBFT consensus
            if self.aggregation_participation_frac is not None:
                # Fixed participation: sort training clients by anomaly score (ascending = least suspicious first)
                n_select = max(1, int(len(training_updates) * self.aggregation_participation_frac))
                sorted_training = sorted(
                    training_updates.keys(),
                    key=lambda cid: self._last_training_anomaly_scores.get(cid, 0.0)
                )
                selected_clients = sorted_training[:n_select]
                flagged_clients = sorted_training[n_select:]
            else:
                # Original pBFT consensus path
                selected_clients, flagged_clients = self.pbft_consensus_adaptive(committee_scores, operation='selection')

            # Filter out committee members from aggregation
            selected_clients = [cid for cid in selected_clients if cid not in self.committee_members]
        else:
            # NO DETECTION: Aggregate all training clients without filtering
            selected_clients = list(training_updates.keys())
            flagged_clients = []
            committee_scores = {}

        # Step 5: Aggregation - Use ONLY selected training clients
        aggregation_clients = selected_clients
        aggregated_params = self._aggregate_with_reputation(
            aggregation_clients,
            client_updates,
            client_data_sizes=client_data_sizes
        )

        # Two-Tier Multi-Round Detection (maintains Recall, reduces FPR)
        # Extract anomaly scores from committee evaluations
        client_anomaly_scores = {}
        if committee_scores:
            # Calculate mean anomaly score from all committee members
            all_scores_per_client = defaultdict(list)
            for committee_id, scores in committee_scores.items():
                for client_id, score in scores.items():
                    all_scores_per_client[client_id].append(score)
            client_anomaly_scores = {cid: np.mean(scores) for cid, scores in all_scores_per_client.items()}

        truly_malicious = []
        strong_signal_flags = []
        weak_signal_flags = []

        for client_id in flagged_clients:
            anomaly_score = client_anomaly_scores.get(client_id, 0.0)

            # STRONG SIGNAL: Immediate flagging (maintains high Recall)
            if anomaly_score >= self.STRONG_SIGNAL_THRESHOLD:
                self.detected_malicious.add(client_id)
                truly_malicious.append(client_id)
                strong_signal_flags.append(client_id)
            else:
                # WEAK SIGNAL: Multi-round tracking (reduces FPR)
                self.suspicious_history[client_id].append(1)
                if len(self.suspicious_history[client_id]) > self.HISTORY_WINDOW:
                    self.suspicious_history[client_id].pop(0)

                suspicious_count = sum(self.suspicious_history[client_id])

                if suspicious_count >= self.SUSPICIOUS_THRESHOLD:
                    self.detected_malicious.add(client_id)
                    truly_malicious.append(client_id)
                    weak_signal_flags.append(client_id)

        # Update history for non-flagged clients (mark as benign this round)
        for client_id in selected_clients:
            if client_id not in flagged_clients:
                self.suspicious_history[client_id].append(0)
                if len(self.suspicious_history[client_id]) > self.HISTORY_WINDOW:
                    self.suspicious_history[client_id].pop(0)

        # Update reputation (use truly_malicious instead of flagged_clients for stronger penalty)
        self._update_reputation(selected_clients, truly_malicious)

        # CRITICAL FIX: Calculate detection metrics BEFORE clearing training_clients
        # This ensures self.training_clients and self.committee_members contain correct participants
        detection_metrics_dict = None
        if all_clients is not None and use_anomaly_detection:
            round_metrics = self.calculate_round_detection_metrics(
                all_clients,
                round_num=self.round_count,
                flagged_this_round=flagged_clients
            )
            detection_metrics_dict = round_metrics

        # ADAPTIVE COMMITTEE SIZE: Adjust based on threat level
        # Calculate threat level from detection rate
        total_evaluated = len(selected_clients) + len(flagged_clients)
        if total_evaluated > 0:
            detection_rate = len(flagged_clients) / total_evaluated
            # Threat level: 0.0 (no detections) to 1.0 (high detection rate)
            # Map detection_rate to threat_level (cap at 50% detection = max threat)
            threat_level = min(1.0, detection_rate / 0.5)  # 50% detection → threat = 1.0
            self._adjust_committee_size_adaptive(threat_level)

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

        # Add detection metrics if they were calculated
        if detection_metrics_dict is not None:
            metrics['detection_metrics'] = detection_metrics_dict

        return aggregated_params, flagged_clients, metrics

    def _aggregate_with_reputation(self,
                                   selected_client_ids: List[int],
                                   all_updates: Dict[int, Dict],
                                   client_data_sizes: Optional[Dict[int, int]] = None) -> Dict:
        """Aggregate using data-size weighted averaging."""
        if len(selected_client_ids) == 0:
            raise ValueError("Cannot aggregate with no selected clients")

        selected_updates = [all_updates[cid] for cid in selected_client_ids]

        if client_data_sizes:
            weights = [client_data_sizes.get(cid, 1) for cid in selected_client_ids]
        else:
            weights = [1.0] * len(selected_client_ids)

        aggregated_params = self.aggregation_method.aggregate(selected_updates, weights)

        return aggregated_params

    def _elect_new_committee(self, training_client_ids: List[int], training_losses: Dict[int, float]) -> List[int]:
        """
        Elect new committee using HYBRID strategy (paper + enhanced adaptive).

        Same approach as CMFLDefense but for adaptive committee.
        """
        # If we don't have stored anomaly scores, fall back to loss-based ranking
        if not hasattr(self, '_last_training_anomaly_scores') or not self._last_training_anomaly_scores:
            anomaly_scores = {cid: training_losses.get(cid, float('inf')) for cid in training_client_ids}
        else:
            anomaly_scores = {cid: self._last_training_anomaly_scores.get(cid, float('inf'))
                            for cid in training_client_ids}

        # Step 1: Sort by anomaly score
        sorted_by_anomaly = sorted(anomaly_scores.items(), key=lambda x: x[1])

        # Step 2: Select MIDDLE-ranked clients (paper's security approach)
        n = len(sorted_by_anomaly)
        if n <= self.committee_size:
            middle_candidates = [cid for cid, _ in sorted_by_anomaly]
        else:
            lower_bound = max(1, int(n * 0.2))
            upper_bound = max(lower_bound + 1, int(n * 0.8))
            middle_candidates = [cid for cid, score in sorted_by_anomaly[lower_bound:upper_bound]]

        # Step 3: Among middle candidates, rank by reputation and loss
        election_scores = {}
        for client_id in middle_candidates:
            loss = training_losses.get(client_id, float('inf'))
            reputation = self.reputation_scores.get(client_id, 0.5)
            score = reputation / (loss + 1.0)
            election_scores[client_id] = score

        sorted_candidates = sorted(election_scores.items(), key=lambda x: x[1], reverse=True)
        n_elect = min(self.committee_size, len(sorted_candidates))
        new_committee = [cid for cid, score in sorted_candidates[:n_elect]]

        return new_committee

    def _step_down_committee(self):
        """Move committee members to idle pool."""
        old_committee = list(self.committee_members)
        self.idle_clients.update(old_committee)

    def _adjust_committee_size_adaptive(self, threat_level: float):
        """
        Dynamically adjust committee size between min (20%) and max (30%) based on threat level.

        Args:
            threat_level: Float between 0.0 (no threat) and 1.0 (high threat)
                         Based on detection rate, false positives, etc.
        """
        # Calculate total participants
        total_participants = self.committee_size + self.training_clients_per_round

        # Calculate target committee size based on threat level
        # threat_level 0.0 → 20% committee (minimum)
        # threat_level 1.0 → 30% committee (maximum)
        target_ratio = self.min_committee_ratio + (threat_level * (self.max_committee_ratio - self.min_committee_ratio))
        target_committee_size = max(2, int(total_participants * target_ratio))

        # Ensure we don't exceed bounds
        min_committee = max(2, int(total_participants * self.min_committee_ratio))
        max_committee = max(min_committee, int(total_participants * self.max_committee_ratio))
        target_committee_size = max(min_committee, min(max_committee, target_committee_size))

        # Adjust sizes if needed
        if target_committee_size != self.committee_size:
            old_size = self.committee_size
            self.committee_size = target_committee_size
            self.training_clients_per_round = total_participants - self.committee_size


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

    def calculate_round_detection_metrics(self, all_clients: List, round_num: int = 0,
                                          flagged_this_round: Optional[List[int]] = None):
        """
        Calculate and store detection metrics for the current round.

        This tracks per-round metrics to allow different aggregation strategies.

        Args:
            all_clients: List of all client objects (to check is_malicious attribute)
            round_num: Current round number
            flagged_this_round: List of client IDs flagged in THIS specific round (recommended).
                               If None, falls back to cumulative detections (less accurate for per-round metrics).
        """
        # Get participating clients this round
        participants = list(self.training_clients | self.committee_members)

        # Identify malicious vs benign among participants
        malicious_participants = [
            cid for cid in participants
            if cid < len(all_clients) and getattr(all_clients[cid], 'is_malicious', False)
        ]
        benign_participants = [
            cid for cid in participants
            if cid < len(all_clients) and not getattr(all_clients[cid], 'is_malicious', False)
        ]

        # FIXED: Use per-round flagged clients if provided, else fall back to cumulative
        # Per-round flagged clients give TRUE per-round metrics
        # Cumulative detections are only used for backward compatibility
        if flagged_this_round is not None:
            detected_ids = flagged_this_round  # ✅ Per-round detections (CORRECT)
        else:
            detected_ids = list(self.detected_malicious)  # Fallback to cumulative (LEGACY)

        # Calculate TRUE POSITIVES: Malicious clients that were detected
        true_positives = [cid for cid in malicious_participants if cid in detected_ids]

        # Calculate FALSE POSITIVES: Benign clients that were detected
        false_positives = [cid for cid in benign_participants if cid in detected_ids]

        # Calculate FALSE NEGATIVES: Malicious clients that were NOT detected
        false_negatives = [cid for cid in malicious_participants if cid not in detected_ids]

        # Calculate TRUE NEGATIVES: Benign clients that were NOT detected
        true_negatives = [cid for cid in benign_participants if cid not in detected_ids]

        # Calculate metrics
        tp_count = len(true_positives)
        fp_count = len(false_positives)
        fn_count = len(false_negatives)
        tn_count = len(true_negatives)

        precision = (tp_count / (tp_count + fp_count) * 100) if (tp_count + fp_count) > 0 else 0.0
        recall = (tp_count / (tp_count + fn_count) * 100) if (tp_count + fn_count) > 0 else 0.0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        fpr = (fp_count / (fp_count + tn_count) * 100) if (fp_count + tn_count) > 0 else 0.0
        dacc = ((tp_count + tn_count) / len(participants) * 100) if len(participants) > 0 else 0.0

        # Store per-round metrics
        round_metrics = {
            'round': round_num,
            'participants': participants,
            'malicious_participants': malicious_participants,
            'benign_participants': benign_participants,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives,
            'tp_count': tp_count,
            'fp_count': fp_count,
            'tn_count': tn_count,
            'fn_count': fn_count,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'fpr': fpr,
            'dacc': dacc,
            'detection_rate': recall,  # Detection rate = Recall
            'num_participants': len(participants),
            'num_malicious_participants': len(malicious_participants),
            'num_benign_participants': len(benign_participants)
        }

        self.detection_history_per_round.append(round_metrics)
        self.participants_per_round.append(participants)
        self.detected_per_round.append(detected_ids.copy())

        return round_metrics

    def get_cumulative_detection_metrics(self, all_clients: List) -> Dict:
        """
        Calculate CUMULATIVE detection metrics based on UNIQUE clients across all rounds.

        CORRECT LOGIC:
        - Per-round metrics: Computed on participants in that specific round ONLY
        - Cumulative metrics: Computed on UNIQUE clients across ALL rounds
        - Each unique client counted ONCE:
          * If EVER detected → TP (malicious) or FP (benign)
          * If NEVER detected → FN (malicious) or TN (benign)

        Args:
            all_clients: List of all client objects

        Returns:
            Dict with cumulative metrics based on unique clients
        """
        # Get all unique participants across all rounds
        all_participants = set()
        for participants in self.participants_per_round:
            all_participants.update(participants)

        # Get all clients that were EVER detected in ANY round
        ever_detected = set()
        for detected_list in self.detected_per_round:
            ever_detected.update(detected_list)

        # Separate unique participants into malicious and benign
        malicious_seen = set([
            cid for cid in all_participants
            if cid < len(all_clients) and getattr(all_clients[cid], 'is_malicious', False)
        ])
        benign_seen = set([
            cid for cid in all_participants
            if cid < len(all_clients) and not getattr(all_clients[cid], 'is_malicious', False)
        ])

        # Calculate confusion matrix based on UNIQUE clients
        # TP: Malicious clients that were EVER detected
        tp_clients = malicious_seen & ever_detected
        tp_count = len(tp_clients)

        # FP: Benign clients that were EVER detected
        fp_clients = benign_seen & ever_detected
        fp_count = len(fp_clients)

        # FN: Malicious clients that were NEVER detected
        fn_clients = malicious_seen - ever_detected
        fn_count = len(fn_clients)

        # TN: Benign clients that were NEVER detected
        tn_clients = benign_seen - ever_detected
        tn_count = len(tn_clients)

        # Total unique participants
        total_unique = len(all_participants)

        # Calculate metrics
        precision = (tp_count / (tp_count + fp_count) * 100) if (tp_count + fp_count) > 0 else 0.0
        recall = (tp_count / (tp_count + fn_count) * 100) if (tp_count + fn_count) > 0 else 0.0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        fpr = (fp_count / (fp_count + tn_count) * 100) if (fp_count + tn_count) > 0 else 0.0
        fnr = 100.0 - recall  # FNR = 100 - Recall
        dacc = ((tp_count + tn_count) / total_unique * 100) if total_unique > 0 else 0.0

        return {
            'aggregation_method': 'cumulative_unique_clients',
            'total_unique_participants': total_unique,
            'total_malicious_seen': len(malicious_seen),
            'total_benign_seen': len(benign_seen),
            'tp_count': tp_count,
            'fp_count': fp_count,
            'tn_count': tn_count,
            'fn_count': fn_count,
            'tp_clients': sorted(list(tp_clients)),  # For debugging
            'fp_clients': sorted(list(fp_clients)),  # For debugging
            'fn_clients': sorted(list(fn_clients)),  # For debugging
            'tn_clients': sorted(list(tn_clients)),  # For debugging
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'fpr': fpr,
            'fnr': fnr,
            'dacc': dacc,
            'detection_rate': recall
        }

    def get_averaged_detection_metrics(self) -> Dict:
        """
        Calculate AVERAGED detection metrics across rounds.

        Simple average of per-round metrics. Less representative than cumulative
        but useful for understanding per-round variance.

        Returns:
            Dict with averaged metrics
        """
        if not self.detection_history_per_round:
            return {}

        num_rounds = len(self.detection_history_per_round)

        avg_precision = sum(r['precision'] for r in self.detection_history_per_round) / num_rounds
        avg_recall = sum(r['recall'] for r in self.detection_history_per_round) / num_rounds
        avg_f1 = sum(r['f1_score'] for r in self.detection_history_per_round) / num_rounds
        avg_fpr = sum(r['fpr'] for r in self.detection_history_per_round) / num_rounds
        avg_dacc = sum(r['dacc'] for r in self.detection_history_per_round) / num_rounds

        return {
            'aggregation_method': 'simple_average',
            'num_rounds': num_rounds,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': avg_f1,
            'fpr': avg_fpr,
            'dacc': avg_dacc,
            'detection_rate': avg_recall
        }

    def get_micro_averaged_detection_metrics(self, all_clients: Optional[List] = None) -> Dict:
        """
        Calculate MICRO-AVERAGED detection metrics by summing per-round
        confusion matrix counts across all rounds.

        This is the standard approach in FL papers:
        - Sum TP, FP, TN, FN across all rounds
        - Compute metrics from the totals

        Why this is correct:
        - CMFL flags the bottom (1-alpha)% of training clients EVERY round
        - With client rotation over 100 rounds, the cumulative "EVER detected"
          approach would mark nearly every client as detected (FPR → 100%)
        - Micro-averaging treats each round's classification independently,
          giving the true per-instance detection performance

        Args:
            all_clients: Optional list of all client objects (for unique participant counts)

        Returns:
            Dict with micro-averaged metrics
        """
        if not self.detection_history_per_round:
            return {}

        total_tp = sum(r['tp_count'] for r in self.detection_history_per_round)
        total_fp = sum(r['fp_count'] for r in self.detection_history_per_round)
        total_tn = sum(r['tn_count'] for r in self.detection_history_per_round)
        total_fn = sum(r['fn_count'] for r in self.detection_history_per_round)
        total = total_tp + total_fp + total_tn + total_fn

        precision = (total_tp / (total_tp + total_fp) * 100) if (total_tp + total_fp) > 0 else 0.0
        recall = (total_tp / (total_tp + total_fn) * 100) if (total_tp + total_fn) > 0 else 0.0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        fpr = (total_fp / (total_fp + total_tn) * 100) if (total_fp + total_tn) > 0 else 0.0
        fnr = 100.0 - recall
        dacc = ((total_tp + total_tn) / total * 100) if total > 0 else 0.0

        # Unique participant counts (for informational purposes)
        all_participants = set()
        for participants in self.participants_per_round:
            all_participants.update(participants)

        if all_clients is not None:
            malicious_seen = len([
                cid for cid in all_participants
                if cid < len(all_clients) and getattr(all_clients[cid], 'is_malicious', False)
            ])
            benign_seen = len([
                cid for cid in all_participants
                if cid < len(all_clients) and not getattr(all_clients[cid], 'is_malicious', False)
            ])
        else:
            malicious_seen = 0
            benign_seen = 0

        return {
            'aggregation_method': 'micro_averaged',
            'num_rounds': len(self.detection_history_per_round),
            'total_unique_participants': len(all_participants),
            'total_malicious_seen': malicious_seen,
            'total_benign_seen': benign_seen,
            'tp_count': total_tp,
            'fp_count': total_fp,
            'tn_count': total_tn,
            'fn_count': total_fn,
            'total_classifications': total,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'fpr': fpr,
            'fnr': fnr,
            'dacc': dacc,
            'detection_rate': recall
        }

    def get_weighted_averaged_detection_metrics(self) -> Dict:
        """
        Calculate WEIGHTED AVERAGE detection metrics across rounds.

        Weights each round by the number of participants, giving more weight
        to rounds with more clients participating.

        Returns:
            Dict with weighted averaged metrics
        """
        if not self.detection_history_per_round:
            return {}

        total_participants = sum(r['num_participants'] for r in self.detection_history_per_round)

        if total_participants == 0:
            return self.get_averaged_detection_metrics()

        weighted_precision = sum(
            r['precision'] * r['num_participants']
            for r in self.detection_history_per_round
        ) / total_participants

        weighted_recall = sum(
            r['recall'] * r['num_participants']
            for r in self.detection_history_per_round
        ) / total_participants

        weighted_f1 = sum(
            r['f1_score'] * r['num_participants']
            for r in self.detection_history_per_round
        ) / total_participants

        weighted_fpr = sum(
            r['fpr'] * r['num_participants']
            for r in self.detection_history_per_round
        ) / total_participants

        weighted_dacc = sum(
            r['dacc'] * r['num_participants']
            for r in self.detection_history_per_round
        ) / total_participants

        return {
            'aggregation_method': 'weighted_average',
            'total_participant_instances': total_participants,
            'num_rounds': len(self.detection_history_per_round),
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1_score': weighted_f1,
            'fpr': weighted_fpr,
            'dacc': weighted_dacc,
            'detection_rate': weighted_recall
        }

    def print_final_detection_summary(self, all_clients: List):
        """
        Print final micro-averaged detection metrics summary after all rounds complete.

        Uses micro-averaging (sum per-round TP/FP/TN/FN counts) which is the
        standard approach in FL papers and avoids the inflation problem of
        cumulative "EVER detected" logic.

        Args:
            all_clients: List of all client objects
        """
        print("\n" + "="*80)
        print("FINAL DETECTION METRICS SUMMARY (Micro-Averaged)")
        print("="*80)

        if not self.detection_history_per_round:
            print("No detection metrics available (no rounds with detection executed)")
            return

        # Get micro-averaged metrics (sum per-round counts — standard FL approach)
        metrics = self.get_micro_averaged_detection_metrics(all_clients)

        # Display Metrics
        print(f"\n  Evaluation Method: Micro-averaged across {metrics['num_rounds']} rounds")
        print(f"  - Total unique participants: {metrics['total_unique_participants']}")
        print(f"  - Malicious participants seen: {metrics['total_malicious_seen']}")
        print(f"  - Honest participants seen: {metrics['total_benign_seen']}")
        print(f"  - Total classifications (sum across rounds): {metrics['total_classifications']}")

        print(f"\n{'='*80}")
        print("CONFUSION MATRIX (summed across rounds)")
        print(f"{'='*80}")
        print(f"{'':25} {'Predicted Malicious':20} {'Predicted Honest':20}")
        print("-" * 80)
        tp_str = f"{metrics['tp_count']} (TP)"
        fn_str = f"{metrics['fn_count']} (FN)"
        fp_str = f"{metrics['fp_count']} (FP)"
        tn_str = f"{metrics['tn_count']} (TN)"
        print(f"{'Actual Malicious':25} {tp_str:20} {fn_str:20}")
        print(f"{'Actual Honest':25} {fp_str:20} {tn_str:20}")

        print(f"\n{'='*80}")
        print("DETECTION METRICS")
        print(f"{'='*80}")
        print(f"  Precision:             {metrics['precision']:>6.2f}%  (TP / (TP + FP))")
        print(f"  Recall (TPR):          {metrics['recall']:>6.2f}%  (TP / (TP + FN))")
        print(f"  Detection Rate:        {metrics['detection_rate']:>6.2f}%  (same as Recall)")
        print(f"  False Positive Rate:   {metrics['fpr']:>6.2f}%  (FP / (FP + TN))")
        print(f"  False Negative Rate:   {metrics['fnr']:>6.2f}%  (FN / (FN + TP))")
        print(f"  Detection Accuracy:    {metrics['dacc']:>6.2f}%  ((TP + TN) / Total)")
        print(f"  F1 Score:              {metrics['f1_score']:>6.2f}%")

        print("\n" + "="*80)

    def _combine_signals_adaptive(self,
                                  distance_scores: np.ndarray,
                                  loss_scores: np.ndarray,
                                  cluster_scores: np.ndarray,
                                  reputation_weights: np.ndarray) -> np.ndarray:
        """
        Combine signals with ADAPTIVE weighting strategy optimized for two-tier detection.

        Two-tier system:
        - Strong signals (composite > 0.8): Immediate flagging
        - Weak signals (0.5 < composite < 0.8): Multi-round tracking

        Optimized weights to better distinguish true attacks from Non-IID variance:
        - 0.50 distance (PRIMARY signal - model update divergence)
        - 0.30 loss (SECONDARY signal - prediction quality)
        - 0.15 cluster (TERTIARY signal - isolation from majority)
        - 0.05 reputation (MINIMAL - builds over time)
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

        # OPTIMIZED: Weights tuned for two-tier detection
        # Distance + Loss = 80% of signal (primary attack indicators)
        # Clustering + Reputation = 20% (secondary indicators)
        composite = (
            0.50 * norm_distance +      # PRIMARY: Model update divergence (strong attack signal)
            0.30 * norm_loss +          # SECONDARY: Prediction quality (data poisoning)
            0.15 * norm_cluster +       # TERTIARY: Isolation (confirms but not primary)
            0.05 * norm_reputation      # MINIMAL: Historical behavior (builds slowly)
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
        # NaN/Inf is a STRONG signal, so we flag immediately (no multi-round waiting)
        if nan_inf_clients:
            # Update detection stats
            for client_id in nan_inf_clients:
                self.detected_malicious.add(client_id)
                # Mark entire history as suspicious (strong signal)
                self.suspicious_history[client_id] = [1] * self.HISTORY_WINDOW

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

        # Flatten once, reuse for distance and cluster scoring
        precomputed = self._flatten_updates(client_updates)

        # Signal 1: Distance-based anomaly detection
        distance_scores = self._compute_distance_scores(client_updates, precomputed=precomputed)

        # Signal 2: Loss-based anomaly detection
        loss_scores = self._compute_loss_scores(filtered_client_losses) if filtered_client_losses else np.zeros(n_clients)

        # Signal 3: Clustering-based anomaly detection
        cluster_scores = self._compute_cluster_scores(client_updates, precomputed=precomputed) if self.use_clustering else np.zeros(n_clients)

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

    def _flatten_updates(self, client_updates: List[Dict]):
        """Flatten all model update dicts to numpy vectors ONCE.
        Returns (vectors_array, nan_mask) tuple for reuse by scoring methods."""
        vectors = []
        nan_mask = []
        for update in client_updates:
            vec = torch.cat([v.flatten() for v in update.values()]).cpu().numpy()
            has_nan_inf = not np.isfinite(vec).all()
            nan_mask.append(has_nan_inf)
            if has_nan_inf:
                vec = np.nan_to_num(vec, nan=1e10, posinf=1e10, neginf=-1e10)
            vectors.append(vec)
        return np.array(vectors) if vectors else np.array([]).reshape(0, 0), nan_mask

    def _compute_distance_scores(self, client_updates: List[Dict], precomputed=None) -> np.ndarray:
        """Original CMFL paper approach: cosine similarity to global update.

        Computes a contribution score per client as the cosine similarity between
        the client's update vector and the mean (global) update direction.
        Clients whose updates diverge from the global direction get high anomaly
        scores.  Complexity: O(n × d) instead of the old O(n² × d) pairwise method.

        Reference: CMFL paper — contribution-based client scoring.
        """
        n_clients = len(client_updates)

        if n_clients == 0:
            return np.array([])

        if n_clients == 1:
            return np.array([0.0])

        # Use precomputed flattened vectors if available
        if precomputed is not None:
            vectors, nan_mask = precomputed
        else:
            vectors, nan_mask = self._flatten_updates(client_updates)

        # Global update direction = mean of all client update vectors
        global_update = np.mean(vectors, axis=0)  # shape: (d,)

        # Vectorized cosine similarity between each client and the global direction
        global_norm = np.linalg.norm(global_update) + 1e-10
        client_norms = np.linalg.norm(vectors, axis=1) + 1e-10  # shape: (n_clients,)
        cos_sims = vectors @ global_update / (client_norms * global_norm)  # shape: (n_clients,)
        # Convert similarity to anomaly score: lower similarity = higher anomaly
        # cos_sim ∈ [-1, 1]; anomaly = 1 - cos_sim ∈ [0, 2]
        scores = 1.0 - cos_sims

        # Clients with NaN/Inf get maximum anomaly score
        scores = np.where(nan_mask, np.max(scores) + 1.0 if len(scores) > 0 else 1.0, scores)

        # Handle case where all scores are identical
        if np.std(scores) > 1e-6:
            scores = (scores - np.mean(scores)) / np.std(scores)
        else:
            scores = np.zeros(n_clients)

        return scores

    # ----- COMMENTED OUT: Old pairwise L2 distance approach -----
    # Kept for potential future study.  Complexity: O(n² × d) — creates an
    # (n, n, d) tensor which is very expensive for large models.
    #
    # def _compute_distance_scores_pairwise(self, client_updates, precomputed=None):
    #     """Pairwise L2 distance-based anomaly scores (old approach)."""
    #     n_clients = len(client_updates)
    #     if n_clients == 0:
    #         return np.array([])
    #     if n_clients == 1:
    #         return np.array([0.0])
    #     if precomputed is not None:
    #         vectors, nan_mask = precomputed
    #     else:
    #         vectors, nan_mask = self._flatten_updates(client_updates)
    #     diff = vectors[:, None, :] - vectors[None, :, :]
    #     distance_matrix = (diff ** 2).sum(axis=2)
    #     distance_matrix = np.nan_to_num(distance_matrix, nan=1e20, posinf=1e20)
    #     np.fill_diagonal(distance_matrix, np.inf)
    #     scores = np.median(distance_matrix, axis=1)
    #     scores = np.where(nan_mask, np.max(scores) + 1.0 if len(scores) > 0 else 1.0, scores)
    #     if np.std(scores) > 1e-6:
    #         scores = (scores - np.mean(scores)) / np.std(scores)
    #     else:
    #         scores = np.zeros(n_clients)
    #     return scores

    def _compute_loss_scores(self, client_losses: List[float]) -> np.ndarray:
        """Compute loss-based anomaly scores (low loss = suspicious)."""
        losses = np.array(client_losses)

        # Normalize losses
        if np.std(losses) > 1e-6:
            normalized = (losses - np.mean(losses)) / np.std(losses)
            # Negative because lower loss is more suspicious
            return -normalized

        return np.zeros_like(losses)

    def _compute_cluster_scores(self, client_updates: List[Dict], precomputed=None) -> np.ndarray:
        """OPTIMIZED: Use K-means clustering for fast outlier detection (10x faster than DBSCAN)."""
        n_clients = len(client_updates)

        if n_clients < 5:
            return np.zeros(n_clients)

        try:
            # Use precomputed flattened vectors if available
            if precomputed is not None:
                X, nan_mask = precomputed
            else:
                X, nan_mask = self._flatten_updates(client_updates)

            # If any clients have NaN/Inf, mark them immediately as outliers
            if any(nan_mask):
                print(f"[WARN] {sum(nan_mask)}/{n_clients} clients have NaN/Inf values - marking as outliers")
                return np.array([1.0 if has_nan else 0.0 for has_nan in nan_mask])

            # Normalize for better clustering (only if no NaN/Inf)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Final NaN check after scaling
            if not np.isfinite(X_scaled).all():
                print(f"[WARN] NaN/Inf detected after scaling - skipping clustering")
                return np.zeros(n_clients)

            # OPTIMIZATION: Use K-means (10x faster than DBSCAN)
            n_clusters = min(2, n_clients)
            kmeans = KMeans(n_clusters=n_clusters, n_init=3, max_iter=10, random_state=42)
            labels = kmeans.fit_predict(X_scaled)

            # Vectorized: distance from each point to its assigned cluster center
            assigned_centers = kmeans.cluster_centers_[labels]  # shape: (n_clients, d)
            distances = np.linalg.norm(X_scaled - assigned_centers, axis=1)  # shape: (n_clients,)

            # Outliers are points far from their cluster center (top 30%)
            threshold = np.percentile(distances, 70)
            scores = (distances > threshold).astype(float)
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
        """Combine multiple anomaly signals (optimized for two-tier detection)."""
        # OPTIMIZED: Same weights as _combine_signals_adaptive for consistency
        w_distance = 0.50  # PRIMARY: Model update divergence (strong attack signal)
        w_loss = 0.30     # SECONDARY: Prediction quality (data poisoning)
        w_cluster = 0.15  # TERTIARY: Isolation (confirms but not primary)
        w_reputation = 0.05  # MINIMAL: Historical behavior (builds slowly)

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

        # If we're detecting too few, lower threshold (faster adaptation for better detection)
        if detection_rate < 0.10 and self.threshold > 0.5:
            self.threshold *= 0.95  # FIXED: Faster adaptation (was 0.97)

        # If we're detecting too many, raise threshold (slower adaptation)
        elif detection_rate > self.max_exclude_frac * 0.8:
            self.threshold *= 1.03

        # FIXED: Clamp threshold to new range (0.5 to 2.5 instead of 2.0 to 4.0)
        self.threshold = max(0.5, min(2.5, self.threshold))

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

        return aggregated_params


# =============================================================================
# Ensemble Defense
# =============================================================================

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

        # Per-round detection tracking for detailed analysis
        self.detection_history_per_round = []
        self.participants_per_round = []
        self.detected_per_round = []

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
        dist_matrix = cdist(training_matrix, committee_matrix, metric='euclidean')
        sq_dist_matrix = dist_matrix ** 2  # shape: (T, C)

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

    # Detection metric methods — shared with AdaptiveCommitteeDefense
    calculate_round_detection_metrics = AdaptiveCommitteeDefense.calculate_round_detection_metrics
    get_cumulative_detection_metrics = AdaptiveCommitteeDefense.get_cumulative_detection_metrics
    get_micro_averaged_detection_metrics = AdaptiveCommitteeDefense.get_micro_averaged_detection_metrics
    get_averaged_detection_metrics = AdaptiveCommitteeDefense.get_averaged_detection_metrics
    get_weighted_averaged_detection_metrics = AdaptiveCommitteeDefense.get_weighted_averaged_detection_metrics
    print_final_detection_summary = AdaptiveCommitteeDefense.print_final_detection_summary


def get_defense(strategy: str = "adaptive_committee", num_clients: int = 10, **kwargs):
    """
    Factory function to create committee-based defense instances.

    Args:
        strategy: Defense strategy ('adaptive_committee', 'cmfl')
        num_clients: Number of clients in the system
        **kwargs: Additional arguments for specific defenses

    Returns:
        Defense instance
    """
    if strategy == "adaptive_committee":
        return AdaptiveCommitteeDefense(num_clients=num_clients, **kwargs)
    elif strategy == "cmfl":
        return CMFLDefense(num_clients, **kwargs)
    else:
        raise ValueError(f"Unknown committee defense strategy: {strategy}. Use 'adaptive_committee' or 'cmfl'.")
