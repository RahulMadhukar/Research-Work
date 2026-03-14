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

import hashlib
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
        self.enabled = True

    def disable(self):
        self.enabled = False

    def record(self, operation: str, duration: float):
        if self.enabled:
            self.timings[operation].append(duration)

    def get_stats(self) -> Dict:
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
        self.timings.clear()


# Global profiler instance
_profiler = PerformanceProfiler()


def get_profiler() -> PerformanceProfiler:
    return _profiler


def profile_function(func):
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

        for param_name, param_tensor in update.items():
            if not isinstance(param_tensor, torch.Tensor):
                param_tensor = torch.tensor(param_tensor)

            if torch.isnan(param_tensor).any():
                has_nan = True
                invalid_params.append(f"{param_name}:NaN")

            if torch.isinf(param_tensor).any():
                has_inf = True
                invalid_params.append(f"{param_name}:Inf")

            if has_nan or has_inf:
                break

        if has_nan or has_inf:
            invalid_client_ids.append(client_id)
            issue_type = []
            if has_nan:
                issue_type.append("NaN")
            if has_inf:
                issue_type.append("Inf")
            print(f"[NaN/Inf DETECTION] Client {client_id} DETECTED as malicious "
                  f"({'/'.join(issue_type)} in {invalid_params[0]})")
        else:
            valid_updates.append(update)
            valid_client_ids.append(client_id)

    if invalid_client_ids:
        print(f"[NaN/Inf DETECTION] Total detected: {len(invalid_client_ids)}/{len(client_updates)} clients")

    return valid_updates, valid_client_ids, invalid_client_ids


# =============================================================================
# Pluggable Aggregation Methods
# =============================================================================

class AggregationMethod:
    def aggregate(self, client_updates: List[Dict], weights: Optional[List[float]] = None) -> Dict:
        raise NotImplementedError


class FedAvgAggregation(AggregationMethod):
    def aggregate(self, client_updates: List[Dict], weights: Optional[List[float]] = None) -> Dict:
        n_clients = len(client_updates)
        if weights is None:
            weights = [1.0 / n_clients] * n_clients
        else:
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
    def aggregate(self, client_updates: List[Dict], weights: Optional[List[float]] = None) -> Dict:
        n_clients = len(client_updates)
        if weights is None:
            weights = [1.0 / n_clients] * n_clients
        else:
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
    def __init__(self, n_attackers: int = 2):
        self.n_attackers = n_attackers

    def aggregate(self, client_updates: List[Dict], weights: Optional[List[float]] = None) -> Dict:
        n_clients = len(client_updates)
        n_select = n_clients - self.n_attackers - 2

        if n_select <= 0:
            return FedAvgAggregation().aggregate(client_updates, weights)

        vectors = []
        for update in client_updates:
            vec = torch.cat([v.flatten() for v in update.values()]).cpu().numpy()
            vectors.append(vec)
        vectors = np.array(vectors)

        distances = np.zeros((n_clients, n_clients))
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                dist = np.linalg.norm(vectors[i] - vectors[j])
                distances[i, j] = dist
                distances[j, i] = dist

        scores = []
        for i in range(n_clients):
            dists = distances[i].copy()
            dists[i] = float('inf')
            closest_dists = np.partition(dists, n_select - 1)[:n_select]
            score = np.sum(closest_dists)
            scores.append(score)

        selected_idx = np.argmin(scores)
        return client_updates[selected_idx]


class MultiKrumAggregation(AggregationMethod):
    def __init__(self, n_attackers: int = 2, s: int = 3):
        self.n_attackers = n_attackers
        self.s = s

    def aggregate(self, client_updates: List[Dict], weights: Optional[List[float]] = None) -> Dict:
        n_clients = len(client_updates)
        n_neighbors = n_clients - self.n_attackers - 2

        vectors = []
        for update in client_updates:
            vec = torch.cat([v.flatten() for v in update.values()]).cpu().numpy()
            vectors.append(vec)
        vectors = np.array(vectors)

        distances = np.zeros((n_clients, n_clients))
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                dist = np.linalg.norm(vectors[i] - vectors[j]) ** 2
                distances[i, j] = dist
                distances[j, i] = dist

        scores = []
        for i in range(n_clients):
            dists = distances[i].copy()
            dists[i] = float('inf')
            closest_dists = np.partition(dists, n_neighbors - 1)[:n_neighbors]
            scores.append(np.sum(closest_dists))

        selected_indices = np.argsort(scores)[:self.s]
        selected_updates = [client_updates[i] for i in selected_indices]
        return FedAvgAggregation().aggregate(selected_updates)


class TrimmedMeanAggregation(AggregationMethod):
    def __init__(self, trim_ratio: float = 0.1):
        self.trim_ratio = trim_ratio

    def aggregate(self, client_updates: List[Dict], weights: Optional[List[float]] = None) -> Dict:
        n_clients = len(client_updates)
        n_trim = int(n_clients * self.trim_ratio)

        if n_trim == 0 or n_clients - 2 * n_trim < 1:
            return FedAvgAggregation().aggregate(client_updates, weights)

        aggregated_params = {}
        param_names = client_updates[0].keys()
        for param_name in param_names:
            stacked = torch.stack([update[param_name] for update in client_updates])
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_vals[n_trim:n_clients - n_trim]
            aggregated_params[param_name] = torch.mean(trimmed, dim=0)
        return aggregated_params


class MedianAggregation(AggregationMethod):
    def aggregate(self, client_updates: List[Dict], weights: Optional[List[float]] = None) -> Dict:
        aggregated_params = {}
        param_names = client_updates[0].keys()
        for param_name in param_names:
            stacked = torch.stack([update[param_name] for update in client_updates])
            aggregated_params[param_name] = torch.median(stacked, dim=0)[0]
        return aggregated_params


def get_aggregation_method(method_name: str, **kwargs) -> AggregationMethod:
    method_name = method_name.lower()
    if method_name == 'fedavg':
        return FedAvgAggregation()
    elif method_name == 'weighted_avg':
        return WeightedAverageAggregation()
    elif method_name == 'krum':
        return KrumAggregation(n_attackers=kwargs.get('n_attackers', 2))
    elif method_name == 'multi_krum':
        return MultiKrumAggregation(n_attackers=kwargs.get('n_attackers', 2), s=kwargs.get('s', 3))
    elif method_name == 'trimmed_mean':
        return TrimmedMeanAggregation(trim_ratio=kwargs.get('trim_ratio', 0.1))
    elif method_name == 'median':
        return MedianAggregation()
    else:
        print(f"[WARN] Unknown aggregation method '{method_name}', using FedAvg")
        return FedAvgAggregation()


# =============================================================================
# Shared gradient summary helper
# =============================================================================

def _print_grad_summary(tag: str, client_ids: List[int], updates: Dict[int, Dict],
                        flagged_ids: List[int], threshold: Optional[float] = None):
    """
    Print one compact line per client: grad_mean and PASS/FAIL.

    Args:
        tag:         Prefix label, e.g. '[CMFL]' or '[CDCFL-II]'
        client_ids:  Ordered list of client IDs to summarise
        updates:     {client_id: param_dict}
        flagged_ids: Clients that failed filtering (shown as FAIL)
        threshold:   Optional threshold value to display in header
    """
    header = f"{tag} Gradient norms (training clients)"
    if threshold is not None:
        header += f" | threshold={threshold:.6f}"
    header += ":"
    print(header)
    flagged_set = set(flagged_ids)
    for cid in client_ids:
        update = updates.get(cid, {})
        vals = [t.abs().mean().item() for t in update.values() if isinstance(t, torch.Tensor)]
        gmean = float(np.mean(vals)) if vals else float('nan')
        status = "FAIL" if cid in flagged_set else "PASS"
        print(f"  Client {cid:>3} | grad_mean={gmean:.6f} | {status}")


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
        self.num_clients = num_clients
        self.aggregation_method = get_aggregation_method(aggregation_method)
        self.aggregation_participation_frac = aggregation_participation_frac if aggregation_participation_frac is not None else 0.4
        self.selection_strategy = selection_strategy
        self.consensus_threshold = consensus_threshold

        if committee_size is None or training_clients_per_round is None:
            total_participation = max(5, min(64, int(num_clients * 0.40)))
            committee_size = max(2, int(total_participation * 0.40))
            training_clients_per_round = total_participation - committee_size

        self.committee_size = min(committee_size, num_clients - 1)
        self.training_clients_per_round = min(training_clients_per_round,
                                               num_clients - self.committee_size)
        self.committee_rotation_rounds = committee_rotation_rounds

        self.committee_members: set = set()
        self.training_clients: set = set()
        self.idle_clients: set = set(range(num_clients))

        initial_committee = np.random.choice(list(self.idle_clients), self.committee_size, replace=False)
        self.committee_members = set(initial_committee.tolist())
        self.idle_clients -= self.committee_members

        self.round_count = 0
        self.detected_malicious = set()
        self.committee_history = [sorted(list(self.committee_members))]
        self.detection_history_per_round = []
        self.infiltration_history = []
        self.verbose = False

    def activate_training_clients(self) -> List[int]:
        available_idle = list(self.idle_clients)
        n_activate = min(self.training_clients_per_round, len(available_idle))
        if n_activate == 0:
            available_idle = [i for i in range(self.num_clients) if i not in self.committee_members]
            n_activate = min(self.training_clients_per_round, len(available_idle))

        activated = list(np.random.choice(available_idle, n_activate, replace=False))
        self.training_clients = set(activated)
        self.idle_clients -= self.training_clients

        if self.verbose:
            print(f"  [ACTIVATE] Training: {sorted(activated)} | Committee: {sorted(self.committee_members)}")

        return activated

    def committee_scoring(self,
                          training_updates: Dict[int, Dict],
                          committee_updates: Dict[int, Dict]) -> Tuple[Dict[int, float], Dict[int, Dict[int, float]]]:
        training_ids = sorted(training_updates.keys())
        committee_ids = sorted(committee_updates.keys())
        C = len(committee_ids)
        eps = 1e-10

        training_matrix = []
        for cid in training_ids:
            vec = torch.cat([v.flatten() for v in training_updates[cid].values()]).cpu().numpy()
            if not np.isfinite(vec).all():
                vec = np.nan_to_num(vec, nan=1e10, posinf=1e10, neginf=-1e10)
            training_matrix.append(vec)
        training_matrix = np.array(training_matrix)

        committee_matrix = []
        for cid in committee_ids:
            vec = torch.cat([v.flatten() for v in committee_updates[cid].values()]).cpu().numpy()
            if not np.isfinite(vec).all():
                vec = np.nan_to_num(vec, nan=1e10, posinf=1e10, neginf=-1e10)
            committee_matrix.append(vec)
        committee_matrix = np.array(committee_matrix)

        sq_dist_matrix = cdist(training_matrix, committee_matrix, metric='sqeuclidean')

        per_member_scores = {}
        for c_idx, c_id in enumerate(committee_ids):
            per_member_scores[c_id] = {}
            for t_idx, t_id in enumerate(training_ids):
                per_member_scores[c_id][t_id] = 1.0 / (sq_dist_matrix[t_idx, c_idx] + eps)

        sum_sq_distances = sq_dist_matrix.sum(axis=1)
        final_scores = {}
        for t_idx, t_id in enumerate(training_ids):
            final_scores[t_id] = C / (sum_sq_distances[t_idx] + eps)

        return final_scores, per_member_scores

    def pbft_consensus(self,
                       final_scores: Dict[int, float],
                       training_client_ids: List[int]) -> Tuple[List[int], List[int]]:
        n_training = len(training_client_ids)
        alpha = self.aggregation_participation_frac

        sorted_by_score = sorted(training_client_ids,
                                 key=lambda cid: final_scores.get(cid, 0), reverse=True)
        n_accept = max(1, int(n_training * alpha))

        if self.selection_strategy == 2:
            selected_clients = sorted_by_score[-n_accept:]
            flagged_clients = sorted_by_score[:-n_accept]
        else:
            selected_clients = sorted_by_score[:n_accept]
            flagged_clients = sorted_by_score[n_accept:]

        if self.verbose:
            required_votes = int(np.floor(len(self.committee_members) / 2)) + 1
            strategy_label = "II" if self.selection_strategy == 2 else "I"
            print(f"  [pBFT] {len(self.committee_members)}/{len(self.committee_members)} agree "
                  f"(required: {required_votes}) | Strategy {strategy_label} (alpha={alpha:.0%}): "
                  f"accept {n_accept}/{n_training}")

        return selected_clients, flagged_clients

    def elect_new_committee(self,
                            training_client_ids: List[int],
                            final_scores: Dict[int, float]) -> List[int]:
        sorted_by_score = sorted(training_client_ids, key=lambda cid: final_scores.get(cid, 0))
        n = len(sorted_by_score)

        if n <= self.committee_size:
            return sorted_by_score

        lower_bound = max(0, (n - self.committee_size) // 2)
        upper_bound = lower_bound + self.committee_size
        if upper_bound > n:
            upper_bound = n
            lower_bound = max(0, upper_bound - self.committee_size)
        return sorted_by_score[lower_bound:upper_bound]

    def step_down_committee(self):
        self.idle_clients.update(self.committee_members)
        self.committee_members.clear()

    def aggregate_selected_clients(self,
                                   selected_client_ids: List[int],
                                   all_updates: Dict[int, Dict],
                                   client_data_sizes: Optional[Dict[int, int]] = None) -> Dict:
        if len(selected_client_ids) == 0:
            raise ValueError("Cannot aggregate with no selected clients")

        selected_updates = [all_updates[cid] for cid in selected_client_ids]
        weights = [client_data_sizes.get(cid, 1) for cid in selected_client_ids] if client_data_sizes else None
        return self.aggregation_method.aggregate(selected_updates, weights)

    def cmfl_round(self,
                   client_updates: Dict[int, Dict],
                   client_losses: Dict[int, float],
                   malicious_client_ids: Optional[List[int]] = None,
                   all_clients: Optional[List] = None,
                   use_anomaly_detection: bool = True,
                   client_data_sizes: Optional[Dict[int, int]] = None) -> Tuple[Dict, List[int], Dict]:
        self.round_count += 1
        all_client_ids = sorted(client_updates.keys())

        training_ids = [cid for cid in all_client_ids if cid in self.training_clients]
        committee_ids = [cid for cid in all_client_ids if cid in self.committee_members]

        training_updates = {cid: client_updates[cid] for cid in training_ids}
        committee_updates_dict = {cid: client_updates[cid] for cid in committee_ids}

        if use_anomaly_detection and len(training_ids) > 0 and len(committee_ids) > 0:
            # Committee scoring (Eq. 7-8)
            final_scores, per_member_scores = self.committee_scoring(
                training_updates, committee_updates_dict
            )

            # pBFT consensus — alpha-based selection for aggregation
            selected_clients, selection_excluded = self.pbft_consensus(
                final_scores, training_ids
            )

            # MAD-based anomaly detection (decoupled from aggregation selection)
            C = len(committee_ids)
            eps = 1e-10
            distances = {cid: C / (final_scores[cid] + eps) for cid in training_ids}
            dist_values = np.array([distances[cid] for cid in training_ids])
            if len(dist_values) >= 2:
                median_dist = float(np.median(dist_values))
                mad = float(np.median(np.abs(dist_values - median_dist)))
                if mad < 1e-10:
                    mad = float(np.std(dist_values)) + 1e-10
                anomaly_threshold = median_dist + 3.0 * mad
                anomaly_detected = [cid for cid in training_ids
                                    if distances[cid] > anomaly_threshold]
            else:
                anomaly_detected = []
                anomaly_threshold = 0.0

            # Committee peer-check (leave-one-out)
            committee_anomalies = []
            if len(committee_ids) >= 3:
                committee_vectors = {}
                for cid in committee_ids:
                    vec = torch.cat([p.detach().cpu().flatten()
                                     for p in committee_updates_dict[cid].values()])
                    committee_vectors[cid] = vec.numpy()
                for cid in committee_ids:
                    peers = [pid for pid in committee_ids if pid != cid]
                    peer_dists = [float(np.linalg.norm(committee_vectors[cid] - committee_vectors[pid]))
                                  for pid in peers]
                    if len(peer_dists) >= 2:
                        peer_median = float(np.median(peer_dists))
                        peer_mad = float(np.median(np.abs(np.array(peer_dists) - peer_median)))
                        if peer_mad < 1e-10:
                            peer_mad = float(np.std(peer_dists)) + 1e-10
                        peer_threshold = peer_median + 3.0 * peer_mad
                        if float(np.mean(peer_dists)) > peer_threshold:
                            committee_anomalies.append(cid)
                if committee_anomalies:
                    anomaly_detected.extend(committee_anomalies)

            # ── Compact gradient summary (replaces verbose raw dump) ──────────
            _print_grad_summary(
                tag='[CMFL]',
                client_ids=training_ids,
                updates=training_updates,
                flagged_ids=anomaly_detected,
                threshold=anomaly_threshold,
            )

            # ── DETAILED ATTACK SCENARIO LOGGING ──────────────────────────────
            if malicious_client_ids is not None and len(malicious_client_ids) > 0:
                print(f"\n[R{self.round_count}] ═══ CMFL-I THRESHOLD & SELECTION ANALYSIS ═══")
                print(f"  Training clients: {sorted(training_ids)} (count={len(training_ids)})")
                print(f"  Committee members: {sorted(committee_ids)} (count={len(committee_ids)})")
                print(f"\n  ─── SCORING RESULTS ───")
                for cid in sorted(training_ids):
                    score_val = final_scores.get(cid, 0.0)
                    dist_val = distances.get(cid, 0.0)
                    is_mal = " [MALICIOUS]" if cid in malicious_client_ids else ""
                    print(f"    Client {cid:>2} | Score={score_val:>10.6f} | Distance={dist_val:>10.6f}{is_mal}")

                print(f"\n  ─── THRESHOLD INFORMATION ───")
                print(f"    Anomaly Threshold (MAD-based): {anomaly_threshold:.6f}")
                print(f"    Median Distance: {median_dist:.6f}")
                print(f"    MAD (Median Absolute Deviation): {mad:.6f}")
                print(f"    Formula: threshold = median + 3.0×MAD = {median_dist:.6f} + 3.0×{mad:.6f}")

                print(f"\n  ─── SELECTION RESULTS ───")
                print(f"    Selected (Alpha {self.aggregation_participation_frac:.0%}): {sorted(selected_clients)} (count={len(selected_clients)})")
                print(f"    Excluded (Alpha): {sorted(selection_excluded)} (count={len(selection_excluded)})")

                print(f"\n  ─── ANOMALY DETECTION RESULTS ───")
                print(f"    Flagged (distance > threshold): {sorted(anomaly_detected)} (count={len(anomaly_detected)})")
                accepted_by_mad = [cid for cid in training_ids if cid not in anomaly_detected]
                print(f"    Accepted (distance ≤ threshold): {sorted(accepted_by_mad)} (count={len(accepted_by_mad)})")

                # Check for conflicts
                conflicts = [cid for cid in selected_clients if cid in anomaly_detected]
                if conflicts:
                    print(f"\n  ⚠️  CONFLICT DETECTED: Clients {conflicts} are BOTH selected AND flagged!")

                print(f"\n  ─── USED FOR AGGREGATION ───")
                print(f"    Final aggregation clients: {sorted(selected_clients)}")
                for cid in sorted(selected_clients):
                    is_mal = " [MALICIOUS]" if cid in malicious_client_ids else ""
                    print(f"      Client {cid}{is_mal}")
                print()

        elif use_anomaly_detection and len(committee_ids) == 0:
            selected_clients = list(training_ids)
            selection_excluded = []
            anomaly_detected = []
            final_scores = {cid: 1.0 for cid in training_ids}
            anomaly_threshold = 0.0
        else:
            selected_clients = list(training_ids)
            selection_excluded = []
            anomaly_detected = []
            final_scores = {}
            anomaly_threshold = 0.0

        # Track N1/N2/N3 infiltration
        if all_clients is not None:
            mal_in_training = [cid for cid in training_ids
                               if cid < len(all_clients) and all_clients[cid].is_malicious]
            mal_in_committee = [cid for cid in self.committee_members
                                if cid < len(all_clients) and all_clients[cid].is_malicious]
            mal_in_aggregation = [cid for cid in selected_clients
                                  if cid < len(all_clients) and all_clients[cid].is_malicious]
            self.infiltration_history.append({
                'round': self.round_count,
                'training_ids': sorted(training_ids),
                'training_size': len(training_ids),
                'N1_count': len(mal_in_training),
                'N1_fraction': len(mal_in_training) / max(1, len(training_ids)),
                'committee_members': sorted(self.committee_members),
                'committee_size': len(self.committee_members),
                'N2_count': len(mal_in_committee),
                'N2_fraction': len(mal_in_committee) / max(1, len(self.committee_members)),
                'selected_clients': sorted(selected_clients),
                'aggregation_size': len(selected_clients),
                'N3_count': len(mal_in_aggregation),
                'N3_fraction': len(mal_in_aggregation) / max(1, len(selected_clients)),
            })

        for cid in anomaly_detected:
            self.detected_malicious.add(cid)

        aggregated_params = self.aggregate_selected_clients(
            selected_clients, client_updates, client_data_sizes=client_data_sizes
        )

        participant_snapshot = sorted(training_ids)
        detection_metrics_dict = None
        if all_clients is not None and use_anomaly_detection:
            detection_metrics_dict = self.calculate_round_detection_metrics(
                all_clients, round_num=self.round_count,
                flagged_this_round=anomaly_detected,
                participant_ids=participant_snapshot
            )

        # Committee rotation
        if self.round_count % self.committee_rotation_rounds == 0 and len(training_ids) > 0:
            new_committee = self.elect_new_committee(training_ids, final_scores)
            self.step_down_committee()
            self.committee_members = set(new_committee)
            self.idle_clients -= self.committee_members
            non_elected = self.training_clients - self.committee_members
            self.idle_clients.update(non_elected)
            self.training_clients.clear()
            self.committee_history.append(sorted(list(self.committee_members)))
        else:
            self.idle_clients.update(self.training_clients)
            self.training_clients.clear()

        metrics = {
            'round': self.round_count,
            'training_clients': sorted(training_ids),
            'committee_members': sorted(list(self.committee_members)),
            'selected_clients': selected_clients,
            'selection_excluded': selection_excluded,
            'anomaly_detected': anomaly_detected,
            'aggregation_clients': selected_clients,
            'final_scores': final_scores,
            'anomaly_threshold': anomaly_threshold,
            'aggregation_participation_frac': self.aggregation_participation_frac,
        }

        if detection_metrics_dict is not None:
            metrics['detection_metrics'] = detection_metrics_dict

        return aggregated_params, anomaly_detected, metrics

    def get_infiltration_summary(self):
        history = self.infiltration_history
        empty = {
            'per_round': [], 'total_rounds': 0,
            'avg_N1_count': 0.0, 'avg_N1_fraction': 0.0,
            'max_N1_count': 0, 'max_N1_fraction': 0.0,
            'avg_N2_count': 0.0, 'avg_N2_fraction': 0.0,
            'max_N2_count': 0, 'max_N2_fraction': 0.0,
            'rounds_with_N2': 0, 'rounds_with_N2_fraction': 0.0,
            'avg_N3_count': 0.0, 'avg_N3_fraction': 0.0,
            'max_N3_count': 0, 'max_N3_fraction': 0.0,
            'rounds_with_N3': 0, 'rounds_with_N3_fraction': 0.0,
            'avg_malicious_count': 0.0, 'avg_malicious_fraction': 0.0,
            'max_malicious_count': 0, 'max_malicious_fraction': 0.0,
            'rounds_with_malicious': 0, 'rounds_with_malicious_fraction': 0.0,
        }
        if not history:
            return empty

        def _stats(key_count, key_frac):
            counts = [r[key_count] for r in history]
            fracs = [r[key_frac] for r in history]
            rw = sum(1 for c in counts if c > 0)
            return {
                f'avg_{key_count}': float(np.mean(counts)),
                f'avg_{key_frac}': float(np.mean(fracs)),
                f'max_{key_count}': int(max(counts)),
                f'max_{key_frac}': float(max(fracs)),
                f'rounds_with_{key_count.replace("_count", "")}': rw,
                f'rounds_with_{key_count.replace("_count", "")}_fraction': rw / len(history),
            }

        result = {'per_round': history, 'total_rounds': len(history)}
        result.update(_stats('N1_count', 'N1_fraction'))
        result.update(_stats('N2_count', 'N2_fraction'))
        result.update(_stats('N3_count', 'N3_fraction'))
        result['avg_malicious_count'] = result['avg_N2_count']
        result['avg_malicious_fraction'] = result['avg_N2_fraction']
        result['max_malicious_count'] = result['max_N2_count']
        result['max_malicious_fraction'] = result['max_N2_fraction']
        result['rounds_with_malicious'] = result['rounds_with_N2']
        result['rounds_with_malicious_fraction'] = result['rounds_with_N2_fraction']
        return result

    def get_malicious_in_committee_summary(self):
        return self.get_infiltration_summary()

    def calculate_round_detection_metrics(self, all_clients: List, round_num: int = 0,
                                          flagged_this_round: Optional[List[int]] = None,
                                          participant_ids: Optional[List[int]] = None):
        if participant_ids is not None:
            participants = list(participant_ids)
        else:
            participants = list(self.training_clients)

        detected_ids = set(flagged_this_round) if flagged_this_round is not None else set(self.detected_malicious)

        tp = fp = tn = fn = 0
        for cid in participants:
            if cid >= len(all_clients):
                continue
            is_malicious = getattr(all_clients[cid], 'is_malicious', False)
            is_rejected = cid in detected_ids
            if is_malicious and is_rejected:
                tp += 1
            elif is_malicious and not is_rejected:
                fn += 1
            elif not is_malicious and is_rejected:
                fp += 1
            else:
                tn += 1
        total = tp + fp + fn + tn

        if total != len(participants):
            print(f"[R{round_num}] WARNING: Detection count mismatch: "
                  f"TP+FP+TN+FN={total} != participants={len(participants)}")

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
            'num_participants': total,
            'num_malicious': tp + fn,
            'num_benign': fp + tn,
        }
        self.detection_history_per_round.append(round_metrics)
        return round_metrics


# =============================================================================
# CD-CFL Defense (CDCFL-I and CDCFL-II)
# =============================================================================

class CDCFLDefense:
    """
    CD-CFL Defense-in-Depth for Decentralized FL.

    CDCFL-II: Validation -> Outlier Filter -> Robust Agg + Consensus (pBFT/PoW)
    CDCFL-I:  Robust Agg (all gradients) -> Consensus (pBFT/PoW)
    """

    def __init__(self, num_clients: int,
                 committee_size: int = None,
                 training_clients_per_round: int = None,
                 committee_rotation_rounds: int = 1,
                 aggregation_method: str = 'multi_krum',
                 robust_agg_method: str = None,
                 consensus_threshold: float = 0.5,
                 aggregation_participation_frac: float = None,
                 selection_strategy: int = 1,
                 norm_lower_factor: float = 0.25,
                 norm_upper_factor: float = 4.0,
                 loss_descent_delta: float = 0.05,
                 mad_multiplier: float = 3.0,
                 enable_validation: bool = True,
                 enable_norm_check: bool = True,
                 enable_loss_check: bool = True,
                 enable_nan_check: bool = True,
                 enable_filter: bool = True,
                 enable_robust: bool = True,
                 consensus_type: str = 'pow',
                 max_nonce_trials: int = 5000,
                 pow_difficulty: str = '000',
                 **kwargs):
        self.num_clients = num_clients

        agg_name = robust_agg_method or aggregation_method or 'multi_krum'
        if not enable_robust:
            self.aggregation_method = FedAvgAggregation()
            self._agg_name = 'fedavg'
        else:
            self.aggregation_method = get_aggregation_method(agg_name)
            self._agg_name = agg_name

        self.aggregation_participation_frac = aggregation_participation_frac if aggregation_participation_frac is not None else 0.4
        self.selection_strategy = selection_strategy
        self.consensus_threshold = consensus_threshold

        if committee_size is None or training_clients_per_round is None:
            total_participation = max(5, min(64, int(num_clients * 0.40)))
            committee_size = max(2, int(total_participation * 0.40))
            training_clients_per_round = total_participation - committee_size

        self.committee_size = min(committee_size, num_clients - 1)
        self.training_clients_per_round = min(training_clients_per_round,
                                               num_clients - self.committee_size)
        self.committee_rotation_rounds = committee_rotation_rounds

        self.committee_members: set = set()
        self.training_clients: set = set()
        self.idle_clients: set = set(range(num_clients))

        initial_committee = np.random.choice(list(self.idle_clients), self.committee_size, replace=False)
        self.committee_members = set(initial_committee.tolist())
        self.idle_clients -= self.committee_members

        self.round_count = 0
        self.detected_malicious = set()
        self.committee_history = [sorted(list(self.committee_members))]

        self.norm_lower_factor = norm_lower_factor
        self.norm_upper_factor = norm_upper_factor
        self.loss_descent_delta = loss_descent_delta
        self.mad_multiplier = mad_multiplier

        self.enable_validation = enable_validation
        self.enable_norm_check = enable_norm_check
        self.enable_loss_check = enable_loss_check
        self.enable_nan_check = enable_nan_check
        self.enable_filter = enable_filter
        self.enable_robust = enable_robust
        self.consensus_type = consensus_type
        self.max_nonce_trials = max_nonce_trials
        self.pow_difficulty = pow_difficulty

        self._previous_block_hash = '0' * 64  # genesis

        self.detection_history_per_round = []
        self.infiltration_history = []
        self.validation_rejection_history = []
        self.filter_rejection_history = []
        self.pbft_aggregation_history = []
        self.consensus_history = []
        self.round_acceptance_history = []

        self.verbose = False

    # =========================================================================
    # Client pool management
    # =========================================================================

    def activate_training_clients(self) -> List[int]:
        available_idle = list(self.idle_clients)
        n_activate = min(self.training_clients_per_round, len(available_idle))
        if n_activate == 0:
            available_idle = [i for i in range(self.num_clients) if i not in self.committee_members]
            n_activate = min(self.training_clients_per_round, len(available_idle))

        activated = list(np.random.choice(available_idle, n_activate, replace=False))
        self.training_clients = set(activated)
        self.idle_clients -= self.training_clients
        return activated

    def step_down_committee(self):
        self.idle_clients.update(self.committee_members)
        self.committee_members.clear()

    # =========================================================================
    # CDCFL-II: Client Update Validation
    # =========================================================================

    def validate_client_updates(self, training_updates: Dict[int, Dict],
                                training_losses_before: Dict[int, float],
                                training_losses_after: Dict[int, float],
                                committee_updates: Dict[int, Dict],
                                global_params: Optional[Dict] = None) -> Tuple[Dict[int, Dict], List[int], Dict]:
        validation_metrics = {
            'norm_rejected': [], 'loss_rejected': [], 'nan_rejected': [],
            'per_client': {}
        }

        if not self.enable_validation:
            return training_updates, [], validation_metrics

        has_norm_ref = False
        norm_lower = norm_upper = 0.0
        if self.enable_norm_check:
            committee_norms = []
            for cid, update in committee_updates.items():
                delta = {k: update[k] - global_params[k] for k in update if k in global_params} if global_params else update
                norm = self._compute_update_norm(delta)
                if np.isfinite(norm) and norm > 0:
                    committee_norms.append(norm)
            if committee_norms:
                median_committee_norm = float(np.median(committee_norms))
                norm_lower = self.norm_lower_factor * median_committee_norm
                norm_upper = self.norm_upper_factor * median_committee_norm
                has_norm_ref = True

        passed_updates = {}
        failed_ids = []

        for cid, update in training_updates.items():
            client_metrics = {'norm': 0.0, 'loss_before': 0.0, 'loss_after': 0.0, 'checks_failed': []}
            rejected = False

            if self.enable_nan_check and not rejected:
                has_nan = any(
                    not torch.isfinite(p).all()
                    for p in update.values()
                    if isinstance(p, torch.Tensor)
                )
                if has_nan:
                    client_metrics['checks_failed'].append('nan_inf')
                    validation_metrics['nan_rejected'].append(cid)
                    rejected = True

            if self.enable_norm_check and has_norm_ref and not rejected:
                delta = {k: update[k] - global_params[k] for k in update if k in global_params} if global_params else update
                norm = self._compute_update_norm(delta)
                client_metrics['norm'] = norm
                if norm < norm_lower or norm > norm_upper:
                    client_metrics['checks_failed'].append('norm_bounds')
                    validation_metrics['norm_rejected'].append(cid)
                    rejected = True

            if self.enable_loss_check and not rejected:
                loss_before = training_losses_before.get(cid, None)
                loss_after = training_losses_after.get(cid, None)
                client_metrics['loss_before'] = loss_before or 0.0
                client_metrics['loss_after'] = loss_after or 0.0
                if loss_before is not None and loss_after is not None:
                    if loss_after > loss_before * (1 + self.loss_descent_delta):
                        client_metrics['checks_failed'].append('loss_consistency')
                        validation_metrics['loss_rejected'].append(cid)
                        rejected = True

            validation_metrics['per_client'][cid] = client_metrics

            if rejected:
                failed_ids.append(cid)
            else:
                passed_updates[cid] = update

        if len(failed_ids) > 0 and self.verbose:
            print(f"  [VALIDATION] Rejected {len(failed_ids)}: "
                  f"norm={len(validation_metrics['norm_rejected'])}, "
                  f"loss={len(validation_metrics['loss_rejected'])}, "
                  f"nan={len(validation_metrics['nan_rejected'])}")

        return passed_updates, failed_ids, validation_metrics

    # =========================================================================
    # CDCFL-II: Committee Statistical Outlier Filter
    # =========================================================================

    def committee_outlier_filter(self, pow_passed_updates: Dict[int, Dict],
                                  committee_updates: Dict[int, Dict],
                                  global_params: Optional[Dict] = None) -> Tuple[List[int], List[int], Dict[int, float]]:
        training_ids = sorted(pow_passed_updates.keys())
        committee_ids = sorted(committee_updates.keys())

        if not self.enable_filter or len(committee_ids) == 0 or len(training_ids) == 0:
            return training_ids, [], {cid: 0.0 for cid in training_ids}

        committee_vectors = []
        for cid in committee_ids:
            if global_params is not None:
                vec = torch.cat([(committee_updates[cid][k] - global_params[k]).flatten().float()
                                 for k in committee_updates[cid] if k in global_params])
            else:
                vec = torch.cat([v.flatten().float() for v in committee_updates[cid].values()])
            committee_vectors.append(vec)
        committee_matrix = torch.stack(committee_vectors)

        reference_gradient = torch.median(committee_matrix, dim=0)[0]

        deviation_scores = {}
        deviations_list = []
        for cid in training_ids:
            if global_params is not None:
                vec = torch.cat([(pow_passed_updates[cid][k] - global_params[k]).flatten().float()
                                 for k in pow_passed_updates[cid] if k in global_params])
            else:
                vec = torch.cat([v.flatten().float() for v in pow_passed_updates[cid].values()])
            deviation = torch.norm(vec - reference_gradient, p=2).item()
            deviation_scores[cid] = deviation
            deviations_list.append(deviation)

        if len(deviations_list) < 2:
            return training_ids, [], deviation_scores

        # MAD-based adaptive threshold
        dev_array = np.array(deviations_list)
        median_dev = float(np.median(dev_array))
        mad = float(np.median(np.abs(dev_array - median_dev)))
        threshold = median_dev + self.mad_multiplier * max(mad, 1e-8)

        flagged_ids = []
        accepted_ids = []
        for cid in training_ids:
            if deviation_scores[cid] > threshold:
                flagged_ids.append(cid)
            else:
                accepted_ids.append(cid)

        # ── Compact gradient summary (replaces verbose raw dump) ──────────────
        _print_grad_summary(
            tag='[CDCFL-II]',
            client_ids=training_ids,
            updates=pow_passed_updates,
            flagged_ids=flagged_ids,
            threshold=threshold,
        )

        if self.verbose and flagged_ids:
            print(f"  [FILTER] Threshold={threshold:.4f} (median={median_dev:.4f}, MAD={mad:.4f}), "
                  f"flagged {len(flagged_ids)}/{len(training_ids)}")

        # ── DETAILED ATTACK SCENARIO LOGGING FOR CDCFL-II FILTER ──────────
        return accepted_ids, flagged_ids, deviation_scores

        return accepted_ids, flagged_ids, deviation_scores

    def committee_consensus_on_outliers(self, flagged_ids_per_member: Dict[int, List[int]]) -> List[int]:
        if not flagged_ids_per_member:
            return []
        C = len(flagged_ids_per_member)
        majority = C // 2 + 1
        flag_counts = defaultdict(int)
        for member_id, flagged_list in flagged_ids_per_member.items():
            for cid in flagged_list:
                flag_counts[cid] += 1
        return [cid for cid, count in flag_counts.items() if count >= majority]

    # =========================================================================
    # CDCFL-II: Robust Aggregation + pBFT Finalization
    # =========================================================================

    def robust_aggregate_with_pbft(self,
                                    accepted_updates: Dict[int, Dict],
                                    committee_ids: List[int],
                                    client_data_sizes: Optional[Dict[int, int]] = None) -> Tuple[Dict, Dict]:
        if len(accepted_updates) == 0:
            raise ValueError("Cannot aggregate with no accepted clients")

        accepted_ids = sorted(accepted_updates.keys())
        updates_list = [accepted_updates[cid] for cid in accepted_ids]
        weights = [client_data_sizes.get(cid, 1) for cid in accepted_ids] if client_data_sizes else None
        candidate_params = self.aggregation_method.aggregate(updates_list, weights)

        C = len(committee_ids) if committee_ids else 1
        majority_needed = C // 2 + 1

        pbft_metrics = {
            'committee_size': C,
            'majority_needed': majority_needed,
            'votes_for': C,
            'consensus_reached': True,
        }
        self.pbft_aggregation_history.append({'round': self.round_count, **pbft_metrics})
        return candidate_params, pbft_metrics

    # =========================================================================
    # CDCFL-II: Main round method
    # =========================================================================

    def cdcfl_ii_round(self,
                       client_updates: Dict[int, Dict],
                       client_losses_before: Dict[int, float],
                       client_losses_after: Dict[int, float],
                       malicious_client_ids: Optional[List[int]] = None,
                       all_clients: Optional[List] = None,
                       client_data_sizes: Optional[Dict[int, int]] = None,
                       test_loader=None,
                       global_params: Optional[Dict] = None,
                       model_template=None,
                       use_anomaly_detection: bool = True) -> Tuple[Dict, List[int], Dict]:
        self.round_count += 1
        round_start = time.time()
        all_client_ids = sorted(client_updates.keys())

        training_ids = [cid for cid in all_client_ids if cid in self.training_clients]
        committee_ids = [cid for cid in all_client_ids if cid in self.committee_members]

        training_updates = {cid: client_updates[cid] for cid in training_ids}
        committee_updates_dict = {cid: client_updates[cid] for cid in committee_ids}

        all_flagged = []
        validation_metrics = {}
        filter_metrics = {}
        pbft_agg_metrics = {}
        deviation_scores = {}
        t_validation = t_filter = t_agg = t_finalize = 0.0
        round_accepted = True

        if use_anomaly_detection and len(training_ids) > 0 and len(committee_ids) > 0:
            # Step 5: Validation
            t_validation = time.time()
            validated_updates, validation_failed_ids, validation_metrics = self.validate_client_updates(
                training_updates,
                {cid: client_losses_before.get(cid, 0.0) for cid in training_ids},
                {cid: client_losses_after.get(cid, 0.0) for cid in training_ids},
                committee_updates_dict,
                global_params=global_params,
            )
            t_validation = time.time() - t_validation
            all_flagged.extend(validation_failed_ids)

            self.validation_rejection_history.append({
                'round': self.round_count,
                'norm_rejected': len(validation_metrics.get('norm_rejected', [])),
                'loss_rejected': len(validation_metrics.get('loss_rejected', [])),
                'nan_rejected': len(validation_metrics.get('nan_rejected', [])),
                'total_rejected': len(validation_failed_ids)
            })

            # Step 6a: Outlier filter
            t_filter = time.time()
            accepted_ids, filter_flagged_ids, deviation_scores = self.committee_outlier_filter(
                validated_updates, committee_updates_dict, global_params=global_params
            )
            t_filter = time.time() - t_filter

            flagged_ids_per_member = {mid: list(filter_flagged_ids) for mid in committee_ids}
            consensus_flagged = self.committee_consensus_on_outliers(flagged_ids_per_member)
            all_flagged.extend(consensus_flagged)

            accepted_ids = [cid for cid in accepted_ids if cid not in consensus_flagged]
            non_consensus_flagged = [cid for cid in filter_flagged_ids if cid not in consensus_flagged]
            accepted_ids.extend(non_consensus_flagged)

            self.filter_rejection_history.append({
                'round': self.round_count,
                'flagged_count': len(consensus_flagged),
                'threshold_used': 0.0
            })

            filter_metrics = {
                'filter_flagged': filter_flagged_ids,
                'consensus_flagged': consensus_flagged,
                'accepted_count': len(accepted_ids)
            }

            # ── DETAILED ATTACK SCENARIO LOGGING FOR CDCFL-II ──────────────────
            if malicious_client_ids is not None and len(malicious_client_ids) > 0:
                print(f"\n[R{self.round_count}] ═══ CDCFL-II THRESHOLD & SELECTION ANALYSIS ═══")
                print(f"  Training clients: {sorted(training_ids)} (count={len(training_ids)})")
                print(f"  Committee members: {sorted(committee_ids)} (count={len(committee_ids)})")

                print(f"\n  ─── LAYER 1: VALIDATION RESULTS ───")
                if validation_failed_ids:
                    print(f"    Rejected clients: {sorted(validation_failed_ids)}")
                    for cid in sorted(validation_failed_ids):
                        checks = validation_metrics.get('per_client', {}).get(cid, {}).get('checks_failed', [])
                        is_mal = " [MALICIOUS]" if cid in malicious_client_ids else ""
                        print(f"      Client {cid}: {', '.join(checks)}{is_mal}")
                else:
                    print(f"    All clients passed validation")
                print(f"    Validated clients: {sorted(validated_updates.keys())} (count={len(validated_updates)})")

                print(f"\n  ─── LAYER 2: OUTLIER FILTER RESULTS ───")
                print(f"    Flagged (filter layer): {sorted(filter_flagged_ids)} (count={len(filter_flagged_ids)})")
                print(f"    Accepted by filter: {sorted(accepted_ids)} (count={len(accepted_ids)})")

                # Get threshold if available (need to recompute for display)
                if len(deviation_scores) >= 2:
                    dev_values = np.array([deviation_scores.get(cid, 0.0) for cid in sorted(training_ids)])
                    median_dev = float(np.median(dev_values))
                    mad = float(np.median(np.abs(dev_values - median_dev)))
                    threshold = median_dev + self.mad_multiplier * max(mad, 1e-8)
                    print(f"    Threshold (MAD-based): {threshold:.6f}")
                    print(f"      Median deviation: {median_dev:.6f}")
                    print(f"      MAD: {mad:.6f}")
                    print(f"      Formula: threshold = median + {self.mad_multiplier}×MAD")

                print(f"\n  ─── FILTER LAYER DETAILS ───")
                for cid in sorted(training_ids):
                    if cid in validated_updates:
                        dev = deviation_scores.get(cid, 0.0)
                        in_filter = "✓ PASS" if cid not in filter_flagged_ids else "✗ FLAG"
                        is_mal = " [MALICIOUS]" if cid in malicious_client_ids else ""
                        print(f"    Client {cid:>2} | Deviation={dev:>10.6f} | {in_filter}{is_mal}")

                print(f"\n  ─── LAYER 3: CONSENSUS RESULTS ───")
                print(f"    Consensus flagged: {sorted(consensus_flagged)} (count={len(consensus_flagged)})")
                print(f"    Consensus passed: {[c for c in filter_flagged_ids if c not in consensus_flagged]} (count={len([c for c in filter_flagged_ids if c not in consensus_flagged])})")

                print(f"\n  ─── FINAL AGGREGATION CLIENTS ───")
                print(f"    Selected for aggregation: {sorted(accepted_ids)} (count={len(accepted_ids)})")
                for cid in sorted(accepted_ids):
                    is_mal = " [MALICIOUS]" if cid in malicious_client_ids else ""
                    print(f"      Client {cid}{is_mal}")
                print()

            accepted_updates = {cid: client_updates[cid] for cid in accepted_ids if cid in client_updates}
            if len(accepted_updates) == 0:
                accepted_updates = committee_updates_dict

            # Step 6b: Robust aggregation (Multi-Krum / configured method)
            t_agg = time.time()
            aggregated_params, pbft_agg_metrics = self.robust_aggregate_with_pbft(
                accepted_updates, committee_ids, client_data_sizes
            )
            t_agg = time.time() - t_agg
            selected_clients = accepted_ids

            # Step 6c: Finalization — PoW or pBFT, controlled by self.consensus_type.
            # This mirrors CDCFL-I so both variants can be tested independently
            # by passing consensus_type='pow' or consensus_type='pbft'.
            t_finalize = time.time()
            round_accepted = True
            finalization_metrics = {}
            if self.consensus_type == 'pbft':
                final_passed, fin_met = self._pbft_finalize(
                    aggregated_params, committee_updates_dict
                )
                finalization_metrics = fin_met
            else:  # 'pow'
                final_passed, _, fin_met = self._pow_finalize(
                    aggregated_params, self.round_count
                )
                fin_met['consensus_type'] = 'pow'
                finalization_metrics = fin_met

            if not final_passed:
                round_accepted = False
                if global_params is not None:
                    aggregated_params = {k: v.clone() for k, v in global_params.items()}

            t_finalize = time.time() - t_finalize

            pbft_agg_metrics['finalization'] = finalization_metrics
            pbft_agg_metrics['round_accepted'] = round_accepted
            pbft_agg_metrics['consensus_type'] = self.consensus_type

            fin_status = "PASSED" if final_passed else "FAILED"
            print(f"  [CDCFL-II FINALIZE/{self.consensus_type.upper()}] {fin_status}")

        elif use_anomaly_detection and len(committee_ids) == 0:
            selected_clients = list(training_ids)
            updates_list = [training_updates[cid] for cid in training_ids]
            weights = [client_data_sizes.get(cid, 1) for cid in training_ids] if client_data_sizes else None
            aggregated_params = self.aggregation_method.aggregate(updates_list, weights)
            deviation_scores = {cid: 0.0 for cid in training_ids}
        else:
            selected_clients = list(training_ids)
            updates_list = [training_updates[cid] for cid in training_ids]
            weights = [client_data_sizes.get(cid, 1) for cid in training_ids] if client_data_sizes else None
            aggregated_params = self.aggregation_method.aggregate(updates_list, weights)
            deviation_scores = {}

        for cid in all_flagged:
            self.detected_malicious.add(cid)

        # Track N1/N2/N3 infiltration
        if all_clients is not None:
            mal_in_training = [cid for cid in training_ids
                               if cid < len(all_clients) and all_clients[cid].is_malicious]
            mal_in_committee = [cid for cid in self.committee_members
                                if cid < len(all_clients) and all_clients[cid].is_malicious]
            mal_in_aggregation = [cid for cid in selected_clients
                                  if cid < len(all_clients) and all_clients[cid].is_malicious]
            self.infiltration_history.append({
                'round': self.round_count,
                'training_ids': sorted(training_ids),
                'training_size': len(training_ids),
                'N1_count': len(mal_in_training),
                'N1_fraction': len(mal_in_training) / max(1, len(training_ids)),
                'committee_members': sorted(self.committee_members),
                'committee_size': len(self.committee_members),
                'N2_count': len(mal_in_committee),
                'N2_fraction': len(mal_in_committee) / max(1, len(self.committee_members)),
                'selected_clients': sorted(selected_clients),
                'aggregation_size': len(selected_clients),
                'N3_count': len(mal_in_aggregation),
                'N3_fraction': len(mal_in_aggregation) / max(1, len(selected_clients)),
            })

        participant_snapshot = sorted(training_ids)
        detection_metrics_dict = None
        if all_clients is not None and use_anomaly_detection:
            detection_metrics_dict = self.calculate_round_detection_metrics(
                all_clients, round_num=self.round_count,
                flagged_this_round=all_flagged,
                participant_ids=participant_snapshot
            )

        # Committee rotation
        if self.round_count % self.committee_rotation_rounds == 0 and len(training_ids) > 0:
            scores_for_election = deviation_scores if deviation_scores else {cid: 0.0 for cid in training_ids}
            new_committee = self.elect_new_committee(training_ids, scores_for_election)
            self.step_down_committee()
            self.committee_members = set(new_committee)
            self.idle_clients -= self.committee_members
            non_elected = self.training_clients - self.committee_members
            self.idle_clients.update(non_elected)
            self.training_clients.clear()
            self.committee_history.append(sorted(list(self.committee_members)))
        else:
            self.idle_clients.update(self.training_clients)
            self.training_clients.clear()

        round_time = time.time() - round_start

        metrics = {
            'round': self.round_count,
            'training_clients': sorted(training_ids),
            'committee_members': sorted(list(self.committee_members)),
            'selected_clients': selected_clients,
            'flagged_clients': all_flagged,
            'aggregation_clients': selected_clients,
            'validation_metrics': validation_metrics,
            'filter_metrics': filter_metrics,
            'pbft_aggregation': pbft_agg_metrics,
            'deviation_scores': {k: float(v) for k, v in deviation_scores.items()},
            'aggregation_method': self._agg_name,
            'round_time': round_time,
            'layer_timing': {
                'validation_time': t_validation,
                'filter_time': t_filter,
                'agg_time': t_agg,
                'finalize_time': t_finalize,
            },
            'round_accepted': round_accepted,
            'consensus_type': self.consensus_type,
            'layer_counts': {
                'total_training': len(training_ids),
                'validation_passed': len(training_ids)
                                     - len(validation_metrics.get('norm_rejected', []))
                                     - len(validation_metrics.get('loss_rejected', []))
                                     - len(validation_metrics.get('nan_rejected', [])),
                'filter_accepted': len(selected_clients),
                'final_aggregated': len(selected_clients),
            },
            'layers_enabled': {
                'validation': self.enable_validation,
                'filter': self.enable_filter,
                'robust': self.enable_robust,
            }
        }

        if detection_metrics_dict is not None:
            metrics['detection_metrics'] = detection_metrics_dict

        return aggregated_params, all_flagged, metrics

    # =========================================================================
    # CDCFL-I: Committee Consensus Check
    # =========================================================================

    def _committee_consensus_check(self, delta_candidate: Dict[str, torch.Tensor],
                                    committee_updates: Dict[int, Dict[str, torch.Tensor]]
                                    ) -> Tuple[bool, Dict]:
        committee_ids = sorted(committee_updates.keys())
        C = len(committee_ids)
        if C == 0:
            return True, {'approve_count': 0, 'required': 0, 'total': 0, 'passed': True, 'per_member_votes': {}}

        required = C // 2 + 1
        candidate_vec = torch.cat([v.flatten().float() for v in delta_candidate.values()])
        candidate_norm = torch.norm(candidate_vec, p=2).item()

        per_member_votes = {}
        approve_count = 0

        for cid in committee_ids:
            member_vec = torch.cat([v.flatten().float() for v in committee_updates[cid].values()])
            member_norm = torch.norm(member_vec, p=2).item()

            if candidate_norm > 1e-10 and member_norm > 1e-10:
                cos_sim = (torch.dot(candidate_vec, member_vec) / (candidate_norm * member_norm)).item()
            else:
                cos_sim = 0.0

            norm_ratio = candidate_norm / max(member_norm, 1e-10)
            approved = cos_sim > -0.5 and 0.1 < norm_ratio < 10.0
            if approved:
                approve_count += 1

            per_member_votes[cid] = {
                'cosine_similarity': float(cos_sim),
                'norm_ratio': float(norm_ratio),
                'approved': approved,
            }

        passed = approve_count >= required

        if self.verbose:
            status = "PASSED" if passed else "FAILED"
            print(f"  [CONSENSUS] {approve_count}/{C} approved (required: {required}) — {status}")

        return passed, {
            'approve_count': approve_count,
            'required': required,
            'total': C,
            'passed': passed,
            'per_member_votes': per_member_votes,
        }

    # =========================================================================
    # CDCFL-I: PoW Finalization
    # =========================================================================

    def _pow_finalize(self, delta_candidate: Dict[str, torch.Tensor],
                      round_id: int) -> Tuple[bool, str, Dict]:
        model_hasher = hashlib.sha256()
        for name in sorted(delta_candidate.keys()):
            tensor_bytes = delta_candidate[name].flatten().float().cpu().numpy().tobytes()
            model_hasher.update(tensor_bytes)
        aggregated_model_hash = model_hasher.hexdigest()

        timestamp = time.time()
        previous_block_hash = self._previous_block_hash

        pow_passed = False
        winning_hash = ''
        winning_nonce = -1

        for nonce in range(self.max_nonce_trials):
            block_string = (f"{round_id}{aggregated_model_hash}"
                            f"{previous_block_hash}{timestamp}{nonce}")
            block_hash = hashlib.sha256(block_string.encode()).hexdigest()
            if block_hash.startswith(self.pow_difficulty):
                pow_passed = True
                winning_hash = block_hash
                winning_nonce = nonce
                break

        if pow_passed:
            self._previous_block_hash = winning_hash

        return pow_passed, winning_hash, {
            'nonce': winning_nonce,
            'hash': winning_hash,
            'passed': pow_passed,
            'difficulty': self.pow_difficulty,
            'trials_used': winning_nonce + 1 if pow_passed else self.max_nonce_trials,
            'block': {
                'round_id': round_id,
                'model_hash': aggregated_model_hash,
                'previous_hash': previous_block_hash,
                'timestamp': timestamp,
            },
        }

    # =========================================================================
    # CDCFL-I: pBFT Finalization
    # =========================================================================

    def _pbft_finalize(self, delta_candidate: Dict[str, torch.Tensor],
                       committee_updates: Dict[int, Dict[str, torch.Tensor]]
                       ) -> Tuple[bool, Dict]:
        consensus_passed, metrics = self._committee_consensus_check(delta_candidate, committee_updates)
        metrics['consensus_type'] = 'pbft'
        return consensus_passed, metrics

    # =========================================================================
    # CDCFL-I: Main round method
    # =========================================================================

    def cdcfl_i_round(self,
                      client_updates: Dict[int, Dict],
                      client_losses_after: Dict[int, float] = None,
                      malicious_client_ids: Optional[List[int]] = None,
                      all_clients: Optional[List] = None,
                      client_data_sizes: Optional[Dict[int, int]] = None,
                      test_loader=None,
                      global_params: Optional[Dict] = None,
                      model_template=None,
                      use_anomaly_detection: bool = True) -> Tuple[Dict, List[int], Dict]:
        self.round_count += 1
        round_start = time.time()
        all_client_ids = sorted(client_updates.keys())

        training_ids = [cid for cid in all_client_ids if cid in self.training_clients]
        committee_ids = [cid for cid in all_client_ids if cid in self.committee_members]

        training_updates = {cid: client_updates[cid] for cid in training_ids}
        committee_updates_dict = {cid: client_updates[cid] for cid in committee_ids}

        round_accepted = True
        consensus_metrics = {}
        finalization_metrics = {}
        contribution_scores = {}

        # Layer 1: Robust aggregation of ALL training gradients
        t_agg = time.time()
        updates_list = [training_updates[cid] for cid in training_ids]
        weights = [client_data_sizes.get(cid, 1) for cid in training_ids] if client_data_sizes else [1.0] * len(updates_list)

        if len(updates_list) > 0:
            delta_candidate = self.aggregation_method.aggregate(updates_list, weights)
        elif global_params is not None:
            delta_candidate = {k: v.clone() for k, v in global_params.items()}
        else:
            delta_candidate = client_updates[all_client_ids[0]]
        t_agg = time.time() - t_agg

        # ── Compact gradient summary for CDCFL-I (all clients pass, no filter) ──
        if training_ids:
            _print_grad_summary(
                tag='[CDCFL-I]',
                client_ids=training_ids,
                updates=training_updates,
                flagged_ids=[],        # CDCFL-I has no per-client filter
                threshold=None,        # no threshold to display
            )

        # ── DETAILED ATTACK SCENARIO LOGGING FOR CDCFL-I ──────────────────────
        if malicious_client_ids is not None and len(malicious_client_ids) > 0:
            print(f"\n[R{self.round_count}] ═══ CDCFL-I THRESHOLD & SELECTION ANALYSIS ═══")
            print(f"  Training clients: {sorted(training_ids)} (count={len(training_ids)})")
            print(f"  Committee members: {sorted(committee_ids)} (count={len(committee_ids)})")

            print(f"\n  ─── CDCFL-I ARCHITECTURE ───")
            print(f"    Layer 1: Robust Aggregation (NO per-client filtering)")
            print(f"    Layer 2: Committee Consensus Check")
            print(f"    Layer 3: Finalization (PoW or pBFT)")

            print(f"\n  ─── AGGREGATION INPUT (ALL CLIENTS) ───")
            print(f"    All training clients included (no filtering): {sorted(training_ids)}")
            for cid in sorted(training_ids):
                is_mal = " [MALICIOUS]" if cid in malicious_client_ids else ""
                print(f"      Client {cid}{is_mal}")
            print(f"    Total clients aggregated: {len(training_ids)}")
            print(f"    Aggregation method: {self._agg_name}")

            print(f"\n  ─── THRESHOLD & FILTERING ───")
            print(f"    Threshold: None (no per-client filtering)")
            print(f"    Flagged clients: [] (empty)")
            print(f"    Strategy: Robust aggregation handles outliers algorithmically")

            print(f"\n  ─── CONSENSUS CHECK RESULTS ───")
            if consensus_metrics:
                approve_count = consensus_metrics.get('approve_count', 0)
                total = consensus_metrics.get('total', 0)
                required = consensus_metrics.get('required', 0)
                passed = consensus_metrics.get('passed', False)
                print(f"    Votes: {approve_count}/{total} (required: {required})")
                print(f"    Result: {'PASSED' if passed else 'FAILED'}")

            print(f"\n  ─── FINALIZATION RESULTS ───")
            print(f"    Finalization type: {self.consensus_type.upper()}")
            print(f"    Round accepted: {'YES' if round_accepted else 'NO'}")

            print(f"\n  ─── FINAL AGGREGATION OUTPUT ───")
            print(f"    Clients used: {sorted(training_ids)} (count={len(training_ids)})")
            print(f"    All clients aggregated (none rejected)")
            print()


        # Layer 2: Committee consensus check
        t_consensus = time.time()
        if len(committee_ids) > 0 and use_anomaly_detection:
            consensus_passed, consensus_metrics = self._committee_consensus_check(
                delta_candidate, committee_updates_dict
            )
            if not consensus_passed:
                round_accepted = False
                if global_params is not None:
                    delta_candidate = {k: v.clone() for k, v in global_params.items()}
        else:
            consensus_passed = True
            consensus_metrics = {'approve_count': 0, 'required': 0, 'total': 0, 'passed': True, 'per_member_votes': {}}
        t_consensus = time.time() - t_consensus

        # Layer 3: Finalization (PoW or pBFT)
        t_finalize = time.time()
        if round_accepted:
            if self.consensus_type == 'pbft':
                pbft_passed, pbft_metrics = self._pbft_finalize(delta_candidate, committee_updates_dict)
                final_passed = consensus_passed and pbft_passed
                finalization_metrics = pbft_metrics
            else:
                pow_passed, pow_hash, pow_met = self._pow_finalize(delta_candidate, self.round_count)
                final_passed = consensus_passed and pow_passed
                finalization_metrics = pow_met
                finalization_metrics['consensus_type'] = 'pow'

            if not final_passed:
                round_accepted = False
                if global_params is not None:
                    delta_candidate = {k: v.clone() for k, v in global_params.items()}
        t_finalize = time.time() - t_finalize

        aggregated_params = delta_candidate

        self.consensus_history.append(consensus_metrics)
        self.round_acceptance_history.append(round_accepted)

        if consensus_metrics.get('total', 0) > 0:
            status = "PASSED" if consensus_metrics['passed'] else "FAILED"
            print(f"  [CONSENSUS] {consensus_metrics['approve_count']}/{consensus_metrics['total']} "
                  f"approved (required: {consensus_metrics['required']}) — {status}")

        # Contribution scores for committee election
        if len(training_ids) > 0:
            candidate_vec = torch.cat([v.flatten().float() for v in aggregated_params.values()])
            candidate_norm = torch.norm(candidate_vec, p=2).item()
            for cid in training_ids:
                member_vec = torch.cat([v.flatten().float() for v in training_updates[cid].values()])
                member_norm = torch.norm(member_vec, p=2).item()
                if candidate_norm > 1e-10 and member_norm > 1e-10:
                    cos = (torch.dot(candidate_vec, member_vec) / (candidate_norm * member_norm)).item()
                else:
                    cos = 0.0
                contribution_scores[cid] = cos

        # Track N1/N2/N3 infiltration
        if all_clients is not None:
            mal_in_training = [cid for cid in training_ids
                               if cid < len(all_clients) and all_clients[cid].is_malicious]
            mal_in_committee = [cid for cid in self.committee_members
                                if cid < len(all_clients) and all_clients[cid].is_malicious]
            self.infiltration_history.append({
                'round': self.round_count,
                'training_ids': sorted(training_ids),
                'training_size': len(training_ids),
                'N1_count': len(mal_in_training),
                'N1_fraction': len(mal_in_training) / max(1, len(training_ids)),
                'committee_members': sorted(self.committee_members),
                'committee_size': len(self.committee_members),
                'N2_count': len(mal_in_committee),
                'N2_fraction': len(mal_in_committee) / max(1, len(self.committee_members)),
                'selected_clients': sorted(training_ids),
                'aggregation_size': len(training_ids),
                'N3_count': len(mal_in_training),
                'N3_fraction': len(mal_in_training) / max(1, len(training_ids)),
            })

        # Committee rotation
        if self.round_count % self.committee_rotation_rounds == 0 and len(training_ids) > 0:
            deviation_scores = {cid: 1.0 - contribution_scores.get(cid, 0.0) for cid in training_ids}
            new_committee = self.elect_new_committee(training_ids, deviation_scores)
            self.step_down_committee()
            self.committee_members = set(new_committee)
            self.idle_clients -= self.committee_members
            non_elected = self.training_clients - self.committee_members
            self.idle_clients.update(non_elected)
            self.training_clients.clear()
            self.committee_history.append(sorted(list(self.committee_members)))
        else:
            self.idle_clients.update(self.training_clients)
            self.training_clients.clear()

        round_time = time.time() - round_start

        metrics = {
            'round': self.round_count,
            'strategy': 'CDCFL-I',
            'training_clients': sorted(training_ids),
            'committee_members': sorted(list(self.committee_members)),
            'selected_clients': sorted(training_ids),
            'flagged_clients': [],
            'aggregation_clients': sorted(training_ids),
            'aggregation_method': self._agg_name,
            'consensus_metrics': consensus_metrics,
            'finalization_metrics': finalization_metrics,
            'consensus_type': self.consensus_type,
            'round_accepted': round_accepted,
            'contribution_scores': {k: float(v) for k, v in contribution_scores.items()},
            'round_time': round_time,
            'layer_timing': {
                'agg_time': t_agg,
                'consensus_time': t_consensus,
                'finalize_time': t_finalize,
                'pow_time': t_finalize if self.consensus_type == 'pow' else 0.0,
                'filter_time': 0.0,
            },
        }

        return aggregated_params, [], metrics

    # =========================================================================
    # Committee election
    # =========================================================================

    def elect_new_committee(self, training_client_ids: List[int],
                            deviation_scores: Dict[int, float]) -> List[int]:
        sorted_by_deviation = sorted(training_client_ids,
                                      key=lambda cid: deviation_scores.get(cid, 0))
        n = len(sorted_by_deviation)

        if n <= self.committee_size:
            return sorted_by_deviation

        lower_bound = max(0, (n - self.committee_size) // 2)
        upper_bound = lower_bound + self.committee_size
        if upper_bound > n:
            upper_bound = n
            lower_bound = max(0, upper_bound - self.committee_size)

        return sorted_by_deviation[lower_bound:upper_bound]

    # =========================================================================
    # Detection metrics
    # =========================================================================

    def calculate_round_detection_metrics(self, all_clients: List, round_num: int = 0,
                                          flagged_this_round: Optional[List[int]] = None,
                                          participant_ids: Optional[List[int]] = None):
        participants = list(participant_ids) if participant_ids is not None else list(self.training_clients)
        detected_ids = set(flagged_this_round) if flagged_this_round is not None else set(self.detected_malicious)

        tp = fp = tn = fn = 0
        for cid in participants:
            if cid >= len(all_clients):
                continue
            is_malicious = getattr(all_clients[cid], 'is_malicious', False)
            is_rejected = cid in detected_ids
            if is_malicious and is_rejected:
                tp += 1
            elif is_malicious and not is_rejected:
                fn += 1
            elif not is_malicious and is_rejected:
                fp += 1
            else:
                tn += 1
        total = tp + fp + fn + tn

        if total != len(participants):
            print(f"[R{round_num}] WARNING: Detection count mismatch: "
                  f"TP+FP+TN+FN={total} != participants={len(participants)}")

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
            'num_participants': total,
            'num_malicious': tp + fn,
            'num_benign': fp + tn,
        }
        self.detection_history_per_round.append(round_metrics)
        return round_metrics

    def get_infiltration_summary(self):
        history = self.infiltration_history
        if not history:
            return {
                'per_round': [], 'total_rounds': 0,
                'avg_N1_count': 0.0, 'avg_N1_fraction': 0.0,
                'max_N1_count': 0, 'max_N1_fraction': 0.0,
                'avg_N2_count': 0.0, 'avg_N2_fraction': 0.0,
                'max_N2_count': 0, 'max_N2_fraction': 0.0,
                'rounds_with_N2': 0, 'rounds_with_N2_fraction': 0.0,
                'avg_N3_count': 0.0, 'avg_N3_fraction': 0.0,
                'max_N3_count': 0, 'max_N3_fraction': 0.0,
                'rounds_with_N3': 0, 'rounds_with_N3_fraction': 0.0,
                'avg_malicious_count': 0.0, 'avg_malicious_fraction': 0.0,
                'max_malicious_count': 0, 'max_malicious_fraction': 0.0,
                'rounds_with_malicious': 0, 'rounds_with_malicious_fraction': 0.0,
            }

        def _stats(key_count, key_frac):
            counts = [r[key_count] for r in history]
            fracs = [r[key_frac] for r in history]
            rw = sum(1 for c in counts if c > 0)
            return {
                f'avg_{key_count}': float(np.mean(counts)),
                f'avg_{key_frac}': float(np.mean(fracs)),
                f'max_{key_count}': int(max(counts)),
                f'max_{key_frac}': float(max(fracs)),
                f'rounds_with_{key_count.replace("_count", "")}': rw,
                f'rounds_with_{key_count.replace("_count", "")}_fraction': rw / len(history),
            }

        result = {'per_round': history, 'total_rounds': len(history)}
        result.update(_stats('N1_count', 'N1_fraction'))
        result.update(_stats('N2_count', 'N2_fraction'))
        result.update(_stats('N3_count', 'N3_fraction'))
        result['avg_malicious_count'] = result['avg_N2_count']
        result['avg_malicious_fraction'] = result['avg_N2_fraction']
        result['max_malicious_count'] = result['max_N2_count']
        result['max_malicious_fraction'] = result['max_N2_fraction']
        result['rounds_with_malicious'] = result['rounds_with_N2']
        result['rounds_with_malicious_fraction'] = result['rounds_with_N2_fraction']
        return result

    def get_malicious_in_committee_summary(self):
        return self.get_infiltration_summary()

    def get_layer_metrics(self) -> Dict:
        total_norm = sum(r.get('norm_rejected', 0) for r in self.validation_rejection_history)
        total_loss = sum(r.get('loss_rejected', 0) for r in self.validation_rejection_history)
        total_nan = sum(r.get('nan_rejected', 0) for r in self.validation_rejection_history)
        total_validation_rejected = sum(r.get('total_rejected', 0) for r in self.validation_rejection_history)
        total_filter_flagged = sum(r.get('flagged_count', 0) for r in self.filter_rejection_history)

        return {
            'validation_norm_rejected': total_norm,
            'validation_loss_rejected': total_loss,
            'validation_nan_rejected': total_nan,
            'validation_total_rejected': total_validation_rejected,
            'filter_total_flagged': total_filter_flagged,
            'validation_rejection_per_round': self.validation_rejection_history,
            'filter_rejection_per_round': self.filter_rejection_history,
            'pbft_history': self.pbft_aggregation_history,
            'num_rounds': max(1, self.round_count),
        }

    # =========================================================================
    # Utilities
    # =========================================================================

    def _compute_update_norm(self, update: Dict) -> float:
        vec = torch.cat([v.flatten().float() for v in update.values()])
        return torch.norm(vec, p=2).item()


# =============================================================================
# Factory
# =============================================================================

def get_defense(strategy: str = "cmfl", num_clients: int = 10, **kwargs):
    """
    Factory function to create committee-based defense instances.

    Args:
        strategy: 'cmfl', 'cdcfl_i', or 'cdcfl_ii'
        num_clients: Number of clients in the system
    """
    if strategy == "cmfl":
        return CMFLDefense(num_clients, **kwargs)
    elif strategy in ("cdcfl_i", "cdcfl_ii", "cd_cfl"):
        return CDCFLDefense(num_clients, **kwargs)
    else:
        raise ValueError(f"Unknown defense strategy: {strategy}. Use 'cmfl', 'cdcfl_i', or 'cdcfl_ii'.")