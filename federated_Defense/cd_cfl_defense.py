"""
CD-CFL: Committee-Driven CFL Byzantine Defense for Decentralized Federated Learning

Implements a defense-in-depth pipeline that replaces CMFL's single-layer defense:
  Layer 1 - Proof of Work (PoW) Verification: gradient norm bounds, loss descent, NaN/Inf
  Layer 2 - Committee Statistical Outlier Filter: median-based reference + MAD threshold
  Layer 3 - Multi-Krum Robust Aggregation + pBFT Finalization:
            Each committee member independently runs Multi-Krum on accepted updates,
            then pBFT majority vote finalizes the global model (same flow as CMFL).

Keeps the same committee/training/idle client pool management as CMFLDefense.
"""

import hashlib
import numpy as np
import torch
import time
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from defense import (
    CMFLDefense, MultiKrumAggregation, FedAvgAggregation,
    get_aggregation_method
)


class CDCFLDefense:
    """
    CD-CFL Defense-in-Depth for Decentralized FL.

    Same committee pool management as CMFLDefense, but replaces L2 scoring
    with a CD-CFL pipeline: PoW -> Outlier Filter -> Multi-Krum + pBFT.
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
                 # CD-CFL specific params
                 norm_lower_factor: float = 0.25,
                 norm_upper_factor: float = 4.0,
                 loss_descent_delta: float = 0.05,
                 mad_multiplier: float = 3.0,
                 # Layer ablation flags
                 enable_pow: bool = True,
                 enable_filter: bool = True,
                 enable_robust: bool = True,
                 # CDCFL-I finalization method
                 finalization_method: str = 'pow',
                 **kwargs):
        self.num_clients = num_clients

        # Robust aggregation method — Multi-Krum by default
        agg_name = robust_agg_method or aggregation_method or 'multi_krum'
        if not enable_robust:
            # When robust agg is disabled, fall back to FedAvg
            self.aggregation_method = FedAvgAggregation()
            self._agg_name = 'fedavg'
        else:
            self.aggregation_method = get_aggregation_method(agg_name)
            self._agg_name = agg_name

        self.aggregation_participation_frac = aggregation_participation_frac if aggregation_participation_frac is not None else 0.4
        self.selection_strategy = selection_strategy
        self.consensus_threshold = consensus_threshold

        # Auto-scale committee and training sizes
        if committee_size is None or training_clients_per_round is None:
            total_participation = max(5, min(64, int(num_clients * 0.40)))
            committee_size = max(2, int(total_participation * 0.40))
            training_clients_per_round = total_participation - committee_size

        self.committee_size = min(committee_size, num_clients - 1)
        self.training_clients_per_round = min(training_clients_per_round,
                                               num_clients - self.committee_size)
        self.committee_rotation_rounds = committee_rotation_rounds

        # Client pools (same as CMFLDefense)
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

        # CD-CFL parameters
        self.norm_lower_factor = norm_lower_factor
        self.norm_upper_factor = norm_upper_factor
        self.loss_descent_delta = loss_descent_delta
        self.mad_multiplier = mad_multiplier

        # Layer enable flags (for ablation)
        self.enable_pow = enable_pow
        self.enable_filter = enable_filter
        self.enable_robust = enable_robust
        self.finalization_method = finalization_method  # 'pow' or 'pbft'

        # Per-round tracking
        self.detection_history_per_round = []
        self.infiltration_history = []
        self.pow_rejection_history = []
        self.filter_rejection_history = []
        self.pbft_aggregation_history = []

        # CDCFL-I specific tracking
        self.consensus_history = []
        self.round_acceptance_history = []

        self.verbose = False

    # =========================================================================
    # Client pool management (identical to CMFLDefense)
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
        old_committee = list(self.committee_members)
        self.idle_clients.update(old_committee)
        self.committee_members.clear()

    # =========================================================================
    # Layer 1: Proof of Work Verification
    # =========================================================================

    def pow_verify(self, training_updates: Dict[int, Dict],
                   training_losses_before: Dict[int, float],
                   training_losses_after: Dict[int, float],
                   committee_updates: Dict[int, Dict],
                   global_params: Optional[Dict] = None) -> Tuple[Dict[int, Dict], List[int], Dict]:
        """
        Layer 1: PoW checks on UPDATE DELTAS (not raw params).

        Checks:
          1c. NaN/Inf in submitted params
          1a. ||delta|| outside [lower, upper] relative to committee delta norms
          1b. Post-poisoning loss increased vs pre-training loss

        When global_params is provided, delta = client_params - global_params.
        This correctly measures the actual work done (or faked) by each client.

        Returns:
            (passed_updates, failed_ids, pow_metrics)
        """
        pow_metrics = {
            'norm_rejected': [], 'loss_rejected': [], 'nan_rejected': [],
            'per_client': {}
        }

        if not self.enable_pow:
            return training_updates, [], pow_metrics

        # Compute UPDATE DELTA norms for committee (reference)
        committee_norms = []
        for cid, update in committee_updates.items():
            if global_params is not None:
                delta = {k: update[k] - global_params[k] for k in update if k in global_params}
            else:
                delta = update
            norm = self._compute_update_norm(delta)
            if np.isfinite(norm) and norm > 0:
                committee_norms.append(norm)

        if not committee_norms:
            return training_updates, [], pow_metrics

        median_committee_norm = float(np.median(committee_norms))
        norm_lower = self.norm_lower_factor * median_committee_norm
        norm_upper = self.norm_upper_factor * median_committee_norm

        passed_updates = {}
        failed_ids = []

        for cid, update in training_updates.items():
            client_metrics = {'norm': 0.0, 'loss_before': 0.0, 'loss_after': 0.0, 'checks_failed': []}
            rejected = False

            # Check 1c: NaN/Inf detection
            has_nan = False
            for param_name, param_tensor in update.items():
                if isinstance(param_tensor, torch.Tensor):
                    if not torch.isfinite(param_tensor).all():
                        has_nan = True
                        break
            if has_nan:
                client_metrics['checks_failed'].append('nan_inf')
                pow_metrics['nan_rejected'].append(cid)
                rejected = True

            if not rejected:
                # Check 1a: UPDATE DELTA norm bounds
                if global_params is not None:
                    delta = {k: update[k] - global_params[k] for k in update if k in global_params}
                else:
                    delta = update
                norm = self._compute_update_norm(delta)
                client_metrics['norm'] = norm
                if norm < norm_lower or norm > norm_upper:
                    client_metrics['checks_failed'].append('norm_bounds')
                    pow_metrics['norm_rejected'].append(cid)
                    rejected = True

            if not rejected:
                # Check 1b: Loss descent (now uses post-poisoning loss from client)
                loss_before = training_losses_before.get(cid, None)
                loss_after = training_losses_after.get(cid, None)
                client_metrics['loss_before'] = loss_before if loss_before is not None else 0.0
                client_metrics['loss_after'] = loss_after if loss_after is not None else 0.0

                if loss_before is not None and loss_after is not None:
                    if loss_after > loss_before * (1 + self.loss_descent_delta):
                        client_metrics['checks_failed'].append('loss_descent')
                        pow_metrics['loss_rejected'].append(cid)
                        rejected = True

            pow_metrics['per_client'][cid] = client_metrics

            if rejected:
                failed_ids.append(cid)
            else:
                passed_updates[cid] = update

        total_rejected = len(failed_ids)
        if total_rejected > 0 and self.verbose:
            print(f"  [PoW] Rejected {total_rejected}: norm={len(pow_metrics['norm_rejected'])}, "
                  f"loss={len(pow_metrics['loss_rejected'])}, nan={len(pow_metrics['nan_rejected'])}")

        return passed_updates, failed_ids, pow_metrics

    # =========================================================================
    # Layer 2: Committee Statistical Outlier Filter
    # =========================================================================

    def committee_outlier_filter(self, pow_passed_updates: Dict[int, Dict],
                                  committee_updates: Dict[int, Dict],
                                  global_params: Optional[Dict] = None) -> Tuple[List[int], List[int], Dict[int, float]]:
        """
        Layer 2: Compute reference delta from committee, flag outliers via MAD.

        Uses update deltas (params - global_params) when global_params is provided,
        making deviation scores independent of model magnitude and purely
        reflecting the direction/magnitude of each client's contribution.

        Returns:
            (accepted_ids, flagged_ids, deviation_scores)
        """
        training_ids = sorted(pow_passed_updates.keys())
        committee_ids = sorted(committee_updates.keys())

        if not self.enable_filter or len(committee_ids) == 0 or len(training_ids) == 0:
            return training_ids, [], {cid: 0.0 for cid in training_ids}

        # Flatten committee UPDATE DELTAS to vectors
        committee_vectors = []
        for cid in committee_ids:
            if global_params is not None:
                vec = torch.cat([(committee_updates[cid][k] - global_params[k]).flatten().float()
                                 for k in committee_updates[cid] if k in global_params])
            else:
                vec = torch.cat([v.flatten().float() for v in committee_updates[cid].values()])
            committee_vectors.append(vec)
        committee_matrix = torch.stack(committee_vectors)  # (C, d)

        # Reference = coordinate-wise median of committee deltas
        reference_gradient = torch.median(committee_matrix, dim=0)[0]  # (d,)

        # Compute deviation of each training client's delta from reference
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

        # MAD-based adaptive threshold with floor to prevent over-sensitivity
        dev_array = np.array(deviations_list)
        median_dev = float(np.median(dev_array))
        mad = float(np.median(np.abs(dev_array - median_dev)))
        threshold = median_dev + self.mad_multiplier * max(mad, 1e-8)

        # Flag clients with deviation > threshold
        flagged_ids = []
        accepted_ids = []
        for cid in training_ids:
            if deviation_scores[cid] > threshold:
                flagged_ids.append(cid)
            else:
                accepted_ids.append(cid)

        if self.verbose and flagged_ids:
            print(f"  [FILTER] Threshold={threshold:.4f} (median={median_dev:.4f}, MAD={mad:.4f}), "
                  f"flagged {len(flagged_ids)}/{len(training_ids)}")

        return accepted_ids, flagged_ids, deviation_scores

    def committee_consensus_on_outliers(self, flagged_ids_per_member: Dict[int, List[int]]) -> List[int]:
        """
        pBFT consensus on outlier flags: a client is removed only if
        floor(C/2)+1 committee members flag it.
        """
        if not flagged_ids_per_member:
            return []

        C = len(flagged_ids_per_member)
        majority = C // 2 + 1

        flag_counts = defaultdict(int)
        for member_id, flagged_list in flagged_ids_per_member.items():
            for cid in flagged_list:
                flag_counts[cid] += 1

        final_flagged = [cid for cid, count in flag_counts.items() if count >= majority]
        return final_flagged

    # =========================================================================
    # Layer 3: Multi-Krum Robust Aggregation + pBFT Finalization
    # =========================================================================

    def robust_aggregate_with_pbft(self,
                                    accepted_updates: Dict[int, Dict],
                                    committee_ids: List[int],
                                    client_data_sizes: Optional[Dict[int, int]] = None) -> Tuple[Dict, Dict]:
        """
        Layer 3: Each committee member independently runs Multi-Krum on
        the accepted updates, then pBFT majority vote finalizes the
        global model.

        Same pattern as CMFL: aggregate -> pBFT consensus -> return.

        Since all committee members see the same accepted_updates and
        Multi-Krum is deterministic, every member produces the identical
        aggregated model.  pBFT consensus is therefore automatically
        achieved (all votes agree).

        Returns:
            (aggregated_params, pbft_metrics)
        """
        if len(accepted_updates) == 0:
            raise ValueError("Cannot aggregate with no accepted clients")

        accepted_ids = sorted(accepted_updates.keys())
        updates_list = [accepted_updates[cid] for cid in accepted_ids]

        if client_data_sizes:
            weights = [client_data_sizes.get(cid, 1) for cid in accepted_ids]
        else:
            weights = [1.0] * len(updates_list)

        # --- Each committee member independently aggregates ---
        # (deterministic: same input -> same output for every member)
        candidate_params = self.aggregation_method.aggregate(updates_list, weights)

        # --- pBFT consensus vote ---
        # Every member produced the same candidate (deterministic Multi-Krum).
        # Tally: C votes for this candidate, 0 against.
        C = len(committee_ids) if committee_ids else 1
        majority_needed = C // 2 + 1
        votes_for = C  # all members agree (deterministic)

        pbft_metrics = {
            'committee_size': C,
            'majority_needed': majority_needed,
            'votes_for': votes_for,
            'consensus_reached': votes_for >= majority_needed,
        }

        self.pbft_aggregation_history.append({
            'round': self.round_count,
            **pbft_metrics
        })

        if self.verbose:
            print(f"  [pBFT-AGG] Consensus: {votes_for}/{C} members agree "
                  f"(need {majority_needed})")

        return candidate_params, pbft_metrics

    # =========================================================================
    # Main round method
    # =========================================================================

    def cd_cfl_round(self,
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
        """
        Execute one full CD-CFL defense round.

        Flow (mirrors CMFL's cmfl_round):
        1. Separate training vs committee updates
        2. Layer 1 — PoW verify
        3. Layer 2 — Committee outlier filter + pBFT on flags
        4. Layer 3 — Multi-Krum aggregation + pBFT finalization
        5. Detection metrics
        6. Committee election + rotation
        7. Return (aggregated_params, flagged_clients, metrics)
        """
        self.round_count += 1
        round_start = time.time()
        all_client_ids = sorted(client_updates.keys())

        # Step 1: Separate training vs committee updates
        training_ids = [cid for cid in all_client_ids if cid in self.training_clients]
        committee_ids = [cid for cid in all_client_ids if cid in self.committee_members]

        training_updates = {cid: client_updates[cid] for cid in training_ids}
        committee_updates_dict = {cid: client_updates[cid] for cid in committee_ids}

        all_flagged = []
        pow_metrics = {}
        filter_metrics = {}
        pbft_agg_metrics = {}
        deviation_scores = {}

        if use_anomaly_detection and len(training_ids) > 0 and len(committee_ids) > 0:
            # ----- Layer 1: PoW Verification -----
            t_pow = time.time()
            pow_passed_updates, pow_failed_ids, pow_metrics = self.pow_verify(
                training_updates,
                {cid: client_losses_before.get(cid, 0.0) for cid in training_ids},
                {cid: client_losses_after.get(cid, 0.0) for cid in training_ids},
                committee_updates_dict,
                global_params=global_params,
            )
            t_pow = time.time() - t_pow
            all_flagged.extend(pow_failed_ids)

            self.pow_rejection_history.append({
                'round': self.round_count,
                'norm_rejected': len(pow_metrics.get('norm_rejected', [])),
                'loss_rejected': len(pow_metrics.get('loss_rejected', [])),
                'nan_rejected': len(pow_metrics.get('nan_rejected', [])),
                'total_rejected': len(pow_failed_ids)
            })

            # ----- Layer 2: Committee Outlier Filter -----
            t_filter = time.time()
            accepted_ids, filter_flagged_ids, deviation_scores = self.committee_outlier_filter(
                pow_passed_updates, committee_updates_dict, global_params=global_params
            )
            t_filter = time.time() - t_filter

            # pBFT consensus on outlier flags
            flagged_ids_per_member = {
                member_id: list(filter_flagged_ids) for member_id in committee_ids
            }
            consensus_flagged = self.committee_consensus_on_outliers(flagged_ids_per_member)
            all_flagged.extend(consensus_flagged)

            # Remove consensus-flagged from accepted; restore non-consensus-flagged
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

            # Build accepted updates dict
            accepted_updates = {cid: client_updates[cid] for cid in accepted_ids
                                if cid in client_updates}

            # Fall back to committee updates if every training client rejected
            if len(accepted_updates) == 0:
                accepted_updates = committee_updates_dict

            # ----- Layer 3: Multi-Krum + pBFT finalization -----
            t_agg = time.time()
            aggregated_params, pbft_agg_metrics = self.robust_aggregate_with_pbft(
                accepted_updates, committee_ids, client_data_sizes
            )
            t_agg = time.time() - t_agg

            selected_clients = accepted_ids

        elif use_anomaly_detection and len(committee_ids) == 0:
            # No committee — just aggregate all training updates
            selected_clients = list(training_ids)
            accepted_ids_list = sorted(training_updates.keys())
            updates_list = [training_updates[cid] for cid in accepted_ids_list]
            weights = [client_data_sizes.get(cid, 1) for cid in accepted_ids_list] if client_data_sizes else None
            aggregated_params = self.aggregation_method.aggregate(updates_list, weights)
            deviation_scores = {cid: 0.0 for cid in training_ids}
            t_pow = t_filter = t_agg = 0.0
        else:
            # Anomaly detection disabled
            selected_clients = list(training_ids)
            accepted_ids_list = sorted(training_updates.keys())
            updates_list = [training_updates[cid] for cid in accepted_ids_list]
            weights = [client_data_sizes.get(cid, 1) for cid in accepted_ids_list] if client_data_sizes else None
            aggregated_params = self.aggregation_method.aggregate(updates_list, weights)
            deviation_scores = {}
            t_pow = t_filter = t_agg = 0.0

        # Track cumulative detected malicious
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

        # Detection metrics
        detection_metrics_dict = None
        if all_clients is not None and use_anomaly_detection:
            detection_metrics_dict = self.calculate_round_detection_metrics(
                all_clients, round_num=self.round_count, flagged_this_round=all_flagged
            )

        # Committee rotation (same as CMFLDefense)
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

        # Build metrics dict
        metrics = {
            'round': self.round_count,
            'training_clients': sorted(training_ids),
            'committee_members': sorted(list(self.committee_members)),
            'selected_clients': selected_clients,
            'flagged_clients': all_flagged,
            'aggregation_clients': selected_clients,
            'pow_metrics': pow_metrics,
            'filter_metrics': filter_metrics,
            'pbft_aggregation': pbft_agg_metrics,
            'deviation_scores': {k: float(v) for k, v in deviation_scores.items()},
            'aggregation_method': self._agg_name,
            'round_time': round_time,
            'layer_timing': {
                'pow_time': t_pow,
                'filter_time': t_filter,
                'agg_time': t_agg,
            },
            'layer_counts': {
                'total_training': len(training_ids),
                'pow_passed': len(training_ids) - len(pow_metrics.get('norm_rejected', []))
                              - len(pow_metrics.get('loss_rejected', []))
                              - len(pow_metrics.get('nan_rejected', [])),
                'filter_accepted': len(selected_clients),
                'final_aggregated': len(selected_clients),
            },
            'layers_enabled': {
                'pow': self.enable_pow,
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
        """
        Layer 2 of CDCFL-I: Committee Consensus Check.

        Each committee member verifies delta_candidate by:
        1. Cosine similarity between candidate and their own gradient
        2. Norm ratio check (candidate vs member gradient)
        Supermajority (floor(C/2)+1) required.
        """
        committee_ids = sorted(committee_updates.keys())
        C = len(committee_ids)
        if C == 0:
            return True, {'approve_count': 0, 'required': 0, 'total': 0, 'passed': True, 'per_member_votes': {}}

        required = C // 2 + 1

        # Flatten candidate
        candidate_vec = torch.cat([v.flatten().float() for v in delta_candidate.values()])
        candidate_norm = torch.norm(candidate_vec, p=2).item()

        per_member_votes = {}
        approve_count = 0

        for cid in committee_ids:
            member_vec = torch.cat([v.flatten().float() for v in committee_updates[cid].values()])
            member_norm = torch.norm(member_vec, p=2).item()

            # Cosine similarity
            if candidate_norm > 1e-10 and member_norm > 1e-10:
                cos_sim = (torch.dot(candidate_vec, member_vec) / (candidate_norm * member_norm)).item()
            else:
                cos_sim = 0.0

            # Norm ratio
            norm_ratio = candidate_norm / max(member_norm, 1e-10)

            # Vote: approve if cosine > -0.5 AND 0.1 < norm_ratio < 10.0
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

        metrics = {
            'approve_count': approve_count,
            'required': required,
            'total': C,
            'passed': passed,
            'per_member_votes': per_member_votes,
        }
        return passed, metrics

    # =========================================================================
    # CDCFL-I: PoW Finalization
    # =========================================================================

    def _pow_finalize(self, delta_candidate: Dict[str, torch.Tensor],
                      nonce: int) -> Tuple[bool, str, Dict]:
        """
        PoW Finalization for CDCFL-I (Step 6).
        Computes H(delta_candidate || nonce) as a commitment hash.
        In simulation this always passes.
        """
        hasher = hashlib.sha256()
        for name in sorted(delta_candidate.keys()):
            tensor_bytes = delta_candidate[name].flatten().float().cpu().numpy().tobytes()[:64]
            hasher.update(tensor_bytes)
        hasher.update(str(nonce).encode())
        pow_hash = hasher.hexdigest()

        # Simulation: always passes
        pow_passed = True

        metrics = {
            'nonce': nonce,
            'hash': pow_hash,
            'passed': pow_passed,
        }
        return pow_passed, pow_hash, metrics

    # =========================================================================
    # CDCFL-I: pBFT Finalization (alternative to PoW)
    # =========================================================================

    def _pbft_finalize(self, delta_candidate: Dict[str, torch.Tensor],
                       committee_updates: Dict[int, Dict[str, torch.Tensor]]
                       ) -> Tuple[bool, Dict]:
        """
        pBFT finalization for CDCFL-I: committee votes on the aggregated result.
        Alternative to PoW — pure majority vote without hash computation.
        """
        consensus_passed, metrics = self._committee_consensus_check(
            delta_candidate, committee_updates
        )
        metrics['finalization_method'] = 'pbft'
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
        """
        Execute one full CDCFL-I round.

        Pipeline: Robust Aggregation (ALL gradients) → Consensus Check → PoW Finalization
        NO pre-filtering. The committee verifies the aggregated result, not individual clients.

        Returns:
            (aggregated_params, flagged_clients, metrics)
            flagged_clients is always [] for CDCFL-I (no per-client detection)
        """
        self.round_count += 1
        round_start = time.time()
        all_client_ids = sorted(client_updates.keys())

        # Step 1: Separate training vs committee updates
        training_ids = [cid for cid in all_client_ids if cid in self.training_clients]
        committee_ids = [cid for cid in all_client_ids if cid in self.committee_members]

        training_updates = {cid: client_updates[cid] for cid in training_ids}
        committee_updates_dict = {cid: client_updates[cid] for cid in committee_ids}

        round_accepted = True
        consensus_metrics = {}
        finalization_metrics = {}
        contribution_scores = {}

        # ---- Layer 1: Robust Aggregation of ALL training gradients ----
        updates_list = [training_updates[cid] for cid in training_ids]
        if client_data_sizes:
            weights = [client_data_sizes.get(cid, 1) for cid in training_ids]
        else:
            weights = [1.0] * len(updates_list)

        if len(updates_list) > 0:
            delta_candidate = self.aggregation_method.aggregate(updates_list, weights)
        elif global_params is not None:
            delta_candidate = {k: v.clone() for k, v in global_params.items()}
        else:
            delta_candidate = client_updates[all_client_ids[0]]

        # ---- Layer 2: Committee Consensus Check ----
        if len(committee_ids) > 0 and use_anomaly_detection:
            consensus_passed, consensus_metrics = self._committee_consensus_check(
                delta_candidate, committee_updates_dict
            )
            if not consensus_passed:
                round_accepted = False
                if global_params is not None:
                    delta_candidate = {k: v.clone() for k, v in global_params.items()}
        else:
            consensus_metrics = {'approve_count': 0, 'required': 0, 'total': 0, 'passed': True, 'per_member_votes': {}}

        # ---- Step 6: Finalization (PoW or pBFT) ----
        if round_accepted:
            if self.finalization_method == 'pbft':
                pbft_passed, pbft_metrics = self._pbft_finalize(
                    delta_candidate, committee_updates_dict
                )
                final_passed = consensus_passed and pbft_passed
                finalization_metrics = pbft_metrics
            else:
                pow_passed, pow_hash, pow_metrics = self._pow_finalize(
                    delta_candidate, self.round_count
                )
                final_passed = consensus_passed and pow_passed
                finalization_metrics = pow_metrics
                finalization_metrics['finalization_method'] = 'pow'

            if not final_passed:
                round_accepted = False
                if global_params is not None:
                    delta_candidate = {k: v.clone() for k, v in global_params.items()}

        aggregated_params = delta_candidate

        # Track consensus and round acceptance
        self.consensus_history.append(consensus_metrics)
        self.round_acceptance_history.append(round_accepted)

        # Print consensus status
        if consensus_metrics.get('total', 0) > 0:
            status = "PASSED" if consensus_metrics['passed'] else "FAILED"
            print(f"  [CONSENSUS] {consensus_metrics['approve_count']}/{consensus_metrics['total']} "
                  f"approved (required: {consensus_metrics['required']}) — {status}")

        # Compute contribution scores for committee election
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

        # Committee rotation (same as CDCFL-II)
        if self.round_count % self.committee_rotation_rounds == 0 and len(training_ids) > 0:
            # Use contribution scores (cosine similarity) for election
            # Convert to deviation-like scores: lower = more aligned = more honest
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
            'finalization_method': self.finalization_method,
            'round_accepted': round_accepted,
            'contribution_scores': {k: float(v) for k, v in contribution_scores.items()},
            'round_time': round_time,
        }

        return aggregated_params, [], metrics

    # =========================================================================
    # Committee election (uses deviation scores instead of L2 scores)
    # =========================================================================

    def elect_new_committee(self, training_client_ids: List[int],
                            deviation_scores: Dict[int, float]) -> List[int]:
        """
        Paper-exact committee election using deviation scores.
        Sort by deviation (ascending), pick middle-ranked clients.
        Lower deviation = closer to consensus (honest), higher = outlier.
        """
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
    # Detection metrics (same interface as CMFLDefense)
    # =========================================================================

    def calculate_round_detection_metrics(self, all_clients: List, round_num: int = 0,
                                          flagged_this_round: Optional[List[int]] = None):
        participants = list(self.training_clients | self.committee_members)

        malicious_participants = [
            cid for cid in participants
            if cid < len(all_clients) and getattr(all_clients[cid], 'is_malicious', False)
        ]
        benign_participants = [
            cid for cid in participants
            if cid < len(all_clients) and not getattr(all_clients[cid], 'is_malicious', False)
        ]

        detected_ids = flagged_this_round if flagged_this_round is not None else list(self.detected_malicious)

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
        """Summarize PoW rejections, filter rejections, and per-layer timing
        across all rounds. Used by E4 (PoW analysis) and E7 (overhead)."""
        # PoW rejection totals
        total_norm = sum(r.get('norm_rejected', 0) for r in self.pow_rejection_history)
        total_loss = sum(r.get('loss_rejected', 0) for r in self.pow_rejection_history)
        total_nan = sum(r.get('nan_rejected', 0) for r in self.pow_rejection_history)
        total_pow_rejected = sum(r.get('total_rejected', 0) for r in self.pow_rejection_history)

        # Filter rejection totals
        total_filter_flagged = sum(r.get('flagged_count', 0) for r in self.filter_rejection_history)

        # Per-layer timing (from detection_history, stored in enhanced_metrics via coordinator)
        # We can also compute from pbft_aggregation_history
        n_rounds = max(1, self.round_count)

        return {
            'pow_norm_rejected': total_norm,
            'pow_loss_rejected': total_loss,
            'pow_nan_rejected': total_nan,
            'pow_total_rejected': total_pow_rejected,
            'filter_total_flagged': total_filter_flagged,
            'pow_rejection_per_round': self.pow_rejection_history,
            'filter_rejection_per_round': self.filter_rejection_history,
            'pbft_history': self.pbft_aggregation_history,
            'num_rounds': n_rounds,
        }

    # =========================================================================
    # Utilities
    # =========================================================================

    def _compute_update_norm(self, update: Dict) -> float:
        """Compute L2 norm of a flattened update."""
        vec = torch.cat([v.flatten().float() for v in update.values()])
        return torch.norm(vec, p=2).item()
