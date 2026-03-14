import torch
import numpy as np
from typing import List
from collections import defaultdict
from defense import CMFLDefense
import os
import csv

class DecentralizedFLCoordinator:
    """
    Coordinates decentralized federated learning among peers with committee defense.
    - Manages federated learning rounds
    - Applies committee defense to detect and exclude malicious clients
    - Tracks performance metrics and detection statistics
    - Provides evaluation capabilities for attack success rates
    """

    def __init__(self, clients, use_defense=True, defense_type='cmfl',
             clients_per_round=None, aggregation_method='fedavg', use_anomaly_detection=None,
             lr=0.001, batch_size=None, epochs=1, enable_verification=False,
             verification_kwargs=None, **defense_kwargs):
        self.clients = clients
        self.use_defense = use_defense  # Keep for backward compatibility
        self.defense_type = defense_type
        self.clients_per_round = clients_per_round  # Fixed client participation per round
        self.aggregation_method = aggregation_method  # Store for use in run_federated_learning
        self._lr = lr  # Learning rate passed to client.local_training()
        self._batch_size = batch_size  # Batch size passed to client.local_training()
        self._epochs = epochs  # Local training epochs (paper tau=1)

        # OPTION A: Separate committee structure from anomaly detection
        # If use_anomaly_detection not specified, default to use_defense value (backward compatibility)
        if use_anomaly_detection is None:
            use_anomaly_detection = use_defense
        self.use_anomaly_detection = use_anomaly_detection

        # Initialize committee structure (ALWAYS if defense_type specified)
        if defense_type in ('cd_cfl', 'cdcfl_ii'):
            from defense import CDCFLDefense
            self.defense = CDCFLDefense(
                num_clients=len(clients),
                robust_agg_method=aggregation_method,
                **defense_kwargs
            )
            self.defense_type = 'cdcfl_ii'  # normalize alias
            self.committee_defense = None
        elif defense_type == 'cdcfl_i':
            from defense import CDCFLDefense
            fin_method = defense_kwargs.pop('consensus_type', defense_kwargs.pop('finalization_method', 'pow'))
            self.defense = CDCFLDefense(
                num_clients=len(clients),
                robust_agg_method=aggregation_method,
                consensus_type=fin_method,
                **defense_kwargs
            )
            self.defense_type = 'cdcfl_i'
            self.committee_defense = None
        elif defense_type and defense_type != 'none':
            self.defense = CMFLDefense(
                num_clients=len(clients),
                aggregation_method=aggregation_method,
                **defense_kwargs
            )
            self.committee_defense = None
        else:
            self.defense = None
            self.committee_defense = None

        self.round_history = []
        self.global_accuracy_history = []
        self.poisoned_participation_log = defaultdict(list)

        # TIMING TRACKING: Track time per round for plotting
        self.round_times = []  # Time taken for each round (seconds)
        self.cumulative_times = []  # Cumulative time up to each round (seconds)
        self._training_start_time = None  # Overall training start time

        # Track committee defense metrics
        self.committee_metrics = {
            'detected_malicious': [],
            'actual_malicious': [],
            'false_positives': [],
            'detection_rate': 0.0,
            'false_positive_rate': 0.0
        }

        # Track enhanced defense metrics (Committee-based only)
        self.enhanced_metrics = {
            'reputation_scores': [],
            'adaptive_thresholds': [],
            'defense_stats': []
        }

        # Global model verification (post-consensus, pre-dissemination)
        if enable_verification:
            vkw = verification_kwargs or {}
            from verifier import GlobalModelVerifier
            self._verifier = GlobalModelVerifier(**vkw)
        else:
            self._verifier = None

    def run_federated_learning(self, rounds=10, aggregation_method=None,
                            test_loader=None, model_constructor=None, round_offset=0, total_rounds=None):
        """
        Run decentralized FL for the given number of rounds.
        """
        if aggregation_method is None:
            aggregation_method = self.aggregation_method

        # Identify malicious clients (once)
        malicious_count = sum(1 for c in self.clients if c.is_malicious)
        malicious_ids = [c.client_id for c in self.clients if c.is_malicious]
        self.committee_metrics['actual_malicious'] = malicious_ids

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        accuracy_history = []
        loss_history = []

        # FedAvg: maintain a single global model that all clients train from.
        # Initialize from client 0 (all clients share the same init after build_clients fix).
        if not hasattr(self, '_global_params') or self._global_params is None:
            self._global_params = {
                name: param.data.clone()
                for name, param in self.clients[0].model.named_parameters()
            }

        import time
        if self._training_start_time is None:
            self._training_start_time = time.time()

        for round_num in range(rounds):
            round_start_time = time.time()
            current_round = round_num + 1 + round_offset
            display_total = total_rounds if total_rounds is not None else (rounds + round_offset)

            # Milestone logging: round 1, every 10th, last round
            _milestone = (current_round == 1 or current_round == display_total
                        or current_round % 10 == 0)

            # --- Client selection ---
            if self.defense and self.defense_type in ('cdcfl_ii', 'cdcfl_i', 'cmfl'):
                training_client_ids = self.defense.activate_training_clients()
                active_client_ids = training_client_ids + list(self.defense.committee_members)
            else:
                if self.clients_per_round is not None and self.clients_per_round < len(self.clients):
                    import random
                    active_client_ids = random.sample(range(len(self.clients)), self.clients_per_round)
                else:
                    active_client_ids = list(range(len(self.clients)))

            # --- FedAvg: distribute current global model BEFORE local training ---
            for cid in active_client_ids:
                client = self.clients[cid]
                with torch.no_grad():
                    for name, param in client.model.named_parameters():
                        if name in self._global_params:
                            param.data.copy_(self._global_params[name])

            # --- Per-round attack setup (model-level only) ---
            if malicious_count > 0:
                model_poison_active = [
                    cid for cid in active_client_ids
                    if cid < len(self.clients)
                    and self.clients[cid].is_malicious
                    and self.clients[cid].attack_instance is not None
                    and self.clients[cid].attack_instance.POISONS_MODEL
                ]
                if model_poison_active:
                    for client_id in model_poison_active:
                        try:
                            self.clients[client_id].poison_local_data()
                        except Exception as e:
                            print(f"  [ERROR] Client {client_id}: Attack setup failed - {e}")

            # --- Loss BEFORE training (for CDCFL-II PoW check only) ---
            losses_before = {}
            if self.defense_type == 'cdcfl_ii':
                # Only CDCFL-II needs loss_before for PoW pre-filter
                for cid in active_client_ids:
                    try:
                        losses_before[cid] = self.clients[cid].compute_loss_only()
                    except Exception:
                        losses_before[cid] = 0.0

            # --- Local training ---
            local_updates = []
            client_losses = []
            client_accuracies = []
            client_id_to_update = {}

            _t_train_start = time.time()

            # --- Gradient mean header (only when an attack is present) ---
            if malicious_count > 0:
                print(f"  [R{current_round}/{display_total}] Gradient summary (mean |value| per client):")

            for client_id in active_client_ids:
                client = self.clients[client_id]
                params, loss, acc = client.local_training(epochs=self._epochs, lr=self._lr, batch_size=self._batch_size)

                # --- Per-client gradient scalar (skipped for clean baseline runs) ---
                if malicious_count > 0:
                    all_vals = [
                        tensor.abs().mean().item()
                        for tensor in params.values()
                        if isinstance(tensor, torch.Tensor)
                    ]
                    grad_mean = float(np.mean(all_vals)) if all_vals else float('nan')
                    role = "MAL" if getattr(client, 'is_malicious', False) else "BEN"
                    print(f"    Client {client_id:>3} [{role}] | grad_mean={grad_mean:.6f}")

                if self.defense and self.defense_type in ('cmfl', 'cdcfl_ii', 'cdcfl_i'):
                    client_id_to_update[client_id] = params
                else:
                    local_updates.append(params)
                client_losses.append(loss)
                client_accuracies.append(acc)

                if getattr(client, 'is_malicious', False):
                    self.poisoned_participation_log[client_id].append(current_round)

            _t_train = time.time() - _t_train_start

            # Collect local dataset sizes for data-size weighted aggregation
            client_data_sizes = {cid: len(self.clients[cid].X) for cid in active_client_ids}

            # --- NaN/Inf detection (concise) ---
            nan_loss_clients = []
            valid_loss_clients = []
            for i, (cid, loss) in enumerate(zip(active_client_ids, client_losses)):
                if not np.isfinite(loss):
                    nan_loss_clients.append(cid)
                    client_losses[i] = 1e10
                else:
                    valid_loss_clients.append(cid)

            self._valid_loss_clients = valid_loss_clients
            self._nan_loss_clients_current_round = nan_loss_clients

            if nan_loss_clients:
                print(f"[R{current_round}] NaN/Inf losses: {len(nan_loss_clients)}/{len(active_client_ids)} clients {sorted(nan_loss_clients)}")
                if not hasattr(self, '_nan_flagged_clients'):
                    self._nan_flagged_clients = set()
                self._nan_flagged_clients.update(nan_loss_clients)

            # --- Aggregation ---
            _t_agg_start = time.time()
            anomalous_clients = []
            defense_metrics = {}

            if self.defense:
                malicious_ids_for_diagnostics = list(malicious_ids) if malicious_ids else None

                if self.defense_type == 'cdcfl_ii':
                    client_losses_dict = {cid: client_losses[i] for i, cid in enumerate(active_client_ids)}
                    global_params, anomalous_clients, defense_metrics = self.defense.cdcfl_ii_round(
                        client_id_to_update,
                        client_losses_before=losses_before,
                        client_losses_after=client_losses_dict,
                        malicious_client_ids=malicious_ids_for_diagnostics,
                        all_clients=self.clients,
                        client_data_sizes=client_data_sizes,
                        use_anomaly_detection=self.use_anomaly_detection,
                        global_params=self._global_params,
                    )
                    # Handle round rejection (PoW/pBFT finalization failed)
                    if not defense_metrics.get('round_accepted', True):
                        global_params = self._global_params
                    self.enhanced_metrics['defense_stats'].append(defense_metrics)

                elif self.defense_type == 'cdcfl_i':
                    client_losses_dict = {cid: client_losses[i] for i, cid in enumerate(active_client_ids)}
                    global_params, anomalous_clients, defense_metrics = self.defense.cdcfl_i_round(
                        client_updates=client_id_to_update,
                        client_losses_after=client_losses_dict,
                        malicious_client_ids=malicious_ids_for_diagnostics,
                        all_clients=self.clients,
                        client_data_sizes=client_data_sizes,
                        global_params=self._global_params,
                        use_anomaly_detection=self.use_anomaly_detection,
                    )
                    # Handle round rejection (consensus failed)
                    if not defense_metrics.get('round_accepted', True):
                        global_params = self._global_params
                    self.enhanced_metrics['defense_stats'].append(defense_metrics)

                elif self.defense_type == 'cmfl':
                    client_losses_dict = {cid: client_losses[i] for i, cid in enumerate(active_client_ids)}
                    global_params, anomalous_clients, defense_metrics = self.defense.cmfl_round(
                        client_id_to_update, client_losses_dict,
                        malicious_client_ids=malicious_ids_for_diagnostics,
                        all_clients=self.clients,
                        use_anomaly_detection=self.use_anomaly_detection,
                        client_data_sizes=client_data_sizes
                    )
                    self.enhanced_metrics['defense_stats'].append(defense_metrics)
                    self.enhanced_metrics['reputation_scores'].append(defense_metrics.get('reputation_scores', {}))

                # Merge NaN-flagged clients and update metrics
                if self.use_anomaly_detection:
                    if hasattr(self, '_nan_flagged_clients') and self._nan_flagged_clients:
                        anomalous_clients = list(set(anomalous_clients) | self._nan_flagged_clients)
                    self._update_committee_metrics(malicious_ids, anomalous_clients)
            else:
                global_params = self._aggregate_updates(local_updates, method=aggregation_method,
                                                    active_client_ids=active_client_ids)

            _t_agg = time.time() - _t_agg_start

            # --- NaN check on global model ---
            _t_dist_start = time.time()
            has_nan_params = False
            for name, param_tensor in global_params.items():
                if isinstance(param_tensor, torch.Tensor) and not torch.isfinite(param_tensor).all():
                    has_nan_params = True
                    break

            if has_nan_params:
                print(f"[R{current_round}] CRITICAL: Global model NaN/Inf — skipping distribution")
            else:
                # Update stored global model for next round's distribution
                self._global_params = {
                    name: val.cpu().clone() if isinstance(val, torch.Tensor) else val
                    for name, val in global_params.items()
                }

                # Distribute global model to client 0 (for eval) and
                # on final round to all malicious clients (for ASR eval)
                broadcast_ids = {0}
                _is_final = (total_rounds is not None and current_round >= display_total)
                if _is_final:
                    broadcast_ids.update(i for i, c in enumerate(self.clients) if c.is_malicious)
                for cid in broadcast_ids:
                    client = self.clients[cid]
                    with torch.no_grad():
                        for name, param in client.model.named_parameters():
                            if name in global_params:
                                param.data.copy_(global_params[name])
            _t_dist = time.time() - _t_dist_start

            # --- Evaluate test accuracy & loss on global model each round ---
            test_accuracy = 0.0
            test_loss = np.inf
            if test_loader is not None and not has_nan_params:
                test_accuracy, test_loss = self.evaluate_on_test_set(test_loader)

            # --- Per-round detection metrics ---
            round_detection = None
            if self.defense and self.use_anomaly_detection:
                actual_set = set(malicious_ids)
                detected_set = set(anomalous_clients)

                # Ground truth: only training clients are subject to detection
                if self.defense and self.defense_type in ('cdcfl_ii', 'cdcfl_i', 'cmfl'):
                    evaluatable_clients = set(training_client_ids)
                else:
                    evaluatable_clients = set(active_client_ids)

                mal_in_round = evaluatable_clients & actual_set
                benign_in_round = evaluatable_clients - actual_set

                if self.defense_type == 'cdcfl_i':
                    # CDCFL-I: no per-client detection, report round-level metrics
                    round_detection = {
                        'tp': 0, 'fp': 0, 'tn': len(benign_in_round), 'fn': len(mal_in_round),
                        'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
                        'fpr': 0.0, 'fnr': 100.0, 'dacc': 0.0,
                        'num_participants': len(evaluatable_clients),
                        'num_malicious': len(mal_in_round),
                        'num_benign': len(benign_in_round),
                        'flagged': [],
                        'strategy': 'CDCFL-I',
                        'round_accepted': defense_metrics.get('round_accepted', True),
                        'consensus_approval': defense_metrics.get('consensus_metrics', {}).get('approve_count', 0),
                    }
                else:
                    # Compute confusion matrix per spec requirement #4
                    tp = fp = tn = fn = 0
                    for cid in evaluatable_clients:
                        is_malicious = cid in actual_set
                        is_rejected = cid in detected_set
                        if is_malicious and is_rejected:
                            tp += 1
                        elif is_malicious and not is_rejected:
                            fn += 1
                        elif not is_malicious and is_rejected:
                            fp += 1
                        else:
                            tn += 1
                    total_p = tp + fp + fn + tn

                    # Validate counts (spec requirement #6)
                    if total_p != len(evaluatable_clients):
                        print(f"[R{current_round}] WARNING: Detection count mismatch: "
                            f"TP+FP+TN+FN={total_p} != evaluatable={len(evaluatable_clients)}")

                    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
                    recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
                    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
                    fpr_val = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0.0
                    fnr_val = 100.0 - recall
                    dacc = ((tp + tn) / total_p * 100) if total_p > 0 else 0.0

                    round_detection = {
                        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                        'precision': precision, 'recall': recall,
                        'f1_score': f1, 'fpr': fpr_val, 'fnr': fnr_val, 'dacc': dacc,
                        'num_participants': len(evaluatable_clients),
                        'num_malicious': len(mal_in_round),
                        'num_benign': len(benign_in_round),
                        'flagged': sorted(list(detected_set)),
                        'round_accepted': defense_metrics.get('round_accepted', True),
                        'consensus_type': defense_metrics.get('consensus_type',
                                            getattr(self.defense, 'consensus_type', 'N/A')),
                    }

            # --- Global Model Verification (post-consensus, pre-dissemination) ---
            verification_status = 'BENIGN'
            verification_details = {}
            if self._verifier is not None and test_loader is not None and not has_nan_params:
                verification_status, verification_details = self._verifier.verify(
                    global_params, test_accuracy, current_round
                )

            # --- Track performance ---
            if valid_loss_clients:
                valid_indices = [i for i, cid in enumerate(active_client_ids) if cid in valid_loss_clients]
                valid_losses = [client_losses[i] for i in valid_indices if np.isfinite(client_losses[i])]
                avg_loss = np.mean(valid_losses) if valid_losses else np.inf
            else:
                avg_loss = np.inf

            self.global_accuracy_history.append(test_accuracy)
            loss_history.append(test_loss if test_loader is not None else avg_loss)
            accuracy_history.append(test_accuracy)

            self.round_history.append({
                'round': current_round,
                'client_accuracies': client_accuracies.copy(),
                'avg_accuracy': test_accuracy,
                'client_losses': client_losses.copy(),
                'avg_loss': avg_loss,
                'test_accuracy': test_accuracy,
                'test_loss': test_loss,
                'anomalous_clients': anomalous_clients.copy(),
                'detection_metrics': round_detection,
                'verification_status': verification_status,
                'verification_details': verification_details,
            })

            # --- Timing ---
            round_end_time = time.time()
            round_duration = round_end_time - round_start_time
            cumulative_time = round_end_time - self._training_start_time
            self.round_times.append(round_duration)
            self.cumulative_times.append(cumulative_time)

            # --- Print per-round summary ---
            consensus_label = 'N/A'
            agg_label = aggregation_method
            if self.defense and self.defense_type in ('cdcfl_i', 'cdcfl_ii'):
                consensus_label = getattr(self.defense, 'consensus_type', 'N/A').upper()
                agg_label = getattr(self.defense, '_agg_name', aggregation_method)

            if self._verifier is not None and verification_details:
                prev_acc = self.global_accuracy_history[-2] * 100 if len(self.global_accuracy_history) >= 2 else 0.0
                param_shift = verification_details.get('parameter_deviation', 0.0)
                print(f"  [R{current_round}/{display_total}] "
                    f"Consensus: {consensus_label} | Agg: {agg_label}\n"
                    f"    AccBefore: {prev_acc:.1f}% | AccAfter: {test_accuracy*100:.1f}% | "
                    f"ParamShift: {param_shift:.4f} | "
                    f"Status: {verification_status} | "
                    f"{round_duration:.1f}s", end="")
            else:
                print(f"  [R{current_round}/{display_total}] "
                    f"TestAcc: {test_accuracy*100:.2f}% | TestLoss: {test_loss:.4f} | "
                    f"TrainLoss: {avg_loss:.4f} | "
                    f"{round_duration:.1f}s/round ({len(active_client_ids)} clients)", end="")

            if round_detection:
                rd = round_detection
                # Show consensus acceptance status for CDCFL-I / CDCFL-II
                accepted = rd.get('round_accepted', True)
                ctype = rd.get('consensus_type', None)
                consensus_suffix = ""
                if ctype is not None:
                    status_str = "ACCEPTED" if accepted else "REJECTED"
                    consensus_suffix = f" | {ctype.upper()} {status_str}"
                print(f"\n    Malicious Present: {rd['num_malicious']} | "
                    f"Detected (TP): {rd['tp']} | Missed (FN): {rd['fn']} | "
                    f"FP: {rd['fp']} | TN: {rd['tn']} | "
                    f"Prec={rd['precision']:.1f}% Rec={rd['recall']:.1f}% "
                    f"F1={rd['f1_score']:.1f}% FPR={rd['fpr']:.1f}%"
                    f"{consensus_suffix}")
            else:
                print()

        # --- Final summary ---
        current_final_round = round_offset + rounds
        is_truly_final = (total_rounds is None) or (current_final_round >= total_rounds)

        if is_truly_final:
            self._print_final_summary()

        return self.global_accuracy_history, loss_history, self.round_history

    def _aggregate_updates(self, client_updates, method='mean', active_client_ids=None):
        """
        Simple aggregation without defense.
        """
        method_lower = method.lower()
        if method_lower == 'fedavg':
            method_lower = 'mean'  # FedAvg is standard mean aggregation

        if method_lower in ['krum', 'multi_krum', 'trimmed_mean', 'median']:
            from defense import get_aggregation_method

            n = len(client_updates)
            if active_client_ids is not None:
                n_attackers = sum(
                    1 for cid in active_client_ids
                    if cid < len(self.clients) and self.clients[cid].is_malicious
                )
            else:
                mal_ratio = sum(1 for c in self.clients if c.is_malicious) / max(1, len(self.clients))
                n_attackers = max(1, int(n * mal_ratio))

            s = max(1, n - n_attackers)
            trim_ratio = n_attackers / n if n > 0 else 0.1

            aggregator = get_aggregation_method(
                method_lower, n_attackers=n_attackers, s=s, trim_ratio=trim_ratio
            )
            return aggregator.aggregate(client_updates)

        aggregated_params = {}
        param_names = client_updates[0].keys()

        for name in param_names:
            param_stack = torch.stack([update[name] for update in client_updates])
            if method_lower == 'mean':
                aggregated_params[name] = torch.mean(param_stack, dim=0)
            else:
                raise ValueError(f"Unknown aggregation method: {method}")

        return aggregated_params

    def _update_committee_metrics(self, actual_malicious, detected_malicious):
        """
        Update committee defense performance metrics.
        """
        actual_set = set(actual_malicious)
        detected_set = set(detected_malicious)
        total_clients = len(self.clients)

        self.committee_metrics['actual_malicious'] = list(actual_malicious)

        if len(actual_malicious) > 0:
            true_positives = len(actual_set & detected_set)
            self.committee_metrics['detection_rate'] = true_positives / len(actual_malicious)
        else:
            self.committee_metrics['detection_rate'] = 0.0

        benign_clients = total_clients - len(actual_malicious)
        if benign_clients > 0:
            false_positives = len(detected_set - actual_set)
            self.committee_metrics['false_positive_rate'] = false_positives / benign_clients
        else:
            self.committee_metrics['false_positive_rate'] = 0.0

    def _print_final_summary(self):
        """Print final summary: per-round test accuracy/loss and detection metrics."""
        num_rounds = len(self.round_history)
        has_detection = any(r.get('detection_metrics') is not None for r in self.round_history)

        print("\n" + "=" * 100)

        test_accs = [r.get('test_accuracy', 0.0) for r in self.round_history]
        test_losses = [r.get('test_loss', np.inf) for r in self.round_history]
        final_acc = test_accs[-1] if test_accs else 0.0
        best_acc = max(test_accs) if test_accs else 0.0
        best_round = test_accs.index(best_acc) + 1 if test_accs else 0

        print(f"  Final TestAcc: {final_acc*100:.2f}%  |  Best TestAcc: {best_acc*100:.2f}% (R{best_round})")

        if has_detection:
            rounds_with_det = [r['detection_metrics'] for r in self.round_history if r.get('detection_metrics')]
            if rounds_with_det:
                avg_prec = np.mean([d['precision'] for d in rounds_with_det])
                avg_rec = np.mean([d['recall'] for d in rounds_with_det])
                avg_fpr = np.mean([d['fpr'] for d in rounds_with_det])
                avg_fnr = np.mean([d['fnr'] for d in rounds_with_det])
                avg_f1 = np.mean([d['f1_score'] for d in rounds_with_det])
                avg_dacc = np.mean([d.get('dacc', 0.0) for d in rounds_with_det])
                total_tp = sum(d['tp'] for d in rounds_with_det)
                total_fp = sum(d['fp'] for d in rounds_with_det)
                total_tn = sum(d['tn'] for d in rounds_with_det)
                total_fn = sum(d['fn'] for d in rounds_with_det)

                print(f"\n  {'─'*50}")
                print(f"  Confusion Matrix (cumulative across {len(rounds_with_det)} rounds):")
                print(f"  {'':>20} {'Predicted Mal':>15} {'Predicted Ben':>15}")
                print(f"  {'Actual Malicious':>20} {total_tp:>15} {total_fn:>15}")
                print(f"  {'Actual Benign':>20} {total_fp:>15} {total_tn:>15}")
                print(f"  {'─'*50}")

                print(f"  {'Metric':<25} {'Value':>10}")
                print(f"  {'─'*36}")
                print(f"  {'Precision':<25} {avg_prec:>9.1f}%")
                print(f"  {'Recall (TPR)':<25} {avg_rec:>9.1f}%")
                print(f"  {'F1 Score':<25} {avg_f1:>9.1f}%")
                print(f"  {'FPR':<25} {avg_fpr:>9.1f}%")
                print(f"  {'FNR':<25} {avg_fnr:>9.1f}%")
                print(f"  {'Detection Accuracy':<25} {avg_dacc:>9.1f}%")
                print(f"  {'─'*36}")

        total_time = self.cumulative_times[-1] if self.cumulative_times else 0.0
        print(f"  Total Time: {total_time:.1f}s ({total_time/60:.1f}min)")

    def get_timing_data(self):
        """
        Get timing data for plotting time vs accuracy/round graphs.
        """
        total_time = self.cumulative_times[-1] if self.cumulative_times else 0.0
        avg_round_time = np.mean(self.round_times) if self.round_times else 0.0

        return {
            'round_times': self.round_times.copy(),
            'cumulative_times': self.cumulative_times.copy(),
            'total_time': total_time,
            'avg_round_time': avg_round_time,
            'min_round_time': min(self.round_times) if self.round_times else 0.0,
            'max_round_time': max(self.round_times) if self.round_times else 0.0,
        }

    def get_committee_metrics(self):
        """
        Get detection metrics computed per-round from round_history.
        """
        rounds_with_det = [
            r['detection_metrics'] for r in self.round_history
            if r.get('detection_metrics') is not None
        ]

        malicious_ids = set(c.client_id for c in self.clients if c.is_malicious)
        num_malicious = len(malicious_ids)
        num_benign = len(self.clients) - num_malicious

        if not rounds_with_det:
            return {
                'detection_rate': 0.0, 'false_positive_rate': 0.0,
                'false_negative_rate': 0.0, 'precision': 0.0, 'recall': 0.0,
                'f1_score': 0.0, 'accuracy': 0.0,
                'true_positives': 0, 'true_negatives': 0,
                'false_positives': 0, 'false_negatives': 0,
                'detected_malicious': 0, 'actual_malicious': list(malicious_ids),
                'total_clients': len(self.clients),
                'num_malicious': num_malicious, 'num_benign': num_benign,
            }

        avg_prec = np.mean([d['precision'] for d in rounds_with_det])
        avg_rec = np.mean([d['recall'] for d in rounds_with_det])
        avg_f1 = np.mean([d['f1_score'] for d in rounds_with_det])
        avg_fpr = np.mean([d['fpr'] for d in rounds_with_det])
        avg_fnr = np.mean([d['fnr'] for d in rounds_with_det])
        avg_dacc = np.mean([d['dacc'] for d in rounds_with_det])
        total_tp = sum(d['tp'] for d in rounds_with_det)
        total_fp = sum(d['fp'] for d in rounds_with_det)
        total_tn = sum(d['tn'] for d in rounds_with_det)
        total_fn = sum(d['fn'] for d in rounds_with_det)

        return {
            'detection_rate': avg_rec,
            'false_positive_rate': avg_fpr,
            'false_negative_rate': avg_fnr,
            'precision': avg_prec,
            'recall': avg_rec,
            'f1_score': avg_f1,
            'accuracy': avg_dacc,
            'true_positives': total_tp,
            'true_negatives': total_tn,
            'false_positives': total_fp,
            'false_negatives': total_fn,
            'detected_malicious': total_tp,
            'actual_malicious': list(malicious_ids),
            'total_clients': len(self.clients),
            'num_malicious': num_malicious,
            'num_benign': num_benign,
        }

    def get_malicious_in_committee_metrics(self):
        """Get N1/N2/N3 infiltration tracking from CMFL defense."""
        if self.use_defense and self.defense is not None:
            if hasattr(self.defense, 'get_infiltration_summary'):
                return self.defense.get_infiltration_summary()
            if hasattr(self.defense, 'get_malicious_in_committee_summary'):
                return self.defense.get_malicious_in_committee_summary()
        return {}

    def get_defense_layer_metrics(self):
        """Get CD-CFL layer-specific metrics (PoW rejections, timing)."""
        if self.defense is not None and hasattr(self.defense, 'get_layer_metrics'):
            layer_metrics = self.defense.get_layer_metrics()
            defense_stats = self.enhanced_metrics.get('defense_stats', [])
            validation_times = []
            filter_times = []
            agg_times = []
            consensus_times = []
            finalize_times = []
            for ds in defense_stats:
                lt = ds.get('layer_timing', {})
                if lt:
                    validation_times.append(lt.get('validation_time', lt.get('pow_time', 0.0)))
                    filter_times.append(lt.get('filter_time', 0.0))
                    agg_times.append(lt.get('agg_time', 0.0))
                    consensus_times.append(lt.get('consensus_time', 0.0))
                    finalize_times.append(lt.get('finalize_time', 0.0))
            if validation_times:
                layer_metrics['avg_validation_time'] = float(np.mean(validation_times))
                layer_metrics['avg_filter_time'] = float(np.mean(filter_times))
                layer_metrics['avg_agg_time'] = float(np.mean(agg_times))
                layer_metrics['avg_consensus_time'] = float(np.mean(consensus_times))
                layer_metrics['avg_finalize_time'] = float(np.mean(finalize_times))
                layer_metrics['total_validation_time'] = float(sum(validation_times))
                layer_metrics['total_filter_time'] = float(sum(filter_times))
                layer_metrics['total_agg_time'] = float(sum(agg_times))
                layer_metrics['total_consensus_time'] = float(sum(consensus_times))
                layer_metrics['total_finalize_time'] = float(sum(finalize_times))
            return layer_metrics
        return {}

    def evaluate_on_test_set(self, test_loader):
        """
        Evaluate global model on test set — returns accuracy AND loss in a
        **single forward pass** (avoids the old double-evaluation bottleneck).
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = torch.nn.CrossEntropyLoss()

        client = self.clients[0]
        client.model.to(device)
        client.model.eval()
        correct, total = 0, 0
        total_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = client.model(batch_X)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item() * batch_y.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else float('inf')
        return accuracy, avg_loss

    def evaluate_attack_success(self, test_X: np.ndarray, y_test: np.ndarray) -> List[float]:
        """
        Evaluate attack success rate for malicious clients.
        """
        attack_success_rates = []

        for client in self.clients:
            if client.is_malicious and client.attack_instance is not None:
                try:
                    success_rate = client.attack_instance.evaluate_attack_success(
                        client.model, test_X, y_test
                    )

                    if success_rate is None:
                        success_rate = 0.0
                    elif isinstance(success_rate, bool):
                        success_rate = 1.0 if success_rate else 0.0
                    elif isinstance(success_rate, torch.Tensor):
                        success_rate = success_rate.item() if success_rate.numel() == 1 else success_rate.float().mean().item()
                    elif isinstance(success_rate, (list, tuple, np.ndarray)):
                        success_rate = float(np.mean(success_rate))
                    else:
                        success_rate = float(success_rate)

                    success_rate = max(0.0, min(1.0, success_rate))
                    attack_success_rates.append(success_rate)
                except Exception as e:
                    attack_success_rates.append(0.0)
            else:
                attack_success_rates.append(0.0)

        return attack_success_rates