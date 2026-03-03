import torch
import numpy as np
from typing import List
from collections import defaultdict
from defense import CMFLDefense


class DecentralizedFLCoordinator:
    """
    Coordinates decentralized federated learning among peers with committee defense.
    
    This coordinator:
    - Manages federated learning rounds
    - Applies committee defense to detect and exclude malicious clients
    - Tracks performance metrics and detection statistics
    - Provides evaluation capabilities for attack success rates
    """

    def __init__(self, clients, use_defense=True, defense_type='cmfl',
                 clients_per_round=None, aggregation_method='fedavg', use_anomaly_detection=None,
                 lr=0.001, batch_size=None, epochs=1, **defense_kwargs):
        """
        Initialize the federated learning coordinator.

        Args:
            clients: List of DecentralizedClient instances
            use_defense: Whether to use committee-based architecture (default: True)
            defense_type: Type of defense ('cmfl', 'none')
                         If not 'none', always creates committee structure
            clients_per_round: Number of clients to participate per round (None = all clients)
            aggregation_method: Aggregation method for both defense and non-defense scenarios
                              ('fedavg', 'krum', 'multi_krum', 'median', 'trimmed_mean')
            use_anomaly_detection: Whether to apply anomaly detection/filtering (default: same as use_defense for backward compatibility)
                                  True = Filter anomalous clients (Defense scenario)
                                  False = No filtering (Baseline/Attack scenario)
            **defense_kwargs: Additional arguments for specific defense types
        """
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
        if defense_type and defense_type != 'none':
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

    def run_federated_learning(self, rounds=10, aggregation_method=None,
                               test_loader=None, model_constructor=None, round_offset=0, total_rounds=None):
        """
        Run decentralized FL for the given number of rounds.

        Args:
            rounds: Number of federated learning rounds to execute in this call
            aggregation_method: Aggregation method ('mean', 'median', 'fedavg', 'krum', 'multi_krum', 'trimmed_mean')
                              If None, uses the aggregation_method set during initialization
            test_loader: Optional test data loader for evaluation
            model_constructor: Optional model constructor for evaluation
            round_offset: Offset for round numbering (used when calling with rounds=1 in a loop)
            total_rounds: Total number of rounds in the entire training (for display purposes)

        Returns:
            Tuple of (accuracy_history, loss_history, round_history)
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
            if self.defense and self.defense_type == 'cmfl':
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

            # --- Local training ---
            local_updates = []
            client_losses = []
            client_accuracies = []
            client_id_to_update = {}

            _t_train_start = time.time()
            for client_id in active_client_ids:
                client = self.clients[client_id]
                params, loss, acc = client.local_training(epochs=self._epochs, lr=self._lr, batch_size=self._batch_size)

                if self.defense and self.defense_type == 'cmfl':
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

                if self.defense_type == 'cmfl':
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
                participating = set(active_client_ids)

                mal_in_round = participating & actual_set
                benign_in_round = participating - actual_set

                tp = len(mal_in_round & detected_set)
                fp = len(benign_in_round & detected_set)
                fn = len(mal_in_round - detected_set)
                tn = len(benign_in_round - detected_set)
                total_p = tp + fp + fn + tn

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
                    'num_participants': len(participating),
                    'num_malicious': len(mal_in_round),
                    'num_benign': len(benign_in_round),
                    'flagged': sorted(list(detected_set)),
                }

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
            })

            # --- Timing ---
            round_end_time = time.time()
            round_duration = round_end_time - round_start_time
            cumulative_time = round_end_time - self._training_start_time
            self.round_times.append(round_duration)
            self.cumulative_times.append(cumulative_time)

            # --- Print per-round summary ---
            print(f"  [R{current_round}/{display_total}] "
                  f"TestAcc: {test_accuracy*100:.2f}% | TestLoss: {test_loss:.4f} | "
                  f"TrainLoss: {avg_loss:.4f} | "
                  f"{round_duration:.1f}s/round ({len(active_client_ids)} clients)", end="")

            if round_detection:
                print(f" | TP={round_detection['tp']} FP={round_detection['fp']} "
                      f"TN={round_detection['tn']} FN={round_detection['fn']} "
                      f"Prec={round_detection['precision']:.1f}% "
                      f"Rec={round_detection['recall']:.1f}% "
                      f"FPR={round_detection['fpr']:.1f}%")
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

        Args:
            client_updates: List of client parameter updates
            method: Aggregation method ('mean', 'median', 'fedavg', 'krum', 'multi_krum', 'trimmed_mean')
            active_client_ids: List of client IDs corresponding to client_updates
                              (used to compute actual attacker count for Krum/Multi-Krum)

        Returns:
            Aggregated parameters
        """
        # Map common names to supported methods
        method_lower = method.lower()
        if method_lower == 'fedavg':
            method_lower = 'mean'  # FedAvg is standard mean aggregation

        # For robust aggregation methods (krum, trimmed_mean), use defense.py implementations
        if method_lower in ['krum', 'multi_krum', 'trimmed_mean', 'median']:
            from defense import get_aggregation_method

            # Compute actual attacker count among active clients
            n = len(client_updates)
            if active_client_ids is not None:
                n_attackers = sum(
                    1 for cid in active_client_ids
                    if cid < len(self.clients) and self.clients[cid].is_malicious
                )
            else:
                # Fallback: estimate from overall malicious ratio
                mal_ratio = sum(1 for c in self.clients if c.is_malicious) / max(1, len(self.clients))
                n_attackers = max(1, int(n * mal_ratio))

            # Multi-Krum paper: s = n - f (select all non-Byzantine clients)
            s = max(1, n - n_attackers)

            # Trimmed Mean: trim_ratio = f/n (trim attacker fraction from each side)
            trim_ratio = n_attackers / n if n > 0 else 0.1

            aggregator = get_aggregation_method(
                method_lower, n_attackers=n_attackers, s=s, trim_ratio=trim_ratio
            )
            return aggregator.aggregate(client_updates)

        # Standard aggregation: FedAvg (mean)
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

        Args:
            actual_malicious: List of actual malicious client IDs
            detected_malicious: List of detected malicious client IDs
        """
        # Convert to sets for easier computation
        actual_set = set(actual_malicious)
        detected_set = set(detected_malicious)
        total_clients = len(self.clients)

        # Store actual malicious IDs (needed for get_committee_metrics)
        self.committee_metrics['actual_malicious'] = list(actual_malicious)

        # Calculate metrics
        if len(actual_malicious) > 0:
            true_positives = len(actual_set & detected_set)
            self.committee_metrics['detection_rate'] = true_positives / len(actual_malicious)
        else:
            self.committee_metrics['detection_rate'] = 0.0

        # False positive rate
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
        print("FINAL TRAINING SUMMARY")
        print("=" * 100)

        # Header
        if has_detection:
            print(f"{'Round':>5} | {'TestAcc':>8} | {'TestLoss':>9} | {'TrainLoss':>10} | "
                  f"{'TP':>3} {'FP':>3} {'TN':>3} {'FN':>3} | "
                  f"{'Prec%':>6} {'Rec%':>6} {'FPR%':>6} {'F1%':>6} {'DAcc%':>6}")
            print("-" * 100)
        else:
            print(f"{'Round':>5} | {'TestAcc':>8} | {'TestLoss':>9} | {'TrainLoss':>10}")
            print("-" * 50)

        for r in self.round_history:
            rnd = r['round']
            tacc = r.get('test_accuracy', 0.0) * 100
            tloss = r.get('test_loss', np.inf)
            train_loss = r.get('avg_loss', np.inf)

            if has_detection and r.get('detection_metrics'):
                dm = r['detection_metrics']
                print(f"{rnd:>5} | {tacc:>7.2f}% | {tloss:>9.4f} | {train_loss:>10.4f} | "
                      f"{dm['tp']:>3} {dm['fp']:>3} {dm['tn']:>3} {dm['fn']:>3} | "
                      f"{dm['precision']:>5.1f} {dm['recall']:>5.1f} "
                      f"{dm['fpr']:>5.1f} {dm['f1_score']:>5.1f} {dm['dacc']:>5.1f}")
            else:
                print(f"{rnd:>5} | {tacc:>7.2f}% | {tloss:>9.4f} | {train_loss:>10.4f}")

        print("=" * 100)

        # Averages
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
                avg_f1 = np.mean([d['f1_score'] for d in rounds_with_det])
                total_tp = sum(d['tp'] for d in rounds_with_det)
                total_fp = sum(d['fp'] for d in rounds_with_det)
                total_tn = sum(d['tn'] for d in rounds_with_det)
                total_fn = sum(d['fn'] for d in rounds_with_det)
                print(f"  Avg Detection: Prec={avg_prec:.1f}% Rec={avg_rec:.1f}% FPR={avg_fpr:.1f}% F1={avg_f1:.1f}%")
                print(f"  Total Counts:  TP={total_tp} FP={total_fp} TN={total_tn} FN={total_fn}")

        total_time = self.cumulative_times[-1] if self.cumulative_times else 0.0
        print(f"  Total Time: {total_time:.1f}s ({total_time/60:.1f}min)")
        print("=" * 100 + "\n")

    def get_timing_data(self):
        """
        Get timing data for plotting time vs accuracy/round graphs.

        Returns:
            Dictionary containing:
            - round_times: List of per-round times (seconds)
            - cumulative_times: List of cumulative times (seconds)
            - total_time: Total training time (seconds)
            - avg_round_time: Average time per round (seconds)
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

        Returns the LAST round's detection metrics for final reporting,
        plus averaged metrics across all rounds.
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

        # Average across all rounds
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
        """Get malicious-in-committee tracking from CMFL defense.

        Returns:
            dict with per-round history and aggregate statistics,
            or empty dict if defense doesn't support this tracking.
        """
        if (self.use_defense and self.defense is not None
                and hasattr(self.defense, 'get_malicious_in_committee_summary')):
            return self.defense.get_malicious_in_committee_summary()
        return {}

    def evaluate_on_test_set(self, test_loader):
        """
        Evaluate global model on test set — returns accuracy AND loss in a
        **single forward pass** (avoids the old double-evaluation bottleneck).

        After global model distribution all clients share the same parameters,
        so evaluating one client is sufficient.

        Args:
            test_loader: PyTorch DataLoader for test data

        Returns:
            Tuple of (accuracy, avg_loss)
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

        Args:
            test_X: Test features (numpy array)
            y_test: Test labels (numpy array)

        Returns:
            List of attack success rates for each client (0.0 for benign clients)
        """
        attack_success_rates = []

        for client in self.clients:
            if client.is_malicious and client.attack_instance is not None:
                try:
                    success_rate = client.attack_instance.evaluate_attack_success(
                        client.model, test_X, y_test
                    )

                    # Robust conversion to float
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

