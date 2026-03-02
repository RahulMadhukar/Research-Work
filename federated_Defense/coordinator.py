import torch
import numpy as np
from typing import List
from collections import defaultdict
from defense import AdaptiveCommitteeDefense, CMFLDefense


class DecentralizedFLCoordinator:
    """
    Coordinates decentralized federated learning among peers with committee defense.
    
    This coordinator:
    - Manages federated learning rounds
    - Applies committee defense to detect and exclude malicious clients
    - Tracks performance metrics and detection statistics
    - Provides evaluation capabilities for attack success rates
    """

    def __init__(self, clients, use_defense=True, num_peers=3, defense_type='adaptivecommittee',
                 clients_per_round=None, aggregation_method='fedavg', use_anomaly_detection=None,
                 lr=0.01, batch_size=None, epochs=1, **defense_kwargs):
        """
        Initialize the federated learning coordinator.

        Args:
            clients: List of DecentralizedClient instances
            use_defense: Whether to use committee-based architecture (default: True)
                        DEPRECATED: Use defense_type to control committee structure
            num_peers: Number of peers for decentralized communication (unused in current impl)
            defense_type: Type of defense ('adaptivecommittee', 'cmfl', 'none')
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
            if defense_type == 'cmfl':
                self.defense = CMFLDefense(
                    num_clients=len(clients),
                    aggregation_method=aggregation_method,
                    **defense_kwargs
                )
                self.committee_defense = None
            else:
                self.defense = AdaptiveCommitteeDefense(
                    num_clients=len(clients),
                    aggregation_method=aggregation_method,
                    **defense_kwargs
                )
                self.committee_defense = self.defense
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
            if self.defense and self.defense_type in ['cmfl', 'adaptivecommittee']:
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

                if self.defense and self.defense_type in ['cmfl', 'adaptivecommittee']:
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

                elif self.defense_type == 'adaptivecommittee':
                    client_losses_dict = {cid: client_losses[i] for i, cid in enumerate(active_client_ids)}
                    global_params, anomalous_clients, defense_metrics = self.defense.adaptive_committee_round(
                        client_id_to_update, client_losses_dict,
                        malicious_client_ids=malicious_ids_for_diagnostics,
                        all_clients=self.clients,
                        use_anomaly_detection=self.use_anomaly_detection,
                        client_data_sizes=client_data_sizes
                    )
                    self.enhanced_metrics['defense_stats'].append(defense_metrics)
                    self.enhanced_metrics['reputation_scores'].append(defense_metrics.get('reputation_scores', {}))
                    self.enhanced_metrics['adaptive_thresholds'].append(self.defense.threshold)

                elif self.defense_type == 'adaptive':
                    reputation_scores = getattr(self, 'reputation_scores', None)
                    anomalous_clients = self.defense.detect_anomalies(local_updates, client_losses, reputation_scores)
                    global_params = self.defense.robust_aggregate(local_updates, anomalous_clients, reputation_scores)
                    self.enhanced_metrics['adaptive_thresholds'].append(self.defense.threshold)

                else:
                    anomalous_clients = self.committee_defense.detect_anomalies(local_updates, client_losses=client_losses)
                    global_params = self.committee_defense.robust_aggregate(local_updates, anomalous_clients)

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

            # --- Track performance ---
            if valid_loss_clients:
                valid_indices = [i for i, cid in enumerate(active_client_ids) if cid in valid_loss_clients]
                valid_losses = [client_losses[i] for i in valid_indices if np.isfinite(client_losses[i])]
                avg_accuracy = 0.0  # local accuracy removed for speed
                avg_loss = np.mean(valid_losses) if valid_losses else np.inf
            else:
                avg_accuracy = 0.0
                avg_loss = np.inf

            self.global_accuracy_history.append(avg_accuracy)
            loss_history.append(avg_loss)
            accuracy_history.append(avg_accuracy)

            self.round_history.append({
                'round': current_round,
                'client_accuracies': client_accuracies.copy(),
                'avg_accuracy': avg_accuracy,
                'client_losses': client_losses.copy(),
                'avg_loss': avg_loss,
                'anomalous_clients': anomalous_clients.copy()
            })

            # --- Timing ---
            round_end_time = time.time()
            round_duration = round_end_time - round_start_time
            cumulative_time = round_end_time - self._training_start_time
            self.round_times.append(round_duration)
            self.cumulative_times.append(cumulative_time)

            # Print concise progress at milestones only
            if _milestone:
                model_name = type(self.clients[0].model).__name__
                print(f"  [R{current_round}/{display_total}] {round_duration:.1f}s/round "
                      f"(train={_t_train:.1f}s [{len(active_client_ids)} clients], "
                      f"agg={_t_agg:.1f}s, dist={_t_dist:.1f}s) | "
                      f"Total: {cumulative_time:.0f}s ({cumulative_time/60:.1f}min) | "
                      f"AvgLoss: {avg_loss:.4f} | Model: {model_name}")

        # --- Defense summary on truly final round ---
        current_final_round = round_offset + rounds
        is_truly_final = (total_rounds is None) or (current_final_round >= total_rounds)

        if self.use_defense and self.use_anomaly_detection and is_truly_final:
            self._print_defense_summary()

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

    def _print_defense_summary(self):
        """Print summary of defense effectiveness across all rounds."""
        # UPDATED: Use defense object's comprehensive final summary if available
        if hasattr(self.defense, 'print_final_detection_summary'):
            # Use the new comprehensive final detection metrics summary
            self.defense.print_final_detection_summary(self.clients)
        else:
            # Fallback to legacy summary for backward compatibility
            print("\n[DEFENSE SUMMARY] Overall effectiveness:")

            total_detections = 0
            correct_detections = 0
            false_positives = 0
            actual_malicious_ids = set([c.client_id for c in self.clients if c.is_malicious])

            for round_info in self.round_history:
                detected = set(round_info['anomalous_clients'])
                correct = detected & actual_malicious_ids
                false_pos = detected - actual_malicious_ids

                total_detections += len(detected)
                correct_detections += len(correct)
                false_positives += len(false_pos)

            # Precision
            if total_detections > 0:
                precision = correct_detections / total_detections * 100
                print(f"  Precision: {precision:.1f}% ({correct_detections}/{total_detections} detections correct)")
            else:
                print(f"  Precision: N/A (no detections made)")

            # Recall
            if len(actual_malicious_ids) > 0:
                all_detected_malicious = set()
                for round_info in self.round_history:
                    detected = set(round_info['anomalous_clients'])
                    all_detected_malicious.update(detected & actual_malicious_ids)

                recall = len(all_detected_malicious) / len(actual_malicious_ids) * 100
                print(f"  Recall: {recall:.1f}% ({len(all_detected_malicious)}/{len(actual_malicious_ids)} malicious clients detected)")
            else:
                print(f"  Recall: N/A (no malicious clients)")

            # Get full metrics including TP, TN, FP, FN
            full_metrics = self.get_committee_metrics()

            # Overall rates
            print(f"  Detection Rate: {self.committee_metrics['detection_rate']:.2%}")
            print(f"  False Positive Rate: {self.committee_metrics['false_positive_rate']:.2%}")
            print(f"  False Negative Rate: {full_metrics['false_negative_rate']:.2f}%")
            print(f"  Precision: {full_metrics['precision']:.2f}%")
            print(f"  Recall: {full_metrics['recall']:.2f}%")

            # Confusion Matrix Elements
            print(f"\n  Confusion Matrix:")
            print(f"    True Positives (TP):  {full_metrics['true_positives']}")
            print(f"    True Negatives (TN):  {full_metrics['true_negatives']}")
            print(f"    False Positives (FP): {full_metrics['false_positives']}")
            print(f"    False Negatives (FN): {full_metrics['false_negatives']}")

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
        Get committee defense performance metrics for evaluation.

        Uses MICRO-AVERAGED metrics (sum per-round TP/FP/TN/FN counts).
        This is the standard FL approach and avoids the inflation problem of
        cumulative "EVER detected" logic, where client rotation over many
        rounds causes nearly every client to be flagged at least once.

        Returns:
            Dictionary containing detection_rate, false_positive_rate, TP, TN, FP, FN, and raw counts
        """
        if self.use_defense and self.defense is not None:
            try:
                # Micro-averaged: sum per-round confusion matrix counts
                metrics = self.defense.get_micro_averaged_detection_metrics(self.clients)

                # Extract malicious IDs for backward compatibility
                malicious_ids = set([c.client_id for c in self.clients if c.is_malicious])

                return {
                    'detection_rate': metrics['recall'],
                    'false_positive_rate': metrics['fpr'],
                    'false_negative_rate': metrics['fnr'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'accuracy': metrics['dacc'],
                    'true_positives': metrics['tp_count'],
                    'true_negatives': metrics['tn_count'],
                    'false_positives': metrics['fp_count'],
                    'false_negatives': metrics['fn_count'],
                    'detected_malicious': metrics['tp_count'],
                    'actual_malicious': list(malicious_ids),
                    'total_clients': len(self.clients),
                    'num_malicious': metrics['total_malicious_seen'],
                    'num_benign': metrics['total_benign_seen'],
                    'total_unique_participants': metrics['total_unique_participants'],
                    'aggregation_method': metrics['aggregation_method']
                }
            except Exception as e:
                print(f"[WARN] Failed to get micro-averaged metrics from defense: {e}")
                pass

        # LEGACY FALLBACK: Only used if defense is not available or metrics call fails
        print("[WARN] Using legacy metrics calculation (not participant-based)")

        malicious_ids = set([c.client_id for c in self.clients if c.is_malicious])
        num_malicious = len(malicious_ids)
        num_benign = len(self.clients) - num_malicious

        total_detected_malicious = 0
        total_false_positives = 0

        for round_info in self.round_history:
            detected_set = set(round_info.get('anomalous_clients', []))
            total_detected_malicious += len(detected_set & malicious_ids)
            total_false_positives += len(detected_set - malicious_ids)

        true_positives = total_detected_malicious
        false_positives = total_false_positives
        false_negatives = num_malicious - true_positives
        true_negatives = num_benign - false_positives

        detection_rate = (true_positives / num_malicious * 100) if num_malicious > 0 else 0.0
        false_positive_rate = (false_positives / num_benign * 100) if num_benign > 0 else 0.0
        false_negative_rate = (false_negatives / num_malicious * 100) if num_malicious > 0 else 0.0
        precision = (true_positives / (true_positives + false_positives) * 100) if (true_positives + false_positives) > 0 else 0.0
        recall = detection_rate

        return {
            'detection_rate': detection_rate,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'precision': precision,
            'recall': recall,
            'true_positives': true_positives,
            'true_negatives': true_negatives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'detected_malicious': total_detected_malicious,
            'actual_malicious': list(malicious_ids),
            'total_clients': len(self.clients),
            'num_malicious': num_malicious,
            'num_benign': num_benign
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

