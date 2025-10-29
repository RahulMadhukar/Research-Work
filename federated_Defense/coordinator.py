import torch
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict
from defense import EnsembleDefense, AdaptiveCommitteeDefense, ReputationSystem, GradientAnalyzer


class DecentralizedFLCoordinator:
    """
    Coordinates decentralized federated learning among peers with committee defense.
    
    This coordinator:
    - Manages federated learning rounds
    - Applies committee defense to detect and exclude malicious clients
    - Tracks performance metrics and detection statistics
    - Provides evaluation capabilities for attack success rates
    """

    def __init__(self, clients, use_committee_defense=True, num_peers=3, defense_type='committee'):
        """
        Initialize the federated learning coordinator.

        Args:
            clients: List of DecentralizedClient instances
            use_committee_defense: Whether to use committee defense (default: True)
            num_peers: Number of peers for decentralized communication (unused in current impl)
            defense_type: Type of defense ('committee', 'ensemble', 'adaptive', 'none')
        """
        self.clients = clients
        self.use_committee_defense = use_committee_defense
        self.defense_type = defense_type

        # Initialize appropriate defense mechanism
        if use_committee_defense:
            if defense_type == 'ensemble':
                self.defense = EnsembleDefense(num_clients=len(clients))
                self.committee_defense = self.defense.adaptive_committee  # For compatibility
                print(f"[DEFENSE] Using Ensemble Defense (Committee + Reputation + Gradient Analysis)")
            elif defense_type == 'adaptive':
                self.defense = AdaptiveCommitteeDefense()
                self.committee_defense = self.defense  # For compatibility
                print(f"[DEFENSE] Using Adaptive Committee Defense")
            elif defense_type == 'reputation':
                self.defense = ReputationSystem(num_clients=len(clients))
                self.committee_defense = self.defense  # For compatibility
                print(f"[DEFENSE] Using Reputation System Defense")
            elif defense_type == 'gradient':
                self.defense = GradientAnalyzer()
                self.committee_defense = self.defense  # For compatibility
                print(f"[DEFENSE] Using Gradient Analyzer Defense")
            else:
                self.defense = AdaptiveCommitteeDefense()
                self.committee_defense = self.defense  # For compatibility
                print(f"[DEFENSE] Using Standard Committee Defense")
        else:
            self.defense = None
            self.committee_defense = None
            print(f"[DEFENSE] No defense enabled")

        self.num_peers = num_peers
        self.round_history = []
        self.global_accuracy_history = []
        self.attack_success_history = []
        self.poisoned_participation_log = defaultdict(list)

        # Track committee defense metrics
        self.committee_metrics = {
            'detected_malicious': [],
            'actual_malicious': [],
            'false_positives': [],
            'detection_rate': 0.0,
            'false_positive_rate': 0.0
        }

        # Track enhanced defense metrics
        self.enhanced_metrics = {
            'reputation_scores': [],
            'gradient_norms': [],
            'adaptive_thresholds': [],
            'defense_stats': []
        }

    def run_federated_learning(self, rounds=10, aggregation_method='mean',
                               test_loader=None, model_constructor=None):
        """
        Run decentralized FL for the given number of rounds.
        
        Args:
            rounds: Number of federated learning rounds
            aggregation_method: Aggregation method ('mean' or 'median') - only used if defense disabled
            test_loader: Optional test data loader for evaluation
            model_constructor: Optional model constructor for evaluation
            
        Returns:
            Tuple of (accuracy_history, loss_history, round_history)
        """
        print(f"\nStarting decentralized FL with {len(self.clients)} clients")
        print(f"Committee defense: {'Enabled ✓' if self.use_committee_defense else 'Disabled ✗'}")
        print(f"Aggregation method: {aggregation_method}")
        
        # Identify malicious clients
        malicious_count = sum(1 for c in self.clients if c.is_malicious)
        malicious_ids = [c.client_id for c in self.clients if c.is_malicious]
        print(f"Malicious clients: {malicious_count}/{len(self.clients)} (IDs: {malicious_ids})")
        
        # Store for metrics calculation
        self.committee_metrics['actual_malicious'] = malicious_ids

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        accuracy_history = []
        loss_history = []

        for round_num in range(rounds):
            print(f"\n{'='*60}")
            print(f"Round {round_num + 1}/{rounds}")
            print(f"{'='*60}")

            # Re-apply attacks each round (for malicious clients)
            if malicious_count > 0:
                print(f"[ROUND {round_num + 1}] Applying attacks to make malicious clients...")
                for client in self.clients:
                    if client.is_malicious:
                        try:
                            client.poison_local_data()
                            num_poisoned = np.sum(client.poison_mask) if hasattr(client, 'poison_mask') else 0
                            print(f"  ✓ Client {client.client_id}: {num_poisoned} samples poisoned")
                        except Exception as e:
                            print(f"  ✗ Client {client.client_id}: Poisoning failed - {e}")

            # Local training on all clients
            local_updates = []
            client_losses = []
            client_accuracies = []
            
            for client in self.clients:
                params, loss, acc = client.local_training()
                local_updates.append(params)
                client_losses.append(loss)
                client_accuracies.append(acc)

                # Log malicious participation
                if getattr(client, 'is_malicious', False):
                    self.poisoned_participation_log[client.client_id].append(round_num + 1)

            print(f"Local accuracies: {[f'{acc:.3f}' for acc in client_accuracies]}")
            print(f"Local losses: {[f'{loss:.3f}' for loss in client_losses]}")

            # Aggregate with or without defense
            anomalous_clients = []
            defense_metrics = {}

            if self.use_committee_defense:
                print(f"\n[DEFENSE] Running {self.defense_type.upper()} defense...")

                # Use appropriate defense strategy
                if self.defense_type == 'ensemble':
                    # Ensemble defense with all components
                    global_params, anomalous_clients, defense_metrics = self.defense.defend_and_aggregate(
                        local_updates,
                        client_losses
                    )

                    # Track enhanced metrics
                    self.enhanced_metrics['defense_stats'].append(defense_metrics)
                    self.enhanced_metrics['reputation_scores'].append(defense_metrics.get('reputation_scores', {}))
                    if 'gradient_analysis' in defense_metrics:
                        self.enhanced_metrics['gradient_norms'].append(defense_metrics['gradient_analysis'].get('grad_norms', []))
                    if 'adaptive_threshold' in defense_metrics:
                        self.enhanced_metrics['adaptive_thresholds'].append(defense_metrics['adaptive_threshold'])

                elif self.defense_type == 'adaptive':
                    # Adaptive committee defense
                    reputation_scores = getattr(self, 'reputation_scores', None)
                    anomalous_clients = self.defense.detect_anomalies(
                        local_updates,
                        client_losses,
                        reputation_scores
                    )
                    global_params = self.defense.robust_aggregate(
                        local_updates,
                        anomalous_clients,
                        reputation_scores
                    )
                    self.enhanced_metrics['adaptive_thresholds'].append(self.defense.threshold)

                else:
                    # Standard committee defense
                    anomalous_clients = self.committee_defense.detect_anomalies(
                        local_updates,
                        client_losses=client_losses
                    )
                    global_params = self.committee_defense.robust_aggregate(
                        local_updates,
                        anomalous_clients
                    )

                # Update committee metrics
                self._update_committee_metrics(malicious_ids, anomalous_clients)

                # Defense analysis
                if anomalous_clients:
                    malicious_flagged = [c for c in anomalous_clients if c in malicious_ids]
                    honest_flagged = [c for c in anomalous_clients if c not in malicious_ids]
                    malicious_missed = [c for c in malicious_ids if c not in anomalous_clients]
                    
                    print(f"[DEFENSE] Detected {len(anomalous_clients)} anomalous clients: {anomalous_clients}")
                    print(f"[DEFENSE] ✓ Correctly flagged malicious: {malicious_flagged} ({len(malicious_flagged)}/{len(malicious_ids)})")
                    print(f"[DEFENSE] ✗ False positives (honest flagged): {honest_flagged}")
                    print(f"[DEFENSE] ✗ Missed malicious: {malicious_missed}")
                    
                    if len(malicious_ids) > 0:
                        detection_rate = len(malicious_flagged) / len(malicious_ids) * 100
                        print(f"[DEFENSE] Detection rate: {detection_rate:.1f}%")
                else:
                    print(f"[DEFENSE] No anomalous clients detected")
            else:
                # No defense - simple aggregation
                global_params = self._aggregate_updates(local_updates, method=aggregation_method)
                print(f"[NO DEFENSE] Using simple {aggregation_method} aggregation (all clients)")

            # Distribute global model to all clients
            print(f"\n[UPDATE] Distributing global model to all clients...")
            for client in self.clients:
                with torch.no_grad():
                    for name, param in client.model.named_parameters():
                        if name in global_params:
                            param.data.copy_(global_params[name])

            # Track performance
            avg_accuracy = np.mean(client_accuracies)
            avg_loss = np.mean(client_losses)
            self.global_accuracy_history.append(avg_accuracy)
            loss_history.append(avg_loss)
            accuracy_history.append(avg_accuracy)

            # Store round information
            self.round_history.append({
                'round': round_num + 1,
                'client_accuracies': client_accuracies.copy(),
                'avg_accuracy': avg_accuracy,
                'client_losses': client_losses.copy(),
                'avg_loss': avg_loss,
                'anomalous_clients': anomalous_clients.copy()
            })
            
            print(f"\n[ROUND {round_num + 1} SUMMARY]")
            print(f"  Avg training accuracy: {avg_accuracy:.4f}")
            print(f"  Avg training loss: {avg_loss:.4f}")

        # Final summary
        print(f"\n{'='*60}")
        print("FEDERATED LEARNING COMPLETED")
        print(f"{'='*60}")
        
        # Print defense effectiveness summary
        if self.use_committee_defense:
            self._print_defense_summary()

        print(f"\nFinal Summary:")
        print(f"  Avg accuracy: {np.mean(accuracy_history):.4f}")
        print(f"  Final accuracy: {accuracy_history[-1]:.4f}")

        return self.global_accuracy_history, loss_history, self.round_history

    def _aggregate_updates(self, client_updates, method='mean'):
        """
        Simple aggregation without defense.
        
        Args:
            client_updates: List of client parameter updates
            method: Aggregation method ('mean' or 'median')
            
        Returns:
            Aggregated parameters
        """
        aggregated_params = {}
        param_names = client_updates[0].keys()
        
        for name in param_names:
            param_stack = torch.stack([update[name] for update in client_updates])
            if method == 'mean':
                aggregated_params[name] = torch.mean(param_stack, dim=0)
            elif method == 'median':
                aggregated_params[name] = torch.median(param_stack, dim=0).values
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
        
        # Overall rates
        print(f"  Detection Rate: {self.committee_metrics['detection_rate']:.2%}")
        print(f"  False Positive Rate: {self.committee_metrics['false_positive_rate']:.2%}")

    def get_committee_metrics(self):
        """
        Get committee defense performance metrics for evaluation.
        
        Returns:
            Dictionary containing detection_rate and false_positive_rate
        """
        return {
            'detection_rate': self.committee_metrics['detection_rate'],
            'false_positive_rate': self.committee_metrics['false_positive_rate'],
            'actual_malicious': self.committee_metrics['actual_malicious'],
            'total_clients': len(self.clients)
        }

    def evaluate_on_test_set(self, test_loader):
        """
        Evaluate all clients on test set.
        
        Args:
            test_loader: PyTorch DataLoader for test data
            
        Returns:
            Tuple of (average_accuracy, list_of_client_accuracies)
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        all_accuracies = []
        
        for client in self.clients:
            client.model.to(device)
            client.model.eval()
            correct, total = 0, 0
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = client.model(batch_X)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == batch_y).sum().item()
                    total += batch_y.size(0)
            
            accuracy = correct / total if total > 0 else 0.0
            all_accuracies.append(accuracy)
        
        return np.mean(all_accuracies), all_accuracies

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
                    
                    # Clamp to [0, 1]
                    success_rate = max(0.0, min(1.0, success_rate))
                    attack_success_rates.append(success_rate)
                    print(f"[ASR] Malicious client {client.client_id}: {success_rate:.4f}")
                    
                except Exception as e:
                    print(f"[ERROR] Client {client.client_id} ASR evaluation failed: {e}")
                    attack_success_rates.append(0.0)
            else:
                # Benign client or no attack instance
                attack_success_rates.append(0.0)
        
        return attack_success_rates

    def run_full_evaluation(self, test_loader, model_constructor=None, rounds=10):
        """
        Run complete evaluation: baseline, federated training, and attack assessment.
        
        Args:
            test_loader: PyTorch DataLoader for test data
            model_constructor: Function that returns a fresh model instance
            rounds: Number of FL rounds to run
            
        Returns:
            Dictionary containing evaluation results
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. Baseline evaluation (untrained model)
        print("\n[EVALUATION] Testing baseline (untrained) model...")
        baseline_model = model_constructor().to(device) if model_constructor else self.clients[0].model
        baseline_model.eval()
        correct, total = 0, 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = baseline_model(batch_X)
                preds = outputs.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
        
        baseline_acc = correct / total if total > 0 else 0.0
        print(f"[EVALUATION] Baseline accuracy: {baseline_acc:.4f}")

        # 2. Run federated learning
        print("\n[EVALUATION] Running federated learning...")
        acc_history, loss_history, round_history = self.run_federated_learning(
            rounds=rounds,
            test_loader=test_loader,
            model_constructor=model_constructor
        )

        # 3. Final evaluation on test set
        print("\n[EVALUATION] Final evaluation on test set...")
        final_acc, client_accuracies = self.evaluate_on_test_set(test_loader)
        print(f"[EVALUATION] Final test accuracy: {final_acc:.4f}")

        # 4. Evaluate attack success (if applicable)
        print("\n[EVALUATION] Evaluating attack success rates...")
        try:
            test_X = test_loader.dataset.data.numpy()
            y_test = test_loader.dataset.targets.numpy()
            asr_rates = self.evaluate_attack_success(test_X, y_test)
            avg_asr = np.mean([r for r in asr_rates if r > 0]) if any(r > 0 for r in asr_rates) else 0.0
            print(f"[EVALUATION] Average attack success rate: {avg_asr:.4f}")
        except Exception as e:
            print(f"[WARN] Could not evaluate attack success: {e}")
            asr_rates = [0.0] * len(self.clients)
            avg_asr = 0.0

        # Return results
        return {
            'baseline_accuracy': baseline_acc,
            'final_accuracy': final_acc,
            'accuracy_history': acc_history,
            'loss_history': loss_history,
            'client_accuracies': client_accuracies,
            'attack_success_rates': asr_rates,
            'avg_attack_success': avg_asr,
            'committee_metrics': self.get_committee_metrics() if self.use_committee_defense else None
        }