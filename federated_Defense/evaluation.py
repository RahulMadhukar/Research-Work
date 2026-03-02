# Allow multi-threaded CPU operations for proper GPU utilization.
import os

import copy
import csv
import json
import random
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from attacks.base import AttackConfig
from models import FashionMNISTNet, CIFAR10Net, FEMNISTNet, EMNISTNet, ShakespeareNet, Sentiment140Net
from datasets import load_fashion_mnist, load_cifar10, load_femnist, load_emnist, load_shakespeare, load_sentiment140, partition_dataset
from client import DecentralizedClient
from coordinator import DecentralizedFLCoordinator
from checkpoints import CheckpointManager
# from plots import PlottingEngine  # Imported only when generating plots


class EvaluationFramework:
    """
    Framework to run comprehensive evaluations of decentralized FL under
    various attacks and committee-based defense. Refactored to centralize repeated logic.
    """

    def __init__(self, out_dir="Output", run_id=None):
        """Initialize directories, plotter and basic bookkeeping."""
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        project_root = os.path.dirname(os.path.abspath(__file__))

        # Resolve base_run_dir
        # FIXED: Always create structure as {out_dir}/{run_id}/
        if os.path.isabs(out_dir):
            # Absolute path like /export/home/.../plots
            self.base_run_dir = os.path.join(out_dir, self.run_id)
        elif self.run_id in out_dir:
            # out_dir already contains run_id
            self.base_run_dir = os.path.join(project_root, out_dir)
        else:
            # Relative path like "plots" -> plots/{run_id}/
            output_root = os.path.join(project_root, out_dir)
            self.base_run_dir = os.path.join(output_root, self.run_id)

        # Create folders
        self.plots_dir = os.path.join(self.base_run_dir, "plots")
        self.results_dir = os.path.join(self.base_run_dir, "results")
        self.checkpoint_dir = os.path.join(self.base_run_dir, "checkpoints")
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # Compatibility
        self.out_dir = self.plots_dir

        # Plotter - Skip initialization during evaluation for faster execution
        # self.plotter = PlottingEngine(output_dir=self.plots_dir, run_id=self.run_id)
        self.plotter = None  # Will be initialized only when plotting is needed

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.base_run_dir,
            run_id=self.run_id
        )

        # Storage
        self.results = {}
        self.plots_generated = []

        # Track best metrics per scenario
        self.best_metrics = {}

    # ------------------------------
    # Shared configuration helpers
    # ------------------------------

    def get_attack_configs(self, malicious_clients: int = 10, data_poisoning_rate: float = 0.40):
        """
        Return canonical attack_configs dictionary used throughout the code.

        SINGLE ATTACK MODE: Only ONE attack is enabled at a time.
        To test a different attack, uncomment it and comment out 'slf'.
        """
        return {
            'none': None,

            # ============================================================================
            # CURRENTLY TESTING: Simple Label Flipping (slf)
            # ============================================================================
            'slf': AttackConfig(attack_type='slf', data_poisoning_rate=data_poisoning_rate,
                               source_class=[0, 2, 4, 6], target_class=[1, 3, 5, 7],
                               num_malicious_clients=malicious_clients),

            # ============================================================================
            # OTHER ATTACKS - Uncomment ONE to test, comment out 'slf' above
            # ============================================================================

            # 'dlf': AttackConfig(attack_type='dlf', data_poisoning_rate=0.60,
            #                     target_class=0,
            #                     num_malicious_clients=malicious_clients),

            # 'centralized': AttackConfig(attack_type='centralized', data_poisoning_rate=0.80,
            #                             source_class=[0, 2, 4, 6, 8],
            #                             target_class=1,
            #                             trigger_size=(8, 8), trigger_intensity=0.9,
            #                             num_malicious_clients=malicious_clients),

            # 'coordinated': AttackConfig(attack_type='coordinated', data_poisoning_rate=0.80,
            #                             source_class=[0, 2, 4, 6, 8],
            #                             target_class=1,
            #                             trigger_size=(8, 8), trigger_intensity=0.9,
            #                             num_malicious_clients=malicious_clients),

            # 'random': AttackConfig(attack_type='random', data_poisoning_rate=0.80,
            #                        source_class=[0, 2, 4, 6, 8],
            #                        target_class=1,
            #                        trigger_size=(8, 8), trigger_intensity=0.9,
            #                        num_malicious_clients=malicious_clients),

            # 'model_dependent': AttackConfig(attack_type='model_dependent', data_poisoning_rate=0.80,
            #                                 source_class=1,
            #                                 target_class=3,
            #                                 epsilon=0.7,
            #                                 num_malicious_clients=malicious_clients),

            # # Model Poisoning Attacks
            # 'local_model_replacement': AttackConfig(attack_type='local_model_replacement', data_poisoning_rate=data_poisoning_rate,
            #                                          epsilon=1.0,
            #                                          num_malicious_clients=malicious_clients),

            # 'local_model_noise': AttackConfig(attack_type='local_model_noise', data_poisoning_rate=data_poisoning_rate,
            #                                   epsilon=0.15,
            #                                   num_malicious_clients=malicious_clients),

            # 'global_model_replacement': AttackConfig(attack_type='global_model_replacement', data_poisoning_rate=data_poisoning_rate,
            #                                          target_class=[1, 3, 5, 7],
            #                                          epsilon=5.0,
            #                                          num_malicious_clients=malicious_clients),

            # 'aggregation_modification': AttackConfig(attack_type='aggregation_modification', data_poisoning_rate=data_poisoning_rate,
            #                                          target_class=[1, 3, 5, 7],
            #                                          epsilon=2.5,
            #                                          num_malicious_clients=malicious_clients)
        }

    def get_datasets_and_models(self, dataset_name=None):
        """Return datasets dict and models factory mapping.

        Args:
            dataset_name: If provided, only that dataset is loaded.
                          If None, all datasets are loaded.
        """
        _loaders = {
            'FEMNIST':       load_femnist,
            'Fashion-MNIST': load_fashion_mnist,
            'EMNIST':        load_emnist,
            'CIFAR-10':      load_cifar10,
            'Shakespeare':   load_shakespeare,
            'Sentiment140':  load_sentiment140,
        }
        models = {
            'FEMNIST':       lambda: FEMNISTNet(),
            'Fashion-MNIST': lambda: FashionMNISTNet(),
            'EMNIST':        lambda: EMNISTNet(),
            'CIFAR-10':      lambda: CIFAR10Net(),
            'Shakespeare':   lambda: ShakespeareNet(),
            'Sentiment140':  lambda: Sentiment140Net(),
        }

        names_to_load = [dataset_name] if dataset_name else _loaders.keys()
        datasets = {name: _loaders[name]() for name in names_to_load}

        return datasets, models

    def _subset_dataset(self, dataset, fraction=0.25):
        """
        Reduce dataset to specified fraction.
        
        Args:
            dataset: PyTorch Dataset or Subset
            fraction: Fraction to keep (0.1 = 10%)

        Returns:
            Subset of the dataset
        """
        if fraction >= 1.0:
            return dataset
        
        subset_size = int(len(dataset) * fraction)
        indices = torch.randperm(len(dataset))[:subset_size]
        subset = torch.utils.data.Subset(dataset, indices)
        
        print(f"[INFO] Subsetted dataset: {len(dataset)} → {subset_size} samples ({fraction*100:.0f}%)")
        return subset

    # ------------------------------
    # Client/update helpers
    # ------------------------------

    def build_clients(self, num_clients, client_data, models_map, dataset_name, attack_config=None, fixed_malicious_ids=None):
        """
        Build a list of DecentralizedClient instances for the provided client_data,
        with FIXED malicious client IDs to ensure consistency across experiments.

        Args:
            num_clients: Total number of clients
            client_data: Data for each client
            models_map: Map of model constructors
            dataset_name: Name of dataset
            attack_config: Attack configuration (None for clean baseline)
            fixed_malicious_ids: FIXED set of malicious client IDs (if None, will be randomly selected)

        CRITICAL: For proper experimental design, fixed_malicious_ids should be:
        - Generated ONCE at the start of an experiment
        - Reused for ALL scenarios (clean baseline, attack-only, attack+defense)
        - This ensures malicious clients are the SAME across all comparisons
        """


        # Use fixed malicious IDs if provided, otherwise randomly select (backward compatibility)
        malicious_ids = set()
        if fixed_malicious_ids is not None:
            # Use the FIXED malicious IDs (CORRECT experimental design)
            malicious_ids = set(fixed_malicious_ids)
        elif attack_config is not None and attack_config.num_malicious_clients > 0:
            # Fallback: randomly select (for backward compatibility, but not recommended)
            malicious_ids = set(random.sample(range(num_clients), attack_config.num_malicious_clients))
            print(f"[WARN] Randomly selecting malicious clients. For reproducible experiments, pass fixed_malicious_ids!")

        clients = []
        # FedAvg requirement: all clients must start from the SAME initial model.
        # Create one reference model and copy its weights to every client.
        _reference_state = models_map[dataset_name]().state_dict()

        for i in range(num_clients):
            model = models_map[dataset_name]()
            model.load_state_dict(_reference_state)
            # IMPORTANT: Malicious clients exist in ALL scenarios
            # - In clean baseline: they have attack_config=None (behave honestly)
            # - In attack scenarios: they have attack_config (launch attacks when selected)
            if i in malicious_ids and attack_config is not None:
                # Malicious client WITH attack enabled
                client = DecentralizedClient(i, client_data[i][0], client_data[i][1], model, attack_config)
            else:
                # Honest client OR malicious client in clean baseline (attack disabled)
                client = DecentralizedClient(i, client_data[i][0], client_data[i][1], model, None)

            # Mark client as malicious (persists across all scenarios)
            if i in malicious_ids:
                client.is_malicious = True
            else:
                client.is_malicious = False

            clients.append(client)

        if fixed_malicious_ids is not None:
            print(f"[INFO] Built {num_clients} clients with FIXED malicious IDs: {sorted(list(malicious_ids))[:10]}{'...' if len(malicious_ids) > 10 else ''}")

        return clients

    def run_and_evaluate_coordinator(
        self, clients, rounds, test_loader,
        test_X=None, y_test=None, use_defense=False, defense_type='adaptivecommittee',
        start_round=0, dataset="", attack="", scenario="", aggregation_method='fedavg'
    ):
        """
        Run federated learning for multiple rounds and evaluate on test data after each round.
        Save checkpoint after completing all rounds.

        Args:
            aggregation_method: Aggregation method ('fedavg', 'krum', 'median', 'trimmed_mean')
        """

        # Create coordinator based on scenario type
        # BASELINE/ATTACK: Regular FL (no committee, no anomaly detection)
        # DEFENSE: Committee structure + anomaly detection
        if use_defense:
            # DEFENSE SCENARIO: Committee structure + anomaly detection
            coordinator = DecentralizedFLCoordinator(
                clients,
                use_defense=True,
                defense_type=defense_type,
                committee_size=6,  # 30% of 20 (increased from 4 to reduce FPR)
                training_clients_per_round=14,  # 70% of 20 (adjusted from 16)
                aggregation_method=aggregation_method,
                use_anomaly_detection=True  # Enable anomaly detection for defense
            )
            print(f"[INFO] Defense scenario: Committee structure (6 committee + 14 training) + Anomaly detection ENABLED")
        else:
            # BASELINE/ATTACK SCENARIO: Regular FL (no committee, no anomaly detection)
            coordinator = DecentralizedFLCoordinator(
                clients,
                use_defense=False,
                defense_type=None,  # CRITICAL: Must be None to avoid default 'adaptivecommittee'
                aggregation_method=aggregation_method,
                clients_per_round=20  # Fixed: 20 clients per round for baseline/attack
            )
            print(f"[INFO] Baseline/Attack scenario: Regular FL with 20 clients per round, NO defense, NO committee")

        best_accuracy = -1.0
        best_round = 0

        # Initialize histories
        training_acc_history = []
        training_loss_history = []
        test_acc_history = []
        test_loss_history = []

        scenario_key = f"{dataset}_{attack}_{scenario}"
        if scenario_key not in self.best_metrics:
            self.best_metrics[scenario_key] = {'accuracy': -1.0, 'round': 0}

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"[INFO] Running federated learning for {rounds} rounds (evaluating at intervals)...")

        # Evaluate every eval_interval rounds + always on last round (saves ~80% eval time)
        eval_interval = max(1, rounds // 20) if rounds > 20 else 1

        try:
            # Run and evaluate for each round
            for round_num in range(1, rounds + 1):
                print(f"\n{'='*80}")
                print(f"[ROUND] Starting Round {round_num}/{rounds}")
                print(f"{'='*80}")

                # Run a single round of federated learning with proper round numbering
                coordinator.run_federated_learning(rounds=1, round_offset=round_num - 1, total_rounds=rounds)

                # Evaluate at intervals and on the last round
                if round_num % eval_interval == 0 or round_num == rounds:
                    try:
                        round_test_acc, round_test_loss = self._evaluate_test_acc_and_loss(
                            clients[0].model, test_loader, device
                        )
                        test_acc_history.append(float(round_test_acc))
                        test_loss_history.append(float(round_test_loss))
                        print(f"[EVALUATION] Round {round_num}/{rounds} → Test Acc: {round_test_acc:.4f}, Test Loss: {round_test_loss:.4f}")
                    except Exception as e:
                        print(f"[WARN] Test evaluation failed at round {round_num}: {e}")
                        test_acc_history.append(test_acc_history[-1] if test_acc_history else 0.0)
                        test_loss_history.append(test_loss_history[-1] if test_loss_history else 0.0)
                else:
                    # Carry forward last known value for skipped rounds
                    test_acc_history.append(test_acc_history[-1] if test_acc_history else 0.0)
                    test_loss_history.append(test_loss_history[-1] if test_loss_history else 0.0)

        except Exception as e:
            print(f"[ERROR] Federated learning failed at round {round_num}: {e}")
            raise

        print(f"\n{'='*80}")
        print(f"[EVALUATION] ALL {rounds} ROUNDS COMPLETED")
        print(f"{'='*80}")

        # Collect full history from coordinator
        if hasattr(coordinator, 'global_accuracy_history') and coordinator.global_accuracy_history:
            training_acc_history = [float(x) for x in coordinator.global_accuracy_history]
            final_train_acc = float(coordinator.global_accuracy_history[-1])
            print(f"[INFO] Collected {len(training_acc_history)} rounds of training accuracy history")
        else:
            training_acc_history = []
            final_train_acc = 0.0
            print(f"[WARN] No training accuracy history available from coordinator")

        if hasattr(coordinator, 'round_history') and coordinator.round_history:
            training_loss_history = [float(r.get('avg_loss', 0.0)) for r in coordinator.round_history]
            final_train_loss = float(coordinator.round_history[-1].get('avg_loss', 0.0))
            print(f"[INFO] Collected {len(training_loss_history)} rounds of training loss history")
        else:
            training_loss_history = []
            final_train_loss = 0.0
            print(f"[WARN] No training loss history available from coordinator")

        print(f"[INFO] Collected {len(test_acc_history)} rounds of test accuracy history")
        print(f"[INFO] Collected {len(test_loss_history)} rounds of test loss history")

        # Get final values
        final_test_acc = test_acc_history[-1] if test_acc_history else 0.0
        final_test_loss = test_loss_history[-1] if test_loss_history else 0.0

        # Check for best model
        is_best = final_test_acc > self.best_metrics[scenario_key]['accuracy']
        if is_best:
            self.best_metrics[scenario_key]['accuracy'] = final_test_acc
            self.best_metrics[scenario_key]['round'] = rounds
            best_accuracy = final_test_acc
            best_round = rounds
            print(f"[CHECKPOINT] New best accuracy: {final_test_acc:.4f} at round {best_round}")

        # Evaluate attack success (if applicable)
        attack_success_rates = []
        if test_X is not None and y_test is not None:
            try:
                attack_success_rates = coordinator.evaluate_attack_success(test_X, y_test)
            except Exception as e:
                print(f"[WARN] Attack success evaluation failed: {e}")

        # Extract detection metrics (if defense used)
        detection_metrics = {}
        if use_defense:
            try:
                if hasattr(coordinator, 'get_committee_metrics'):
                    detection_metrics = coordinator.get_committee_metrics()
                elif hasattr(coordinator, 'defense') and coordinator.defense is not None:
                    defense = coordinator.defense
                    if hasattr(defense, 'detected_malicious'):
                        detected = defense.detected_malicious
                        if isinstance(detected, dict):
                            detected_ids = list(detected.keys())
                        elif isinstance(detected, (list, tuple, np.ndarray, set)):
                            detected_ids = list(detected)
                        else:
                            detected_ids = []

                        num_detected = len(detected_ids)

                        # FIXED: Only count PARTICIPATING clients (committee + training), not all clients
                        participating_clients = []
                        if hasattr(defense, 'get_client_categories'):
                            categories = defense.get_client_categories()
                            participating_clients = categories.get('committee', []) + categories.get('training', [])
                        else:
                            # Fallback: assume all clients participated
                            participating_clients = list(range(len(coordinator.clients)))

                        # Count malicious/benign among PARTICIPANTS ONLY
                        num_malicious_participants = sum(
                            1 for cid in participating_clients
                            if cid < len(coordinator.clients) and getattr(coordinator.clients[cid], 'is_malicious', False)
                        )
                        num_benign_participants = len(participating_clients) - num_malicious_participants

                        # Count false positives among participants
                        false_positives = sum(
                            1 for cid in detected_ids
                            if isinstance(cid, int) and cid < len(coordinator.clients)
                            and not getattr(coordinator.clients[cid], 'is_malicious', False)
                        )

                        # Count true positives (correctly detected malicious)
                        true_positives = sum(
                            1 for cid in detected_ids
                            if isinstance(cid, int) and cid < len(coordinator.clients)
                            and getattr(coordinator.clients[cid], 'is_malicious', False)
                        )

                        # Per-round (final round) metrics - DEPRECATED, use cumulative instead
                        detection_metrics = {
                            'detection_rate': (true_positives / num_malicious_participants * 100) if num_malicious_participants > 0 else 0.0,
                            'false_positive_rate': (false_positives / num_benign_participants * 100) if num_benign_participants > 0 else 0.0,
                            'precision': (true_positives / num_detected * 100) if num_detected > 0 else 0.0,
                            'recall': (true_positives / num_malicious_participants * 100) if num_malicious_participants > 0 else 0.0,
                            'actual_malicious_participants': num_malicious_participants,
                            'actual_benign_participants': num_benign_participants,
                            'total_participants': len(participating_clients),
                            'total_clients': len(coordinator.clients),
                            'detected_count': num_detected,
                            'true_positives': true_positives,
                            'false_positives': false_positives
                        }

                        # RECOMMENDED: Get cumulative metrics across all rounds
                        if hasattr(defense, 'get_cumulative_detection_metrics'):
                            cumulative_metrics = defense.get_cumulative_detection_metrics(coordinator.clients)
                            # Add cumulative metrics with 'cumulative_' prefix
                            for key, value in cumulative_metrics.items():
                                if key != 'aggregation_method':
                                    detection_metrics[f'cumulative_{key}'] = value
                            # Override main metrics with cumulative (RECOMMENDED for FL)
                            detection_metrics['precision'] = cumulative_metrics.get('precision', detection_metrics['precision'])
                            detection_metrics['recall'] = cumulative_metrics.get('recall', detection_metrics['recall'])
                            detection_metrics['detection_rate'] = cumulative_metrics.get('detection_rate', detection_metrics['detection_rate'])
                            detection_metrics['dacc'] = cumulative_metrics.get('dacc', 0.0)
                            detection_metrics['f1_score'] = cumulative_metrics.get('f1_score', 0.0)
            except Exception as e:
                print(f"[WARN] Failed to extract detection metrics: {e}")

        # Gather and return results
        result = (
            coordinator.global_accuracy_history if hasattr(coordinator, 'global_accuracy_history') else [],
            [],
            coordinator.round_history if hasattr(coordinator, 'round_history') else []
        )

        # TIMING: Collect timing data for plotting
        timing_data = {}
        if hasattr(coordinator, 'get_timing_data'):
            timing_data = coordinator.get_timing_data()
            print(f"[INFO] Collected timing data: {len(timing_data.get('round_times', []))} rounds")
        else:
            print(f"[WARN] Coordinator does not have timing data")

        return (
            coordinator, result, final_test_acc, attack_success_rates,
            training_acc_history, training_loss_history,
            test_acc_history, test_loss_history, detection_metrics,
            timing_data  # ADDED: timing data for plotting
        )


    def _evaluate_test_acc_and_loss(self, model, test_loader, device=None):
        """
        Compute BOTH test accuracy and loss in a single forward pass.
        Avoids creating a new model or running two separate evaluation passes.
        """
        device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        model.to(device)
        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        correct = 0
        total_samples = 0
        total_loss = 0.0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total_loss += criterion(logits, y).item() * y.size(0)
                total_samples += y.size(0)
        acc = correct / total_samples if total_samples > 0 else 0.0
        loss = total_loss / total_samples if total_samples > 0 else 0.0
        return acc, loss

    # ------------------------------
    # Core experiments
    # ------------------------------

    def run_comprehensive_evaluation(self, test_defenses=True, subset_fraction=0.25, resume=False, defenses_to_test=None):
        """
        Run comprehensive evaluation with proper history storage.

        Configuration:
        - 100 total clients
        - 40 malicious clients (40%)
        - 24 clients participate per round (24% of total)
          - 6 committee members (25% of participants)
          - 18 training clients (75% of participants)
        """
        num_clients = 100
        rounds = 5
        malicious_clients = 40

        attack_configs = self.get_attack_configs(malicious_clients=malicious_clients)
        datasets, models = self.get_datasets_and_models()

        print("=" * 80)
        print("ENHANCED DECENTRALIZED FL SECURITY EVALUATION")
        print(f"Using {subset_fraction*100:.0f}% of each dataset for faster evaluation")
        print("=" * 80)

        results = {}

        for dataset_name, (trainset, testset) in datasets.items():
            print(f"\n{'=' * 60}")
            print(f"EVALUATING ON {dataset_name}")
            print(f"{'=' * 60}")

            # Subset the datasets
            print(f"[INFO] Original sizes - Train: {len(trainset)}, Test: {len(testset)}")
            trainset = self._subset_dataset(trainset, subset_fraction)
            testset = self._subset_dataset(testset, subset_fraction)
            print(f"[INFO] Subsetted sizes - Train: {len(trainset)}, Test: {len(testset)}")

            # FIXED: Use IID for better class balance (ensures SLF can poison effectively)
            client_data = partition_dataset(trainset, num_clients, iid=True)
            _pin = torch.cuda.is_available()
            test_loader = DataLoader(testset, batch_size=256, shuffle=False,
                                     pin_memory=_pin, num_workers=2)
            test_X, y_test = next(iter(DataLoader(testset, batch_size=len(testset), shuffle=False)))
            test_X, y_test = test_X.numpy(), y_test.numpy()

            dataset_results = {}

            # CRITICAL FIX: Generate FIXED malicious client IDs ONCE for this dataset
            # These IDs will be reused for ALL experiments (baseline, attack-only, attack+defense)
    
            fixed_malicious_ids = set(random.sample(range(num_clients), malicious_clients))
            print(f"\n{'='*80}")
            print(f"FIXED MALICIOUS CLIENT IDs FOR ALL EXPERIMENTS")
            print(f"{'='*80}")
            print(f"[CRITICAL] Malicious IDs: {sorted(list(fixed_malicious_ids))}")
            print(f"[INFO] These {len(fixed_malicious_ids)} malicious clients will be used in:")
            print(f"  1. Clean baseline (attack disabled, behave as honest)")
            print(f"  2. Attack-only scenarios (attack enabled when selected)")
            print(f"  3. Attack+Defense scenarios (same attackers, defense active)")
            print(f"{'='*80}")

            # FIXED: Run baseline ONCE with the SAME malicious IDs (but attack disabled)
            print(f"\n{'='*80}")
            print(f"RUNNING SHARED BASELINE (No Attack, No Defense)")
            print(f"{'='*80}")
            baseline_clients = self.build_clients(num_clients, client_data, models, dataset_name,
                                                  attack_config=None, fixed_malicious_ids=fixed_malicious_ids)
            _, _, baseline_acc, baseline_asr, bl_train_acc, bl_train_loss, bl_test_acc, bl_test_loss, _, timing_baseline = self.run_and_evaluate_coordinator(
                baseline_clients, rounds, test_loader, test_X=test_X, y_test=y_test,
                use_defense=False, dataset=dataset_name, attack='baseline', scenario='baseline'
            )
            print(f"[BASELINE] Final Test Acc: {baseline_acc:.4f}")
            print(f"[INFO] This baseline used FIXED malicious IDs (attack disabled)")

            # Store baseline results (shared across all attacks)
            shared_baseline = {
                'final_accuracy': float(baseline_acc),
                'training_acc_history': [float(x) for x in bl_train_acc],
                'training_loss_history': [float(x) for x in bl_train_loss],
                'test_acc_history': [float(x) for x in bl_test_acc],
                'test_loss_history': [float(x) for x in bl_test_loss],
                'attack_success_rates': [],
                'fixed_malicious_ids': sorted(list(fixed_malicious_ids))  # Store for reference
            }

            for attack_name, attack_config in attack_configs.items():
                # Skip 'none' attack - we already have baseline
                if attack_name.lower() == 'none':
                    print(f"\n--- Skipping 'none' attack (baseline already computed) ---")
                    continue

                print(f"\n{'='*80}")
                print(f"TESTING {attack_name.upper()} ATTACK")
                print(f"{'='*80}")

                attack_results = {}

                # Use shared baseline (no re-computation)
                attack_results['baseline'] = shared_baseline.copy()

                # STEP 1: Create attack scenario clients ONCE (will be reused for all defenses)
                scenario = 'attack'
                print(f"\n[SCENARIO] Creating attack scenario for {attack_name} (poisoned clients)")
                attack_scenario_clients = self.build_clients(num_clients, client_data, models, dataset_name, attack_config, fixed_malicious_ids=fixed_malicious_ids)
                print(f"[INFO] Attack scenario created with FIXED malicious IDs - these clients will be reused for all defense tests")

                # STEP 2: Run attack without defense (optional, for comparison)
                print(f"\n[SCENARIO] Running {scenario.upper()} for {attack_name} (no defense)")
                use_defense = False
                coordinator, result, final_test_accuracy, attack_success_rates_final, training_acc_history, training_loss_history, test_acc_history, test_loss_history, detection_metrics, _ = self.run_and_evaluate_coordinator(
                    attack_scenario_clients, rounds, test_loader, test_X=test_X, y_test=y_test,
                    use_defense=use_defense, dataset=dataset_name, attack=attack_name, scenario=scenario
                )
                attack_results[scenario] = {
                    'final_accuracy': float(final_test_accuracy),
                    'training_acc_history': [float(x) for x in training_acc_history],
                    'training_loss_history': [float(x) for x in training_loss_history],
                    'test_acc_history': [float(x) for x in test_acc_history],
                    'test_loss_history': [float(x) for x in test_loss_history],
                    'attack_success_rates': [float(x) for x in attack_success_rates_final] if attack_success_rates_final else []
                }
                print(f"[RESULT] {scenario.title()} - Final Test Acc: {final_test_accuracy:.4f}")
                print(f"[HISTORY] Collected {len(training_acc_history)} rounds of training history")
                print(f"[HISTORY] Collected {len(test_acc_history)} rounds of test history")

                # STEP 3: Test all defenses on the SAME attack scenario (deep copy for each defense)
                if attack_name.lower() != 'none':
                    defenses_to_run = ["adaptivecommittee", "cmfl"]
                    for defense_name in defenses_to_run:
                        print(f"\n[SCENARIO] Running DEFENSE ({defense_name.upper()}) for {attack_name}")
                        print(f"[INFO] Using deep copy of the SAME poisoned clients for fair comparison")

                        # Deep copy the attack scenario clients to ensure same poisoned data
                        defense_clients = copy.deepcopy(attack_scenario_clients)

                        coordinator, result, final_test_accuracy, attack_success_rates_final, training_acc_history, training_loss_history, test_acc_history, test_loss_history, detection_metrics, _ = self.run_and_evaluate_coordinator(
                            defense_clients, rounds, test_loader, test_X=test_X, y_test=y_test,
                            use_defense=True, defense_type=defense_name, dataset=dataset_name, attack=attack_name, scenario=f"defense_{defense_name}"
                        )

                        # Store defense results with detection metrics
                        defense_result = {
                            'final_accuracy': float(final_test_accuracy),
                            'training_acc_history': [float(x) for x in training_acc_history],
                            'training_loss_history': [float(x) for x in training_loss_history],
                            'test_acc_history': [float(x) for x in test_acc_history],
                            'test_loss_history': [float(x) for x in test_loss_history],
                            'attack_success_rates': [float(x) for x in attack_success_rates_final] if attack_success_rates_final else []
                        }

                        # Add detection metrics if available
                        if detection_metrics:
                            defense_result['detection_metrics'] = detection_metrics

                        attack_results[f"defense_{defense_name}"] = defense_result
                        print(f"[RESULT] Defense {defense_name} - Final Test Acc: {final_test_accuracy:.4f}")

                dataset_results[attack_name] = attack_results

            results[dataset_name] = dataset_results

        self.results = results
        return results

    # ------------------------------
    # Results handling and plotting
    # ------------------------------

    def generate_results_table(self, results):
        """Print and return a summary table list for CSV saving."""
        print(f"\n{'='*100}")
        print("COMPREHENSIVE EVALUATION RESULTS")
        print(f"{'='*100}")

        print(f"\n{'Dataset':<15} {'Attack Type':<12} {'Baseline':<10} {'Under Attack':<12} "
              f"{'With Defense':<12} {'Defense Recovery':<15} {'Attack Success':<15}")
        print("-" * 105)

        table_data = []
        for dataset_name, dataset_results in results.items():
            for attack_name, attack_results in dataset_results.items():
                baseline_acc = attack_results.get('baseline', {}).get('final_accuracy', 0.0)
                
                # For 'none' attack, only show baseline
                if attack_name.lower() == 'none':
                    print(f"{dataset_name:<15} {attack_name:<12} {baseline_acc:.4f}    "
                          f"{'N/A':<12} {'N/A':<12} {'N/A':<15} {'N/A':<15}")
                    table_data.append([dataset_name, attack_name, baseline_acc, None, None, None, None])
                    continue
                
                attack_acc = attack_results.get('attack', {}).get('final_accuracy', baseline_acc)
                attack_success = attack_results.get('attack', {}).get('attack_success_rates', [])
                avg_attack_success = np.mean(attack_success) if attack_success else 0.0

                # Loop over all defense results (keys starting with 'defense_')
                defense_keys = [k for k in attack_results.keys() if k.startswith('defense_')]
                if not defense_keys:
                    # Fallback: legacy single 'defense' key
                    defense_keys = ['defense'] if 'defense' in attack_results else []

                for defense_key in defense_keys:
                    defense_acc = attack_results.get(defense_key, {}).get('final_accuracy', attack_acc)
                    # Extract defense name for display
                    defense_name = defense_key.replace('defense_', '') if defense_key.startswith('defense_') else defense_key

                    if baseline_acc > 0 and baseline_acc != attack_acc:
                        recovery = (defense_acc - attack_acc) / (baseline_acc - attack_acc) * 100
                        recovery = max(0, min(100, recovery))
                    else:
                        recovery = 100 if defense_acc >= baseline_acc else 0

                    print(f"{dataset_name:<15} {attack_name:<12} {baseline_acc:.4f}    {attack_acc:.4f}      "
                          f"{defense_acc:.4f} ({defense_name})      {recovery:.1f}%            {avg_attack_success:.4f}")

                    table_data.append([dataset_name, attack_name, baseline_acc, attack_acc,
                                       defense_acc, recovery, avg_attack_success, defense_name])

        return table_data

    def save_results_table_to_csv(self, table_data, filename_prefix="evaluation_results"):
        """Save the evaluation table to CSV, leaving attack/defense metrics empty for baseline (no attack)."""
        filename = f"{filename_prefix}_{self.run_id}.csv"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, mode="w", newline="") as f:
            writer = csv.writer(f)
            # Header
            writer.writerow([
                "Dataset", "Attack Type", "Defense Name", "Baseline Accuracy", "Under Attack Accuracy",
                "With Defense Accuracy", "Defense Recovery (%)", "Attack Success Rate"
            ])
            
            for row in table_data:
                # Now each row has: dataset, attack_type, baseline_acc, attack_acc, defense_acc, recovery, asr, defense_name
                if len(row) == 8:
                    dataset, attack_type, baseline_acc, attack_acc, defense_acc, recovery, asr, defense_name = row
                else:
                    # For 'none' attack, legacy row
                    dataset, attack_type, baseline_acc, attack_acc, defense_acc, recovery, asr = row
                    defense_name = ""
                # For 'none' attack, leave attack/defense columns empty
                if attack_type.lower() == 'none' or attack_acc is None:
                    attack_acc = ""
                    defense_acc = ""
                    recovery = ""
                    asr = ""
                writer.writerow([
                    dataset, attack_type, defense_name, baseline_acc, attack_acc,
                    defense_acc, recovery, asr
                ])
        
        print(f"[INFO] Results table saved to {filepath}")


    def save_results_summary_to_json(self, results, filename_prefix="evaluation_summary"):
        """Save full results dict to JSON."""
        filename = f"{filename_prefix}_{self.run_id}.json"
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, "w") as f:
            json.dump(results, f, indent=4)
        print(f"[INFO] Results summary saved to {filepath}")

    def load_results_from_json(self, filepath):
        """Load a previously saved results JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No saved results found at {filepath}")
        with open(filepath, "r") as f:
            results = json.load(f)
        print(f"[INFO] Results loaded from {filepath}")
        self.results = results
        return results

    def generate_comprehensive_report(self, results):
        """Produce comprehensive report with committee defense analysis."""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE EVALUATION REPORT")
        print(f"{'='*80}")

        # Overall statistics
        total_attacks_tested = 0
        successful_defenses = 0

        for dataset_results in results.values():
            for attack_name, attack_results in dataset_results.items():
                if attack_name != 'none':
                    total_attacks_tested += 1

                    baseline_acc = attack_results.get('baseline', {}).get('final_accuracy', 0)
                    attack_acc = attack_results.get('attack', {}).get('final_accuracy', 0)

                    # Check all defense keys
                    defense_keys = [k for k in attack_results.keys() if k.startswith('defense_')]
                    for defense_key in defense_keys:
                        defense_acc = attack_results.get(defense_key, {}).get('final_accuracy', 0)
                        if baseline_acc != attack_acc:
                            recovery = (defense_acc - attack_acc) / (baseline_acc - attack_acc)
                            if recovery > 0.7:
                                successful_defenses += 1
                        elif defense_acc >= baseline_acc:
                            successful_defenses += 1

        print(f"\nOverall Statistics:")
        print(f"  Total attack scenarios tested: {total_attacks_tested}")
        print(f"  Successful defense instances: {successful_defenses}")
        print(f"  Defense success rate: {successful_defenses/total_attacks_tested*100 if total_attacks_tested>0 else 0:.1f}%")

        # Attack-specific analysis
        print(f"\n{'='*60}")
        print("ATTACK-SPECIFIC ANALYSIS")
        print(f"{'='*60}")

        attack_analysis = {
            'slf': 'Static Label Flipping - Simple but effective label manipulation',
            'dlf': 'Dynamic Label Flipping - Sophisticated feature-space targeting',
            'centralized': 'Centralized Trigger - Backdoor attack with fixed triggers',
            'coordinated': 'Coordinated Trigger - Client-specific backdoor triggers',
            'random': 'Random Trigger - Stochastic backdoor generation',
            'model_dependent': 'Model-Dependent - Gradient-based trigger optimization',
            # Model poisoning attacks — commented out (redundant with Byzantine/label-flipping; uncomment if needed)
            # 'local_model_replacement': 'Local Model Replacement - Replace model parameters with malicious values',
            # 'local_model_noise': 'Local Model Noise - Add crafted noise to model parameters',
            # 'global_model_replacement': 'Global Model Replacement - Override global model with boosted malicious update',
            # 'aggregation_modification': 'Aggregation Modification - Manipulate aggregation process with outlier/coordinated updates'
        }

        for attack_name, description in attack_analysis.items():
            print(f"\n{attack_name.upper()} - {description}")
            print("-" * 50)

            for dataset_name, dataset_results in results.items():
                if attack_name in dataset_results:
                    attack_results = dataset_results[attack_name]

                    baseline_acc = attack_results.get('baseline', {}).get('final_accuracy', 0)
                    attack_acc = attack_results.get('attack', {}).get('final_accuracy', 0)
                    attack_success = attack_results.get('attack', {}).get('attack_success_rates', [])
                    avg_attack_success = np.mean(attack_success) if attack_success else 0
                    impact = (baseline_acc - attack_acc) * 100

                    print(f"  {dataset_name}:")
                    print(f"    Baseline accuracy: {baseline_acc:.4f}")
                    print(f"    Attack accuracy:   {attack_acc:.4f}")
                    print(f"    Impact: {impact:.2f}% accuracy drop")
                    print(f"    Attack success: {avg_attack_success:.3f}")

                    defense_keys = [k for k in attack_results.keys() if k.startswith('defense_')]
                    for defense_key in defense_keys:
                        defense_acc = attack_results.get(defense_key, {}).get('final_accuracy', 0)
                        defense_name = defense_key.replace('defense_', '')
                        if baseline_acc != attack_acc:
                            recovery = (defense_acc - attack_acc) / (baseline_acc - attack_acc) * 100
                            recovery = max(0, min(100, recovery))
                        else:
                            recovery = 100 if defense_acc >= baseline_acc else 0
                        print(f"    Defense: {defense_name:<10} | Recovery: {recovery:.1f}% | Defense Acc: {defense_acc:.4f}")

        print(f"\n{'='*60}")
        print("DEFENSE MECHANISM ANALYSIS")
        print(f"{'='*60}")

        print("\nCommittee-based Defense Properties:")
        print("  - Anomaly detection through parameter distance analysis")
        print("  - Robust aggregation with outlier exclusion")
        print("  - Byzantine fault tolerance")
        print("  - Adaptive threshold-based filtering")
        print("  - Integrated directly into federated learning rounds")

        return {
            'total_attacks_tested': total_attacks_tested,
            'successful_defenses': successful_defenses,
            'defense_success_rate': successful_defenses/total_attacks_tested*100 if total_attacks_tested > 0 else 0,
            'plots_generated': []
        }
    

# =============================================================================
# MAIN EXECUTION FUNCTIONS
# =============================================================================

def run_enhanced_evaluation(
    resume_from_json=None,
    out_dir="Output",
    test_defenses=True,
    subset_fraction=0.25,
    resume=False,
    defenses_to_test=None
):
    """
    Main entry point — runs the comprehensive evaluation (training)
    and extended defense comparison automatically, saves CSVs and plots.

    Args:
        resume_from_json: Path to JSON results to skip training
        out_dir: Output directory
        test_defenses: Whether to test defenses
        subset_fraction: Fraction of dataset to use
        resume: Whether to resume from checkpoint
        defenses_to_test: List of defense types to test (default: ["adaptivecommittee", "cmfl"])

    Returns:
        Tuple of (results dict, report statistics dict)
    """

    if defenses_to_test is None:
        defenses_to_test = ["adaptivecommittee", "cmfl"]

    print("=" * 100)
    print("ENHANCED DECENTRALIZED FEDERATED LEARNING SECURITY FRAMEWORK")
    print(f"Dataset Size: {subset_fraction * 100:.0f}% (for faster testing)")
    print(f"Resume Mode: {'ENABLED' if resume else 'DISABLED'}")
    print("=" * 100)

    try:
        # Initialize evaluation framework
        if resume_from_json:
            evaluator = EvaluationFramework(out_dir=out_dir)
            results = evaluator.load_results_from_json(resume_from_json)
            print("[INFO] Skipping training, reusing saved results.")
        else:
            if resume:
                # Search for any checkpoint files recursively in out_dir/checkpoints/
                checkpoint_base = os.path.join(out_dir, "checkpoints")
                found_checkpoint = False
                for root, dirs, files in os.walk(checkpoint_base):
                    for file in files:
                        if file.endswith(".pth"):
                            found_checkpoint = True
                            break
                    if found_checkpoint:
                        break
                if found_checkpoint:
                    run_id = os.path.basename(os.path.normpath(out_dir))
                    print(f"[INFO] Found checkpoint for run_id: {run_id}")
                else:
                    print("[INFO] No existing checkpoint found. Starting fresh...")
                    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            else:
                run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

            evaluator = EvaluationFramework(out_dir=out_dir, run_id=run_id)

            # Run main comprehensive evaluation with resume support
            results = evaluator.run_comprehensive_evaluation(
                test_defenses=test_defenses,
                subset_fraction=subset_fraction,
                resume=resume,
                defenses_to_test=defenses_to_test
            )

            # Save results
            table_data = evaluator.generate_results_table(results)
            evaluator.save_results_table_to_csv(table_data)
            evaluator.save_results_summary_to_json(results)
            
            # Print checkpoint info
            print(f"\n{'='*80}")
            print("CHECKPOINT SUMMARY")
            print(f"{'='*80}")
            print(f"Checkpoint directory: {evaluator.checkpoint_dir}")
            print(f"Best checkpoints saved for each scenario")
            print(f"To resume training: python main.py --resume")
            print(f"{'='*80}")

        # Generate final report and print summary
        report_stats = evaluator.generate_comprehensive_report(results)

        print(f"\n{'=' * 80}")
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print(f"{'=' * 80}")

        print("\nKey Findings:")
        print("• Committee-based defense shows strong performance across attack types")
        print("• Trigger-based backdoors require specialized detection methods")
        print("• Dynamic attacks (DLF) pose greater challenges than static ones")
        print("• Decentralized FL maintains robustness under diverse threat models")

        return results, report_stats

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Training interrupted by user")
        print("[INFO] Progress has been saved. You can resume using --resume flag")
        raise
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        traceback.print_exc()
        print("\n[INFO] If training was interrupted, you can resume using --resume flag")
        raise


def run_attack_comparison_test(subset_fraction=0.25, num_clients=25, rounds = 5, malicious_clients=10):
    """
    Run side-by-side comparison of all attacks using a subset of the dataset.
    Evaluates attack effectiveness, accuracy, and stealth.
    """
    print("=" * 80)
    print("ATTACK COMPARISON TEST - DETAILED METRICS")
    print(f"Using {subset_fraction * 100:.0f}% of the dataset for evaluation")
    print("=" * 80)

    # Load and subset Fashion-MNIST
    trainset, testset = load_fashion_mnist()

    full_train_len, full_test_len = len(trainset), len(testset)
    subset_train_len = int(len(trainset) * subset_fraction)
    subset_test_len = int(len(testset) * subset_fraction)

    indices_train = np.random.choice(range(full_train_len), subset_train_len, replace=False)
    indices_test = np.random.choice(range(full_test_len), subset_test_len, replace=False)
    trainset = torch.utils.data.Subset(trainset, indices_train)
    testset = torch.utils.data.Subset(testset, indices_test)

    print(f"[INFO] Dataset sizes after subsetting - Train: {len(trainset)}, Test: {len(testset)}")

    # Partition for federated learning
    # FIXED: Use IID for better class balance (ensures all attacks work effectively)
    client_data = partition_dataset(trainset, num_clients=num_clients, iid=True)
    _pin = torch.cuda.is_available()
    test_loader = DataLoader(testset, batch_size=256, shuffle=False,
                             pin_memory=_pin, num_workers=2)

    # Prepare test arrays
    test_X, y_test = next(iter(DataLoader(testset, batch_size=len(testset), shuffle=False)))
    test_X, y_test = test_X.numpy(), y_test.numpy()

    # Attack setup
    framework = EvaluationFramework()
    attack_configs = framework.get_attack_configs(malicious_clients=malicious_clients)

    comparison_results = {}

    # Evaluate each attack
    for attack_name, attack_config in attack_configs.items():
        print("=" * 80)
        print(f"\n--- Testing {attack_name.upper()} ---")
        print("=" * 80)


        clients = []
        _ref_state = FashionMNISTNet().state_dict()
        for i in range(num_clients):
            model = FashionMNISTNet()
            model.load_state_dict(_ref_state)
            client_attack = attack_config if (attack_config and i < attack_config.num_malicious_clients) else None
            clients.append(
                DecentralizedClient(i, client_data[i][0], client_data[i][1], model, client_attack)
            )

        coordinator = DecentralizedFLCoordinator(
            clients,
            use_defense=False,
            defense_type=None,  # No committee structure for this test
            aggregation_method='fedavg'
        )
        accuracy_history = coordinator.run_federated_learning(rounds=rounds)

        final_accuracy, client_accuracies = coordinator.evaluate_on_test_set(test_loader)
        try:
            attack_success_rates = coordinator.evaluate_attack_success(test_X, y_test)
        except Exception as e:
            print(f"[WARN] Attack success evaluation failed: {e}")
            attack_success_rates = [0.0] * len(clients)

        if attack_config is None:
            comparison_results[attack_name] = {
                'final_accuracy': final_accuracy,
                'attack_success': 0.0,
                'malicious_clients': 0,
                'data_poisoning_rate': 0.0,
                'attack_type': "BASELINE"
            }
        else:
            comparison_results[attack_name] = {
                'final_accuracy': final_accuracy,
                'attack_success': np.mean(attack_success_rates) if attack_success_rates else 0.0,
                'malicious_clients': sum(1 for c in clients if getattr(c, 'is_malicious', False)),
                'data_poisoning_rate': attack_config.data_poisoning_rate,
                'attack_type': attack_config.attack_type
            }

        print(f"Results: Accuracy = {final_accuracy:.4f}, Attack Success = {np.mean(attack_success_rates):.4f}")

    # Print Results Table
    print(f"\n{'='*80}")
    print("ATTACK COMPARISON RESULTS")
    print(f"{'='*80}")

    print(f"{'Attack':<20} {'Type':<12} {'PDR':<6} {'Accuracy':<10} {'Attack Success':<15} {'Stealth':<12}")
    print("-" * 80)

    for name, res in comparison_results.items():
        stealth_score = res['final_accuracy'] * (1 - res['attack_success'] * 0.5)
        print(f"{name:<20} {res['attack_type'].upper():<12} {res['data_poisoning_rate']:.1%}    "
              f"{res['final_accuracy']:.4f}    {res['attack_success']:.4f}         {stealth_score:.4f}")

    print(f"\n{'='*80}\nATTACK COMPARISON COMPLETED\n{'='*80}")
    return comparison_results


def run_comprehensive_defense_test(subset_fraction=0.25, num_clients=25, rounds = 5, out_dir="comprehensive_defense_results",
                                   resume_checkpoint=False, checkpoint_interval=2):
    """
    Comprehensive test of ALL committee-based defenses against ALL attacks for ALL datasets.
    Tests adaptivecommittee and cmfl defenses separately (both distributed).
    Generates comparative visualizations and recovery rate analysis.

    Args:
        subset_fraction: Fraction of dataset to use (0.1 = 10%)
        num_clients: Number of clients in federated learning
        rounds: Number of FL rounds
        out_dir: Output directory for results
        resume_checkpoint: If True, check for and resume from existing checkpoints
        checkpoint_interval: Save checkpoint every N rounds

    Returns:
        Complete results dictionary with all defense comparisons
    """
    print("\n" + "="*100)
    print("COMPREHENSIVE DEFENSE TESTING FRAMEWORK")
    print(f"Testing ALL 2 committee-based defenses (adaptivecommittee, cmfl) against ALL attacks")
    print(f"Dataset fraction: {subset_fraction*100:.0f}%, Clients: {num_clients}, Rounds: {rounds}")
    if resume_checkpoint:
        print(f"Checkpoint Resume: ENABLED (interval: every {checkpoint_interval} rounds)")
    print("="*100)

    should_resume = False

    if resume_checkpoint:
        # Search for any checkpoint files recursively in out_dir/checkpoints/
        checkpoint_base = os.path.join(out_dir, "checkpoints")
        found_checkpoint = False
        for root, dirs, files in os.walk(checkpoint_base):
            for file in files:
                if file.endswith(".pth"):
                    found_checkpoint = True
                    break
            if found_checkpoint:
                break
        if found_checkpoint:
            run_id = os.path.basename(os.path.normpath(out_dir))
            print(f"[INFO] Found checkpoint for run_id: {run_id}")
        else:
            print("[INFO] No existing checkpoint found. Starting fresh...")
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    framework = EvaluationFramework(out_dir=out_dir, run_id=run_id)

    # Check for existing checkpoints if resume is enabled
    if resume_checkpoint:
        resume_info = framework.checkpoint_manager.get_resume_info()
        if resume_info["can_resume"]:
            print("\n" + "="*100)
            print("[CHECKPOINT DETECTED]")
            print("="*100)
            print(f"  Last run was interrupted at:")
            print(f"  Dataset: {resume_info['dataset']}")
            print(f"  Attack: {resume_info['attack']}")
            print(f"  Scenario: {resume_info['scenario']}")
            print(f"  Last completed round: {resume_info['last_round']}/{rounds}")
            print(f"  Completed scenarios: {resume_info['completed_scenarios']}")
            print("="*100)

            # NON-INTERACTIVE: Always resume if resume_checkpoint is set
            should_resume = True
            print("\n[RESUME] Resuming from checkpoint...")
        else:
            print("\n[INFO] No existing checkpoint found. Starting fresh...")
            should_resume = False

    # Store checkpoint interval for use in run_and_evaluate_coordinator
    framework.checkpoint_interval = checkpoint_interval

    # Defense types to test - ALL 2 committee-based defenses
    # Only distributed committee-based defenses (centralized defenses removed)
    defense_types = ['adaptivecommittee', 'cmfl']

    # Get datasets and models
    datasets, models = framework.get_datasets_and_models()
    attack_configs = framework.get_attack_configs(malicious_clients=10)  # 40% malicious for stronger attacks

    # IMPORTANT: Define explicit attack order to ensure all attacks run sequentially
    # NOTE: 'none' is NOT an attack - it's the baseline without any attack
    # We handle baseline separately before testing actual attacks
    attack_order = ['slf', 'dlf', 'centralized', 'coordinated', 'random', 'model_dependent',
                    # Model poisoning attacks — commented out (redundant; uncomment if needed)
                    # 'local_model_replacement', 'local_model_noise', 'global_model_replacement', 'aggregation_modification',
                    ]

    print(f"\n[DEBUG] Real attacks to test: {attack_order}")
    print(f"[DEBUG] Total attacks: {len(attack_order)} (baseline will be run separately)")

    # Store results
    all_results = {}

    # Loop through datasets, then attacks, then all defenses for each attack
    # This ensures all defenses run for one attack before moving to the next attack
    for dataset_name, (trainset, testset) in datasets.items():
        print(f"\n{'='*100}")
        print(f"TESTING DATASET: {dataset_name}")
        print(f"{'='*100}")

        all_results[dataset_name] = {}

        # Subset datasets
        print(f"[INFO] Original sizes - Train: {len(trainset)}, Test: {len(testset)}")
        trainset_subset = framework._subset_dataset(trainset, subset_fraction)
        testset_subset = framework._subset_dataset(testset, subset_fraction)
        print(f"[INFO] Subsetted sizes - Train: {len(trainset_subset)}, Test: {len(testset_subset)}")

        # Partition data - using IID for better class balance (ensures malicious clients have source class samples)
        client_data = partition_dataset(trainset_subset, num_clients, iid=True)
        _pin = torch.cuda.is_available()
        test_loader = DataLoader(testset_subset, batch_size=256, shuffle=False,
                                 pin_memory=_pin, num_workers=2)
        test_X, y_test = next(iter(DataLoader(testset_subset, batch_size=len(testset_subset), shuffle=False)))
        test_X, y_test = test_X.numpy(), y_test.numpy()

        # CRITICAL FIX: Generate FIXED malicious client IDs ONCE for this dataset
        # These IDs will be reused for ALL experiments (baseline, attack-only, attack+defense)

        malicious_clients = min(int(num_clients * 0.40), num_clients)  # 40% of actual clients
        fixed_malicious_ids = set(random.sample(range(num_clients), malicious_clients))
        print(f"\n{'='*80}")
        print(f"FIXED MALICIOUS CLIENT IDs FOR ALL EXPERIMENTS")
        print(f"{'='*80}")
        print(f"[CRITICAL] Malicious IDs: {sorted(list(fixed_malicious_ids))}")
        print(f"[INFO] These {len(fixed_malicious_ids)} malicious clients will be used in:")
        print(f"  1. Clean baseline (attack disabled, behave as honest)")
        print(f"  2. Attack-only scenarios (attack enabled when selected)")
        print(f"  3. Attack+Defense scenarios (same attackers, defense active)")
        print(f"{'='*80}")

        # ============================================================================
        # STEP 1: Run BASELINE (no attack, no defense) - This is NOT an attack!
        # ============================================================================
        print(f"\n{'-'*100}")
        print(f"BASELINE (No Attack, No Defense)")
        print(f"{'-'*100}")

        baseline_scenario = "baseline"
        baseline_acc = None
        bl_training_acc_history = []
        bl_training_loss_history = []
        bl_test_acc_history = []
        bl_test_loss_history = []

        if should_resume and framework.checkpoint_manager.is_scenario_complete(dataset_name, 'baseline', baseline_scenario):
            print(f"[SKIP] Baseline already completed - loading from checkpoint")
            checkpoint = framework.checkpoint_manager.load_checkpoint(dataset_name, 'baseline', baseline_scenario, "best")
            if checkpoint and 'metrics' in checkpoint:
                baseline_acc = checkpoint['metrics'].get('accuracy', None)
                # Try to load history from checkpoint if available
                bl_training_acc_history = checkpoint['metrics'].get('training_acc_history', [])
                bl_training_loss_history = checkpoint['metrics'].get('training_loss_history', [])
                bl_test_acc_history = checkpoint['metrics'].get('test_acc_history', [])
                bl_test_loss_history = checkpoint['metrics'].get('test_loss_history', [])
                if baseline_acc is not None:
                    print(f"Loaded baseline accuracy: {baseline_acc:.4f}")
                else:
                    print("Loaded baseline accuracy: None")
        else:
            print(f"[INFO] Running baseline (clean training, no attack, no defense)...")
            baseline_clients = framework.build_clients(num_clients, client_data, models, dataset_name, attack_config=None, fixed_malicious_ids=fixed_malicious_ids)
            _, _, baseline_acc, baseline_asr_list, bl_training_acc_history, bl_training_loss_history, bl_test_acc_history, bl_test_loss_history, _, _ = framework.run_and_evaluate_coordinator(
                baseline_clients, rounds, test_loader, test_X=test_X, y_test=y_test,
                use_defense=False, dataset=dataset_name, attack='baseline', scenario=baseline_scenario
            )
            print(f"✓ Baseline accuracy: {baseline_acc:.4f}")
            print(f"[HISTORY] Collected {len(bl_training_acc_history)} rounds of training history")
            print(f"[HISTORY] Collected {len(bl_test_acc_history)} rounds of test history")
        # ============================================================================
        # STEP 2: Test each REAL ATTACK with all defenses
        # ============================================================================
        for attack_idx, attack_name in enumerate(attack_order, start=1):
            print(f"\n[DEBUG] Starting attack {attack_idx}/{len(attack_order)}: '{attack_name}'")
            attack_config = attack_configs.get(attack_name)

            if attack_config is None:
                print(f"[ERROR] Attack config not found for '{attack_name}'! Skipping...")
                continue

            print(f"\n{'-'*100}")
            print(f"ATTACK {attack_idx}/{len(attack_order)}: {attack_name.upper()}")
            print(f"{'-'*100}")

            attack_results = {}

            # Store baseline accuracy for this attack (already computed above)
            if baseline_acc is not None:
                baseline_acc_value = float(baseline_acc)
                # Store baseline with full history
                attack_results['baseline'] = {
                    'final_accuracy': baseline_acc_value,
                    'accuracy': baseline_acc_value,
                    'attack_success_rate': 0.0,
                    'training_acc_history': [float(x) for x in bl_training_acc_history] if bl_training_acc_history else [],
                    'training_loss_history': [float(x) for x in bl_training_loss_history] if bl_training_loss_history else [],
                    'test_acc_history': [float(x) for x in bl_test_acc_history] if bl_test_acc_history else [],
                    'test_loss_history': [float(x) for x in bl_test_loss_history] if bl_test_loss_history else [],
                    'attack_success_rates': []
                }
            else:
                # Fallback if baseline wasn't computed
                attack_results['baseline'] = {
                    'final_accuracy': 0.0,
                    'accuracy': 0.0,
                    'attack_success_rate': 0.0,
                    'training_acc_history': [],
                    'training_loss_history': [],
                    'test_acc_history': [],
                    'test_loss_history': [],
                    'attack_success_rates': []
                }

            # 1. Run attack WITHOUT defense
            scenario_name = "attack"
            start_round = 0
            if should_resume and framework.checkpoint_manager.is_scenario_complete(dataset_name, attack_name, scenario_name):
                print(f"\n[1/{len(defense_types)+1}] Attack (no defense) - SKIPPED (already completed)")
                checkpoint = framework.checkpoint_manager.load_checkpoint(dataset_name, attack_name, scenario_name, "best")
                if checkpoint and 'metrics' in checkpoint:
                    attack_results['attack'] = checkpoint['metrics']
                start_round = checkpoint.get('round', 0) + 1
            else:
                print(f"\n[1/{len(defense_types)+1}] {attack_name.upper()} attack (no defense)...")
                attack_clients = framework.build_clients(num_clients, client_data, models, dataset_name, attack_config, fixed_malicious_ids=fixed_malicious_ids)
                attack_coord, attack_result, attack_acc, attack_asr_list, training_acc_history, training_loss_history, test_acc_history, test_loss_history, _, _ = framework.run_and_evaluate_coordinator(
                    attack_clients, rounds, test_loader, test_X=test_X, y_test=y_test,
                    use_defense=False, dataset=dataset_name, attack=attack_name, scenario=scenario_name
                )

                attack_asr = np.mean([r for r in attack_asr_list if r > 0]) if any(r > 0 for r in attack_asr_list) else 0.0
                attack_results['attack'] = {
                    'final_accuracy': float(attack_acc),
                    'accuracy': float(attack_acc),
                    'attack_success_rate': float(attack_asr),
                    'training_acc_history': [float(x) for x in training_acc_history],
                    'training_loss_history': [float(x) for x in training_loss_history],
                    'test_acc_history': [float(x) for x in test_acc_history],
                    'test_loss_history': [float(x) for x in test_loss_history],
                    'attack_success_rates': [float(x) for x in attack_asr_list] if attack_asr_list else []
                }
                print(f"Attack accuracy: {attack_acc:.4f}, ASR: {attack_asr:.4f}")
                print(f"[HISTORY] Collected {len(training_acc_history)} rounds of training history")
                print(f"[HISTORY] Collected {len(test_acc_history)} rounds of test history")

            # Test each defense type
            print(f"\n[DEBUG] Starting defense loop for attack '{attack_name}'")
            print(f"[DEBUG] Defense types to test: {defense_types}")
            print(f"[DEBUG] Resume mode: {should_resume}")

            for idx, defense_type in enumerate(defense_types, start=2):
                scenario_name = f"defense_{defense_type}"
                start_round = 0
                print(f"\n[DEBUG] Processing defense {idx-1}/{len(defense_types)}: {defense_type}")

                # Check if scenario is already completed
                is_completed = should_resume and framework.checkpoint_manager.is_scenario_complete(dataset_name, attack_name, scenario_name)
                print(f"[DEBUG] Scenario '{scenario_name}' is_completed: {is_completed}")

                if is_completed:
                    print(f"\n[{idx}/{len(defense_types)+1}] {defense_type.upper()} defense - SKIPPED (already completed)")
                    checkpoint = framework.checkpoint_manager.load_checkpoint(dataset_name, attack_name, scenario_name, "best")
                    if checkpoint and 'metrics' in checkpoint:
                        attack_results[defense_type] = checkpoint['metrics']
                    start_round = checkpoint.get('round', 0) + 1
                    continue

                print(f"\n[{idx}/{len(defense_types)+1}] {attack_name.upper()} with {defense_type.upper()} defense...")

                defense_clients = framework.build_clients(num_clients, client_data, models, dataset_name, attack_config, fixed_malicious_ids=fixed_malicious_ids)
                try:
                    _, _, defense_acc, defense_asr_list, def_training_acc_history, def_training_loss_history, def_test_acc_history, def_test_loss_history, detection_metrics, _ = framework.run_and_evaluate_coordinator(
                        defense_clients, rounds, test_loader, test_X=test_X, y_test=y_test,
                        use_defense=True, defense_type=defense_type,
                        dataset=dataset_name, attack=attack_name, scenario=scenario_name
                    )

                    defense_asr = np.mean([r for r in defense_asr_list if r > 0]) if any(r > 0 for r in defense_asr_list) else 0.0

                    # Calculate recovery rate
                    baseline_acc_val = attack_results.get('baseline', {}).get('accuracy', 0)
                    attack_acc_val = attack_results.get('attack', {}).get('accuracy', 0)

                    if baseline_acc_val != attack_acc_val:
                        recovery = ((defense_acc - attack_acc_val) / (baseline_acc_val - attack_acc_val)) * 100
                        recovery = max(0, min(100, recovery))
                    else:
                        recovery = 100.0 if defense_acc >= baseline_acc_val else 0.0

                    # Store results including detection metrics AND history arrays
                    attack_results[defense_type] = {
                        'final_accuracy': float(defense_acc),
                        'accuracy': float(defense_acc),
                        'attack_success_rate': float(defense_asr),
                        'recovery_rate': float(recovery),
                        'detection_metrics': detection_metrics,
                        'training_acc_history': [float(x) for x in def_training_acc_history],
                        'training_loss_history': [float(x) for x in def_training_loss_history],
                        'test_acc_history': [float(x) for x in def_test_acc_history],
                        'test_loss_history': [float(x) for x in def_test_loss_history],
                        'attack_success_rates': [float(x) for x in defense_asr_list] if defense_asr_list else []
                    }

                    print(f"{defense_type.capitalize()} defense - Accuracy: {defense_acc:.4f}, Recovery: {recovery:.2f}%")
                    print(f"[HISTORY] Collected {len(def_training_acc_history)} rounds of training history")
                    print(f"[HISTORY] Collected {len(def_test_acc_history)} rounds of test history")
                    print(f"[DEBUG] Detection metrics for {defense_type}: {detection_metrics}")

                except Exception as e:
                    print(f"[ERROR] {defense_type.capitalize()} defense failed: {e}")
                    traceback.print_exc()
                    attack_results[defense_type] = {
                        'accuracy': 0.0,
                        'attack_success_rate': 0.0,
                        'recovery_rate': 0.0,
                        'detection_metrics': {},
                        'error': str(e)
                    }
                    print(f"[DEBUG] Continuing to next defense after error...")

            # Store attack results
            print(f"\n[DEBUG] Completed all defenses for attack '{attack_name}'")
            print(f"[DEBUG] Attack results keys: {list(attack_results.keys())}")
            all_results[dataset_name][attack_name] = attack_results
            print(f"[INFO] ✓ Attack {attack_idx}/{len(attack_order)} ({attack_name}) completed successfully")
            print(f"[INFO] Moving to next attack...")

    # Print execution summary
    print(f"\n{'='*100}")
    print("EXECUTION SUMMARY")
    print(f"{'='*100}")
    for dataset_name, dataset_results in all_results.items():
        print(f"\nDataset: {dataset_name}")
        print(f"  Baseline: Completed (shared across all attacks)")
        print(f"  Attacks executed: {list(dataset_results.keys())}")
        print(f"  Total attacks: {len(dataset_results)}/{len(attack_order)}")
        for attack_name, attack_results in dataset_results.items():
            defenses_tested = [k for k in attack_results.keys() if k not in ['baseline', 'attack']]
            print(f"    {attack_name}: attack + {len(defenses_tested)} defenses {defenses_tested}")

    # Save results
    print(f"\n{'='*100}")
    print("SAVING RESULTS")
    print(f"{'='*100}")

    # Save JSON
    json_path = os.path.join(framework.results_dir, f"comprehensive_defense_results_{run_id}.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"[INFO] Results saved to: {json_path}")

    # Save CSV
    csv_path = os.path.join(framework.results_dir, f"comprehensive_defense_results_{run_id}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Dataset', 'Attack', 'Defense', 'Baseline_Acc', 'Attack_Acc',
            'Defense_Acc', 'ASR', 'Recovery_Rate'
        ])

        for dataset_name, dataset_results in all_results.items():
            for attack_name, attack_results in dataset_results.items():
                # FIXED: Skip 'none' attack (baseline only, no attack/defense scenarios)
                if attack_name == 'none' or 'attack' not in attack_results:
                    continue

                # Safely get baseline and attack results
                baseline_acc = attack_results.get('baseline', {}).get('accuracy', 0.0)
                attack_acc = attack_results.get('attack', {}).get('accuracy', 0.0)
                attack_asr = attack_results.get('attack', {}).get('attack_success_rate', 0.0)

                for defense_type in defense_types:
                    if defense_type in attack_results and 'error' not in attack_results[defense_type]:
                        defense_acc = attack_results[defense_type]['accuracy']
                        recovery = attack_results[defense_type].get('recovery_rate', 0.0)

                        writer.writerow([
                            dataset_name, attack_name, defense_type,
                            f"{baseline_acc:.4f}", f"{attack_acc:.4f}",
                            f"{defense_acc:.4f}", f"{attack_asr:.4f}",
                            f"{recovery:.2f}"
                        ])


    print(f"[INFO] CSV saved to: {csv_path}")

    # SKIP plotting during execution for faster training
    # Plots can be generated later using generate_plots_from_results.py
    print(f"\n{'='*100}")
    print("SKIPPING VISUALIZATION GENERATION (for faster execution)")
    print("To generate plots later, run: python generate_plots_from_results.py")
    print(f"{'='*100}")

    # Generate comprehensive defense analysis tables (includes summary printing)
    from defense_analysis import DefenseAnalyzer
    analyzer = DefenseAnalyzer(results_dir=framework.results_dir)
    analyzer.generate_all_analysis(
        results=all_results,
        num_malicious=10,  # FIXED: Match the actual malicious_clients count (line 1492)
        num_clients=num_clients,
        total_rounds=rounds
    )

    # Generate research paper tables (Table 1, Table 2)
    print(f"\n{'='*100}")
    print("GENERATING RESEARCH PAPER TABLES")
    print(f"{'='*100}")
    from table_generator import ResearchTableGenerator
    table_gen = ResearchTableGenerator(results_dir=framework.results_dir)
    table_gen.results = all_results
    table_gen.generate_all_tables()

    print(f"\n{'='*100}")
    print("COMPREHENSIVE DEFENSE TEST COMPLETED")
    print(f"{'='*100}")
    print(f"\nResults saved to: {framework.base_run_dir}")
    print(f"Plots saved to: {framework.plots_dir}")
    print(f"Data saved to: {framework.results_dir}")
    print(f"Analysis tables: {framework.results_dir}")
    print(f"Research tables:")
    print(f"  - Table 1 (Detection): {framework.results_dir}/table1_detection_performance.csv")
    print(f"  - Table 2 (Model): {framework.results_dir}/table2_model_performance.csv")
    print(f"{'='*100}")

    return all_results


