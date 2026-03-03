#!/usr/bin/env python3
"""
Comprehensive Ablation Study for Federated Learning Defense

This script performs a comprehensive ablation study analyzing the impact of:
1. Data Poison Percentage (20%, 40%, 60%, 80%)
2. Client Malicious Ratio (20%, 40%, 60%, 80%)
3. Non-IID Data Distribution (0%, 20%, 40%, 60%, 80%)
4. Aggregation Schemes (FedAvg, Krum, Multi-Krum, Median, Trimmed Mean)

Fixed parameters for scalability study:
- 100 total clients
- 100 clients per round (full participation)
- All other parameters at 40%

Usage:
    # Run full ablation study
    python run_impact_analysis.py --all

    # Run specific ablation
    python run_impact_analysis.py --data-poison-percentage
    python run_impact_analysis.py --client-malicious-ratio
    python run_impact_analysis.py --non-iid
    python run_impact_analysis.py --aggregation

    # Custom parameters
    python run_impact_analysis.py --data-poison-percentage --percentages 0.2 0.4 0.6 0.8
    python run_impact_analysis.py --client-malicious-ratio --ratios 0.2 0.4 0.6 0.8

    # Use dataset fraction for faster testing
    python run_impact_analysis.py --all --dataset-fraction 0.1  # Use 10% of data
    python run_impact_analysis.py --data-poison-percentage --dataset-fraction 0.25  # Use 25% of data
"""

# Thread settings removed: setting to '1' serializes numpy/sklearn ops,
# killing multi-core parallelism and hurting performance by 20-40%.
import os

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Union

# Import evaluation framework for proper client setup
from evaluation import EvaluationFramework


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

    # _compute_test_loss removed — accuracy and loss are now computed
    # in a single forward pass by coordinator.evaluate_on_test_set()


def _resolve_clients(total_clients, dataset: str) -> int:
    """Resolve per-dataset client count.

    total_clients can be:
        int              -> same count for every dataset
        Dict[str, int]   -> per-dataset counts, e.g.
                            {'FEMNIST': 3550, 'Shakespeare': 143, 'Sentiment140': 772}
                            Falls back to 100 if the key is missing.
    """
    if isinstance(total_clients, dict):
        return total_clients.get(dataset, 100)
    return total_clients


# =============================================================================
# NEW HELPER FUNCTIONS: Proper Experimental Design
# =============================================================================

def run_baseline_evaluation(
    framework: EvaluationFramework,
    dataset_name: str,
    num_clients: int,
    rounds: int,
    clients_per_round: int,
    aggregation: str = "fedavg",
    alpha: float = 1.0,
    dataset_fraction: float = 1.0,
    scenario_name: str = "baseline",
    aggregation_participation_frac: float = None,
    committee_size_frac: float = 0.40,
    scoring_mode: str = 'distance_only',
    selection_strategy: int = 1,
    lr: float = None
):
    """
    Run baseline evaluation with clean data (NO attack, NO poisoning).

    This represents the upper bound of model performance with benign clients only.

    Args:
        framework: Evaluation framework instance
        dataset_name: 'FEMNIST', 'Fashion-MNIST', 'EMNIST', 'CIFAR-10', 'Shakespeare', or 'Sentiment140'
        num_clients: Total number of clients
        rounds: Number of FL rounds
        clients_per_round: Clients participating per round
        aggregation: Aggregation method
        alpha: Dirichlet alpha for Non-IID data
        dataset_fraction: Fraction of dataset to use
        scenario_name: Scenario name for tracking

    Returns:
        float: Baseline accuracy (clean data, no attack)
    """
    import torch
    import math
    from coordinator import DecentralizedFLCoordinator

    # Get datasets and models
    datasets_and_loaders, models_map = framework.get_datasets_and_models(dataset_name)

    if dataset_name not in datasets_and_loaders:
        print(f"[ERROR] Dataset {dataset_name} not found")
        return 0.0, [], []

    # Unpack datasets
    train_dataset, test_dataset = datasets_and_loaders[dataset_name]

    # Subset datasets if needed
    if dataset_fraction < 1.0:
        train_dataset = framework._subset_dataset(train_dataset, dataset_fraction)
        test_dataset = framework._subset_dataset(test_dataset, dataset_fraction)

    # Create test loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False
    )

    # Distribute data among clients
    from datasets import partition_dataset
    client_data = partition_dataset(
        train_dataset,
        num_clients,
        iid=False,
        alpha=alpha
    )

    # Build CLEAN clients (NO attack config)
    clients = framework.build_clients(
        num_clients=num_clients,
        client_data=client_data,
        models_map=models_map,
        dataset_name=dataset_name,
        attack_config=None  # NO ATTACK for baseline
    )

    # ---------------------------------------------------------------------------
    # Create coordinator — respects committee schemes even on clean (baseline) data
    # ---------------------------------------------------------------------------
    _COMMITTEE_SCHEMES = {'cmfl', 'cmfl_ii'}

    # Dataset-specific learning rates and batch sizes
    if lr is None:
        lr = DATASET_LR.get(dataset_name, 0.001)
    _bs = DATASET_BATCH_SIZE.get(dataset_name, 32)

    if aggregation.lower() in _COMMITTEE_SCHEMES:
        # cmfl_ii → cmfl defense with Selection Strategy II
        if aggregation.lower() == 'cmfl_ii':
            chosen_defense = 'cmfl'
            selection_strategy = 2
        else:
            chosen_defense = aggregation.lower()
        base_agg         = 'weighted_avg'
        committee_size   = max(2, int(clients_per_round * committee_size_frac))
        training_clients = max(4, clients_per_round - committee_size)

        coordinator = DecentralizedFLCoordinator(
            clients,
            use_defense=True,
            defense_type=chosen_defense,
            committee_size=committee_size,
            training_clients_per_round=training_clients,
            aggregation_method=base_agg,
            use_anomaly_detection=False,  # Baseline: no malicious clients → skip scoring/detection
            clients_per_round=clients_per_round,
            aggregation_participation_frac=aggregation_participation_frac,
            scoring_mode=scoring_mode,
            selection_strategy=selection_strategy,
            lr=lr, batch_size=_bs
        )
        strategy_label = "II" if selection_strategy == 2 else "I"
        print(f"[INFO] Baseline ({chosen_defense.upper()}): committee={committee_size} ({committee_size_frac*100:.0f}%), training={training_clients}, Strategy {strategy_label}, detection=OFF (clean baseline), lr={lr}, batch_size={_bs}")
    else:
        coordinator = DecentralizedFLCoordinator(
            clients,
            use_defense=False,
            defense_type=None,
            aggregation_method=aggregation,
            clients_per_round=clients_per_round,
            lr=lr, batch_size=_bs
        )
        print(f"[INFO] Baseline ({aggregation.upper()}): plain FL, {clients_per_round} clients/round")

    # Track test accuracy and loss per round
    test_acc_history = []
    test_loss_history = []

    print(f"[INFO] Running baseline for {rounds} rounds...")

    # Evaluate every eval_interval rounds + always on last round (saves ~80% eval time)
    eval_interval = max(1, rounds // 20) if rounds > 20 else 1

    # Run FL round by round
    for round_num in range(1, rounds + 1):
        # Run a single round with proper round numbering
        coordinator.run_federated_learning(
            rounds=1, aggregation_method=aggregation,
            test_loader=test_loader,
            round_offset=round_num - 1, total_rounds=rounds
        )

        # Evaluate at intervals and on the last round
        if round_num % eval_interval == 0 or round_num == rounds:
            try:
                round_test_acc, round_test_loss = coordinator.evaluate_on_test_set(test_loader)

                if math.isnan(round_test_acc) or math.isinf(round_test_acc):
                    round_test_acc = 1e10
                if math.isnan(round_test_loss) or math.isinf(round_test_loss):
                    round_test_loss = 1e10

                test_acc_history.append(float(round_test_acc))
                test_loss_history.append(float(round_test_loss))
            except Exception as e:
                print(f"[WARN] Test evaluation failed at round {round_num}: {e}")
                test_acc_history.append(test_acc_history[-1] if test_acc_history else 0.0)
                test_loss_history.append(test_loss_history[-1] if test_loss_history else float('inf'))
        else:
            # Carry forward last known value for skipped rounds
            test_acc_history.append(test_acc_history[-1] if test_acc_history else 0.0)
            test_loss_history.append(test_loss_history[-1] if test_loss_history else float('inf'))

    final_accuracy = test_acc_history[-1] if test_acc_history else 1e10
    if math.isnan(final_accuracy) or math.isinf(final_accuracy):
        final_accuracy = 1e10

    return final_accuracy, test_acc_history, test_loss_history


def create_attack_scenario(
    framework: EvaluationFramework,
    dataset_name: str,
    attack_type: str,
    poison_percentage: float,
    num_clients: int,
    malicious_clients: int,
    alpha: float = 1.0,
    dataset_fraction: float = 1.0
):
    """
    Create attack scenario by building poisoned clients ONCE.

    This creates the poisoned clients that will be reused (via deep copy)
    for testing all defenses, ensuring fair comparison.

    Args:
        framework: Evaluation framework instance
        dataset_name: 'FEMNIST', 'Fashion-MNIST', 'EMNIST', 'CIFAR-10', 'Shakespeare', or 'Sentiment140'
        attack_type: Attack to use ('slf', 'dlf', etc.)
        poison_percentage: Poison data percentage (0.0-1.0)
        num_clients: Total number of clients
        malicious_clients: Number of malicious clients
        alpha: Dirichlet alpha for Non-IID data
        dataset_fraction: Fraction of dataset to use

    Returns:
        tuple: (clients_list, test_loader, test_X, y_test)
    """
    import torch
    import numpy as np
    from attacks.base import AttackConfig

    # Get datasets and models
    datasets_and_loaders, models_map = framework.get_datasets_and_models(dataset_name)

    if dataset_name not in datasets_and_loaders:
        print(f"[ERROR] Dataset {dataset_name} not found")
        return None, None, None, None

    # NOTE: Trigger-based attacks now support both image (pixel patterns) and
    # text (token replacement) datasets — no skip needed for sequence datasets.

    # Unpack datasets
    train_dataset, test_dataset = datasets_and_loaders[dataset_name]

    # Subset datasets if needed
    if dataset_fraction < 1.0:
        train_dataset = framework._subset_dataset(train_dataset, dataset_fraction)
        test_dataset = framework._subset_dataset(test_dataset, dataset_fraction)

    # Create test loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False
    )

    # Extract test data as numpy arrays for ASR evaluation
    test_X_list = []
    y_test_list = []
    for X_batch, y_batch in test_loader:
        test_X_list.append(X_batch.numpy())
        y_test_list.append(y_batch.numpy())
    test_X = np.concatenate(test_X_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    # Create attack configuration with dynamic source/target classes based on attack type
    # Match the configuration from evaluation.py
    if attack_type.lower() == 'slf':
        attack_config = AttackConfig(
            attack_type=attack_type,
            data_poisoning_rate=poison_percentage,
            source_class=[0, 2, 4, 6],
            target_class=[1, 3, 5, 7],
            num_malicious_clients=malicious_clients
        )
    elif attack_type.lower() == 'dlf':
        attack_config = AttackConfig(
            attack_type=attack_type,
            data_poisoning_rate=poison_percentage,
            target_class=0,
            num_malicious_clients=malicious_clients
        )
    elif attack_type.lower() in ['centralized', 'coordinated', 'random']:
        attack_config = AttackConfig(
            attack_type=attack_type,
            data_poisoning_rate=poison_percentage,
            target_class=[1, 3, 5, 7],
            trigger_size=(8, 8),
            trigger_intensity=0.7,
            num_malicious_clients=malicious_clients
        )
    elif attack_type.lower() == 'model_dependent':
        attack_config = AttackConfig(
            attack_type=attack_type,
            data_poisoning_rate=0.90,  # Higher poisoning rate for model_dependent
            source_class=1,
            target_class=[1, 3, 5, 7],
            epsilon=0.65,
            num_malicious_clients=malicious_clients
        )
    elif attack_type.lower() == 'gradient_scaling':
        # Byzantine attack: manipulates gradients, NOT data. data_poisoning_rate=0.
        attack_config = AttackConfig(
            attack_type=attack_type,
            num_malicious_clients=malicious_clients,
            data_poisoning_rate=0.0
        )
    elif attack_type.lower() in ['same_value', 'back_gradient']:
        # Byzantine attacks: manipulate gradients, NOT data. data_poisoning_rate=0.
        attack_config = AttackConfig(
            attack_type=attack_type,
            num_malicious_clients=malicious_clients,
            data_poisoning_rate=0.0
        )
    else:
        # Fallback for unknown attack types
        attack_config = AttackConfig(
            attack_type=attack_type,
            num_malicious_clients=malicious_clients,
            data_poisoning_rate=poison_percentage,
            source_class=1,
            target_class=7
        )

    # Distribute data among clients
    from datasets import partition_dataset
    client_data = partition_dataset(
        train_dataset,
        num_clients,
        iid=False,
        alpha=alpha
    )

    # Build poisoned clients (ONCE!)
    clients = framework.build_clients(
        num_clients=num_clients,
        client_data=client_data,
        models_map=models_map,
        dataset_name=dataset_name,
        attack_config=attack_config
    )

    return clients, test_loader, test_X, y_test


def test_defense_on_scenario(
    clients_list,
    test_loader,
    test_X,
    y_test,
    rounds: int,
    aggregation: str = "fedavg",
    defense_type: str = None,
    clients_per_round: Optional[int] = None,
    framework: EvaluationFramework = None,
    dataset_name: str = None,
    attack_type: str = None,
    scenario_name: str = None,
    aggregation_participation_frac: float = None,
    committee_size_frac: float = 0.40,
    scoring_mode: str = 'distance_only',
    selection_strategy: int = 1,
    lr: float = 0.01,
    batch_size: int = 32
):
    """
    Run one FL scenario (baseline or attack) under a given aggregation scheme.

    ``aggregation`` is the single knob that selects the method:
        'fedavg' | 'krum' | 'multi_krum' | 'median' | 'trimmed_mean'
            -> plain FL, no committee
        'cmfl' | 'cmfl_ii'
            -> committee-based FL (the scheme itself IS the defense)

    Deep-copies clients so the same poisoned set can be reused across schemes.

    Returns:
        tuple: (final_accuracy, attack_success_rate, detection_metrics, test_acc_history, test_loss_history)
    """
    import numpy as np
    import math
    from coordinator import DecentralizedFLCoordinator

    _COMMITTEE_SCHEMES = {'cmfl', 'cmfl_ii'}

    # Reset clients to their post-poisoning snapshot (no expensive deep copy)
    if hasattr(clients_list[0], '_snapshot_model_state'):
        for client in clients_list:
            client.reset_for_new_run()
        clients_copy = clients_list
    else:
        # Fallback: no snapshot available (e.g. called outside _run_per_scheme)
        import copy
        clients_copy = copy.deepcopy(clients_list)

    # ---------------------------------------------------------------------------
    # Decide: plain aggregation  vs  committee-based aggregation
    # ---------------------------------------------------------------------------
    if aggregation.lower() in _COMMITTEE_SCHEMES:
        # cmfl_ii → cmfl defense with Selection Strategy II
        if aggregation.lower() == 'cmfl_ii':
            chosen_defense = 'cmfl'
            selection_strategy = 2
        else:
            chosen_defense = aggregation.lower()          # 'cmfl'
        base_agg       = 'weighted_avg'               # data-size weighted averaging for committee schemes

        if clients_per_round is not None:
            committee_size   = max(2, int(clients_per_round * committee_size_frac))
            training_clients = max(4, clients_per_round - committee_size)
        else:
            committee_size   = None
            training_clients = None

        coordinator = DecentralizedFLCoordinator(
            clients_copy,
            use_defense=True,
            defense_type=chosen_defense,
            committee_size=committee_size,
            training_clients_per_round=training_clients,
            aggregation_method=base_agg,
            use_anomaly_detection=True,
            clients_per_round=clients_per_round,
            aggregation_participation_frac=aggregation_participation_frac,
            scoring_mode=scoring_mode,
            selection_strategy=selection_strategy,
            lr=lr, batch_size=batch_size
        )
        strategy_label = "II" if selection_strategy == 2 else "I"
        print(f"[INFO] {chosen_defense.upper()} scheme: committee={committee_size or 'auto'} ({committee_size_frac*100:.0f}%), "
              f"training={training_clients or 'auto'}, Strategy {strategy_label}, anomaly detection ON, scoring={scoring_mode}")
    else:
        # Plain aggregation — no committee, no anomaly detection
        coordinator = DecentralizedFLCoordinator(
            clients_copy,
            use_defense=False,
            defense_type=None,
            aggregation_method=aggregation,
            clients_per_round=clients_per_round,
            lr=lr, batch_size=batch_size
        )
        print(f"[INFO] {aggregation.upper()} scheme: plain FL, {clients_per_round or len(clients_copy)} clients/round")

    # Track test accuracy and loss per round
    test_acc_history = []
    test_loss_history = []

    if aggregation.lower() in _COMMITTEE_SCHEMES:
        scenario_label = f"{aggregation.upper()} defense"
    elif defense_type:
        scenario_label = f"{defense_type.upper()} defense"
    else:
        scenario_label = "ATTACK (no defense)"
    print(f"[INFO] Running {scenario_label} for {rounds} rounds...")

    # Evaluate every eval_interval rounds + always on last round (saves ~80% eval time)
    eval_interval = max(1, rounds // 20) if rounds > 20 else 1

    # Run FL round by round
    for round_num in range(1, rounds + 1):
        # Run a single round with proper round numbering
        coordinator.run_federated_learning(
            rounds=1,
            aggregation_method=aggregation,
            test_loader=test_loader,
            round_offset=round_num - 1,
            total_rounds=rounds
        )

        # Evaluate at intervals and on the last round
        if round_num % eval_interval == 0 or round_num == rounds:
            try:
                round_test_acc, round_test_loss = coordinator.evaluate_on_test_set(test_loader)

                if math.isnan(round_test_acc) or math.isinf(round_test_acc):
                    round_test_acc = 1e10
                if math.isnan(round_test_loss) or math.isinf(round_test_loss):
                    round_test_loss = 1e10

                test_acc_history.append(float(round_test_acc))
                test_loss_history.append(float(round_test_loss))
            except Exception as e:
                print(f"[WARN] Test evaluation failed at round {round_num}: {e}")
                test_acc_history.append(test_acc_history[-1] if test_acc_history else 0.0)
                test_loss_history.append(test_loss_history[-1] if test_loss_history else float('inf'))
        else:
            # Carry forward last known value for skipped rounds
            test_acc_history.append(test_acc_history[-1] if test_acc_history else 0.0)
            test_loss_history.append(test_loss_history[-1] if test_loss_history else float('inf'))


    # Get final accuracy with NaN/Inf protection - Replace with LARGE value (symmetry with defense.py, coordinator.py)
    final_accuracy = test_acc_history[-1] if test_acc_history else 1e10
    if math.isnan(final_accuracy) or math.isinf(final_accuracy):
        print(f"[WARN] Final accuracy is NaN/Inf, replacing with 1e10 (highly suspicious)")
        final_accuracy = 1e10

    # Calculate attack success rate with NaN/Inf handling
    attack_success_rate = 0.0
    if test_X is not None and y_test is not None:
        try:
            asr_results = coordinator.evaluate_attack_success(test_X, y_test)
            if isinstance(asr_results, dict):
                attack_success_rate = asr_results.get('attack_success_rate', 0.0)
            elif isinstance(asr_results, (list, tuple, np.ndarray)):
                # FIXED: Handle list of ASR values (returns ASR for each client)
                # Compute mean of all malicious clients (filter NaN/Inf and non-zero values)
                asr_values = [
                    float(asr) for asr in asr_results
                    if asr > 0 and not math.isnan(asr) and not math.isinf(asr)
                ]
                attack_success_rate = np.mean(asr_values) if len(asr_values) > 0 else 0.0
            elif isinstance(asr_results, (int, float)):
                attack_success_rate = float(asr_results)

            # Final NaN/Inf check - Replace with LARGE value (symmetry with defense.py, coordinator.py)
            if math.isnan(attack_success_rate) or math.isinf(attack_success_rate):
                print(f"[WARN] ASR is NaN/Inf, replacing with 1e10 (highly suspicious)")
                attack_success_rate = 1e10
        except Exception as e:
            print(f"[WARN] ASR calculation failed: {e}")
            attack_success_rate = 1e10  # Large value = calculation failed/suspicious

    # Get detection metrics for defenses with validation
    detection_metrics = {}
    if coordinator.use_defense:
        try:
            detection_metrics = coordinator.get_committee_metrics()

            # Validate detection metrics - Replace with LARGE values (symmetry with defense.py, coordinator.py)
            if detection_metrics:
                for key, value in detection_metrics.items():
                    if isinstance(value, (int, float)):
                        if math.isnan(value) or math.isinf(value):
                            print(f"[WARN] Detection metric '{key}' is NaN/Inf, replacing with 1e10 (highly suspicious)")
                            detection_metrics[key] = 1e10
                    elif isinstance(value, dict):
                        # Handle nested dict (e.g., confusion matrix)
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (int, float)):
                                if math.isnan(sub_value) or math.isinf(sub_value):
                                    print(f"[WARN] Detection metric '{key}.{sub_key}' is NaN/Inf, replacing with 1e10 (highly suspicious)")
                                    value[sub_key] = 1e10
        except Exception as e:
            print(f"[WARN] Failed to get detection metrics: {e}")
            detection_metrics = {}

    return final_accuracy, attack_success_rate, detection_metrics, test_acc_history, test_loss_history


# =============================================================================
# SHARED INNER-LOOP HELPER  (CMFL-paper style)
# =============================================================================
# All 7 aggregation schemes are treated on equal footing.
# For each dataset × attack the poisoned clients are built ONCE, then
# deep-copied into every scheme run by test_defense_on_scenario.
#
# Result layout (stored per dataset × attack_type):
#   {
#     "<SCHEME>": {
#       "baseline":  { final_accuracy, test_acc_history, test_loss_history },
#       "attack":    { final_accuracy, attack_success_rate, detection_metrics,
#                      test_acc_history, test_loss_history }
#     },
#     ...
#   }
# =============================================================================

# Canonical list of aggregation schemes (plain + committee).
ALL_AGG_SCHEMES = ['fedavg', 'krum', 'multi_krum', 'median', 'trimmed_mean',
                   'cmfl', 'cmfl_ii']

# ---------------------------------------------------------------------------
# Attack categories — controls which attacks run in each ablation study.
#
# DATA_POISONING_ATTACKS  : poison training data (labels or triggers).
#                           Respond to poison_percentage ablation.
# BYZANTINE_ATTACKS       : manipulate model updates (gradient-level).
#                           Do NOT respond to poison_percentage (model-level only).
# ALL_ATTACKS             : union of both — used in ablations where the
#                           varied parameter affects all attack types.
# ---------------------------------------------------------------------------
# NOT IN CMFL PAPER — paper only uses Byzantine attacks (Section 6.3)
# DATA_POISONING_ATTACKS = ['slf', 'dlf',
#                           'centralized', 'coordinated', 'random', 'model_dependent']
DATA_POISONING_ATTACKS = []  # Empty — paper uses no data-poisoning attacks
BYZANTINE_ATTACKS      = ['gradient_scaling', 'same_value', 'back_gradient']
ALL_ATTACKS            = BYZANTINE_ATTACKS  # Paper: only Byzantine attacks

# Paper-exact round counts per dataset (arXiv:2108.00365v2)
DATASET_ROUNDS = {'FEMNIST': 600, 'Shakespeare': 500, 'Sentiment140': 1000}
DATASET_LR = {'FEMNIST': 0.01, 'Sentiment140': 0.005, 'Shakespeare': 0.001}
DATASET_BATCH_SIZE = {'FEMNIST': 32, 'Sentiment140': 32, 'Shakespeare': 32}


def _run_baselines(
    framework,
    dataset: str,
    agg_schemes: List[str],
    rounds: int,
    n_clients: int,
    clients_per_round: int,
    alpha: float,
    dataset_fraction: float,
    aggregation_participation_frac: float = None,
    committee_size_frac: float = 0.40,
    scoring_mode: str = 'distance_only',
    selection_strategy: int = 1,
    lr: float = None
) -> dict:
    """Run baseline (clean clients) for every scheme in *agg_schemes* once.
    Returns  {SCHEME_KEY: (final_acc, acc_history, loss_history)}."""
    baselines = {}
    print(f"\n  [BASELINES] Running once per scheme...")
    for scheme in agg_schemes:
        scheme_key = scheme.upper()
        print(f"\n    [{scheme_key}] ─── Baseline ───")
        try:
            baseline_acc, bl_acc_hist, bl_loss_hist = run_baseline_evaluation(
                framework=framework,
                dataset_name=dataset,
                num_clients=n_clients,
                rounds=rounds,
                clients_per_round=clients_per_round,
                aggregation=scheme,
                alpha=alpha,
                dataset_fraction=dataset_fraction,
                aggregation_participation_frac=aggregation_participation_frac,
                committee_size_frac=committee_size_frac,
                scoring_mode=scoring_mode,
                selection_strategy=selection_strategy,
                lr=lr
            )
            print(f"    [{scheme_key}] Baseline Acc: {baseline_acc:.4f}")
            baselines[scheme_key] = (baseline_acc, bl_acc_hist, bl_loss_hist)
        except Exception as e:
            print(f"    [{scheme_key}] Baseline ❌: {e}")
            baselines[scheme_key] = (0.0, [], [])
    return baselines


def _run_per_scheme(
    framework,
    dataset: str,
    agg_schemes: List[str],
    attacks_to_test: List[str],
    rounds: int,
    n_clients: int,
    num_malicious: int,
    clients_per_round: int,
    poison_percentage: float,
    alpha: float,
    dataset_fraction: float,
    aggregation_participation_frac: float = None,
    committee_size_frac: float = 0.40,
    precomputed_baselines: dict = None,
    scoring_mode: str = 'distance_only',
    selection_strategy: int = 1,
    lr: float = None
) -> dict:
    """
    CMFL-paper-style evaluation for ONE dataset.

    For every aggregation scheme in *agg_schemes*:
        1. Run baseline (clean clients, that scheme).
        2. For every attack in *attacks_to_test*:
            - Build poisoned clients once.
            - Run attack scenario under that scheme.

    Returns a nested dict:  { attack_type: { scheme: { baseline: {…}, attack: {…} } } }
    """
    # Dataset-specific learning rates and batch sizes
    if lr is None:
        lr = DATASET_LR.get(dataset, 0.001)
    _bs = DATASET_BATCH_SIZE.get(dataset, 32)

    dataset_results = {}

    # ---------- Phase 1: baselines — run fresh or reuse pre-computed ----------
    if precomputed_baselines is not None:
        baselines = precomputed_baselines
        print(f"\n  [BASELINES] Reusing {len(baselines)} cached baselines...")
    else:
        baselines = _run_baselines(
            framework=framework, dataset=dataset,
            agg_schemes=agg_schemes, rounds=rounds,
            n_clients=n_clients, clients_per_round=clients_per_round,
            alpha=alpha, dataset_fraction=dataset_fraction,
            aggregation_participation_frac=aggregation_participation_frac,
            committee_size_frac=committee_size_frac,
            scoring_mode=scoring_mode,
            selection_strategy=selection_strategy
        )

    # ---------- Phase 2: attacks, reusing cached baselines ----------
    for attack_type in attacks_to_test:
        print(f"\n  [ATTACK] {attack_type.upper()}")
        dataset_results[attack_type] = {}

        # ---------- build poisoned clients ONCE per attack ----------
        print(f"    Creating attack scenario (poisoned clients)...")
        try:
            clients, test_loader, test_X, y_test = create_attack_scenario(
                framework=framework,
                dataset_name=dataset,
                attack_type=attack_type,
                poison_percentage=poison_percentage,
                num_clients=n_clients,
                malicious_clients=num_malicious,
                alpha=alpha,
                dataset_fraction=dataset_fraction
            )
        except Exception as e:
            print(f"    ❌ Failed to create attack scenario: {e}")
            import traceback; traceback.print_exc()
            continue

        if clients is None:
            print(f"    ❌ Attack scenario returned None — skipping")
            continue

        # Snapshot client state ONCE so we can cheaply reset between schemes
        for client in clients:
            client.snapshot_state()

        # ---------- loop over every aggregation scheme ----------
        for scheme in agg_schemes:
            scheme_key = scheme.upper()
            baseline_acc, bl_acc_hist, bl_loss_hist = baselines[scheme_key]

            # -- Attack (poisoned clients, this scheme) --
            print(f"    [{scheme_key}] ─── Attack ───")
            try:
                atk_acc, atk_asr, atk_metrics, atk_acc_hist, atk_loss_hist = test_defense_on_scenario(
                    clients_list=clients,
                    test_loader=test_loader,
                    test_X=test_X,
                    y_test=y_test,
                    rounds=rounds,
                    aggregation=scheme,
                    clients_per_round=clients_per_round,
                    aggregation_participation_frac=aggregation_participation_frac,
                    committee_size_frac=committee_size_frac,
                    scoring_mode=scoring_mode,
                    selection_strategy=selection_strategy,
                    lr=lr, batch_size=_bs
                )
                print(f"    [{scheme_key}] Attack Acc: {atk_acc:.4f}, ASR: {atk_asr:.4f}")
            except Exception as e:
                print(f"    [{scheme_key}] Attack ❌: {e}")
                import traceback; traceback.print_exc()
                atk_acc, atk_asr, atk_metrics = 0.0, 0.0, {}
                atk_acc_hist, atk_loss_hist = [], []

            dataset_results[attack_type][scheme_key] = {
                'baseline': {
                    'final_accuracy':      baseline_acc,
                    'test_acc_history':    bl_acc_hist,
                    'test_loss_history':   bl_loss_hist
                },
                'attack': {
                    'final_accuracy':           atk_acc,
                    'attack_success_rate':      atk_asr,
                    'detection_metrics':        atk_metrics,
                    'test_acc_history':         atk_acc_hist,
                    'test_loss_history':        atk_loss_hist
                }
            }

        print(f"    ✅ Completed {attack_type.upper()} (all {len(agg_schemes)} schemes)")

    return dataset_results


# =============================================================================
# ABLATION STUDY 6: DATA POISON PERCENTAGE
# =============================================================================

def run_data_poison_percentage_ablation(
    percentages: Optional[List[float]] = None,
    output_base: str = None,
    datasets: Optional[List[str]] = None,
    rounds: int = None,
    total_clients: Union[int, Dict[str, int]] = 100,
    client_participation: int = 40,
    malicious_ratio: float = 0.4,
    aggregation: str = "fedavg",          # kept for back-compat; ignored
    attacks_to_test: Optional[List[str]] = None,
    agg_schemes: Optional[List[str]] = None,
    defenses_to_test: Optional[List[str]] = None,  # back-compat; ignored
    dataset_fraction: float = 1.0,
    aggregation_participation_frac: float = None,
    run_id: str = None,
    selection_strategy: int = 1
):
    """Ablation 1 — data poison percentage.
    NOT IN CMFL PAPER — paper only uses Byzantine attacks which don't poison data.
    Commented out to match paper exactly. Keeping function signature for import compatibility."""
    # NOT IN CMFL PAPER — commented out entire body
    print("[SKIP] Data poison percentage ablation — NOT in CMFL paper (paper uses only Byzantine attacks)")
    return {}


# =============================================================================
# ABLATION STUDY 1: CLIENT MALICIOUS RATIO
# =============================================================================

def run_client_malicious_ratio_ablation(
    ratios: Optional[List[float]] = None,
    output_base: str = None,
    datasets: Optional[List[str]] = None,
    rounds: int = None,
    total_clients: Union[int, Dict[str, int]] = 100,
    client_participation: int = 40,
    poison_percentage: float = 0.4,
    aggregation: str = "fedavg",          # kept for back-compat; ignored
    attacks_to_test: Optional[List[str]] = None,
    agg_schemes: Optional[List[str]] = None,
    defenses_to_test: Optional[List[str]] = None,  # back-compat; ignored
    dataset_fraction: float = 1.0,
    aggregation_participation_frac: float = None,
    run_id: str = None,
    selection_strategy: int = 1
):
    """Ablation 2 — client malicious ratio.  CMFL-paper style: every scheme
    (FedAvg, Krum, Multi-Krum, Median, Trimmed Mean, CMFL)
    is tested under both baseline and attack for each client malicious ratio.

    ALL attacks run by default: both data-poisoning and Byzantine attacks
    are affected by the number of malicious clients."""

    if ratios is None:
        ratios = [0.2, 0.4, 0.6, 0.8]
    if datasets is None:
        datasets = ['FEMNIST', 'Shakespeare', 'Sentiment140']
    if attacks_to_test is None:
        attacks_to_test = ALL_ATTACKS  # All attacks respond to malicious_ratio
    if agg_schemes is None:
        agg_schemes = ALL_AGG_SCHEMES
    if run_id is None:
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    if output_base is None:
        output_base = f"Output/table3_runs/{run_id}/ablation_client_malicious_ratio"
    else:
        output_base = f"{output_base}/{run_id}/ablation_client_malicious_ratio"

    print("\n" + "="*100)
    print("ABLATION STUDY 1: CLIENT MALICIOUS RATIO")
    print("="*100)
    print(f"Run ID: {run_id}")
    print(f"Output: {output_base}")
    print(f"Client malicious ratios: {[f'{r*100:.0f}%' for r in ratios]}")
    print(f"Aggregation schemes    : {[s.upper() for s in agg_schemes]}")
    print(f"Attacks                : {[a.upper() for a in attacks_to_test]}")
    print(f"Datasets               : {datasets}")
    print(f"Rounds                 : {rounds if rounds is not None else 'per-dataset (DATASET_ROUNDS)'}")
    print(f"Learning rate          : {DATASET_LR}")
    print(f"Batch size             : {DATASET_BATCH_SIZE}")
    print(f"Fixed data poison pct  : {poison_percentage*100:.0f}%")
    print("="*100)

    framework = EvaluationFramework(out_dir=output_base)
    results = {f"{int(r*100)}%": {} for r in ratios}

    # ---------- cache baselines per dataset (independent of client_malicious_ratio) ----------
    baseline_cache = {}
    for dataset in datasets:
        _rounds = rounds if rounds is not None else DATASET_ROUNDS.get(dataset, 200)
        n_clients = _resolve_clients(total_clients, dataset)
        print(f"\n[DATASET] {dataset} — running baselines once (reused across all client malicious ratios)...")
        baseline_cache[dataset] = _run_baselines(
            framework=framework, dataset=dataset,
            agg_schemes=agg_schemes, rounds=_rounds,
            n_clients=n_clients, clients_per_round=client_participation,
            alpha=1.0, dataset_fraction=dataset_fraction,
            aggregation_participation_frac=aggregation_participation_frac,
            selection_strategy=selection_strategy
        )

    # ---------- sweep client malicious ratios, reusing cached baselines ----------
    for ratio in ratios:
        ratio_pct = int(ratio * 100)
        print(f"\n{'='*80}\nRunning with {ratio_pct}% client malicious ratio\n{'='*80}")

        for dataset in datasets:
            _rounds = rounds if rounds is not None else DATASET_ROUNDS.get(dataset, 200)
            n_clients     = _resolve_clients(total_clients, dataset)
            num_malicious = int(n_clients * ratio)
            print(f"\n[DATASET] {dataset} (clients={n_clients}, malicious={num_malicious})")
            Path(f"{output_base}/malicious_{ratio_pct}pct/{dataset}").mkdir(parents=True, exist_ok=True)

            results[f"{ratio_pct}%"][dataset] = _run_per_scheme(
                framework=framework, dataset=dataset,
                agg_schemes=agg_schemes, attacks_to_test=attacks_to_test,
                rounds=_rounds, n_clients=n_clients, num_malicious=num_malicious,
                clients_per_round=client_participation,
                poison_percentage=poison_percentage, alpha=1.0,
                dataset_fraction=dataset_fraction,
                aggregation_participation_frac=aggregation_participation_frac,
                precomputed_baselines=baseline_cache[dataset],
                selection_strategy=selection_strategy
            )

    # Save results
    results_dir = Path(output_base) / "aggregated_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"client_malicious_ratio_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*100}\n✅ Client malicious ratio ablation completed\nResults saved to: {results_file}\n{'='*100}")
    return results


# =============================================================================
# ABLATION STUDY 8: NON-IID DATA DISTRIBUTION
# =============================================================================

def run_non_iid_ablation(
    non_iid_levels: Optional[List] = None,
    output_base: str = None,
    datasets: Optional[List[str]] = None,
    rounds: int = None,
    total_clients: Union[int, Dict[str, int]] = 100,
    client_participation: int = 40,
    malicious_ratio: float = 0.4,
    poison_percentage: float = 0.4,
    aggregation: str = "fedavg",          # kept for back-compat; ignored
    attacks_to_test: Optional[List[str]] = None,
    agg_schemes: Optional[List[str]] = None,
    defenses_to_test: Optional[List[str]] = None,  # back-compat; ignored
    dataset_fraction: float = 1.0,
    aggregation_participation_frac: float = None,
    run_id: str = None,
    selection_strategy: int = 1
):
    """Ablation 3 — Non-IID data distribution.

    Tests with different Non-IID levels (Dirichlet alpha):
        0%  = IID (alpha = 1000.0)
        20% = Slightly Non-IID (alpha = 1.0)
        40% = Moderately Non-IID (alpha = 0.5)
        60% = Highly Non-IID (alpha = 0.3)
        80% = Extremely Non-IID (alpha = 0.1)

    CMFL-paper style: every scheme is tested under both baseline and attack
    for each Non-IID level."""

    if non_iid_levels is None:
        non_iid_levels = [
            ('0%',  1000.0),   # IID
            ('20%', 1.0),      # Slightly Non-IID
            ('40%', 0.5),      # Moderately Non-IID
            ('60%', 0.3),      # Highly Non-IID
            ('80%', 0.1),      # Extremely Non-IID
        ]
    if datasets is None:
        datasets = ['FEMNIST', 'Shakespeare', 'Sentiment140']
    if attacks_to_test is None:
        attacks_to_test = ALL_ATTACKS
    if agg_schemes is None:
        agg_schemes = ALL_AGG_SCHEMES
    if run_id is None:
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    if output_base is None:
        output_base = f"Output/table3_runs/{run_id}/ablation_non_iid"
    else:
        output_base = f"{output_base}/{run_id}/ablation_non_iid"

    print("\n" + "="*100)
    print("ABLATION STUDY 8: NON-IID DATA DISTRIBUTION")
    print("="*100)
    print(f"Run ID: {run_id}")
    print(f"Output: {output_base}")
    print(f"Non-IID levels         : {[level[0] for level in non_iid_levels]}")
    print(f"Aggregation schemes    : {[s.upper() for s in agg_schemes]}")
    print(f"Attacks                : {[a.upper() for a in attacks_to_test]}")
    print(f"Datasets               : {datasets}")
    print(f"Rounds                 : {rounds if rounds is not None else 'per-dataset (DATASET_ROUNDS)'}")
    print(f"Learning rate          : {DATASET_LR}")
    print(f"Batch size             : {DATASET_BATCH_SIZE}")
    print(f"Fixed malicious ratio  : {malicious_ratio*100:.0f}%")
    print(f"Fixed data poison pct  : {poison_percentage*100:.0f}%")
    print("="*100)

    framework = EvaluationFramework(out_dir=output_base)
    results = {level_name: {} for level_name, _ in non_iid_levels}

    for level_name, alpha in non_iid_levels:
        print(f"\n{'='*80}")
        print(f"Running with {level_name} Non-IID (alpha={alpha})")
        print(f"{'='*80}")

        for dataset in datasets:
            _rounds = rounds if rounds is not None else DATASET_ROUNDS.get(dataset, 200)
            n_clients = _resolve_clients(total_clients, dataset)
            num_malicious = int(n_clients * malicious_ratio)
            print(f"\n[DATASET] {dataset} (clients={n_clients}, malicious={num_malicious})")
            Path(f"{output_base}/noniid_{level_name.replace('%', 'pct')}/{dataset}").mkdir(parents=True, exist_ok=True)

            # Baselines need to be re-run for each Non-IID level (alpha changes data distribution)
            results[level_name][dataset] = _run_per_scheme(
                framework=framework, dataset=dataset,
                agg_schemes=agg_schemes, attacks_to_test=attacks_to_test,
                rounds=_rounds, n_clients=n_clients, num_malicious=num_malicious,
                clients_per_round=client_participation,
                poison_percentage=poison_percentage, alpha=alpha,
                dataset_fraction=dataset_fraction,
                aggregation_participation_frac=aggregation_participation_frac,
                selection_strategy=selection_strategy
            )

    # Save results
    results_dir = Path(output_base) / "aggregated_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"non_iid_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*100}\n✅ Non-IID ablation completed\nResults saved to: {results_file}\n{'='*100}")
    return results


# =============================================================================
# ABLATION STUDY 7: CLIENT PARTICIPATION RATE
# =============================================================================

def run_client_participation_ablation(
    participation_rates: Optional[List[float]] = None,
    output_base: str = None,
    datasets: Optional[List[str]] = None,
    rounds: int = None,
    total_clients: Union[int, Dict[str, int]] = 100,
    malicious_ratio: float = 0.4,
    poison_percentage: float = 0.4,
    aggregation: str = "fedavg",          # kept for back-compat; ignored
    attacks_to_test: Optional[List[str]] = None,
    agg_schemes: Optional[List[str]] = None,
    defenses_to_test: Optional[List[str]] = None,  # back-compat; ignored
    dataset_fraction: float = 1.0,
    aggregation_participation_frac: float = None,
    run_id: str = None,
    selection_strategy: int = 1
):
    """Ablation 4 — client participation rate per round.

    Tests with different participation rates: 10%, 20%, 30%, 40%, 50%
    The number of clients participating per round = total_clients * rate.

    CMFL-paper style: every scheme is tested under both baseline and attack
    for each participation rate."""

    if participation_rates is None:
        participation_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    if datasets is None:
        datasets = ['FEMNIST', 'Shakespeare', 'Sentiment140']
    if attacks_to_test is None:
        attacks_to_test = ALL_ATTACKS
    if agg_schemes is None:
        agg_schemes = ALL_AGG_SCHEMES
    if run_id is None:
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    if output_base is None:
        output_base = f"Output/table3_runs/{run_id}/ablation_client_participation"
    else:
        output_base = f"{output_base}/{run_id}/ablation_client_participation"

    print("\n" + "="*100)
    print("ABLATION STUDY 7: CLIENT PARTICIPATION RATE")
    print("="*100)
    print(f"Run ID: {run_id}")
    print(f"Output: {output_base}")
    print(f"Participation rates    : {[f'{p*100:.0f}%' for p in participation_rates]}")
    print(f"Aggregation schemes    : {[s.upper() for s in agg_schemes]}")
    print(f"Attacks                : {[a.upper() for a in attacks_to_test]}")
    print(f"Datasets               : {datasets}")
    print(f"Rounds                 : {rounds if rounds is not None else 'per-dataset (DATASET_ROUNDS)'}")
    print(f"Learning rate          : {DATASET_LR}")
    print(f"Batch size             : {DATASET_BATCH_SIZE}")
    print(f"Fixed malicious ratio  : {malicious_ratio*100:.0f}%")
    print(f"Fixed data poison pct  : {poison_percentage*100:.0f}%")
    print("="*100)

    framework = EvaluationFramework(out_dir=output_base)
    results = {f"{int(r*100)}%": {} for r in participation_rates}

    for rate in participation_rates:
        rate_pct = int(rate * 100)

        print(f"\n{'='*80}")
        print(f"Running with {rate_pct}% client participation")
        print(f"{'='*80}")

        for dataset in datasets:
            _rounds = rounds if rounds is not None else DATASET_ROUNDS.get(dataset, 200)
            n_clients = _resolve_clients(total_clients, dataset)
            num_malicious = int(n_clients * malicious_ratio)
            clients_per_round = max(1, int(n_clients * rate))
            print(f"\n[DATASET] {dataset} (clients={n_clients}, participating={clients_per_round}, malicious={num_malicious})")
            Path(f"{output_base}/participation_{rate_pct}pct/{dataset}").mkdir(parents=True, exist_ok=True)

            # Baselines need to be re-run for each participation rate (clients_per_round changes)
            results[f"{rate_pct}%"][dataset] = _run_per_scheme(
                framework=framework, dataset=dataset,
                agg_schemes=agg_schemes, attacks_to_test=attacks_to_test,
                rounds=_rounds, n_clients=n_clients, num_malicious=num_malicious,
                clients_per_round=clients_per_round,
                poison_percentage=poison_percentage, alpha=1.0,
                dataset_fraction=dataset_fraction,
                aggregation_participation_frac=aggregation_participation_frac,
                selection_strategy=selection_strategy
            )

    # Save results
    results_dir = Path(output_base) / "aggregated_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"client_participation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*100}\n✅ Client participation ablation completed\nResults saved to: {results_file}\n{'='*100}")
    return results


# =============================================================================
# ABLATION STUDY 2: AGGREGATION PARTICIPATION FRACTION (CMFL)
# =============================================================================

def run_aggregation_participation_ablation(
    participation_fracs: Optional[List[float]] = None,
    output_base: str = None,
    datasets: Optional[List[str]] = None,
    rounds: int = None,
    total_clients: Union[int, Dict[str, int]] = 100,
    client_participation: int = 40,
    malicious_ratio: float = 0.4,
    poison_percentage: float = 0.4,
    attacks_to_test: Optional[List[str]] = None,
    agg_schemes: Optional[List[str]] = None,
    dataset_fraction: float = 1.0,
    run_id: str = None,
    selection_strategy: int = 1
):
    """Ablation 5 — aggregation participation fraction inside CMFL.
    ``aggregation_participation_frac`` controls the fraction of
    training clients (sorted by anomaly score, least-suspicious first) that are
    included in the final model aggregation.  Plain schemes (FedAvg, Krum, …)
    are unaffected by this parameter and are therefore excluded from this sweep.
    Each level is evaluated under both Baseline and Attack for every attack type,
    using CMFL as the aggregation scheme.

    ALL attacks run by default: aggregation participation affects how many
    client updates get included, impacting both data-poisoning and Byzantine attacks."""

    _COMMITTEE_ONLY = ['cmfl', 'cmfl_ii']

    if participation_fracs is None:
        participation_fracs = [None, 0.1, 0.2, 0.3, 0.4, 0.5]  # None = original pBFT consensus
    if datasets is None:
        datasets = ['FEMNIST', 'Shakespeare', 'Sentiment140']
    if attacks_to_test is None:
        attacks_to_test = ALL_ATTACKS  # All attacks respond to aggregation participation
    if agg_schemes is None:
        agg_schemes = _COMMITTEE_ONLY
    else:
        # Only committee schemes are affected — filter silently
        agg_schemes = [s for s in agg_schemes if s.lower() in {'cmfl', 'cmfl_ii'}]
        if not agg_schemes:
            agg_schemes = _COMMITTEE_ONLY
    if run_id is None:
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    if output_base is None:
        output_base = f"Output/table3_runs/{run_id}/ablation_agg_participation"
    else:
        output_base = f"{output_base}/{run_id}/ablation_agg_participation"

    print("\n" + "="*100)
    print("ABLATION STUDY 2: AGGREGATION PARTICIPATION FRACTION (CMFL)")
    print("="*100)
    print(f"Run ID: {run_id}")
    print(f"Output: {output_base}")
    print(f"Participation fracs    : {['pBFT (original)' if f is None else f'{f*100:.0f}%' for f in participation_fracs]}")
    print(f"Aggregation schemes    : {[s.upper() for s in agg_schemes]}")
    print(f"Attacks                : {[a.upper() for a in attacks_to_test]}")
    print(f"Datasets               : {datasets}")
    print(f"Rounds                 : {rounds if rounds is not None else 'per-dataset (DATASET_ROUNDS)'}")
    print(f"Learning rate          : {DATASET_LR}")
    print(f"Batch size             : {DATASET_BATCH_SIZE}")
    print(f"Fixed client malicious ratio: {malicious_ratio*100:.0f}%")
    print(f"Fixed data poison pct      : {poison_percentage*100:.0f}%")
    print("="*100)

    framework = EvaluationFramework(out_dir=output_base)
    results = {}

    for frac in participation_fracs:
        # None → original pBFT consensus path; otherwise fixed-fraction selection
        label   = "pBFT" if frac is None else f"{int(frac * 100)}%"
        dir_tag = "pBFT" if frac is None else f"{int(frac * 100)}pct"
        print(f"\n{'='*80}\nRunning with {label} aggregation participation\n{'='*80}")
        results[label] = {}

        for dataset in datasets:
            _rounds = rounds if rounds is not None else DATASET_ROUNDS.get(dataset, 200)
            n_clients     = _resolve_clients(total_clients, dataset)
            num_malicious = int(n_clients * malicious_ratio)
            print(f"\n[DATASET] {dataset} (clients={n_clients}, malicious={num_malicious})")
            Path(f"{output_base}/agg_participation_{dir_tag}/{dataset}").mkdir(parents=True, exist_ok=True)

            results[label][dataset] = _run_per_scheme(
                framework=framework, dataset=dataset,
                agg_schemes=agg_schemes, attacks_to_test=attacks_to_test,
                rounds=_rounds, n_clients=n_clients, num_malicious=num_malicious,
                clients_per_round=client_participation,
                poison_percentage=poison_percentage, alpha=1.0,
                dataset_fraction=dataset_fraction,
                aggregation_participation_frac=frac,
                selection_strategy=selection_strategy
            )

    # Save results
    results_dir = Path(output_base) / "aggregated_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"agg_participation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*100}\n✅ Aggregation participation ablation completed\nResults saved to: {results_file}\n{'='*100}")
    return results


# =============================================================================
# ABLATION STUDY 3: COMMITTEE SIZE (CMFL)
# =============================================================================

def run_committee_size_ablation(
    committee_size_fracs: Optional[List[float]] = None,
    output_base: str = None,
    datasets: Optional[List[str]] = None,
    rounds: int = None,
    total_clients: Union[int, Dict[str, int]] = 100,
    client_participation: int = 40,
    malicious_ratio: float = 0.4,
    poison_percentage: float = 0.4,
    attacks_to_test: Optional[List[str]] = None,
    agg_schemes: Optional[List[str]] = None,
    dataset_fraction: float = 1.0,
    run_id: str = None,
    selection_strategy: int = 1
):
    """Ablation 6 — committee size as a fraction of clients_per_round.
    ``committee_size_frac`` controls how many of the participating clients form
    the validation committee; the remainder become training clients.  Only
    CMFL is tested — plain schemes have no committee.
    Each level is evaluated under both Baseline and Attack for every attack type.

    ALL attacks run by default: committee size affects defense detection
    capability, impacting both data-poisoning and Byzantine attacks."""

    _COMMITTEE_ONLY = ['cmfl', 'cmfl_ii']

    if committee_size_fracs is None:
        committee_size_fracs = [0.1, 0.2, 0.3, 0.4, 0.5]
    if datasets is None:
        datasets = ['FEMNIST', 'Shakespeare', 'Sentiment140']
    if attacks_to_test is None:
        attacks_to_test = ALL_ATTACKS  # All attacks respond to committee size
    if agg_schemes is None:
        agg_schemes = _COMMITTEE_ONLY
    else:
        agg_schemes = [s for s in agg_schemes if s.lower() in {'cmfl', 'cmfl_ii'}]
        if not agg_schemes:
            agg_schemes = _COMMITTEE_ONLY
    if run_id is None:
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    if output_base is None:
        output_base = f"Output/table3_runs/{run_id}/ablation_committee_size"
    else:
        output_base = f"{output_base}/{run_id}/ablation_committee_size"

    print("\n" + "="*100)
    print("ABLATION STUDY 3: COMMITTEE SIZE (CMFL)")
    print("="*100)
    print(f"Run ID: {run_id}")
    print(f"Output: {output_base}")
    print(f"Committee size fracs   : {[f'{f*100:.0f}%' for f in committee_size_fracs]}")
    print(f"Aggregation schemes    : {[s.upper() for s in agg_schemes]}")
    print(f"Attacks                : {[a.upper() for a in attacks_to_test]}")
    print(f"Datasets               : {datasets}")
    print(f"Rounds                 : {rounds if rounds is not None else 'per-dataset (DATASET_ROUNDS)'}")
    print(f"Learning rate          : {DATASET_LR}")
    print(f"Batch size             : {DATASET_BATCH_SIZE}")
    print(f"Fixed client malicious ratio: {malicious_ratio*100:.0f}%")
    print(f"Fixed data poison pct      : {poison_percentage*100:.0f}%")
    print("="*100)

    framework = EvaluationFramework(out_dir=output_base)
    results = {}

    for frac in committee_size_fracs:
        frac_pct = int(frac * 100)
        print(f"\n{'='*80}\nRunning with {frac_pct}% committee size\n{'='*80}")
        results[f"{frac_pct}%"] = {}

        for dataset in datasets:
            _rounds = rounds if rounds is not None else DATASET_ROUNDS.get(dataset, 200)
            n_clients     = _resolve_clients(total_clients, dataset)
            num_malicious = int(n_clients * malicious_ratio)
            print(f"\n[DATASET] {dataset} (clients={n_clients}, malicious={num_malicious})")
            Path(f"{output_base}/committee_{frac_pct}pct/{dataset}").mkdir(parents=True, exist_ok=True)

            results[f"{frac_pct}%"][dataset] = _run_per_scheme(
                framework=framework, dataset=dataset,
                agg_schemes=agg_schemes, attacks_to_test=attacks_to_test,
                rounds=_rounds, n_clients=n_clients, num_malicious=num_malicious,
                clients_per_round=client_participation,
                poison_percentage=poison_percentage, alpha=1.0,
                dataset_fraction=dataset_fraction,
                committee_size_frac=frac,
                selection_strategy=selection_strategy
            )

    # Save results
    results_dir = Path(output_base) / "aggregated_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"committee_size_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*100}\n✅ Committee size ablation completed\nResults saved to: {results_file}\n{'='*100}")
    return results


# =============================================================================
# ABLATION STUDY 4: COMMITTEE MEMBER MALICIOUS TRACKING
# =============================================================================

def run_committee_malicious_tracking_ablation(
    ratios: Optional[List[float]] = None,
    output_base: str = None,
    datasets: Optional[List[str]] = None,
    rounds: int = None,
    total_clients: Union[int, Dict[str, int]] = 100,
    client_participation: int = 20,
    poison_percentage: float = 0.4,
    attacks_to_test: Optional[List[str]] = None,
    committee_size_frac: float = 0.40,
    aggregation_participation_frac: float = 0.40,
    dataset_fraction: float = 1.0,
    run_id: str = None,
    selection_strategy: int = 1
):
    """Ablation 7 — Committee member malicious tracking.

    Tracks how many malicious clients end up in the CMFL committee across
    rounds as the malicious ratio varies.  This measures the defense's ability
    to keep malicious clients OUT of the trusted validation committee.

    For each malicious ratio, runs CMFL Strategy I and II with every attack
    and records:
      - Per-round committee composition and malicious overlap
      - Aggregate stats: avg/max malicious fraction in committee
      - Fraction of rounds where at least one malicious client is in committee
      - Standard detection metrics and accuracy for context

    Only CMFL/CMFL_II schemes are tested (plain schemes have no committee).
    """
    import math
    from coordinator import DecentralizedFLCoordinator

    _COMMITTEE_ONLY = ['cmfl', 'cmfl_ii']

    if ratios is None:
        ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    if datasets is None:
        datasets = ['FEMNIST', 'Shakespeare', 'Sentiment140']
    if attacks_to_test is None:
        attacks_to_test = ALL_ATTACKS
    if run_id is None:
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    if output_base is None:
        output_base = f"Output/table3_runs/{run_id}/ablation_committee_malicious_tracking"
    else:
        output_base = f"{output_base}/{run_id}/ablation_committee_malicious_tracking"

    print("\n" + "=" * 100)
    print("ABLATION STUDY 4: COMMITTEE MEMBER MALICIOUS TRACKING")
    print("=" * 100)
    print(f"Run ID: {run_id}")
    print(f"Output: {output_base}")
    print(f"Malicious ratios       : {[f'{r*100:.0f}%' for r in ratios]}")
    print(f"Aggregation schemes    : {[s.upper() for s in _COMMITTEE_ONLY]}")
    print(f"Attacks                : {[a.upper() for a in attacks_to_test]}")
    print(f"Datasets               : {datasets}")
    print(f"Rounds                 : {rounds if rounds is not None else 'per-dataset (DATASET_ROUNDS)'}")
    print(f"Learning rate          : {DATASET_LR}")
    print(f"Batch size             : {DATASET_BATCH_SIZE}")
    print(f"Committee size frac    : {committee_size_frac*100:.0f}%")
    print(f"Agg participation frac : {aggregation_participation_frac*100:.0f}%")
    print(f"Fixed data poison pct  : {poison_percentage*100:.0f}%")
    print("=" * 100)

    framework = EvaluationFramework(out_dir=output_base)
    results = {}

    for ratio in ratios:
        ratio_pct = int(ratio * 100)
        print(f"\n{'=' * 80}\nMalicious ratio: {ratio_pct}%\n{'=' * 80}")
        results[f"{ratio_pct}%"] = {}

        for dataset in datasets:
            _rounds = rounds if rounds is not None else DATASET_ROUNDS.get(dataset, 200)
            n_clients = _resolve_clients(total_clients, dataset)
            num_malicious = int(n_clients * ratio)
            clients_per_round = client_participation if isinstance(client_participation, int) else max(1, int(n_clients * 0.2))
            print(f"\n[DATASET] {dataset} (clients={n_clients}, malicious={num_malicious})")
            Path(f"{output_base}/malicious_{ratio_pct}pct/{dataset}").mkdir(parents=True, exist_ok=True)

            dataset_attack_results = {}

            # ── Baselines (clean, no attack) for CMFL schemes ──
            baselines = _run_baselines(
                framework=framework, dataset=dataset,
                agg_schemes=_COMMITTEE_ONLY, rounds=_rounds,
                n_clients=n_clients, clients_per_round=clients_per_round,
                alpha=1.0, dataset_fraction=dataset_fraction,
                aggregation_participation_frac=aggregation_participation_frac,
                committee_size_frac=committee_size_frac,
                selection_strategy=selection_strategy
            )

            # ── Attack scenarios ──
            for attack_type in attacks_to_test:
                print(f"\n  [ATTACK] {attack_type.upper()}")
                dataset_attack_results[attack_type] = {}

                # Build poisoned clients ONCE per attack
                try:
                    clients, test_loader, test_X, y_test = create_attack_scenario(
                        framework=framework, dataset_name=dataset,
                        attack_type=attack_type,
                        poison_percentage=poison_percentage,
                        num_clients=n_clients,
                        malicious_clients=num_malicious,
                        alpha=1.0, dataset_fraction=dataset_fraction
                    )
                except Exception as e:
                    print(f"    Failed to create attack scenario: {e}")
                    import traceback; traceback.print_exc()
                    continue

                if clients is None:
                    print(f"    Attack scenario returned None — skipping")
                    continue

                # Snapshot for reset between schemes
                for c in clients:
                    c.snapshot_state()

                # ── Run each CMFL scheme ──
                for scheme in _COMMITTEE_ONLY:
                    scheme_key = scheme.upper()
                    baseline_acc, bl_acc_hist, bl_loss_hist = baselines[scheme_key]

                    print(f"    [{scheme_key}] Running attack + committee tracking...")

                    # Reset clients
                    for c in clients:
                        c.reset_for_new_run()

                    # Determine defense params
                    if scheme.lower() == 'cmfl_ii':
                        chosen_defense = 'cmfl'
                        sel_strat = 2
                    else:
                        chosen_defense = scheme.lower()
                        sel_strat = selection_strategy

                    cs = max(2, int(clients_per_round * committee_size_frac))
                    tc = max(4, clients_per_round - cs)

                    # Dataset-specific learning rate
                    _lr = DATASET_LR.get(dataset, 0.001)
                    _bs = DATASET_BATCH_SIZE.get(dataset, 32)

                    try:
                        coordinator = DecentralizedFLCoordinator(
                            clients, use_defense=True,
                            defense_type=chosen_defense,
                            committee_size=cs,
                            training_clients_per_round=tc,
                            aggregation_method='weighted_avg',
                            use_anomaly_detection=True,
                            clients_per_round=clients_per_round,
                            aggregation_participation_frac=aggregation_participation_frac,
                            selection_strategy=sel_strat,
                            lr=_lr, batch_size=_bs
                        )

                        test_acc_history = []
                        eval_interval = max(1, _rounds // 20) if _rounds > 20 else 1

                        for rnd in range(1, _rounds + 1):
                            coordinator.run_federated_learning(
                                rounds=1, aggregation_method=scheme,
                                test_loader=test_loader,
                                round_offset=rnd - 1, total_rounds=_rounds
                            )
                            if rnd % eval_interval == 0 or rnd == _rounds:
                                try:
                                    acc, _ = coordinator.evaluate_on_test_set(test_loader)
                                    if math.isnan(acc) or math.isinf(acc):
                                        acc = 1e10
                                    test_acc_history.append(float(acc))
                                except Exception:
                                    test_acc_history.append(test_acc_history[-1] if test_acc_history else 0.0)
                            else:
                                test_acc_history.append(test_acc_history[-1] if test_acc_history else 0.0)

                        final_accuracy = test_acc_history[-1] if test_acc_history else 0.0

                        # Get committee tracking metrics
                        committee_tracking = coordinator.get_malicious_in_committee_metrics()
                        detection_metrics = coordinator.get_committee_metrics()

                        # Strip per_round details for JSON (keep summary stats)
                        per_round_summary = []
                        for r in committee_tracking.get('per_round', []):
                            per_round_summary.append({
                                'round': r['round'],
                                'committee_size': r['committee_size'],
                                'malicious_count': r['malicious_count'],
                                'malicious_fraction': r['malicious_fraction'],
                            })

                        print(f"    [{scheme_key}] Acc={final_accuracy:.4f}, "
                              f"Avg mal in committee={committee_tracking.get('avg_malicious_count', 0):.2f}/{cs}, "
                              f"Rounds with mal={committee_tracking.get('rounds_with_malicious', 0)}/{_rounds}")

                        dataset_attack_results[attack_type][scheme_key] = {
                            'baseline': {
                                'final_accuracy': baseline_acc,
                                'test_acc_history': bl_acc_hist,
                            },
                            'attack': {
                                'final_accuracy': final_accuracy,
                                'test_acc_history': test_acc_history,
                                'detection_metrics': detection_metrics,
                            },
                            'committee_tracking': {
                                'total_rounds': committee_tracking.get('total_rounds', 0),
                                'avg_malicious_count': committee_tracking.get('avg_malicious_count', 0.0),
                                'avg_malicious_fraction': committee_tracking.get('avg_malicious_fraction', 0.0),
                                'max_malicious_count': committee_tracking.get('max_malicious_count', 0),
                                'max_malicious_fraction': committee_tracking.get('max_malicious_fraction', 0.0),
                                'rounds_with_malicious': committee_tracking.get('rounds_with_malicious', 0),
                                'rounds_with_malicious_fraction': committee_tracking.get('rounds_with_malicious_fraction', 0.0),
                                'per_round': per_round_summary,
                            },
                        }
                    except Exception as e:
                        print(f"    [{scheme_key}] FAILED: {e}")
                        import traceback; traceback.print_exc()
                        dataset_attack_results[attack_type][scheme_key] = {
                            'baseline': {'final_accuracy': baseline_acc},
                            'attack': {'final_accuracy': 0.0},
                            'committee_tracking': {},
                        }

                print(f"    Completed {attack_type.upper()} (all CMFL schemes)")

            results[f"{ratio_pct}%"][dataset] = dataset_attack_results

    # Save results
    results_dir = Path(output_base) / "aggregated_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"committee_malicious_tracking_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 100}")
    print(f"✅ Committee malicious tracking ablation completed")
    print(f"Results saved to: {results_file}")
    print(f"{'=' * 100}")
    return results


# =============================================================================
# ABLATION STUDY 5: EFFICIENCY EXPERIMENT (CMFL Paper Section 6.5)
# =============================================================================

def run_efficiency_experiment(
    datasets=None,
    rounds=None,
    total_clients=108,
    committee_size_frac=0.40,
    aggregation_participation_frac=0.40,
    transmission_rates=None,
    dataset_fraction=1.0,
    output_base=None,
    selection_strategy=1,
    run_id=None
):
    """Ablation 7 — Efficiency experiment (CMFL paper Section 6.5).

    Compares communication overhead and wall-clock time across 4 FL frameworks:
    - Typical FL (FedAvg): actual training run
    - BrainTorrent: analytical simulation (all-to-all communication)
    - GossipFL: analytical simulation (k-neighbor gossip, k=3)
    - CMFL Strategy I: actual training run

    BrainTorrent and GossipFL share Typical FL's per-round computation time
    (same local training) but have different communication cost formulas.

    Paper specification:
    - Datasets: FEMNIST, Sentiment140 only (no Shakespeare)
    - Rounds: 600, Clients: 108 (43 committee + 65 training)
    - Transmission rates: 1, 10, 100 Mbps
    - No attacks (clean baseline)
    """
    import time
    import torch
    from models import get_model_size_bytes
    from coordinator import DecentralizedFLCoordinator
    from datasets import partition_dataset

    if datasets is None:
        datasets = ['FEMNIST', 'Sentiment140']
    if transmission_rates is None:
        transmission_rates = [1, 10, 100]  # Mbps
    if run_id is None:
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_base is None:
        output_base = f"Output/efficiency_experiment/{run_id}"
    else:
        output_base = f"{output_base}/{run_id}"

    # Only FEMNIST and Sentiment140 (paper Section 6.5)
    datasets = [d for d in datasets if d in ('FEMNIST', 'Sentiment140')]
    if not datasets:
        print("[EFFICIENCY] No applicable datasets (need FEMNIST or Sentiment140). Skipping.")
        return {}

    print("\n" + "=" * 100)
    print("ABLATION STUDY 5: EFFICIENCY EXPERIMENT (CMFL Paper Section 6.5)")
    print("=" * 100)
    print(f"Run ID: {run_id}")
    print(f"Output: {output_base}")
    print(f"Datasets             : {datasets}")
    print(f"Rounds               : {rounds if rounds is not None else 'per-dataset (DATASET_ROUNDS)'}")
    print(f"Learning rate        : {DATASET_LR}")
    print(f"Batch size           : {DATASET_BATCH_SIZE}")
    print(f"Total clients        : {total_clients}")
    print(f"Committee size frac  : {committee_size_frac * 100:.0f}%")
    print(f"Agg participation    : {aggregation_participation_frac * 100:.0f}%")
    print(f"Transmission rates   : {transmission_rates} Mbps")
    print(f"Dataset fraction     : {dataset_fraction}")
    print("=" * 100)

    framework = EvaluationFramework(out_dir=output_base)
    results = {}

    n_clients = total_clients if isinstance(total_clients, int) else 108
    clients_per_round = n_clients  # full participation for efficiency experiment
    committee_size = max(2, int(clients_per_round * committee_size_frac))
    training_clients = max(4, clients_per_round - committee_size)
    gossip_k = 3  # GossipFL default neighbor count

    for dataset in datasets:
        _rounds = rounds if rounds is not None else DATASET_ROUNDS.get(dataset, 200)
        print(f"\n{'#' * 100}")
        print(f"#  EFFICIENCY: {dataset}")
        print(f"{'#' * 100}")

        # Get datasets and models
        datasets_and_loaders, models_map = framework.get_datasets_and_models(dataset)
        if dataset not in datasets_and_loaders:
            print(f"[ERROR] Dataset {dataset} not found, skipping")
            continue

        train_dataset, test_dataset = datasets_and_loaders[dataset]
        if dataset_fraction < 1.0:
            train_dataset = framework._subset_dataset(train_dataset, dataset_fraction)
            test_dataset = framework._subset_dataset(test_dataset, dataset_fraction)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

        client_data = partition_dataset(train_dataset, n_clients, iid=False, alpha=1.0)

        # Build clean clients (no attacks)
        clients = framework.build_clients(
            num_clients=n_clients,
            client_data=client_data,
            models_map=models_map,
            dataset_name=dataset,
            attack_config=None
        )

        # Model size
        model_size_bytes = get_model_size_bytes(clients[0].model)
        model_size_bits = model_size_bytes * 8
        print(f"[INFO] Model size: {model_size_bytes:,} bytes ({model_size_bytes / 1024:.1f} KB)")

        # ── Run 1: Typical FL (FedAvg) ──
        print(f"\n  [Typical FL] Running {_rounds} rounds...")
        # Snapshot client state for reuse
        for c in clients:
            c.snapshot_state()

        _lr_eff = DATASET_LR.get(dataset, 0.001)
        _bs_eff = DATASET_BATCH_SIZE.get(dataset, 32)

        coordinator_fl = DecentralizedFLCoordinator(
            clients, use_defense=False, defense_type=None,
            aggregation_method='fedavg', clients_per_round=clients_per_round,
            lr=_lr_eff, batch_size=_bs_eff
        )

        eval_interval = max(1, _rounds // 20) if _rounds > 20 else 1
        fl_acc_history = []
        for rnd in range(1, _rounds + 1):
            coordinator_fl.run_federated_learning(
                rounds=1, aggregation_method='fedavg',
                test_loader=test_loader, round_offset=rnd - 1, total_rounds=_rounds
            )
            if rnd % eval_interval == 0 or rnd == _rounds:
                try:
                    acc, _ = coordinator_fl.evaluate_on_test_set(test_loader)
                    fl_acc_history.append(float(acc))
                except Exception:
                    fl_acc_history.append(fl_acc_history[-1] if fl_acc_history else 0.0)
            else:
                fl_acc_history.append(fl_acc_history[-1] if fl_acc_history else 0.0)

        fl_round_times = list(coordinator_fl.round_times)  # per-round computation time
        fl_total_comp = sum(fl_round_times)
        fl_accuracy = fl_acc_history[-1] if fl_acc_history else 0.0
        print(f"  [Typical FL] Done. Accuracy={fl_accuracy:.4f}, Total comp={fl_total_comp:.1f}s")

        # ── Run 2: CMFL Strategy I ──
        print(f"\n  [CMFL] Running {_rounds} rounds...")
        for c in clients:
            c.reset_for_new_run()

        coordinator_cmfl = DecentralizedFLCoordinator(
            clients, use_defense=True, defense_type='cmfl',
            committee_size=committee_size,
            training_clients_per_round=training_clients,
            aggregation_method='weighted_avg',
            use_anomaly_detection=False,  # clean baseline
            clients_per_round=clients_per_round,
            aggregation_participation_frac=aggregation_participation_frac,
            selection_strategy=selection_strategy,
            lr=_lr_eff, batch_size=_bs_eff
        )

        cmfl_acc_history = []
        for rnd in range(1, _rounds + 1):
            coordinator_cmfl.run_federated_learning(
                rounds=1, aggregation_method='cmfl',
                test_loader=test_loader, round_offset=rnd - 1, total_rounds=_rounds
            )
            if rnd % eval_interval == 0 or rnd == _rounds:
                try:
                    acc, _ = coordinator_cmfl.evaluate_on_test_set(test_loader)
                    cmfl_acc_history.append(float(acc))
                except Exception:
                    cmfl_acc_history.append(cmfl_acc_history[-1] if cmfl_acc_history else 0.0)
            else:
                cmfl_acc_history.append(cmfl_acc_history[-1] if cmfl_acc_history else 0.0)

        cmfl_round_times = list(coordinator_cmfl.round_times)
        cmfl_total_comp = sum(cmfl_round_times)
        cmfl_accuracy = cmfl_acc_history[-1] if cmfl_acc_history else 0.0
        print(f"  [CMFL] Done. Accuracy={cmfl_accuracy:.4f}, Total comp={cmfl_total_comp:.1f}s")

        # ── Compute efficiency metrics for each transmission rate ──
        ds_results = {
            'model_size_bytes': model_size_bytes,
            'rounds': _rounds,
            'total_clients': n_clients,
            'committee_size': committee_size,
            'training_clients': training_clients,
            'fl_total_comp_time': fl_total_comp,
            'cmfl_total_comp_time': cmfl_total_comp,
            'fl_accuracy': fl_accuracy,
            'cmfl_accuracy': cmfl_accuracy,
        }

        N = clients_per_round  # active clients per round (Typical FL)
        s = model_size_bytes
        T = training_clients
        C = committee_size

        for rate_mbps in transmission_rates:
            r = rate_mbps * 1e6  # bits per second
            rate_key = f"{rate_mbps}Mbps"

            # ── Per-round communication time & total data transferred ──

            # Typical FL: N uploads + N downloads (parallel)
            fl_comm_per_round = 2.0 * model_size_bits / r  # parallel clients
            fl_data_per_round = 2 * N * s  # bytes

            # BrainTorrent: each client sends to all others (sequential per client)
            bt_comm_per_round = (N - 1) * model_size_bits / r
            bt_data_per_round = N * (N - 1) * s

            # GossipFL: k neighbors, bidirectional
            gfl_comm_per_round = 2.0 * gossip_k * model_size_bits / r
            gfl_data_per_round = 2 * N * gossip_k * s

            # CMFL: T training clients upload + committee broadcasts back
            cmfl_comm_per_round = 2.0 * model_size_bits / r  # parallel, similar latency
            cmfl_data_per_round = (T + C) * s  # fewer total bytes

            # ── Total wall-clock time = sum(comp_time + comm_time) per round ──
            fl_total_time = sum(ct + fl_comm_per_round for ct in fl_round_times)
            bt_total_time = sum(ct + bt_comm_per_round for ct in fl_round_times)  # same comp as FL
            gfl_total_time = sum(ct + gfl_comm_per_round for ct in fl_round_times)
            cmfl_total_time = sum(ct + cmfl_comm_per_round for ct in cmfl_round_times)

            # ── Total communication overhead (bytes) ──
            fl_total_comm = fl_data_per_round * _rounds
            bt_total_comm = bt_data_per_round * _rounds
            gfl_total_comm = gfl_data_per_round * _rounds
            cmfl_total_comm = cmfl_data_per_round * _rounds

            ds_results[rate_key] = {
                'typical_fl': {
                    'total_time': fl_total_time,
                    'comm_overhead_bytes': fl_total_comm,
                    'comp_time': fl_total_comp,
                    'comm_time_total': fl_comm_per_round * _rounds,
                    'accuracy': fl_accuracy,
                },
                'braintorrent': {
                    'total_time': bt_total_time,
                    'comm_overhead_bytes': bt_total_comm,
                    'comp_time': fl_total_comp,
                    'comm_time_total': bt_comm_per_round * _rounds,
                },
                'gossipfl': {
                    'total_time': gfl_total_time,
                    'comm_overhead_bytes': gfl_total_comm,
                    'comp_time': fl_total_comp,
                    'comm_time_total': gfl_comm_per_round * _rounds,
                },
                'cmfl': {
                    'total_time': cmfl_total_time,
                    'comm_overhead_bytes': cmfl_total_comm,
                    'comp_time': cmfl_total_comp,
                    'comm_time_total': cmfl_comm_per_round * _rounds,
                    'accuracy': cmfl_accuracy,
                },
            }

            # Print summary for this rate
            print(f"\n  [{rate_key}] Communication overhead & wall-clock time:")
            for fw_name in ['typical_fl', 'braintorrent', 'gossipfl', 'cmfl']:
                fw = ds_results[rate_key][fw_name]
                comm_mb = fw['comm_overhead_bytes'] / (1024 ** 2)
                print(f"    {fw_name:15s}: time={fw['total_time']:.1f}s, comm={comm_mb:.1f} MB")

        results[dataset] = ds_results

    # Save results
    results_dir = Path(output_base) / "aggregated_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"efficiency_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 100}")
    print(f"✅ Efficiency experiment completed")
    print(f"Results saved to: {results_file}")
    print(f"{'=' * 100}")
    return results


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Ablation Study for Federated Learning Defense",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all ablation studies
  python run_impact_analysis.py --all

  # Run specific ablation
  python run_impact_analysis.py --data-poison-percentage
  python run_impact_analysis.py --client-malicious-ratio
  python run_impact_analysis.py --non-iid
  python run_impact_analysis.py --aggregation
  python run_impact_analysis.py --client-participation

  # Custom parameters
  python run_impact_analysis.py --data-poison-percentage --percentages 0.2 0.4 0.6 0.8
  python run_impact_analysis.py --client-malicious-ratio --ratios 0.2 0.4 0.6 0.8
  python run_impact_analysis.py --aggregation --schemes fedavg krum multi_krum median trimmed_mean
  python run_impact_analysis.py --client-participation --participation-rates 0.3 0.4 0.6 0.8

  # Run with specific datasets
  python run_impact_analysis.py --all --datasets Fashion-MNIST
  python run_impact_analysis.py --data-poison-percentage --datasets CIFAR-10

  # Adjust rounds and attacks/defenses
  python run_impact_analysis.py --all --rounds 15 --attacks slf dlf centralized

  # Use dataset fraction for faster testing
  python run_impact_analysis.py --all --dataset-fraction 0.1
  python run_impact_analysis.py --data-poison-percentage --dataset-fraction 0.25
        """
    )

    # Ablation study flags
    parser.add_argument('--all', action='store_true',
                       help='Run all ablation studies')
    parser.add_argument('--data-poison-percentage', action='store_true',
                       help='Ablation: Data poison percentage (20%%, 40%%, 60%%, 80%%) — data-poisoning attacks only')
    parser.add_argument('--client-malicious-ratio', action='store_true',
                       help='Ablation: Client malicious ratio (20%%, 40%%, 60%%, 80%%) — all attacks')
    parser.add_argument('--non-iid', action='store_true',
                       help='Ablation: Non-IID data distribution (0%%, 20%%, 40%%, 60%%, 80%%)')
    parser.add_argument('--client-participation', action='store_true',
                       help='Ablation: Client participation rate (20%%, 40%%, 60%%, 80%%)')
    parser.add_argument('--aggregation-participation', action='store_true',
                       help='Ablation: Aggregation participation fraction (CMFL only)')
    parser.add_argument('--committee-size', action='store_true',
                       help='Ablation: Committee size fraction (CMFL only)')

    # Parameter customization
    parser.add_argument('--percentages', nargs='+', type=float,
                       help='Data poison percentages (e.g., 0.2 0.4 0.6 0.8)')
    parser.add_argument('--ratios', nargs='+', type=float,
                       help='Client malicious ratios (e.g., 0.2 0.4 0.6 0.8)')
    parser.add_argument('--schemes', nargs='+', type=str,
                       help='Aggregation schemes (e.g., fedavg krum multi_krum median trimmed_mean)')
    parser.add_argument('--participation-rates', nargs='+', type=float,
                       help='Client participation rates (e.g., 0.3 0.4 0.6 0.8)')

    # General parameters
    parser.add_argument('--datasets', nargs='+', type=str, default=None,
                       help='Datasets to use (default: FEMNIST Shakespeare Sentiment140)')
    parser.add_argument('--rounds', type=int, default=None,
                       help='Number of FL rounds (default: per-dataset — FEMNIST=600, Shakespeare=500, Sentiment140=1000)')
    parser.add_argument('--total-clients', type=int, default=100,
                       help='Total number of clients (default: 100 for scalability study)')
    parser.add_argument('--attacks', nargs='+', type=str, default=None,
                       help='Attacks to test (default: per-ablation — data-poisoning for poison%%, all 9 for others)')
    parser.add_argument('--defenses', nargs='+', type=str, default=None,
                       help='Defenses to test (default: cmfl)')
    parser.add_argument('--dataset-fraction', type=float, default=1.0,
                       help='Fraction of dataset to use (default: 1.0 = 100%%, 0.1 = 10%%)')

    args = parser.parse_args()

    # Check if any ablation is selected
    if not (args.all or args.data_poison_percentage or args.client_malicious_ratio or
            args.non_iid or args.client_participation or
            args.aggregation_participation or args.committee_size):
        print("[ERROR] Please specify at least one ablation study:")
        print("  --all, --data-poison-percentage, --client-malicious-ratio, --non-iid,")
        print("  --client-participation, --aggregation-participation, or --committee-size")
        parser.print_help()
        sys.exit(1)

    # Determine datasets
    datasets = args.datasets if args.datasets else ['FEMNIST', 'Shakespeare', 'Sentiment140']

    print("\n" + "="*100)
    print("COMPREHENSIVE ABLATION STUDY FOR FEDERATED LEARNING DEFENSE")
    print("="*100)
    print(f"\nConfiguration:")
    print(f"  Datasets: {datasets}")
    print(f"  Rounds: {args.rounds if args.rounds is not None else 'per-dataset (FEMNIST=600, Shakespeare=500, Sentiment140=1000)'}")
    print(f"  Total clients: {args.total_clients}")
    if args.attacks:
        print(f"  Attacks: {args.attacks}")
    if args.defenses:
        print(f"  Defenses: {args.defenses}")
    print()

    # Run ablation studies
    all_results = {}

    if args.all or args.data_poison_percentage:
        print("\n[STARTING] Data Poison Percentage Ablation...")
        # Filter out Byzantine attacks — they don't poison data, so results are
        # constant across data-poison percentages and would be meaningless.
        ablation1_attacks = args.attacks
        if ablation1_attacks is not None:
            ablation1_attacks = [a for a in ablation1_attacks if a not in BYZANTINE_ATTACKS]
            if not ablation1_attacks:
                ablation1_attacks = None  # fall back to per-ablation default
        results = run_data_poison_percentage_ablation(
            percentages=args.percentages,
            datasets=datasets,
            rounds=args.rounds,
            total_clients=args.total_clients,
            attacks_to_test=ablation1_attacks,
            defenses_to_test=args.defenses,
            dataset_fraction=args.dataset_fraction
        )
        all_results['data_poison_percentage'] = results

    if args.all or args.client_malicious_ratio:
        print("\n[STARTING] Client Malicious Ratio Ablation...")
        results = run_client_malicious_ratio_ablation(
            ratios=args.ratios,
            datasets=datasets,
            rounds=args.rounds,
            total_clients=args.total_clients,
            attacks_to_test=args.attacks,
            defenses_to_test=args.defenses,
            dataset_fraction=args.dataset_fraction
        )
        all_results['client_malicious_ratio'] = results

    if args.all or args.non_iid:
        print("\n[STARTING] Non-IID Distribution Ablation...")
        results = run_non_iid_ablation(
            datasets=datasets,
            rounds=args.rounds,
            total_clients=args.total_clients,
            attacks_to_test=args.attacks,
            defenses_to_test=args.defenses,
            dataset_fraction=args.dataset_fraction
        )
        all_results['non_iid'] = results

    if args.all or args.client_participation:
        print("\n[STARTING] Client Participation Rate Ablation...")
        results = run_client_participation_ablation(
            participation_rates=args.participation_rates,
            datasets=datasets,
            rounds=args.rounds,
            total_clients=args.total_clients,
            attacks_to_test=args.attacks,
            defenses_to_test=args.defenses,
            dataset_fraction=args.dataset_fraction
        )
        all_results['client_participation'] = results

    if args.all or args.aggregation_participation:
        print("\n[STARTING] Aggregation Participation Fraction Ablation...")
        results = run_aggregation_participation_ablation(
            datasets=datasets,
            rounds=args.rounds,
            total_clients=args.total_clients,
            attacks_to_test=args.attacks,
            dataset_fraction=args.dataset_fraction
        )
        all_results['aggregation_participation'] = results

    if args.all or args.committee_size:
        print("\n[STARTING] Committee Size Ablation...")
        results = run_committee_size_ablation(
            datasets=datasets,
            rounds=args.rounds,
            total_clients=args.total_clients,
            attacks_to_test=args.attacks,
            dataset_fraction=args.dataset_fraction
        )
        all_results['committee_size'] = results

    # Save combined results
    if all_results:
        combined_dir = Path("Output/ablation_study_combined")
        combined_dir.mkdir(parents=True, exist_ok=True)
        combined_file = combined_dir / f"all_ablations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'='*100}")
        print("✅ ALL ABLATION STUDIES COMPLETED")
        print(f"{'='*100}")
        print(f"\nCombined results saved to: {combined_file}")
        print(f"\nIndividual results saved in:")
        if 'data_poison_percentage' in all_results:
            print("  - Output/ablation_data_poison_percentage/")
        if 'client_malicious_ratio' in all_results:
            print("  - Output/ablation_client_malicious_ratio/")
        if 'non_iid' in all_results:
            print("  - Output/ablation_non_iid/")
        if 'client_participation' in all_results:
            print("  - Output/ablation_client_participation/")
        if 'aggregation_participation' in all_results:
            print("  - Output/ablation_agg_participation/")
        if 'committee_size' in all_results:
            print("  - Output/ablation_committee_size/")

        print(f"\n{'='*100}")
        print("Next steps:")
        print("1. Use table_generator.py to create tables from results")
        print("2. Use plot_tables.py to visualize the ablation study results")
        print("3. Analyze impact of each parameter on defense performance")
        print(f"{'='*100}\n")


if __name__ == "__main__":
    import time
    overall_start = time.time()

    main()

    overall_elapsed = time.time() - overall_start
    print(f"\n" + "="*100)
    print(f"⏱️  OVERALL TIME: {overall_elapsed:.1f}s (~{overall_elapsed/60:.2f} minutes, ~{overall_elapsed/3600:.2f} hours)")
    print("="*100)
