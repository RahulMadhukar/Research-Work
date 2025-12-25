#!/usr/bin/env python3
"""
Comprehensive Ablation Study for Federated Learning Defense

This script performs a comprehensive ablation study analyzing the impact of:
1. Poison Data Percentage (20%, 40%, 60%, 80%)
2. Malicious Client Ratio (20%, 40%, 60%, 80%)
3. Non-IID Data Distribution (0%, 20%, 40%, 60%, 80%)
4. Total Client Participation (20, 40, 60, 80, 100 from 100 clients)
5. Aggregation Schemes (FedAvg, Krum, Median, Trimmed Mean)

The evaluation.py is kept for baseline evaluation with:
- 10 clients
- All parameters at 40%
- FedAvg aggregation

Usage:
    # Run full ablation study
    python run_impact_analysis.py --all

    # Run specific ablation
    python run_impact_analysis.py --poison-percentage
    python run_impact_analysis.py --malicious-ratio
    python run_impact_analysis.py --non-iid
    python run_impact_analysis.py --client-participation
    python run_impact_analysis.py --aggregation

    # Custom parameters
    python run_impact_analysis.py --poison-percentage --percentages 0.2 0.4 0.6 0.8
    python run_impact_analysis.py --malicious-ratio --ratios 0.2 0.4 0.6 0.8

    # Use dataset fraction for faster testing
    python run_impact_analysis.py --all --dataset-fraction 0.1  # Use 10% of data
    python run_impact_analysis.py --poison-percentage --dataset-fraction 0.25  # Use 25% of data
"""

# FIX: Set thread limits BEFORE importing numpy/torch to prevent OpenBLAS warnings
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Import evaluation framework for proper client setup
from evaluation import EvaluationFramework


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _compute_test_loss(aggregated_params, model_factory, test_loader, device=None):
    """
    Compute average test loss for the aggregated model.
    """
    import torch
    device = device or (torch.device('cuda') if torch.cuda.is_available() else 'cpu')
    model = model_factory()
    model.to(device)

    try:
        model.load_state_dict(aggregated_params, strict=False)
    except Exception:
        sd = model.state_dict()
        mismatched = []
        for k, v in sd.items():
            if k in aggregated_params:
                ap = aggregated_params[k]
                if isinstance(ap, torch.Tensor) and ap.shape == v.shape:
                    try:
                        sd[k] = ap.clone().to(device).type(v.dtype)
                    except Exception:
                        mismatched.append(k)
                else:
                    mismatched.append(k)
        if mismatched:
            print(f"[WARN] _compute_test_loss: skipped mismatched keys: {mismatched}")
        model.load_state_dict(sd, strict=False)

    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss


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
    dataset_fraction: float = 1.0
):
    """
    Run baseline evaluation with clean data (NO attack, NO poisoning).

    This represents the upper bound of model performance with benign clients only.

    Args:
        framework: Evaluation framework instance
        dataset_name: 'MNIST', 'Fashion-MNIST', 'EMNIST', or 'CIFAR-10'
        num_clients: Total number of clients
        rounds: Number of FL rounds
        clients_per_round: Clients participating per round
        aggregation: Aggregation method
        alpha: Dirichlet alpha for Non-IID data
        dataset_fraction: Fraction of dataset to use

    Returns:
        float: Baseline accuracy (clean data, no attack)
    """
    import torch
    from coordinator import DecentralizedFLCoordinator

    # Get datasets and models
    datasets_and_loaders, models_map = framework.get_datasets_and_models()

    if dataset_name not in datasets_and_loaders:
        print(f"[ERROR] Dataset {dataset_name} not found")
        return 0.0

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

    # Create coordinator without defense
    coordinator = DecentralizedFLCoordinator(
        clients,
        use_defense=False,
        defense_type=None
    )

    # Track test accuracy and loss per round
    test_acc_history = []
    test_loss_history = []

    import torch
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    print(f"[INFO] Running baseline for {rounds} rounds (evaluating after each round)...")

    # Run FL round by round
    for round_num in range(1, rounds + 1):
        print(f"\n{'='*80}")
        print(f"[ROUND] Starting Round {round_num}/{rounds}")
        print(f"{'='*80}")

        # Run a single round with proper round numbering
        coordinator.run_federated_learning(rounds=1, aggregation_method=aggregation, test_loader=test_loader, round_offset=round_num - 1, total_rounds=rounds)

        # Evaluate after each round
        try:
            round_test_acc, _ = coordinator.evaluate_on_test_set(test_loader)

            # Get model params for test loss computation
            model_params = coordinator.global_model.state_dict() if hasattr(coordinator, 'global_model') else clients[0].model.state_dict()
            round_test_loss = _compute_test_loss(
                model_params,
                lambda: type(clients[0].model)(),
                test_loader,
                device
            )

            test_acc_history.append(float(round_test_acc))
            test_loss_history.append(float(round_test_loss))
            print(f"[EVALUATION] Round {round_num}/{rounds} → Test Acc: {round_test_acc:.4f}, Test Loss: {round_test_loss:.4f}")
        except Exception as e:
            print(f"[WARN] Test evaluation failed at round {round_num}: {e}")
            test_acc_history.append(test_acc_history[-1] if test_acc_history else 0.0)
            test_loss_history.append(test_loss_history[-1] if test_loss_history else 0.0)

    # Get final accuracy
    final_accuracy = test_acc_history[-1] if test_acc_history else 0.0

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
        dataset_name: 'MNIST', 'Fashion-MNIST', 'EMNIST', or 'CIFAR-10'
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
    datasets_and_loaders, models_map = framework.get_datasets_and_models()

    if dataset_name not in datasets_and_loaders:
        print(f"[ERROR] Dataset {dataset_name} not found")
        return None, None, None, None

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
            poisoning_rate=poison_percentage,
            source_class=[0, 2, 4, 6],
            target_class=[1, 3, 5, 7],
            num_malicious_clients=malicious_clients
        )
    elif attack_type.lower() == 'dlf':
        attack_config = AttackConfig(
            attack_type=attack_type,
            poisoning_rate=poison_percentage,
            target_class=0,
            num_malicious_clients=malicious_clients
        )
    elif attack_type.lower() in ['centralized', 'coordinated', 'random']:
        attack_config = AttackConfig(
            attack_type=attack_type,
            poisoning_rate=poison_percentage,
            target_class=[1, 3, 5, 7],
            trigger_size=(8, 8),
            trigger_intensity=0.7,
            num_malicious_clients=malicious_clients
        )
    elif attack_type.lower() == 'model_dependent':
        attack_config = AttackConfig(
            attack_type=attack_type,
            poisoning_rate=0.90,  # Higher poisoning rate for model_dependent
            source_class=1,
            target_class=[1, 3, 5, 7],
            epsilon=0.65,
            num_malicious_clients=malicious_clients
        )
    else:
        # Fallback for unknown attack types
        attack_config = AttackConfig(
            attack_type=attack_type,
            num_malicious_clients=malicious_clients,
            poisoning_rate=poison_percentage,
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
    clients_per_round: Optional[int] = None
):
    """
    Test a defense mechanism on a given attack scenario.

    Uses deep copy of clients to ensure the same poisoned clients are used
    across all defense tests for fair comparison.

    Args:
        clients_list: List of clients (will be deep copied)
        test_loader: Test data loader
        test_X: Test data as numpy array
        y_test: Test labels as numpy array
        rounds: Number of FL rounds
        aggregation: Aggregation method
        defense_type: Defense mechanism name (None for no defense/attack scenario)
        clients_per_round: Optional number of clients participating per round

    Returns:
        tuple: (final_accuracy, attack_success_rate, detection_metrics, test_acc_history, test_loss_history)
    """
    import copy
    from coordinator import DecentralizedFLCoordinator

    # Deep copy clients to avoid modifying the original
    clients_copy = copy.deepcopy(clients_list)

    # Create coordinator with defense
    use_defense = defense_type is not None
    coordinator = DecentralizedFLCoordinator(
        clients_copy,
        use_defense=use_defense,
        defense_type=defense_type if defense_type else 'adaptivecommittee'
    )

    # Track test accuracy and loss per round
    test_acc_history = []
    test_loss_history = []

    import torch
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    scenario_name = f"{defense_type.upper()} defense" if use_defense else "ATTACK (no defense)"
    print(f"[INFO] Running {scenario_name} for {rounds} rounds (evaluating after each round)...")

    # Run FL round by round
    for round_num in range(1, rounds + 1):
        print(f"\n{'='*80}")
        print(f"[ROUND] Starting Round {round_num}/{rounds}")
        print(f"{'='*80}")

        # Run a single round with proper round numbering
        coordinator.run_federated_learning(
            rounds=1,
            aggregation_method=aggregation,
            test_loader=test_loader,
            round_offset=round_num - 1,
            total_rounds=rounds
        )

        # Evaluate after each round
        try:
            round_test_acc, _ = coordinator.evaluate_on_test_set(test_loader)

            # Get model params for test loss computation
            model_params = coordinator.global_model.state_dict() if hasattr(coordinator, 'global_model') else clients_copy[0].model.state_dict()
            round_test_loss = _compute_test_loss(
                model_params,
                lambda: type(clients_copy[0].model)(),
                test_loader,
                device
            )

            test_acc_history.append(float(round_test_acc))
            test_loss_history.append(float(round_test_loss))
            print(f"[EVALUATION] Round {round_num}/{rounds} → Test Acc: {round_test_acc:.4f}, Test Loss: {round_test_loss:.4f}")
        except Exception as e:
            print(f"[WARN] Test evaluation failed at round {round_num}: {e}")
            test_acc_history.append(test_acc_history[-1] if test_acc_history else 0.0)
            test_loss_history.append(test_loss_history[-1] if test_loss_history else 0.0)

    # Get final accuracy
    final_accuracy = test_acc_history[-1] if test_acc_history else 0.0

    # Calculate attack success rate
    attack_success_rate = 0.0
    if test_X is not None and y_test is not None:
        try:
            asr_results = coordinator.evaluate_attack_success(test_X, y_test)
            if isinstance(asr_results, dict):
                attack_success_rate = asr_results.get('attack_success_rate', 0.0)
            elif isinstance(asr_results, (int, float)):
                attack_success_rate = float(asr_results)
        except:
            attack_success_rate = 0.0

    # Get detection metrics for defenses
    detection_metrics = {}
    if use_defense:
        try:
            detection_metrics = coordinator.get_committee_metrics()
        except:
            detection_metrics = {}

    return final_accuracy, attack_success_rate, detection_metrics, test_acc_history, test_loss_history


# =============================================================================
# ABLATION STUDY 1: POISON DATA PERCENTAGE
# =============================================================================

def run_poison_percentage_ablation(
    percentages: Optional[List[float]] = None,
    output_base: str = "Output/ablation_poison_percentage",
    datasets: Optional[List[str]] = None,
    rounds: int = 1,
    total_clients: int = 10,
    client_participation: int = 40,
    malicious_ratio: float = 0.4,
    aggregation: str = "fedavg",
    attacks_to_test: Optional[List[str]] = None,
    defenses_to_test: Optional[List[str]] = None,
    dataset_fraction: float = 1.0
):
    """
    Ablation Study 1: Impact of poison data percentage.

    Tests with different poison percentages: 20%, 40%, 60%, 80%
    Fixed: 100 clients, 40 participate, 40% malicious, FedAvg aggregation

    Args:
        percentages: List of poison percentages (e.g., [0.2, 0.4, 0.6, 0.8])
        output_base: Base output directory
        datasets: Datasets to test (default: ['MNIST', 'Fashion-MNIST', 'EMNIST', 'CIFAR-10'])
        rounds: Number of FL rounds
        total_clients: Total number of clients
        client_participation: Number of clients participating per round
        malicious_ratio: Ratio of malicious clients
        aggregation: Aggregation scheme
        attacks_to_test: List of attacks to test
        defenses_to_test: List of defenses to test
    """
    if percentages is None:
        percentages = [0.2, 0.4, 0.6, 0.8]  # 20%, 40%, 60%, 80%

    if datasets is None:
        datasets = ['MNIST', 'Fashion-MNIST', 'EMNIST', 'CIFAR-10']

    if attacks_to_test is None:
        attacks_to_test = ['slf', 'dlf', 'centralized', 'coordinated', 'random', 'model_dependent',
                           'local_model_replacement', 'local_model_noise', 'global_model_replacement', 'aggregation_modification']

    if defenses_to_test is None:
        defenses_to_test = ['adaptivecommittee', 'cmfl']

    print("\n" + "="*100)
    print("ABLATION STUDY 1: POISON DATA PERCENTAGE")
    print("="*100)
    print(f"Testing poison percentages: {[f'{p*100:.0f}%' for p in percentages]}")
    print(f"Fixed parameters:")
    print(f"  - Total clients: {total_clients}")
    print(f"  - Client participation: {client_participation}")
    print(f"  - Malicious ratio: {malicious_ratio*100:.0f}%")
    print(f"  - Aggregation: {aggregation.upper()}")
    print(f"Datasets: {datasets}")
    print(f"Rounds: {rounds}")

    # Create evaluation framework
    framework = EvaluationFramework(out_dir=output_base)

    results_by_percentage = {}
    num_malicious = int(total_clients * malicious_ratio)

    for percentage in percentages:
        pct = int(percentage * 100)
        print(f"\n{'='*80}")
        print(f"Running with {pct}% poison data")
        print(f"{'='*80}")

        results_by_percentage[f"{pct}%"] = {}

        for dataset in datasets:
            print(f"\n[DATASET] {dataset}")
            output_dir = f"{output_base}/poison_{pct}pct/{dataset}"
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            results_by_percentage[f"{pct}%"][dataset] = {}

            # Step 1: Run baseline ONCE per dataset (clean data, no attack)
            print(f"  [BASELINE] Running baseline (clean data, no attack)...")
            try:
                baseline_acc, baseline_test_acc_history, baseline_test_loss_history = run_baseline_evaluation(
                    framework=framework,
                    dataset_name=dataset,
                    num_clients=total_clients,
                    rounds=rounds,
                    clients_per_round=client_participation,
                    aggregation=aggregation,
                    alpha=1.0,
                    dataset_fraction=dataset_fraction
                )
                print(f"  [BASELINE] ✅ Baseline accuracy: {baseline_acc:.4f}")
            except Exception as e:
                print(f"  [BASELINE] ❌ Failed: {e}")
                baseline_acc = 0.0
                baseline_test_acc_history = []
                baseline_test_loss_history = []

            # Step 2: For each attack, create scenario once and test all defenses
            for attack_type in attacks_to_test:
                print(f"\n  [ATTACK] {attack_type.upper()}")

                try:
                    # Create attack scenario ONCE (poisoned clients)
                    print(f"    Creating attack scenario (poisoned clients)...")
                    clients, test_loader, test_X, y_test = create_attack_scenario(
                        framework=framework,
                        dataset_name=dataset,
                        attack_type=attack_type,
                        poison_percentage=percentage,
                        num_clients=total_clients,
                        malicious_clients=num_malicious,
                        alpha=1.0,
                        dataset_fraction=dataset_fraction
                    )

                    if clients is None:
                        print(f"    ❌ Failed to create attack scenario")
                        continue

                    # STEP 2: Run attack scenario WITHOUT defense (CRITICAL - was missing!)
                    print(f"    Running attack WITHOUT defense (establishing attack baseline)...")
                    try:
                        attack_acc, attack_asr, _, attack_test_acc_history, attack_test_loss_history = test_defense_on_scenario(
                            clients_list=clients,
                            test_loader=test_loader,
                            test_X=test_X,
                            y_test=y_test,
                            rounds=rounds,
                            aggregation=aggregation,
                            defense_type=None  # NO DEFENSE!
                        )
                        print(f"    [ATTACK] ✅ Attack accuracy: {attack_acc:.4f}, ASR: {attack_asr:.4f}")
                    except Exception as e:
                        print(f"    [ATTACK] ❌ Failed: {e}")
                        attack_acc = 0.0
                        attack_asr = 0.0
                        attack_test_acc_history = []
                        attack_test_loss_history = []

                    # Store baseline and attack results
                    results_by_percentage[f"{pct}%"][dataset][attack_type] = {
                        'baseline': {
                            'final_accuracy': baseline_acc,
                            'test_acc_history': baseline_test_acc_history,
                            'test_loss_history': baseline_test_loss_history
                        },
                        'attack_no_defense': {
                            'final_accuracy': attack_acc,
                            'attack_success_rate': attack_asr,
                            'test_acc_history': attack_test_acc_history,
                            'test_loss_history': attack_test_loss_history
                        }
                    }

                    # Step 3: Test each defense on the SAME attack scenario
                    for defense_name in defenses_to_test:
                        print(f"    Testing {defense_name.upper()} defense...")

                        defense_acc, defense_asr, defense_metrics, defense_test_acc_history, defense_test_loss_history = test_defense_on_scenario(
                            clients_list=clients,
                            test_loader=test_loader,
                            test_X=test_X,
                            y_test=y_test,
                            rounds=rounds,
                            aggregation=aggregation,
                            defense_type=defense_name
                        )

                        results_by_percentage[f"{pct}%"][dataset][attack_type][defense_name] = {
                            'final_accuracy': defense_acc,
                            'attack_success_rate': defense_asr,
                            'detection_metrics': defense_metrics,
                            'test_acc_history': defense_test_acc_history,
                            'test_loss_history': defense_test_loss_history
                        }

                    print(f"    ✅ Completed {attack_type.upper()}")

                except Exception as e:
                    print(f"    ❌ Failed {attack_type.upper()}: {e}")
                    import traceback
                    traceback.print_exc()

    # Save results
    results_dir = Path(output_base) / "aggregated_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"poison_percentage_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(results_file, 'w') as f:
        json.dump(results_by_percentage, f, indent=2)

    print(f"\n{'='*100}")
    print(f"✅ Poison percentage ablation completed")
    print(f"Results saved to: {results_file}")
    print(f"{'='*100}")

    return results_by_percentage


# =============================================================================
# ABLATION STUDY 2: MALICIOUS CLIENT RATIO
# =============================================================================

def run_malicious_ratio_ablation(
    ratios: Optional[List[float]] = None,
    output_base: str = "Output/ablation_malicious_ratio",
    datasets: Optional[List[str]] = None,
    rounds: int = 1,
    total_clients: int = 10,
    client_participation: int = 40,
    poison_percentage: float = 0.4,
    aggregation: str = "fedavg",
    attacks_to_test: Optional[List[str]] = None,
    defenses_to_test: Optional[List[str]] = None,
    dataset_fraction: float = 1.0
):
    """
    Ablation Study 2: Impact of malicious client ratio.

    Tests with different malicious ratios: 20%, 40%, 60%, 80%
    Fixed: 100 clients, 40 participate, 40% poison data, FedAvg aggregation
    """
    if ratios is None:
        ratios = [0.2, 0.4, 0.6, 0.8]  # 20%, 40%, 60%, 80%

    if datasets is None:
        datasets = ['MNIST', 'Fashion-MNIST', 'EMNIST', 'CIFAR-10']

    if attacks_to_test is None:
        attacks_to_test = ['slf', 'dlf', 'centralized', 'coordinated', 'random', 'model_dependent',
                           'local_model_replacement', 'local_model_noise', 'global_model_replacement', 'aggregation_modification']

    if defenses_to_test is None:
        defenses_to_test = ['adaptivecommittee', 'cmfl']

    print("\n" + "="*100)
    print("ABLATION STUDY 2: MALICIOUS CLIENT RATIO")
    print("="*100)
    print(f"Testing malicious ratios: {[f'{r*100:.0f}%' for r in ratios]}")
    print(f"Fixed parameters:")
    print(f"  - Total clients: {total_clients}")
    print(f"  - Client participation: {client_participation}")
    print(f"  - Poison percentage: {poison_percentage*100:.0f}%")
    print(f"  - Aggregation: {aggregation.upper()}")
    print(f"Datasets: {datasets}")
    print(f"Rounds: {rounds}")

    # Create evaluation framework
    framework = EvaluationFramework(out_dir=output_base)

    results_by_ratio = {}

    for ratio in ratios:
        ratio_pct = int(ratio * 100)
        num_malicious = int(total_clients * ratio)

        print(f"\n{'='*80}")
        print(f"Running with {ratio_pct}% malicious clients ({num_malicious}/{total_clients})")
        print(f"{'='*80}")

        results_by_ratio[f"{ratio_pct}%"] = {}

        for dataset in datasets:
            print(f"\n[DATASET] {dataset}")
            output_dir = f"{output_base}/malicious_{ratio_pct}pct/{dataset}"
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            results_by_ratio[f"{ratio_pct}%"][dataset] = {}

            # Step 1: Run baseline ONCE per dataset (clean data, no attack)
            print(f"  [BASELINE] Running baseline (clean data, no attack)...")
            try:
                baseline_acc, baseline_test_acc_history, baseline_test_loss_history = run_baseline_evaluation(
                    framework=framework,
                    dataset_name=dataset,
                    num_clients=total_clients,
                    rounds=rounds,
                    clients_per_round=client_participation,
                    aggregation=aggregation,
                    alpha=1.0,
                    dataset_fraction=dataset_fraction
                )
                print(f"  [BASELINE] ✅ Baseline accuracy: {baseline_acc:.4f}")
            except Exception as e:
                print(f"  [BASELINE] ❌ Failed: {e}")
                baseline_acc = 0.0
                baseline_test_acc_history = []
                baseline_test_loss_history = []

            # Step 2: For each attack, create scenario once and test all defenses
            for attack_type in attacks_to_test:
                print(f"\n  [ATTACK] {attack_type.upper()}")

                try:
                    # Create attack scenario ONCE (poisoned clients)
                    print(f"    Creating attack scenario (poisoned clients)...")
                    clients, test_loader, test_X, y_test = create_attack_scenario(
                        framework=framework,
                        dataset_name=dataset,
                        attack_type=attack_type,
                        poison_percentage=poison_percentage,
                        num_clients=total_clients,
                        malicious_clients=num_malicious,  # Variable parameter
                        alpha=1.0,
                        dataset_fraction=dataset_fraction
                    )

                    if clients is None:
                        print(f"    ❌ Failed to create attack scenario")
                        continue

                    # STEP 2: Run attack scenario WITHOUT defense (CRITICAL - was missing!)
                    print(f"    Running attack WITHOUT defense (establishing attack baseline)...")
                    try:
                        attack_acc, attack_asr, _, attack_test_acc_history, attack_test_loss_history = test_defense_on_scenario(
                            clients_list=clients,
                            test_loader=test_loader,
                            test_X=test_X,
                            y_test=y_test,
                            rounds=rounds,
                            aggregation=aggregation,
                            defense_type=None  # NO DEFENSE!
                        )
                        print(f"    [ATTACK] ✅ Attack accuracy: {attack_acc:.4f}, ASR: {attack_asr:.4f}")
                    except Exception as e:
                        print(f"    [ATTACK] ❌ Failed: {e}")
                        attack_acc = 0.0
                        attack_asr = 0.0
                        attack_test_acc_history = []
                        attack_test_loss_history = []

                    # Store baseline and attack results
                    results_by_ratio[f"{ratio_pct}%"][dataset][attack_type] = {
                        'baseline': {
                            'final_accuracy': baseline_acc,
                            'test_acc_history': baseline_test_acc_history,
                            'test_loss_history': baseline_test_loss_history
                        },
                        'attack_no_defense': {
                            'final_accuracy': attack_acc,
                            'attack_success_rate': attack_asr,
                            'test_acc_history': attack_test_acc_history,
                            'test_loss_history': attack_test_loss_history
                        }
                    }

                    # Step 3: Test each defense on the SAME attack scenario
                    for defense_name in defenses_to_test:
                        print(f"    Testing {defense_name.upper()} defense...")

                        defense_acc, defense_asr, defense_metrics, defense_test_acc_history, defense_test_loss_history = test_defense_on_scenario(
                            clients_list=clients,
                            test_loader=test_loader,
                            test_X=test_X,
                            y_test=y_test,
                            rounds=rounds,
                            aggregation=aggregation,
                            defense_type=defense_name
                        )

                        results_by_ratio[f"{ratio_pct}%"][dataset][attack_type][defense_name] = {
                            'final_accuracy': defense_acc,
                            'attack_success_rate': defense_asr,
                            'detection_metrics': defense_metrics,
                            'test_acc_history': defense_test_acc_history,
                            'test_loss_history': defense_test_loss_history
                        }

                    print(f"    ✅ Completed {attack_type.upper()}")

                except Exception as e:
                    print(f"    ❌ Failed {attack_type.upper()}: {e}")
                    import traceback
                    traceback.print_exc()

    # Save results
    results_dir = Path(output_base) / "aggregated_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"malicious_ratio_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(results_file, 'w') as f:
        json.dump(results_by_ratio, f, indent=2)

    print(f"\n{'='*100}")
    print(f"✅ Malicious ratio ablation completed")
    print(f"Results saved to: {results_file}")
    print(f"{'='*100}")

    return results_by_ratio


# =============================================================================
# ABLATION STUDY 3: NON-IID DATA DISTRIBUTION
# =============================================================================

def run_non_iid_ablation(
    non_iid_levels: Optional[List[float]] = None,
    output_base: str = "Output/ablation_non_iid",
    datasets: Optional[List[str]] = None,
    rounds: int = 1,
    total_clients: int = 10,
    client_participation: int = 40,
    malicious_ratio: float = 0.4,
    poison_percentage: float = 0.4,
    aggregation: str = "fedavg",
    attacks_to_test: Optional[List[str]] = None,
    defenses_to_test: Optional[List[str]] = None,
    dataset_fraction: float = 1.0
):
    """
    Ablation Study 3: Impact of Non-IID data distribution.

    Tests with different Non-IID levels: 0%, 20%, 40%, 60%, 80%
    - 0% = IID (alpha = infinity)
    - 20% = Slightly Non-IID (alpha = 1.0)
    - 40% = Moderately Non-IID (alpha = 0.5)
    - 60% = Highly Non-IID (alpha = 0.3)
    - 80% = Extremely Non-IID (alpha = 0.1)

    Fixed: 100 clients, 40 participate, 40% malicious, 40% poison, FedAvg
    """
    if non_iid_levels is None:
        # Map Non-IID percentage to Dirichlet alpha
        non_iid_levels = [
            ('0%', None),      # IID
            ('20%', 1.0),      # Slightly Non-IID
            ('40%', 0.5),      # Moderately Non-IID
            ('60%', 0.3),      # Highly Non-IID
            ('80%', 0.1)       # Extremely Non-IID
        ]

    if datasets is None:
        datasets = ['MNIST', 'Fashion-MNIST', 'EMNIST', 'CIFAR-10']

    if attacks_to_test is None:
        attacks_to_test = ['slf', 'dlf', 'centralized', 'coordinated', 'random', 'model_dependent',
                           'local_model_replacement', 'local_model_noise', 'global_model_replacement', 'aggregation_modification']

    if defenses_to_test is None:
        defenses_to_test = ['adaptivecommittee', 'cmfl']

    print("\n" + "="*100)
    print("ABLATION STUDY 3: NON-IID DATA DISTRIBUTION")
    print("="*100)
    print(f"Testing Non-IID levels: {[level[0] for level in non_iid_levels]}")
    print(f"Fixed parameters:")
    print(f"  - Total clients: {total_clients}")
    print(f"  - Client participation: {client_participation}")
    print(f"  - Malicious ratio: {malicious_ratio*100:.0f}%")
    print(f"  - Poison percentage: {poison_percentage*100:.0f}%")
    print(f"  - Aggregation: {aggregation.upper()}")
    print(f"Datasets: {datasets}")
    print(f"Rounds: {rounds}")

    # Create evaluation framework
    framework = EvaluationFramework(out_dir=output_base)

    results_by_noniid = {}
    num_malicious = int(total_clients * malicious_ratio)

    for level_name, alpha in non_iid_levels:
        print(f"\n{'='*80}")
        print(f"Running with {level_name} Non-IID (alpha={'IID' if alpha is None else alpha})")
        print(f"{'='*80}")

        results_by_noniid[level_name] = {}

        for dataset in datasets:
            print(f"\n[DATASET] {dataset}")
            output_dir = f"{output_base}/noniid_{level_name.replace('%', 'pct')}/{dataset}"
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            results_by_noniid[level_name][dataset] = {}

            # Use very large alpha for IID data, otherwise use specified alpha
            actual_alpha = 1000.0 if alpha is None else alpha

            # Step 1: Run baseline ONCE per dataset (clean data, no attack)
            print(f"  [BASELINE] Running baseline (clean data, no attack)...")
            try:
                baseline_acc, baseline_test_acc_history, baseline_test_loss_history = run_baseline_evaluation(
                    framework=framework,
                    dataset_name=dataset,
                    num_clients=total_clients,
                    rounds=rounds,
                    clients_per_round=client_participation,
                    aggregation=aggregation,
                    alpha=actual_alpha,  # Variable parameter
                    dataset_fraction=dataset_fraction
                )
                print(f"  [BASELINE] ✅ Baseline accuracy: {baseline_acc:.4f}")
            except Exception as e:
                print(f"  [BASELINE] ❌ Failed: {e}")
                baseline_acc = 0.0
                baseline_test_acc_history = []
                baseline_test_loss_history = []

            # Step 2: For each attack, create scenario once and test all defenses
            for attack_type in attacks_to_test:
                print(f"\n  [ATTACK] {attack_type.upper()}")

                try:
                    # Create attack scenario ONCE (poisoned clients)
                    print(f"    Creating attack scenario (poisoned clients)...")
                    clients, test_loader, test_X, y_test = create_attack_scenario(
                        framework=framework,
                        dataset_name=dataset,
                        attack_type=attack_type,
                        poison_percentage=poison_percentage,
                        num_clients=total_clients,
                        malicious_clients=num_malicious,
                        alpha=actual_alpha,  # Variable parameter
                        dataset_fraction=dataset_fraction
                    )

                    if clients is None:
                        print(f"    ❌ Failed to create attack scenario")
                        continue

                    # STEP 2: Run attack scenario WITHOUT defense (CRITICAL - was missing!)
                    print(f"    Running attack WITHOUT defense (establishing attack baseline)...")
                    try:
                        attack_acc, attack_asr, _, attack_test_acc_history, attack_test_loss_history = test_defense_on_scenario(
                            clients_list=clients,
                            test_loader=test_loader,
                            test_X=test_X,
                            y_test=y_test,
                            rounds=rounds,
                            aggregation=aggregation,
                            defense_type=None  # NO DEFENSE!
                        )
                        print(f"    [ATTACK] ✅ Attack accuracy: {attack_acc:.4f}, ASR: {attack_asr:.4f}")
                    except Exception as e:
                        print(f"    [ATTACK] ❌ Failed: {e}")
                        attack_acc = 0.0
                        attack_asr = 0.0
                        attack_test_acc_history = []
                        attack_test_loss_history = []

                    # Store baseline and attack results
                    results_by_noniid[level_name][dataset][attack_type] = {
                        'baseline': {
                            'final_accuracy': baseline_acc,
                            'test_acc_history': baseline_test_acc_history,
                            'test_loss_history': baseline_test_loss_history
                        },
                        'attack_no_defense': {
                            'final_accuracy': attack_acc,
                            'attack_success_rate': attack_asr,
                            'test_acc_history': attack_test_acc_history,
                            'test_loss_history': attack_test_loss_history
                        }
                    }

                    # Step 3: Test each defense on the SAME attack scenario
                    for defense_name in defenses_to_test:
                        print(f"    Testing {defense_name.upper()} defense...")

                        defense_acc, defense_asr, defense_metrics, defense_test_acc_history, defense_test_loss_history = test_defense_on_scenario(
                            clients_list=clients,
                            test_loader=test_loader,
                            test_X=test_X,
                            y_test=y_test,
                            rounds=rounds,
                            aggregation=aggregation,
                            defense_type=defense_name
                        )

                        results_by_noniid[level_name][dataset][attack_type][defense_name] = {
                            'final_accuracy': defense_acc,
                            'attack_success_rate': defense_asr,
                            'detection_metrics': defense_metrics,
                            'test_acc_history': defense_test_acc_history,
                            'test_loss_history': defense_test_loss_history
                        }

                    print(f"    ✅ Completed {attack_type.upper()}")

                except Exception as e:
                    print(f"    ❌ Failed {attack_type.upper()}: {e}")
                    import traceback
                    traceback.print_exc()

    # Save results
    results_dir = Path(output_base) / "aggregated_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"non_iid_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(results_file, 'w') as f:
        json.dump(results_by_noniid, f, indent=2)

    print(f"\n{'='*100}")
    print(f"✅ Non-IID ablation completed")
    print(f"Results saved to: {results_file}")
    print(f"{'='*100}")

    return results_by_noniid


# =============================================================================
# ABLATION STUDY 4: CLIENT PARTICIPATION
# =============================================================================

def run_client_participation_ablation(
    participation_counts: Optional[List[int]] = None,
    output_base: str = "Output/ablation_client_participation",
    datasets: Optional[List[str]] = None,
    rounds: int = 1,
    total_clients: int = 10,
    malicious_ratio: float = 0.4,
    poison_percentage: float = 0.4,
    aggregation: str = "fedavg",
    attacks_to_test: Optional[List[str]] = None,
    defenses_to_test: Optional[List[str]] = None,
    dataset_fraction: float = 1.0
):
    """
    Ablation Study 4: Impact of client participation per round.

    Tests with different participation: 20, 40, 60, 80, 100 clients per round (from 100 total)
    Fixed: 100 clients total, 40% malicious, 40% poison, FedAvg aggregation
    """
    if participation_counts is None:
        participation_counts = [20, 40, 60, 80, 100]

    if datasets is None:
        datasets = ['MNIST', 'Fashion-MNIST', 'EMNIST', 'CIFAR-10']

    if attacks_to_test is None:
        attacks_to_test = ['slf', 'dlf', 'centralized', 'coordinated', 'random', 'model_dependent',
                           'local_model_replacement', 'local_model_noise', 'global_model_replacement', 'aggregation_modification']

    if defenses_to_test is None:
        defenses_to_test = ['adaptivecommittee', 'cmfl']

    print("\n" + "="*100)
    print("ABLATION STUDY 4: CLIENT PARTICIPATION")
    print("="*100)
    print(f"Testing participation counts: {participation_counts} clients per round (from {total_clients} total)")
    print(f"Fixed parameters:")
    print(f"  - Total clients: {total_clients}")
    print(f"  - Malicious ratio: {malicious_ratio*100:.0f}%")
    print(f"  - Poison percentage: {poison_percentage*100:.0f}%")
    print(f"  - Aggregation: {aggregation.upper()}")
    print(f"Datasets: {datasets}")
    print(f"Rounds: {rounds}")

    # Create evaluation framework
    framework = EvaluationFramework(out_dir=output_base)

    results_by_participation = {}
    num_malicious = int(total_clients * malicious_ratio)

    for participation in participation_counts:
        print(f"\n{'='*80}")
        print(f"Running with {participation}/{total_clients} clients participating per round")
        print(f"{'='*80}")

        results_by_participation[f"{participation}"] = {}

        for dataset in datasets:
            print(f"\n[DATASET] {dataset}")
            output_dir = f"{output_base}/participation_{participation}/{dataset}"
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            results_by_participation[f"{participation}"][dataset] = {}

            # Step 1: Run baseline ONCE per dataset (clean data, no attack)
            print(f"  [BASELINE] Running baseline (clean data, no attack)...")
            try:
                baseline_acc, baseline_test_acc_history, baseline_test_loss_history = run_baseline_evaluation(
                    framework=framework,
                    dataset_name=dataset,
                    num_clients=total_clients,
                    rounds=rounds,
                    clients_per_round=participation,  # Variable parameter
                    aggregation=aggregation,
                    alpha=1.0,
                    dataset_fraction=dataset_fraction
                )
                print(f"  [BASELINE] ✅ Baseline accuracy: {baseline_acc:.4f}")
            except Exception as e:
                print(f"  [BASELINE] ❌ Failed: {e}")
                baseline_acc = 0.0
                baseline_test_acc_history = []
                baseline_test_loss_history = []

            # Step 2: For each attack, create scenario once and test all defenses
            for attack_type in attacks_to_test:
                print(f"\n  [ATTACK] {attack_type.upper()}")

                try:
                    # Create attack scenario ONCE (poisoned clients)
                    print(f"    Creating attack scenario (poisoned clients)...")
                    clients, test_loader, test_X, y_test = create_attack_scenario(
                        framework=framework,
                        dataset_name=dataset,
                        attack_type=attack_type,
                        poison_percentage=poison_percentage,
                        num_clients=total_clients,
                        malicious_clients=num_malicious,
                        alpha=1.0,
                        dataset_fraction=dataset_fraction
                    )

                    if clients is None:
                        print(f"    ❌ Failed to create attack scenario")
                        continue

                    # STEP 2: Run attack scenario WITHOUT defense (CRITICAL - was missing!)
                    print(f"    Running attack WITHOUT defense (establishing attack baseline)...")
                    try:
                        attack_acc, attack_asr, _, attack_test_acc_history, attack_test_loss_history = test_defense_on_scenario(
                            clients_list=clients,
                            test_loader=test_loader,
                            test_X=test_X,
                            y_test=y_test,
                            rounds=rounds,
                            aggregation=aggregation,
                            defense_type=None,  # NO DEFENSE!
                            clients_per_round=participation  # Variable parameter
                        )
                        print(f"    [ATTACK] ✅ Attack accuracy: {attack_acc:.4f}, ASR: {attack_asr:.4f}")
                    except Exception as e:
                        print(f"    [ATTACK] ❌ Failed: {e}")
                        attack_acc = 0.0
                        attack_asr = 0.0
                        attack_test_acc_history = []
                        attack_test_loss_history = []

                    # Store baseline and attack results
                    results_by_participation[f"{participation}"][dataset][attack_type] = {
                        'baseline': {
                            'final_accuracy': baseline_acc,
                            'test_acc_history': baseline_test_acc_history,
                            'test_loss_history': baseline_test_loss_history
                        },
                        'attack_no_defense': {
                            'final_accuracy': attack_acc,
                            'attack_success_rate': attack_asr,
                            'test_acc_history': attack_test_acc_history,
                            'test_loss_history': attack_test_loss_history
                        }
                    }

                    # Step 3: Test each defense on the SAME attack scenario
                    for defense_name in defenses_to_test:
                        print(f"    Testing {defense_name.upper()} defense...")

                        defense_acc, defense_asr, defense_metrics, defense_test_acc_history, defense_test_loss_history = test_defense_on_scenario(
                            clients_list=clients,
                            test_loader=test_loader,
                            test_X=test_X,
                            y_test=y_test,
                            rounds=rounds,
                            aggregation=aggregation,
                            defense_type=defense_name,
                            clients_per_round=participation  # Variable parameter
                        )

                        results_by_participation[f"{participation}"][dataset][attack_type][defense_name] = {
                            'final_accuracy': defense_acc,
                            'attack_success_rate': defense_asr,
                            'detection_metrics': defense_metrics,
                            'test_acc_history': defense_test_acc_history,
                            'test_loss_history': defense_test_loss_history
                        }

                    print(f"    ✅ Completed {attack_type.upper()}")

                except Exception as e:
                    print(f"    ❌ Failed {attack_type.upper()}: {e}")
                    import traceback
                    traceback.print_exc()

    # Save results
    results_dir = Path(output_base) / "aggregated_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"client_participation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(results_file, 'w') as f:
        json.dump(results_by_participation, f, indent=2)

    print(f"\n{'='*100}")
    print(f"✅ Client participation ablation completed")
    print(f"Results saved to: {results_file}")
    print(f"{'='*100}")

    return results_by_participation


# =============================================================================
# ABLATION STUDY 5: AGGREGATION SCHEMES
# =============================================================================

def run_aggregation_ablation(
    aggregation_schemes: Optional[List[str]] = None,
    output_base: str = "Output/ablation_aggregation",
    datasets: Optional[List[str]] = None,
    rounds: int = 1,
    total_clients: int = 10,
    client_participation: int = 40,
    malicious_ratio: float = 0.4,
    poison_percentage: float = 0.4,
    attacks_to_test: Optional[List[str]] = None,
    defenses_to_test: Optional[List[str]] = None,
    dataset_fraction: float = 1.0
):
    """
    Ablation Study 5: Impact of different aggregation schemes.

    Tests with: FedAvg, Krum, Median, Trimmed Mean
    Fixed: 100 clients, 40 participate, 40% malicious, 40% poison
    """
    if aggregation_schemes is None:
        aggregation_schemes = ['fedavg', 'krum', 'median', 'trimmed_mean']

    if datasets is None:
        datasets = ['MNIST', 'Fashion-MNIST', 'EMNIST', 'CIFAR-10']

    if attacks_to_test is None:
        attacks_to_test = ['slf', 'dlf', 'centralized', 'coordinated', 'random', 'model_dependent',
                           'local_model_replacement', 'local_model_noise', 'global_model_replacement', 'aggregation_modification']

    if defenses_to_test is None:
        defenses_to_test = ['adaptivecommittee', 'cmfl']

    print("\n" + "="*100)
    print("ABLATION STUDY 5: AGGREGATION SCHEMES")
    print("="*100)
    print(f"Testing aggregation schemes: {[agg.upper() for agg in aggregation_schemes]}")
    print(f"Fixed parameters:")
    print(f"  - Total clients: {total_clients}")
    print(f"  - Client participation: {client_participation}")
    print(f"  - Malicious ratio: {malicious_ratio*100:.0f}%")
    print(f"  - Poison percentage: {poison_percentage*100:.0f}%")
    print(f"Datasets: {datasets}")
    print(f"Rounds: {rounds}")

    # Create evaluation framework
    framework = EvaluationFramework(out_dir=output_base)

    results_by_aggregation = {}
    num_malicious = int(total_clients * malicious_ratio)

    for aggregation in aggregation_schemes:
        print(f"\n{'='*80}")
        print(f"Running with {aggregation.upper()} aggregation")
        print(f"{'='*80}")

        results_by_aggregation[aggregation.upper()] = {}

        for dataset in datasets:
            print(f"\n[DATASET] {dataset}")
            output_dir = f"{output_base}/aggregation_{aggregation}/{dataset}"
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            results_by_aggregation[aggregation.upper()][dataset] = {}

            # Step 1: Run baseline ONCE per dataset (clean data, no attack)
            print(f"  [BASELINE] Running baseline (clean data, no attack)...")
            try:
                baseline_acc, baseline_test_acc_history, baseline_test_loss_history = run_baseline_evaluation(
                    framework=framework,
                    dataset_name=dataset,
                    num_clients=total_clients,
                    rounds=rounds,
                    clients_per_round=client_participation,
                    aggregation=aggregation,  # Variable parameter
                    alpha=1.0,
                    dataset_fraction=dataset_fraction
                )
                print(f"  [BASELINE] ✅ Baseline accuracy: {baseline_acc:.4f}")
            except Exception as e:
                print(f"  [BASELINE] ❌ Failed: {e}")
                baseline_acc = 0.0
                baseline_test_acc_history = []
                baseline_test_loss_history = []

            # Step 2: For each attack, create scenario once and test all defenses
            for attack_type in attacks_to_test:
                print(f"\n  [ATTACK] {attack_type.upper()}")

                try:
                    # Create attack scenario ONCE (poisoned clients)
                    print(f"    Creating attack scenario (poisoned clients)...")
                    clients, test_loader, test_X, y_test = create_attack_scenario(
                        framework=framework,
                        dataset_name=dataset,
                        attack_type=attack_type,
                        poison_percentage=poison_percentage,
                        num_clients=total_clients,
                        malicious_clients=num_malicious,
                        alpha=1.0,
                        dataset_fraction=dataset_fraction
                    )

                    if clients is None:
                        print(f"    ❌ Failed to create attack scenario")
                        continue

                    # STEP 2: Run attack scenario WITHOUT defense (CRITICAL - was missing!)
                    print(f"    Running attack WITHOUT defense (establishing attack baseline)...")
                    try:
                        attack_acc, attack_asr, _, attack_test_acc_history, attack_test_loss_history = test_defense_on_scenario(
                            clients_list=clients,
                            test_loader=test_loader,
                            test_X=test_X,
                            y_test=y_test,
                            rounds=rounds,
                            aggregation=aggregation,  # Variable parameter
                            defense_type=None  # NO DEFENSE!
                        )
                        print(f"    [ATTACK] ✅ Attack accuracy: {attack_acc:.4f}, ASR: {attack_asr:.4f}")
                    except Exception as e:
                        print(f"    [ATTACK] ❌ Failed: {e}")
                        attack_acc = 0.0
                        attack_asr = 0.0
                        attack_test_acc_history = []
                        attack_test_loss_history = []

                    # Store baseline and attack results
                    results_by_aggregation[aggregation.upper()][dataset][attack_type] = {
                        'baseline': {
                            'final_accuracy': baseline_acc,
                            'test_acc_history': baseline_test_acc_history,
                            'test_loss_history': baseline_test_loss_history
                        },
                        'attack_no_defense': {
                            'final_accuracy': attack_acc,
                            'attack_success_rate': attack_asr,
                            'test_acc_history': attack_test_acc_history,
                            'test_loss_history': attack_test_loss_history
                        }
                    }

                    # Step 3: Test each defense on the SAME attack scenario
                    for defense_name in defenses_to_test:
                        print(f"    Testing {defense_name.upper()} defense...")

                        defense_acc, defense_asr, defense_metrics, defense_test_acc_history, defense_test_loss_history = test_defense_on_scenario(
                            clients_list=clients,
                            test_loader=test_loader,
                            test_X=test_X,
                            y_test=y_test,
                            rounds=rounds,
                            aggregation=aggregation,  # Variable parameter
                            defense_type=defense_name
                        )

                        results_by_aggregation[aggregation.upper()][dataset][attack_type][defense_name] = {
                            'final_accuracy': defense_acc,
                            'attack_success_rate': defense_asr,
                            'detection_metrics': defense_metrics,
                            'test_acc_history': defense_test_acc_history,
                            'test_loss_history': defense_test_loss_history
                        }

                    print(f"    ✅ Completed {attack_type.upper()}")

                except Exception as e:
                    print(f"    ❌ Failed {attack_type.upper()}: {e}")
                    import traceback
                    traceback.print_exc()

    # Save results
    results_dir = Path(output_base) / "aggregated_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"aggregation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(results_file, 'w') as f:
        json.dump(results_by_aggregation, f, indent=2)

    print(f"\n{'='*100}")
    print(f"✅ Aggregation scheme ablation completed")
    print(f"Results saved to: {results_file}")
    print(f"{'='*100}")

    return results_by_aggregation


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
  python run_impact_analysis.py --poison-percentage
  python run_impact_analysis.py --malicious-ratio
  python run_impact_analysis.py --non-iid
  python run_impact_analysis.py --client-participation
  python run_impact_analysis.py --aggregation

  # Custom parameters
  python run_impact_analysis.py --poison-percentage --percentages 0.2 0.4 0.6 0.8
  python run_impact_analysis.py --malicious-ratio --ratios 0.2 0.4 0.6 0.8
  python run_impact_analysis.py --client-participation --participation 20 40 60 80
  python run_impact_analysis.py --aggregation --schemes fedavg krum median trimmed_mean

  # Run with specific datasets
  python run_impact_analysis.py --all --datasets Fashion-MNIST
  python run_impact_analysis.py --poison-percentage --datasets CIFAR-10

  # Adjust rounds and attacks/defenses
  python run_impact_analysis.py --all --rounds 15 --attacks slf dlf centralized

  # Use dataset fraction for faster testing
  python run_impact_analysis.py --all --dataset-fraction 0.1
  python run_impact_analysis.py --poison-percentage --dataset-fraction 0.25
        """
    )

    # Ablation study flags
    parser.add_argument('--all', action='store_true',
                       help='Run all ablation studies')
    parser.add_argument('--poison-percentage', action='store_true',
                       help='Ablation: Poison data percentage (20%%, 40%%, 60%%, 80%%)')
    parser.add_argument('--malicious-ratio', action='store_true',
                       help='Ablation: Malicious client ratio (20%%, 40%%, 60%%, 80%%)')
    parser.add_argument('--non-iid', action='store_true',
                       help='Ablation: Non-IID data distribution (0%%, 20%%, 40%%, 60%%, 80%%)')
    parser.add_argument('--client-participation', action='store_true',
                       help='Ablation: Client participation (20, 40, 60, 80 from 100)')
    parser.add_argument('--aggregation', action='store_true',
                       help='Ablation: Aggregation schemes (FedAvg, Krum, Median, Trimmed Mean)')

    # Parameter customization
    parser.add_argument('--percentages', nargs='+', type=float,
                       help='Poison percentages (e.g., 0.2 0.4 0.6 0.8)')
    parser.add_argument('--ratios', nargs='+', type=float,
                       help='Malicious ratios (e.g., 0.2 0.4 0.6 0.8)')
    parser.add_argument('--participation', nargs='+', type=int,
                       help='Client participation counts (e.g., 20 40 60 80)')
    parser.add_argument('--schemes', nargs='+', type=str,
                       help='Aggregation schemes (e.g., fedavg krum median trimmed_mean)')

    # General parameters
    parser.add_argument('--datasets', nargs='+', type=str, default=None,
                       help='Datasets to use (default: MNIST Fashion-MNIST EMNIST CIFAR-10)')
    parser.add_argument('--rounds', type=int, default=10,
                       help='Number of FL rounds (default: 10)')
    parser.add_argument('--total-clients', type=int, default=100,
                       help='Total number of clients (default: 10)')
    parser.add_argument('--attacks', nargs='+', type=str, default=None,
                       help='Attacks to test (default: all 10 attacks - 6 data poisoning + 4 model poisoning)')
    parser.add_argument('--defenses', nargs='+', type=str, default=None,
                       help='Defenses to test (default: both committee-based defenses - adaptivecommittee, cmfl)')
    parser.add_argument('--dataset-fraction', type=float, default=1.0,
                       help='Fraction of dataset to use (default: 1.0 = 100%%, 0.1 = 10%%)')

    args = parser.parse_args()

    # Check if any ablation is selected
    if not (args.all or args.poison_percentage or args.malicious_ratio or
            args.non_iid or args.client_participation or args.aggregation):
        print("[ERROR] Please specify at least one ablation study:")
        print("  --all, --poison-percentage, --malicious-ratio, --non-iid,")
        print("  --client-participation, or --aggregation")
        parser.print_help()
        sys.exit(1)

    # Determine datasets
    datasets = args.datasets if args.datasets else ['MNIST', 'Fashion-MNIST', 'EMNIST', 'CIFAR-10']

    print("\n" + "="*100)
    print("COMPREHENSIVE ABLATION STUDY FOR FEDERATED LEARNING DEFENSE")
    print("="*100)
    print(f"\nConfiguration:")
    print(f"  Datasets: {datasets}")
    print(f"  Rounds: {args.rounds}")
    print(f"  Total clients: {args.total_clients}")
    if args.attacks:
        print(f"  Attacks: {args.attacks}")
    if args.defenses:
        print(f"  Defenses: {args.defenses}")
    print()

    # Run ablation studies
    all_results = {}

    if args.all or args.poison_percentage:
        print("\n[STARTING] Poison Percentage Ablation...")
        results = run_poison_percentage_ablation(
            percentages=args.percentages,
            datasets=datasets,
            rounds=args.rounds,
            total_clients=args.total_clients,
            attacks_to_test=args.attacks,
            defenses_to_test=args.defenses,
            dataset_fraction=args.dataset_fraction
        )
        all_results['poison_percentage'] = results

    if args.all or args.malicious_ratio:
        print("\n[STARTING] Malicious Client Ratio Ablation...")
        results = run_malicious_ratio_ablation(
            ratios=args.ratios,
            datasets=datasets,
            rounds=args.rounds,
            total_clients=args.total_clients,
            attacks_to_test=args.attacks,
            defenses_to_test=args.defenses,
            dataset_fraction=args.dataset_fraction
        )
        all_results['malicious_ratio'] = results

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
        print("\n[STARTING] Client Participation Ablation...")
        results = run_client_participation_ablation(
            participation_counts=args.participation,
            datasets=datasets,
            rounds=args.rounds,
            total_clients=args.total_clients,
            attacks_to_test=args.attacks,
            defenses_to_test=args.defenses,
            dataset_fraction=args.dataset_fraction
        )
        all_results['client_participation'] = results

    if args.all or args.aggregation:
        print("\n[STARTING] Aggregation Scheme Ablation...")
        results = run_aggregation_ablation(
            aggregation_schemes=args.schemes,
            datasets=datasets,
            rounds=args.rounds,
            total_clients=args.total_clients,
            attacks_to_test=args.attacks,
            defenses_to_test=args.defenses,
            dataset_fraction=args.dataset_fraction
        )
        all_results['aggregation'] = results

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
        if 'poison_percentage' in all_results:
            print("  - Output/ablation_poison_percentage/")
        if 'malicious_ratio' in all_results:
            print("  - Output/ablation_malicious_ratio/")
        if 'non_iid' in all_results:
            print("  - Output/ablation_non_iid/")
        if 'client_participation' in all_results:
            print("  - Output/ablation_client_participation/")
        if 'aggregation' in all_results:
            print("  - Output/ablation_aggregation/")

        print(f"\n{'='*100}")
        print("Next steps:")
        print("1. Use table_generator.py to create tables from results")
        print("2. Use plot_tables.py to visualize the ablation study results")
        print("3. Analyze impact of each parameter on defense performance")
        print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
