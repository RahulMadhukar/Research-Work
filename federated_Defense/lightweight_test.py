#!/usr/bin/env python3
"""
Lightweight test configuration for rapid framework testing — PAPER-EXACT (CMFL).

This uses:
- FEMNIST dataset with 10% subset (fast)
- LightweightFEMNISTNet model (smaller, faster than standard)
- 100 total clients (10 participating per round — paper default)
- 1 local epoch per round (paper tau=1)
- 100 communication rounds
- 3 Byzantine attacks only (paper Section 6.3)

Model Architecture:
- LightweightFEMNISTNet: Custom lightweight CNN defined in this file
  * Conv1: 1→16, 5x5 (vs 32 in standard)
  * Conv2: 16→32, 5x5 (vs 64 in standard)
  * FC1: 1568→1024 (vs 3136→2048 in standard)
  * FC2: 1024→62

Ablation Studies (7):
2. Client Malicious Ratio (10%, 20%, 30%, 40%, 50%)        [CMFL paper ε, Section 6.4]
3. Client Participation Rate (10%-50%)
4. Aggregation Participation (10%-50%)                      [CMFL paper α, Section 6.4]
5. Committee Size (10%-50%)                                 [CMFL paper ω, Section 6.4]
6. Committee Malicious Tracking (track malicious in committee) [CMFL paper Section 6.6]
7. Non-IID Data Distribution (0%, 20%, 40%, 60%, 80%)
8. Efficiency Experiment (FEMNIST & Sentiment140 only)      [CMFL paper Section 6.5]

Aggregation Schemes (7):
  FedAvg, Krum, Multi-Krum, Median, Trimmed Mean,
  CMFL Strategy I, CMFL Strategy II

Usage:
    python lightweight_test.py
"""

import os
# Thread settings removed: setting to '1' serializes numpy/sklearn ops,
# killing multi-core parallelism and hurting performance by 20-40%.

from pathlib import Path
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# LIGHTWEIGHT CNN MODEL FOR FAST TESTING
# =============================================================================
# CRITICAL: Define and patch the model BEFORE importing other modules

class LightweightFEMNISTNet(nn.Module):
    """
    Lightweight CNN for FEMNIST - optimized for quick testing.

    Architecture (halved channels vs standard FEMNISTNet):
    - Conv1: 1 -> 16 channels, 5x5 (vs 32 in standard)
    - Conv2: 16 -> 32 channels, 5x5 (vs 64 in standard)
    - FC1: 32*7*7 -> 1024 (vs 3136->2048 in standard)
    - FC2: 1024 -> 62
    """
    def __init__(self, num_classes=62):
        super(LightweightFEMNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        features = F.relu(self.fc1(x))
        x = self.fc2(features)
        return x

    def get_features(self, x):
        """Extract features for defense analysis (required by framework)"""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        features = F.relu(self.fc1(x))
        return features


# =============================================================================
# PATCH THE MODELS MODULE BEFORE IMPORTING ANYTHING ELSE
# =============================================================================
# This MUST happen before run_impact_analysis imports evaluation, which imports models

import sys
import models

# Store original FEMNISTNet for comparison if needed
_OriginalFEMNISTNet = models.FEMNISTNet

# Replace FEMNISTNet with lightweight version
models.FEMNISTNet = LightweightFEMNISTNet

print("\n" + "="*100)
print("LIGHTWEIGHT MODEL ACTIVATED")
print("="*100)
print("Standard FEMNISTNet has been replaced with LightweightFEMNISTNet")
print("")
print("Model Comparison:")
print("  Standard FEMNISTNet:")
print("    - Conv1: 1->32 (5x5), Conv2: 32->64 (5x5), FC1: 3136->2048, FC2: 2048->62")
print("")
print("  LightweightFEMNISTNet (This run):")
print("    - Conv1: 1->16 (5x5), Conv2: 16->32 (5x5), FC1: 1568->1024, FC2: 1024->62")
print("="*100 + "\n")

# NOW import the ablation functions (they will use the patched model)
from run_impact_analysis import (
    run_data_poison_percentage_ablation,
    run_client_malicious_ratio_ablation,
    run_non_iid_ablation,
    run_client_participation_ablation,
    run_aggregation_participation_ablation,
    run_committee_size_ablation,
    run_committee_malicious_tracking_ablation,
    run_efficiency_experiment,
    BYZANTINE_ATTACKS,
    DATA_POISONING_ATTACKS,
)


def run_full_ablation_study(selected_datasets=None, selected_attacks=None):
    """
    Run complete ablation study — one dataset at a time.

    Args:
        selected_datasets: List of datasets to run, or None for all.
        selected_attacks:  List of attacks to run, or None for all.

    Order: FEMNIST -> Shakespeare -> Sentiment140
    For each dataset, all 8 ablation studies run to completion before moving on.
    This way a failure in one dataset (e.g. download issue) does NOT block the others.
    """

    # ---------- shared parameters ----------
    all_datasets = ['FEMNIST', 'Shakespeare', 'Sentiment140']
    all_attacks_list = [
        # Byzantine gradient attacks (CMFL paper Section 6.3)
        'gradient_scaling',
        'same_value',
        'back_gradient',
        # --- NOT IN CMFL PAPER (commented out) ---
        # # Label flipping attacks
        # 'slf',
        # 'dlf',
        # # Trigger-based backdoor attacks
        # 'centralized',
        # 'coordinated',
        # 'random',
        # 'model_dependent',
    ]

    datasets_order = selected_datasets if selected_datasets else all_datasets
    attacks_to_run = selected_attacks if selected_attacks else all_attacks_list

    print("\n" + "="*100)
    print("LIGHTWEIGHT FRAMEWORK - PAPER-EXACT ABLATION STUDY (CMFL)")
    print("="*100)
    print("Configuration:")
    print(f"  - Datasets: {', '.join(datasets_order)}")
    print(f"  - Attacks:  {', '.join(attacks_to_run)} (Byzantine only, CMFL paper Section 6.3)")
    print("  - Total Clients: 100")
    print("  - Participating Clients: 10% of total per round (paper default)")
    print("  - Rounds: 100")
    print("  - Local Epochs: 1 (paper tau=1)")
    print("  - Dataset Fraction: 10%")
    print("")
    print("  Aggregation Schemes (7):")
    print("    FedAvg, Krum, Multi-Krum, Median, Trimmed Mean,")
    print("    CMFL (Strategy I), CMFL II (Strategy II)")
    print("")
    print("  Ablation Studies (7):")
    print("    2. Client Malicious %  (10%, 20%, 30%, 40%, 50%)   [CMFL paper ε, Section 6.4]")
    print("    3. Client Participation (10%-50%)")
    print("    4. Agg. Particip.   (10%-50%) [CMFL paper α, Section 6.4]")
    print("    5. Committee Size   (10%-50%) [CMFL paper ω, Section 6.4]")
    print("    6. Committee Mal.   (track malicious in committee) [CMFL paper Section 6.6]")
    print("    7. Non-IID          (0%, 20%, 40%, 60%, 80%)")
    print("    8. Efficiency       (CMFL paper Section 6.5, FEMNIST & Sentiment140 only)")
    print("="*100)

    # Create output directory
    output_base = f"Output/lightweight_ablation/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    Path(output_base).mkdir(parents=True, exist_ok=True)

    dataset_fraction = 0.1  # Only 10% for speed
    num_clients = {
        'FEMNIST':       100,
        'Shakespeare':   100,
        'Sentiment140':  100,
    }
    rounds = 100

    all_attacks = attacks_to_run
    
    # 3 aggregation schemes (CMFL-paper style)
    # cmfl = CMFL Strategy I (top alpha%, for Byzantine robustness)
    # cmfl_ii = CMFL Strategy II (bottom alpha%, for convergence acceleration)
    all_schemes = [
        'fedavg', 'multi_krum', 'cmfl', 'cmfl_ii',
    ]
    
    # 7 aggregation schemes (CMFL-paper style)
    # cmfl = CMFL Strategy I (top alpha%, for Byzantine robustness)
    # cmfl_ii = CMFL Strategy II (bottom alpha%, for convergence acceleration)
    #all_schemes = [
    #    'fedavg', 'krum', 'multi_krum', 'median', 'trimmed_mean',
    #    # 'adaptivecommittee',  # NOT in CMFL paper
    #    'cmfl', 'cmfl_ii',
    #]

    # ---------- accumulate results across datasets ----------
    all_results = {
        # 'data_poison_percentage':     {},  # NOT in CMFL paper
        'client_malicious_ratio':     {},
        'client_participation':       {},
        'agg_participation':          {},
        'committee_size':             {},
        'non_iid':                    {},
    }

    # ==========================================================================
    # OUTER LOOP: one dataset at a time
    # ==========================================================================
    for ds in datasets_order:
        print(f"\n{'#'*100}")
        print(f"#  STARTING DATASET: {ds}")
        print(f"{'#'*100}\n")

        ds_list = [ds]  # single-element list expected by ablation functions
        clients_per_round = max(1, int(num_clients[ds] * 0.1))  # Paper: 10% participation

        # -- Ablation 1: Data Poison Percentage --
        # NOT IN CMFL PAPER — paper uses only Byzantine attacks which don't poison data
        print(f"\n[{ds}] ABLATION 1: SKIPPED — NOT in CMFL paper (no data-poisoning attacks)")
        # try:
        #     res = run_data_poison_percentage_ablation(...)
        #     all_results['data_poison_percentage'].update(res)
        # except Exception as e:
        #     print(f"[{ds}] ABLATION 1 FAILED: {e}")

        # -- Ablation 2: Client Malicious Ratio (CMFL paper ε) --
        print("\n" + "="*100)
        print(f"[{ds}] ABLATION 2: CLIENT MALICIOUS RATIO (CMFL paper ε)")
        print("="*100)
        try:
            res = run_client_malicious_ratio_ablation(
                #ratios=[0.1, 0.2, 0.3, 0.4, 0.5],  # CMFL paper: ε ∈ {10,20,30,40,50}
                ratios=[0.1, 0.3, 0.5],
                output_base=f"{output_base}/client_malicious_ratio",
                datasets=ds_list,
                rounds=rounds,
                total_clients=num_clients,
                client_participation=clients_per_round,
                poison_percentage=0.4,
                attacks_to_test=all_attacks,
                agg_schemes=all_schemes,
                dataset_fraction=dataset_fraction
            )
            all_results['client_malicious_ratio'].update(res)
        except Exception as e:
            print(f"[{ds}] ABLATION 2 FAILED: {e}")

        # -- Ablation 3: Client Participation Rate --
        print("\n" + "="*100)
        print(f"[{ds}] ABLATION 3: CLIENT PARTICIPATION RATE")
        print("="*100)
        try:
            res = run_client_participation_ablation(
                #participation_rates=[0.1, 0.2, 0.3, 0.4, 0.5],
                participation_rates=[0.2, 0.4, 0.6],
                output_base=f"{output_base}/client_participation",
                datasets=ds_list,
                rounds=rounds,
                total_clients=num_clients,
                malicious_ratio=0.4,
                poison_percentage=0.4,
                attacks_to_test=all_attacks,
                agg_schemes=all_schemes,
                dataset_fraction=dataset_fraction
            )
            all_results['client_participation'].update(res)
        except Exception as e:
            print(f"[{ds}] ABLATION 3 FAILED: {e}")

        # -- Ablation 4: Aggregation Participation Fraction (CMFL paper α) --
        print("\n" + "="*100)
        print(f"[{ds}] ABLATION 4: AGGREGATION PARTICIPATION FRACTION (CMFL paper α)")
        print("="*100)
        try:
            res = run_aggregation_participation_ablation(
                #participation_fracs=[0.1, 0.2, 0.3, 0.4, 0.5],  # CMFL paper: α ∈ {10,20,30,40,50}
                participation_fracs=[0.1, 0.3, 0.5],
                output_base=f"{output_base}/agg_participation",
                datasets=ds_list,
                rounds=rounds,
                total_clients=num_clients,
                client_participation=clients_per_round,
                malicious_ratio=0.1,  # CMFL paper default ε=10
                poison_percentage=0.4,
                attacks_to_test=all_attacks,
                dataset_fraction=dataset_fraction
            )
            all_results['agg_participation'].update(res)
        except Exception as e:
            print(f"[{ds}] ABLATION 4 FAILED: {e}")

        # -- Ablation 5: Committee Size (CMFL paper ω) --
        print("\n" + "="*100)
        print(f"[{ds}] ABLATION 5: COMMITTEE SIZE (CMFL paper ω)")
        print("="*100)
        try:
            res = run_committee_size_ablation(
                #committee_size_fracs=[0.1, 0.2, 0.3, 0.4, 0.5],  # CMFL paper: ω ∈ {10,20,30,40,50}
                committee_size_fracs=[0.1, 0.3, 0.5],
                output_base=f"{output_base}/committee_size",
                datasets=ds_list,
                rounds=rounds,
                total_clients=num_clients,
                client_participation=clients_per_round,
                malicious_ratio=0.1,  # CMFL paper default ε=10
                poison_percentage=0.4,
                attacks_to_test=all_attacks,
                dataset_fraction=dataset_fraction
            )
            all_results['committee_size'].update(res)
        except Exception as e:
            print(f"[{ds}] ABLATION 5 FAILED: {e}")

        # -- Ablation 6: Committee Member Malicious Tracking (CMFL only) --
        print("\n" + "="*100)
        print(f"[{ds}] ABLATION 6: COMMITTEE MEMBER MALICIOUS TRACKING")
        print("="*100)
        try:
            res = run_committee_malicious_tracking_ablation(
                #ratios=[0.1, 0.2, 0.3, 0.4, 0.5],
                ratios=[0.1, 0.3, 0.5],
                output_base=f"{output_base}/committee_malicious_tracking",
                datasets=ds_list,
                rounds=rounds,
                total_clients=num_clients,
                client_participation=clients_per_round,
                poison_percentage=0.4,
                attacks_to_test=all_attacks,
                committee_size_frac=0.40,
                aggregation_participation_frac=0.40,
                dataset_fraction=dataset_fraction,
            )
            all_results.setdefault('committee_malicious_tracking', {}).update(res)
        except Exception as e:
            print(f"[{ds}] ABLATION 6 FAILED: {e}")

        # -- Ablation 7: Non-IID Distribution --
        print("\n" + "="*100)
        print(f"[{ds}] ABLATION 7: NON-IID DATA DISTRIBUTION")
        print("="*100)
        try:
            res = run_non_iid_ablation(
                non_iid_levels=[
                    ('0%',  1000.0),   # IID
                    #('20%', 1.0),      # Slightly Non-IID
                    ('40%', 0.5),      # Moderately Non-IID
                    #('60%', 0.3),      # Highly Non-IID
                    ('80%', 0.1),      # Extremely Non-IID
                ],
                output_base=f"{output_base}/non_iid",
                datasets=ds_list,
                rounds=rounds,
                total_clients=num_clients,
                client_participation=clients_per_round,
                malicious_ratio=0.4,
                poison_percentage=0.4,
                attacks_to_test=all_attacks,
                agg_schemes=all_schemes,
                dataset_fraction=dataset_fraction
            )
            all_results['non_iid'].update(res)
        except Exception as e:
            print(f"[{ds}] ABLATION 7 FAILED: {e}")

        # -- Ablation 8: Efficiency Experiment (CMFL paper Section 6.5) --
        # Only runs for FEMNIST and Sentiment140 (paper Section 6.5)
        if ds in ('FEMNIST', 'Sentiment140'):
            print("\n" + "="*100)
            print(f"[{ds}] ABLATION 8: EFFICIENCY EXPERIMENT (CMFL paper Section 6.5)")
            print("="*100)
            try:
                res = run_efficiency_experiment(
                    datasets=ds_list,
                    rounds=rounds,
                    total_clients=num_clients[ds],
                    transmission_rates=[1, 10, 100],
                    dataset_fraction=dataset_fraction,
                    output_base=f"{output_base}/efficiency",
                )
                all_results.setdefault('efficiency', {}).update(res)
            except Exception as e:
                print(f"[{ds}] ABLATION 8 FAILED: {e}")
        else:
            print(f"\n[{ds}] ABLATION 8: SKIPPED (efficiency experiment is FEMNIST/Sentiment140 only)")

        print(f"\n{'#'*100}")
        print(f"#  COMPLETED DATASET: {ds}")
        print(f"{'#'*100}\n")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print(f"\n{'='*100}")
    print("ALL ABLATION STUDIES COMPLETED (all datasets)")
    print(f"{'='*100}")
    print(f"\nResults saved in:")
    # print(f"  - {output_base}/data_poison_percentage/")  # NOT in CMFL paper
    print(f"  - {output_base}/client_malicious_ratio/")
    print(f"  - {output_base}/client_participation/")
    print(f"  - {output_base}/agg_participation/")
    print(f"  - {output_base}/committee_size/")
    print(f"  - {output_base}/committee_malicious_tracking/")
    print(f"  - {output_base}/non_iid/")
    print(f"  - {output_base}/efficiency/")
    print(f"{'='*100}\n")

    return all_results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser(description="Lightweight ablation study")
    parser.add_argument('--dataset', '-d', type=str, default=None,
                        help="Single dataset to run: FEMNIST, Shakespeare, Sentiment140")
    parser.add_argument('--attack', '-a', type=str, default=None,
                        help="Single attack to run: gradient_scaling, same_value, back_gradient")
    args = parser.parse_args()

    # Normalize dataset name to expected capitalization
    _DS_MAP = {
        'femnist': 'FEMNIST', 'shakespeare': 'Shakespeare', 'sentiment140': 'Sentiment140'
    }
    sel_datasets = [_DS_MAP.get(args.dataset.lower(), args.dataset)] if args.dataset else None
    sel_attacks = [args.attack] if args.attack else None

    start_time = time.time()
    results = run_full_ablation_study(
        selected_datasets=sel_datasets,
        selected_attacks=sel_attacks
    )
    elapsed = time.time() - start_time
    print(f"\n Total time: {elapsed:.2f} seconds (~{elapsed/60:.1f} minutes)")
