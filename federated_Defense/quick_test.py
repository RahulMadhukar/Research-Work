#!/usr/bin/env python3
"""
Quick multi-dataset test — PAPER-EXACT (CMFL).

Datasets: FEMNIST (25% subset), Shakespeare, Sentiment140
- Natural client counts (FEMNIST=3550, Shakespeare=143, Sentiment140=772)
- 10% participating per round (paper default)
- 1 local epoch per round (paper tau=1)
- 1 communication round (quick test)
- 3 Byzantine gradient attacks only (CMFL paper Section 6.3)

Ablation Studies (7):
2. Client Malicious Ratio (20%, 40%, 60%, 80%)              [CMFL paper ε]
3. Client Participation Rate (10%-50%)
4. Aggregation Participation (pBFT, 10%-50%)                 [CMFL paper α]
5. Committee Size (10%-50%)                                   [CMFL paper ω]
6. Committee Malicious Tracking (track malicious in committee) [CMFL paper Section 6.6]
7. Non-IID Data Distribution (0%, 20%, 40%, 60%, 80%)
8. Efficiency Experiment (FEMNIST & Sentiment140 only)        [CMFL paper Section 6.5]

Usage:
    python quick_test.py
"""

import os
# Thread settings removed: setting to '1' serializes numpy/sklearn ops,
# killing multi-core parallelism and hurting performance by 20-40%.

from pathlib import Path
from datetime import datetime

from run_impact_analysis import (
    run_data_poison_percentage_ablation,
    run_client_malicious_ratio_ablation,
    run_non_iid_ablation,
    run_client_participation_ablation,
    run_aggregation_participation_ablation,
    run_committee_size_ablation,
    run_committee_malicious_tracking_ablation,
    run_efficiency_experiment,
)


def run_full_ablation_study(selected_datasets=None, selected_attacks=None):
    """Run complete ablation study.

    Args:
        selected_datasets: List of datasets to run, or None for all.
        selected_attacks:  List of attacks to run, or None for all.
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

    datasets = selected_datasets if selected_datasets else all_datasets
    attacks_to_run = selected_attacks if selected_attacks else all_attacks_list

    print("\n" + "="*100)
    print("QUICK MULTI-DATASET TEST — PAPER-EXACT (CMFL)")
    print("="*100)
    print("Configuration:")
    print(f"  - Datasets: {', '.join(datasets)}")
    print(f"  - Attacks:  {', '.join(attacks_to_run)} (Byzantine only, CMFL paper Section 6.3)")
    print("  - Total Clients: FEMNIST=3550, Shakespeare=143, Sentiment140=772")
    print("  - Participating Clients: 10% of total per round (paper default)")
    print("  - Rounds: 1 (quick test)")
    print("  - Local Epochs: 1 (paper tau=1)")
    print("  - Dataset Fraction: 25%")
    print("")
    print("  Aggregation Schemes (7):")
    print("    FedAvg, Krum, Multi-Krum, Median, Trimmed Mean,")
    print("    CMFL (Strategy I), CMFL II (Strategy II)")
    print("")
    print("  Ablation Studies (7):")
    print("    2. Client Malicious %  (20%, 40%, 60%, 80%)   [CMFL paper ε]")
    print("    3. Client Particip. (10%-50%)")
    print("    4. Agg. Particip.   (pBFT, 10%-50%) [CMFL paper α]")
    print("    5. Committee Size   (10%-50%) [CMFL paper ω]")
    print("    6. Committee Mal.   (track malicious in committee) [CMFL paper Section 6.6]")
    print("    7. Non-IID          (0%, 20%, 40%, 60%, 80%)")
    print("    8. Efficiency       (CMFL paper Section 6.5, FEMNIST & Sentiment140 only)")
    print("="*100)

    # Create output directory
    output_base = f"Output/quick_femnist_ablation/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    Path(output_base).mkdir(parents=True, exist_ok=True)

    dataset_fraction = 0.25  # 25% of dataset
    num_clients = {
        'FEMNIST':       3550,
        'Fashion-MNIST': 3550,
        'EMNIST':        3550,
        'CIFAR-10':      3550,
        'Shakespeare':    143,
        'Sentiment140':   772,
    }
    rounds = 1
    local_epochs = 10

    all_attacks = attacks_to_run

    # 7 aggregation schemes on equal footing (CMFL-paper style)
    all_schemes = [
        'fedavg', 'krum', 'multi_krum', 'median', 'trimmed_mean',
        # 'adaptivecommittee',  # NOT in CMFL paper
        'cmfl', 'cmfl_ii',
    ]

    # ---------- accumulate results across datasets ----------
    all_results = {
        # 'data_poison_percentage':    {},  # NOT in CMFL paper
        'client_malicious_ratio':    {},
        'client_participation':      {},
        'agg_participation':         {},
        'committee_size':            {},
        'non_iid':                   {},
    }

    # ==========================================================================
    # OUTER LOOP: one dataset at a time (so clients_per_round = 20% of each)
    # ==========================================================================
    for ds in datasets:
        print(f"\n{'#'*100}")
        print(f"#  STARTING DATASET: {ds}")
        print(f"{'#'*100}\n")

        ds_list = [ds]
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
                ratios=[0.2, 0.4, 0.6, 0.8],
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
                participation_rates=[0.1, 0.2, 0.3, 0.4, 0.5],
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
                participation_fracs=[0.1, 0.2, 0.3, 0.4, 0.5],
                output_base=f"{output_base}/agg_participation",
                datasets=ds_list,
                rounds=rounds,
                total_clients=num_clients,
                client_participation=clients_per_round,
                malicious_ratio=0.4,
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
                committee_size_fracs=[0.1, 0.2, 0.3, 0.4, 0.5],
                output_base=f"{output_base}/committee_size",
                datasets=ds_list,
                rounds=rounds,
                total_clients=num_clients,
                client_participation=clients_per_round,
                malicious_ratio=0.4,
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
                ratios=[0.1, 0.2, 0.3, 0.4, 0.5],
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
                    ('20%', 1.0),      # Slightly Non-IID
                    ('40%', 0.5),      # Moderately Non-IID
                    ('60%', 0.3),      # Highly Non-IID
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

    # =============================================================================
    # Summary
    # =============================================================================
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


if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser(description="Quick multi-dataset ablation study")
    parser.add_argument('--dataset', '-d', type=str, default=None,
                        help="Single dataset: FEMNIST, Shakespeare, Sentiment140")
    parser.add_argument('--attack', '-a', type=str, default=None,
                        help="Single attack: gradient_scaling, same_value, back_gradient")
    args = parser.parse_args()

    # Normalize dataset name to expected capitalization
    _DS_MAP = {
        'femnist': 'FEMNIST', 'shakespeare': 'Shakespeare', 'sentiment140': 'Sentiment140'
    }
    sel_datasets = [_DS_MAP.get(args.dataset.lower(), args.dataset)] if args.dataset else None
    sel_attacks = [args.attack] if args.attack else None

    overall_start = time.time()
    results = run_full_ablation_study(
        selected_datasets=sel_datasets,
        selected_attacks=sel_attacks
    )
    overall_elapsed = time.time() - overall_start
    print(f"\n" + "="*100)
    print(f"OVERALL TIME: {overall_elapsed:.1f}s (~{overall_elapsed/60:.2f} minutes)")
    print("="*100)
