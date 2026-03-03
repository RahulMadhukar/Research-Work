#!/usr/bin/env python3
"""
Lightweight test configuration for rapid framework testing — PAPER-EXACT (CMFL).

Standard models, 50% dataset, 100 clients, dataset-specific rounds.
Only CMFL paper ablations (Sections 6.3-6.6).

Usage:
    python lightweight_test.py
    python lightweight_test.py -d femnist -a gradient_scaling
"""

import os
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
    BYZANTINE_ATTACKS,
    DATA_POISONING_ATTACKS,
)


def run_full_ablation_study(selected_datasets=None, selected_attacks=None):
    """
    Run CMFL paper ablation studies — one dataset at a time.

    Order: FEMNIST -> Shakespeare -> Sentiment140
    """

    all_datasets = ['FEMNIST', 'Shakespeare', 'Sentiment140']
    all_attacks_list = ['gradient_scaling', 'same_value', 'back_gradient']

    datasets_order = selected_datasets if selected_datasets else all_datasets
    attacks_to_run = selected_attacks if selected_attacks else all_attacks_list

    output_base = f"Output/lightweight_ablation/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    Path(output_base).mkdir(parents=True, exist_ok=True)

    dataset_fraction = 0.5
    num_clients = {
        'FEMNIST': 100, 'Shakespeare': 143, 'Sentiment140': 772,
    }
    dataset_rounds = {
        'FEMNIST': 100, 'Shakespeare': 500, 'Sentiment140': 1000,
    }
    all_attacks = attacks_to_run
    all_schemes = ['fedavg', 'multi_krum', 'cmfl', 'cmfl_ii']

    # Print experiment configuration
    from run_impact_analysis import DATASET_LR, DATASET_BATCH_SIZE
    print("\n" + "="*100)
    print("LIGHTWEIGHT ABLATION STUDY — CONFIGURATION")
    print("="*100)
    print(f"Datasets             : {datasets_order}")
    print(f"Attacks              : {attacks_to_run}")
    print(f"Schemes              : {all_schemes}")
    print(f"Dataset fraction     : {dataset_fraction}")
    print(f"Total clients        : {num_clients}")
    print(f"Rounds per dataset   : {dataset_rounds}")
    print(f"Learning rate        : {DATASET_LR}")
    print(f"Batch size           : {DATASET_BATCH_SIZE}")
    print("="*100)

    all_results = {
        'client_malicious_ratio': {},
        'agg_participation':      {},
        'committee_size':         {},
    }

    for ds in datasets_order:
        clients_per_round = max(1, int(num_clients[ds] * 0.1))
        print(f"\n[DATASET] {ds} (clients={num_clients[ds]}, active={clients_per_round}, rounds={dataset_rounds[ds]}, fraction={dataset_fraction}, lr={DATASET_LR.get(ds, 0.001)}, batch_size={DATASET_BATCH_SIZE.get(ds, 32)})")

        ds_list = [ds]

        # 1. Client Malicious Ratio (Section 6.3/6.4)
        try:
            res = run_client_malicious_ratio_ablation(
                ratios=[0.1, 0.3, 0.5],
                output_base=f"{output_base}/client_malicious_ratio",
                datasets=ds_list,
                rounds=dataset_rounds[ds],
                total_clients=num_clients,
                client_participation=clients_per_round,
                poison_percentage=0.4,
                attacks_to_test=all_attacks,
                agg_schemes=all_schemes,
                dataset_fraction=dataset_fraction
            )
            all_results['client_malicious_ratio'].update(res)
        except Exception as e:
            print(f"  [FAILED] Client Malicious Ratio: {e}")

        # 2. Aggregation Participation (Section 6.4, alpha)
        try:
            res = run_aggregation_participation_ablation(
                participation_fracs=[0.1, 0.3, 0.5],
                output_base=f"{output_base}/agg_participation",
                datasets=ds_list,
                rounds=dataset_rounds[ds],
                total_clients=num_clients,
                client_participation=clients_per_round,
                malicious_ratio=0.1,
                poison_percentage=0.4,
                attacks_to_test=all_attacks,
                dataset_fraction=dataset_fraction
            )
            all_results['agg_participation'].update(res)
        except Exception as e:
            print(f"  [FAILED] Aggregation Participation: {e}")

        # 3. Committee Size (Section 6.4, omega)
        try:
            res = run_committee_size_ablation(
                committee_size_fracs=[0.1, 0.3, 0.5],
                output_base=f"{output_base}/committee_size",
                datasets=ds_list,
                rounds=dataset_rounds[ds],
                total_clients=num_clients,
                client_participation=clients_per_round,
                malicious_ratio=0.1,
                poison_percentage=0.4,
                attacks_to_test=all_attacks,
                dataset_fraction=dataset_fraction
            )
            all_results['committee_size'].update(res)
        except Exception as e:
            print(f"  [FAILED] Committee Size: {e}")

        # 4. Committee Malicious Tracking (Section 6.6)
        try:
            res = run_committee_malicious_tracking_ablation(
                ratios=[0.1, 0.3, 0.5],
                output_base=f"{output_base}/committee_malicious_tracking",
                datasets=ds_list,
                rounds=dataset_rounds[ds],
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
            print(f"  [FAILED] Committee Malicious Tracking: {e}")

        # 5. Efficiency (Section 6.5) — FEMNIST & Sentiment140 only
        if ds in ('FEMNIST', 'Sentiment140'):
            try:
                res = run_efficiency_experiment(
                    datasets=ds_list,
                    rounds=dataset_rounds[ds],
                    total_clients=num_clients[ds],
                    transmission_rates=[1, 10, 100],
                    dataset_fraction=dataset_fraction,
                    output_base=f"{output_base}/efficiency",
                )
                all_results.setdefault('efficiency', {}).update(res)
            except Exception as e:
                print(f"  [FAILED] Efficiency: {e}")

        # 6-8: Client Participation, Data Poisoning, Non-IID — not in CMFL paper
        # (commented out, see docstring)

    print(f"\nAll ablations completed. Results in: {output_base}/")
    return all_results


if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser(description="Lightweight ablation study (CMFL paper)")
    parser.add_argument('--dataset', '-d', type=str, default=None,
                        help="Single dataset: FEMNIST, Shakespeare, Sentiment140")
    parser.add_argument('--attack', '-a', type=str, default=None,
                        help="Single attack: gradient_scaling, same_value, back_gradient")
    args = parser.parse_args()

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
    print(f"Total time: {elapsed:.1f}s (~{elapsed/60:.1f} min)")
