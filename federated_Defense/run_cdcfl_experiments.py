#!/usr/bin/env python3
"""
Master Experiment Runner for CD-CFL Byzantine Defense

Runs experiments E1-E7 with a shared cache to ensure ZERO duplicate runs.

Experiments:
  E1  Convergence: no attack, all methods
  E2  Robustness: all methods under 3 Byzantine attacks at eps=10%
  E3  Design Choice Ablation:
      E3-A: CDCFL-II layer ablation (5 variants)
      E3-B: CDCFL-I vs CDCFL-II strategy comparison (+ pBFT)
  E4  PoW effectiveness (from E2, no new runs)
  E5  Stress test: eps=10%-50%
  E6  Convergence speed (no attack)
  E7  Computational overhead (from E2, no new runs)

Usage:
    python run_cd_cfl_experiments.py                      # Run everything
    python run_cd_cfl_experiments.py -e E1                # Specific experiment
    python run_cd_cfl_experiments.py -d FEMNIST            # Specific dataset
    python run_cd_cfl_experiments.py -d femnist -a gradient_scaling -e E2
    python run_cd_cfl_experiments.py --resume Output/cd_cfl/20260305_120000/
    python run_cd_cfl_experiments.py --dev                 # Dev mode
"""

import os
import json
import argparse
import time
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

from evaluation import EvaluationFramework
from run_impact_analysis import (
    create_attack_scenario, test_defense_on_scenario,
    run_baseline_evaluation, DATASET_ROUNDS, DATASET_LR, DATASET_BATCH_SIZE,
)

# =============================================================================
# GLOBAL PARAMETERS
# =============================================================================
DATASETS = ['FEMNIST', 'Shakespeare', 'Sentiment140']
ATTACKS = ['gradient_scaling', 'same_value', 'back_gradient']
ROUNDS = {'FEMNIST': 600, 'Shakespeare': 500, 'Sentiment140': 1000}
LR = {'FEMNIST': 0.001, 'Shakespeare': 0.001, 'Sentiment140': 0.005}
NUM_CLIENTS = 100
DATASET_FRACTION = 0.25
EPS_DEFAULT = 0.1
ALPHA = 0.4   # aggregation participation fraction
OMEGA = 0.4   # committee size fraction
OUTPUT_PREFIX = 'cd_cfl'  # output subdirectory (overridden by lightweight wrapper)

# Dev mode overrides
DEV_ROUNDS = {'FEMNIST': 50, 'Shakespeare': 50, 'Sentiment140': 50}
DEV_FRACTION = 0.10

# Dataset name normalization (lowercase → canonical)
_DS = {'femnist': 'FEMNIST', 'shakespeare': 'Shakespeare', 'sentiment140': 'Sentiment140'}

# ---------------------------------------------------------------------------
# Method groups (override in lightweight_test.py for fewer baselines)
# ---------------------------------------------------------------------------
E2_METHODS = ['fedavg', 'krum', 'multi_krum', 'median', 'trimmed_mean',
              'cmfl', 'cmfl_ii', 'cdcfl_i', 'cdcfl_ii']
E5_METHODS = ['fedavg', 'cmfl', 'cmfl_ii', 'cdcfl_i', 'cdcfl_ii']
E6_METHODS = ['fedavg', 'cmfl', 'cmfl_ii', 'cdcfl_i', 'cdcfl_ii']

# CDCFL-II layer ablation variants
CD_CFL_VARIANTS = {
    'cdcfl_ii':             {'enable_pow': True,  'enable_filter': True,  'enable_robust': True},
    'cdcfl_ii_no_pow':      {'enable_pow': False, 'enable_filter': True,  'enable_robust': True},
    'cdcfl_ii_no_filter':   {'enable_pow': True,  'enable_filter': False, 'enable_robust': True},
    'cdcfl_ii_no_robust':   {'enable_pow': True,  'enable_filter': True,  'enable_robust': False},
    'cdcfl_ii_only_robust': {'enable_pow': False, 'enable_filter': False, 'enable_robust': True},
}

# E3-B strategy comparison variants
E3B_VARIANTS = ['cdcfl_i', 'cdcfl_i_pbft', 'cdcfl_ii_no_filter',
                'cdcfl_ii_no_pow', 'cdcfl_ii']

# Display labels and colors for plots
METHOD_LABELS = {
    'fedavg': 'FedAvg', 'krum': 'Krum', 'multi_krum': 'Multi-Krum',
    'median': 'Median', 'trimmed_mean': 'Trimmed Mean',
    'cmfl': 'CMFL-I', 'cmfl_ii': 'CMFL-II',
    'cdcfl_i': 'CDCFL-I', 'cdcfl_ii': 'CDCFL-II',
    'cdcfl_i_pbft': 'CDCFL-I (pBFT)',
    'cdcfl_ii_no_pow': 'CDCFL-II - PoW', 'cdcfl_ii_no_filter': 'CDCFL-II - Filter',
    'cdcfl_ii_no_robust': 'CDCFL-II - Robust', 'cdcfl_ii_only_robust': 'CDCFL-II only Robust',
}
METHOD_COLORS = {
    'fedavg': '#1f77b4', 'krum': '#aec7e8', 'multi_krum': '#ffbb78',
    'median': '#98df8a', 'trimmed_mean': '#ff9896',
    'cmfl': '#ff7f0e', 'cmfl_ii': '#2ca02c',
    'cdcfl_i': '#d62728', 'cdcfl_ii': '#9467bd',
    'cdcfl_i_pbft': '#e377c2',
    'cdcfl_ii_no_pow': '#8c564b', 'cdcfl_ii_no_filter': '#bcbd22',
    'cdcfl_ii_no_robust': '#17becf', 'cdcfl_ii_only_robust': '#7f7f7f',
}
E3A_LABELS = {
    'cdcfl_ii': 'Full', 'cdcfl_ii_no_pow': '- PoW',
    'cdcfl_ii_no_filter': '- Filter', 'cdcfl_ii_no_robust': '- Robust',
    'cdcfl_ii_only_robust': 'Only Robust',
}
E3B_LABELS = {
    'cdcfl_i': 'CDCFL-I (PoW)', 'cdcfl_i_pbft': 'CDCFL-I (pBFT)',
    'cdcfl_ii_no_filter': '+ PoW prefilter', 'cdcfl_ii_no_pow': '+ Outlier filter',
    'cdcfl_ii': 'CDCFL-II (full)',
}


# =============================================================================
# SHARED RESULTS CACHE
# =============================================================================
RESULTS_CACHE = {}


def _cache_key(dataset, method, attack, epsilon, alpha=None, omega=None, rounds=None):
    a = alpha if alpha is not None else ALPHA
    o = omega if omega is not None else OMEGA
    r = rounds if rounds is not None else ROUNDS.get(dataset, 500)
    return (dataset, method, attack, epsilon, a, o, r)


def run_or_fetch(key, run_fn):
    """Run if not cached."""
    if key in RESULTS_CACHE:
        print(f"  [CACHE HIT] {key[:4]}")
        return RESULTS_CACHE[key]
    result = run_fn()
    RESULTS_CACHE[key] = result
    return result


def save_cache_checkpoint(output_dir):
    """Save cache to disk for resumability."""
    cache_path = os.path.join(output_dir, 'results_cache.json')
    serializable = {}
    for key, val in RESULTS_CACHE.items():
        str_key = '|'.join(str(k) for k in key)
        serializable[str_key] = _serialize_result(val)
    with open(cache_path, 'w') as f:
        json.dump(serializable, f, indent=2)


def load_cache_checkpoint(resume_dir):
    """Load cache from disk."""
    cache_path = os.path.join(resume_dir, 'results_cache.json')
    if not os.path.exists(cache_path):
        print(f"[WARN] No cache found at {cache_path}")
        return
    with open(cache_path, 'r') as f:
        serializable = json.load(f)
    for str_key, val in serializable.items():
        parts = str_key.split('|')
        key = (parts[0], parts[1], parts[2], float(parts[3]),
               float(parts[4]), float(parts[5]), int(parts[6]))
        RESULTS_CACHE[key] = val
    print(f"[CACHE] Loaded {len(RESULTS_CACHE)} cached results from {cache_path}")


def _serialize_result(result):
    """Make result JSON-serializable."""
    if result is None:
        return None
    out = {}
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.float32, np.float64)):
            out[k] = float(v)
        elif isinstance(v, (np.int32, np.int64)):
            out[k] = int(v)
        elif isinstance(v, dict):
            out[k] = _serialize_result(v)
        elif isinstance(v, list):
            out[k] = [float(x) if isinstance(x, (np.float32, np.float64)) else x for x in v]
        else:
            out[k] = v
    return out


def save_results_json(output_dir):
    """Save results.json in simple format for notebook compatibility."""
    serializable = {}
    for key, val in RESULTS_CACHE.items():
        ds, method, attack, eps, alpha, omega, rds = key
        simple_key = f"{ds}|{method}|{attack}|{eps}"
        if val is None:
            serializable[simple_key] = None
            continue
        serializable[simple_key] = {
            'final_accuracy': val.get('final_accuracy', 0),
            'asr': val.get('attack_success_rate', 0),
            'acc_history': val.get('acc_history', []),
            'loss_history': val.get('loss_history', []),
            'detection': val.get('detection_metrics', {}),
        }
    fpath = os.path.join(output_dir, 'results.json')
    with open(fpath, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"  [JSON] {fpath}")


# =============================================================================
# CORE RUN FUNCTION
# =============================================================================

def _resolve_method(method):
    """Resolve method name to (aggregation_scheme, extra_kwargs)."""
    if method == 'cdcfl_i':
        return 'cdcfl_i', {}
    elif method == 'cdcfl_i_pbft':
        return 'cdcfl_i', {'finalization_method': 'pbft'}
    elif method in CD_CFL_VARIANTS:
        return 'cdcfl_ii', dict(CD_CFL_VARIANTS[method])
    else:
        return method, {}


def run_single_experiment(framework, dataset, method, attack, epsilon,
                          rounds, fraction, clients_per_round):
    """Run a single FL experiment and return result dict."""
    n_clients = NUM_CLIENTS
    malicious_clients = max(1, int(n_clients * epsilon))
    lr = LR.get(dataset, 0.001)
    bs = DATASET_BATCH_SIZE.get(dataset, 32)
    agg, extra = _resolve_method(method)

    tag = f"[{dataset}][{METHOD_LABELS.get(method, method.upper())}][{attack}][eps={epsilon}]"
    print(f"\n{'='*70}")
    print(f"  {tag} Starting — {rounds} rounds")
    print(f"{'='*70}")

    start_time = time.time()

    if attack == 'none':
        final_acc, acc_hist, loss_hist = run_baseline_evaluation(
            framework=framework, dataset_name=dataset, num_clients=n_clients,
            rounds=rounds, clients_per_round=clients_per_round, aggregation=agg,
            alpha=1.0, dataset_fraction=fraction,
            aggregation_participation_frac=ALPHA, committee_size_frac=OMEGA,
            lr=lr, **extra
        )
        total_time = time.time() - start_time
        return {
            'final_accuracy': float(final_acc),
            'acc_history': [float(x) for x in acc_hist],
            'loss_history': [float(x) for x in loss_hist],
            'attack_success_rate': 0.0,
            'detection_metrics': {},
            'total_time': total_time,
            'config': {'dataset': dataset, 'method': method, 'attack': attack,
                       'epsilon': epsilon, 'rounds': rounds},
        }
    else:
        clients, test_loader, test_X, y_test = create_attack_scenario(
            framework=framework, dataset_name=dataset, attack_type=attack,
            poison_percentage=0.0, num_clients=n_clients,
            malicious_clients=malicious_clients, alpha=1.0, dataset_fraction=fraction,
        )
        if clients is None:
            print(f"  {tag} FAILED to create scenario")
            return None

        for c in clients:
            c.snapshot_state()

        final_acc, asr, det_metrics, acc_hist, loss_hist = test_defense_on_scenario(
            clients_list=clients, test_loader=test_loader, test_X=test_X, y_test=y_test,
            rounds=rounds, aggregation=agg, clients_per_round=clients_per_round,
            framework=framework, dataset_name=dataset, attack_type=attack,
            aggregation_participation_frac=ALPHA, committee_size_frac=OMEGA,
            lr=lr, batch_size=bs, **extra
        )

        total_time = time.time() - start_time
        return {
            'final_accuracy': float(final_acc) if final_acc is not None else 0.0,
            'acc_history': [float(x) for x in (acc_hist or [])],
            'loss_history': [float(x) for x in (loss_hist or [])],
            'attack_success_rate': float(asr) if asr is not None else 0.0,
            'detection_metrics': det_metrics or {},
            'total_time': total_time,
            'config': {'dataset': dataset, 'method': method, 'attack': attack,
                       'epsilon': epsilon, 'rounds': rounds},
        }


# =============================================================================
# HELPER
# =============================================================================

def _run_cached(framework, ds, method, attack, eps, rds, fraction, cpr, output_dir):
    """Run or fetch a single experiment with caching."""
    key = _cache_key(ds, method, attack, eps, ALPHA, OMEGA, rds)
    return run_or_fetch(key, lambda: run_single_experiment(
        framework, ds, method, attack, eps, rds, fraction, cpr
    ))


# =============================================================================
# EXPERIMENT GROUPS
# =============================================================================

def run_E1(framework, datasets, rounds_map, fraction, cpr, output_dir):
    """E1: Convergence (no attack) — all methods."""
    print("\n" + "="*80)
    print("  E1: Convergence (no attack)")
    print("="*80)
    for ds in datasets:
        rds = rounds_map[ds]
        for method in E6_METHODS:  # same method set as E6
            _run_cached(framework, ds, method, 'none', 0.0, rds, fraction, cpr, output_dir)
        save_cache_checkpoint(output_dir)


def run_E2(framework, datasets, attacks, rounds_map, fraction, cpr, output_dir):
    """E2: All methods under attack (main robustness table)."""
    print("\n" + "="*80)
    print(f"  E2: All Methods Under Attack (eps={EPS_DEFAULT:.0%})")
    print("="*80)
    for ds in datasets:
        rds = rounds_map[ds]
        for method in E2_METHODS:
            for attack in attacks:
                _run_cached(framework, ds, method, attack, EPS_DEFAULT, rds, fraction, cpr, output_dir)
            save_cache_checkpoint(output_dir)

    # Summary table
    print("\n" + "="*80)
    print("  E2 SUMMARY — Final Accuracy (%)")
    print("="*80)
    for ds in datasets:
        print(f"\n  {ds}:")
        hdr = f"  {'Method':18s}"
        for atk in attacks:
            hdr += f" | {atk:>18s}"
        print(hdr)
        print("  " + "-"*(18 + 21*len(attacks)))
        for m in E2_METHODS:
            row = f"  {METHOD_LABELS.get(m, m):18s}"
            for atk in attacks:
                key = _cache_key(ds, m, atk, EPS_DEFAULT, ALPHA, OMEGA, rounds_map[ds])
                r = RESULTS_CACHE.get(key)
                row += f" | {r['final_accuracy']*100:>17.2f}%" if r else f" | {'FAIL':>18s}"
            print(row)


def run_E3(framework, datasets, attacks, rounds_map, fraction, cpr, output_dir):
    """E3: Design Choice Ablation — E3-A (layers) + E3-B (strategies)."""

    # --- E3-A: CDCFL-II Layer Ablation ---
    print("\n" + "="*80)
    print("  E3-A: CDCFL-II Layer Ablation")
    print("="*80)

    e3a_variants = list(CD_CFL_VARIANTS.keys())
    for ds in datasets:
        rds = rounds_map[ds]
        for variant in e3a_variants:
            for attack in attacks:
                _run_cached(framework, ds, variant, attack, EPS_DEFAULT, rds, fraction, cpr, output_dir)
        save_cache_checkpoint(output_dir)

    # E3-A summary
    for ds in datasets:
        for atk in attacks:
            print(f"\n  E3-A: {ds} / {atk}")
            print(f"  {'Variant':<35s} | Accuracy")
            print(f"  {'-'*50}")
            for v in e3a_variants:
                key = _cache_key(ds, v, atk, EPS_DEFAULT, ALPHA, OMEGA, rounds_map[ds])
                r = RESULTS_CACHE.get(key)
                acc = f"{r['final_accuracy']*100:.1f}%" if r else "N/A"
                print(f"  {E3A_LABELS.get(v, v):<35s} | {acc}")

    # --- E3-B: Strategy Comparison ---
    print("\n" + "="*80)
    print("  E3-B: CDCFL-I vs CDCFL-II Strategy Comparison")
    print("  (includes PoW vs pBFT finalization test)")
    print("="*80)

    for ds in datasets:
        rds = rounds_map[ds]
        for variant in E3B_VARIANTS:
            for attack in attacks:
                _run_cached(framework, ds, variant, attack, EPS_DEFAULT, rds, fraction, cpr, output_dir)
        save_cache_checkpoint(output_dir)

    # E3-B summary
    for ds in datasets:
        for atk in attacks:
            print(f"\n  E3-B Strategy Comparison: {ds} / {atk}")
            print(f"  {'Variant':<35s} | Accuracy")
            print(f"  {'-'*50}")
            for v in E3B_VARIANTS:
                key = _cache_key(ds, v, atk, EPS_DEFAULT, ALPHA, OMEGA, rounds_map[ds])
                r = RESULTS_CACHE.get(key)
                acc = f"{r['final_accuracy']*100:.1f}%" if r else "N/A"
                print(f"  {E3B_LABELS.get(v, v):<35s} | {acc}")

    # Combined summary
    print("\n" + "="*80)
    print("  E3 COMBINED SUMMARY")
    print("="*80)
    for ds in datasets:
        print(f"\n  {ds}:")
        atk_hdr = f"  {'Variant':<35s}"
        for atk in attacks:
            atk_hdr += f" | {atk:>18s}"
        print(atk_hdr)
        print("  " + "-"*(35 + 21*len(attacks)))
        for v in E3B_VARIANTS:
            row = f"  {E3B_LABELS.get(v, v):<35s}"
            for atk in attacks:
                key = _cache_key(ds, v, atk, EPS_DEFAULT, ALPHA, OMEGA, rounds_map[ds])
                r = RESULTS_CACHE.get(key)
                acc = f"{r['final_accuracy']*100:.1f}%" if r else "N/A"
                row += f" | {acc:>18s}"
            print(row)
        print(f"  --- CDCFL-II Layer Ablation ---")
        for v in e3a_variants:
            row = f"  {E3A_LABELS.get(v, v):<35s}"
            for atk in attacks:
                key = _cache_key(ds, v, atk, EPS_DEFAULT, ALPHA, OMEGA, rounds_map[ds])
                r = RESULTS_CACHE.get(key)
                acc = f"{r['final_accuracy']*100:.1f}%" if r else "N/A"
                row += f" | {acc:>18s}"
            print(row)


def run_E4(datasets, attacks, rounds_map):
    """E4: PoW effectiveness analysis (from E2 results, no new runs)."""
    print("\n" + "="*80)
    print("  E4: PoW Effectiveness Analysis")
    print("="*80)
    for ds in datasets:
        print(f"\n  {ds}:")
        hdr = f"  {'Attack':20s} | {'Norm Rej':>10s} | {'Loss Rej':>10s} | {'NaN Rej':>10s} | {'Total Rej':>10s}"
        print(hdr)
        print("  " + "-"*72)
        for atk in attacks:
            key = _cache_key(ds, 'cdcfl_ii', atk, EPS_DEFAULT, ALPHA, OMEGA, rounds_map[ds])
            r = RESULTS_CACHE.get(key)
            if r is None:
                print(f"  {atk:20s} | {'N/A (run E2 first)':>46s}")
                continue
            det = r.get('detection_metrics', {})
            norm_r = det.get('pow_norm_rejected', 0)
            loss_r = det.get('pow_loss_rejected', 0)
            nan_r  = det.get('pow_nan_rejected', 0)
            total_r = det.get('pow_total_rejected', norm_r + loss_r + nan_r)
            print(f"  {atk:20s} | {norm_r:>10} | {loss_r:>10} | {nan_r:>10} | {total_r:>10}")
    print("\n  Note: PoW rejection counts are cumulative across all rounds.")


def run_E5(framework, datasets, attacks, rounds_map, fraction, cpr, output_dir):
    """E5: Stress test (eps 10%-50%)."""
    print("\n" + "="*80)
    print("  E5: Stress Test (eps 10%-50%)")
    print("="*80)
    epsilons = [0.1, 0.2, 0.3, 0.4, 0.5]
    for ds in datasets:
        rds = rounds_map[ds]
        for method in E5_METHODS:
            for attack in attacks:
                for eps in epsilons:
                    _run_cached(framework, ds, method, attack, eps, rds, fraction, cpr, output_dir)
            save_cache_checkpoint(output_dir)

    # Summary
    print("\n" + "="*80)
    print("  E5 STRESS TEST — Final Accuracy (%)")
    print("="*80)
    for ds in datasets:
        for atk in attacks:
            print(f"\n  {ds} / {atk}:")
            hdr = f"  {'Method':18s}"
            for eps in epsilons:
                hdr += f" |  {eps:<5.0%}"
            print(hdr)
            print("  " + "-"*(18 + 9*len(epsilons)))
            for m in E5_METHODS:
                row = f"  {METHOD_LABELS.get(m, m):18s}"
                for eps in epsilons:
                    key = _cache_key(ds, m, atk, eps, ALPHA, OMEGA, rounds_map[ds])
                    r = RESULTS_CACHE.get(key)
                    row += f" | {r['final_accuracy']*100:>5.1f}%" if r else f" | {'FAIL':>6s}"
                print(row)


def run_E6(framework, datasets, rounds_map, fraction, cpr, output_dir):
    """E6: Convergence speed (no attack)."""
    print("\n" + "="*80)
    print("  E6: Convergence Speed (no attack)")
    print("="*80)
    for ds in datasets:
        rds = rounds_map[ds]
        for method in E6_METHODS:
            _run_cached(framework, ds, method, 'none', 0.0, rds, fraction, cpr, output_dir)
        save_cache_checkpoint(output_dir)

    # Summary
    print("\n  Convergence Summary — Final Accuracy (%):")
    for ds in datasets:
        print(f"\n  {ds}:")
        for m in E6_METHODS:
            key = _cache_key(ds, m, 'none', 0.0, ALPHA, OMEGA, rounds_map[ds])
            r = RESULTS_CACHE.get(key)
            if r:
                acc = r['final_accuracy'] * 100
                acc_h = r.get('acc_history', [])
                half_target = r['final_accuracy'] * 0.5
                half_round = '-'
                for i, a in enumerate(acc_h):
                    if a >= half_target:
                        half_round = str(i + 1)
                        break
                print(f"    {METHOD_LABELS.get(m, m):18s}: {acc:6.2f}%  (50% reached at round {half_round})")
            else:
                print(f"    {METHOD_LABELS.get(m, m):18s}: FAIL")


def run_E7(datasets, attacks, rounds_map):
    """E7: Computational overhead (from E2 results, no new runs)."""
    print("\n" + "="*80)
    print("  E7: Computational Overhead")
    print("="*80)
    methods = ['fedavg', 'cmfl', 'cdcfl_ii']
    for ds in datasets:
        print(f"\n  {ds}:")
        hdr = f"  {'Method':18s} | {'Total Time':>12s} | {'Time/Round':>12s} | {'PoW':>8s} | {'Filter':>8s} | {'Agg':>8s}"
        print(hdr)
        print("  " + "-"*78)
        rds = rounds_map[ds]
        for m in methods:
            found = False
            for atk in attacks:
                key = _cache_key(ds, m, atk, EPS_DEFAULT, ALPHA, OMEGA, rds)
                r = RESULTS_CACHE.get(key)
                if r and 'total_time' in r:
                    found = True
                    break
            if not found:
                print(f"  {METHOD_LABELS.get(m, m):18s} | {'N/A (run E2 first)':>60s}")
                continue
            n_rounds = len(r.get('acc_history', [])) or 1
            dm = r.get('detection_metrics', {})
            avg_pow = dm.get('avg_pow_time', 0.0)
            avg_filt = dm.get('avg_filter_time', 0.0)
            avg_agg = dm.get('avg_agg_time', 0.0)
            if avg_pow > 0 or avg_filt > 0 or avg_agg > 0:
                total_overhead = (avg_pow + avg_filt + avg_agg) * n_rounds
                print(f"  {METHOD_LABELS.get(m, m):18s} | {total_overhead:>10.1f}s | "
                      f"{(total_overhead/n_rounds):>10.3f}s | "
                      f"{avg_pow:>7.3f}s | {avg_filt:>7.3f}s | {avg_agg:>7.3f}s")
            else:
                print(f"  {METHOD_LABELS.get(m, m):18s} |          - |            - |        - |        - |        -")
    print("\n  Note: PoW/Filter/Agg breakdown only available for CDCFL-II.")


# =============================================================================
# PLOTTING
# =============================================================================

def _save_fig(fig, path_prefix):
    """Save figure as PNG and EPS."""
    fig.savefig(f"{path_prefix}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{path_prefix}.eps", bbox_inches='tight')
    plt.close(fig)
    print(f"  [PLOT] {path_prefix}.png + .eps")


def plot_convergence_curves(rounds_map, datasets, save_path):
    """E1/E6: Accuracy vs rounds for all methods (no attack)."""
    ncols = len(datasets)
    fig, axes = plt.subplots(1, ncols, figsize=(6*ncols, 5), squeeze=False)
    for di, ds in enumerate(datasets):
        ax = axes[0][di]
        rds = rounds_map[ds]
        for method in E6_METHODS:
            key = _cache_key(ds, method, 'none', 0.0, ALPHA, OMEGA, rds)
            res = RESULTS_CACHE.get(key)
            if res and res.get('acc_history'):
                ax.plot([x*100 for x in res['acc_history']],
                        label=METHOD_LABELS.get(method, method),
                        color=METHOD_COLORS.get(method), linewidth=1.5)
        ax.set_xlabel('Round')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title(ds)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Convergence (No Attack)', fontsize=14)
    fig.tight_layout()
    _save_fig(fig, save_path)


def plot_robustness_grid(rounds_map, datasets, attacks, save_path):
    """E2: Grid of accuracy curves — attacks x datasets."""
    nrows, ncols = len(attacks), len(datasets)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows), squeeze=False)
    for ai, attack in enumerate(attacks):
        for di, ds in enumerate(datasets):
            ax = axes[ai][di]
            rds = rounds_map[ds]
            for method in E2_METHODS:
                key = _cache_key(ds, method, attack, EPS_DEFAULT, ALPHA, OMEGA, rds)
                res = RESULTS_CACHE.get(key)
                if res and res.get('acc_history'):
                    ax.plot([x*100 for x in res['acc_history']],
                            label=METHOD_LABELS.get(method, method),
                            color=METHOD_COLORS.get(method), linewidth=1.2)
            ax.set_title(f"{ds} / {attack}")
            ax.set_xlabel('Round')
            ax.set_ylabel('Acc (%)')
            ax.grid(True, alpha=0.3)
            if ai == 0 and di == ncols-1:
                ax.legend(fontsize=5, loc='lower right')
    fig.suptitle(f'Robustness Under Attack (eps={EPS_DEFAULT:.0%})', fontsize=14)
    fig.tight_layout()
    _save_fig(fig, save_path)


def plot_robustness_bars(rounds_map, datasets, attacks, save_path):
    """E2: Grouped bar chart — methods x attacks, one per dataset."""
    for ds in datasets:
        n_atk, n_m = len(attacks), len(E2_METHODS)
        fig, ax = plt.subplots(figsize=(max(8, 3*n_atk), 6))
        x = np.arange(n_atk)
        width = 0.8 / n_m
        rds = rounds_map[ds]
        for i, m in enumerate(E2_METHODS):
            accs = []
            for atk in attacks:
                key = _cache_key(ds, m, atk, EPS_DEFAULT, ALPHA, OMEGA, rds)
                r = RESULTS_CACHE.get(key)
                accs.append(r['final_accuracy']*100 if r else 0)
            bars = ax.bar(x + i*width - 0.4 + width/2, accs, width,
                          label=METHOD_LABELS.get(m, m), color=METHOD_COLORS.get(m))
            for bar, acc in zip(bars, accs):
                if acc > 0:
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                            f'{acc:.1f}', ha='center', va='bottom', fontsize=5)
        ax.set_title(f'{ds} — Robustness (eps={EPS_DEFAULT:.0%})', fontsize=14)
        ax.set_xlabel('Attack')
        ax.set_ylabel('Final Accuracy (%)')
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace('_', '\n') for a in attacks], fontsize=9)
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        _save_fig(fig, f"{save_path}_{ds}")


def plot_E3A_ablation(rounds_map, datasets, attacks, save_path):
    """E3-A: CDCFL-II layer ablation bar chart."""
    variants = list(CD_CFL_VARIANTS.keys())
    for ds in datasets:
        n_atk, n_v = len(attacks), len(variants)
        fig, ax = plt.subplots(figsize=(max(8, 3*n_atk), 6))
        x = np.arange(n_atk)
        width = 0.8 / n_v
        rds = rounds_map[ds]
        for i, v in enumerate(variants):
            accs = []
            for atk in attacks:
                key = _cache_key(ds, v, atk, EPS_DEFAULT, ALPHA, OMEGA, rds)
                r = RESULTS_CACHE.get(key)
                accs.append(r['final_accuracy']*100 if r else 0)
            ax.bar(x + i*width - 0.4 + width/2, accs, width,
                   label=E3A_LABELS.get(v, v), color=METHOD_COLORS.get(v))
        ax.set_title(f'{ds} — E3-A: CDCFL-II Layer Ablation', fontsize=14)
        ax.set_xlabel('Attack')
        ax.set_ylabel('Final Accuracy (%)')
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace('_', '\n') for a in attacks], fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        _save_fig(fig, f"{save_path}_{ds}")


def plot_E3B_strategy(rounds_map, datasets, attacks, save_path):
    """E3-B: Strategy comparison bar chart."""
    for ds in datasets:
        n_atk, n_v = len(attacks), len(E3B_VARIANTS)
        fig, ax = plt.subplots(figsize=(max(8, 3*n_atk), 6))
        x = np.arange(n_atk)
        width = 0.8 / n_v
        rds = rounds_map[ds]
        for i, v in enumerate(E3B_VARIANTS):
            accs = []
            for atk in attacks:
                key = _cache_key(ds, v, atk, EPS_DEFAULT, ALPHA, OMEGA, rds)
                r = RESULTS_CACHE.get(key)
                accs.append(r['final_accuracy']*100 if r else 0)
            ax.bar(x + i*width - 0.4 + width/2, accs, width,
                   label=E3B_LABELS.get(v, v), color=METHOD_COLORS.get(v))
        ax.set_title(f'{ds} — E3-B: Strategy Comparison', fontsize=14)
        ax.set_xlabel('Attack')
        ax.set_ylabel('Final Accuracy (%)')
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace('_', '\n') for a in attacks], fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        _save_fig(fig, f"{save_path}_{ds}")


def plot_pow_analysis(rounds_map, datasets, attacks, save_path):
    """E4: Stacked bar chart of PoW rejection breakdown."""
    ncols = len(datasets)
    fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 5), squeeze=False)
    for di, ds in enumerate(datasets):
        ax = axes[0][di]
        rds = rounds_map[ds]
        norm_rej, loss_rej, nan_rej = [], [], []
        for atk in attacks:
            key = _cache_key(ds, 'cdcfl_ii', atk, EPS_DEFAULT, ALPHA, OMEGA, rds)
            res = RESULTS_CACHE.get(key)
            dm = res.get('detection_metrics', {}) if res else {}
            norm_rej.append(dm.get('pow_norm_rejected', 0))
            loss_rej.append(dm.get('pow_loss_rejected', 0))
            nan_rej.append(dm.get('pow_nan_rejected', 0))
        x = np.arange(len(attacks))
        ax.bar(x, norm_rej, label='Norm Rejected')
        ax.bar(x, loss_rej, bottom=norm_rej, label='Loss Rejected')
        b2 = [n+l for n, l in zip(norm_rej, loss_rej)]
        ax.bar(x, nan_rej, bottom=b2, label='NaN Rejected')
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace('_', '\n') for a in attacks], fontsize=8)
        ax.set_ylabel('Total Rejections')
        ax.set_title(ds)
        ax.legend(fontsize=7)
    fig.suptitle('PoW Rejection Breakdown', fontsize=14)
    fig.tight_layout()
    _save_fig(fig, save_path)


def plot_stress_lines(rounds_map, datasets, attacks, save_path):
    """E5: Accuracy vs epsilon for all methods — grid of attacks x datasets."""
    epsilons = [0.1, 0.2, 0.3, 0.4, 0.5]
    nrows, ncols = len(attacks), len(datasets)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows), squeeze=False)
    for ai, atk in enumerate(attacks):
        for di, ds in enumerate(datasets):
            ax = axes[ai][di]
            rds = rounds_map[ds]
            for method in E5_METHODS:
                accs = []
                for eps in epsilons:
                    key = _cache_key(ds, method, atk, eps, ALPHA, OMEGA, rds)
                    r = RESULTS_CACHE.get(key)
                    accs.append(r['final_accuracy']*100 if r else 0)
                ax.plot(epsilons, accs, 'o-',
                        label=METHOD_LABELS.get(method, method),
                        color=METHOD_COLORS.get(method), linewidth=1.5)
            ax.set_xlabel('Malicious Ratio')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(f"{ds} / {atk}")
            ax.set_xticks(epsilons)
            ax.set_xticklabels([f'{e:.0%}' for e in epsilons])
            ax.grid(True, alpha=0.3)
            if ai == 0 and di == 0:
                ax.legend(fontsize=7)
    fig.suptitle('Stress Test: Accuracy vs Malicious Ratio', fontsize=14)
    fig.tight_layout()
    _save_fig(fig, save_path)


def plot_overhead_bars(rounds_map, datasets, attacks, save_path):
    """E7: Computational overhead breakdown."""
    methods = ['fedavg', 'cmfl', 'cdcfl_ii']
    ncols = len(datasets)
    fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 5), squeeze=False)
    for di, ds in enumerate(datasets):
        ax = axes[0][di]
        rds = rounds_map[ds]
        total_times, pow_times, filter_times, agg_times = [], [], [], []
        for m in methods:
            found = False
            for atk in attacks:
                key = _cache_key(ds, m, atk, EPS_DEFAULT, ALPHA, OMEGA, rds)
                res = RESULTS_CACHE.get(key)
                if res and 'total_time' in res:
                    avg_total = res['total_time'] / max(1, rds)
                    total_times.append(avg_total)
                    dm = res.get('detection_metrics', {})
                    pow_times.append(dm.get('avg_pow_time', 0.0))
                    filter_times.append(dm.get('avg_filter_time', 0.0))
                    agg_times.append(dm.get('avg_agg_time', 0.0))
                    found = True
                    break
            if not found:
                total_times.append(0)
                pow_times.append(0)
                filter_times.append(0)
                agg_times.append(0)
        x = np.arange(len(methods))
        train_t = [max(0, t-p-f-a) for t, p, f, a in zip(total_times, pow_times, filter_times, agg_times)]
        ax.bar(x, train_t, label='Training')
        ax.bar(x, pow_times, bottom=train_t, label='PoW')
        b2 = [tr+pw for tr, pw in zip(train_t, pow_times)]
        ax.bar(x, filter_times, bottom=b2, label='Filter')
        b3 = [b+f for b, f in zip(b2, filter_times)]
        ax.bar(x, agg_times, bottom=b3, label='Aggregation')
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods])
        ax.set_ylabel('Avg Time / Round (s)')
        ax.set_title(ds)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')
    fig.suptitle('Computational Overhead Breakdown', fontsize=14)
    fig.tight_layout()
    _save_fig(fig, save_path)


# =============================================================================
# TABLE GENERATION
# =============================================================================

def generate_latex_table(rounds_map, datasets, attacks, save_path):
    """LaTeX table: methods x (attacks x datasets)."""
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        r'\caption{Test Accuracy (\%) Under Byzantine Attacks}',
        r'\begin{tabular}{l' + 'c' * (len(attacks) * len(datasets)) + '}',
        r'\toprule',
    ]
    header = 'Method'
    for ds in datasets:
        for atk in attacks:
            header += f' & {ds[:3]}/{atk[:4]}'
    header += r' \\'
    lines.append(header)
    lines.append(r'\midrule')
    for method in E2_METHODS:
        row = METHOD_LABELS.get(method, method).replace('_', r'\_')
        for ds in datasets:
            rds = rounds_map[ds]
            for atk in attacks:
                key = _cache_key(ds, method, atk, EPS_DEFAULT, ALPHA, OMEGA, rds)
                res = RESULTS_CACHE.get(key)
                acc = res['final_accuracy'] * 100 if res else 0.0
                row += f' & {acc:.1f}'
        row += r' \\'
        lines.append(row)
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    tex = '\n'.join(lines)
    with open(f"{save_path}.tex", 'w') as f:
        f.write(tex)
    print(f"  [TABLE] {save_path}.tex")


def generate_summary_csv(save_path):
    """One row per run: config + metrics."""
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'method', 'attack', 'epsilon', 'rounds',
                          'final_accuracy', 'best_accuracy', 'asr', 'total_time'])
        for key, res in sorted(RESULTS_CACHE.items()):
            if res is None:
                continue
            ds, method, attack, eps, alpha, omega, rds = key
            best_acc = max(res.get('acc_history', [0.0])) if res.get('acc_history') else 0.0
            asr = res.get('attack_success_rate', 0.0)
            writer.writerow([
                ds, METHOD_LABELS.get(method, method), attack, eps, rds,
                f"{res.get('final_accuracy', 0.0):.6f}",
                f"{best_acc:.6f}",
                f"{asr:.6f}",
                f"{res.get('total_time', 0.0):.1f}",
            ])
    print(f"  [CSV] {save_path}")


def generate_E3_csv(rounds_map, datasets, attacks, save_path):
    """E3 ablation CSV tables."""
    with open(f"{save_path}_E3A.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Dataset', 'Variant', 'Attack', 'Accuracy(%)'])
        for ds in datasets:
            rds = rounds_map[ds]
            for v in CD_CFL_VARIANTS:
                for atk in attacks:
                    key = _cache_key(ds, v, atk, EPS_DEFAULT, ALPHA, OMEGA, rds)
                    r = RESULTS_CACHE.get(key)
                    w.writerow([ds, E3A_LABELS.get(v, v), atk,
                                f"{r['final_accuracy']*100:.2f}" if r else 'N/A'])
    print(f"  [CSV] {save_path}_E3A.csv")
    with open(f"{save_path}_E3B.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Dataset', 'Variant', 'Attack', 'Accuracy(%)'])
        for ds in datasets:
            rds = rounds_map[ds]
            for v in E3B_VARIANTS:
                for atk in attacks:
                    key = _cache_key(ds, v, atk, EPS_DEFAULT, ALPHA, OMEGA, rds)
                    r = RESULTS_CACHE.get(key)
                    w.writerow([ds, E3B_LABELS.get(v, v), atk,
                                f"{r['final_accuracy']*100:.2f}" if r else 'N/A'])
    print(f"  [CSV] {save_path}_E3B.csv")


def generate_E5_csv(rounds_map, datasets, attacks, save_path):
    """E5 stress test CSV."""
    epsilons = [0.1, 0.2, 0.3, 0.4, 0.5]
    with open(save_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Dataset', 'Method', 'Attack', 'Epsilon', 'Accuracy(%)'])
        for ds in datasets:
            rds = rounds_map[ds]
            for m in E5_METHODS:
                for atk in attacks:
                    for eps in epsilons:
                        key = _cache_key(ds, m, atk, eps, ALPHA, OMEGA, rds)
                        r = RESULTS_CACHE.get(key)
                        w.writerow([ds, METHOD_LABELS.get(m, m), atk, f"{eps:.0%}",
                                    f"{r['final_accuracy']*100:.2f}" if r else 'N/A'])
    print(f"  [CSV] {save_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CD-CFL Defense Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_cd_cfl_experiments.py                      # Run everything
  python run_cd_cfl_experiments.py -e E2                # Specific experiment
  python run_cd_cfl_experiments.py -d femnist            # Specific dataset
  python run_cd_cfl_experiments.py -d femnist -a gradient_scaling -e E2
  python run_cd_cfl_experiments.py --resume Output/cd_cfl/20260305_120000/
  python run_cd_cfl_experiments.py --dev                 # Dev mode
""")
    parser.add_argument('-e', '--experiment', type=str, default=None,
                        help='E1-E7 (comma-separated, default: all)')
    parser.add_argument('-d', '--dataset', type=str, default=None,
                        help='Dataset filter (femnist, shakespeare, sentiment140)')
    parser.add_argument('-a', '--attack', type=str, default=None,
                        help='Attack filter (gradient_scaling, same_value, back_gradient)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from cached results directory')
    parser.add_argument('--dev', action='store_true',
                        help='Dev mode (fewer rounds, smaller subset)')
    parser.add_argument('--fraction', type=float, default=None,
                        help='Dataset fraction override')
    args = parser.parse_args()

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join('Output', OUTPUT_PREFIX, timestamp)
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Config
    if args.dev:
        rounds_map = DEV_ROUNDS.copy()
        fraction = DEV_FRACTION
        print("[DEV MODE] Reduced rounds and data")
    else:
        rounds_map = ROUNDS.copy()
        fraction = DATASET_FRACTION

    if args.fraction is not None:
        fraction = args.fraction

    # Dataset filter
    if args.dataset:
        ds_key = args.dataset.lower()
        datasets = [_DS.get(ds_key, args.dataset)]
    else:
        datasets = DATASETS

    # Attack filter
    if args.attack:
        attacks = [args.attack.lower()]
    else:
        attacks = ATTACKS

    # Experiment filter
    ALL_EXPS = {'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7'}
    if args.experiment:
        run_exps = {e.strip().upper() for e in args.experiment.split(',')}
    else:
        run_exps = ALL_EXPS

    # Resume
    if args.resume:
        load_cache_checkpoint(args.resume)

    cpr = max(5, int(NUM_CLIENTS * 0.10))
    framework = EvaluationFramework(out_dir=output_dir, run_id=timestamp)

    print(f"\n{'='*80}")
    print(f"  CD-CFL EXPERIMENTS ({OUTPUT_PREFIX})")
    print(f"{'='*80}")
    print(f"  Datasets    : {datasets}")
    print(f"  Attacks     : {attacks}")
    print(f"  Experiments : {sorted(run_exps)}")
    print(f"  Rounds      : { {ds: rounds_map[ds] for ds in datasets} }")
    print(f"  Fraction    : {fraction}")
    print(f"  Clients     : {NUM_CLIENTS} (active={cpr})")
    print(f"  Output      : {output_dir}")
    print(f"{'='*80}")

    total_start = time.time()

    # ---- Training experiments ----
    if 'E1' in run_exps:
        run_E1(framework, datasets, rounds_map, fraction, cpr, output_dir)
    if 'E2' in run_exps:
        run_E2(framework, datasets, attacks, rounds_map, fraction, cpr, output_dir)
    if 'E3' in run_exps:
        run_E3(framework, datasets, attacks, rounds_map, fraction, cpr, output_dir)
    if 'E5' in run_exps:
        run_E5(framework, datasets, attacks, rounds_map, fraction, cpr, output_dir)
    if 'E6' in run_exps:
        run_E6(framework, datasets, rounds_map, fraction, cpr, output_dir)

    # ---- Post-processing (no new runs) ----
    if 'E4' in run_exps:
        run_E4(datasets, attacks, rounds_map)
    if 'E7' in run_exps:
        run_E7(datasets, attacks, rounds_map)

    # ---- Generate plots, tables, and output files ----
    print(f"\n{'='*80}")
    print("  GENERATING PLOTS & TABLES")
    print(f"{'='*80}")

    try:
        if 'E1' in run_exps or 'E6' in run_exps:
            plot_convergence_curves(rounds_map, datasets, os.path.join(plots_dir, 'E1_E6_convergence'))

        if 'E2' in run_exps:
            plot_robustness_grid(rounds_map, datasets, attacks, os.path.join(plots_dir, 'E2_robustness_grid'))
            plot_robustness_bars(rounds_map, datasets, attacks, os.path.join(plots_dir, 'E2_robustness_bars'))
            generate_latex_table(rounds_map, datasets, attacks, os.path.join(plots_dir, 'E2_table'))

        if 'E3' in run_exps:
            plot_E3A_ablation(rounds_map, datasets, attacks, os.path.join(plots_dir, 'E3A_ablation'))
            plot_E3B_strategy(rounds_map, datasets, attacks, os.path.join(plots_dir, 'E3B_strategy'))
            generate_E3_csv(rounds_map, datasets, attacks, os.path.join(plots_dir, 'E3'))

        if 'E4' in run_exps or 'E2' in run_exps:
            plot_pow_analysis(rounds_map, datasets, attacks, os.path.join(plots_dir, 'E4_pow'))

        if 'E5' in run_exps:
            plot_stress_lines(rounds_map, datasets, attacks, os.path.join(plots_dir, 'E5_stress'))
            generate_E5_csv(rounds_map, datasets, attacks, os.path.join(plots_dir, 'E5_stress.csv'))

        if 'E7' in run_exps or 'E2' in run_exps:
            plot_overhead_bars(rounds_map, datasets, attacks, os.path.join(plots_dir, 'E7_overhead'))

        # Summary CSV
        generate_summary_csv(os.path.join(plots_dir, 'summary.csv'))

        # results.json (notebook-compatible format)
        save_results_json(output_dir)

    except Exception as e:
        print(f"  [WARNING] Plot/table generation failed: {e}")
        import traceback
        traceback.print_exc()

    # Final cache save
    save_cache_checkpoint(output_dir)

    # List generated files
    gen_files = sorted(os.listdir(plots_dir))
    png_count = sum(1 for f in gen_files if f.endswith('.png'))
    csv_count = sum(1 for f in gen_files if f.endswith('.csv'))
    tex_count = sum(1 for f in gen_files if f.endswith('.tex'))
    eps_count = sum(1 for f in gen_files if f.endswith('.eps'))
    print(f"\n  Generated: {png_count} PNG, {eps_count} EPS, {csv_count} CSV, {tex_count} LaTeX")
    for f in gen_files:
        print(f"    {f}")

    total_time = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"  DONE — {len(RESULTS_CACHE)} unique runs, {total_time/60:.1f} min total")
    print(f"  Output: {output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
