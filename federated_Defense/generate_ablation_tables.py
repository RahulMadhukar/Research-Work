#!/usr/bin/env python3
"""
Generate tables and plots from ablation study results.

This script reads aggregated ablation study results and generates:
- CSV and LaTeX tables showing metrics across parameter values
- Heatmaps and line plots for visualization

Usage:
    python generate_ablation_tables.py <results_json_file>

Example:
    python generate_ablation_tables.py Output/quick_mnist_ablation/20260107_103125/client_participation/aggregated_results/client_participation_results.json
"""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_ablation_results(json_path):
    """Load ablation study results from JSON file."""
    print(f"[INFO] Loading ablation results from: {json_path}")

    with open(json_path, 'r') as f:
        results = json.load(f)

    # Debug: Print top-level keys
    print(f"[DEBUG] Top-level keys in JSON: {list(results.keys())}")

    # If only one key and it contains the actual data, unwrap it
    if len(results.keys()) == 1:
        key = list(results.keys())[0]
        if isinstance(results[key], dict):
            print(f"[DEBUG] Unwrapping single key: '{key}'")
            results = results[key]
            print(f"[DEBUG] New top-level keys: {list(results.keys())}")

    return results


def get_parameter_column_name(ablation_type):
    """Get the appropriate column name for the parameter based on ablation type."""
    ablation_type_lower = ablation_type.lower()

    if 'data_poison' in ablation_type_lower or 'poison_percent' in ablation_type_lower:
        return 'Data Poison %'
    elif 'aggregation' in ablation_type_lower or 'agg_scheme' in ablation_type_lower:
        return 'Agg Scheme'
    elif 'client_participation' in ablation_type_lower:
        return 'Client Participation %'
    elif 'malicious' in ablation_type_lower:
        return 'Malicious Client %'
    elif 'non_iid' in ablation_type_lower or 'noniid' in ablation_type_lower:
        return 'C value'
    else:
        return 'Parameter Value'


def detect_ablation_type(json_path, results):
    """Detect the ablation type from file path or results."""
    path_str = str(json_path).lower()

    if 'data_poison' in path_str or 'poison_percent' in path_str:
        return 'data_poison_percentage'
    elif 'aggregation' in path_str or 'agg_scheme' in path_str:
        return 'aggregation_schemes'
    elif 'client_participation' in path_str:
        return 'client_participation'
    elif 'malicious' in path_str:
        return 'malicious_client_ratio'
    elif 'non_iid' in path_str or 'noniid' in path_str:
        return 'non_iid'

    elif 'efficiency' in path_str:
        return 'efficiency'
    elif 'committee' in path_str and 'malicious' in path_str:
        return 'committee_malicious_tracking'
    elif 'committee' in path_str:
        return 'committee_size'
    elif 'agg_participation' in path_str:
        return 'agg_participation'

    # Try to detect from results structure
    if 'ablation_type' in results:
        return results['ablation_type']

    # Detect efficiency from structure (dataset → rate keys like "1Mbps")
    for key, val in results.items():
        if isinstance(val, dict) and any(k.endswith('Mbps') for k in val):
            return 'efficiency'

    return 'unknown'


def _extract_row(param_value, dataset, attack_type, scheme, scenario_key, data):
    """Build a single row dict from a scenario data dict."""
    detection_metrics = data.get('detection_metrics', {})
    return {
        'parameter_value': param_value,
        'dataset': dataset,
        'attack': attack_type,
        'scheme': scheme,
        'scenario': scenario_key,
        'test_accuracy': data.get('final_test_accuracy', data.get('final_accuracy', None)),
        'attack_success_rate': data.get('final_attack_success_rate', data.get('attack_success_rate', None)),
        'dacc': detection_metrics.get('dacc', detection_metrics.get('accuracy', detection_metrics.get('detection_rate', None))),
        'precision': detection_metrics.get('precision', None),
        'recall': detection_metrics.get('recall', None),
        'fpr': detection_metrics.get('fpr', detection_metrics.get('false_positive_rate', None)),
        'fnr': detection_metrics.get('fnr', detection_metrics.get('false_negative_rate', None)),
        'f1_score': detection_metrics.get('f1_score', None),
    }


def extract_metrics_from_ablation(results):
    """
    Extract metrics from ablation study results.

    Handles multiple JSON formats:
    - New format:  param -> dataset -> attack_type -> scheme -> baseline/attack
    - Old format:  param -> dataset -> attack_type -> baseline/attack/defense
    - Legacy:      param -> dataset -> baseline/attack/defense

    Returns a DataFrame with columns:
    - parameter_value, dataset, attack, scheme, scenario
    - test_accuracy, attack_success_rate
    - dacc, precision, recall, fpr, fnr, f1_score
    """
    rows = []

    print(f"[DEBUG] Extracting metrics from {len(results)} top-level keys")

    for param_value, param_data in results.items():
        if param_value in ['ablation_type', 'timestamp', 'configuration']:
            continue
        if not isinstance(param_data, dict):
            continue

        print(f"[DEBUG] Processing parameter value: {param_value}")

        for dataset, dataset_data in param_data.items():
            if not isinstance(dataset_data, dict):
                continue

            print(f"[DEBUG]   Dataset: {dataset}, keys: {list(dataset_data.keys())}")

            for attack_type, attack_data in dataset_data.items():
                if not isinstance(attack_data, dict):
                    continue

                # Detect depth: check if values are scenario dicts (have final_accuracy)
                # or aggregation-scheme dicts (have keys like FEDAVG, KRUM, etc.)
                sample_val = next(iter(attack_data.values()), None)

                if isinstance(sample_val, dict) and \
                   'final_accuracy' not in sample_val and \
                   'final_test_accuracy' not in sample_val and \
                   any(k in sample_val for k in ('baseline', 'attack', 'attack_no_defense')):
                    # New format: attack_data[SCHEME] = {baseline: {...}, attack: {...}}
                    for scheme, scheme_data in attack_data.items():
                        if not isinstance(scheme_data, dict):
                            continue
                        for scenario_key in ('baseline', 'attack', 'attack_no_defense'):
                            if scenario_key in scheme_data and isinstance(scheme_data[scenario_key], dict):
                                sc = 'attack' if scenario_key == 'attack_no_defense' else scenario_key
                                rows.append(_extract_row(
                                    param_value, dataset, attack_type,
                                    scheme.upper(), sc, scheme_data[scenario_key]))
                else:
                    # Old format: attack_data = {baseline: {...}, attack: {...}, ...}
                    scheme = 'N/A'
                    for scenario_key in ('baseline', 'attack', 'attack_no_defense'):
                        if scenario_key in attack_data and isinstance(attack_data[scenario_key], dict):
                            sc = 'attack' if scenario_key == 'attack_no_defense' else scenario_key
                            rows.append(_extract_row(
                                param_value, dataset, attack_type,
                                scheme, sc, attack_data[scenario_key]))
                    for def_key in ('cmfl',):
                        if def_key in attack_data and isinstance(attack_data[def_key], dict):
                            rows.append(_extract_row(
                                param_value, dataset, attack_type,
                                scheme, def_key, attack_data[def_key]))

    df = pd.DataFrame(rows)
    print(f"[DEBUG] Extracted {len(df)} rows of data")

    if len(df) > 0:
        print(f"[DEBUG] DataFrame columns: {list(df.columns)}")
        print(f"[DEBUG] Unique datasets: {df['dataset'].unique()}")
        print(f"[DEBUG] Unique parameter values: {sorted(df['parameter_value'].unique())}")
        print(f"[DEBUG] Unique scenarios: {df['scenario'].unique()}")

    return df


def generate_summary_table(df, output_dir, ablation_type=""):
    """Generate summary table showing all metrics across parameter values."""

    print("\n" + "="*100)
    print("GENERATING ABLATION STUDY SUMMARY TABLE")
    print("="*100)

    param_values = sorted(df['parameter_value'].unique())

    # Detect if we have per-scheme data (new format) vs old format
    has_schemes = 'scheme' in df.columns and not (df['scheme'] == 'N/A').all()

    summary_rows = []

    if has_schemes:
        # New format: rows grouped by (dataset, attack, scheme, scenario)
        attacks = sorted(df['attack'].unique())
        schemes = sorted(df['scheme'].unique())

        for dataset in df['dataset'].unique():
            for attack in attacks:
                for scheme in schemes:
                    for scenario in ['baseline', 'attack']:
                        sub = df[
                            (df['dataset'] == dataset) &
                            (df['attack'] == attack) &
                            (df['scheme'] == scheme) &
                            (df['scenario'] == scenario)
                        ]
                        if len(sub) == 0:
                            continue

                        row = {
                            'Dataset': dataset,
                            'Attack': attack,
                            'Scheme': scheme,
                            'Scenario': scenario.capitalize(),
                        }
                        for pv in param_values:
                            pv_df = sub[sub['parameter_value'] == pv]
                            if len(pv_df) > 0:
                                acc = pv_df.iloc[0]['test_accuracy']
                                asr = pv_df.iloc[0]['attack_success_rate']
                                row[f'{pv}_Acc'] = f"{acc:.3f}" if acc is not None else "N/A"
                                row[f'{pv}_ASR'] = f"{asr:.4f}" if (scenario == 'attack' and asr is not None) else "N/A"
                            else:
                                row[f'{pv}_Acc'] = "N/A"
                                row[f'{pv}_ASR'] = "N/A"
                        summary_rows.append(row)

        id_cols = ['Dataset', 'Attack', 'Scheme', 'Scenario']
    else:
        # Old format: rows grouped by (dataset, scenario)
        _SCENARIO_ORDER = ['baseline', 'attack', 'cmfl']
        _SCENARIO_LABELS = {
            'baseline': 'Baseline', 'attack': 'Attack (no defense)',
            'cmfl': 'CMFL',
        }
        for dataset in df['dataset'].unique():
            for scenario in _SCENARIO_ORDER:
                sub = df[(df['dataset'] == dataset) & (df['scenario'] == scenario)]
                if len(sub) == 0:
                    continue
                row = {
                    'Dataset': dataset,
                    'Scenario': _SCENARIO_LABELS.get(scenario, scenario.capitalize()),
                }
                for pv in param_values:
                    pv_df = sub[sub['parameter_value'] == pv]
                    if len(pv_df) > 0:
                        acc = pv_df.iloc[0]['test_accuracy']
                        asr = pv_df.iloc[0]['attack_success_rate']
                        row[f'{pv}_Acc'] = f"{acc:.3f}" if acc is not None else "N/A"
                        row[f'{pv}_ASR'] = f"{asr:.4f}" if (scenario != 'baseline' and asr is not None) else "N/A"
                    else:
                        row[f'{pv}_Acc'] = "N/A"
                        row[f'{pv}_ASR'] = "N/A"
                summary_rows.append(row)

        id_cols = ['Dataset', 'Scenario']

    summary_df = pd.DataFrame(summary_rows)

    ordered_columns = list(id_cols)
    for pv in param_values:
        ordered_columns.append(f'{pv}_Acc')
        ordered_columns.append(f'{pv}_ASR')
    summary_df = summary_df[ordered_columns]

    # Save CSV
    csv_path = output_dir / "ablation_summary_table.csv"
    with open(csv_path, 'w') as f:
        header1 = list(id_cols)
        for pv in param_values:
            header1.extend([str(pv), ''])
        f.write(','.join(header1) + '\n')
        header2 = [''] * len(id_cols)
        for _ in param_values:
            header2.extend(['Acc', 'ASR'])
        f.write(','.join(header2) + '\n')
        for _, row in summary_df.iterrows():
            row_data = [str(row[c]) for c in id_cols]
            for pv in param_values:
                row_data.append(str(row[f'{pv}_Acc']))
                row_data.append(str(row[f'{pv}_ASR']))
            f.write(','.join(row_data) + '\n')
    print(f"[INFO] Summary table saved to: {csv_path}")

    # Save LaTeX
    latex_path = output_dir / "ablation_summary_table.tex"
    with open(latex_path, 'w') as f:
        col_spec = 'l' * len(id_cols) + 'r' * (len(param_values) * 2)
        f.write('\\begin{tabular}{' + col_spec + '}\n\\hline\n')
        h1 = list(id_cols)
        for pv in param_values:
            h1.append(f'\\multicolumn{{2}}{{c}}{{{pv}}}')
        f.write(' & '.join(h1) + ' \\\\\n')
        h2 = [''] * len(id_cols)
        for _ in param_values:
            h2.extend(['Acc', 'ASR'])
        f.write(' & '.join(h2) + ' \\\\\n\\hline\n')
        for _, row in summary_df.iterrows():
            parts = [str(row[c]) for c in id_cols]
            for pv in param_values:
                parts.append(str(row[f'{pv}_Acc']))
                parts.append(str(row[f'{pv}_ASR']))
            f.write(' & '.join(parts) + ' \\\\\n')
        f.write('\\hline\n\\end{tabular}\n')
    print(f"[INFO] LaTeX table saved to: {latex_path}")

    # Print to console
    print("\n" + "="*120)
    print("ABLATION STUDY SUMMARY")
    print("="*120)
    id_widths = {'Dataset': 12, 'Attack': 18, 'Scheme': 14, 'Scenario': 10}
    header_line = ""
    for c in id_cols:
        w = id_widths.get(c, 15)
        header_line += f"{c:<{w}} "
    for pv in param_values:
        header_line += f" {str(pv):^20}"
    print(header_line)
    sub_line = ""
    for c in id_cols:
        w = id_widths.get(c, 15)
        sub_line += " " * (w + 1)
    for _ in param_values:
        sub_line += f" {'Acc':>9} {'ASR':>9}"
    print(sub_line)
    print("-" * 120)
    for _, row in summary_df.iterrows():
        line = ""
        for c in id_cols:
            w = id_widths.get(c, 15)
            line += f"{str(row[c]):<{w}} "
        for pv in param_values:
            line += f" {row[f'{pv}_Acc']:>9} {row[f'{pv}_ASR']:>9}"
        print(line)
    print("="*120 + "\n")

    return summary_df


def generate_detection_metrics_table(df, output_dir, ablation_type=""):
    """Generate table showing detection metrics for defense scenarios."""

    print("\n" + "="*100)
    print("GENERATING DETECTION METRICS TABLE (Defense Only)")
    print("="*100)

    has_schemes = 'scheme' in df.columns and not (df['scheme'] == 'N/A').all()

    # Select defense rows: new format uses scheme column, old format uses scenario
    if has_schemes:
        defense_keys = ['CMFL']
        defense_df = df[
            (df['scheme'].str.upper().isin(defense_keys)) &
            (df['scenario'] == 'attack')
        ].copy()
        label_col = 'scheme'
    else:
        defense_keys = ['cmfl']
        defense_df = df[df['scenario'].isin(defense_keys)].copy()
        label_col = 'scenario'

    _DEFENSE_LABELS = {
        'cmfl': 'CMFL', 'CMFL': 'CMFL',
    }

    if len(defense_df) == 0:
        print("[WARN] No defense scenarios found. Skipping detection metrics table.")
        return None

    param_col_name = get_parameter_column_name(ablation_type)
    detection_rows = []

    for dataset in defense_df['dataset'].unique():
        for dk in defense_keys:
            if has_schemes:
                sub = defense_df[
                    (defense_df['dataset'] == dataset) &
                    (defense_df['scheme'].str.upper() == dk)
                ]
            else:
                sub = defense_df[
                    (defense_df['dataset'] == dataset) &
                    (defense_df['scenario'] == dk)
                ]
            if len(sub) == 0:
                continue

            param_values = sorted(sub['parameter_value'].unique())
            for idx, pv in enumerate(param_values):
                pv_df = sub[sub['parameter_value'] == pv]
                if len(pv_df) > 0:
                    m = pv_df.iloc[0]
                    detection_rows.append({
                        'Dataset': dataset if idx == 0 else '',
                        'Defense': _DEFENSE_LABELS.get(dk, dk),
                        param_col_name: pv,
                        'DACC': f"{m['dacc']:.2f}" if pd.notna(m.get('dacc')) else "N/A",
                        'Precision': f"{m['precision']:.2f}" if pd.notna(m.get('precision')) else "N/A",
                        'Recall': f"{m['recall']:.2f}" if pd.notna(m.get('recall')) else "N/A",
                        'FPR': f"{m['fpr']:.2f}" if pd.notna(m.get('fpr')) else "N/A",
                        'FNR': f"{m['fnr']:.2f}" if pd.notna(m.get('fnr')) else "N/A",
                        'F1': f"{m['f1_score']:.2f}" if pd.notna(m.get('f1_score')) else "N/A",
                    })

    detection_df_table = pd.DataFrame(detection_rows)

    csv_path = output_dir / "detection_metrics_table.csv"
    detection_df_table.to_csv(csv_path, index=False)
    print(f"[INFO] Detection metrics table saved to: {csv_path}")

    latex_path = output_dir / "detection_metrics_table.tex"
    with open(latex_path, 'w') as f:
        f.write('\\begin{tabular}{lllrrrrrr}\n\\hline\n')
        f.write(f'Dataset & Defense & {param_col_name} & DACC & Precision & Recall & FPR & FNR & F1 \\\\\n\\hline\n')
        for _, row in detection_df_table.iterrows():
            ds = str(row['Dataset']) if row['Dataset'] else ''
            pv = str(row[param_col_name]).replace('_', '\\_')
            f.write(f"{ds} & {row['Defense']} & {pv} & {row['DACC']} & {row['Precision']} & {row['Recall']} & {row['FPR']} & {row['FNR']} & {row['F1']} \\\\\n")
        f.write('\\hline\n\\end{tabular}\n')
    print(f"[INFO] LaTeX table saved to: {latex_path}")

    print("\n" + "="*115)
    print("DETECTION METRICS (Defense Scenarios)")
    print("="*115)
    print(f"\n{'Dataset':<15} {'Defense':<22} {param_col_name:<20} {'DACC':>8} {'Precision':>10} {'Recall':>8} {'FPR':>8} {'FNR':>8} {'F1':>8}")
    print("-" * 115)
    for _, row in detection_df_table.iterrows():
        ds = row['Dataset'] if row['Dataset'] else ''
        print(f"{ds:<15} {row['Defense']:<22} {str(row[param_col_name]):<20} {row['DACC']:>8} {row['Precision']:>10} {row['Recall']:>8} {row['FPR']:>8} {row['FNR']:>8} {row['F1']:>8}")
    print("="*115 + "\n")

    return detection_df_table


def generate_plots(df, output_dir):
    """Generate visualization plots."""

    print("\n[PLOTS] Generating ablation study plots...")

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    has_schemes = 'scheme' in df.columns and not (df['scheme'] == 'N/A').all()
    datasets = df['dataset'].unique()
    param_values = sorted(df['parameter_value'].unique())
    x_positions = list(range(len(param_values)))
    x_labels = [str(pv) for pv in param_values]

    if has_schemes:
        schemes = sorted(df['scheme'].unique())

        # 1. Accuracy vs Parameter — one subplot per scheme, lines per (attack, scenario)
        n_schemes = len(schemes)
        ncols = min(3, n_schemes)
        nrows = (n_schemes + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
        fig.suptitle('Test Accuracy vs Parameter Value (by Scheme)', fontsize=16, fontweight='bold')

        for si, scheme in enumerate(schemes):
            ax = axes[si // ncols, si % ncols]
            scheme_df = df[df['scheme'] == scheme]
            for attack in sorted(scheme_df['attack'].unique()):
                for scenario in ['baseline', 'attack']:
                    sub = scheme_df[(scheme_df['attack'] == attack) & (scheme_df['scenario'] == scenario)]
                    if len(sub) == 0:
                        continue
                    accs = []
                    for pv in param_values:
                        pv_sub = sub[sub['parameter_value'] == pv]
                        accs.append(pv_sub.iloc[0]['test_accuracy'] if len(pv_sub) > 0 else None)
                    valid = [(x, a) for x, a in zip(x_positions, accs) if a is not None]
                    if valid:
                        xs, ys = zip(*valid)
                        ax.plot(xs, ys, marker='o', label=f"{attack} ({scenario})", linewidth=1.5)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=45)
            ax.set_title(scheme, fontsize=12, fontweight='bold')
            ax.set_ylabel('Test Accuracy')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for si in range(n_schemes, nrows * ncols):
            axes[si // ncols, si % ncols].set_visible(False)

        plt.tight_layout()
        plot_path = plots_dir / "accuracy_vs_parameter.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved {plot_path}")

        # 2. ASR vs Parameter — similar layout
        attack_df = df[df['scenario'] == 'attack']
        if len(attack_df) > 0:
            fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
            fig.suptitle('Attack Success Rate vs Parameter Value (by Scheme)', fontsize=16, fontweight='bold')

            for si, scheme in enumerate(schemes):
                ax = axes[si // ncols, si % ncols]
                scheme_df = attack_df[attack_df['scheme'] == scheme]
                for attack in sorted(scheme_df['attack'].unique()):
                    sub = scheme_df[scheme_df['attack'] == attack]
                    sub = sub[sub['attack_success_rate'].notna()]
                    if len(sub) == 0:
                        continue
                    asrs = []
                    for pv in param_values:
                        pv_sub = sub[sub['parameter_value'] == pv]
                        asrs.append(pv_sub.iloc[0]['attack_success_rate'] if len(pv_sub) > 0 else None)
                    valid = [(x, a) for x, a in zip(x_positions, asrs) if a is not None]
                    if valid:
                        xs, ys = zip(*valid)
                        ax.plot(xs, ys, marker='o', label=attack, linewidth=1.5)
                ax.set_xticks(x_positions)
                ax.set_xticklabels(x_labels, rotation=45)
                ax.set_title(scheme, fontsize=12, fontweight='bold')
                ax.set_ylabel('Attack Success Rate')
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)

            for si in range(n_schemes, nrows * ncols):
                axes[si // ncols, si % ncols].set_visible(False)

            plt.tight_layout()
            plot_path = plots_dir / "asr_vs_parameter.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved {plot_path}")
    else:
        # Old format plotting (scenario-based)
        _PLOT_LABELS = {
            'baseline': 'Baseline', 'attack': 'Attack (no defense)',
            'cmfl': 'CMFL',
        }

        n_ds = min(4, len(datasets))
        nrows = max(1, (n_ds + 1) // 2)
        fig, axes = plt.subplots(nrows, 2, figsize=(16, 6 * nrows), squeeze=False)
        fig.suptitle('Test Accuracy vs Parameter Value', fontsize=16, fontweight='bold')
        for idx, dataset in enumerate(datasets[:4]):
            ax = axes[idx // 2, idx % 2]
            ds_df = df[df['dataset'] == dataset]
            for scenario in ['baseline', 'attack', 'cmfl']:
                sub = ds_df[ds_df['scenario'] == scenario]
                if len(sub) > 0:
                    sub = sub.sort_values('parameter_value')
                    ax.plot(range(len(sub)), sub['test_accuracy'],
                            marker='o', label=_PLOT_LABELS.get(scenario, scenario), linewidth=2)
                    ax.set_xticks(range(len(sub)))
                    ax.set_xticklabels([str(v) for v in sub['parameter_value']], rotation=45)
            ax.set_ylabel('Test Accuracy')
            ax.set_title(dataset, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "accuracy_vs_parameter.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved {plots_dir / 'accuracy_vs_parameter.png'}")

    print("[PLOTS] All plots generated successfully!\n")


def extract_metrics_from_efficiency(results):
    """Extract metrics from efficiency experiment results.

    Efficiency JSON structure:
        dataset -> {metadata, "1Mbps"/"10Mbps"/"100Mbps" -> framework -> metrics}

    Returns a DataFrame with columns:
        dataset, rate, framework, total_time, comm_overhead_mb, comp_time, accuracy
    """
    rows = []
    _METADATA_KEYS = {
        'model_size_bytes', 'rounds', 'total_clients', 'committee_size',
        'training_clients', 'fl_total_comp_time', 'cmfl_total_comp_time',
        'fl_accuracy', 'cmfl_accuracy',
    }
    _FRAMEWORK_LABELS = {
        'typical_fl': 'Typical FL',
        'braintorrent': 'BrainTorrent',
        'gossipfl': 'GossipFL',
        'cmfl': 'CMFL',
    }

    for dataset, ds_data in results.items():
        if not isinstance(ds_data, dict):
            continue

        for key, val in ds_data.items():
            if key in _METADATA_KEYS or not isinstance(val, dict):
                continue
            # key is a rate like "1Mbps", "10Mbps", "100Mbps"
            rate = key
            for fw_key, fw_data in val.items():
                if not isinstance(fw_data, dict):
                    continue
                rows.append({
                    'dataset': dataset,
                    'rate': rate,
                    'framework': _FRAMEWORK_LABELS.get(fw_key, fw_key),
                    'total_time': fw_data.get('total_time', None),
                    'comm_overhead_mb': fw_data.get('comm_overhead_bytes', 0) / (1024 ** 2),
                    'comp_time': fw_data.get('comp_time', None),
                    'accuracy': fw_data.get('accuracy', None),
                })

    df = pd.DataFrame(rows)
    print(f"[DEBUG] Extracted {len(df)} efficiency rows")
    if len(df) > 0:
        print(f"[DEBUG] Datasets: {df['dataset'].unique().tolist()}")
        print(f"[DEBUG] Rates: {df['rate'].unique().tolist()}")
        print(f"[DEBUG] Frameworks: {df['framework'].unique().tolist()}")
    return df


def generate_efficiency_table(df, output_dir):
    """Generate efficiency summary table (CSV + LaTeX + console)."""
    print("\n" + "=" * 100)
    print("GENERATING EFFICIENCY EXPERIMENT TABLE")
    print("=" * 100)

    rates = sorted(df['rate'].unique(), key=lambda r: float(r.replace('Mbps', '')))
    frameworks = ['Typical FL', 'BrainTorrent', 'GossipFL', 'CMFL']
    frameworks = [f for f in frameworks if f in df['framework'].values]

    summary_rows = []
    for dataset in df['dataset'].unique():
        for fw in frameworks:
            row = {'Dataset': dataset, 'Framework': fw}
            for rate in rates:
                sub = df[(df['dataset'] == dataset) & (df['framework'] == fw) & (df['rate'] == rate)]
                if len(sub) > 0:
                    m = sub.iloc[0]
                    row[f'{rate}_Time'] = f"{m['total_time']:.1f}" if m['total_time'] is not None else "N/A"
                    row[f'{rate}_Comm'] = f"{m['comm_overhead_mb']:.1f}" if m['comm_overhead_mb'] is not None else "N/A"
                else:
                    row[f'{rate}_Time'] = "N/A"
                    row[f'{rate}_Comm'] = "N/A"
            # Add accuracy (same across rates)
            acc_sub = df[(df['dataset'] == dataset) & (df['framework'] == fw) & df['accuracy'].notna()]
            row['Accuracy'] = f"{acc_sub.iloc[0]['accuracy']:.4f}" if len(acc_sub) > 0 else "N/A"
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Save CSV
    csv_path = output_dir / "efficiency_summary_table.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"[INFO] Efficiency table saved to: {csv_path}")

    # Save LaTeX
    latex_path = output_dir / "efficiency_summary_table.tex"
    with open(latex_path, 'w') as f:
        ncols = 2 + len(rates) * 2 + 1  # Dataset, Framework, (Time,Comm) per rate, Accuracy
        f.write('\\begin{tabular}{ll' + 'rr' * len(rates) + 'r}\n\\hline\n')
        h1 = ['Dataset', 'Framework']
        for rate in rates:
            h1.append(f'\\multicolumn{{2}}{{c}}{{{rate}}}')
        h1.append('Accuracy')
        f.write(' & '.join(h1) + ' \\\\\n')
        h2 = ['', '']
        for _ in rates:
            h2.extend(['Time(s)', 'Comm(MB)'])
        h2.append('')
        f.write(' & '.join(h2) + ' \\\\\n\\hline\n')
        for _, row in summary_df.iterrows():
            parts = [str(row['Dataset']), str(row['Framework'])]
            for rate in rates:
                parts.append(str(row[f'{rate}_Time']))
                parts.append(str(row[f'{rate}_Comm']))
            parts.append(str(row['Accuracy']))
            f.write(' & '.join(parts) + ' \\\\\n')
        f.write('\\hline\n\\end{tabular}\n')
    print(f"[INFO] LaTeX table saved to: {latex_path}")

    # Console output
    print("\n" + "=" * 120)
    print("EFFICIENCY EXPERIMENT SUMMARY")
    print("=" * 120)
    header = f"{'Dataset':<15} {'Framework':<15}"
    for rate in rates:
        header += f" {'Time(s)':>10} {'Comm(MB)':>10}"
    header += f" {'Accuracy':>10}"
    print(header)
    print("-" * 120)
    for _, row in summary_df.iterrows():
        line = f"{row['Dataset']:<15} {row['Framework']:<15}"
        for rate in rates:
            line += f" {row[f'{rate}_Time']:>10} {row[f'{rate}_Comm']:>10}"
        line += f" {row['Accuracy']:>10}"
        print(line)
    print("=" * 120 + "\n")

    return summary_df


def generate_efficiency_plots(df, output_dir):
    """Generate efficiency comparison plots."""
    print("\n[PLOTS] Generating efficiency plots...")

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    rates = sorted(df['rate'].unique(), key=lambda r: float(r.replace('Mbps', '')))
    rate_values = [float(r.replace('Mbps', '')) for r in rates]
    frameworks = ['Typical FL', 'BrainTorrent', 'GossipFL', 'CMFL']
    frameworks = [f for f in frameworks if f in df['framework'].values]
    datasets = df['dataset'].unique()

    # 1. Total time vs transmission rate (one subplot per dataset)
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(7 * n_ds, 5), squeeze=False)
    fig.suptitle('Total Time vs Transmission Rate', fontsize=14, fontweight='bold')
    for di, dataset in enumerate(datasets):
        ax = axes[0, di]
        for fw in frameworks:
            sub = df[(df['dataset'] == dataset) & (df['framework'] == fw)]
            times = []
            for rate in rates:
                r_sub = sub[sub['rate'] == rate]
                times.append(r_sub.iloc[0]['total_time'] if len(r_sub) > 0 and r_sub.iloc[0]['total_time'] is not None else None)
            valid = [(x, t) for x, t in zip(rate_values, times) if t is not None]
            if valid:
                xs, ys = zip(*valid)
                ax.plot(xs, ys, marker='o', label=fw, linewidth=2)
        ax.set_xlabel('Transmission Rate (Mbps)')
        ax.set_ylabel('Total Time (s)')
        ax.set_title(dataset, fontweight='bold')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "efficiency_time_vs_rate.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {plots_dir / 'efficiency_time_vs_rate.png'}")

    # 2. Communication overhead bar chart
    fig, axes = plt.subplots(1, n_ds, figsize=(7 * n_ds, 5), squeeze=False)
    fig.suptitle('Communication Overhead by Framework', fontsize=14, fontweight='bold')
    for di, dataset in enumerate(datasets):
        ax = axes[0, di]
        x = np.arange(len(rates))
        width = 0.8 / len(frameworks)
        for fi, fw in enumerate(frameworks):
            comms = []
            for rate in rates:
                sub = df[(df['dataset'] == dataset) & (df['framework'] == fw) & (df['rate'] == rate)]
                comms.append(sub.iloc[0]['comm_overhead_mb'] if len(sub) > 0 else 0)
            ax.bar(x + fi * width, comms, width, label=fw)
        ax.set_xlabel('Transmission Rate')
        ax.set_ylabel('Communication Overhead (MB)')
        ax.set_title(dataset, fontweight='bold')
        ax.set_xticks(x + width * (len(frameworks) - 1) / 2)
        ax.set_xticklabels(rates)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(plots_dir / "efficiency_comm_overhead.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {plots_dir / 'efficiency_comm_overhead.png'}")

    print("[PLOTS] Efficiency plots generated successfully!\n")


# =============================================================================
# Committee Malicious Tracking (Section 6.6) — N1/N2/N3 infiltration
# =============================================================================

def extract_metrics_from_committee_tracking(results):
    """Extract N1/N2/N3 infiltration metrics from committee tracking JSON.

    Expected JSON structure:
        epsilon% -> dataset -> attack -> scheme -> {committee_tracking: {avg_N1_count, ...}}

    Returns a DataFrame with columns:
        epsilon, dataset, attack, scheme, N1_avg, N1_frac, N2_avg, N2_frac, N3_avg, N3_frac,
        accuracy, rounds_with_N2, rounds_with_N3
    """
    rows = []
    for eps_key, eps_data in results.items():
        if not isinstance(eps_data, dict):
            continue
        for dataset, ds_data in eps_data.items():
            if not isinstance(ds_data, dict):
                continue
            for attack, atk_data in ds_data.items():
                if not isinstance(atk_data, dict):
                    continue
                for scheme, sch_data in atk_data.items():
                    if not isinstance(sch_data, dict):
                        continue
                    ct = sch_data.get('committee_tracking', {})
                    atk_info = sch_data.get('attack', {})
                    rows.append({
                        'epsilon': eps_key,
                        'dataset': dataset,
                        'attack': attack,
                        'scheme': scheme,
                        'accuracy': atk_info.get('final_accuracy', None),
                        'N1_avg': ct.get('avg_N1_count', 0.0),
                        'N1_frac': ct.get('avg_N1_fraction', 0.0),
                        'N2_avg': ct.get('avg_N2_count', ct.get('avg_malicious_count', 0.0)),
                        'N2_frac': ct.get('avg_N2_fraction', ct.get('avg_malicious_fraction', 0.0)),
                        'N3_avg': ct.get('avg_N3_count', 0.0),
                        'N3_frac': ct.get('avg_N3_fraction', 0.0),
                        'rounds_with_N2': ct.get('rounds_with_N2', ct.get('rounds_with_malicious', 0)),
                        'rounds_with_N3': ct.get('rounds_with_N3', 0),
                        'total_rounds': ct.get('total_rounds', 0),
                    })

    df = pd.DataFrame(rows)
    print(f"[DEBUG] Committee tracking: extracted {len(df)} rows")
    return df


def generate_committee_tracking_table(df, output_dir):
    """Generate CSV/LaTeX/console table showing avg N1/N2/N3 per epsilon."""
    output_dir = Path(output_dir)
    print("\n" + "=" * 100)
    print("COMMITTEE MALICIOUS TRACKING TABLE (Section 6.6)")
    print("=" * 100)

    if len(df) == 0:
        print("[WARN] No committee tracking data to tabulate.")
        return

    eps_values = sorted(df['epsilon'].unique(), key=lambda x: float(x.replace('%', '')))
    datasets = sorted(df['dataset'].unique())
    attacks = sorted(df['attack'].unique())
    schemes = sorted(df['scheme'].unique())

    summary_rows = []
    for dataset in datasets:
        for attack in attacks:
            for scheme in schemes:
                row = {'Dataset': dataset, 'Attack': attack, 'Scheme': scheme}
                for eps in eps_values:
                    sub = df[(df['dataset'] == dataset) & (df['attack'] == attack) &
                             (df['scheme'] == scheme) & (df['epsilon'] == eps)]
                    if len(sub) > 0:
                        r = sub.iloc[0]
                        row[f'{eps}_N1'] = f"{r['N1_avg']:.2f}"
                        row[f'{eps}_N2'] = f"{r['N2_avg']:.2f}"
                        row[f'{eps}_N3'] = f"{r['N3_avg']:.2f}"
                        row[f'{eps}_Acc'] = f"{r['accuracy']:.3f}" if r['accuracy'] is not None else "N/A"
                    else:
                        row[f'{eps}_N1'] = "N/A"
                        row[f'{eps}_N2'] = "N/A"
                        row[f'{eps}_N3'] = "N/A"
                        row[f'{eps}_Acc'] = "N/A"
                summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Build ordered columns
    id_cols = ['Dataset', 'Attack', 'Scheme']
    ordered = list(id_cols)
    for eps in eps_values:
        ordered.extend([f'{eps}_N1', f'{eps}_N2', f'{eps}_N3', f'{eps}_Acc'])
    summary_df = summary_df[[c for c in ordered if c in summary_df.columns]]

    # Save CSV
    csv_path = output_dir / "committee_tracking_table.csv"
    with open(csv_path, 'w') as f:
        h1 = list(id_cols)
        for eps in eps_values:
            h1.extend([f'ε={eps}', '', '', ''])
        f.write(','.join(h1) + '\n')
        h2 = [''] * len(id_cols)
        for _ in eps_values:
            h2.extend(['N1', 'N2', 'N3', 'Acc'])
        f.write(','.join(h2) + '\n')
        for _, row in summary_df.iterrows():
            vals = [str(row.get(c, '')) for c in ordered if c in summary_df.columns]
            f.write(','.join(vals) + '\n')
    print(f"[INFO] CSV saved to: {csv_path}")

    # Save LaTeX
    latex_path = output_dir / "committee_tracking_table.tex"
    with open(latex_path, 'w') as f:
        col_spec = 'l' * len(id_cols) + 'r' * (len(eps_values) * 4)
        f.write('\\begin{tabular}{' + col_spec + '}\n\\hline\n')
        h1 = list(id_cols)
        for eps in eps_values:
            h1.append(f'\\multicolumn{{4}}{{c}}{{$\\varepsilon$={eps}}}')
        f.write(' & '.join(h1) + ' \\\\\n')
        h2 = [''] * len(id_cols)
        for _ in eps_values:
            h2.extend(['N1', 'N2', 'N3', 'Acc'])
        f.write(' & '.join(h2) + ' \\\\\n\\hline\n')
        for _, row in summary_df.iterrows():
            parts = [str(row.get(c, '')) for c in ordered if c in summary_df.columns]
            f.write(' & '.join(parts) + ' \\\\\n')
        f.write('\\hline\n\\end{tabular}\n')
    print(f"[INFO] LaTeX saved to: {latex_path}")

    # Print to console
    print("\n" + "-" * 120)
    header = f"{'Dataset':<12} {'Attack':<18} {'Scheme':<10}"
    for eps in eps_values:
        header += f" | {eps:^24}"
    print(header)
    sub_header = f"{'':12} {'':18} {'':10}"
    for _ in eps_values:
        sub_header += f" | {'N1':>5} {'N2':>5} {'N3':>5}  {'Acc':>6}"
    print(sub_header)
    print("-" * 120)
    for _, row in summary_df.iterrows():
        line = f"{row.get('Dataset',''):<12} {row.get('Attack',''):<18} {row.get('Scheme',''):<10}"
        for eps in eps_values:
            n1 = row.get(f'{eps}_N1', 'N/A')
            n2 = row.get(f'{eps}_N2', 'N/A')
            n3 = row.get(f'{eps}_N3', 'N/A')
            acc = row.get(f'{eps}_Acc', 'N/A')
            line += f" | {n1:>5} {n2:>5} {n3:>5}  {acc:>6}"
        print(line)
    print("-" * 120)


def generate_committee_tracking_plots(df, output_dir):
    """Line plots of N1/N2/N3 counts vs epsilon."""
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if len(df) == 0:
        print("[WARN] No committee tracking data to plot.")
        return

    eps_values = sorted(df['epsilon'].unique(), key=lambda x: float(x.replace('%', '')))
    eps_numeric = [float(e.replace('%', '')) for e in eps_values]
    datasets = sorted(df['dataset'].unique())
    schemes = sorted(df['scheme'].unique())
    attacks = sorted(df['attack'].unique())

    # One figure per dataset, subplots per attack
    for dataset in datasets:
        ds_df = df[df['dataset'] == dataset]
        n_attacks = len(attacks)
        if n_attacks == 0:
            continue

        fig, axes = plt.subplots(1, max(n_attacks, 1), figsize=(7 * n_attacks, 5), squeeze=False)
        fig.suptitle(f'{dataset} — Malicious Infiltration vs ε', fontsize=14, fontweight='bold')

        for ai, attack in enumerate(attacks):
            ax = axes[0, ai]
            for scheme in schemes:
                sub = ds_df[(ds_df['attack'] == attack) & (ds_df['scheme'] == scheme)]
                if len(sub) == 0:
                    continue
                # Align by epsilon order
                n1s, n2s, n3s = [], [], []
                for eps in eps_values:
                    eps_sub = sub[sub['epsilon'] == eps]
                    if len(eps_sub) > 0:
                        n1s.append(eps_sub.iloc[0]['N1_avg'])
                        n2s.append(eps_sub.iloc[0]['N2_avg'])
                        n3s.append(eps_sub.iloc[0]['N3_avg'])
                    else:
                        n1s.append(0); n2s.append(0); n3s.append(0)

                ls = '-' if 'II' not in scheme else '--'
                ax.plot(eps_numeric, n1s, marker='o', linestyle=ls, label=f'{scheme} N1 (train)')
                ax.plot(eps_numeric, n2s, marker='s', linestyle=ls, label=f'{scheme} N2 (comm.)')
                ax.plot(eps_numeric, n3s, marker='^', linestyle=ls, label=f'{scheme} N3 (agg.)')

            ax.set_xlabel('Malicious Ratio ε (%)')
            ax.set_ylabel('Avg Malicious Count')
            ax.set_title(attack.upper())
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = plots_dir / f"committee_tracking_{dataset.lower()}.png"
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved {fname}")

    print("[PLOTS] Committee tracking plots generated successfully!\n")


def generate_tables_and_plots(json_path):
    """Programmatic entry point — called after each ablation saves its JSON.

    Args:
        json_path: Path (str or Path) to the results JSON file.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        print(f"[WARN] Results file not found, skipping table generation: {json_path}")
        return

    results = load_ablation_results(json_path)
    ablation_type = detect_ablation_type(json_path, results)
    print(f"[INFO] Detected ablation type: {ablation_type}")

    output_dir = json_path.parent

    if ablation_type == 'efficiency':
        df = extract_metrics_from_efficiency(results)
        if len(df) == 0:
            print("[WARN] No data extracted from efficiency results.")
            return
        generate_efficiency_table(df, output_dir)
        generate_efficiency_plots(df, output_dir)
    elif ablation_type == 'committee_malicious_tracking':
        df = extract_metrics_from_committee_tracking(results)
        if len(df) == 0:
            print("[WARN] No data extracted from committee tracking results.")
            return
        generate_committee_tracking_table(df, output_dir)
        generate_committee_tracking_plots(df, output_dir)
    else:
        df = extract_metrics_from_ablation(results)
        if len(df) == 0:
            print("[WARN] No data extracted from results file.")
            return
        generate_summary_table(df, output_dir, ablation_type)
        generate_detection_metrics_table(df, output_dir, ablation_type)
        generate_plots(df, output_dir)

    print(f"[INFO] Tables and plots saved to: {output_dir}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_ablation_tables.py <results_json_file>")
        print("\nExample:")
        print("  python generate_ablation_tables.py Output/quick_mnist_ablation/.../client_participation_results.json")
        sys.exit(1)

    json_path = Path(sys.argv[1])

    if not json_path.exists():
        print(f"[ERROR] File not found: {json_path}")
        sys.exit(1)

    # Load results
    results = load_ablation_results(json_path)

    # Detect ablation type
    ablation_type = detect_ablation_type(json_path, results)
    print(f"[INFO] Detected ablation type: {ablation_type}")

    # Create output directory
    output_dir = json_path.parent

    print(f"\n[INFO] Generating tables and plots...")
    print(f"[INFO] Output directory: {output_dir}\n")

    if ablation_type == 'efficiency':
        # Efficiency has a different JSON structure — use dedicated functions
        df = extract_metrics_from_efficiency(results)
        if len(df) == 0:
            print("[ERROR] No data extracted from efficiency results. Check JSON structure.")
            sys.exit(1)
        generate_efficiency_table(df, output_dir)
        generate_efficiency_plots(df, output_dir)
        print("\n" + "=" * 100)
        print("EFFICIENCY TABLE GENERATION COMPLETE")
        print("=" * 100)
        print(f"\nResults saved in: {output_dir}")
        print(f"Plots saved in: {output_dir / 'plots'}")
        print("\nFiles generated:")
        print("  - efficiency_summary_table.csv/.tex")
        print("  - efficiency_time_vs_rate.png")
        print("  - efficiency_comm_overhead.png")
        print("=" * 100 + "\n")
    elif ablation_type == 'committee_malicious_tracking':
        df = extract_metrics_from_committee_tracking(results)
        if len(df) == 0:
            print("[ERROR] No data extracted from committee tracking results. Check JSON structure.")
            sys.exit(1)
        generate_committee_tracking_table(df, output_dir)
        generate_committee_tracking_plots(df, output_dir)
        print("\n" + "=" * 100)
        print("COMMITTEE TRACKING TABLE GENERATION COMPLETE")
        print("=" * 100)
        print(f"\nResults saved in: {output_dir}")
        print(f"Plots saved in: {output_dir / 'plots'}")
        print("\nFiles generated:")
        print("  - committee_tracking_table.csv/.tex")
        print("  - committee_tracking_<dataset>.png")
        print("=" * 100 + "\n")
    else:
        # Standard ablation extraction
        df = extract_metrics_from_ablation(results)
        if len(df) == 0:
            print("[ERROR] No data extracted from results file. Check JSON structure.")
            sys.exit(1)

        # Generate tables
        summary_table = generate_summary_table(df, output_dir, ablation_type)
        detection_table = generate_detection_metrics_table(df, output_dir, ablation_type)

        # Generate plots
        generate_plots(df, output_dir)

        # Summary
        print("\n" + "="*100)
        print("ABLATION STUDY TABLE GENERATION COMPLETE")
        print("="*100)
        print(f"\nResults saved in: {output_dir}")
        print(f"Plots saved in: {output_dir / 'plots'}")
        print("\nFiles generated:")
        print("  - ablation_summary_table.csv/.tex")
        print("  - detection_metrics_table.csv/.tex")
        print("  - accuracy_vs_parameter.png")
        print("  - asr_vs_parameter.png")
        print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
