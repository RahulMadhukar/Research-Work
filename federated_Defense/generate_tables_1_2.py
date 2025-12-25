#!/usr/bin/env python3
"""
Auto-generate Table 1 and Table 2 from evaluation results.

This script is called automatically after evaluation.py completes.
It generates:
- Table 1: Detection Performance (DACC, FPR, FNR)
- Table 2: Model Performance (TACC, ASR)
- Associated plots for both tables
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import sys

class Tables1And2Generator:
    """Generate Table 1 and Table 2 from evaluation results."""

    def __init__(self, results_dir: Path):
        """
        Args:
            results_dir: Directory containing evaluation results JSON
        """
        self.results_dir = Path(results_dir)
        self.results = None
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def load_results(self, json_file: str = None):
        """Load results from JSON file."""
        if json_file is None:
            # Find the most recent evaluation_summary JSON
            json_files = list(self.results_dir.glob("evaluation_summary_*.json"))

            # If no evaluation_summary files found, try to find any JSON files
            if not json_files:
                print(f"[WARN] No evaluation_summary JSON found in {self.results_dir}")
                json_files = list(self.results_dir.glob("*.json"))

            if not json_files:
                # Try looking in parent directory
                parent_json = list(self.results_dir.parent.glob("*.json"))
                if parent_json:
                    print(f"[INFO] Found JSON files in parent directory: {self.results_dir.parent}")
                    for jf in parent_json:
                        print(f"  - {jf.name}")
                    json_file = max(parent_json, key=lambda p: p.stat().st_mtime)
                    filepath = json_file
                else:
                    raise FileNotFoundError(
                        f"No JSON files found in {self.results_dir}\n"
                        f"Please ensure your results JSON file exists.\n"
                        f"Looking for files matching: evaluation_summary_*.json or *.json"
                    )
            else:
                json_file = max(json_files, key=lambda p: p.stat().st_mtime)
                filepath = json_file
        else:
            # Handle provided json_file
            json_path = Path(json_file)
            if json_path.is_absolute():
                filepath = json_path
            elif (self.results_dir / json_file).exists():
                filepath = self.results_dir / json_file
            elif (self.results_dir.parent / json_file).exists():
                filepath = self.results_dir.parent / json_file
            elif json_path.exists():
                filepath = json_path
            else:
                raise FileNotFoundError(
                    f"JSON file not found: {json_file}\n"
                    f"Searched in:\n"
                    f"  - {json_path}\n"
                    f"  - {self.results_dir / json_file}\n"
                    f"  - {self.results_dir.parent / json_file}"
                )

        print(f"[INFO] Loading results from: {filepath}")

        with open(filepath, 'r') as f:
            self.results = json.load(f)
        return self.results

    def generate_table1_detection_performance(self) -> pd.DataFrame:
        """
        Generate Table 1: Detection performance (DACC, FPR, FNR, Precision, Recall).

        Table format:
        Attack | Defense | MNIST (DACC↑, FPR↓, FNR↓, Precision↑, Recall↑) | Fashion-MNIST (...) | EMNIST (...) | CIFAR-10 (...)
        """
        print("\n" + "="*100)
        print("GENERATING TABLE 1: Detection Performance (DACC↑, FPR↓, FNR↓, Precision↑, Recall↑)")
        print("="*100)

        results = self.results
        datasets = ['MNIST', 'Fashion-MNIST', 'EMNIST', 'CIFAR-10']
        attacks = ['slf', 'dlf', 'centralized', 'coordinated', 'random', 'model_dependent',
                   'local_model_replacement', 'local_model_noise', 'global_model_replacement', 'aggregation_modification']
        defenses = ['adaptivecommittee', 'cmfl']

        table_data = []

        for attack in attacks:
            for defense in defenses:
                row = {
                    'Attack': attack.upper().replace('_', ' '),
                    'Defense': defense.capitalize()
                }

                for dataset in datasets:
                    try:
                        attack_results = results.get(dataset, {}).get(attack, {})
                        # Try with defense_ prefix first, then without
                        defense_key = f'defense_{defense}'
                        defense_results = attack_results.get(defense_key, attack_results.get(defense, {}))

                        if defense_results and 'detection_metrics' in defense_results:
                            metrics = defense_results['detection_metrics']

                            # Compute DACC from TP, TN, FP, FN
                            tp = metrics.get('true_positives', 0)
                            tn = metrics.get('true_negatives', 0)
                            fp = metrics.get('false_positives', 0)
                            fn = metrics.get('false_negatives', 0)

                            total = tp + tn + fp + fn
                            dacc = ((tp + tn) / total * 100) if total > 0 else 0

                            # Note: coordinator already returns rates as percentages (0-100)
                            fpr = metrics.get('false_positive_rate', 0)
                            fnr = metrics.get('false_negative_rate', 0)
                            precision = metrics.get('precision', 0)
                            recall = metrics.get('recall', 0)

                            row[f'{dataset}_DACC'] = round(dacc, 2)
                            row[f'{dataset}_FPR'] = round(fpr, 2)
                            row[f'{dataset}_FNR'] = round(fnr, 2)
                            row[f'{dataset}_Precision'] = round(precision, 2)
                            row[f'{dataset}_Recall'] = round(recall, 2)
                        else:
                            row[f'{dataset}_DACC'] = None
                            row[f'{dataset}_FPR'] = None
                            row[f'{dataset}_FNR'] = None
                            row[f'{dataset}_Precision'] = None
                            row[f'{dataset}_Recall'] = None
                            if defense == 'adaptivecommittee':  # Only print once per attack
                                print(f"[WARN] No detection_metrics found for {dataset}/{attack}/{defense}")
                                print(f"[WARN] Re-run evaluation with updated coordinator.py to get detection metrics")
                    except Exception as e:
                        print(f"[WARN] Error processing {dataset}/{attack}/{defense}: {e}")
                        row[f'{dataset}_DACC'] = None
                        row[f'{dataset}_FPR'] = None
                        row[f'{dataset}_FNR'] = None
                        row[f'{dataset}_Precision'] = None
                        row[f'{dataset}_Recall'] = None

                table_data.append(row)

        # Create DataFrame
        df = pd.DataFrame(table_data)

        # Reorder columns
        column_order = ['Attack', 'Defense']
        for dataset in datasets:
            column_order.extend([f'{dataset}_DACC', f'{dataset}_FPR', f'{dataset}_FNR',
                               f'{dataset}_Precision', f'{dataset}_Recall'])
        df = df[column_order]

        # Save CSV
        csv_file = self.results_dir / "table1_detection_performance.csv"
        df.to_csv(csv_file, index=False)
        print(f"[INFO] Table 1 CSV saved to {csv_file}")

        # Save LaTeX
        latex_file = self.results_dir / "table1_detection_performance.tex"
        with open(latex_file, 'w') as f:
            f.write(df.to_latex(index=False, float_format="%.2f"))
        print(f"[INFO] Table 1 LaTeX saved to {latex_file}")

        # Print table
        print("\n" + "="*140)
        print("TABLE 1: Detection Performance (DACC↑, FPR↓, FNR↓, Precision↑, Recall↑)")
        print("="*140)
        print(df.to_string(index=False))
        print("="*140)

        return df

    def generate_table2_model_performance(self) -> pd.DataFrame:
        """
        Generate Table 2: Model performance (TACC, ASR).

        Table format:
        Attack | Defense | MNIST (TACC↑, ASR↓) | Fashion-MNIST (TACC↑, ASR↓) | EMNIST (TACC↑, ASR↓) | CIFAR-10 (TACC↑, ASR↓)
        """
        print("\n" + "="*100)
        print("GENERATING TABLE 2: Model Performance (TACC↑, ASR↓)")
        print("="*100)

        results = self.results
        datasets = ['MNIST', 'Fashion-MNIST', 'EMNIST', 'CIFAR-10']
        attacks = ['slf', 'dlf', 'centralized', 'coordinated', 'random', 'model_dependent',
                   'local_model_replacement', 'local_model_noise', 'global_model_replacement', 'aggregation_modification']
        defenses = ['adaptivecommittee', 'cmfl']

        table_data = []

        for attack in attacks:
            for defense in defenses:
                row = {
                    'Attack': attack.upper().replace('_', ' '),
                    'Defense': defense.capitalize()
                }

                for dataset in datasets:
                    try:
                        attack_results = results.get(dataset, {}).get(attack, {})
                        # Try with defense_ prefix first, then without
                        defense_key = f'defense_{defense}'
                        defense_results = attack_results.get(defense_key, attack_results.get(defense, {}))

                        if defense_results:
                            tacc = defense_results.get('final_accuracy', 0) * 100
                            # attack_success_rates is a list, calculate mean
                            asr_list = defense_results.get('attack_success_rates', [])
                            if asr_list:
                                asr = sum(asr_list) / len(asr_list) * 100
                            else:
                                asr = defense_results.get('attack_success_rate', 0) * 100

                            row[f'{dataset}_TACC'] = round(tacc, 2)
                            row[f'{dataset}_ASR'] = round(asr, 2)
                        else:
                            row[f'{dataset}_TACC'] = None
                            row[f'{dataset}_ASR'] = None
                    except Exception as e:
                        print(f"[WARN] Error processing {dataset}/{attack}/{defense}: {e}")
                        row[f'{dataset}_TACC'] = None
                        row[f'{dataset}_ASR'] = None

                table_data.append(row)

        # Create DataFrame
        df = pd.DataFrame(table_data)

        # Reorder columns
        column_order = ['Attack', 'Defense']
        for dataset in datasets:
            column_order.extend([f'{dataset}_TACC', f'{dataset}_ASR'])
        df = df[column_order]

        # Save CSV
        csv_file = self.results_dir / "table2_model_performance.csv"
        df.to_csv(csv_file, index=False)
        print(f"[INFO] Table 2 CSV saved to {csv_file}")

        # Save LaTeX
        latex_file = self.results_dir / "table2_model_performance.tex"
        with open(latex_file, 'w') as f:
            f.write(df.to_latex(index=False, float_format="%.2f"))
        print(f"[INFO] Table 2 LaTeX saved to {latex_file}")

        # Print table
        print("\n" + "="*100)
        print("TABLE 2: Model Performance (TACC↑, ASR↓)")
        print("="*100)
        print(df.to_string(index=False))
        print("="*100)

        return df

    def plot_table1_heatmaps(self, df: pd.DataFrame):
        """Generate heatmap visualizations for Table 1."""
        print("\n[PLOTS] Generating Table 1 heatmaps...")

        datasets = ['MNIST', 'Fashion-MNIST', 'EMNIST', 'CIFAR-10']
        metrics = ['DACC', 'FPR', 'FNR', 'Precision', 'Recall']

        for metric in metrics:
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            fig.suptitle(f'Detection Performance: {metric}', fontsize=16, fontweight='bold')
            axes = axes.flatten()  # Flatten to easily index

            for idx, dataset in enumerate(datasets):
                col_name = f'{dataset}_{metric}'

                # Prepare data for heatmap
                pivot_data = df.pivot_table(
                    values=col_name,
                    index='Defense',
                    columns='Attack',
                    aggfunc='first'
                )

                # Create heatmap
                ax = axes[idx]
                cmap = 'RdYlGn_r' if metric in ['FPR', 'FNR'] else 'RdYlGn'  # FPR and FNR lower is better
                sns.heatmap(
                    pivot_data,
                    annot=True,
                    fmt='.2f',
                    cmap=cmap,
                    cbar_kws={'label': metric},
                    ax=ax,
                    vmin=0,
                    vmax=100
                )
                ax.set_title(f'{dataset}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Attack Type', fontsize=12)
                ax.set_ylabel('Defense Mechanism', fontsize=12)

            plt.tight_layout()
            plot_file = self.plots_dir / f'table1_{metric.lower()}_heatmap.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved {plot_file}")

    def plot_table2_heatmaps(self, df: pd.DataFrame):
        """Generate heatmap visualizations for Table 2."""
        print("\n[PLOTS] Generating Table 2 heatmaps...")

        datasets = ['MNIST', 'Fashion-MNIST', 'EMNIST', 'CIFAR-10']
        metrics = ['TACC', 'ASR']

        for metric in metrics:
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            fig.suptitle(f'Model Performance: {metric}', fontsize=16, fontweight='bold')
            axes = axes.flatten()  # Flatten to easily index

            for idx, dataset in enumerate(datasets):
                col_name = f'{dataset}_{metric}'

                # Prepare data for heatmap
                pivot_data = df.pivot_table(
                    values=col_name,
                    index='Defense',
                    columns='Attack',
                    aggfunc='first'
                )

                # Create heatmap
                ax = axes[idx]
                cmap = 'RdYlGn' if metric == 'TACC' else 'RdYlGn_r'
                sns.heatmap(
                    pivot_data,
                    annot=True,
                    fmt='.2f',
                    cmap=cmap,
                    cbar_kws={'label': metric},
                    ax=ax,
                    vmin=0,
                    vmax=100
                )
                ax.set_title(f'{dataset}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Attack Type', fontsize=12)
                ax.set_ylabel('Defense Mechanism', fontsize=12)

            plt.tight_layout()
            plot_file = self.plots_dir / f'table2_{metric.lower()}_heatmap.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved {plot_file}")

    def plot_defense_comparison(self, df1: pd.DataFrame, df2: pd.DataFrame):
        """Generate defense mechanism comparison plots."""
        print("\n[PLOTS] Generating defense comparison plots...")

        datasets = ['MNIST', 'Fashion-MNIST', 'EMNIST', 'CIFAR-10']

        for dataset in datasets:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Defense Mechanism Comparison - {dataset}', fontsize=16, fontweight='bold')

            # Average DACC by defense
            ax = axes[0, 0]
            avg_dacc = df1.groupby('Defense')[f'{dataset}_DACC'].mean()
            avg_dacc.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title('Average Detection Accuracy (DACC)', fontsize=12)
            ax.set_ylabel('DACC (%)', fontsize=10)
            ax.set_xlabel('')
            ax.grid(axis='y', alpha=0.3)

            # Average FPR by defense
            ax = axes[0, 1]
            avg_fpr = df1.groupby('Defense')[f'{dataset}_FPR'].mean()
            avg_fpr.plot(kind='bar', ax=ax, color='salmon')
            ax.set_title('Average False Positive Rate (FPR)', fontsize=12)
            ax.set_ylabel('FPR (%)', fontsize=10)
            ax.set_xlabel('')
            ax.grid(axis='y', alpha=0.3)

            # Average TACC by defense
            ax = axes[1, 0]
            avg_tacc = df2.groupby('Defense')[f'{dataset}_TACC'].mean()
            avg_tacc.plot(kind='bar', ax=ax, color='lightgreen')
            ax.set_title('Average Test Accuracy (TACC)', fontsize=12)
            ax.set_ylabel('TACC (%)', fontsize=10)
            ax.set_xlabel('Defense Mechanism', fontsize=10)
            ax.grid(axis='y', alpha=0.3)

            # Average ASR by defense
            ax = axes[1, 1]
            avg_asr = df2.groupby('Defense')[f'{dataset}_ASR'].mean()
            avg_asr.plot(kind='bar', ax=ax, color='orange')
            ax.set_title('Average Attack Success Rate (ASR)', fontsize=12)
            ax.set_ylabel('ASR (%)', fontsize=10)
            ax.set_xlabel('Defense Mechanism', fontsize=10)
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            plot_file = self.plots_dir / f'defense_comparison_{dataset.lower().replace("-", "_")}.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved {plot_file}")

    def generate_all(self):
        """Generate all tables and plots."""
        print("\n" + "="*100)
        print("AUTO-GENERATING TABLE 1 AND TABLE 2 FROM EVALUATION RESULTS")
        print("="*100)

        # Generate tables
        df1 = self.generate_table1_detection_performance()
        df2 = self.generate_table2_model_performance()

        # Generate plots
        self.plot_table1_heatmaps(df1)
        self.plot_table2_heatmaps(df2)
        self.plot_defense_comparison(df1, df2)

        print("\n" + "="*100)
        print("✅ TABLE 1 AND TABLE 2 GENERATION COMPLETE")
        print("="*100)
        print(f"Results saved in: {self.results_dir}")
        print(f"Plots saved in: {self.plots_dir}")
        print("\nFiles generated:")
        print("  - table1_detection_performance.csv/.tex")
        print("  - table2_model_performance.csv/.tex")
        print("  - Various heatmap and comparison plots (PNG)")
        print("="*100)


def list_available_json_files(directory: Path):
    """List all available JSON files in the directory and subdirectories."""
    print(f"\n[INFO] Searching for JSON files in: {directory}")
    print("="*80)

    # Search in multiple locations
    search_paths = [
        directory,
        directory / "results",
        directory.parent,
        directory.parent / "results"
    ]

    found_files = []
    for search_path in search_paths:
        if search_path.exists():
            json_files = list(search_path.glob("*.json"))
            if json_files:
                print(f"\n  In {search_path}:")
                for jf in sorted(json_files):
                    print(f"    - {jf.name}")
                    found_files.append(jf)

    if not found_files:
        print("\n  No JSON files found in any of the searched locations.")
        print("\n  Searched in:")
        for sp in search_paths:
            print(f"    - {sp}")

    print("="*80)
    return found_files


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python generate_tables_1_2.py <results_directory> [json_file]")
        print("       python generate_tables_1_2.py --list <directory>")
        print("")
        print("Example:")
        print("  # List available JSON files")
        print("  python generate_tables_1_2.py --list plots/20251209_113842")
        print("")
        print("  # Generate tables from auto-detected JSON")
        print("  python generate_tables_1_2.py plots/20251209_113842/results")
        print("")
        print("  # Generate tables from specific JSON file")
        print("  python generate_tables_1_2.py plots/20251209_113842/results evaluation_summary_20251209_113842.json")
        sys.exit(1)

    # Handle --list option
    if sys.argv[1] == '--list':
        if len(sys.argv) < 3:
            print("[ERROR] Please provide a directory to search")
            print("Usage: python generate_tables_1_2.py --list <directory>")
            sys.exit(1)
        directory = Path(sys.argv[2])
        list_available_json_files(directory)
        sys.exit(0)

    results_dir = Path(sys.argv[1])
    json_file = sys.argv[2] if len(sys.argv) > 2 else None

    # Create generator
    generator = Tables1And2Generator(results_dir)

    # Load results
    generator.load_results(json_file)

    # Generate all tables and plots
    generator.generate_all()


if __name__ == '__main__':
    main()
