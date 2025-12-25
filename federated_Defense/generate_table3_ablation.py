#!/usr/bin/env python3
"""
Generate Table 3: Impact Analysis from Ablation Studies

This is COMPLETELY SEPARATE from Table 1 & Table 2.
Run this script AFTER ablation studies complete to analyze:
- Impact of poison percentage
- Impact of malicious client ratio
- Impact of Non-IID data distribution
- Impact of client participation
- Impact of aggregation schemes
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys

class Table3AblationAnalyzer:
    """Analyze ablation study results and generate Table 3 + plots."""

    def __init__(self, output_dir: Path = None):
        """
        Args:
            output_dir: Directory to save tables and plots
        """
        self.output_dir = Path(output_dir) if output_dir else Path("Output/table3_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.ablation_results = {}

    def auto_find_ablation_results(self):
        """Automatically find the most recent ablation result files."""
        base_path = Path("Output")

        mappings = {
            'poison_percentage': 'ablation_poison_percentage',
            'malicious_ratio': 'ablation_malicious_ratio',
            'non_iid': 'ablation_non_iid',
            'client_participation': 'ablation_client_participation',
            'aggregation': 'ablation_aggregation'
        }

        found_files = {}

        for key, dir_name in mappings.items():
            search_dir = base_path / dir_name / "aggregated_results"
            if search_dir.exists():
                # Find all JSON files matching the pattern
                json_files = list(search_dir.glob(f"{key}_results_*.json"))
                if json_files:
                    # Get the most recent file
                    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
                    found_files[key] = str(latest_file)
                    print(f"[AUTO-FOUND] {key}: {latest_file.name}")

        return found_files

    def load_ablation_results(
        self,
        poison_pct_file: str = None,
        malicious_ratio_file: str = None,
        non_iid_file: str = None,
        participation_file: str = None,
        aggregation_file: str = None,
        auto_discover: bool = False
    ):
        """Load all ablation study results."""
        print("\n" + "="*100)
        print("LOADING ABLATION STUDY RESULTS")
        print("="*100)

        # Auto-discovery if requested
        if auto_discover:
            print("[INFO] Auto-discovering ablation result files...")
            auto_files = self.auto_find_ablation_results()
            # Use auto-discovered files if not explicitly provided
            poison_pct_file = poison_pct_file or auto_files.get('poison_percentage')
            malicious_ratio_file = malicious_ratio_file or auto_files.get('malicious_ratio')
            non_iid_file = non_iid_file or auto_files.get('non_iid')
            participation_file = participation_file or auto_files.get('client_participation')
            aggregation_file = aggregation_file or auto_files.get('aggregation')

        files = {
            'poison_percentage': poison_pct_file,
            'malicious_ratio': malicious_ratio_file,
            'non_iid': non_iid_file,
            'client_participation': participation_file,
            'aggregation': aggregation_file
        }

        for study_name, filepath in files.items():
            if filepath and Path(filepath).exists():
                print(f"[INFO] Loading {study_name}: {filepath}")
                with open(filepath, 'r') as f:
                    self.ablation_results[study_name] = json.load(f)
            else:
                print(f"[WARN] Skipping {study_name}: file not provided or not found")

        return self.ablation_results

    def generate_table3a_malicious_ratio(self) -> pd.DataFrame:
        """
        Generate Table 3(a): Impact of Malicious Client Ratio.
        Rows: Attack + Defense (MultiIndex), Columns: Malicious ratio values.
        """
        if 'malicious_ratio' not in self.ablation_results:
            print("[WARN] No malicious ratio results available")
            return None

        print("\n" + "="*100)
        print("GENERATING TABLE 3(a): Impact of Malicious Client Ratio")
        print("="*100)

        results = self.ablation_results['malicious_ratio']
        table_data = []

        for ratio, datasets_data in results.items():
            for dataset, attacks_data in datasets_data.items():
                for attack, defenses_data in attacks_data.items():
                    for defense, metrics in defenses_data.items():
                        # Skip baseline
                        if defense in ['baseline', 'baseline_accuracy'] or not isinstance(metrics, dict):
                            continue

                        # Calculate DACC
                        dacc = None
                        if 'detection_metrics' in metrics and metrics['detection_metrics']:
                            dm = metrics['detection_metrics']
                            tp = dm.get('true_positives', 0)
                            tn = dm.get('true_negatives', 0)
                            fp = dm.get('false_positives', 0)
                            fn = dm.get('false_negatives', 0)
                            total = tp + tn + fp + fn
                            dacc = (tp + tn) / total * 100 if total > 0 else 0

                        table_data.append({
                            'Ratio': ratio,
                            'Dataset': dataset,
                            'Attack': attack.upper(),
                            'Defense': defense.capitalize(),
                            'DACC': dacc
                        })

        df_long = pd.DataFrame(table_data)

        # Create pivot table for each dataset
        for dataset in df_long['Dataset'].unique():
            df_dataset = df_long[df_long['Dataset'] == dataset]

            # Pivot: MultiIndex rows = (Attack, Defense), columns = Ratio
            df_pivot = df_dataset.pivot_table(
                index=['Attack', 'Defense'],
                columns='Ratio',
                values='DACC',
                aggfunc='mean'
            )

            # Round to 2 decimals
            df_pivot = df_pivot.round(2)

            # Save CSV
            csv_file = self.output_dir / f"table3a_malicious_ratio_{dataset.lower().replace('-', '_')}.csv"
            df_pivot.to_csv(csv_file)
            print(f"[INFO] Table 3(a) for {dataset} saved to {csv_file}")

            # Save LaTeX
            latex_file = self.output_dir / f"table3a_malicious_ratio_{dataset.lower().replace('-', '_')}.tex"
            with open(latex_file, 'w') as f:
                f.write(df_pivot.to_latex(multirow=True, float_format="%.2f"))
            print(f"[INFO] Table 3(a) LaTeX for {dataset} saved to {latex_file}")

        return df_long

    def generate_table3b_client_participation(self) -> pd.DataFrame:
        """
        Generate Table 3(b): Impact of Total Client Number (Client Participation).
        Rows: Attack + Defense (MultiIndex), Columns: Participation values.
        """
        if 'client_participation' not in self.ablation_results:
            print("[WARN] No client participation results available")
            return None

        print("\n" + "="*100)
        print("GENERATING TABLE 3(b): Impact of Total Client Number (Client Participation)")
        print("="*100)

        results = self.ablation_results['client_participation']
        table_data = []

        for participation, datasets_data in results.items():
            for dataset, attacks_data in datasets_data.items():
                for attack, defenses_data in attacks_data.items():
                    for defense, metrics in defenses_data.items():
                        # Skip baseline
                        if defense in ['baseline', 'baseline_accuracy'] or not isinstance(metrics, dict):
                            continue

                        # Calculate DACC
                        dacc = None
                        if 'detection_metrics' in metrics and metrics['detection_metrics']:
                            dm = metrics['detection_metrics']
                            tp = dm.get('true_positives', 0)
                            tn = dm.get('true_negatives', 0)
                            fp = dm.get('false_positives', 0)
                            fn = dm.get('false_negatives', 0)
                            total = tp + tn + fp + fn
                            dacc = (tp + tn) / total * 100 if total > 0 else 0

                        table_data.append({
                            'Participation': participation,
                            'Dataset': dataset,
                            'Attack': attack.upper(),
                            'Defense': defense.capitalize(),
                            'DACC': dacc
                        })

        df_long = pd.DataFrame(table_data)

        # Create pivot table for each dataset
        for dataset in df_long['Dataset'].unique():
            df_dataset = df_long[df_long['Dataset'] == dataset]

            # Pivot: MultiIndex rows = (Attack, Defense), columns = Participation
            df_pivot = df_dataset.pivot_table(
                index=['Attack', 'Defense'],
                columns='Participation',
                values='DACC',
                aggfunc='mean'
            )

            # Round to 2 decimals
            df_pivot = df_pivot.round(2)

            # Save CSV
            csv_file = self.output_dir / f"table3b_client_participation_{dataset.lower().replace('-', '_')}.csv"
            df_pivot.to_csv(csv_file)
            print(f"[INFO] Table 3(b) for {dataset} saved to {csv_file}")

            # Save LaTeX
            latex_file = self.output_dir / f"table3b_client_participation_{dataset.lower().replace('-', '_')}.tex"
            with open(latex_file, 'w') as f:
                f.write(df_pivot.to_latex(multirow=True, float_format="%.2f"))
            print(f"[INFO] Table 3(b) LaTeX for {dataset} saved to {latex_file}")

        return df_long

    def generate_table3c_aggregation(self) -> pd.DataFrame:
        """
        Generate Table 3(c): Impact of Aggregation Scheme.
        Rows: Attack + Defense (MultiIndex), Columns: Aggregation schemes.
        """
        if 'aggregation' not in self.ablation_results:
            print("[WARN] No aggregation results available")
            return None

        print("\n" + "="*100)
        print("GENERATING TABLE 3(c): Impact of Aggregation Scheme")
        print("="*100)

        results = self.ablation_results['aggregation']
        table_data = []

        for aggregation, datasets_data in results.items():
            for dataset, attacks_data in datasets_data.items():
                for attack, defenses_data in attacks_data.items():
                    for defense, metrics in defenses_data.items():
                        # Skip baseline
                        if defense in ['baseline', 'baseline_accuracy'] or not isinstance(metrics, dict):
                            continue

                        # Calculate DACC
                        dacc = None
                        if 'detection_metrics' in metrics and metrics['detection_metrics']:
                            dm = metrics['detection_metrics']
                            tp = dm.get('true_positives', 0)
                            tn = dm.get('true_negatives', 0)
                            fp = dm.get('false_positives', 0)
                            fn = dm.get('false_negatives', 0)
                            total = tp + tn + fp + fn
                            dacc = (tp + tn) / total * 100 if total > 0 else 0

                        table_data.append({
                            'Aggregation': aggregation,
                            'Dataset': dataset,
                            'Attack': attack.upper(),
                            'Defense': defense.capitalize(),
                            'DACC': dacc
                        })

        df_long = pd.DataFrame(table_data)

        # Create pivot table for each dataset
        for dataset in df_long['Dataset'].unique():
            df_dataset = df_long[df_long['Dataset'] == dataset]

            # Pivot: MultiIndex rows = (Attack, Defense), columns = Aggregation
            df_pivot = df_dataset.pivot_table(
                index=['Attack', 'Defense'],
                columns='Aggregation',
                values='DACC',
                aggfunc='mean'
            )

            # Round to 2 decimals
            df_pivot = df_pivot.round(2)

            # Save CSV
            csv_file = self.output_dir / f"table3c_aggregation_{dataset.lower().replace('-', '_')}.csv"
            df_pivot.to_csv(csv_file)
            print(f"[INFO] Table 3(c) for {dataset} saved to {csv_file}")

            # Save LaTeX
            latex_file = self.output_dir / f"table3c_aggregation_{dataset.lower().replace('-', '_')}.tex"
            with open(latex_file, 'w') as f:
                f.write(df_pivot.to_latex(multirow=True, float_format="%.2f"))
            print(f"[INFO] Table 3(c) LaTeX for {dataset} saved to {latex_file}")

        return df_long

    def generate_table3d_poison_percentage(self) -> pd.DataFrame:
        """
        Generate Table 3(d): Impact of Poisoning Data Percentage.
        Rows: Attack + Defense (MultiIndex), Columns: Poison percentage values.
        """
        if 'poison_percentage' not in self.ablation_results:
            print("[WARN] No poison percentage results available")
            return None

        print("\n" + "="*100)
        print("GENERATING TABLE 3(d): Impact of Poisoning Data Percentage")
        print("="*100)

        results = self.ablation_results['poison_percentage']
        table_data = []

        for poison_pct, datasets_data in results.items():
            for dataset, attacks_data in datasets_data.items():
                for attack, defenses_data in attacks_data.items():
                    for defense, metrics in defenses_data.items():
                        # Skip baseline
                        if defense in ['baseline', 'baseline_accuracy'] or not isinstance(metrics, dict):
                            continue

                        # Calculate DACC
                        dacc = None
                        if 'detection_metrics' in metrics and metrics['detection_metrics']:
                            dm = metrics['detection_metrics']
                            tp = dm.get('true_positives', 0)
                            tn = dm.get('true_negatives', 0)
                            fp = dm.get('false_positives', 0)
                            fn = dm.get('false_negatives', 0)
                            total = tp + tn + fp + fn
                            dacc = (tp + tn) / total * 100 if total > 0 else 0

                        table_data.append({
                            'Poison%': poison_pct,
                            'Dataset': dataset,
                            'Attack': attack.upper(),
                            'Defense': defense.capitalize(),
                            'DACC': dacc
                        })

        df_long = pd.DataFrame(table_data)

        # Create pivot table for each dataset
        for dataset in df_long['Dataset'].unique():
            df_dataset = df_long[df_long['Dataset'] == dataset]

            # Pivot: MultiIndex rows = (Attack, Defense), columns = Poison%
            df_pivot = df_dataset.pivot_table(
                index=['Attack', 'Defense'],
                columns='Poison%',
                values='DACC',
                aggfunc='mean'
            )

            # Round to 2 decimals
            df_pivot = df_pivot.round(2)

            # Save CSV
            csv_file = self.output_dir / f"table3d_poison_percentage_{dataset.lower().replace('-', '_')}.csv"
            df_pivot.to_csv(csv_file)
            print(f"[INFO] Table 3(d) for {dataset} saved to {csv_file}")

            # Save LaTeX
            latex_file = self.output_dir / f"table3d_poison_percentage_{dataset.lower().replace('-', '_')}.tex"
            with open(latex_file, 'w') as f:
                f.write(df_pivot.to_latex(multirow=True, float_format="%.2f"))
            print(f"[INFO] Table 3(d) LaTeX for {dataset} saved to {latex_file}")

        return df_long

    def generate_table3e_non_iid(self) -> pd.DataFrame:
        """
        Generate Table 3(e): Impact of Non-IID Degree.
        Rows: Attack + Defense (MultiIndex), Columns: Non-IID degree values.
        """
        if 'non_iid' not in self.ablation_results:
            print("[WARN] No Non-IID results available")
            return None

        print("\n" + "="*100)
        print("GENERATING TABLE 3(e): Impact of Non-IID Degree")
        print("="*100)

        results = self.ablation_results['non_iid']
        table_data = []

        for level, datasets_data in results.items():
            for dataset, attacks_data in datasets_data.items():
                for attack, defenses_data in attacks_data.items():
                    for defense, metrics in defenses_data.items():
                        # Skip baseline
                        if defense in ['baseline', 'baseline_accuracy'] or not isinstance(metrics, dict):
                            continue

                        # Calculate DACC
                        dacc = None
                        if 'detection_metrics' in metrics and metrics['detection_metrics']:
                            dm = metrics['detection_metrics']
                            tp = dm.get('true_positives', 0)
                            tn = dm.get('true_negatives', 0)
                            fp = dm.get('false_positives', 0)
                            fn = dm.get('false_negatives', 0)
                            total = tp + tn + fp + fn
                            dacc = (tp + tn) / total * 100 if total > 0 else 0

                        table_data.append({
                            'Non-IID Level': level,
                            'Dataset': dataset,
                            'Attack': attack.upper(),
                            'Defense': defense.capitalize(),
                            'DACC': dacc
                        })

        df_long = pd.DataFrame(table_data)

        # Create pivot table for each dataset
        for dataset in df_long['Dataset'].unique():
            df_dataset = df_long[df_long['Dataset'] == dataset]

            # Pivot: MultiIndex rows = (Attack, Defense), columns = Non-IID Level
            df_pivot = df_dataset.pivot_table(
                index=['Attack', 'Defense'],
                columns='Non-IID Level',
                values='DACC',
                aggfunc='mean'
            )

            # Round to 2 decimals
            df_pivot = df_pivot.round(2)

            # Save CSV
            csv_file = self.output_dir / f"table3e_non_iid_{dataset.lower().replace('-', '_')}.csv"
            df_pivot.to_csv(csv_file)
            print(f"[INFO] Table 3(e) for {dataset} saved to {csv_file}")

            # Save LaTeX
            latex_file = self.output_dir / f"table3e_non_iid_{dataset.lower().replace('-', '_')}.tex"
            with open(latex_file, 'w') as f:
                f.write(df_pivot.to_latex(multirow=True, float_format="%.2f"))
            print(f"[INFO] Table 3(e) LaTeX for {dataset} saved to {latex_file}")

        return df_long

    def plot_poison_percentage_impact(self, df: pd.DataFrame):
        """Plot impact of poison percentage - DACC only."""
        if df is None:
            return

        print("\n[PLOTS] Generating poison percentage impact plots...")

        # Group by poison% and defense, average DACC across attacks
        for dataset in df['Dataset'].unique():
            df_dataset = df[df['Dataset'] == dataset]

            fig, ax = plt.subplots(figsize=(10, 6))
            fig.suptitle(f'Impact of Poison Percentage on DACC - {dataset}', fontsize=14, fontweight='bold')

            for defense in df_dataset['Defense'].unique():
                df_defense = df_dataset[df_dataset['Defense'] == defense]
                avg_metrics = df_defense.groupby('Poison%').agg({
                    'DACC': 'mean'
                })

                # Plot DACC
                ax.plot(avg_metrics.index, avg_metrics['DACC'], marker='o', label=defense, linewidth=2)

            ax.set_title('Detection Accuracy (DACC) vs Poison Percentage', fontsize=12)
            ax.set_xlabel('Poison Percentage', fontsize=10)
            ax.set_ylabel('DACC (%)', fontsize=10)
            ax.grid(alpha=0.3)
            ax.legend(title='Defense', fontsize=9)

            plt.tight_layout()
            plot_file = self.plots_dir / f'poison_percentage_dacc_{dataset.lower().replace("-", "_")}.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved {plot_file}")

    def plot_malicious_ratio_impact(self, df: pd.DataFrame):
        """Plot impact of malicious client ratio - DACC only."""
        if df is None:
            return

        print("\n[PLOTS] Generating malicious ratio impact plots...")

        for dataset in df['Dataset'].unique():
            df_dataset = df[df['Dataset'] == dataset]

            fig, ax = plt.subplots(figsize=(10, 6))
            fig.suptitle(f'Impact of Malicious Client Ratio on DACC - {dataset}', fontsize=14, fontweight='bold')

            for defense in df_dataset['Defense'].unique():
                df_defense = df_dataset[df_dataset['Defense'] == defense]
                avg_metrics = df_defense.groupby('Ratio').agg({
                    'DACC': 'mean'
                })

                # Plot DACC
                ax.plot(avg_metrics.index, avg_metrics['DACC'], marker='o', label=defense, linewidth=2)

            ax.set_title('Detection Accuracy (DACC) vs Malicious Client Ratio', fontsize=12)
            ax.set_xlabel('Malicious Client Ratio', fontsize=10)
            ax.set_ylabel('DACC (%)', fontsize=10)
            ax.grid(alpha=0.3)
            ax.legend(title='Defense', fontsize=9)

            plt.tight_layout()
            plot_file = self.plots_dir / f'malicious_ratio_dacc_{dataset.lower().replace("-", "_")}.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved {plot_file}")

    def plot_non_iid_impact(self, df: pd.DataFrame):
        """Plot impact of Non-IID data distribution - DACC only."""
        if df is None:
            return

        print("\n[PLOTS] Generating Non-IID impact plots...")

        for dataset in df['Dataset'].unique():
            df_dataset = df[df['Dataset'] == dataset]

            fig, ax = plt.subplots(figsize=(10, 6))
            fig.suptitle(f'Impact of Non-IID Degree on DACC - {dataset}', fontsize=14, fontweight='bold')

            for defense in df_dataset['Defense'].unique():
                df_defense = df_dataset[df_dataset['Defense'] == defense]
                avg_metrics = df_defense.groupby('Non-IID Level').agg({
                    'DACC': 'mean'
                })

                # Plot DACC
                ax.plot(avg_metrics.index, avg_metrics['DACC'], marker='o', label=defense, linewidth=2)

            ax.set_title('Detection Accuracy (DACC) vs Non-IID Level', fontsize=12)
            ax.set_xlabel('Non-IID Level', fontsize=10)
            ax.set_ylabel('DACC (%)', fontsize=10)
            ax.grid(alpha=0.3)
            ax.legend(title='Defense', fontsize=9)

            plt.tight_layout()
            plot_file = self.plots_dir / f'non_iid_dacc_{dataset.lower().replace("-", "_")}.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved {plot_file}")

    def plot_client_participation_impact(self, df: pd.DataFrame):
        """Plot impact of client participation - DACC only."""
        if df is None:
            return

        print("\n[PLOTS] Generating client participation impact plots...")

        for dataset in df['Dataset'].unique():
            df_dataset = df[df['Dataset'] == dataset]

            fig, ax = plt.subplots(figsize=(10, 6))
            fig.suptitle(f'Impact of Client Participation on DACC - {dataset}', fontsize=14, fontweight='bold')

            for defense in df_dataset['Defense'].unique():
                df_defense = df_dataset[df_dataset['Defense'] == defense]
                avg_metrics = df_defense.groupby('Participation').agg({
                    'DACC': 'mean'
                })

                # Plot DACC
                ax.plot(avg_metrics.index, avg_metrics['DACC'], marker='o', label=defense, linewidth=2)

            ax.set_title('Detection Accuracy (DACC) vs Client Participation', fontsize=12)
            ax.set_xlabel('Client Participation (%)', fontsize=10)
            ax.set_ylabel('DACC (%)', fontsize=10)
            ax.grid(alpha=0.3)
            ax.legend(title='Defense', fontsize=9)

            plt.tight_layout()
            plot_file = self.plots_dir / f'client_participation_dacc_{dataset.lower().replace("-", "_")}.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved {plot_file}")

    def generate_all(self):
        """Generate all Table 3 variants (a-e) showing DACC for all attacks and defenses."""
        print("\n" + "="*100)
        print("GENERATING TABLE 3: ABLATION STUDY IMPACT ANALYSIS")
        print("="*100)
        print("Each table shows DACC (Detection Accuracy) for all attacks and defenses")
        print("="*100)

        # Generate all 5 tables
        df3a = self.generate_table3a_malicious_ratio()
        df3b = self.generate_table3b_client_participation()
        df3c = self.generate_table3c_aggregation()
        df3d = self.generate_table3d_poison_percentage()
        df3e = self.generate_table3e_non_iid()

        # Generate plots (if needed)
        if df3d is not None:
            self.plot_poison_percentage_impact(df3d)
        if df3a is not None:
            self.plot_malicious_ratio_impact(df3a)
        if df3b is not None:
            self.plot_client_participation_impact(df3b)
        if df3e is not None:
            self.plot_non_iid_impact(df3e)

        print("\n" + "="*100)
        print("✅ TABLE 3 GENERATION COMPLETE")
        print("="*100)
        print(f"Results saved in: {self.output_dir}")
        print(f"Plots saved in: {self.plots_dir}")
        print("\nFiles generated:")
        if df3a is not None:
            print("  - table3a_malicious_ratio.csv/.tex")
        if df3b is not None:
            print("  - table3b_client_participation.csv/.tex")
        if df3c is not None:
            print("  - table3c_aggregation.csv/.tex")
        if df3d is not None:
            print("  - table3d_poison_percentage.csv/.tex")
        if df3e is not None:
            print("  - table3e_non_iid.csv/.tex")
        print("  - Various impact analysis plots (PNG)")
        print("\nTable Structure:")
        print("  Table 3(a): Impact of Malicious Client Ratio")
        print("  Table 3(b): Impact of Total Client Number (Client Participation)")
        print("  Table 3(c): Impact of Aggregation Scheme")
        print("  Table 3(d): Impact of Poisoning Data Percentage")
        print("  Table 3(e): Impact of Non-IID Degree")
        print("="*100)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python generate_table3_ablation.py [options]")
        print("")
        print("Options:")
        print("  --all                    Auto-discover all ablation results (recommended)")
        print("  --poison-pct <file>      Poison percentage results JSON")
        print("  --malicious-ratio <file> Malicious ratio results JSON")
        print("  --non-iid <file>         Non-IID results JSON")
        print("  --participation <file>   Client participation results JSON")
        print("  --aggregation <file>     Aggregation scheme results JSON")
        print("  --output-dir <dir>       Output directory (default: Output/table3_analysis)")
        print("")
        print("Examples:")
        print("  # Auto-discover (easiest)")
        print("  python generate_table3_ablation.py --all")
        print("")
        print("  # Manual file specification")
        print("  python generate_table3_ablation.py \\")
        print("    --poison-pct Output/ablation_poison_percentage/aggregated_results/poison_percentage_results_*.json \\")
        print("    --malicious-ratio Output/ablation_malicious_ratio/aggregated_results/malicious_ratio_results_*.json \\")
        print("    --non-iid Output/ablation_non_iid/aggregated_results/non_iid_results_*.json \\")
        print("    --output-dir Output/table3_analysis")
        sys.exit(1)

    # Parse arguments
    args = sys.argv[1:]
    files = {}
    output_dir = None
    auto_discover = False

    i = 0
    while i < len(args):
        if args[i] == '--all':
            auto_discover = True
            i += 1
        elif args[i] == '--poison-pct' and i + 1 < len(args):
            files['poison_pct_file'] = args[i + 1]
            i += 2
        elif args[i] == '--malicious-ratio' and i + 1 < len(args):
            files['malicious_ratio_file'] = args[i + 1]
            i += 2
        elif args[i] == '--non-iid' and i + 1 < len(args):
            files['non_iid_file'] = args[i + 1]
            i += 2
        elif args[i] == '--participation' and i + 1 < len(args):
            files['participation_file'] = args[i + 1]
            i += 2
        elif args[i] == '--aggregation' and i + 1 < len(args):
            files['aggregation_file'] = args[i + 1]
            i += 2
        elif args[i] == '--output-dir' and i + 1 < len(args):
            output_dir = args[i + 1]
            i += 2
        else:
            i += 1

    # Create analyzer
    analyzer = Table3AblationAnalyzer(output_dir)

    # Load ablation results with auto-discovery if requested
    analyzer.load_ablation_results(**files, auto_discover=auto_discover)

    # Generate all tables and plots
    analyzer.generate_all()


if __name__ == '__main__':
    main()
