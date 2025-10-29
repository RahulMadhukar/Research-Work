"""
Defense Analysis and Reporting Module

Generates comprehensive tables and visualizations for defense effectiveness analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import os


class DefenseAnalyzer:
    """Analyzes defense effectiveness and generates comprehensive reports."""

    def __init__(self, results_dir: str):
        """
        Args:
            results_dir: Directory to save analysis results
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        # Attack name mappings
        self.attack_names = {
            'slf': 'Static Label Flip',
            'dlf': 'Dynamic Label Flip',
            'centralized': 'Centralized Backdoor',
            'coordinated': 'Coordinated Backdoor',
            'random': 'Random Backdoor',
            'model_dependent': 'Model-Dependent'
        }

    def generate_detection_effectiveness_table(self, results: Dict, num_malicious: int, num_clients: int) -> pd.DataFrame:
        """
        Generate Detection Effectiveness table.

        Format:
        Attack Type          | Detection Rate | False Positive Rate
        ---------------------|----------------|--------------------
        Static Label Flip    | 95-100%        | 5-10%
        ...
        """
        data = []

        for dataset_name, dataset_results in results.items():
            for attack_name, attack_results in dataset_results.items():
                if attack_name == 'none':
                    continue

                # Get defense results (committee or any defense)
                # Handle both 'defense_X' and 'X' formats for defense keys
                defense_results = None
                for def_type in ['committee', 'adaptive', 'ensemble', 'reputation', 'gradient']:
                    defense_key = f'defense_{def_type}' if f'defense_{def_type}' in attack_results else def_type
                    if defense_key in attack_results:
                        defense_results = attack_results[defense_key]
                        break

                if defense_results and 'detection_metrics' in defense_results:
                    metrics = defense_results['detection_metrics']

                    # Calculate detection rate (true positives / total malicious)
                    detected = metrics.get('detected_malicious', 0)
                    detection_rate = (detected / num_malicious) * 100 if num_malicious > 0 else 0

                    # Calculate false positive rate (benign flagged as malicious / total benign)
                    false_positives = metrics.get('false_positives', 0)
                    num_benign = num_clients - num_malicious
                    fpr = (false_positives / num_benign) * 100 if num_benign > 0 else 0

                    data.append({
                        'Attack Type': self.attack_names.get(attack_name, attack_name),
                        'Detection Rate': f"{detection_rate:.1f}%",
                        'False Positive Rate': f"{fpr:.1f}%",
                        'Dataset': dataset_name
                    })

        df = pd.DataFrame(data)

        # Save to CSV
        csv_path = os.path.join(self.results_dir, 'detection_effectiveness.csv')
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Detection effectiveness table saved to: {csv_path}")

        # Print formatted table
        print("\n" + "="*80)
        print("DETECTION EFFECTIVENESS")
        print("="*80)
        print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

        return df

    def generate_reputation_tracking_table(self, results: Dict, total_rounds: int) -> pd.DataFrame:
        """
        Generate Reputation Tracking table.

        Format:
        Attack Pattern              | Detection Round | Final Reputation
        ----------------------------|-----------------|------------------
        Continuous malicious (100%) | Round 3-5       | R < 0.3
        ...
        """
        data = []

        for dataset_name, dataset_results in results.items():
            for attack_name, attack_results in dataset_results.items():
                if attack_name == 'none':
                    continue

                # Get reputation-based defense results
                # Handle both 'defense_X' and 'X' formats for defense keys
                for def_type in ['reputation', 'ensemble', 'committee']:
                    defense_key = f'defense_{def_type}' if f'defense_{def_type}' in attack_results else def_type
                    if defense_key in attack_results and 'reputation_history' in attack_results[defense_key]:
                        rep_history = attack_results[defense_key]['reputation_history']

                        # Analyze malicious client reputation
                        if 'malicious_clients' in rep_history:
                            mal_clients = rep_history['malicious_clients']
                            mal_rep_history = rep_history.get('malicious_reputation', [])

                            if mal_rep_history:
                                # Find detection round (when reputation drops below 0.5)
                                detection_round = None
                                for round_idx, rep_val in enumerate(mal_rep_history):
                                    if rep_val < 0.5:
                                        detection_round = round_idx + 1
                                        break

                                final_rep = mal_rep_history[-1] if mal_rep_history else 1.0

                                # Determine attack pattern
                                attack_pattern = f"{self.attack_names.get(attack_name, attack_name)}"

                                data.append({
                                    'Attack Pattern': attack_pattern,
                                    'Detection Round': f"Round {detection_round}" if detection_round else "Not detected",
                                    'Final Reputation': f"R = {final_rep:.3f}",
                                    'Defense': def_type,
                                    'Dataset': dataset_name
                                })
                        break

        df = pd.DataFrame(data)

        # Save to CSV
        csv_path = os.path.join(self.results_dir, 'reputation_tracking.csv')
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Reputation tracking table saved to: {csv_path}")

        # Print formatted table
        print("\n" + "="*80)
        print("REPUTATION-BASED DETECTION TRACKING")
        print("="*80)
        print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

        return df

    def generate_defense_comparison_table(self, results: Dict, dataset_name: str) -> pd.DataFrame:
        """
        Generate Defense Comparison table for a specific dataset.

        Format:
        Defense         | Static LF | Dynamic LF | Backdoor | Avg Acc | Avg FPR
        ----------------|-----------|------------|----------|---------|--------
        None            | 25.3%     | 32.1%      | 28.6%    | 28.7%   | N/A
        ...
        """
        defense_types = ['none', 'committee', 'adaptive', 'reputation', 'gradient', 'ensemble']
        attack_groups = {
            'Static LF': ['slf'],
            'Dynamic LF': ['dlf'],
            'Backdoor': ['centralized', 'coordinated', 'random'],
            'Model-Dep': ['model_dependent']
        }

        data = []
        dataset_results = results.get(dataset_name, {})

        for def_type in defense_types:
            row = {'Defense': def_type.capitalize()}
            accuracies = []
            fprs = []

            for group_name, attack_list in attack_groups.items():
                group_accs = []

                for attack_name in attack_list:
                    if attack_name in dataset_results:
                        attack_res = dataset_results[attack_name]

                        # Get accuracy based on defense type
                        if def_type == 'none':
                            # No defense = attack accuracy
                            acc = attack_res.get('attack', {}).get('final_accuracy', attack_res.get('attack', {}).get('accuracy', 0)) * 100
                        else:
                            # Defense accuracy - handle both 'defense_X' and 'X' formats
                            defense_key = f'defense_{def_type}' if f'defense_{def_type}' in attack_res else def_type
                            acc = attack_res.get(defense_key, {}).get('final_accuracy', attack_res.get(defense_key, {}).get('accuracy', 0)) * 100

                        group_accs.append(acc)
                        accuracies.append(acc)

                        # Get FPR if available
                        if def_type != 'none':
                            defense_key = f'defense_{def_type}' if f'defense_{def_type}' in attack_res else def_type
                            if 'detection_metrics' in attack_res.get(defense_key, {}):
                                fpr = attack_res[defense_key]['detection_metrics'].get('false_positive_rate', 0)
                                fprs.append(fpr)

                # Average for this attack group
                if group_accs:
                    row[group_name] = f"{np.mean(group_accs):.1f}%"
                else:
                    row[group_name] = "N/A"

            # Overall averages
            row['Avg Acc'] = f"{np.mean(accuracies):.1f}%" if accuracies else "N/A"
            row['Avg FPR'] = f"{np.mean(fprs):.1f}%" if fprs else "N/A"

            data.append(row)

        df = pd.DataFrame(data)

        # Save to CSV
        csv_path = os.path.join(self.results_dir, f'defense_comparison_{dataset_name}.csv')
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Defense comparison table saved to: {csv_path}")

        # Print formatted table
        print("\n" + "="*80)
        print(f"DEFENSE COMPARISON - {dataset_name}")
        print("="*80)
        print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

        return df

    def plot_defense_comparison(self, results: Dict, output_dir: str):
        """Generate comprehensive defense comparison plots."""

        # 1. Defense Accuracy Comparison (Bar Chart)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for idx, dataset_name in enumerate(['Fashion-MNIST', 'CIFAR-10']):
            if dataset_name not in results:
                continue

            dataset_results = results[dataset_name]
            defense_types = ['none', 'committee', 'adaptive', 'reputation', 'gradient', 'ensemble']
            attack_types = ['slf', 'dlf', 'centralized', 'coordinated', 'random', 'model_dependent']

            # Prepare data for grouped bar chart
            defense_accs = {def_type: [] for def_type in defense_types}

            for attack in attack_types:
                if attack not in dataset_results:
                    continue

                attack_res = dataset_results[attack]

                for def_type in defense_types:
                    if def_type == 'none':
                        acc = attack_res.get('attack', {}).get('final_accuracy', attack_res.get('attack', {}).get('accuracy', 0)) * 100
                    else:
                        # Handle both 'defense_X' and 'X' formats for defense keys
                        defense_key = f'defense_{def_type}' if f'defense_{def_type}' in attack_res else def_type
                        acc = attack_res.get(defense_key, {}).get('final_accuracy', attack_res.get(defense_key, {}).get('accuracy', 0)) * 100
                    defense_accs[def_type].append(acc)

            # Plot grouped bars
            x = np.arange(len(attack_types))
            width = 0.12
            colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

            for i, (def_type, color) in enumerate(zip(defense_types, colors)):
                if defense_accs[def_type]:
                    offset = (i - len(defense_types)/2) * width
                    axes[idx].bar(x + offset, defense_accs[def_type], width,
                                 label=def_type.capitalize(), color=color, alpha=0.8)

            axes[idx].set_xlabel('Attack Type', fontsize=12)
            axes[idx].set_ylabel('Accuracy (%)', fontsize=12)
            axes[idx].set_title(f'{dataset_name}', fontsize=14, fontweight='bold')
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels([a.upper() for a in attack_types], rotation=45, ha='right')
            axes[idx].legend(loc='upper right', fontsize=9)
            axes[idx].grid(axis='y', alpha=0.3)
            axes[idx].set_ylim(0, 100)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'defense_accuracy_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Defense comparison plot saved to: {plot_path}")

        # 2. Detection Rate Heatmap
        self._plot_detection_heatmap(results, output_dir)

    def _plot_detection_heatmap(self, results: Dict, output_dir: str):
        """Generate heatmap of detection rates across defenses and attacks."""

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for idx, dataset_name in enumerate(['Fashion-MNIST', 'CIFAR-10']):
            if dataset_name not in results:
                continue

            dataset_results = results[dataset_name]
            defense_types = ['committee', 'adaptive', 'reputation', 'gradient', 'ensemble']
            attack_types = ['slf', 'dlf', 'centralized', 'coordinated', 'random', 'model_dependent']

            # Build detection rate matrix
            detection_matrix = []

            for def_type in defense_types:
                row = []
                for attack in attack_types:
                    if attack in dataset_results:
                        # Handle both 'defense_X' and 'X' formats for defense keys
                        defense_key = f'defense_{def_type}' if f'defense_{def_type}' in dataset_results[attack] else def_type
                        if defense_key in dataset_results[attack]:
                            metrics = dataset_results[attack][defense_key].get('detection_metrics', {})
                            detection_rate = metrics.get('detection_rate', 0)
                            # Handle both percentage (0-100) and fraction (0-1) formats
                            if detection_rate > 1:
                                row.append(detection_rate)
                            else:
                                row.append(detection_rate * 100)
                        else:
                            row.append(0)
                    else:
                        row.append(0)
                detection_matrix.append(row)

            # Plot heatmap
            im = axes[idx].imshow(detection_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

            axes[idx].set_xticks(np.arange(len(attack_types)))
            axes[idx].set_yticks(np.arange(len(defense_types)))
            axes[idx].set_xticklabels([a.upper() for a in attack_types], rotation=45, ha='right')
            axes[idx].set_yticklabels([d.capitalize() for d in defense_types])
            axes[idx].set_title(f'Detection Rates - {dataset_name}', fontsize=14, fontweight='bold')

            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[idx])
            cbar.set_label('Detection Rate (%)', rotation=270, labelpad=20)

            # Annotate cells
            for i in range(len(defense_types)):
                for j in range(len(attack_types)):
                    text = axes[idx].text(j, i, f'{detection_matrix[i][j]:.0f}%',
                                        ha="center", va="center", color="black", fontsize=9)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'detection_rate_heatmap.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Detection heatmap saved to: {plot_path}")

    def print_comprehensive_summary(self, results: Dict):
        """Print comprehensive summary of all defense test results (moved from evaluation.py)."""
        print("\n" + "="*120)
        print("COMPREHENSIVE DEFENSE TEST SUMMARY")
        print("="*120)

        for dataset_name, dataset_results in results.items():
            print(f"\n{dataset_name}:")
            print("-" * 140)
            print(f"{'Attack':<18} {'Baseline':<10} {'Attack':<10} {'Committee':<11} {'Adaptive':<11} {'Reputation':<11} {'Gradient':<11} {'Ensemble':<11} {'Best'}")
            print("-" * 140)

            for attack_name, attack_results in dataset_results.items():
                if 'attack' not in attack_results:
                    continue  # Skip if attack phase failed
                baseline_acc = attack_results['baseline'].get('final_accuracy', attack_results['baseline'].get('accuracy', 0.0))
                attack_acc = attack_results['attack'].get('final_accuracy', attack_results['attack'].get('accuracy', 0.0))

                # Handle both 'defense_X' and 'X' formats for defense keys
                committee_acc = attack_results.get('defense_committee', attack_results.get('committee', {})).get('final_accuracy', attack_results.get('defense_committee', attack_results.get('committee', {})).get('accuracy', 0.0))
                adaptive_acc = attack_results.get('defense_adaptive', attack_results.get('adaptive', {})).get('final_accuracy', attack_results.get('defense_adaptive', attack_results.get('adaptive', {})).get('accuracy', 0.0))
                reputation_acc = attack_results.get('defense_reputation', attack_results.get('reputation', {})).get('final_accuracy', attack_results.get('defense_reputation', attack_results.get('reputation', {})).get('accuracy', 0.0))
                gradient_acc = attack_results.get('defense_gradient', attack_results.get('gradient', {})).get('final_accuracy', attack_results.get('defense_gradient', attack_results.get('gradient', {})).get('accuracy', 0.0))
                ensemble_acc = attack_results.get('defense_ensemble', attack_results.get('ensemble', {})).get('final_accuracy', attack_results.get('defense_ensemble', attack_results.get('ensemble', {})).get('accuracy', 0.0))

                # Find best defense
                defense_accs = {
                    'committee': committee_acc,
                    'adaptive': adaptive_acc,
                    'reputation': reputation_acc,
                    'gradient': gradient_acc,
                    'ensemble': ensemble_acc
                }
                best_defense = max(defense_accs, key=defense_accs.get)

                print(f"{attack_name:<18} {baseline_acc:>8.4f}  {attack_acc:>8.4f}  "
                      f"{committee_acc:>9.4f}  {adaptive_acc:>9.4f}  {reputation_acc:>9.4f}  "
                      f"{gradient_acc:>9.4f}  {ensemble_acc:>9.4f}  {best_defense.capitalize()}")

            print("\nRecovery Rates (%):")
            print("-" * 140)
            print(f"{'Attack':<18} {'Committee':<13} {'Adaptive':<13} {'Reputation':<13} {'Gradient':<13} {'Ensemble':<13} {'Best'}")
            print("-" * 140)

            for attack_name, attack_results in dataset_results.items():
                if 'attack' not in attack_results:
                    continue
                # Handle both 'defense_X' and 'X' formats for defense keys
                committee_rec = attack_results.get('defense_committee', attack_results.get('committee', {})).get('recovery_rate', 0.0)
                adaptive_rec = attack_results.get('defense_adaptive', attack_results.get('adaptive', {})).get('recovery_rate', 0.0)
                reputation_rec = attack_results.get('defense_reputation', attack_results.get('reputation', {})).get('recovery_rate', 0.0)
                gradient_rec = attack_results.get('defense_gradient', attack_results.get('gradient', {})).get('recovery_rate', 0.0)
                ensemble_rec = attack_results.get('defense_ensemble', attack_results.get('ensemble', {})).get('recovery_rate', 0.0)

                # Find best recovery
                recovery_rates = {
                    'committee': committee_rec,
                    'adaptive': adaptive_rec,
                    'reputation': reputation_rec,
                    'gradient': gradient_rec,
                    'ensemble': ensemble_rec
                }
                best_recovery = max(recovery_rates, key=recovery_rates.get)

                print(f"{attack_name:<18} {committee_rec:>11.2f}%  {adaptive_rec:>11.2f}%  {reputation_rec:>11.2f}%  "
                      f"{gradient_rec:>11.2f}%  {ensemble_rec:>11.2f}%  {best_recovery.capitalize()}")

        print("\n" + "="*120)

    def generate_all_analysis(self, results: Dict, num_malicious: int, num_clients: int, total_rounds: int):
        """Generate all analysis tables and plots."""

        print("\n" + "="*100)
        print("GENERATING COMPREHENSIVE DEFENSE ANALYSIS")
        print("="*100)

        # NOTE: Detection metrics and reputation history are not currently tracked
        # by the coordinator, so we skip those tables for now

        # Only generate tables that work with available data (accuracy, recovery_rate)

        # 1. Defense Comparison for each dataset (works with accuracy data)
        for dataset_name in results.keys():
            self.generate_defense_comparison_table(results, dataset_name)

        # 2. Visualization plots (works with accuracy data)
        try:
            self.plot_defense_comparison(results, self.results_dir)
        except Exception as e:
            print(f"[WARN] Failed to generate plots: {e}")

        # 3. Print comprehensive summary (works with accuracy and recovery_rate)
        self.print_comprehensive_summary(results)

        print("\n" + "="*100)
        print("ANALYSIS COMPLETE")
        print(f"All tables and plots saved to: {self.results_dir}")
        print("="*100)
