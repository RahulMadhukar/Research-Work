import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# Attack color scheme
ATTACK_COLORS = {
    'slf': '#FF6B6B',           # Red
    'dlf': '#4ECDC4',           # Teal
    'centralized': "#1F06AD",   # Blue
    'coordinated': '#FFD700',   # Gold
    'random': "#4B7907",        # Green
    'model_dependent': '#FF69B4' # Pink
}


class PlottingEngine:
    """
    Centralized plotting engine for all evaluation visualizations.
    Focuses on committee-based defense analysis.
    """
    
    def __init__(self, output_dir, plots_generated=None, run_id=None):
        """
        Initialize plotting engine.
        
        Args:
            output_dir: Base directory for saving plots
            plots_generated: List to track generated plot filenames
            run_id: Run identifier for filenames (optional)
        """
        self.out_dir = output_dir
        self.plots_generated = plots_generated if plots_generated is not None else []
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.out_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8')
    
    def _save_plot(self, fig, filename):
        """Helper to save plot and track it."""
        filepath = os.path.join(self.out_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        self.plots_generated.append(filepath)
        return 
    
    def _filter_valid_results(self, all_results):
        """
        Filter out datasets or attacks that lack valid 'attack' results.
        Prevents KeyError when accessing missing entries.
        """
        filtered_results = {}
        for dataset_name, dataset_results in all_results.items():
            valid_attacks = {}
            for attack_name, attack_data in dataset_results.items():
                # Only keep attack entries that contain the 'attack' scenario
                if isinstance(attack_data, dict) and 'attack' in attack_data:
                    valid_attacks[attack_name] = attack_data
                else:
                    print(f"[SKIP] {dataset_name} → {attack_name}: Missing 'attack' key (baseline-only).")
            if valid_attacks:
                filtered_results[dataset_name] = valid_attacks

        if not filtered_results:
            print("[WARNING] No valid attack results found after filtering.")
        else:
            print(f"[INFO] {sum(len(v) for v in filtered_results.values())} valid attack scenarios retained.")
        return filtered_results

    

    def _get_scenario_data(self, dataset_results, attack_name, scenario, key):
        """
        Safely extract data from nested results structure.

        Args:
            dataset_results: Results dict for a dataset
            attack_name: Name of attack (use 'none' for baseline-only data)
            scenario: 'baseline', 'attack', 'defense', or defense type like 'committee', 'adaptive', etc.
            key: Data key to extract

        Returns:
            List of values or empty list if not found
        """
        try:
            # Handle both 'defense_X' and 'X' formats for defense keys
            attack_data = dataset_results.get(attack_name, {})

            # If attack_data is empty and this is a baseline request
            if not attack_data and scenario == 'baseline':
                # Try 'none' first
                if 'none' in dataset_results:
                    attack_data = dataset_results.get('none', {})

                # If still empty after trying 'none', try getting baseline from first available attack
                if not attack_data:  # Changed from elif to if
                    for atk_key in dataset_results.keys():
                        if atk_key.lower() != 'none' and dataset_results[atk_key].get('baseline'):
                            attack_data = dataset_results[atk_key]
                            break

            if not attack_data:
                return []

            # Try original scenario key first
            scenario_data = attack_data.get(scenario, {})

            # If not found and scenario is a defense type, try with 'defense_' prefix
            if not scenario_data and scenario not in ['baseline', 'attack', 'defense']:
                scenario_data = attack_data.get(f'defense_{scenario}', {})

            # Debug: Log when defense data is not found (only once per combo to avoid spam)
            if not scenario_data and scenario not in ['baseline', 'attack']:
                # Check what keys are actually available
                if isinstance(attack_data, dict):
                    available_keys = list(attack_data.keys())
                    if len(available_keys) > 0:  # Only log if there are keys available
                        print(f"[DEBUG] Defense '{scenario}' not found for attack '{attack_name}'. Available: {available_keys[:5]}")  # Show first 5 keys only

            data = scenario_data.get(key, [])
            if data is None:
                return []
            if not isinstance(data, (list, tuple, np.ndarray)):
                return [float(data)]

            # Convert to list and add debug info for history data
            result = [float(x) for x in data]
            if 'history' in key.lower() and len(result) == 1:
                print(f"[DEBUG] Only 1 round found for {attack_name}/{scenario}/{key}. Expected 5 rounds!")
            return result
        except Exception as e:
            print(f"[WARN] Failed to extract {key} for {attack_name}/{scenario}: {e}")
            return []

    def _get_final_accuracy(self, dataset_results, attack_name, scenario):
        """
        Get the final accuracy value from results (last round).

        Args:
            dataset_results: Results dict for a dataset
            attack_name: Name of attack (use 'none' for baseline-only data)
            scenario: 'baseline', 'attack', or defense type like 'committee', 'adaptive', etc.

        Returns:
            Final accuracy value (float) or 0.0 if not found
        """
        try:
            attack_data = dataset_results.get(attack_name, {})

            # If attack_data is empty and this is a baseline request
            if not attack_data and scenario == 'baseline':
                # Try 'none' first
                if 'none' in dataset_results:
                    attack_data = dataset_results.get('none', {})

                # If still empty after trying 'none', try getting baseline from first available attack
                if not attack_data:  # Changed from elif to if
                    for atk_key in dataset_results.keys():
                        if atk_key.lower() != 'none' and dataset_results[atk_key].get('baseline'):
                            attack_data = dataset_results[atk_key]
                            break

            if not attack_data:
                return 0.0

            # Try original scenario key first
            scenario_data = attack_data.get(scenario, {})

            # If not found and scenario is a defense type, try with 'defense_' prefix
            if not scenario_data and scenario not in ['baseline', 'attack', 'defense']:
                scenario_data = attack_data.get(f'defense_{scenario}', {})

            # Try to get final_accuracy first
            final_acc = scenario_data.get('final_accuracy', None)
            if final_acc is not None:
                return float(final_acc)

            # Try to get accuracy
            acc = scenario_data.get('accuracy', None)
            if acc is not None:
                return float(acc)

            # Try to get last value from test_acc_history
            test_acc_hist = scenario_data.get('test_acc_history', [])
            if test_acc_hist and len(test_acc_hist) > 0:
                return float(test_acc_hist[-1])

            # Try to get last value from training_acc_history
            train_acc_hist = scenario_data.get('training_acc_history', [])
            if train_acc_hist and len(train_acc_hist) > 0:
                return float(train_acc_hist[-1])

            return 0.0
        except Exception as e:
            print(f"[WARN] Failed to get final accuracy for {attack_name}/{scenario}: {e}")
            return 0.0

    def _calculate_recovery_rate(self, baseline_acc, attack_acc, defense_acc):
        """
        Calculate defense recovery rate as a percentage.

        Recovery Rate = (Defense Acc - Attack Acc) / (Baseline Acc - Attack Acc) * 100

        Args:
            baseline_acc: Baseline accuracy (no attack)
            attack_acc: Accuracy under attack
            defense_acc: Accuracy with defense

        Returns:
            Recovery rate percentage (0-100)
        """
        try:
            if baseline_acc == attack_acc:
                # No degradation from attack
                return 100.0 if defense_acc >= baseline_acc else 0.0

            recovery = (defense_acc - attack_acc) / (baseline_acc - attack_acc) * 100
            return max(0.0, min(100.0, recovery))
        except Exception as e:
            print(f"[WARN] Failed to calculate recovery rate: {e}")
            return 0.0

    def _calculate_attack_success_rate(self, baseline_acc, attack_acc):
        """
        Calculate attack success rate (how much accuracy degradation).

        Attack Success Rate = (Baseline Acc - Attack Acc) / Baseline Acc

        Args:
            baseline_acc: Baseline accuracy (no attack)
            attack_acc: Accuracy under attack

        Returns:
            Attack success rate (0.0-1.0)
        """
        try:
            if baseline_acc == 0:
                return 0.0
            asr = (baseline_acc - attack_acc) / baseline_acc
            return max(0.0, min(1.0, asr))
        except Exception as e:
            print(f"[WARN] Failed to calculate attack success rate: {e}")
            return 0.0
    

    # =====================================================================
    # CONSOLIDATED METRIC PLOTS (ONE PLOT PER METRIC, ALL ATTACKS AS SUBPLOTS)
    # =====================================================================

    def plot_training_accuracy_vs_round(self, results):
        """
        Create one plot for Training Accuracy vs Round with 6 subplots (one per attack).
        Each subplot shows baseline, attack, and all 5 defenses.
        """
        defense_types = ['committee', 'adaptive', 'reputation', 'gradient', 'ensemble']
        defense_colors = {
            'committee': '#4169E1',
            'adaptive': '#FF8C00',
            'reputation': '#9370DB',
            'gradient': '#20B2AA',
            'ensemble': '#DC143C'
        }

        for dataset_name, dataset_results in results.items():
            # Get all attacks except 'none'
            attacks = [atk for atk in dataset_results.keys() if atk.lower() != 'none']
            if not attacks:
                continue

            # Create subplots (2 rows x 3 columns for 6 attacks)
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(f'{dataset_name} - Training Accuracy vs Round (All Attacks)',
                        fontsize=18, fontweight='bold', y=0.995)
            axes = axes.flatten()

            for idx, attack_name in enumerate(attacks):
                if idx >= 6:  # Only show first 6 attacks
                    break
                ax = axes[idx]

                # Baseline
                base_hist = self._get_scenario_data(dataset_results, 'none', 'baseline', 'training_acc_history')
                if base_hist:
                    ax.plot(range(1, len(base_hist)+1), base_hist, label='Baseline',
                           color='#2E8B57', linewidth=2, marker='o', markersize=3)

                # Attack
                att_hist = self._get_scenario_data(dataset_results, attack_name, 'attack', 'training_acc_history')
                if att_hist:
                    color = ATTACK_COLORS.get(attack_name.lower(), '#DC143C')
                    ax.plot(range(1, len(att_hist)+1), att_hist, label=f'{attack_name.upper()}',
                           color=color, linewidth=2, marker='s', markersize=3)

                # Defenses
                for def_type in defense_types:
                    def_hist = self._get_scenario_data(dataset_results, attack_name, def_type, 'training_acc_history')
                    if def_hist:
                        color = defense_colors.get(def_type, '#666666')
                        ax.plot(range(1, len(def_hist)+1), def_hist, label=f'{def_type.capitalize()}',
                               color=color, linewidth=2, marker='x', markersize=3, alpha=0.8)

                ax.set_title(f'{attack_name.upper()} Attack', fontweight='bold', fontsize=12)
                ax.set_xlabel('Rounds', fontweight='bold')
                ax.set_ylabel('Training Accuracy', fontweight='bold')
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8, loc='best')

            # Hide unused subplots
            for idx in range(len(attacks), 6):
                axes[idx].axis('off')

            plt.tight_layout()
            self._save_plot(fig, f'{dataset_name.lower()}_training_accuracy_vs_round.png')

    def plot_training_loss_vs_round(self, results):
        """
        Create one plot for Training Loss vs Round with 6 subplots (one per attack).
        Each subplot shows baseline, attack, and all 5 defenses.
        """
        defense_types = ['committee', 'adaptive', 'reputation', 'gradient', 'ensemble']
        defense_colors = {
            'committee': '#4169E1',
            'adaptive': '#FF8C00',
            'reputation': '#9370DB',
            'gradient': '#20B2AA',
            'ensemble': '#DC143C'
        }

        for dataset_name, dataset_results in results.items():
            attacks = [atk for atk in dataset_results.keys() if atk.lower() != 'none']
            if not attacks:
                continue

            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(f'{dataset_name} - Training Loss vs Round (All Attacks)',
                        fontsize=18, fontweight='bold', y=0.995)
            axes = axes.flatten()

            for idx, attack_name in enumerate(attacks):
                if idx >= 6:
                    break
                ax = axes[idx]

                # Baseline
                base_hist = self._get_scenario_data(dataset_results, 'none', 'baseline', 'training_loss_history')
                if base_hist:
                    ax.plot(range(1, len(base_hist)+1), base_hist, label='Baseline',
                           color='#2E8B57', linewidth=2, marker='o', markersize=3)

                # Attack
                att_hist = self._get_scenario_data(dataset_results, attack_name, 'attack', 'training_loss_history')
                if att_hist:
                    color = ATTACK_COLORS.get(attack_name.lower(), '#DC143C')
                    ax.plot(range(1, len(att_hist)+1), att_hist, label=f'{attack_name.upper()}',
                           color=color, linewidth=2, marker='s', markersize=3)

                # Defenses
                for def_type in defense_types:
                    def_hist = self._get_scenario_data(dataset_results, attack_name, def_type, 'training_loss_history')
                    if def_hist:
                        color = defense_colors.get(def_type, '#666666')
                        ax.plot(range(1, len(def_hist)+1), def_hist, label=f'{def_type.capitalize()}',
                               color=color, linewidth=2, marker='x', markersize=3, alpha=0.8)

                ax.set_title(f'{attack_name.upper()} Attack', fontweight='bold', fontsize=12)
                ax.set_xlabel('Rounds', fontweight='bold')
                ax.set_ylabel('Training Loss', fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8, loc='best')

            for idx in range(len(attacks), 6):
                axes[idx].axis('off')

            plt.tight_layout()
            self._save_plot(fig, f'{dataset_name.lower()}_training_loss_vs_round.png')

    def plot_testing_accuracy_vs_round(self, results):
        """
        Create one plot for Testing Accuracy vs Round with 6 subplots (one per attack).
        Each subplot shows baseline, attack, and all 5 defenses.
        """
        defense_types = ['committee', 'adaptive', 'reputation', 'gradient', 'ensemble']
        defense_colors = {
            'committee': '#4169E1',
            'adaptive': '#FF8C00',
            'reputation': '#9370DB',
            'gradient': '#20B2AA',
            'ensemble': '#DC143C'
        }

        for dataset_name, dataset_results in results.items():
            attacks = [atk for atk in dataset_results.keys() if atk.lower() != 'none']
            if not attacks:
                continue

            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(f'{dataset_name} - Testing Accuracy vs Round (All Attacks)',
                        fontsize=18, fontweight='bold', y=0.995)
            axes = axes.flatten()

            for idx, attack_name in enumerate(attacks):
                if idx >= 6:
                    break
                ax = axes[idx]

                # Baseline
                base_hist = self._get_scenario_data(dataset_results, 'none', 'baseline', 'test_acc_history')
                if base_hist:
                    ax.plot(range(1, len(base_hist)+1), base_hist, label='Baseline',
                           color='#2E8B57', linewidth=2, marker='o', markersize=3)

                # Attack
                att_hist = self._get_scenario_data(dataset_results, attack_name, 'attack', 'test_acc_history')
                if att_hist:
                    color = ATTACK_COLORS.get(attack_name.lower(), '#DC143C')
                    ax.plot(range(1, len(att_hist)+1), att_hist, label=f'{attack_name.upper()}',
                           color=color, linewidth=2, marker='s', markersize=3)

                # Defenses
                for def_type in defense_types:
                    def_hist = self._get_scenario_data(dataset_results, attack_name, def_type, 'test_acc_history')
                    if def_hist:
                        color = defense_colors.get(def_type, '#666666')
                        ax.plot(range(1, len(def_hist)+1), def_hist, label=f'{def_type.capitalize()}',
                               color=color, linewidth=2, marker='x', markersize=3, alpha=0.8)

                ax.set_title(f'{attack_name.upper()} Attack', fontweight='bold', fontsize=12)
                ax.set_xlabel('Rounds', fontweight='bold')
                ax.set_ylabel('Testing Accuracy', fontweight='bold')
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8, loc='best')

            for idx in range(len(attacks), 6):
                axes[idx].axis('off')

            plt.tight_layout()
            self._save_plot(fig, f'{dataset_name.lower()}_testing_accuracy_vs_round.png')

    def plot_testing_loss_vs_round(self, results):
        """
        Create one plot for Testing Loss vs Round with 6 subplots (one per attack).
        Each subplot shows baseline, attack, and all 5 defenses.
        """
        defense_types = ['committee', 'adaptive', 'reputation', 'gradient', 'ensemble']
        defense_colors = {
            'committee': '#4169E1',
            'adaptive': '#FF8C00',
            'reputation': '#9370DB',
            'gradient': '#20B2AA',
            'ensemble': '#DC143C'
        }

        for dataset_name, dataset_results in results.items():
            attacks = [atk for atk in dataset_results.keys() if atk.lower() != 'none']
            if not attacks:
                continue

            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(f'{dataset_name} - Testing Loss vs Round (All Attacks)',
                        fontsize=18, fontweight='bold', y=0.995)
            axes = axes.flatten()

            for idx, attack_name in enumerate(attacks):
                if idx >= 6:
                    break
                ax = axes[idx]

                # Baseline
                base_hist = self._get_scenario_data(dataset_results, 'none', 'baseline', 'test_loss_history')
                if base_hist:
                    ax.plot(range(1, len(base_hist)+1), base_hist, label='Baseline',
                           color='#2E8B57', linewidth=2, marker='o', markersize=3)

                # Attack
                att_hist = self._get_scenario_data(dataset_results, attack_name, 'attack', 'test_loss_history')
                if att_hist:
                    color = ATTACK_COLORS.get(attack_name.lower(), '#DC143C')
                    ax.plot(range(1, len(att_hist)+1), att_hist, label=f'{attack_name.upper()}',
                           color=color, linewidth=2, marker='s', markersize=3)

                # Defenses
                for def_type in defense_types:
                    def_hist = self._get_scenario_data(dataset_results, attack_name, def_type, 'test_loss_history')
                    if def_hist:
                        color = defense_colors.get(def_type, '#666666')
                        ax.plot(range(1, len(def_hist)+1), def_hist, label=f'{def_type.capitalize()}',
                               color=color, linewidth=2, marker='x', markersize=3, alpha=0.8)

                ax.set_title(f'{attack_name.upper()} Attack', fontweight='bold', fontsize=12)
                ax.set_xlabel('Rounds', fontweight='bold')
                ax.set_ylabel('Testing Loss', fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8, loc='best')

            for idx in range(len(attacks), 6):
                axes[idx].axis('off')

            plt.tight_layout()
            self._save_plot(fig, f'{dataset_name.lower()}_testing_loss_vs_round.png')

    def plot_train_vs_test_accuracy(self, results):
        """
        Create one plot for Train vs Test Accuracy with 6 subplots (one per attack).
        Each subplot shows train/test for baseline, attack, and all 5 defenses.
        """
        defense_types = ['committee', 'adaptive', 'reputation', 'gradient', 'ensemble']
        defense_colors = {
            'committee': '#4169E1',
            'adaptive': '#FF8C00',
            'reputation': '#9370DB',
            'gradient': '#20B2AA',
            'ensemble': '#DC143C'
        }

        for dataset_name, dataset_results in results.items():
            attacks = [atk for atk in dataset_results.keys() if atk.lower() != 'none']
            if not attacks:
                continue

            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(f'{dataset_name} - Train vs Test Accuracy (All Attacks)',
                        fontsize=18, fontweight='bold', y=0.995)
            axes = axes.flatten()

            for idx, attack_name in enumerate(attacks):
                if idx >= 6:
                    break
                ax = axes[idx]

                # Baseline
                train_acc = self._get_scenario_data(dataset_results, 'none', 'baseline', 'training_acc_history')
                test_acc = self._get_scenario_data(dataset_results, 'none', 'baseline', 'test_acc_history')
                if train_acc and test_acc:
                    rounds = range(1, len(train_acc)+1)
                    ax.plot(rounds, train_acc, label='Baseline Train', color='#2E8B57',
                           linewidth=2, marker='o', markersize=3, linestyle='-')
                    ax.plot(rounds, test_acc, label='Baseline Test', color='#2E8B57',
                           linewidth=2, marker='x', markersize=3, linestyle='--', alpha=0.7)

                # Attack
                train_acc = self._get_scenario_data(dataset_results, attack_name, 'attack', 'training_acc_history')
                test_acc = self._get_scenario_data(dataset_results, attack_name, 'attack', 'test_acc_history')
                if train_acc and test_acc:
                    rounds = range(1, len(train_acc)+1)
                    color = ATTACK_COLORS.get(attack_name.lower(), '#DC143C')
                    ax.plot(rounds, train_acc, label=f'{attack_name.upper()} Train', color=color,
                           linewidth=2, marker='o', markersize=3, linestyle='-')
                    ax.plot(rounds, test_acc, label=f'{attack_name.upper()} Test', color=color,
                           linewidth=2, marker='x', markersize=3, linestyle='--', alpha=0.7)

                # Defenses
                for def_type in defense_types:
                    train_acc = self._get_scenario_data(dataset_results, attack_name, def_type, 'training_acc_history')
                    test_acc = self._get_scenario_data(dataset_results, attack_name, def_type, 'test_acc_history')
                    if train_acc and test_acc:
                        rounds = range(1, len(train_acc)+1)
                        color = defense_colors.get(def_type, '#666666')
                        ax.plot(rounds, train_acc, label=f'{def_type.capitalize()} Train', color=color,
                               linewidth=2, marker='o', markersize=3, linestyle='-', alpha=0.8)
                        ax.plot(rounds, test_acc, label=f'{def_type.capitalize()} Test', color=color,
                               linewidth=2, marker='x', markersize=3, linestyle='--', alpha=0.6)

                ax.set_title(f'{attack_name.upper()} Attack', fontweight='bold', fontsize=12)
                ax.set_xlabel('Rounds', fontweight='bold')
                ax.set_ylabel('Accuracy', fontweight='bold')
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=7, loc='best', ncol=2)

            for idx in range(len(attacks), 6):
                axes[idx].axis('off')

            plt.tight_layout()
            self._save_plot(fig, f'{dataset_name.lower()}_train_vs_test_accuracy.png')

    def plot_train_vs_test_loss(self, results):
        """
        Create one plot for Train vs Test Loss with 6 subplots (one per attack).
        Each subplot shows train/test for baseline, attack, and all 5 defenses.
        """
        defense_types = ['committee', 'adaptive', 'reputation', 'gradient', 'ensemble']
        defense_colors = {
            'committee': '#4169E1',
            'adaptive': '#FF8C00',
            'reputation': '#9370DB',
            'gradient': '#20B2AA',
            'ensemble': '#DC143C'
        }

        for dataset_name, dataset_results in results.items():
            attacks = [atk for atk in dataset_results.keys() if atk.lower() != 'none']
            if not attacks:
                continue

            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(f'{dataset_name} - Train vs Test Loss (All Attacks)',
                        fontsize=18, fontweight='bold', y=0.995)
            axes = axes.flatten()

            for idx, attack_name in enumerate(attacks):
                if idx >= 6:
                    break
                ax = axes[idx]

                # Baseline
                train_loss = self._get_scenario_data(dataset_results, 'none', 'baseline', 'training_loss_history')
                test_loss = self._get_scenario_data(dataset_results, 'none', 'baseline', 'test_loss_history')
                if train_loss and test_loss:
                    rounds = range(1, len(train_loss)+1)
                    ax.plot(rounds, train_loss, label='Baseline Train', color='#2E8B57',
                           linewidth=2, marker='o', markersize=3, linestyle='-')
                    ax.plot(rounds, test_loss, label='Baseline Test', color='#2E8B57',
                           linewidth=2, marker='x', markersize=3, linestyle='--', alpha=0.7)

                # Attack
                train_loss = self._get_scenario_data(dataset_results, attack_name, 'attack', 'training_loss_history')
                test_loss = self._get_scenario_data(dataset_results, attack_name, 'attack', 'test_loss_history')
                if train_loss and test_loss:
                    rounds = range(1, len(train_loss)+1)
                    color = ATTACK_COLORS.get(attack_name.lower(), '#DC143C')
                    ax.plot(rounds, train_loss, label=f'{attack_name.upper()} Train', color=color,
                           linewidth=2, marker='o', markersize=3, linestyle='-')
                    ax.plot(rounds, test_loss, label=f'{attack_name.upper()} Test', color=color,
                           linewidth=2, marker='x', markersize=3, linestyle='--', alpha=0.7)

                # Defenses
                for def_type in defense_types:
                    train_loss = self._get_scenario_data(dataset_results, attack_name, def_type, 'training_loss_history')
                    test_loss = self._get_scenario_data(dataset_results, attack_name, def_type, 'test_loss_history')
                    if train_loss and test_loss:
                        rounds = range(1, len(train_loss)+1)
                        color = defense_colors.get(def_type, '#666666')
                        ax.plot(rounds, train_loss, label=f'{def_type.capitalize()} Train', color=color,
                               linewidth=2, marker='o', markersize=3, linestyle='-', alpha=0.8)
                        ax.plot(rounds, test_loss, label=f'{def_type.capitalize()} Test', color=color,
                               linewidth=2, marker='x', markersize=3, linestyle='--', alpha=0.6)

                ax.set_title(f'{attack_name.upper()} Attack', fontweight='bold', fontsize=12)
                ax.set_xlabel('Rounds', fontweight='bold')
                ax.set_ylabel('Loss', fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=7, loc='best', ncol=2)

            for idx in range(len(attacks), 6):
                axes[idx].axis('off')

            plt.tight_layout()
            self._save_plot(fig, f'{dataset_name.lower()}_train_vs_test_loss.png')

    def plot_training_convergence(self, results):
        """
        Create one plot for Training Convergence with 6 subplots (one per attack).
        Each subplot shows convergence for baseline, attack, and all 5 defenses.
        """
        defense_types = ['committee', 'adaptive', 'reputation', 'gradient', 'ensemble']
        defense_colors = {
            'committee': '#4169E1',
            'adaptive': '#FF8C00',
            'reputation': '#9370DB',
            'gradient': '#20B2AA',
            'ensemble': '#DC143C'
        }

        for dataset_name, dataset_results in results.items():
            attacks = [atk for atk in dataset_results.keys() if atk.lower() != 'none']
            if not attacks:
                continue

            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(f'{dataset_name} - Training Convergence (All Attacks)',
                        fontsize=18, fontweight='bold', y=0.995)
            axes = axes.flatten()

            for idx, attack_name in enumerate(attacks):
                if idx >= 6:
                    break
                ax = axes[idx]

                # Baseline
                base_hist = self._get_scenario_data(dataset_results, 'none', 'baseline', 'training_acc_history')
                if base_hist:
                    ax.plot(range(1, len(base_hist)+1), base_hist, label='Baseline',
                           color='#2E8B57', linewidth=2, marker='o', markersize=3)

                # Attack
                att_hist = self._get_scenario_data(dataset_results, attack_name, 'attack', 'training_acc_history')
                if att_hist:
                    color = ATTACK_COLORS.get(attack_name.lower(), '#DC143C')
                    ax.plot(range(1, len(att_hist)+1), att_hist, label=f'{attack_name.upper()}',
                           color=color, linewidth=2, marker='s', markersize=3)

                # Defenses
                for def_type in defense_types:
                    def_hist = self._get_scenario_data(dataset_results, attack_name, def_type, 'training_acc_history')
                    if def_hist:
                        color = defense_colors.get(def_type, '#666666')
                        ax.plot(range(1, len(def_hist)+1), def_hist, label=f'{def_type.capitalize()}',
                               color=color, linewidth=2, marker='x', markersize=3, alpha=0.8)

                ax.set_title(f'{attack_name.upper()} Attack', fontweight='bold', fontsize=12)
                ax.set_xlabel('Communication Rounds', fontweight='bold')
                ax.set_ylabel('Global Model Accuracy', fontweight='bold')
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8, loc='best')

            for idx in range(len(attacks), 6):
                axes[idx].axis('off')

            plt.tight_layout()
            self._save_plot(fig, f'{dataset_name.lower()}_training_convergence.png')


    # =====================================================================
    # DEFENSE EFFECTIVENESS PLOTS
    # =====================================================================

    def plot_defense_comparison(self, summary, dataset_name, run_id):
        """
        Generate separate bar charts showing committee defense effectiveness:
        1 Accuracy comparison (Clean, Attack, Defense)
        2 Recovery rate (%)
        """
        try:
            attacks = []
            clean_accs = []
            attack_accs = []
            defense_accs = []
            recovery_rates = []

            # Collect data for committee defense across attacks
            for attack_name, defenses in summary.items():
                if not defenses or 'committee' not in defenses:
                    continue

                metrics = defenses['committee']
                clean_acc = metrics.get("clean_accuracy", 0.0) or 0.0
                attack_acc = metrics.get("attack_accuracy", 0.0) or 0.0
                defense_acc = metrics.get("defense_accuracy", 0.0) or 0.0
                recovery = metrics.get("defense_recovery_rate", 0.0) or 0.0

                attacks.append(attack_name.upper())
                clean_accs.append(clean_acc)
                attack_accs.append(attack_acc)
                defense_accs.append(defense_acc)
                recovery_rates.append(recovery)

            if not attacks:
                print("[WARN] No committee defense data to plot.")
                return

            x = np.arange(len(attacks))
            width = 0.25

            # ---------------------------------------------------------------------
            # Plot 1: Accuracy Comparison (Clean vs Attack vs Defense)
            # ---------------------------------------------------------------------
            fig1, ax1 = plt.subplots(figsize=(9, 6))

            bars1 = ax1.bar(x - width, clean_accs, width, label='Clean', 
                            color='#2E8B57', alpha=0.85, edgecolor='black')
            bars2 = ax1.bar(x, attack_accs, width, label='Under Attack', 
                            color='#DC143C', alpha=0.85, edgecolor='black')
            bars3 = ax1.bar(x + width, defense_accs, width, label='Committee Defense', 
                            color='#4169E1', alpha=0.85, edgecolor='black')

            ax1.set_xlabel("Attack Types", fontweight='bold')
            ax1.set_ylabel("Accuracy", fontweight='bold')
            ax1.set_title(f"{dataset_name} - Committee Defense Accuracy", fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(attacks, rotation=45, ha='right')
            ax1.set_ylim(0, 1.0)
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.legend()

            # Value labels
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    h = bar.get_height()
                    if h > 0:
                        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                                f"{h:.3f}", ha='center', va='bottom', fontsize=8)

            plt.tight_layout()
            acc_filename = f"{dataset_name.lower()}_committee_defense_accuracy_{run_id}.png"
            self._save_plot(fig1, acc_filename)
            print(f"[INFO] Accuracy comparison plot saved: {acc_filename}")

            # ---------------------------------------------------------------------
            # Plot 2: Recovery Rate (%)
            # ---------------------------------------------------------------------
            fig2, ax2 = plt.subplots(figsize=(9, 6))
            colors = [ATTACK_COLORS.get(a.lower(), '#666666') for a in attacks]

            bars = ax2.bar(attacks, recovery_rates, color=colors, alpha=0.85, edgecolor='black')

            ax2.set_xlabel("Attack Types", fontweight='bold')
            ax2.set_ylabel("Recovery Rate (%)", fontweight='bold')
            ax2.set_title(f"{dataset_name} - Committee Defense Recovery", fontweight='bold')
            ax2.set_ylim(0, 110)
            ax2.grid(True, alpha=0.3, axis='y')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax2.text(bar.get_x() + bar.get_width() / 2, h + 2,
                            f"{h:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

            plt.tight_layout()
            rec_filename = f"{dataset_name.lower()}_committee_defense_recovery_{run_id}.png"
            self._save_plot(fig2, rec_filename)
            print(f"[INFO] Recovery rate plot saved: {rec_filename}")

        except Exception as e:
            print(f"[WARN] Failed to generate defense comparison plots: {e}")
            import traceback
            traceback.print_exc()


    def plot_defense_effectiveness(self, results):
        """Plot Defense Recovery Rate (%) for each attack type across all datasets and all defenses."""
        attack_list = ['slf', 'dlf', 'centralized', 'coordinated', 'random', 'model_dependent']
        defense_types = ['committee', 'adaptive', 'reputation', 'gradient', 'ensemble']
        defense_colors = {
            'committee': '#4169E1',
            'adaptive': '#FF8C00',
            'reputation': '#9370DB',
            'gradient': '#20B2AA',
            'ensemble': '#DC143C'
        }

        for dataset_name, dataset_results in results.items():
            fig, ax = plt.subplots(figsize=(12, 7))
            width = 0.13
            x = np.arange(len(attack_list))
            bars = []
            if not dataset_results:
                print(f"[WARNING] No results found for dataset: {dataset_name}")
                plt.close(fig)
                continue

            for i, def_type in enumerate(defense_types):
                recovery_rates = []
                for attack_name in attack_list:
                    if attack_name not in dataset_results:
                        recovery_rates.append(0)
                        continue
                    # Use helper methods to get accuracies
                    baseline_acc = self._get_final_accuracy(dataset_results, attack_name, 'baseline')
                    attack_acc = self._get_final_accuracy(dataset_results, attack_name, 'attack')
                    defense_acc = self._get_final_accuracy(dataset_results, attack_name, def_type)

                    # Calculate recovery rate using helper method
                    recovery = self._calculate_recovery_rate(baseline_acc, attack_acc, defense_acc)
                    recovery_rates.append(recovery)
                bar = ax.bar(x + (i - 2) * width, recovery_rates, width, label=def_type.capitalize(), color=defense_colors[def_type], alpha=0.85, edgecolor='black')
                bars.append(bar)

            ax.set_xlabel('Attack Types', fontweight='bold')
            ax.set_ylabel('Defense Recovery Rate (%)', fontweight='bold')
            ax.set_title(f'{dataset_name} - Defense Recovery Rate (All Defenses)', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([a.upper() for a in attack_list], rotation=30, ha='right')
            ax.set_ylim(0, 110)
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            ax.legend()

            # Annotate bar values
            for bar_group in bars:
                for bar in bar_group:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 2, f'{height:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

            plt.tight_layout()
            self._save_plot(fig, f'{dataset_name.lower()}_defenses_recovery_rate.png')

    # =====================================================================
    # SPECIALIZED PLOTS
    # =====================================================================
    
    def plot_all_defense_performance(self, results):
        """
        Plot defense performance metrics across attacks - SEPARATE plot for EACH dataset.
        Shows all 5 defenses: committee, adaptive, reputation, gradient, ensemble.
        """
        defense_types = ['committee', 'adaptive', 'reputation', 'gradient', 'ensemble']
        defense_colors = {
            'committee': '#4169E1',
            'adaptive': '#FF8C00',
            'reputation': '#9370DB',
            'gradient': '#20B2AA',
            'ensemble': '#DC143C'
        }

        # Create separate plot for each dataset
        for dataset_name, dataset_results in results.items():
            all_attacks = []
            baseline_accs = []
            attack_accs = []
            defense_accs = {def_type: [] for def_type in defense_types}

            for attack_name, attack_results in dataset_results.items():
                if not isinstance(attack_name, str) or attack_name.lower() == 'none':
                    continue

                # Use helper method to get final accuracies
                baseline_acc = self._get_final_accuracy(dataset_results, attack_name, 'baseline')
                attack_acc = self._get_final_accuracy(dataset_results, attack_name, 'attack')

                # Skip if no attack data
                if not baseline_acc or not attack_acc:
                    continue

                all_attacks.append(attack_name.upper())
                baseline_accs.append(baseline_acc)
                attack_accs.append(attack_acc)

                # Get all defense accuracies using helper method
                for def_type in defense_types:
                    def_acc = self._get_final_accuracy(dataset_results, attack_name, def_type)
                    defense_accs[def_type].append(def_acc)

            if not all_attacks:
                print(f"[WARNING] No defense performance data found for {dataset_name}")
                continue

            # Calculate number of bars and width
            num_defenses = len(defense_types)
            num_bars = num_defenses + 2  # baseline + attack + all defenses
            x = np.arange(len(all_attacks))
            width = 0.8 / num_bars  # Total width of 0.8 divided by number of bars

            fig, ax = plt.subplots(figsize=(16, 7))

            # Calculate bar positions
            offset = -0.4 + width/2  # Start from left edge

            # Plot baseline
            bars1 = ax.bar(x + offset, baseline_accs, width, label='Baseline',
                          color='#2E8B57', alpha=0.85, edgecolor='black', linewidth=0.5)
            offset += width

            # Plot attack
            bars2 = ax.bar(x + offset, attack_accs, width, label='Under Attack',
                          color='#8B0000', alpha=0.85, edgecolor='black', linewidth=0.5)
            offset += width

            # Plot all defenses
            defense_bars = []
            for def_type in defense_types:
                bars = ax.bar(x + offset, defense_accs[def_type], width,
                             label=def_type.capitalize(),
                             color=defense_colors[def_type], alpha=0.85,
                             edgecolor='black', linewidth=0.5)
                defense_bars.append(bars)
                offset += width

            ax.set_xlabel('Attack Type', fontweight='bold', fontsize=12)
            ax.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
            ax.set_title(f'{dataset_name} - All Defenses Performance Comparison',
                        fontweight='bold', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(all_attacks, rotation=45, ha="right", fontsize=10)
            ax.set_ylim(0, 1.0)
            ax.legend(loc='upper right', ncol=2, fontsize=9)
            ax.grid(axis='y', linestyle='--', alpha=0.4)

            # Add bar value annotations (only for bars > 0.1 to avoid clutter)
            all_bars = [bars1, bars2] + defense_bars
            for bars in all_bars:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0.1:  # Only show labels for significant values
                        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                               f'{height:.2f}', ha='center', va='bottom', fontsize=6, rotation=90)

            plt.tight_layout()
            safe_dataset_name = dataset_name.replace(' ', '_').replace('-', '_').lower()
            self._save_plot(fig, f'{safe_dataset_name}_all_defenses_performance.png')


    def plot_attack_sophistication_radar(self):
        """Radar chart showing attack sophistication characteristics"""
        attack_characteristics = {
            'slf': {'complexity': 1, 'stealth': 2, 'effectiveness': 3, 
                   'label': 'Static Label Flipping'},
            'dlf': {'complexity': 4, 'stealth': 3, 'effectiveness': 4, 
                   'label': 'Dynamic Label Flipping'},
            'centralized': {'complexity': 3, 'stealth': 4, 'effectiveness': 5, 
                           'label': 'Centralized Trigger'},
            'coordinated': {'complexity': 5, 'stealth': 5, 'effectiveness': 5, 
                           'label': 'Coordinated Trigger'},
            'random': {'complexity': 2, 'stealth': 4, 'effectiveness': 3, 
                      'label': 'Random Trigger'},
            'model_dependent': {'complexity': 5, 'stealth': 5, 'effectiveness': 5, 
                               'label': 'Model-Dependent'}
        }
        
        categories = ['Complexity', 'Stealth', 'Effectiveness']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        colors = ['#FF6B6B', '#4ECDC4', "#1F06AD", '#FFD700', "#4B7907", '#FF69B4']
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        for i, (attack_name, chars) in enumerate(attack_characteristics.items()):
            values = [chars['complexity'], chars['stealth'], chars['effectiveness']]
            values += [values[0]]
            ax.plot(angles, values, 'o-', linewidth=2, label=chars['label'], color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 5)
        ax.set_title('Attack Sophistication Analysis', fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        self._save_plot(fig, 'attack_sophistication_radar.png')

    def plot_attack_comparison(self, results):
        """Plot attack accuracy comparison for each dataset, including all defenses."""
        defense_types = ['committee', 'adaptive', 'reputation', 'gradient', 'ensemble']
        defense_colors = {
            'committee': '#4169E1',
            'adaptive': '#FF8C00',
            'reputation': '#9370DB',
            'gradient': '#20B2AA',
            'ensemble': '#DC143C'
        }
        for dataset_name, dataset_results in results.items():
            for attack_name in dataset_results.keys():
                if attack_name.lower() == 'none':
                    continue
            fig, ax = plt.subplots(figsize=(12, 7))
            attacks = []
            baselines = []
            attacks_vals = []
            defense_accs = {def_type: [] for def_type in defense_types}
            for attack_name in dataset_results.keys():
                if attack_name.lower() == 'none':
                    continue
                # Use helper method to get accuracies
                baseline_acc = self._get_final_accuracy(dataset_results, attack_name, 'baseline')
                attack_acc = self._get_final_accuracy(dataset_results, attack_name, 'attack')
                attacks.append(attack_name.upper())
                baselines.append(baseline_acc)
                attacks_vals.append(attack_acc)
                for def_type in defense_types:
                    defense_acc = self._get_final_accuracy(dataset_results, attack_name, def_type)
                    defense_accs[def_type].append(defense_acc)
            x = np.arange(len(attacks))
            width = 0.8 / (2 + len(defense_types))
            offset = -0.4 + width/2
            bars1 = ax.bar(x + offset, baselines, width, label='Baseline', color='#2E8B57', alpha=0.85, edgecolor='black')
            offset += width
            bars2 = ax.bar(x + offset, attacks_vals, width, label='Under Attack', color='#DC143C', alpha=0.85, edgecolor='black')
            offset += width
            defense_bars = []
            for def_type in defense_types:
                bars = ax.bar(x + offset, defense_accs[def_type], width, label=def_type.capitalize(), color=defense_colors[def_type], alpha=0.85, edgecolor='black')
                defense_bars.append(bars)
                offset += width
            ax.set_xlabel('Attack Types', fontweight='bold')
            ax.set_ylabel('Accuracy', fontweight='bold')
            ax.set_title(f'{dataset_name} - Model Accuracy Comparison (All Defenses)', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(attacks, rotation=45, ha='right')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            # Add value labels
            all_bars = [bars1, bars2] + defense_bars
            for bars in all_bars:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            plt.tight_layout()
            self._save_plot(fig, f'{dataset_name.lower()}_attack_accuracy_comparison.png')    

    def plot_attack_evaluation(self, results):
        """Bar chart of attack evaluation results with all defenses."""
        defense_types = ['committee', 'adaptive', 'reputation', 'gradient', 'ensemble']
        defense_colors = {
            'committee': '#4169E1',
            'adaptive': '#FF8C00',
            'reputation': '#9370DB',
            'gradient': '#20B2AA',
            'ensemble': '#DC143C'
        }
        for dataset_name, dataset_results in results.items():
            attacks = []
            baselines = []
            attacks_vals = []
            defense_accs = {def_type: [] for def_type in defense_types}
            for attack_name in dataset_results.keys():
                if attack_name.lower() == 'none':
                    continue
                # Use helper method to get accuracies
                baseline_acc = self._get_final_accuracy(dataset_results, attack_name, 'baseline')
                attack_acc = self._get_final_accuracy(dataset_results, attack_name, 'attack')
                attacks.append(attack_name.upper())
                baselines.append(baseline_acc)
                attacks_vals.append(attack_acc)
                for def_type in defense_types:
                    defense_acc = self._get_final_accuracy(dataset_results, attack_name, def_type)
                    defense_accs[def_type].append(defense_acc)
            x = np.arange(len(attacks))
            width = 0.8 / (2 + len(defense_types))
            offset = -0.4 + width/2
            fig, ax = plt.subplots(figsize=(12, 7))
            bars1 = ax.bar(x + offset, baselines, width, label='Baseline', color='#2E8B57', alpha=0.85, edgecolor='black')
            offset += width
            bars2 = ax.bar(x + offset, attacks_vals, width, label='Under Attack', color='#DC143C', alpha=0.85, edgecolor='black')
            offset += width
            defense_bars = []
            for def_type in defense_types:
                bars = ax.bar(x + offset, defense_accs[def_type], width, label=def_type.capitalize(), color=defense_colors[def_type], alpha=0.85, edgecolor='black')
                defense_bars.append(bars)
                offset += width
            ax.set_xlabel('Attack Types', fontweight='bold')
            ax.set_ylabel('Accuracy', fontweight='bold')
            ax.set_title(f'{dataset_name} - Attack Evaluation (All Defenses)', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(attacks, rotation=45, ha='right')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            # Add value labels
            all_bars = [bars1, bars2] + defense_bars
            for bars in all_bars:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            plt.tight_layout()
            self._save_plot(fig, f'{dataset_name.lower()}_attack_evaluation.png')

    def plot_attack_success_rate(self, results):
        """Bar chart showing attack success rates"""
        for dataset_name, dataset_results in results.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            attacks_success = []
            success_rates = []
            
            for attack_name in dataset_results.keys():
                if attack_name.lower() != 'none':
                    attack_success = dataset_results[attack_name].get('attack', {}).get('attack_success_rates', [])
                    if attack_success:
                        attacks_success.append(attack_name.upper())
                        success_rates.append(np.mean(attack_success))
            
            if attacks_success and success_rates:
                colors = [ATTACK_COLORS.get(a.lower(), '#DC143C') for a in attacks_success]
                bars = ax.bar(attacks_success, success_rates, alpha=0.8, color=colors, edgecolor='black')
                
                ax.set_xlabel('Attack Types', fontweight='bold')
                ax.set_ylabel('Attack Success Rate', fontweight='bold')
                ax.set_title(f'{dataset_name} - Attack Success Rate', fontweight='bold')
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3, axis='y')
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', 
                           fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            self._save_plot(fig, f'{dataset_name.lower()}_attack_success_rate.png')
            plt.close(fig)

    # =====================================================================
    # UTILITY METHODS
    # =====================================================================
    
    
    def generate_all_plots(self, results_df, output_dir, dataset_name):
        """
        Generate all available plots for the given results DataFrame.
        Now supports all five defenses and is compatible with new results structure.
        """
        # Set output directory for saving plots
        self.out_dir = output_dir
        print("\n" + "="*80)
        print(f"GENERATING COMPREHENSIVE VISUALIZATIONS for {dataset_name}")
        print("="*80 + "\n")

        # If results_df is a DataFrame, convert to dict for legacy plotting functions
        if hasattr(results_df, 'to_dict'):
            results = results_df.to_dict()
        else:
            results = results_df

        # Only plot for the given dataset_name
        available_datasets = [dataset_name]
        print(f"[INFO] Plotting for dataset: {dataset_name}")

        for ds in available_datasets:
            dataset_results = results.get(ds, results)
            print(f"\n[PLOTTING] Processing {ds}...")
            try:
                # Consolidated plots: One plot per metric with all attacks as subplots
                print("  - Training Accuracy vs Round (All Attacks)")
                self.plot_training_accuracy_vs_round({ds: dataset_results})
                print("  - Training Loss vs Round (All Attacks)")
                self.plot_training_loss_vs_round({ds: dataset_results})
                print("  - Testing Accuracy vs Round (All Attacks)")
                self.plot_testing_accuracy_vs_round({ds: dataset_results})
                print("  - Testing Loss vs Round (All Attacks)")
                self.plot_testing_loss_vs_round({ds: dataset_results})
                print("  - Train vs Test Accuracy (All Attacks)")
                self.plot_train_vs_test_accuracy({ds: dataset_results})
                print("  - Train vs Test Loss (All Attacks)")
                self.plot_train_vs_test_loss({ds: dataset_results})
                print("  - Training Convergence (All Attacks)")
                self.plot_training_convergence({ds: dataset_results})
            except Exception as e:
                print(f"  [ERROR] Failed to generate plots for {ds}: {e}")
                import traceback
                traceback.print_exc()

        # Comprehensive Defense Comparison (includes ALL defense plots - NO DUPLICATES)
        print(f"\n[PLOTTING] Comprehensive Defense Analysis...")
        try:
            print("  - All Defense Comparisons (Accuracy, Recovery, ASR, Heatmaps)")
            self.plot_comprehensive_defense_comparison(results)
        except Exception as e:
            print(f"  [ERROR] Comprehensive defense comparison: {e}")
            import traceback
            traceback.print_exc()

        # Attack sophistication radar (UNIQUE plot)
        try:
            print("  - Attack Sophistication Radar")
            self.plot_attack_sophistication_radar()
        except Exception as e:
            print(f"  [ERROR] Attack sophistication radar: {e}")

        print(f"\n{'='*80}")
        print(f"✓ COMPLETED: Generated {len(self.plots_generated)} plots")
        print(f"✓ Saved to: {self.out_dir}")
        print(f"{'='*80}\n")
        return self.plots_generated

    # =====================================================================
    # COMPREHENSIVE DEFENSE COMPARISON PLOTS
    # =====================================================================

    def plot_comprehensive_defense_comparison(self, results):
        """
        Generate comprehensive defense comparison plots for all 5 defenses.

        Safely handles missing 'attack', 'baseline', or defense entries
        to avoid KeyErrors, while maintaining full visual structure.
        """
        defense_colors = {
            'committee': '#2E86AB',      # Blue
            'adaptive': '#A23B72',       # Purple
            'reputation': '#F77F00',     # Orange
            'gradient': '#06A77D',       # Green
            'ensemble': '#D62828'        # Red
        }

        print(f"\n[PLOTTING] Generating comprehensive defense comparison plots...")

        # ✅ Step 1: Filter out datasets/attacks without 'attack' data
        filtered_results = {}
        for dataset_name, dataset_results in results.items():
            valid_attacks = {
                atk: data for atk, data in dataset_results.items()
                if isinstance(data, dict) and "attack" in data
            }
            if valid_attacks:
                filtered_results[dataset_name] = valid_attacks
            else:
                print(f"[SKIP] Dataset '{dataset_name}' has no valid 'attack' entries (baseline only).")

        if not filtered_results:
            print("[WARNING] No datasets with valid 'attack' entries. Skipping plots.")
            return

        results = filtered_results
        all_defenses = ['committee', 'adaptive', 'reputation', 'gradient', 'ensemble']

        # ✅ Plot 1: Defense Comparison by Dataset (Accuracy)
        for dataset_name, dataset_results in results.items():
            fig, ax = plt.subplots(figsize=(16, 8))
            attacks = list(dataset_results.keys())
            x = np.arange(len(attacks))
            width = 0.11

            # Use helper method to get accuracies
            baseline_accs = [self._get_final_accuracy(dataset_results, atk, 'baseline') for atk in attacks]
            attack_accs = [self._get_final_accuracy(dataset_results, atk, 'attack') for atk in attacks]

            ax.bar(x - 3*width, baseline_accs, width, label='Baseline', color='#2ecc71', alpha=0.8)
            ax.bar(x - 2*width, attack_accs, width, label='Attack', color='#e74c3c', alpha=0.8)

            for idx, defense in enumerate(all_defenses):
                defense_accs = [self._get_final_accuracy(dataset_results, atk, defense) for atk in attacks]
                ax.bar(x + (idx - 1)*width, defense_accs, width,
                    label=f'{defense.capitalize()} Defense',
                    color=defense_colors[defense], alpha=0.8)

            ax.set_xlabel('Attack Type', fontsize=12, fontweight='bold')
            ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
            ax.set_title(f'All 5 Defenses Comparison - {dataset_name}', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([atk.upper() for atk in attacks], rotation=45, ha='right')
            ax.legend(loc='upper right', fontsize=9, ncol=2)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1.0])
            plt.tight_layout()
            # Save with consistent naming
            self._save_plot(fig, f"{dataset_name.lower()}_all_defenses_comparison.png")

        # ✅ Plot 2: Recovery Rates by Dataset
        for dataset_name, dataset_results in results.items():
            fig, ax = plt.subplots(figsize=(16, 8))
            attacks = list(dataset_results.keys())
            x = np.arange(len(attacks))
            width = 0.15

            for idx, defense in enumerate(all_defenses):
                recovery_rates = []
                for atk in attacks:
                    # Calculate recovery rate from actual accuracies
                    baseline_acc = self._get_final_accuracy(dataset_results, atk, 'baseline')
                    attack_acc = self._get_final_accuracy(dataset_results, atk, 'attack')
                    defense_acc = self._get_final_accuracy(dataset_results, atk, defense)
                    recovery_rate = self._calculate_recovery_rate(baseline_acc, attack_acc, defense_acc)
                    recovery_rates.append(recovery_rate)
                ax.bar(x + (idx - 2)*width, recovery_rates, width,
                    label=f'{defense.capitalize()} Defense',
                    color=defense_colors[defense], alpha=0.8)

            ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Full Recovery')
            ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50% Recovery')

            ax.set_xlabel('Attack Type', fontsize=12, fontweight='bold')
            ax.set_ylabel('Recovery Rate (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'All 5 Defenses Recovery Rates - {dataset_name}', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([atk.upper() for atk in attacks], rotation=45, ha='right')
            ax.legend(loc='upper right', fontsize=9, ncol=2)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 110])
            plt.tight_layout()
            # Save with consistent naming
            self._save_plot(fig, f"{dataset_name.lower()}_recovery_rates.png")

        # ✅ Plot 3: Cross-Dataset Comparison (one subplot per defense)
        # First, get the union of all attacks across all datasets
        all_attacks = set()
        for dataset_results in results.values():
            all_attacks.update(dataset_results.keys())
        all_attacks = sorted([atk for atk in all_attacks if atk.lower() != 'none'])

        if not all_attacks:
            print("[WARNING] No attacks found for cross-dataset comparison")
        else:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()

            for idx, defense in enumerate(all_defenses):
                ax = axes[idx]
                for dataset_name in results.keys():
                    dataset_results = results[dataset_name]
                    # Calculate recovery rates for ALL attacks (use 0 if attack not in this dataset)
                    recovery_rates = []
                    for atk in all_attacks:
                        if atk in dataset_results:
                            baseline_acc = self._get_final_accuracy(dataset_results, atk, 'baseline')
                            attack_acc = self._get_final_accuracy(dataset_results, atk, 'attack')
                            defense_acc = self._get_final_accuracy(dataset_results, atk, defense)
                            recovery_rate = self._calculate_recovery_rate(baseline_acc, attack_acc, defense_acc)
                            recovery_rates.append(recovery_rate)
                        else:
                            # Attack doesn't exist for this dataset, use None to skip
                            recovery_rates.append(None)

                    x = np.arange(len(all_attacks))
                    # Filter out None values for plotting
                    x_valid = [x[i] for i in range(len(recovery_rates)) if recovery_rates[i] is not None]
                    y_valid = [r for r in recovery_rates if r is not None]

                    if y_valid:  # Only plot if there's data
                        ax.plot(x_valid, y_valid, marker='o', linewidth=2, markersize=8,
                                label=dataset_name, alpha=0.8)

                ax.axhline(y=100, color='green', linestyle='--', alpha=0.3)
                ax.set_xlabel('Attack Type', fontsize=11, fontweight='bold')
                ax.set_ylabel('Recovery Rate (%)', fontsize=11, fontweight='bold')
                ax.set_title(f'{defense.capitalize()} Defense', fontsize=12, fontweight='bold')
                ax.set_xticks(np.arange(len(all_attacks)))
                ax.set_xticklabels([atk.upper() for atk in all_attacks], rotation=45, ha='right', fontsize=9)
                ax.legend(fontsize=9)
                ax.grid(alpha=0.3)
                ax.set_ylim([0, 110])

            axes[5].axis('off')
            plt.suptitle('Cross-Dataset Defense Comparison (All 5 Defenses)', fontsize=14, fontweight='bold', y=1.00)
            plt.tight_layout()
            # Save with consistent naming
            self._save_plot(fig, f"cross_dataset_defense_comparison.png")

        # ✅ Plot 4: Attack Success Rate Comparison
        for dataset_name, dataset_results in results.items():
            fig, ax = plt.subplots(figsize=(16, 8))
            # Exclude 'none' from attacks - it's baseline with no attack!
            attacks = [atk for atk in dataset_results.keys() if atk.lower() != 'none']

            if not attacks:
                print(f"[WARNING] No attacks found for ASR plot in {dataset_name}")
                plt.close(fig)
                continue

            x = np.arange(len(attacks))
            width = 0.15

            # Calculate attack success rate without defense
            attack_asr = []
            for atk in attacks:
                baseline_acc = self._get_final_accuracy(dataset_results, atk, 'baseline')
                attack_acc = self._get_final_accuracy(dataset_results, atk, 'attack')
                asr = self._calculate_attack_success_rate(baseline_acc, attack_acc)
                attack_asr.append(asr)
            ax.plot(x, attack_asr, 'ro-', linewidth=2, markersize=8, label='No Defense', alpha=0.7)

            # Calculate attack success rate with each defense
            for idx, defense in enumerate(all_defenses):
                asr_rates = []
                for atk in attacks:
                    baseline_acc = self._get_final_accuracy(dataset_results, atk, 'baseline')
                    defense_acc = self._get_final_accuracy(dataset_results, atk, defense)
                    # Attack success rate with defense = degradation from baseline even with defense
                    asr = self._calculate_attack_success_rate(baseline_acc, defense_acc)
                    asr_rates.append(asr)
                ax.bar(x + (idx - 2)*width, asr_rates, width,
                    label=f'{defense.capitalize()} Defense',
                    color=defense_colors[defense], alpha=0.8)

            ax.set_xlabel('Attack Type', fontsize=12, fontweight='bold')
            ax.set_ylabel('Attack Success Rate', fontsize=12, fontweight='bold')
            ax.set_title(f'Attack Success Rate with All 5 Defenses - {dataset_name}', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([atk.upper() for atk in attacks], rotation=45, ha='right')
            ax.legend(loc='upper right', fontsize=9, ncol=2)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1.0])
            plt.tight_layout()
            # Save with consistent naming
            self._save_plot(fig, f"{dataset_name.lower()}_attack_success_rates.png")

        # ✅ Plot 5: Defense Accuracy Heatmap
        for dataset_name, dataset_results in results.items():
            # Exclude 'none' from attacks - only show defense performance against actual attacks
            attacks = [atk for atk in dataset_results.keys() if atk.lower() != 'none']

            if not attacks:
                print(f"[WARNING] No attacks found for heatmap in {dataset_name}")
                continue

            n_attacks = len(attacks)
            accuracy_matrix = np.zeros((len(all_defenses), n_attacks))

            for i, defense in enumerate(all_defenses):
                for j, attack in enumerate(attacks):
                    # Get final accuracy using helper method
                    accuracy_matrix[i, j] = self._get_final_accuracy(dataset_results, attack, defense)

            fig, ax = plt.subplots(figsize=(14, 8))
            im = ax.imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            ax.set_xticks(np.arange(n_attacks))
            ax.set_yticks(np.arange(len(all_defenses)))
            ax.set_xticklabels([atk.upper() for atk in attacks], rotation=45, ha='right')
            ax.set_yticklabels([d.capitalize() for d in all_defenses])
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Accuracy', rotation=270, labelpad=20, fontweight='bold')

            for i in range(len(all_defenses)):
                for j in range(n_attacks):
                    ax.text(j, i, f'{accuracy_matrix[i, j]:.3f}', ha="center", va="center",
                            color="black", fontsize=9)

            ax.set_title(f'Defense Accuracy Heatmap - {dataset_name}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Attack Type', fontsize=12, fontweight='bold')
            ax.set_ylabel('Defense Type', fontsize=12, fontweight='bold')
            plt.tight_layout()
            # Save with consistent naming
            self._save_plot(fig, f"{dataset_name.lower()}_defense_accuracy_heatmap.png")

        print(f"  ✓ Generated comprehensive defense comparison plots (including heatmaps and ASR plots)")


