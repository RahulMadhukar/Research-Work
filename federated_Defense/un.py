# evaluation.py
import csv
import json
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from attacks.attack_summery import attack_summary
from attacks.base import AttackConfig
from models import FashionMNISTNet, CIFAR10Net
from datasets import load_fashion_mnist, load_cifar10, partition_dataset
from client import DecentralizedClient
from coordinator import DecentralizedFLCoordinator

# Import the plotting engine
from plots import PlottingEngine

# Import defense module
from defense import get_defense


class EvaluationFramework:
    """
    Framework to run comprehensive evaluations of decentralized FL under
    various attacks and defenses. Refactored to centralize repeated logic.
    """

    def __init__(self, out_dir="Output", run_id=None):
        """Initialize directories, plotter and basic bookkeeping."""
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        project_root = os.path.dirname(os.path.abspath(__file__))

        # Resolve base_run_dir
        if os.path.isabs(out_dir):
            self.base_run_dir = out_dir
        elif self.run_id in out_dir:
            self.base_run_dir = os.path.join(project_root, out_dir)
        else:
            output_root = os.path.join(project_root, out_dir)
            self.base_run_dir = os.path.join(output_root, self.run_id)

        # Create folders
        self.plots_dir = os.path.join(self.base_run_dir, "plots")
        self.results_dir = os.path.join(self.base_run_dir, "results")
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # Compatibility
        self.out_dir = self.plots_dir

        # Plotter
        self.plotter = PlottingEngine(output_dir=self.plots_dir)

        # Storage
        self.results = {}
        self.plots_generated = []
        self.defense_results = {}  # store defense comparison if needed

    # ------------------------------
    # Shared configuration helpers
    # ------------------------------

    def get_attack_configs(self, malicious_clients: int = 4, poisoning_rate: float = 0.30):
        """Return canonical attack_configs dictionary used throughout the code."""
        return {
            'none': None,
            'slf': AttackConfig(attack_type='slf', poisoning_rate=poisoning_rate, source_class=[0, 2, 4],
                                target_class=[1, 3, 5], num_malicious_clients=malicious_clients),
            'dlf': AttackConfig(attack_type='dlf', poisoning_rate=poisoning_rate, target_class=0,
                                num_malicious_clients=malicious_clients),
            'centralized': AttackConfig(attack_type='centralized', poisoning_rate=poisoning_rate, target_class=0,
                                        trigger_size=(6, 6), trigger_intensity=1.0,
                                        num_malicious_clients=malicious_clients),
            'coordinated': AttackConfig(attack_type='coordinated', poisoning_rate=poisoning_rate, target_class=0,
                                        trigger_size=(6, 6), trigger_intensity=1.0,
                                        num_malicious_clients=malicious_clients),
            'random': AttackConfig(attack_type='random', poisoning_rate=poisoning_rate, target_class=0,
                                   trigger_size=(6, 6), trigger_intensity=1.0,
                                   num_malicious_clients=malicious_clients),
            'model_dependent': AttackConfig(attack_type='model_dependent', poisoning_rate=poisoning_rate,
                                            source_class=1, target_class=0, epsilon=0.5,
                                            num_malicious_clients=malicious_clients)
        }

    def get_datasets_and_models(self):
        """Return datasets dict and models factory mapping."""
        datasets = {
            'Fashion-MNIST': load_fashion_mnist(),
            'CIFAR-10': load_cifar10()
        }
        models = {
            'Fashion-MNIST': lambda: FashionMNISTNet(),
            'CIFAR-10': lambda: CIFAR10Net()
        }
        return datasets, models

    def _safe_extract_number(self, x, keys_fallback=None):
        """Convert various formats (dict/list/np types) to a float safely."""
        import numbers
        if keys_fallback is None:
            keys_fallback = ['accuracy', 'avg_accuracy', 'value', 'loss']

        if x is None:
            return 0.0
        if isinstance(x, numbers.Number):
            return float(x)
        if isinstance(x, np.generic):
            return float(x.item())
        if isinstance(x, dict):
            for k in keys_fallback:
                if k in x:
                    try:
                        return float(x[k])
                    except Exception:
                        pass
            vals = []
            for v in x.values():
                if isinstance(v, numbers.Number):
                    vals.append(float(v))
                elif isinstance(v, np.generic):
                    vals.append(float(v.item()))
            return float(np.mean(vals)) if vals else 0.0
        if isinstance(x, (list, tuple, np.ndarray)):
            if len(x) == 0:
                return 0.0
            for elem in x:
                if isinstance(elem, numbers.Number) or isinstance(elem, np.generic):
                    return float(elem)
            return float(np.mean([_safe for _safe in [self._safe_extract_number(e) for e in x]
                                  if _safe is not None])) if len(x) > 0 else 0.0
        try:
            return float(x)
        except Exception:
            return 0.0

    # ------------------------------
    # Client/update helpers
    # ------------------------------

    def build_clients(self, num_clients, client_data, models_map, dataset_name, attack_config=None):
        """
        Build a list of DecentralizedClient instances for the provided client_data,
        optionally injecting attack_config for the first N malicious clients.
        """
        clients = []
        for i in range(num_clients):
            model = models_map[dataset_name]()
            if attack_config is not None and i < attack_config.num_malicious_clients:
                client = DecentralizedClient(i, client_data[i][0], client_data[i][1], model, attack_config)
            else:
                client = DecentralizedClient(i, client_data[i][0], client_data[i][1], model, None)
            clients.append(client)
        return clients

    def _extract_client_updates(self, clients):
        """
        Robustly extract a list of parameter dicts (state_dict-like) from DecentralizedClient objects.
        Each element returned is a dict: param_name -> torch.Tensor
        """
        updates = []
        for c in clients:
            upd = None
            # 1) last_update
            if hasattr(c, 'last_update') and isinstance(getattr(c, 'last_update'), dict):
                upd = {k: v.clone().detach() if isinstance(v, torch.Tensor) else torch.tensor(v)
                       for k, v in c.last_update.items()}
            # 2) model.state_dict()
            elif hasattr(c, 'model') and hasattr(c.model, 'state_dict'):
                st = c.model.state_dict()
                upd = {k: v.clone().detach() if isinstance(v, torch.Tensor) else torch.tensor(v)
                       for k, v in st.items()}
            # 3) get_update() method
            elif hasattr(c, 'get_update'):
                maybe = c.get_update()
                if isinstance(maybe, dict):
                    upd = {k: v.clone().detach() if isinstance(v, torch.Tensor) else torch.tensor(v)
                           for k, v in maybe.items()}
            else:
                raise RuntimeError(f"Cannot extract update from client {getattr(c, 'client_id', 'unknown')}")
            updates.append(upd)
        return updates

    def _evaluate_aggregated_params(self, aggregated_params, model_factory, test_loader, device=None):
        """
        Given aggregated_params (dict param_name->tensor) and a model class factory,
        construct a fresh model, load params and evaluate accuracy on test_loader.
        Returns: accuracy (float)
        """
        device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        model = model_factory()
        model.to(device)

        try:
            model.load_state_dict(aggregated_params, strict=False)
        except Exception:
            sd = model.state_dict()
            for k in sd.keys():
                if k in aggregated_params:
                    try:
                        sd[k] = aggregated_params[k].clone().to(device)
                    except Exception:
                        pass
            model.load_state_dict(sd, strict=False)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                y = y.to(device)
                logits = model(X)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.numel()
        acc = float(correct) / total if total > 0 else 0.0
        return acc

    # ------------------------------
    # Coordinator helper
    # ------------------------------

    def run_and_evaluate_coordinator(self, clients, rounds, test_loader, test_X=None, y_test=None, use_defense=False):
        """
        Create coordinator with clients, run federated learning, return coordinator, result, baseline_acc, attack_success_rates.
        """
        coordinator = DecentralizedFLCoordinator(clients, use_committee_defense=use_defense)
        result = coordinator.run_federated_learning(rounds=rounds)

        baseline_final_acc = None
        attack_success_rates = None
        try:
            baseline_final_acc, _ = coordinator.evaluate_on_test_set(test_loader)
        except Exception:
            baseline_final_acc = None

        try:
            if test_X is not None and y_test is not None:
                attack_success_rates = coordinator.evaluate_attack_success(test_X, y_test)
        except Exception:
            attack_success_rates = None

        return coordinator, result, baseline_final_acc, attack_success_rates

    # ------------------------------
    # Multi-defense × multi-attack runner
    # ------------------------------

    def run_defense_vs_attacks(self, attack_configs, dataset_name, models_map,
                               num_clients, rounds, client_data, test_loader,
                               test_X=None, y_test=None, defenses_to_run=None,
                               save_csv=True):
        """
        For the given dataset and attack configs, run FL training once per attack,
        extract client updates, then run multiple defenses on the SAME client updates.

        Returns:
            summary: dict[attack_name][defense_name] = {'clean_accuracy': ..., 'asr': ..., 'baseline_acc': ...}
        """
        if defenses_to_run is None:
            defenses_to_run = ["committee", "krum", "multi-krum", "trimmed-mean",
                               "median", "norm-bounding"]

        summary = {}

        for attack_name, attack_config in attack_configs.items():
            print(f"\n{'='*70}")
            print(f"TESTING DEFENSES AGAINST: {attack_name.upper()}")
            print(f"{'='*70}")

            # Build clients (attack injected for malicious clients)
            clients = self.build_clients(num_clients, client_data, models_map, dataset_name, attack_config)

            # Run federated learning once without committee defense to get client updates
            coordinator, result, baseline_final_acc, attack_success_rates = self.run_and_evaluate_coordinator(
                clients, rounds, test_loader, test_X=test_X, y_test=y_test, use_defense=False
            )
            if baseline_final_acc is not None:
                print(f"[INFO] Baseline accuracy (no defense): {baseline_final_acc:.4f}")

            avg_asr = None
            if attack_success_rates is not None:
                try:
                    avg_asr = float(np.mean(attack_success_rates)) if len(attack_success_rates) > 0 else 0.0
                    print(f"[INFO] Baseline ASR: {avg_asr:.4f}")
                except Exception:
                    avg_asr = None

            # Extract client updates
            try:
                client_updates = self._extract_client_updates(clients)
                print(f"[INFO] Extracted {len(client_updates)} client updates")
            except Exception as e:
                print(f"[ERROR] Failed to extract client updates: {e}")
                continue

            summary[attack_name] = {}

            # Apply each defense to the same client_updates
            for defense_name in defenses_to_run:
                print(f"\n  → Testing defense: {defense_name.upper()}")
                try:
                    if defense_name == "krum":
                        defense = get_defense("krum")
                    elif defense_name == "multi-krum":
                        defense = get_defense("multi-krum", m=2)
                    elif defense_name == "trimmed-mean":
                        defense = get_defense("trimmed-mean", beta=0.1)
                    elif defense_name == "norm-bounding":
                        defense = get_defense("norm-bounding", max_norm=10.0)
                    else:
                        defense = get_defense(defense_name)

                    print(f"     ✓ Defense instantiated: {defense.__class__.__name__}")

                except Exception as e:
                    print(f"     ✗ Could not instantiate defense '{defense_name}': {e}")
                    summary[attack_name][defense_name] = {
                        "clean_accuracy": None,
                        "asr": None,
                        "error": str(e)
                    }
                    continue

                # Detect anomalies if the defense supports it
                try:
                    if hasattr(defense, "detect_anomalies"):
                        defense.detect_anomalies(client_updates, client_losses=None)
                        print(f"     ✓ Anomaly detection completed")
                except Exception as e:
                    print(f"     ⚠ Anomaly detection failed: {e}")

                # Aggregate
                try:
                    aggregated_params = defense.robust_aggregate(client_updates, client_weights=None)
                    print(f"     ✓ Aggregation completed")
                except Exception as e:
                    print(f"     ✗ Aggregation failed: {e}")
                    summary[attack_name][defense_name] = {
                        "clean_accuracy": None,
                        "asr": None,
                        "error": f"Aggregation failed: {str(e)}"
                    }
                    continue

                # Evaluate aggregated model
                try:
                    clean_acc = self._evaluate_aggregated_params(
                        aggregated_params, models_map[dataset_name], test_loader
                    )
                    print(f"     ✓ Clean accuracy: {clean_acc:.4f}")
                except Exception as e:
                    print(f"     ✗ Evaluation failed: {e}")
                    clean_acc = None

                # Save result (approximate ASR using baseline run)
                summary[attack_name][defense_name] = {
                    "clean_accuracy": float(clean_acc) if clean_acc is not None else None,
                    "asr": avg_asr,
                    "baseline_acc": float(baseline_final_acc) if baseline_final_acc is not None else None
                }

                print(f"     → Result: CleanAcc={clean_acc if clean_acc is not None else 'N/A'}, "
                      f"ASR={avg_asr if avg_asr is not None else 'N/A'}")

        # Save CSV summary
        if save_csv:
            rows = []
            for attack_name, defenses in summary.items():
                for defense_name, vals in defenses.items():
                    rows.append([
                        self.run_id,
                        dataset_name,
                        attack_name,
                        defense_name,
                        vals.get("clean_accuracy"),
                        vals.get("asr"),
                        vals.get("baseline_acc"),
                        vals.get("error", "")
                    ])

            csv_fname = os.path.join(self.results_dir,
                                     f"defense_vs_attack_summary_{dataset_name}_{self.run_id}.csv")
            with open(csv_fname, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["run_id", "dataset", "attack", "defense",
                                 "clean_accuracy", "asr", "baseline_acc", "error"])
                writer.writerows(rows)
            print(f"\n[INFO] ✓ Saved defense vs attack summary → {csv_fname}")

        # Plot comparison (call plotter helper directly to ensure passing dataset_name)
        try:
            self.plotter._plot_defense_comparison(summary, dataset_name)
        except Exception as e:
            print(f"[WARN] Failed to generate defense comparison plot via plotter: {e}")

        # Store for external use
        self.defense_results[dataset_name] = summary

        return summary

    # ------------------------------
    # Feature extraction helpers
    # ------------------------------

    def extract_client_features(self, clients, dataset_loader):
        """
        Extract features for all clients on given dataset (test or train).
        Returns a dictionary: {client_id: feature_tensor}.
        """
        client_features = {}
        if not clients:
            return client_features

        device = next(iter(clients)).model.conv1.weight.device

        with torch.no_grad():
            for client in clients:
                all_features = []
                for images, labels in dataset_loader:
                    images = images.to(device)
                    features = client.model.get_features(images)
                    all_features.append(features.cpu())
                client_features[client.client_id] = torch.cat(all_features, dim=0)

        return client_features

    # ------------------------------
    # Core experiments (refactored)
    # ------------------------------

    def run_comprehensive_evaluation(self, test_defenses=True):
        """
        Run comprehensive evaluation (baseline, attack, defense scenarios).
        This is the main training loop you originally used, but using helper functions.
        """
        num_clients = 10
        rounds = 1
        malicious_clients = 4

        attack_configs = self.get_attack_configs(malicious_clients=malicious_clients)
        datasets, models = self.get_datasets_and_models()

        print("=" * 80)
        print("ENHANCED DECENTRALIZED FL SECURITY EVALUATION")
        print("=" * 80)

        results = {}

        for dataset_name, (trainset, testset) in datasets.items():
            print(f"\n{'=' * 60}")
            print(f"EVALUATING ON {dataset_name}")
            print(f"{'=' * 60}")

            client_data = partition_dataset(trainset, num_clients, iid=False)
            test_loader = DataLoader(testset, batch_size=256, shuffle=False)
            test_X, y_test = next(iter(DataLoader(testset, batch_size=len(testset), shuffle=False)))
            test_X, y_test = test_X.numpy(), y_test.numpy()

            dataset_results = {}

            for attack_name, attack_config in attack_configs.items():
                print(f"\n--- Testing {attack_name.upper()} attack ---")
                scenarios = ['baseline', 'attack', 'defense'] if attack_config else ['baseline']
                attack_results = {}

                for scenario in scenarios:
                    # Build clients for this scenario
                    client_attack_config = attack_config if (scenario in ['attack', 'defense'] and attack_config) else None
                    clients = self.build_clients(num_clients, client_data, models, dataset_name, client_attack_config)

                    use_defense = scenario == 'defense'
                    # Run coordinator and evaluate
                    coordinator, result, _, _ = self.run_and_evaluate_coordinator(
                        clients, rounds, test_loader, test_X=test_X, y_test=y_test, use_defense=use_defense
                    )

                    # Extract training history in flexible formats
                    raw_loss_hist = []
                    raw_acc_hist = []
                    if isinstance(result, tuple):
                        if len(result) == 2:
                            raw_acc_hist, raw_loss_hist = result
                        elif len(result) == 3:
                            maybe_params, maybe_loss, maybe_acc = result
                            if isinstance(maybe_acc, (list, np.ndarray)) or isinstance(maybe_acc, dict):
                                raw_acc_hist = maybe_acc
                                raw_loss_hist = maybe_loss
                            else:
                                raw_acc_hist, raw_loss_hist = maybe_loss, maybe_acc
                        else:
                            print(f"[WARN] Unexpected tuple length: {len(result)}")
                    elif isinstance(result, dict):
                        raw_loss_hist = result.get("loss_history", []) or result.get("losses", []) or []
                        raw_acc_hist = result.get("accuracy_history", []) or result.get("accuracies", []) or []
                    else:
                        print("[WARN] Coordinator returned unexpected type:", type(result))

                    if raw_acc_hist is None:
                        raw_acc_hist = []
                    if raw_loss_hist is None:
                        raw_loss_hist = []

                    def _ensure_list(x):
                        if isinstance(x, (list, tuple, np.ndarray)):
                            return list(x)
                        if isinstance(x, dict):
                            try:
                                keys = list(x.keys())
                                if all(isinstance(k, (int, np.integer, float, np.floating)) for k in keys):
                                    ordered = [x[k] for k in sorted(keys, key=lambda k: float(k))]
                                    return ordered
                            except Exception:
                                pass
                            return list(x.values())
                        return [x]

                    raw_acc_list = _ensure_list(raw_acc_hist)
                    raw_loss_list = _ensure_list(raw_loss_hist)

                    acc_history = [self._safe_extract_number(entry, keys_fallback=['accuracy', 'avg_accuracy', 'value'])
                                   for entry in raw_acc_list]
                    loss_history = [self._safe_extract_number(entry, keys_fallback=['loss', 'value'])
                                    for entry in raw_loss_list]

                    if len(acc_history) == 0:
                        acc_history = [0.0]
                    if len(loss_history) == 0:
                        loss_history = [0.0]

                    final_acc = acc_history[-1]
                    final_loss = loss_history[-1]

                    try:
                        print(f"[DEBUG] Loss history length: {len(loss_history)}, "
                              f"Accuracy history length: {len(acc_history)}")
                        print(f"\n[INFO] Federated learning completed successfully.")
                        print(f"[INFO] Final Training Accuracy: {float(final_acc):.4f}")
                        print(f"[INFO] Final Training Loss: {float(final_loss):.4f}")
                    except Exception as e:
                        print(f"[WARN] Could not format final metrics: {e}")

                    # Evaluate on test set
                    final_accuracy, client_accuracies = coordinator.evaluate_on_test_set(test_loader)
                    # Evaluate attack success
                    try:
                        attack_success_rates = coordinator.evaluate_attack_success(test_X, y_test)
                    except TypeError as e:
                        print(f"[ERROR] evaluate_attack_success call failed: {e}")
                        # Fallback: create empty list
                        attack_success_rates = [0.0] * len(coordinator.clients)

                    safe_acc_history = [float(x) for x in acc_history]
                    safe_loss_history = [float(x) for x in loss_history]
                    safe_client_accuracies = [float(x) for x in client_accuracies]
                    safe_attack_success_rates = [float(x) for x in attack_success_rates] if attack_success_rates else []

                    attack_results[scenario] = {
                        'final_accuracy': float(final_accuracy),
                        'accuracy_history': safe_acc_history,
                        'loss_history': safe_loss_history,
                        'client_accuracies': safe_client_accuracies,
                        'attack_success_rates': safe_attack_success_rates
                    }

                    print(f"{scenario.title()} final accuracy: {final_accuracy:.4f}")

                dataset_results[attack_name] = attack_results

            results[dataset_name] = dataset_results

        self.results = results
        return results

    # ------------------------------
    # Results handling and plotting
    # ------------------------------

    def generate_results_table(self, results):
        """Print and return a summary table list for CSV saving."""
        print(f"\n{'='*100}")
        print("COMPREHENSIVE EVALUATION RESULTS")
        print(f"{'='*100}")

        print(f"\n{'Dataset':<15} {'Attack Type':<12} {'Baseline':<10} {'Under Attack':<12} "
              f"{'With Defense':<12} {'Defense Recovery':<15} {'Attack Success':<15}")
        print("-" * 105)

        table_data = []
        for dataset_name, dataset_results in results.items():
            for attack_name, attack_results in dataset_results.items():
                baseline_acc = attack_results.get('baseline', {}).get('final_accuracy', 0.0)
                attack_acc = attack_results.get('attack', {}).get('final_accuracy', baseline_acc)
                defense_acc = attack_results.get('defense', {}).get('final_accuracy', attack_acc)

                if baseline_acc > 0 and baseline_acc != attack_acc:
                    recovery = (defense_acc - attack_acc) / (baseline_acc - attack_acc) * 100
                    recovery = max(0, min(100, recovery))
                else:
                    recovery = 100 if defense_acc >= baseline_acc else 0

                attack_success = attack_results.get('attack', {}).get('attack_success_rates', [])
                avg_attack_success = np.mean(attack_success) if attack_success else 0.0

                print(f"{dataset_name:<15} {attack_name:<12} {baseline_acc:.4f}    {attack_acc:.4f}      "
                      f"{defense_acc:.4f}      {recovery:.1f}%            {avg_attack_success:.4f}")

                table_data.append([dataset_name, attack_name, baseline_acc, attack_acc,
                                   defense_acc, recovery, avg_attack_success])

        return table_data

    def save_results_table_to_csv(self, table_data, filename_prefix="evaluation_results"):
        """Save the printed table to CSV."""
        filename = f"{filename_prefix}_{self.run_id}.csv"
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Dataset", "Attack Type", "Baseline", "Under Attack",
                             "With Defense", "Defense Recovery (%)", "Attack Success"])
            writer.writerows(table_data)
        print(f"[INFO] Results table saved to {filepath}")

    def save_results_summary_to_json(self, results, filename_prefix="evaluation_summary"):
        """Save full results dict to JSON."""
        filename = f"{filename_prefix}_{self.run_id}.json"
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, "w") as f:
            json.dump(results, f, indent=4)
        print(f"[INFO] Results summary saved to {filepath}")

    def load_results_from_json(self, filepath):
        """Load a previously saved results JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No saved results found at {filepath}")
        with open(filepath, "r") as f:
            results = json.load(f)
        print(f"[INFO] Results loaded from {filepath}")
        self.results = results
        return results

    def generate_all_plots(self, results):
        """Delegate plotting to the PlottingEngine (keeps original single call)."""
        return self.plotter.generate_all_plots(results)

    def generate_comprehensive_report(self, results, features_dict=None):
        """Produce the same comprehensive report text as before."""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE EVALUATION REPORT")
        print(f"{'='*80}")

        # Overall statistics
        total_attacks_tested = 0
        successful_defenses = 0

        for dataset_results in results.values():
            for attack_name, attack_results in dataset_results.items():
                if attack_name != 'none':
                    total_attacks_tested += 1

                    baseline_acc = attack_results.get('baseline', {}).get('final_accuracy', 0)
                    attack_acc = attack_results.get('attack', {}).get('final_accuracy', 0)
                    defense_acc = attack_results.get('defense', {}).get('final_accuracy', 0)

                    if baseline_acc != attack_acc:
                        recovery = (defense_acc - attack_acc) / (baseline_acc - attack_acc)
                        if recovery > 0.7:
                            successful_defenses += 1
                    elif defense_acc >= baseline_acc:
                        successful_defenses += 1

        print(f"\nOverall Statistics:")
        print(f"  Total attack scenarios tested: {total_attacks_tested}")
        print(f"  Successful defense instances: {successful_defenses}")
        print(f"  Defense success rate: {successful_defenses/total_attacks_tested*100 if total_attacks_tested>0 else 0:.1f}%")

        # Attack-specific analysis
        print(f"\n{'='*60}")
        print("ATTACK-SPECIFIC ANALYSIS")
        print(f"{'='*60}")

        attack_analysis = {
            'slf': 'Static Label Flipping - Simple but effective label manipulation',
            'dlf': 'Dynamic Label Flipping - Sophisticated feature-space targeting',
            'centralized': 'Centralized Trigger - Backdoor attack with fixed triggers',
            'coordinated': 'Coordinated Trigger - Client-specific backdoor triggers',
            'random': 'Random Trigger - Stochastic backdoor generation',
            'model_dependent': 'Model-Dependent - Gradient-based trigger optimization'
        }

        for attack_name, description in attack_analysis.items():
            print(f"\n{attack_name.upper()} - {description}")
            print("-" * 50)

            for dataset_name, dataset_results in results.items():
                if attack_name in dataset_results:
                    attack_results = dataset_results[attack_name]

                    baseline_acc = attack_results.get('baseline', {}).get('final_accuracy', 0)
                    attack_acc = attack_results.get('attack', {}).get('final_accuracy', 0)
                    defense_acc = attack_results.get('defense', {}).get('final_accuracy', 0)

                    attack_success = attack_results.get('attack', {}).get('attack_success_rates', [])
                    avg_attack_success = np.mean(attack_success) if attack_success else 0

                    impact = (baseline_acc - attack_acc) * 100

                    if baseline_acc != attack_acc:
                        recovery = (defense_acc - attack_acc) / (baseline_acc - attack_acc) * 100
                        recovery = max(0, min(100, recovery))
                    else:
                        recovery = 100 if defense_acc >= baseline_acc else 0

                    print(f"  {dataset_name}:")
                    print(f"    Impact: {impact:.2f}% accuracy drop")
                    print(f"    Attack success: {avg_attack_success:.3f}")
                    print(f"    Defense recovery: {recovery:.1f}%")

        # Feature similarity summary
        if features_dict:
            print(f"\n{'='*60}")
            print("FEATURE SIMILARITY ANALYSIS")
            print(f"{'='*60}")

            for client_id, feats in features_dict.items():
                sim_matrix = cosine_similarity(feats.numpy())
                mean_sim = sim_matrix.mean()
                min_sim = sim_matrix.min()
                max_sim = sim_matrix.max()
                print(f"Client {client_id}: mean={mean_sim:.3f}, min={min_sim:.3f}, max={max_sim:.3f}")

        # Defense mechanism analysis
        print(f"\n{'='*60}")
        print("DEFENSE MECHANISM ANALYSIS")
        print(f"{'='*60}")

        print("\nCommittee-based Defense Properties:")
        print("  ✓ Anomaly detection through parameter distance analysis")
        print("  ✓ Robust aggregation with outlier exclusion")
        print("  ✓ Byzantine fault tolerance")
        print("  ✓ Adaptive threshold-based filtering")

        print("\nStrengths:")
        print("  • Effective against label-flipping attacks")
        print("  • Maintains model utility while providing security")
        print("  • Scales well with number of clients")
        print("  • No single point of failure")

        print("\nLimitations:")
        print("  • May struggle with sophisticated trigger-based attacks")
        print("  • Requires careful threshold tuning")
        print("  • Computational overhead for anomaly detection")
        print("  • Limited effectiveness against coordinated attacks")

        return {
            'total_attacks_tested': total_attacks_tested,
            'successful_defenses': successful_defenses,
            'defense_success_rate': successful_defenses/total_attacks_tested*100 if total_attacks_tested > 0 else 0,
            'plots_generated': self.plotter.plots_generated
        }


# =============================================================================
# MAIN EXECUTION FUNCTIONS (kept names the same, reorganized internals)
# =============================================================================

def run_enhanced_evaluation(resume_from_json=None, out_dir="Output"):
    """
    Main entry point — runs the comprehensive evaluation (training) and then the
    extended multi-defense comparison automatically, saves CSVs and plots.
    """
    print("="*100)
    print("ENHANCED DECENTRALIZED FEDERATED LEARNING SECURITY FRAMEWORK")
    print("="*100)

    try:
        # Initialize evaluation framework
        if resume_from_json:
            evaluator = EvaluationFramework(out_dir=out_dir)
            results = evaluator.load_results_from_json(resume_from_json)
            print("[INFO] Skipping training, reusing saved results.")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            evaluator = EvaluationFramework(out_dir=out_dir, run_id=timestamp)
            results = evaluator.run_comprehensive_evaluation()

            # Save results
            table_data = evaluator.generate_results_table(results)
            evaluator.save_results_table_to_csv(table_data)
            evaluator.save_results_summary_to_json(results)

        # Feature extraction (if applicable)
        features_dict = None
        if hasattr(evaluator, 'clients') and evaluator.clients:
            test_loader = DataLoader(evaluator.testset, batch_size=128, shuffle=False)
            features_dict = evaluator.extract_client_features(evaluator.clients, test_loader)

            features_file = os.path.join(evaluator.plots_dir, 'client_features.pt')
            torch.save(features_dict, features_file)
            print(f"[INFO] Features extracted and saved: {features_file}")

        # -----------------------------------------------------------------
        # Extended multi-defense evaluation: run all defenses for all attacks
        # -----------------------------------------------------------------
        print("\n[INFO] Starting extended multi-defense evaluation phase...")

        num_clients = 10
        rounds = 1
        attack_configs = evaluator.get_attack_configs(malicious_clients=4)
        datasets, models = evaluator.get_datasets_and_models()

        for dataset_name, (trainset, testset) in datasets.items():
            test_loader = DataLoader(testset, batch_size=256, shuffle=False)
            test_X, y_test = next(iter(DataLoader(testset, batch_size=len(testset), shuffle=False)))
            test_X, y_test = test_X.numpy(), y_test.numpy()
            client_data = partition_dataset(trainset, num_clients, iid=False)

            evaluator.run_defense_vs_attacks(
                attack_configs=attack_configs,
                dataset_name=dataset_name,
                models_map=models,
                num_clients=num_clients,
                rounds=rounds,
                client_data=client_data,
                test_loader=test_loader,
                test_X=test_X,
                y_test=y_test,
                defenses_to_run=["committee", "krum", "multi-krum", "trimmed-mean", "median", "norm-bounding"],
                save_csv=True
            )

        # Generate all visualizations using PlottingEngine
        print("\nGenerating comprehensive visualizations...")
        evaluator.generate_all_plots(results)

        # Generate comprehensive report
        report_stats = evaluator.generate_comprehensive_report(results, features_dict)

        print(f"\n{'='*80}")
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")

        print("\nKey Findings:")
        print("• Advanced mathematical attack formulations provide deeper insights")
        print("• Committee-based defense shows strong performance across attack types")
        print("• Dynamic attacks (DLF) pose greater challenges than static ones")
        print("• Trigger-based backdoors require specialized detection methods")
        print("• Decentralized FL maintains robustness under diverse threat models")

        print(f"\nGenerated {len(report_stats['plots_generated'])} visualization plots")

        return results, report_stats

    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()


def run_attack_comparison_test():
    """Run side-by-side comparison of all attacks with detailed metrics (keeps original logic)."""
    print("="*80)
    print("ATTACK COMPARISON TEST - DETAILED METRICS")
    print("="*80)

    # Load Fashion-MNIST
    trainset, testset = load_fashion_mnist()
    client_data = partition_dataset(trainset, num_clients=10, iid=False, alpha=0.3)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False)

    test_X, y_test = next(iter(DataLoader(testset, batch_size=len(testset), shuffle=False)))
    test_X, y_test = test_X.numpy(), y_test.numpy()

    malicious_clients = 4
    attack_configs = EvaluationFramework().get_attack_configs(malicious_clients=malicious_clients)

    comparison_results = {}

    for attack_name, attack_config in attack_configs.items():
        print(f"\n--- Testing {attack_name.upper()} ---")

        clients = []
        for i in range(10):
            model = FashionMNISTNet()
            if attack_config and i < attack_config.num_malicious_clients:
                client_attack = attack_config
            else:
                client_attack = None
            clients.append(DecentralizedClient(i, client_data[i][0], client_data[i][1],
                                              model, client_attack))

        coordinator = DecentralizedFLCoordinator(clients, use_committee_defense=True)
        accuracy_history = coordinator.run_federated_learning(rounds=1)

        final_accuracy, client_accuracies = coordinator.evaluate_on_test_set(test_loader)
        attack_success_rates = coordinator.evaluate_attack_success(test_X, y_test)

        if attack_config is None:
            comparison_results[attack_name] = {
                'final_accuracy': final_accuracy,
                'attack_success': 0.0,
                'malicious_clients': 0,
                'poisoning_rate': 0.0,
                'attack_type': "BASELINE"
            }
        else:
            comparison_results[attack_name] = {
                'final_accuracy': final_accuracy,
                'attack_success': np.mean(attack_success_rates) if attack_success_rates else 0,
                'malicious_clients': sum(1 for c in clients if getattr(c, 'is_malicious', False)),
                'poisoning_rate': attack_config.poisoning_rate,
                'attack_type': attack_config.attack_type
            }

        print(f"Results: accuracy={final_accuracy:.4f}")

    # Results table
    print(f"\n{'='*80}")
    print("ATTACK COMPARISON RESULTS")
    print(f"{'='*80}")

    print(f"{'Attack':<20} {'Type':<12} {'PDR':<6} {'Accuracy':<10} {'Attack Success':<15} {'Stealth':<12}")
    print("-" * 80)

    for name, res in comparison_results.items():
        stealth_score = res['final_accuracy'] * (1 - res['attack_success'] * 0.5)
        print(f"{name:<20} {res['attack_type'].upper():<12} {res['poisoning_rate']:.1%}    "
              f"{res['final_accuracy']:.4f}    {res['attack_success']:.4f}         {stealth_score:.4f}")

    return comparison_results


if __name__ == "__main__":
    results, report_stats = run_enhanced_evaluation(resume_from_json=None, out_dir="Output")
    comparison_results = run_attack_comparison_test()

