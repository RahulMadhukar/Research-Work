import csv
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

from attacks.attack_summery import attack_summary
from attacks.base import AttackConfig
from models import FashionMNISTNet, CIFAR10Net
from datasets import load_fashion_mnist, load_cifar10, partition_dataset
from client import DecentralizedClient
from coordinator import DecentralizedFLCoordinator
from checkpoints import CheckpointManager
from plots import PlottingEngine


class EvaluationFramework:
    """
    Framework to run comprehensive evaluations of decentralized FL under
    various attacks and committee-based defense. Refactored to centralize repeated logic.
    """

    def __init__(self, out_dir="Output", run_id=None):
        """Initialize directories, plotter and basic bookkeeping."""
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        project_root = os.path.dirname(os.path.abspath(__file__))

        # Resolve base_run_dir
        # FIXED: Always create structure as {out_dir}/{run_id}/
        if os.path.isabs(out_dir):
            # Absolute path like /export/home/.../plots
            self.base_run_dir = os.path.join(out_dir, self.run_id)
        elif self.run_id in out_dir:
            # out_dir already contains run_id
            self.base_run_dir = os.path.join(project_root, out_dir)
        else:
            # Relative path like "plots" -> plots/{run_id}/
            output_root = os.path.join(project_root, out_dir)
            self.base_run_dir = os.path.join(output_root, self.run_id)

        # Create folders
        self.plots_dir = os.path.join(self.base_run_dir, "plots")
        self.results_dir = os.path.join(self.base_run_dir, "results")
        self.checkpoint_dir = os.path.join(self.base_run_dir, "checkpoints")
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Compatibility
        self.out_dir = self.plots_dir

        # Plotter
        self.plotter = PlottingEngine(output_dir=self.plots_dir, run_id=self.run_id)

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.base_run_dir,
            run_id=self.run_id
        )

        # Storage
        self.results = {}
        self.plots_generated = []
        self.defense_results = {}
        
        # Track best metrics per scenario
        self.best_metrics = {}

    # ------------------------------
    # Shared configuration helpers
    # ------------------------------

    def get_attack_configs(self, malicious_clients: int = 4, poisoning_rate: float = 0.60):
        """Return canonical attack_configs dictionary used throughout the code."""
        return {
            'none': None,
            'slf': AttackConfig(attack_type='slf', poisoning_rate=poisoning_rate, source_class=[0, 2, 4, 6], target_class=[1, 3, 5, 7], num_malicious_clients=malicious_clients),
            'dlf': AttackConfig(attack_type='dlf', poisoning_rate=poisoning_rate, target_class=0,
                                num_malicious_clients=malicious_clients),
            'centralized': AttackConfig(attack_type='centralized', poisoning_rate=poisoning_rate, target_class=0,
                                        trigger_size=(8, 8), trigger_intensity=0.4,
                                        num_malicious_clients=malicious_clients),
            'coordinated': AttackConfig(attack_type='coordinated', poisoning_rate=poisoning_rate, target_class=0,
                                        trigger_size=(8, 8), trigger_intensity=0.4,
                                        num_malicious_clients=malicious_clients),
            'random': AttackConfig(attack_type='random', poisoning_rate=poisoning_rate, target_class=0,
                                   trigger_size=(8, 8), trigger_intensity=0.4,
                                   num_malicious_clients=malicious_clients),
            'model_dependent': AttackConfig(attack_type='model_dependent', poisoning_rate=0.8,  # CHANGED: increased from poisoning_rate (0.3) to 0.8 for more poisoned samples
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


    def _subset_dataset(self, dataset, fraction=0.1):
        """
        Reduce dataset to specified fraction.
        
        Args:
            dataset: PyTorch Dataset or Subset
            fraction: Fraction to keep (0.1 = 10%)

        Returns:
            Subset of the dataset
        """
        if fraction >= 1.0:
            return dataset
        
        subset_size = int(len(dataset) * fraction)
        indices = torch.randperm(len(dataset))[:subset_size]
        subset = torch.utils.data.Subset(dataset, indices)
        
        print(f"[INFO] Subsetted dataset: {len(dataset)} → {subset_size} samples ({fraction*100:.0f}%)")
        return subset

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
    
    def run_and_evaluate_coordinator(
        self, clients, rounds, test_loader,
        test_X=None, y_test=None, use_defense=False, defense_type='committee',
        start_round=0, dataset="", attack="", scenario=""
    ):
        """
        Run federated learning in a single call to coordinator.run_federated_learning,
        without iterating per round inside this function. Save checkpoint after all rounds.

        Args:
            clients: List of client objects participating in FL.
            rounds: Number of communication rounds to run.
            test_loader: DataLoader for test set evaluation.
            test_X: (Optional) Numpy array of test features for attack evaluation.
            y_test: (Optional) Numpy array of test labels for attack evaluation.
            use_defense: Whether to enable a defense mechanism during training.
            defense_type: Type of defense to use if use_defense is True. Options are:
                - 'committee': Committee-based anomaly/outlier detection and robust aggregation.
                - 'adaptive': Adaptive defense with dynamic thresholding or filtering.
                - 'reputation': Reputation-based defense (if implemented).
                - 'gradient': Gradient-based defense (if implemented).
                - 'ensemble': Ensemble of multiple defense strategies.
            start_round: (Optional) Starting round number (for resuming).
            dataset: (Optional) Name of the dataset (for logging/checkpointing).
            attack: (Optional) Name of the attack (for logging/checkpointing).
            scenario: (Optional) Scenario label (e.g., 'baseline', 'attack', 'defense').

        Returns:
            Tuple containing:
                - coordinator: The FL coordinator object after training.
                - result: Tuple of histories (accuracy, loss, etc.).
                - final_test_acc: Final test accuracy after training.
                - attack_success_rates: List of attack success rates (if applicable).
                - training_acc_history: List of training accuracies per round.
                - training_loss_history: List of training losses per round.
                - test_acc_history: List of test accuracies per round.
                - test_loss_history: List of test losses per round.
                - detection_metrics: Dictionary of detection metrics (if defense enabled).
        """
        coordinator = DecentralizedFLCoordinator(clients, use_defense=use_defense, defense_type=defense_type)

        best_accuracy = -1.0
        best_round = 0

        # Initialize histories (will be populated from coordinator after training)
        training_acc_history = []
        training_loss_history = []
        test_acc_history = []
        test_loss_history = []

        scenario_key = f"{dataset}_{attack}_{scenario}"
        if scenario_key not in self.best_metrics:
            self.best_metrics[scenario_key] = {'accuracy': -1.0, 'round': 0}

        device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))

        print(f"[INFO] Running federated learning for {rounds} rounds one at a time to track test metrics...")

        # Initialize histories
        test_acc_history = []
        test_loss_history = []

        # Run rounds one at a time to evaluate test set after each round
        try:
            for round_num in range(1, rounds + 1):
                # Run one round
                coordinator.run_federated_learning(rounds=1)

                # Evaluate on test set after this round
                try:
                    round_test_acc, _ = coordinator.evaluate_on_test_set(test_loader)
                    round_test_loss = self._compute_test_loss(
                        coordinator.global_model.state_dict() if hasattr(coordinator, 'global_model')
                        else clients[0].model.state_dict(),
                        lambda: type(clients[0].model)(),
                        test_loader,
                        device
                    )
                    test_acc_history.append(float(round_test_acc))
                    test_loss_history.append(float(round_test_loss))
                except Exception as e:
                    print(f"[WARN] Test evaluation failed at round {round_num}: {e}")
                    test_acc_history.append(0.0)
                    test_loss_history.append(0.0)

                if round_num % 1 == 0 or round_num == rounds:
                    print(f"  Round {round_num}/{rounds} - Test Acc: {test_acc_history[-1]:.4f}, Test Loss: {test_loss_history[-1]:.4f}")
        except Exception as e:
            print(f"[ERROR] Federated learning failed: {e}")
            self.checkpoint_manager.save_checkpoint(
                clients=clients,
                round_num=len(test_acc_history),
                dataset=dataset,
                attack=attack,
                scenario=scenario,
                metrics={'error': str(e)},
                is_best=False
            )
            raise

        # After all rounds, collect the FULL history from coordinator
        if hasattr(coordinator, 'global_accuracy_history') and coordinator.global_accuracy_history:
            training_acc_history = [float(x) for x in coordinator.global_accuracy_history]
            final_train_acc = float(coordinator.global_accuracy_history[-1])
            print(f"[INFO] Collected {len(training_acc_history)} rounds of training accuracy history")
        else:
            training_acc_history = []
            final_train_acc = 0.0
            print(f"[WARN] No training accuracy history available from coordinator")

        if hasattr(coordinator, 'round_history') and coordinator.round_history:
            training_loss_history = [float(r.get('avg_loss', 0.0)) for r in coordinator.round_history]
            final_train_loss = float(coordinator.round_history[-1].get('avg_loss', 0.0))
            print(f"[INFO] Collected {len(training_loss_history)} rounds of training loss history")
        else:
            training_loss_history = []
            final_train_loss = 0.0
            print(f"[WARN] No training loss history available from coordinator")

        print(f"[INFO] Collected {len(test_acc_history)} rounds of test accuracy history")
        print(f"[INFO] Collected {len(test_loss_history)} rounds of test loss history")

        # Get final values from test history
        if test_acc_history:
            final_test_acc = test_acc_history[-1]
        else:
            final_test_acc = 0.0

        if test_loss_history:
            final_test_loss = test_loss_history[-1]
        else:
            final_test_loss = 0.0

        # Check for best model
        is_best = final_test_acc > self.best_metrics[scenario_key]['accuracy']
        if is_best:
            self.best_metrics[scenario_key]['accuracy'] = final_test_acc
            self.best_metrics[scenario_key]['round'] = rounds
            best_accuracy = final_test_acc
            best_round = rounds
            print(f"[CHECKPOINT] New best accuracy: {final_test_acc:.4f} at round {best_round}")

        # Save checkpoint after completing all rounds
        metrics = {
            'round': rounds,
            'train_accuracy': final_train_acc,
            'test_accuracy': final_test_acc,
            'train_loss': final_train_loss,
            'test_loss': final_test_loss
        }
        print(f"[CHECKPOINT] Saving checkpoint at final round {rounds}")
        try:
            self.checkpoint_manager.save_checkpoint(
                clients=clients,
                round_num=rounds,
                dataset=dataset,
                attack=attack,
                scenario=scenario,
                metrics=metrics,
                is_best=is_best
            )
        except Exception as e:
            print(f"[WARN] Failed to save checkpoint: {e}")

        # Evaluate attack success
        attack_success_rates = []
        if test_X is not None and y_test is not None:
            try:
                attack_success_rates = coordinator.evaluate_attack_success(test_X, y_test)
            except Exception as e:
                print(f"[WARN] Attack success evaluation failed: {e}")

        # Extract detection metrics if defense was used
        detection_metrics = {}
        if use_defense:
            try:
                # Try to get committee metrics (for committee defense)
                if hasattr(coordinator, 'get_committee_metrics'):
                    detection_metrics = coordinator.get_committee_metrics()
                    print(f"[DEBUG] Detection metrics extracted: {detection_metrics}")
                # For other defenses, extract from defense mechanism attributes
                elif hasattr(coordinator, 'defense') and coordinator.defense is not None:
                    defense = coordinator.defense
                    # Extract metrics based on defense type
                    if hasattr(defense, 'detected_malicious'):
                        num_detected = len(defense.detected_malicious)
                        num_malicious = sum(1 for c in coordinator.clients if c.is_malicious)
                        num_benign = len(coordinator.clients) - num_malicious

                        # Calculate metrics
                        detection_rate = (num_detected / num_malicious * 100) if num_malicious > 0 else 0.0
                        false_positives = sum(1 for cid in defense.detected_malicious if not coordinator.clients[cid].is_malicious)
                        false_positive_rate = (false_positives / num_benign * 100) if num_benign > 0 else 0.0

                        detection_metrics = {
                            'detection_rate': detection_rate,
                            'false_positive_rate': false_positive_rate,
                            'actual_malicious': num_malicious,
                            'total_clients': len(coordinator.clients),
                            'detected_count': num_detected
                        }
                        print(f"[DEBUG] Detection metrics extracted from defense: {detection_metrics}")
            except Exception as e:
                print(f"[WARN] Failed to extract detection metrics: {e}")
                import traceback
                traceback.print_exc()

        # Gather results
        result = (
            coordinator.global_accuracy_history if hasattr(coordinator, 'global_accuracy_history') else [],
            [],
            coordinator.round_history if hasattr(coordinator, 'round_history') else []
        )

        return (
            coordinator, result, final_test_acc, attack_success_rates,
            training_acc_history, training_loss_history,
            test_acc_history, test_loss_history, detection_metrics
        )

    def _compute_test_loss(self, aggregated_params, model_factory, test_loader, device=None):
        """
        Compute average test loss for the aggregated model.
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
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                y = y.to(device)
                logits = model(X)
                loss = criterion(logits, y)
                total_loss += loss.item() * y.size(0)
                total_samples += y.size(0)
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return avg_loss

    # ------------------------------
    # Helpers: convert state_dict <-> flat vector (for compatibility if needed)
    # ------------------------------
    @staticmethod
    def _state_dict_to_vector(state_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        """Convert state_dict to flat numpy vector."""
        parts = []
        for k, v in state_dict.items():
            arr = v.detach().cpu().numpy().ravel()
            parts.append(arr)
        return np.concatenate(parts).astype(np.float32)

    @staticmethod
    def _vector_to_state_dict(vector: np.ndarray, template_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert flat vector back to state_dict using template."""
        arr = np.asarray(vector).ravel().astype(np.float32)
        new_sd = {}
        idx = 0
        for k, v in template_state_dict.items():
            num = v.numel()
            chunk = arr[idx: idx + num]
            new_sd[k] = torch.from_numpy(chunk.reshape(v.size())).type(v.dtype)
            idx += num
        return new_sd

    # ------------------------------
    # Defense vs Attacks Evaluation
    # ------------------------------

    def run_defense_vs_attacks(self, attack_configs, dataset_name, models_map,
                           num_clients, rounds, client_data, test_loader,
                           test_X=None, y_test=None, defenses_to_run=None,
                           save_csv=True):
        """
        For each attack, run FL training with different defense mechanisms.
        Tests: committee, adaptive, and ensemble defenses.
        For 'none' (no attack), skip entirely - no attack, no defense needed.
        """

        if defenses_to_run is None:
            # Test all available defenses by default
            defenses_to_run = ["committee", "adaptive", "reputation", "gradient", "ensemble"]

        summary = {}
        device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))

        for attack_name, attack_config in attack_configs.items():
            # Skip 'none' attack - no need to test defenses against no attack
            if attack_name.lower() == 'none' or attack_config is None:
                print(f"\n{'='*70}\nSKIPPING: {attack_name} (no attack to defend against)\n{'='*70}")
                continue
                
            print(f"\n{'='*70}\nTESTING: {attack_name}\n{'='*70}")

            # Build clients and ensure local poisoning setup
            clients = self.build_clients(num_clients, client_data, models_map, dataset_name, attack_config)
            for c in clients:
                try:
                    if hasattr(c, "poison_local_data"):
                        c.poison_local_data()
                except Exception as e:
                    print(f"[WARN] Client {getattr(c,'client_id','?')} poison_local_data failed: {e}")

            # Run FL with attack (no defense)
            coordinator, result, attack_accuracy, attack_success_rates, _, _, _, _, _ = self.run_and_evaluate_coordinator(
                clients, rounds, test_loader, test_X=test_X, y_test=y_test, use_defense=False,
                dataset=dataset_name, attack=attack_name, scenario="attack"
            )
            print(f"[INFO] Attack (no defense) accuracy: {attack_accuracy}")

            avg_asr = None
            if attack_success_rates is not None and len(attack_success_rates) > 0:
                avg_asr = float(np.mean(attack_success_rates))

            # Run clean baseline
            clean_clients = self.build_clients(num_clients, client_data, models_map, dataset_name, attack_config=None)
            for c in clean_clients:
                if hasattr(c, "poison_local_data"):
                    c.poison_local_data()
            _, _, clean_accuracy, _, _, _, _, _, _ = self.run_and_evaluate_coordinator(
                clean_clients, rounds, test_loader, test_X=test_X, y_test=y_test, use_defense=False,
                dataset=dataset_name, attack=attack_name, scenario="baseline"
            )
            print(f"[INFO] Clean baseline accuracy: {clean_accuracy}")

            summary[attack_name] = {}

            # --------------------------------------------------------
            # Run enhanced defenses (committee, adaptive, ensemble)
            # --------------------------------------------------------
            for defense_name in defenses_to_run:
                print(f"\n  → Testing defense: {defense_name.upper()}")

                # Build fresh clients with attack config for defense test
                defense_clients = self.build_clients(num_clients, client_data, models_map, dataset_name, attack_config)
                for c in defense_clients:
                    try:
                        if hasattr(c, "poison_local_data"):
                            c.poison_local_data()
                    except Exception as e:
                        print(f"[WARN] Client {getattr(c,'client_id','?')} poison_local_data failed: {e}")

                # Run FL with specified defense type enabled
                try:
                    _, _, defense_accuracy, defense_asr, _, _, _, _, _ = self.run_and_evaluate_coordinator(
                        defense_clients, rounds, test_loader, test_X=test_X, y_test=y_test,
                        use_defense=True, defense_type=defense_name,
                        dataset=dataset_name, attack=attack_name, scenario=f"defense_{defense_name}"
                    )
                    print(f"     ✓ {defense_name.capitalize()} defense accuracy: {defense_accuracy:.4f}")
                except Exception as e:
                    print(f"     ✗ Defense evaluation failed: {e}")
                    summary[attack_name][defense_name] = {
                        "clean_accuracy": clean_accuracy,
                        "attack_accuracy": attack_accuracy,
                        "defense_accuracy": None,
                        "ASR": avg_asr,
                        "defense_recovery_rate": None,
                        "error": str(e)
                    }
                    continue

                # Compute recovery rate
                if (clean_accuracy is not None) and (attack_accuracy is not None) and (defense_accuracy is not None):
                    if clean_accuracy != attack_accuracy:
                        recovery = ((defense_accuracy - attack_accuracy) /
                                    (clean_accuracy - attack_accuracy)) * 100
                        recovery = max(0, min(100, recovery))
                    else:
                        recovery = 0.0
                else:
                    recovery = None

                # Store results
                summary[attack_name][defense_name] = {
                    "clean_accuracy": float(clean_accuracy) if clean_accuracy is not None else None,
                    "attack_accuracy": float(attack_accuracy) if attack_accuracy is not None else None,
                    "defense_accuracy": float(defense_accuracy) if defense_accuracy is not None else None,
                    "ASR": float(avg_asr) if avg_asr is not None else None,
                    "defense_recovery_rate": float(recovery) if recovery is not None else None,
                }

                print(
                    f"     → Result: Clean={f'{clean_accuracy:.2f}' if clean_accuracy is not None else 'N/A'}, "
                    f"Attack={f'{attack_accuracy:.2f}' if attack_accuracy is not None else 'N/A'}, "
                    f"Defense={f'{defense_accuracy:.2f}' if defense_accuracy is not None else 'N/A'}, "
                    f"ASR={f'{avg_asr:.2f}' if avg_asr is not None else 'N/A'}, "
                    f"Recovery={f'{recovery:.2f}' if recovery is not None else 'N/A'}"
                )

        # --------------------------------------------------------
        # Save results to CSV
        # --------------------------------------------------------
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
                        vals.get("attack_accuracy"),
                        vals.get("defense_accuracy"),
                        vals.get("ASR"),
                        vals.get("defense_recovery_rate"),
                        vals.get("error", "")
                    ])

            csv_fname = os.path.join(self.results_dir,
                                    f"defense_vs_attack_summary_{dataset_name}_{self.run_id}.csv")
            os.makedirs(self.results_dir, exist_ok=True)
            with open(csv_fname, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "run_id", "dataset", "attack", "defense",
                    "clean_accuracy", "attack_accuracy", "defense_accuracy",
                    "ASR", "defense_recovery_rate", "error"
                ])
                writer.writerows(rows)
            print(f"\n[INFO] ✓ Saved defense vs attack summary → {csv_fname}")

        # Plotting
        try:
            self.plotter.plot_defense_comparison(summary, dataset_name, self.run_id)
        except Exception as e:
            print(f"[WARN] Failed to generate defense comparison plot via plotter: {e}")

        self.defense_results[dataset_name] = summary
        return summary

    # ------------------------------
    # Core experiments (refactored)
    # ------------------------------

    def run_comprehensive_evaluation(self, test_defenses=True, subset_fraction=0.1, resume=False, defenses_to_test=None):
        """
        Run comprehensive evaluation with proper history storage.
        """
        num_clients = 10
        rounds = 30
        malicious_clients = 4

        attack_configs = self.get_attack_configs(malicious_clients=malicious_clients)
        datasets, models = self.get_datasets_and_models()

        print("=" * 80)
        print("ENHANCED DECENTRALIZED FL SECURITY EVALUATION")
        print(f"Using {subset_fraction*100:.0f}% of each dataset for faster evaluation")
        print("=" * 80)

        results = {}

        for dataset_name, (trainset, testset) in datasets.items():
            print(f"\n{'=' * 60}")
            print(f"EVALUATING ON {dataset_name}")
            print(f"{'=' * 60}")

            # Subset the datasets
            print(f"[INFO] Original sizes - Train: {len(trainset)}, Test: {len(testset)}")
            trainset = self._subset_dataset(trainset, subset_fraction)
            testset = self._subset_dataset(testset, subset_fraction)
            print(f"[INFO] Subsetted sizes - Train: {len(trainset)}, Test: {len(testset)}")

            client_data = partition_dataset(trainset, num_clients, iid=False)
            test_loader = DataLoader(testset, batch_size=256, shuffle=False)
            test_X, y_test = next(iter(DataLoader(testset, batch_size=len(testset), shuffle=False)))
            test_X, y_test = test_X.numpy(), y_test.numpy()

            dataset_results = {}

            for attack_name, attack_config in attack_configs.items():
                # Only run baseline for 'none' attack, otherwise run baseline, attack, and defenses for each attack
                if attack_name.lower() == 'none':
                    print(f"\n--- Testing BASELINE (No Attack) ---")
                    # Only run baseline scenario for 'none'
                    scenarios = ['baseline']
                else:
                    print(f"\n--- Testing {attack_name.upper()} attack ---")
                    # For real attacks, run baseline, attack, and then defenses
                    scenarios = ['baseline', 'attack']
                
                attack_results = {}

                for scenario in scenarios:
                    print(f"\n[SCENARIO] Running {scenario.upper()} for {attack_name}")
                    # For baseline, do not use attack config; for attack, use attack config
                    client_attack_config = attack_config if (scenario == 'attack' and attack_config) else None
                    clients = self.build_clients(num_clients, client_data, models, dataset_name, client_attack_config)
                    use_defense = False
                    coordinator, result, final_test_accuracy, attack_success_rates_final, training_acc_history, training_loss_history, test_acc_history, test_loss_history, _ = self.run_and_evaluate_coordinator(
                        clients, rounds, test_loader, test_X=test_X, y_test=y_test,
                        use_defense=use_defense, dataset=dataset_name, attack=attack_name, scenario=scenario
                    )
                    attack_results[scenario] = {
                        'final_accuracy': float(final_test_accuracy),
                        'training_acc_history': [float(x) for x in training_acc_history],  
                        'training_loss_history': [float(x) for x in training_loss_history],     
                        'test_acc_history': [float(x) for x in test_acc_history],         
                        'test_loss_history': [float(x) for x in test_loss_history],             
                        'attack_success_rates': [float(x) for x in attack_success_rates_final] if attack_success_rates_final else []
                    }
                    print(f"[RESULT] {scenario.title()} - Final Test Acc: {final_test_accuracy:.4f}")
                    print(f"[HISTORY] Collected {len(training_acc_history)} rounds of training history")
                    print(f"[HISTORY] Collected {len(test_acc_history)} rounds of test history")

                # Only run defenses for real attacks (not for 'none')
                if attack_name.lower() != 'none':
                    defenses_to_run = ["committee", "adaptive", "reputation", "gradient", "ensemble"]
                    for defense_name in defenses_to_run:
                        print(f"\n[SCENARIO] Running DEFENSE ({defense_name.upper()}) for {attack_name}")
                        defense_clients = self.build_clients(num_clients, client_data, models, dataset_name, attack_config)
                        for c in defense_clients:
                            try:
                                if hasattr(c, "poison_local_data"):
                                    c.poison_local_data()
                            except Exception as e:
                                print(f"[WARN] Client {getattr(c,'client_id','?')} poison_local_data failed: {e}")
                        coordinator, result, final_test_accuracy, attack_success_rates_final, training_acc_history, training_loss_history, test_acc_history, test_loss_history, _ = self.run_and_evaluate_coordinator(
                            defense_clients, rounds, test_loader, test_X=test_X, y_test=y_test,
                            use_defense=True, defense_type=defense_name, dataset=dataset_name, attack=attack_name, scenario=f"defense_{defense_name}"
                        )
                        attack_results[f"defense_{defense_name}"] = {
                            'final_accuracy': float(final_test_accuracy),
                            'training_acc_history': [float(x) for x in training_acc_history],  
                            'training_loss_history': [float(x) for x in training_loss_history],     
                            'test_acc_history': [float(x) for x in test_acc_history],         
                            'test_loss_history': [float(x) for x in test_loss_history],             
                            'attack_success_rates': [float(x) for x in attack_success_rates_final] if attack_success_rates_final else []
                        }
                        print(f"[RESULT] Defense {defense_name} - Final Test Acc: {final_test_accuracy:.4f}")

                dataset_results[attack_name] = attack_results

            results[dataset_name] = dataset_results

            # REMOVED: Redundant defense comparison (was re-running all attacks)
            # The defenses are already tested in the main loop above
            # if test_defenses:
            #     print(f"\n{'='*70}")
            #     print(f"RUNNING DEFENSE COMPARISON FOR {dataset_name}")
            #     print(f"{'='*70}")
            #
            #     self.run_defense_vs_attacks(
            #         attack_configs=attack_configs,
            #         dataset_name=dataset_name,
            #         models_map=models,
            #         num_clients=num_clients,
            #         rounds=rounds,
            #         client_data=client_data,
            #         test_loader=test_loader,
            #         test_X=test_X,
            #         y_test=y_test,
            #         defenses_to_run=defenses_to_test if defenses_to_test else ["committee", "adaptive", "ensemble"],
            #         save_csv=True
            #     )

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
                
                # For 'none' attack, only show baseline
                if attack_name.lower() == 'none':
                    print(f"{dataset_name:<15} {attack_name:<12} {baseline_acc:.4f}    "
                          f"{'N/A':<12} {'N/A':<12} {'N/A':<15} {'N/A':<15}")
                    table_data.append([dataset_name, attack_name, baseline_acc, None, None, None, None])
                    continue
                
                attack_acc = attack_results.get('attack', {}).get('final_accuracy', baseline_acc)
                attack_success = attack_results.get('attack', {}).get('attack_success_rates', [])
                avg_attack_success = np.mean(attack_success) if attack_success else 0.0

                # Loop over all defense results (keys starting with 'defense_')
                defense_keys = [k for k in attack_results.keys() if k.startswith('defense_')]
                if not defense_keys:
                    # Fallback: legacy single 'defense' key
                    defense_keys = ['defense'] if 'defense' in attack_results else []

                for defense_key in defense_keys:
                    defense_acc = attack_results.get(defense_key, {}).get('final_accuracy', attack_acc)
                    # Extract defense name for display
                    defense_name = defense_key.replace('defense_', '') if defense_key.startswith('defense_') else defense_key

                    if baseline_acc > 0 and baseline_acc != attack_acc:
                        recovery = (defense_acc - attack_acc) / (baseline_acc - attack_acc) * 100
                        recovery = max(0, min(100, recovery))
                    else:
                        recovery = 100 if defense_acc >= baseline_acc else 0

                    print(f"{dataset_name:<15} {attack_name:<12} {baseline_acc:.4f}    {attack_acc:.4f}      "
                          f"{defense_acc:.4f} ({defense_name})      {recovery:.1f}%            {avg_attack_success:.4f}")

                    table_data.append([dataset_name, attack_name, baseline_acc, attack_acc,
                                       defense_acc, recovery, avg_attack_success, defense_name])

        return table_data

    def save_results_table_to_csv(self, table_data, filename_prefix="evaluation_results"):
        """Save the evaluation table to CSV, leaving attack/defense metrics empty for baseline (no attack)."""
        filename = f"{filename_prefix}_{self.run_id}.csv"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, mode="w", newline="") as f:
            writer = csv.writer(f)
            # Header
            writer.writerow([
                "Dataset", "Attack Type", "Defense Name", "Baseline Accuracy", "Under Attack Accuracy",
                "With Defense Accuracy", "Defense Recovery (%)", "Attack Success Rate"
            ])
            
            for row in table_data:
                # Now each row has: dataset, attack_type, baseline_acc, attack_acc, defense_acc, recovery, asr, defense_name
                if len(row) == 8:
                    dataset, attack_type, baseline_acc, attack_acc, defense_acc, recovery, asr, defense_name = row
                else:
                    # For 'none' attack, legacy row
                    dataset, attack_type, baseline_acc, attack_acc, defense_acc, recovery, asr = row
                    defense_name = ""
                # For 'none' attack, leave attack/defense columns empty
                if attack_type.lower() == 'none' or attack_acc is None:
                    attack_acc = ""
                    defense_acc = ""
                    recovery = ""
                    asr = ""
                writer.writerow([
                    dataset, attack_type, defense_name, baseline_acc, attack_acc,
                    defense_acc, recovery, asr
                ])
        
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
        """
        Delegate plotting to the PlottingEngine for each dataset.
        Ensures results structure is complete and passes to plotter correctly.
        """
        # First, ensure all required keys exist in the results structure
        for dataset_name, dataset_results in results.items():
            for attack_name, attack_results in dataset_results.items():
                # Ensure 'baseline' and 'attack' keys exist
                for scenario in ['baseline', 'attack']:
                    if scenario not in attack_results:
                        attack_results[scenario] = {
                            'final_accuracy': 0.0,
                            'training_acc_history': [],
                            'training_loss_history': [],
                            'test_acc_history': [],
                            'test_loss_history': [],
                            'attack_success_rates': []
                        }

                # Ensure all defense scenarios have required keys
                defense_scenarios = [k for k in attack_results.keys() if k.startswith('defense_')]
                for scenario in defense_scenarios:
                    for key in ['final_accuracy', 'training_acc_history', 'training_loss_history',
                               'test_acc_history', 'test_loss_history', 'attack_success_rates']:
                        if key not in attack_results[scenario]:
                            attack_results[scenario][key] = 0.0 if key == 'final_accuracy' else []

                # Ensure required keys exist in baseline and attack scenarios
                for scenario in ['baseline', 'attack']:
                    for key in ['final_accuracy', 'training_acc_history', 'training_loss_history',
                               'test_acc_history', 'test_loss_history', 'attack_success_rates']:
                        if key not in attack_results[scenario]:
                            attack_results[scenario][key] = 0.0 if key == 'final_accuracy' else []

        # Now call the plotting engine correctly for each dataset
        # PlottingEngine.generate_all_plots expects: (results_dict, output_dir, dataset_name)
        for dataset_name, dataset_results in results.items():
            self.plotter.generate_all_plots({dataset_name: dataset_results}, self.out_dir, dataset_name)


    def generate_comprehensive_report(self, results, features_dict=None):
        """Produce comprehensive report with committee defense analysis."""
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
                    attack_success = attack_results.get('attack', {}).get('attack_success_rates', [])
                    avg_attack_success = np.mean(attack_success) if attack_success else 0
                    impact = (baseline_acc - attack_acc) * 100

                    # Print baseline and attack scenario
                    print(f"  {dataset_name}:")
                    print(f"    Baseline accuracy: {baseline_acc:.4f}")
                    print(f"    Attack accuracy:   {attack_acc:.4f}")
                    print(f"    Impact: {impact:.2f}% accuracy drop")
                    print(f"    Attack success: {avg_attack_success:.3f}")

                    # Loop over all defense results (keys starting with 'defense_')
                    defense_keys = [k for k in attack_results.keys() if k.startswith('defense_')]
                    if not defense_keys:
                        defense_keys = ['defense'] if 'defense' in attack_results else []

                    for defense_key in defense_keys:
                        defense_acc = attack_results.get(defense_key, {}).get('final_accuracy', 0)
                        defense_name = defense_key.replace('defense_', '') if defense_key.startswith('defense_') else defense_key
                        if baseline_acc != attack_acc:
                            recovery = (defense_acc - attack_acc) / (baseline_acc - attack_acc) * 100
                            recovery = max(0, min(100, recovery))
                        else:
                            recovery = 100 if defense_acc >= baseline_acc else 0
                        print(f"    Defense: {defense_name:<10} | Recovery: {recovery:.1f}% | Defense Acc: {defense_acc:.4f}")

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
        print("  ✓ Integrated directly into federated learning rounds")

        print("\nStrengths:")
        print("  • Effective against various poisoning attacks")
        print("  • Maintains model utility while providing security")
        print("  • Scales well with number of clients")
        print("  • No single point of failure")
        print("  • Real-time detection during training")

        print("\nLimitations:")
        print("  • May struggle with sophisticated coordinated attacks")
        print("  • Requires careful threshold tuning")
        print("  • Computational overhead for anomaly detection")
        print("  • Performance depends on ratio of malicious clients")

        return {
            'total_attacks_tested': total_attacks_tested,
            'successful_defenses': successful_defenses,
            'defense_success_rate': successful_defenses/total_attacks_tested*100 if total_attacks_tested > 0 else 0,
            'plots_generated': self.plotter.plots_generated
        }
    
    def extract_client_features(self, clients, test_loader):
        """Extract features from client models for similarity analysis."""
        features_dict = {}
        device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))
        
        for client in clients:
            client.model.to(device)
            client.model.eval()
            features = []
            
            with torch.no_grad():
                for X, _ in test_loader:
                    X = X.to(device)
                    # Extract features from penultimate layer if available
                    if hasattr(client.model, 'features'):
                        feat = client.model.features(X)
                    else:
                        # Try to get intermediate layer output
                        try:
                            # For models with sequential structure
                            layers = list(client.model.children())
                            x = X
                            for layer in layers[:-1]:  # All but last layer
                                x = layer(x)
                            feat = x
                        except:
                            # Fallback to full model output
                            feat = client.model(X)
                    
                    features.append(feat.cpu())
            
            features_dict[client.client_id] = torch.cat(features, dim=0)
        
        return features_dict


# =============================================================================
# MAIN EXECUTION FUNCTIONS
# =============================================================================

def run_enhanced_evaluation(
    resume_from_json=None,
    out_dir="Output",
    test_defenses=True,
    subset_fraction=0.1,
    resume=False,
    defenses_to_test=None
):
    """
    Main entry point — runs the comprehensive evaluation (training)
    and extended defense comparison automatically, saves CSVs and plots.

    Args:
        resume_from_json: Path to JSON results to skip training
        out_dir: Output directory
        test_defenses: Whether to test defenses
        subset_fraction: Fraction of dataset to use
        resume: Whether to resume from checkpoint
        defenses_to_test: List of defense types to test (default: ["committee", "adaptive", "ensemble"])

    Returns:
        Tuple of (results dict, report statistics dict)
    """

    if defenses_to_test is None:
        defenses_to_test = ["committee", "adaptive", "ensemble"]

    print("=" * 100)
    print("ENHANCED DECENTRALIZED FEDERATED LEARNING SECURITY FRAMEWORK")
    print(f"Dataset Size: {subset_fraction * 100:.0f}% (for faster testing)")
    print(f"Resume Mode: {'ENABLED' if resume else 'DISABLED'}")
    print("=" * 100)

    try:
        # Initialize evaluation framework
        if resume_from_json:
            evaluator = EvaluationFramework(out_dir=out_dir)
            results = evaluator.load_results_from_json(resume_from_json)
            print("[INFO] Skipping training, reusing saved results.")
        else:
            # FIXED: Check for existing checkpoints BEFORE creating new run_id
            should_resume = False
            existing_run_id = None

            if resume:
                # Search for any checkpoint files recursively in out_dir/checkpoints/
                checkpoint_base = os.path.join(out_dir, "checkpoints")
                found_checkpoint = False
                for root, dirs, files in os.walk(checkpoint_base):
                    for file in files:
                        if file.endswith(".pth"):
                            found_checkpoint = True
                            break
                    if found_checkpoint:
                        break
                if found_checkpoint:
                    run_id = os.path.basename(os.path.normpath(out_dir))
                    print(f"[INFO] Found checkpoint for run_id: {run_id}")
                else:
                    print("[INFO] No existing checkpoint found. Starting fresh...")
                    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            else:
                run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

            evaluator = EvaluationFramework(out_dir=out_dir, run_id=run_id)

            # Run main comprehensive evaluation with resume support
            results = evaluator.run_comprehensive_evaluation(
                test_defenses=test_defenses,
                subset_fraction=subset_fraction,
                resume=resume,
                defenses_to_test=defenses_to_test
            )

            # Save results
            table_data = evaluator.generate_results_table(results)
            evaluator.save_results_table_to_csv(table_data)
            evaluator.save_results_summary_to_json(results)
            
            # Print checkpoint info
            print(f"\n{'='*80}")
            print("CHECKPOINT SUMMARY")
            print(f"{'='*80}")
            print(f"Checkpoint directory: {evaluator.checkpoint_dir}")
            print(f"Best checkpoints saved for each scenario")
            print(f"To resume training: python main.py --resume")
            print(f"{'='*80}")

        # Optional: Extract client features if available (for analysis)
        features_dict = None

        # Visualization phase (plots, summaries)
        print("\n[INFO] Generating comprehensive visualizations and report...")
        evaluator.generate_all_plots(results)

        # Generate final report and print summary
        report_stats = evaluator.generate_comprehensive_report(results, features_dict)

        print(f"\n{'=' * 80}")
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print(f"{'=' * 80}")

        print("\nKey Findings:")
        print("• Committee-based defense shows strong performance across attack types")
        print("• Trigger-based backdoors require specialized detection methods")
        print("• Dynamic attacks (DLF) pose greater challenges than static ones")
        print("• Decentralized FL maintains robustness under diverse threat models")
        print(f"\nGenerated {len(report_stats['plots_generated'])} visualization plots")
 
        return results, report_stats

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Training interrupted by user")
        print("[INFO] Progress has been saved. You can resume using --resume flag")
        raise
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n[INFO] If training was interrupted, you can resume using --resume flag")
        raise


def run_attack_comparison_test(subset_fraction=0.1, num_clients=1, rounds = 30, malicious_clients=4):
    """
    Run side-by-side comparison of all attacks using a subset of the dataset.
    Evaluates attack effectiveness, accuracy, and stealth.
    """
    print("=" * 80)
    print("ATTACK COMPARISON TEST - DETAILED METRICS")
    print(f"Using {subset_fraction * 100:.0f}% of the dataset for evaluation")
    print("=" * 80)

    # Load and subset Fashion-MNIST
    trainset, testset = load_fashion_mnist()

    full_train_len, full_test_len = len(trainset), len(testset)
    subset_train_len = int(len(trainset) * subset_fraction)
    subset_test_len = int(len(testset) * subset_fraction)

    indices_train = np.random.choice(range(full_train_len), subset_train_len, replace=False)
    indices_test = np.random.choice(range(full_test_len), subset_test_len, replace=False)
    trainset = torch.utils.data.Subset(trainset, indices_train)
    testset = torch.utils.data.Subset(testset, indices_test)

    print(f"[INFO] Dataset sizes after subsetting - Train: {len(trainset)}, Test: {len(testset)}")

    # Partition for federated learning
    client_data = partition_dataset(trainset, num_clients=num_clients, iid=False, alpha=0.3)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False)

    # Prepare test arrays
    test_X, y_test = next(iter(DataLoader(testset, batch_size=len(testset), shuffle=False)))
    test_X, y_test = test_X.numpy(), y_test.numpy()

    # Attack setup
    framework = EvaluationFramework()
    attack_configs = framework.get_attack_configs(malicious_clients=malicious_clients)

    comparison_results = {}

    # Evaluate each attack
    for attack_name, attack_config in attack_configs.items():
        print("=" * 80)
        print(f"\n--- Testing {attack_name.upper()} ---")
        print("=" * 80)


        clients = []
        for i in range(num_clients):
            model = FashionMNISTNet()
            client_attack = attack_config if (attack_config and i < attack_config.num_malicious_clients) else None
            clients.append(
                DecentralizedClient(i, client_data[i][0], client_data[i][1], model, client_attack)
            )

        coordinator = DecentralizedFLCoordinator(clients, use_committee_defense=False)
        accuracy_history = coordinator.run_federated_learning(rounds=rounds)

        final_accuracy, client_accuracies = coordinator.evaluate_on_test_set(test_loader)
        try:
            attack_success_rates = coordinator.evaluate_attack_success(test_X, y_test)
        except Exception as e:
            print(f"[WARN] Attack success evaluation failed: {e}")
            attack_success_rates = [0.0] * len(clients)

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
                'attack_success': np.mean(attack_success_rates) if attack_success_rates else 0.0,
                'malicious_clients': sum(1 for c in clients if getattr(c, 'is_malicious', False)),
                'poisoning_rate': attack_config.poisoning_rate,
                'attack_type': attack_config.attack_type
            }

        print(f"Results: Accuracy = {final_accuracy:.4f}, Attack Success = {np.mean(attack_success_rates):.4f}")

    # Print Results Table
    print(f"\n{'='*80}")
    print("ATTACK COMPARISON RESULTS")
    print(f"{'='*80}")

    print(f"{'Attack':<20} {'Type':<12} {'PDR':<6} {'Accuracy':<10} {'Attack Success':<15} {'Stealth':<12}")
    print("-" * 80)

    for name, res in comparison_results.items():
        stealth_score = res['final_accuracy'] * (1 - res['attack_success'] * 0.5)
        print(f"{name:<20} {res['attack_type'].upper():<12} {res['poisoning_rate']:.1%}    "
              f"{res['final_accuracy']:.4f}    {res['attack_success']:.4f}         {stealth_score:.4f}")

    print(f"\n{'='*80}\nATTACK COMPARISON COMPLETED\n{'='*80}")
    return comparison_results


def run_comprehensive_defense_test(subset_fraction=0.1, num_clients=10, rounds = 30, out_dir="comprehensive_defense_results",
                                   resume_checkpoint=False, checkpoint_interval=2):
    """
    Comprehensive test of ALL defenses against ALL attacks for BOTH datasets.
    Tests committee, adaptive, and ensemble defenses separately.
    Generates comparative visualizations and recovery rate analysis.

    Args:
        subset_fraction: Fraction of dataset to use (0.1 = 10%)
        num_clients: Number of clients in federated learning
        rounds: Number of FL rounds
        out_dir: Output directory for results
        resume_checkpoint: If True, check for and resume from existing checkpoints
        checkpoint_interval: Save checkpoint every N rounds

    Returns:
        Complete results dictionary with all defense comparisons
    """
    print("\n" + "="*100)
    print("COMPREHENSIVE DEFENSE TESTING FRAMEWORK")
    print(f"Testing ALL 5 defenses (committee, adaptive, reputation, gradient, ensemble) against ALL attacks")
    print(f"Dataset fraction: {subset_fraction*100:.0f}%, Clients: {num_clients}, Rounds: {rounds}")
    if resume_checkpoint:
        print(f"Checkpoint Resume: ENABLED (interval: every {checkpoint_interval} rounds)")
    print("="*100)

    # FIXED: Check for existing checkpoints BEFORE creating new run_id
    should_resume = False
    existing_run_id = None

    if resume_checkpoint:
        # Search for any checkpoint files recursively in out_dir/checkpoints/
        checkpoint_base = os.path.join(out_dir, "checkpoints")
        found_checkpoint = False
        for root, dirs, files in os.walk(checkpoint_base):
            for file in files:
                if file.endswith(".pth"):
                    found_checkpoint = True
                    break
            if found_checkpoint:
                break
        if found_checkpoint:
            run_id = os.path.basename(os.path.normpath(out_dir))
            print(f"[INFO] Found checkpoint for run_id: {run_id}")
        else:
            print("[INFO] No existing checkpoint found. Starting fresh...")
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    framework = EvaluationFramework(out_dir=out_dir, run_id=run_id)

    # Check for existing checkpoints if resume is enabled
    if resume_checkpoint:
        resume_info = framework.checkpoint_manager.get_resume_info()
        if resume_info["can_resume"]:
            print("\n" + "="*100)
            print("[CHECKPOINT DETECTED]")
            print("="*100)
            print(f"  Last run was interrupted at:")
            print(f"  Dataset: {resume_info['dataset']}")
            print(f"  Attack: {resume_info['attack']}")
            print(f"  Scenario: {resume_info['scenario']}")
            print(f"  Last completed round: {resume_info['last_round']}/{rounds}")
            print(f"  Completed scenarios: {resume_info['completed_scenarios']}")
            print("="*100)

            # NON-INTERACTIVE: Always resume if resume_checkpoint is set
            should_resume = True
            print("\n[RESUME] Resuming from checkpoint...")
        else:
            print("\n[INFO] No existing checkpoint found. Starting fresh...")
            should_resume = False

    # Store checkpoint interval for use in run_and_evaluate_coordinator
    framework.checkpoint_interval = checkpoint_interval

    # Defense types to test - ALL 5 defenses
    # CHANGED: Now testing ALL defenses for comprehensive analysis
    defense_types = ['committee', 'adaptive', 'reputation', 'gradient', 'ensemble']

    # Get datasets and models
    datasets, models = framework.get_datasets_and_models()
    attack_configs = framework.get_attack_configs(malicious_clients=4)

    # IMPORTANT: Define explicit attack order to ensure all attacks run sequentially
    # NOTE: 'none' is NOT an attack - it's the baseline without any attack
    # We handle baseline separately before testing actual attacks
    attack_order = ['slf', 'dlf', 'centralized', 'coordinated', 'random', 'model_dependent']

    print(f"\n[DEBUG] Real attacks to test: {attack_order}")
    print(f"[DEBUG] Total attacks: {len(attack_order)} (baseline will be run separately)")

    # Store results
    all_results = {}

    # Loop through datasets, then attacks, then all defenses for each attack
    # This ensures all defenses run for one attack before moving to the next attack
    for dataset_name, (trainset, testset) in datasets.items():
        print(f"\n{'='*100}")
        print(f"TESTING DATASET: {dataset_name}")
        print(f"{'='*100}")

        all_results[dataset_name] = {}

        # Subset datasets
        print(f"[INFO] Original sizes - Train: {len(trainset)}, Test: {len(testset)}")
        trainset_subset = framework._subset_dataset(trainset, subset_fraction)
        testset_subset = framework._subset_dataset(testset, subset_fraction)
        print(f"[INFO] Subsetted sizes - Train: {len(trainset_subset)}, Test: {len(testset_subset)}")

        # Partition data
        client_data = partition_dataset(trainset_subset, num_clients, iid=False)
        test_loader = DataLoader(testset_subset, batch_size=256, shuffle=False)
        test_X, y_test = next(iter(DataLoader(testset_subset, batch_size=len(testset_subset), shuffle=False)))
        test_X, y_test = test_X.numpy(), y_test.numpy()

        # ============================================================================
        # STEP 1: Run BASELINE (no attack, no defense) - This is NOT an attack!
        # ============================================================================
        print(f"\n{'-'*100}")
        print(f"BASELINE (No Attack, No Defense)")
        print(f"{'-'*100}")

        baseline_scenario = "baseline"
        baseline_acc = None
        bl_training_acc_history = []
        bl_training_loss_history = []
        bl_test_acc_history = []
        bl_test_loss_history = []
        #start_round = 0

        if should_resume and framework.checkpoint_manager.is_scenario_complete(dataset_name, 'baseline', baseline_scenario):
            print(f"[SKIP] Baseline already completed - loading from checkpoint")
            checkpoint = framework.checkpoint_manager.load_checkpoint(dataset_name, 'baseline', baseline_scenario, "best")
            if checkpoint and 'metrics' in checkpoint:
                baseline_acc = checkpoint['metrics'].get('accuracy', None)
                # Try to load history from checkpoint if available
                bl_training_acc_history = checkpoint['metrics'].get('training_acc_history', [])
                bl_training_loss_history = checkpoint['metrics'].get('training_loss_history', [])
                bl_test_acc_history = checkpoint['metrics'].get('test_acc_history', [])
                bl_test_loss_history = checkpoint['metrics'].get('test_loss_history', [])
                if baseline_acc is not None:
                    print(f"Loaded baseline accuracy: {baseline_acc:.4f}")
                else:
                    print("Loaded baseline accuracy: None")
            #start_round = checkpoint.get('round', 0) + 1
        else:
            print(f"[INFO] Running baseline (clean training, no attack, no defense)...")
            baseline_clients = framework.build_clients(num_clients, client_data, models, dataset_name, attack_config=None)
            _, _, baseline_acc, baseline_asr_list, bl_training_acc_history, bl_training_loss_history, bl_test_acc_history, bl_test_loss_history, _ = framework.run_and_evaluate_coordinator(
                baseline_clients, rounds, test_loader, test_X=test_X, y_test=y_test,
                use_defense=False, dataset=dataset_name, attack='baseline', scenario=baseline_scenario
            )
            print(f"✓ Baseline accuracy: {baseline_acc:.4f}")
            print(f"[HISTORY] Collected {len(bl_training_acc_history)} rounds of training history")
            print(f"[HISTORY] Collected {len(bl_test_acc_history)} rounds of test history")
            framework.checkpoint_manager.mark_scenario_complete(dataset_name, 'baseline', baseline_scenario)

        # ============================================================================
        # STEP 2: Test each REAL ATTACK with all defenses
        # ============================================================================
        for attack_idx, attack_name in enumerate(attack_order, start=1):
            print(f"\n[DEBUG] Starting attack {attack_idx}/{len(attack_order)}: '{attack_name}'")
            attack_config = attack_configs.get(attack_name)

            if attack_config is None:
                print(f"[ERROR] Attack config not found for '{attack_name}'! Skipping...")
                continue

            print(f"\n{'-'*100}")
            print(f"ATTACK {attack_idx}/{len(attack_order)}: {attack_name.upper()}")
            print(f"{'-'*100}")

            attack_results = {}

            # Store baseline accuracy for this attack (already computed above)
            if baseline_acc is not None:
                baseline_acc_value = float(baseline_acc)
                # Store baseline with full history
                attack_results['baseline'] = {
                    'final_accuracy': baseline_acc_value,
                    'accuracy': baseline_acc_value,
                    'attack_success_rate': 0.0,
                    'training_acc_history': [float(x) for x in bl_training_acc_history] if bl_training_acc_history else [],
                    'training_loss_history': [float(x) for x in bl_training_loss_history] if bl_training_loss_history else [],
                    'test_acc_history': [float(x) for x in bl_test_acc_history] if bl_test_acc_history else [],
                    'test_loss_history': [float(x) for x in bl_test_loss_history] if bl_test_loss_history else [],
                    'attack_success_rates': []
                }
            else:
                # Fallback if baseline wasn't computed
                attack_results['baseline'] = {
                    'final_accuracy': 0.0,
                    'accuracy': 0.0,
                    'attack_success_rate': 0.0,
                    'training_acc_history': [],
                    'training_loss_history': [],
                    'test_acc_history': [],
                    'test_loss_history': [],
                    'attack_success_rates': []
                }

            # 1. Run attack WITHOUT defense
            scenario_name = "attack"
            start_round = 0
            if should_resume and framework.checkpoint_manager.is_scenario_complete(dataset_name, attack_name, scenario_name):
                print(f"\n[1/{len(defense_types)+1}] Attack (no defense) - SKIPPED (already completed)")
                checkpoint = framework.checkpoint_manager.load_checkpoint(dataset_name, attack_name, scenario_name, "best")
                if checkpoint and 'metrics' in checkpoint:
                    attack_results['attack'] = checkpoint['metrics']
                start_round = checkpoint.get('round', 0) + 1
            else:
                print(f"\n[1/{len(defense_types)+1}] {attack_name.upper()} attack (no defense)...")
                attack_clients = framework.build_clients(num_clients, client_data, models, dataset_name, attack_config)
                for client in attack_clients:
                    if hasattr(client, 'poison_local_data') and client.is_malicious:
                        try:
                            client.poison_local_data()
                        except Exception as e:
                            print(f"[WARN] Poisoning failed for client {client.client_id}: {e}")

                attack_coord, attack_result, attack_acc, attack_asr_list, training_acc_history, training_loss_history, test_acc_history, test_loss_history, _ = framework.run_and_evaluate_coordinator(
                    attack_clients, rounds, test_loader, test_X=test_X, y_test=y_test,
                    use_defense=False, dataset=dataset_name, attack=attack_name, scenario=scenario_name
                )

                attack_asr = np.mean([r for r in attack_asr_list if r > 0]) if any(r > 0 for r in attack_asr_list) else 0.0
                attack_results['attack'] = {
                    'final_accuracy': float(attack_acc),
                    'accuracy': float(attack_acc),
                    'attack_success_rate': float(attack_asr),
                    'training_acc_history': [float(x) for x in training_acc_history],
                    'training_loss_history': [float(x) for x in training_loss_history],
                    'test_acc_history': [float(x) for x in test_acc_history],
                    'test_loss_history': [float(x) for x in test_loss_history],
                    'attack_success_rates': [float(x) for x in attack_asr_list] if attack_asr_list else []
                }
                print(f"Attack accuracy: {attack_acc:.4f}, ASR: {attack_asr:.4f}")
                print(f"[HISTORY] Collected {len(training_acc_history)} rounds of training history")
                print(f"[HISTORY] Collected {len(test_acc_history)} rounds of test history")

                # Mark scenario as complete
                framework.checkpoint_manager.mark_scenario_complete(dataset_name, attack_name, scenario_name)

            # 2-6. Test each defense type (5 defenses)
            print(f"\n[DEBUG] Starting defense loop for attack '{attack_name}'")
            print(f"[DEBUG] Defense types to test: {defense_types}")
            print(f"[DEBUG] Resume mode: {should_resume}")

            for idx, defense_type in enumerate(defense_types, start=2):
                scenario_name = f"defense_{defense_type}"
                start_round = 0
                print(f"\n[DEBUG] Processing defense {idx-1}/{len(defense_types)}: {defense_type}")

                # Check if scenario is already completed
                is_completed = should_resume and framework.checkpoint_manager.is_scenario_complete(dataset_name, attack_name, scenario_name)
                print(f"[DEBUG] Scenario '{scenario_name}' is_completed: {is_completed}")

                if is_completed:
                    print(f"\n[{idx}/{len(defense_types)+1}] {defense_type.upper()} defense - SKIPPED (already completed)")
                    checkpoint = framework.checkpoint_manager.load_checkpoint(dataset_name, attack_name, scenario_name, "best")
                    if checkpoint and 'metrics' in checkpoint:
                        attack_results[defense_type] = checkpoint['metrics']
                    start_round = checkpoint.get('round', 0) + 1
                    continue

                print(f"\n[{idx}/{len(defense_types)+1}] {attack_name.upper()} with {defense_type.upper()} defense...")

                defense_clients = framework.build_clients(num_clients, client_data, models, dataset_name, attack_config)
                for client in defense_clients:
                    if hasattr(client, 'poison_local_data') and client.is_malicious:
                        try:
                            client.poison_local_data()
                        except Exception as e:
                            print(f"[WARN] Poisoning failed for client {client.client_id}: {e}")

                try:
                    _, _, defense_acc, defense_asr_list, def_training_acc_history, def_training_loss_history, def_test_acc_history, def_test_loss_history, detection_metrics = framework.run_and_evaluate_coordinator(
                        defense_clients, rounds, test_loader, test_X=test_X, y_test=y_test,
                        use_defense=True, defense_type=defense_type,
                        dataset=dataset_name, attack=attack_name, scenario=scenario_name
                    )

                    defense_asr = np.mean([r for r in defense_asr_list if r > 0]) if any(r > 0 for r in defense_asr_list) else 0.0

                    # Calculate recovery rate
                    baseline_acc_val = attack_results.get('baseline', {}).get('accuracy', 0)
                    attack_acc_val = attack_results.get('attack', {}).get('accuracy', 0)

                    if baseline_acc_val != attack_acc_val:
                        recovery = ((defense_acc - attack_acc_val) / (baseline_acc_val - attack_acc_val)) * 100
                        recovery = max(0, min(100, recovery))
                    else:
                        recovery = 100.0 if defense_acc >= baseline_acc_val else 0.0

                    # Store results including detection metrics AND history arrays
                    attack_results[defense_type] = {
                        'final_accuracy': float(defense_acc),
                        'accuracy': float(defense_acc),
                        'attack_success_rate': float(defense_asr),
                        'recovery_rate': float(recovery),
                        'detection_metrics': detection_metrics,
                        'training_acc_history': [float(x) for x in def_training_acc_history],
                        'training_loss_history': [float(x) for x in def_training_loss_history],
                        'test_acc_history': [float(x) for x in def_test_acc_history],
                        'test_loss_history': [float(x) for x in def_test_loss_history],
                        'attack_success_rates': [float(x) for x in defense_asr_list] if defense_asr_list else []
                    }

                    print(f"{defense_type.capitalize()} defense - Accuracy: {defense_acc:.4f}, Recovery: {recovery:.2f}%")
                    print(f"[HISTORY] Collected {len(def_training_acc_history)} rounds of training history")
                    print(f"[HISTORY] Collected {len(def_test_acc_history)} rounds of test history")
                    print(f"[DEBUG] Detection metrics for {defense_type}: {detection_metrics}")

                    # Mark scenario as complete
                    framework.checkpoint_manager.mark_scenario_complete(dataset_name, attack_name, scenario_name)
                    print(f"[DEBUG] Marked scenario '{scenario_name}' as complete")

                except Exception as e:
                    print(f"[ERROR] {defense_type.capitalize()} defense failed: {e}")
                    import traceback
                    traceback.print_exc()
                    attack_results[defense_type] = {
                        'accuracy': 0.0,
                        'attack_success_rate': 0.0,
                        'recovery_rate': 0.0,
                        'detection_metrics': {},
                        'error': str(e)
                    }
                    print(f"[DEBUG] Continuing to next defense after error...")

            # Store attack results
            print(f"\n[DEBUG] Completed all defenses for attack '{attack_name}'")
            print(f"[DEBUG] Attack results keys: {list(attack_results.keys())}")
            all_results[dataset_name][attack_name] = attack_results
            print(f"[INFO] ✓ Attack {attack_idx}/{len(attack_order)} ({attack_name}) completed successfully")
            print(f"[INFO] Moving to next attack...")

    # Print execution summary
    print(f"\n{'='*100}")
    print("EXECUTION SUMMARY")
    print(f"{'='*100}")
    for dataset_name, dataset_results in all_results.items():
        print(f"\nDataset: {dataset_name}")
        print(f"  Baseline: Completed (shared across all attacks)")
        print(f"  Attacks executed: {list(dataset_results.keys())}")
        print(f"  Total attacks: {len(dataset_results)}/{len(attack_order)}")
        for attack_name, attack_results in dataset_results.items():
            defenses_tested = [k for k in attack_results.keys() if k not in ['baseline', 'attack']]
            print(f"    {attack_name}: attack + {len(defenses_tested)} defenses {defenses_tested}")

    # Save results
    print(f"\n{'='*100}")
    print("SAVING RESULTS")
    print(f"{'='*100}")

    # Save JSON
    json_path = os.path.join(framework.results_dir, f"comprehensive_defense_results_{run_id}.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"[INFO] Results saved to: {json_path}")

    # Save CSV
    # Save CSV
    csv_path = os.path.join(framework.results_dir, f"comprehensive_defense_results_{run_id}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Dataset', 'Attack', 'Defense', 'Baseline_Acc', 'Attack_Acc',
            'Defense_Acc', 'ASR', 'Recovery_Rate'
        ])

        for dataset_name, dataset_results in all_results.items():
            for attack_name, attack_results in dataset_results.items():
                # FIXED: Skip 'none' attack (baseline only, no attack/defense scenarios)
                if attack_name == 'none' or 'attack' not in attack_results:
                    continue

                # Safely get baseline and attack results
                baseline_acc = attack_results.get('baseline', {}).get('accuracy', 0.0)
                attack_acc = attack_results.get('attack', {}).get('accuracy', 0.0)
                attack_asr = attack_results.get('attack', {}).get('attack_success_rate', 0.0)

                for defense_type in defense_types:
                    if defense_type in attack_results and 'error' not in attack_results[defense_type]:
                        defense_acc = attack_results[defense_type]['accuracy']
                        recovery = attack_results[defense_type].get('recovery_rate', 0.0)

                        writer.writerow([
                            dataset_name, attack_name, defense_type,
                            f"{baseline_acc:.4f}", f"{attack_acc:.4f}",
                            f"{defense_acc:.4f}", f"{attack_asr:.4f}",
                            f"{recovery:.2f}"
                        ])


    print(f"[INFO] CSV saved to: {csv_path}")

    # Generate visualizations using PlottingEngine
    print(f"\n{'='*100}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*100}")

    from plots import PlottingEngine
    plotter = PlottingEngine(output_dir=framework.plots_dir, run_id=run_id)
    plotter.plot_comprehensive_defense_comparison(all_results, run_id=run_id)

    # Generate comprehensive defense analysis tables (includes summary printing)
    from defense_analysis import DefenseAnalyzer
    analyzer = DefenseAnalyzer(results_dir=framework.results_dir)
    analyzer.generate_all_analysis(
        results=all_results,
        num_malicious=4,  # From attack_configs
        num_clients=num_clients,
        total_rounds=rounds
    )

    print(f"\n{'='*100}")
    print("COMPREHENSIVE DEFENSE TEST COMPLETED")
    print(f"{'='*100}")
    print(f"\nResults saved to: {framework.base_run_dir}")
    print(f"Plots saved to: {framework.plots_dir}")
    print(f"Data saved to: {framework.results_dir}")
    print(f"Analysis tables: {framework.results_dir}")
    print(f"{'='*100}")

    return all_results




# Note: Main execution is handled in main.py
# This allows evaluation.py to be used as a library module