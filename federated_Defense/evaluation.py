import os
import copy
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from models import FEMNISTNet, ShakespeareNet, Sentiment140Net
from datasets import load_femnist, load_shakespeare, load_sentiment140, partition_dataset
from client import DecentralizedClient
from coordinator import DecentralizedFLCoordinator

DATASETS = ['FEMNIST', 'Shakespeare', 'Sentiment140']
MODELS = {
    'FEMNIST': FEMNISTNet,
    'Shakespeare': ShakespeareNet,
    'Sentiment140': Sentiment140Net,
}
ATTACKS = ['gradient_scaling', 'same_value', 'back_gradient']

DATASET_ROUNDS = {'FEMNIST': 600, 'Shakespeare': 500, 'Sentiment140': 1000}
DATASET_LR = {'FEMNIST': 0.001, 'Shakespeare': 0.001, 'Sentiment140': 0.005}
DATASET_BATCH_SIZE = {'FEMNIST': 32, 'Shakespeare': 32, 'Sentiment140': 32}

class EvaluationFramework:
    """
    Framework to run evaluations of decentralized FL under various attacks and committee-based defense.
    """
    def __init__(self, out_dir="Output", run_id=None):
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        project_root = os.path.dirname(os.path.abspath(__file__))
        self.base_run_dir = os.path.join(project_root, out_dir, self.run_id)
        self.results_dir = os.path.join(self.base_run_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)
        self.results = {}

    def get_datasets_and_models(self, dataset_name=None):
        _loaders = {
            'FEMNIST': load_femnist,
            'Shakespeare': load_shakespeare,
            'Sentiment140': load_sentiment140,
        }
        models = {
            'FEMNIST': lambda: FEMNISTNet(),
            'Shakespeare': lambda: ShakespeareNet(),
            'Sentiment140': lambda: Sentiment140Net(),
        }
        names_to_load = [dataset_name] if dataset_name else _loaders.keys()
        datasets = {name: _loaders[name]() for name in names_to_load}
        return datasets, models

    def build_clients(self, num_clients, client_data, models_map, dataset_name, attack_config=None, fixed_malicious_ids=None):
        malicious_ids = set(fixed_malicious_ids) if fixed_malicious_ids is not None else set()
        clients = []
        _reference_state = models_map[dataset_name]().state_dict()
        for i in range(num_clients):
            model = models_map[dataset_name]()
            #for name, param in model.named_parameters():
                #print(name, param.shape)
            model.load_state_dict(_reference_state)
            if i in malicious_ids and attack_config is not None:
                client = DecentralizedClient(i, client_data[i][0], client_data[i][1], model, attack_config)
                client.is_malicious = True
            else:
                client = DecentralizedClient(i, client_data[i][0], client_data[i][1], model, None)
                client.is_malicious = False
            clients.append(client)
        return clients

    def _subset_dataset(self, dataset, fraction=0.25):
        if fraction >= 1.0:
            return dataset
        subset_size = int(len(dataset) * fraction)
        indices = torch.randperm(len(dataset))[:subset_size]
        subset = torch.utils.data.Subset(dataset, indices)
        return subset

# ---------------------------------------------------------------------------
# Baseline Evaluation (NO attack, NO poisoning)
# ---------------------------------------------------------------------------
def run_baseline_evaluation(
    framework: EvaluationFramework,
    dataset_name: str,
    num_clients: int,
    rounds: int,
    clients_per_round: int,
    aggregation: str = "fedavg",
    alpha: float = 1.0,
    dataset_fraction: float = 1.0,
    scenario_name: str = "baseline",
    aggregation_participation_frac: float = None,
    committee_size_frac: float = 0.40,
    scoring_mode: str = 'distance_only',
    selection_strategy: int = 1,
    lr: float = None,
    **defense_kwargs
):
    import math
    # Get datasets and models
    datasets_and_loaders, models_map = framework.get_datasets_and_models(dataset_name)
    if dataset_name not in datasets_and_loaders:
        print(f"[ERROR] Dataset {dataset_name} not found")
        return 0.0, [], []
    train_dataset, test_dataset = datasets_and_loaders[dataset_name]
    if dataset_fraction < 1.0:
        train_dataset = framework._subset_dataset(train_dataset, dataset_fraction)
        test_dataset = framework._subset_dataset(test_dataset, dataset_fraction)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
    client_data = partition_dataset(train_dataset, num_clients, iid=False, alpha=alpha)
    clients = framework.build_clients(
        num_clients=num_clients,
        client_data=client_data,
        models_map=models_map,
        dataset_name=dataset_name,
        attack_config=None
    )
    _COMMITTEE_SCHEMES = {'cmfl', 'cmfl_ii', 'cd_cfl', 'cdcfl_i', 'cdcfl_ii'}
    if lr is None:
        lr = DATASET_LR.get(dataset_name, 0.001)
    _bs = DATASET_BATCH_SIZE.get(dataset_name, 32)
    if aggregation.lower() in _COMMITTEE_SCHEMES:
        if aggregation.lower() == 'cmfl_ii':
            chosen_defense = 'cmfl'
            selection_strategy = 2
        elif aggregation.lower() == 'cdcfl_i':
            chosen_defense = 'cdcfl_i'
        elif aggregation.lower() in ('cdcfl_ii', 'cd_cfl'):
            chosen_defense = 'cdcfl_ii'
        else:
            chosen_defense = aggregation.lower()
        base_agg = 'multi_krum' if chosen_defense in ('cdcfl_ii', 'cdcfl_i') else 'weighted_avg'
        committee_size   = max(2, int(clients_per_round * committee_size_frac))
        training_clients = max(4, clients_per_round - committee_size)
        coordinator = DecentralizedFLCoordinator(
            clients,
            use_defense=True,
            defense_type=chosen_defense,
            committee_size=committee_size,
            training_clients_per_round=training_clients,
            aggregation_method=base_agg,
            use_anomaly_detection=False,
            clients_per_round=clients_per_round,
            aggregation_participation_frac=aggregation_participation_frac,
            scoring_mode=scoring_mode,
            selection_strategy=selection_strategy,
            lr=lr, batch_size=_bs,
            **defense_kwargs
        )
    else:
        coordinator = DecentralizedFLCoordinator(
            clients,
            use_defense=False,
            defense_type=None,
            aggregation_method='fedavg',
            clients_per_round=clients_per_round,
            lr=lr, batch_size=_bs
        )

    # if aggregation.lower() in _COMMITTEE_SCHEMES:
    #     if aggregation.lower() == 'cmfl_ii':
    #         chosen_defense = 'cmfl'
    #         selection_strategy = 2
    #     elif aggregation.lower() == 'cdcfl_i':
    #         chosen_defense = 'cdcfl_i'
    #     elif aggregation.lower() in ('cdcfl_ii', 'cd_cfl'):
    #         chosen_defense = 'cdcfl_ii'
    #     else:
    #         chosen_defense = aggregation.lower()
    #     base_agg = 'multi_krum' if chosen_defense in ('cdcfl_ii', 'cdcfl_i') else 'weighted_avg'
    #     committee_size   = max(2, int(clients_per_round * committee_size_frac))
    #     training_clients = max(4, clients_per_round - committee_size)
    #     coordinator = DecentralizedFLCoordinator(
    #         clients,
    #         use_defense=True,  # <-- ENABLE defense logic for robust methods, even in baseline!
    #         defense_type=chosen_defense,
    #         committee_size=committee_size,
    #         training_clients_per_round=training_clients,
    #         aggregation_method=base_agg,
    #         use_anomaly_detection=False,
    #         clients_per_round=clients_per_round,
    #         aggregation_participation_frac=aggregation_participation_frac,
    #         scoring_mode=scoring_mode,
    #         selection_strategy=selection_strategy,
    #         lr=lr, batch_size=_bs,
    #         **defense_kwargs
    #     )
    # else:
    #     coordinator = DecentralizedFLCoordinator(
    #         clients,
    #         use_defense=False,
    #         defense_type=None,
    #         aggregation_method=aggregation,
    #         clients_per_round=clients_per_round,
    #         lr=lr, batch_size=_bs
    #     )

    test_acc_history = []
    test_loss_history = []
    eval_interval = max(1, rounds // 20) if rounds > 20 else 1
    for round_num in range(1, rounds + 1):
        coordinator.run_federated_learning(
            rounds=1, aggregation_method=aggregation,
            test_loader=test_loader,
            round_offset=round_num - 1, total_rounds=rounds
        )
        if round_num % eval_interval == 0 or round_num == rounds:
            try:
                round_test_acc, round_test_loss = coordinator.evaluate_on_test_set(test_loader)
                if math.isnan(round_test_acc) or math.isinf(round_test_acc):
                    round_test_acc = 1e10
                if math.isnan(round_test_loss) or math.isinf(round_test_loss):
                    round_test_loss = 1e10
                test_acc_history.append(float(round_test_acc))
                test_loss_history.append(float(round_test_loss))
            except Exception as e:
                print(f"[WARN] Test evaluation failed at round {round_num}: {e}")
                test_acc_history.append(test_acc_history[-1] if test_acc_history else 0.0)
                test_loss_history.append(test_loss_history[-1] if test_loss_history else float('inf'))
        else:
            test_acc_history.append(test_acc_history[-1] if test_acc_history else 0.0)
            test_loss_history.append(test_loss_history[-1] if test_loss_history else float('inf'))
    final_accuracy = test_acc_history[-1] if test_acc_history else 1e10
    if math.isnan(final_accuracy) or math.isinf(final_accuracy):
        final_accuracy = 1e10
    return final_accuracy, test_acc_history, test_loss_history

# ---------------------------------------------------------------------------
# Attack Scenario Creation
# ---------------------------------------------------------------------------
def create_attack_scenario(
    framework: EvaluationFramework,
    dataset_name: str,
    attack_type: str,
    poison_percentage: float,
    num_clients: int,
    malicious_clients: int,
    alpha: float = 1.0,
    dataset_fraction: float = 1.0
):
    import numpy as np
    from attacks.base import AttackConfig
    datasets_and_loaders, models_map = framework.get_datasets_and_models(dataset_name)
    if dataset_name not in datasets_and_loaders:
        print(f"[ERROR] Dataset {dataset_name} not found")
        return None, None, None, None
    train_dataset, test_dataset = datasets_and_loaders[dataset_name]
    if dataset_fraction < 1.0:
        train_dataset = framework._subset_dataset(train_dataset, dataset_fraction)
        test_dataset = framework._subset_dataset(test_dataset, dataset_fraction)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
    test_X_list = []
    y_test_list = []
    for X_batch, y_batch in test_loader:
        test_X_list.append(X_batch.numpy())
        y_test_list.append(y_batch.numpy())
    test_X = np.concatenate(test_X_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    # AttackConfig logic (simplified for your attacks)
    if attack_type.lower() == 'gradient_scaling':
        attack_config = AttackConfig(
            attack_type=attack_type,
            num_malicious_clients=malicious_clients,
            data_poisoning_rate=0.0,
            epsilon=0.5
        )
    elif attack_type.lower() in ['same_value', 'back_gradient']:
        attack_config = AttackConfig(
            attack_type=attack_type,
            num_malicious_clients=malicious_clients,
            data_poisoning_rate=0.0
        )
    else:
        attack_config = AttackConfig(
            attack_type=attack_type,
            num_malicious_clients=malicious_clients,
            data_poisoning_rate=poison_percentage
        )
    client_data = partition_dataset(train_dataset, num_clients, iid=False, alpha=alpha)
    clients = framework.build_clients(
        num_clients=num_clients,
        client_data=client_data,
        models_map=models_map,
        dataset_name=dataset_name,
        attack_config=attack_config
    )
    return clients, test_loader, test_X, y_test

# ---------------------------------------------------------------------------
# Test Defense on Scenario
# ---------------------------------------------------------------------------
def test_defense_on_scenario(
    clients_list,
    test_loader,
    test_X,
    y_test,
    rounds: int,
    aggregation: str = "fedavg",
    defense_type: str = None,
    clients_per_round: Optional[int] = None,
    framework: EvaluationFramework = None,
    dataset_name: str = None,
    attack_type: str = None,
    scenario_name: str = None,
    aggregation_participation_frac: float = None,
    committee_size_frac: float = 0.40,
    scoring_mode: str = 'distance_only',
    selection_strategy: int = 1,
    lr: float = 0.01,
    batch_size: int = 32,
    **defense_kwargs
):
    import math
    import numpy as np
    _COMMITTEE_SCHEMES = {'cmfl', 'cmfl_ii', 'cd_cfl', 'cdcfl_i', 'cdcfl_ii'}
    if hasattr(clients_list[0], '_snapshot_model_state'):
        for client in clients_list:
            client.reset_for_new_run()
        clients_copy = clients_list
    else:
        import copy
        clients_copy = copy.deepcopy(clients_list)
    if aggregation.lower() in _COMMITTEE_SCHEMES:
        if aggregation.lower() == 'cmfl_ii':
            chosen_defense = 'cmfl'
            selection_strategy = 2
        elif aggregation.lower() == 'cdcfl_i':
            chosen_defense = 'cdcfl_i'
        elif aggregation.lower() in ('cdcfl_ii', 'cd_cfl'):
            chosen_defense = 'cdcfl_ii'
        else:
            chosen_defense = aggregation.lower()
        base_agg = 'multi_krum' if chosen_defense in ('cdcfl_ii', 'cdcfl_i') else 'weighted_avg'
        if clients_per_round is not None:
            committee_size   = max(2, int(clients_per_round * committee_size_frac))
            training_clients = max(4, clients_per_round - committee_size)
        else:
            committee_size   = None
            training_clients = None
        _enable_verification = defense_kwargs.pop('enable_verification', False)
        _verification_kwargs = defense_kwargs.pop('verification_kwargs', None)
        coordinator = DecentralizedFLCoordinator(
            clients_copy,
            use_defense=True,
            defense_type=chosen_defense,
            committee_size=committee_size,
            training_clients_per_round=training_clients,
            aggregation_method=base_agg,
            use_anomaly_detection=True,
            clients_per_round=clients_per_round,
            aggregation_participation_frac=aggregation_participation_frac,
            scoring_mode=scoring_mode,
            selection_strategy=selection_strategy,
            lr=lr, batch_size=batch_size,
            enable_verification=_enable_verification,
            verification_kwargs=_verification_kwargs,
            **defense_kwargs
        )
    else:
        coordinator = DecentralizedFLCoordinator(
            clients_copy,
            use_defense=False,
            defense_type=None,
            aggregation_method=aggregation,
            clients_per_round=clients_per_round,
            lr=lr, batch_size=batch_size
        )
    test_acc_history = []
    test_loss_history = []
    eval_interval = max(1, rounds // 20) if rounds > 20 else 1
    for round_num in range(1, rounds + 1):
        coordinator.run_federated_learning(
            rounds=1,
            aggregation_method=aggregation,
            test_loader=test_loader,
            round_offset=round_num - 1,
            total_rounds=rounds
        )
        if round_num % eval_interval == 0 or round_num == rounds:
            try:
                round_test_acc, round_test_loss = coordinator.evaluate_on_test_set(test_loader)
                if math.isnan(round_test_acc) or math.isinf(round_test_acc):
                    round_test_acc = 1e10
                if math.isnan(round_test_loss) or math.isinf(round_test_loss):
                    round_test_loss = 1e10
                test_acc_history.append(float(round_test_acc))
                test_loss_history.append(float(round_test_loss))
            except Exception as e:
                print(f"[WARN] Test evaluation failed at round {round_num}: {e}")
                test_acc_history.append(test_acc_history[-1] if test_acc_history else 0.0)
                test_loss_history.append(test_loss_history[-1] if test_loss_history else float('inf'))
        else:
            test_acc_history.append(test_acc_history[-1] if test_acc_history else 0.0)
            test_loss_history.append(test_loss_history[-1] if test_loss_history else float('inf'))
    final_accuracy = test_acc_history[-1] if test_acc_history else 1e10
    if math.isnan(final_accuracy) or math.isinf(final_accuracy):
        final_accuracy = 1e10
    attack_success_rate = 0.0
    if test_X is not None and y_test is not None:
        try:
            asr_results = coordinator.evaluate_attack_success(test_X, y_test)
            if isinstance(asr_results, dict):
                attack_success_rate = asr_results.get('attack_success_rate', 0.0)
            elif isinstance(asr_results, (list, tuple, np.ndarray)):
                asr_values = [
                    float(asr) for asr in asr_results
                    if asr > 0 and not math.isnan(asr) and not math.isinf(asr)
                ]
                attack_success_rate = np.mean(asr_values) if len(asr_values) > 0 else 0.0
            elif isinstance(asr_results, (int, float)):
                attack_success_rate = float(asr_results)
            if math.isnan(attack_success_rate) or math.isinf(attack_success_rate):
                attack_success_rate = 1e10
        except Exception as e:
            print(f"[WARN] ASR calculation failed: {e}")
            attack_success_rate = 1e10
    detection_metrics = {}
    if coordinator.use_defense:
        try:
            detection_metrics = coordinator.get_committee_metrics()
            if detection_metrics:
                for key, value in detection_metrics.items():
                    if isinstance(value, (int, float)):
                        if math.isnan(value) or math.isinf(value):
                            detection_metrics[key] = 1e10
                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (int, float)):
                                if math.isnan(sub_value) or math.isinf(sub_value):
                                    value[sub_key] = 1e10
        except Exception as e:
            print(f"[WARN] Failed to get detection metrics: {e}")
            detection_metrics = {}
        try:
            layer_metrics = coordinator.get_defense_layer_metrics()
            if layer_metrics:
                detection_metrics.update(layer_metrics)
        except Exception:
            pass
    return final_accuracy, attack_success_rate, detection_metrics, test_acc_history, test_loss_history

def run_and_evaluate_coordinator(
        self, clients, rounds, test_loader,
        test_X=None, y_test=None, use_defense=False, defense_type='cmfl',
        start_round=0, dataset="", attack="", scenario="", aggregation_method='fedavg'
    ):
        if use_defense:
            coordinator = DecentralizedFLCoordinator(
                clients,
                use_defense=True,
                defense_type=defense_type,
                aggregation_method=aggregation_method
            )
        else:
            coordinator = DecentralizedFLCoordinator(
                clients,
                use_defense=False,
                defense_type=None,
                aggregation_method=aggregation_method
            )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_acc_history = []
        test_loss_history = []
        for round_num in range(1, rounds + 1):
            coordinator.run_federated_learning(rounds=1, round_offset=round_num - 1, total_rounds=rounds, test_loader=test_loader)
            acc, loss = self._evaluate_test_acc_and_loss(clients[0].model, test_loader, device)
            test_acc_history.append(float(acc))
            test_loss_history.append(float(loss))
        final_test_acc = test_acc_history[-1] if test_acc_history else 0.0
        attack_success_rates = []
        detection_metrics = {}
        timing_data = {}
        return (
            coordinator, [], final_test_acc, attack_success_rates,
            [], [], test_acc_history, test_loss_history, detection_metrics, timing_data
        )

def _evaluate_test_acc_and_loss(self, model, test_loader, device=None):
        device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        model.to(device)
        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        correct = 0
        total_samples = 0
        total_loss = 0.0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total_loss += criterion(logits, y).item() * y.size(0)
                total_samples += y.size(0)
        acc = correct / total_samples if total_samples > 0 else 0.0
        loss = total_loss / total_samples if total_samples > 0 else 0.0
        return acc, loss

def run_comprehensive_evaluation(self, test_defenses=True, subset_fraction=0.25, defenses_to_test=None):
        num_clients = 100
        rounds = 5
        malicious_clients = 40
        datasets, models = self.get_datasets_and_models()
        results = {}
        for dataset_name, (trainset, testset) in datasets.items():
            trainset = self._subset_dataset(trainset, subset_fraction)
            testset = self._subset_dataset(testset, subset_fraction)
            client_data = partition_dataset(trainset, num_clients, iid=True)
            _pin = torch.cuda.is_available()
            test_loader = DataLoader(testset, batch_size=256, shuffle=False, pin_memory=_pin, num_workers=2)
            test_X, y_test = next(iter(DataLoader(testset, batch_size=len(testset), shuffle=False)))
            test_X, y_test = test_X.numpy(), y_test.numpy()
            fixed_malicious_ids = set(random.sample(range(num_clients), malicious_clients))
            baseline_clients = self.build_clients(num_clients, client_data, models, dataset_name, attack_config=None, fixed_malicious_ids=fixed_malicious_ids)
            _, _, baseline_acc, _, _, _, bl_test_acc, bl_test_loss, _, _ = self.run_and_evaluate_coordinator(
                baseline_clients, rounds, test_loader, test_X=test_X, y_test=y_test,
                use_defense=False, dataset=dataset_name, attack='baseline', scenario='baseline'
            )
            dataset_results = {
                'baseline': {
                    'final_accuracy': float(baseline_acc),
                    'test_acc_history': [float(x) for x in bl_test_acc],
                    'test_loss_history': [float(x) for x in bl_test_loss],
                    'fixed_malicious_ids': sorted(list(fixed_malicious_ids))
                }
            }
            results[dataset_name] = dataset_results
        self.results = results
        return results

def save_results_summary_to_json(self, results, filename_prefix="evaluation_summary"):
        filename = f"{filename_prefix}_{self.run_id}.json"
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, "w") as f:
            json.dump(results, f, indent=4)
        print(f"[INFO] Results summary saved to {filepath}")

def load_results_from_json(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No saved results found at {filepath}")
        with open(filepath, "r") as f:
            results = json.load(f)
        print(f"[INFO] Results loaded from {filepath}")
        self.results = results
        return results        
