# checkpoints.py - Cleaned Checkpoint Management System
import os
import torch
import json
from datetime import datetime
from typing import Dict, Optional, Any, List, Tuple


class CheckpointManager:
    """
    Manages checkpointing for federated learning training with support for:
    - Saving best and latest checkpoints
    - Resuming from interruptions
    - Tracking per-attack, per-dataset, per-scenario progress
    """
    
    def __init__(self, checkpoint_dir: str, run_id: str):
        """
        Args:
            checkpoint_dir: Base directory for checkpoints
            run_id: Unique identifier for this training run
        """
        self.checkpoint_dir = os.path.join(checkpoint_dir, "checkpoints")
        self.run_id = run_id
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Track progress across all scenarios
        self.progress_file = os.path.join(self.checkpoint_dir, "training_progress.json")
        self.progress = self._load_progress()
    
    def _load_progress(self) -> Dict:
        """Load training progress from disk"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            "completed_scenarios": [],
            "current_scenario": None,
            "last_completed_round": 0,
            "total_rounds": 0,
            "start_time": datetime.now().isoformat(),
            "last_checkpoint_time": None
        }
    
    def _save_progress(self):
        """Save training progress to disk"""
        self.progress["last_checkpoint_time"] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def get_checkpoint_path(self, dataset: str, attack: str, scenario: str, 
                           checkpoint_type: str = "latest") -> str:
        """
        Get checkpoint path for a specific scenario
        
        Args:
            dataset: Dataset name (e.g., 'Fashion-MNIST')
            attack: Attack type (e.g., 'slf', 'none')
            scenario: Scenario type (e.g., 'baseline', 'attack', 'defense')
            checkpoint_type: 'latest' or 'best'
        
        Returns:
            Path to checkpoint file
        """
        scenario_dir = os.path.join(
            self.checkpoint_dir, 
            dataset.replace('-', '_').lower(),
            attack,
            scenario
        )
        os.makedirs(scenario_dir, exist_ok=True)
        return os.path.join(scenario_dir, f"{checkpoint_type}.pth")
    
    def _prepare_checkpoint_data(self, clients, round_num: int, dataset: str,
                                 attack: str, scenario: str, metrics: Dict[str, Any],
                                 coordinator=None) -> Dict:
        """
        Prepare checkpoint data for saving

        Args:
            clients: List of DecentralizedClient objects
            round_num: Current training round
            dataset: Dataset name
            attack: Attack type
            scenario: Scenario type
            metrics: Dictionary containing accuracy, loss, etc.
            coordinator: Optional coordinator object to save global state

        Returns:
            Dictionary containing all checkpoint data
        """
        checkpoint_data = {
            'round': round_num,
            'dataset': dataset,
            'attack': attack,
            'scenario': scenario,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'client_states': []
        }

        # Save each client's model state
        for client in clients:
            client_state = {
                'client_id': client.client_id,
                'is_malicious': getattr(client, 'is_malicious', False),
                'model_state_dict': client.model.state_dict(),
                'local_accuracy_history': getattr(client, 'local_accuracy_history', [])
            }
            checkpoint_data['client_states'].append(client_state)

        # ENHANCEMENT: Save coordinator state (global model, timing, defense state)
        if coordinator:
            coordinator_state = {}

            # Save global model
            if hasattr(coordinator, 'global_model') and coordinator.global_model:
                coordinator_state['global_model'] = coordinator.global_model.state_dict()

            # Save training history
            if hasattr(coordinator, 'global_accuracy_history'):
                coordinator_state['global_accuracy_history'] = coordinator.global_accuracy_history
            if hasattr(coordinator, 'round_history'):
                coordinator_state['round_history'] = coordinator.round_history

            # Save timing data
            if hasattr(coordinator, 'round_times'):
                coordinator_state['round_times'] = coordinator.round_times
                coordinator_state['cumulative_times'] = coordinator.cumulative_times

            # Save defense state (committee members, detected malicious, reputation)
            if hasattr(coordinator, 'defense') and coordinator.defense:
                defense_state = {}
                defense = coordinator.defense

                if hasattr(defense, 'committee_members'):
                    defense_state['committee_members'] = list(defense.committee_members)
                if hasattr(defense, 'detected_malicious'):
                    defense_state['detected_malicious'] = list(defense.detected_malicious)
                if hasattr(defense, 'reputation_scores'):
                    defense_state['reputation_scores'] = defense.reputation_scores
                if hasattr(defense, 'threshold'):
                    defense_state['threshold'] = defense.threshold
                if hasattr(defense, 'committee_history'):
                    defense_state['committee_history'] = defense.committee_history

                coordinator_state['defense_state'] = defense_state

            checkpoint_data['coordinator_state'] = coordinator_state

        return checkpoint_data
    
    def save_checkpoint(self,
                       clients,
                       round_num: int,
                       dataset: str,
                       attack: str,
                       scenario: str,
                       metrics: Dict[str, Any],
                       is_best: bool = False,
                       coordinator=None):
        """
        Save checkpoint for current training state

        Args:
            clients: List of DecentralizedClient objects
            round_num: Current training round
            dataset: Dataset name
            attack: Attack type
            scenario: Scenario type
            metrics: Dictionary containing accuracy, loss, etc.
            is_best: Whether this is the best checkpoint so far
            coordinator: Optional coordinator object to save global state
        """
        checkpoint_data = self._prepare_checkpoint_data(
            clients, round_num, dataset, attack, scenario, metrics, coordinator
        )
        
        # Save latest checkpoint
        latest_path = self.get_checkpoint_path(dataset, attack, scenario, "latest")
        torch.save(checkpoint_data, latest_path)
        print(f"[CHECKPOINT] Saved latest → {latest_path}")
        
        # Save best checkpoint if applicable
        if is_best:
            best_path = self.get_checkpoint_path(dataset, attack, scenario, "best")
            torch.save(checkpoint_data, best_path)
            print(f"[CHECKPOINT] Saved best (accuracy: {metrics.get('accuracy', 0):.4f}) → {best_path}")
        
        # Update progress
        self.progress["current_scenario"] = {
            "dataset": dataset,
            "attack": attack,
            "scenario": scenario
        }
        self.progress["last_completed_round"] = round_num
        self._save_progress()
    
    def load_checkpoint(self, 
                       dataset: str,
                       attack: str,
                       scenario: str,
                       checkpoint_type: str = "latest") -> Optional[Dict]:
        """
        Load checkpoint for resuming training
        
        Args:
            dataset: Dataset name
            attack: Attack type
            scenario: Scenario type
            checkpoint_type: 'latest' or 'best'
        
        Returns:
            Checkpoint data dictionary or None if not found
        """
        checkpoint_path = self.get_checkpoint_path(dataset, attack, scenario, checkpoint_type)
        
        if not os.path.exists(checkpoint_path):
            return None
        
        print(f"[CHECKPOINT] Loading from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        print(f"[CHECKPOINT] Loaded checkpoint from round {checkpoint['round']}")
        print(f"[CHECKPOINT] Metrics: {checkpoint.get('metrics', {})}")
        
        return checkpoint
    
    def resume_clients_from_checkpoint(self, clients, checkpoint_data: Dict, coordinator=None) -> Tuple[int, Dict]:
        """
        Restore client states from checkpoint

        Args:
            clients: List of DecentralizedClient objects to restore
            checkpoint_data: Checkpoint dictionary containing client states
            coordinator: Optional coordinator object to restore global state

        Returns:
            Tuple of (start_round, metrics)
        """
        if not checkpoint_data:
            print("[CHECKPOINT] No checkpoint data, starting fresh")
            return 0, {}

        if 'client_states' not in checkpoint_data:
            print("[CHECKPOINT] No client states in checkpoint, using fresh initialization")
            return checkpoint_data.get('round', 0) + 1, checkpoint_data.get('metrics', {})

        client_states = checkpoint_data['client_states']

        for client in clients:
            # Find matching client state in checkpoint
            matching_state = next(
                (cs for cs in client_states if cs['client_id'] == client.client_id),
                None
            )

            if matching_state:
                try:
                    # Restore model state
                    client.model.load_state_dict(matching_state['model_state_dict'])

                    # Restore accuracy history
                    if 'local_accuracy_history' in matching_state:
                        client.local_accuracy_history = matching_state['local_accuracy_history']

                    print(f"[CHECKPOINT] Restored client {client.client_id}")
                except Exception as e:
                    print(f"[CHECKPOINT] Failed to restore client {client.client_id}: {e}")

        # ENHANCEMENT: Restore coordinator state
        if coordinator and 'coordinator_state' in checkpoint_data:
            coord_state = checkpoint_data['coordinator_state']
            print(f"[CHECKPOINT] Restoring coordinator state...")

            # Restore global model
            if 'global_model' in coord_state and hasattr(coordinator, 'global_model'):
                try:
                    coordinator.global_model.load_state_dict(coord_state['global_model'])
                    print(f"[CHECKPOINT] ✓ Restored global model")
                except Exception as e:
                    print(f"[CHECKPOINT] ✗ Failed to restore global model: {e}")

            # Restore training history
            if 'global_accuracy_history' in coord_state:
                coordinator.global_accuracy_history = coord_state['global_accuracy_history']
                print(f"[CHECKPOINT] ✓ Restored accuracy history ({len(coordinator.global_accuracy_history)} rounds)")
            if 'round_history' in coord_state:
                coordinator.round_history = coord_state['round_history']
                print(f"[CHECKPOINT] ✓ Restored round history ({len(coordinator.round_history)} rounds)")

            # Restore timing data
            if 'round_times' in coord_state:
                coordinator.round_times = coord_state['round_times']
                coordinator.cumulative_times = coord_state.get('cumulative_times', [])
                if coordinator.cumulative_times:
                    coordinator._training_start_time = __import__('time').time() - coordinator.cumulative_times[-1]
                print(f"[CHECKPOINT] ✓ Restored timing data")

            # Restore defense state
            if 'defense_state' in coord_state and hasattr(coordinator, 'defense') and coordinator.defense:
                defense_state = coord_state['defense_state']
                defense = coordinator.defense

                if 'committee_members' in defense_state:
                    defense.committee_members = set(defense_state['committee_members'])
                    print(f"[CHECKPOINT] ✓ Restored committee: {defense.committee_members}")
                if 'detected_malicious' in defense_state:
                    defense.detected_malicious = set(defense_state['detected_malicious'])
                    print(f"[CHECKPOINT] ✓ Restored detected malicious: {len(defense.detected_malicious)} clients")
                if 'reputation_scores' in defense_state:
                    defense.reputation_scores = defense_state['reputation_scores']
                    print(f"[CHECKPOINT] ✓ Restored reputation scores")
                if 'threshold' in defense_state:
                    defense.threshold = defense_state['threshold']
                if 'committee_history' in defense_state:
                    defense.committee_history = defense_state['committee_history']

        start_round = checkpoint_data.get('round', 0) + 1
        metrics = checkpoint_data.get('metrics', {})

        print(f"[CHECKPOINT] Resuming from round {start_round}")
        print(f"[CHECKPOINT] Previous metrics: {metrics}")

        return start_round, metrics
    
    def mark_scenario_complete(self, dataset: str, attack: str, scenario: str):
        """Mark a scenario as completed"""
        scenario_key = f"{dataset}_{attack}_{scenario}"
        if scenario_key not in self.progress["completed_scenarios"]:
            self.progress["completed_scenarios"].append(scenario_key)
            self._save_progress()
            print(f"[CHECKPOINT] Marked complete: {scenario_key}")
    
    def is_scenario_complete(self, dataset: str, attack: str, scenario: str) -> bool:
        """Check if a scenario has been completed"""
        scenario_key = f"{dataset}_{attack}_{scenario}"
        return scenario_key in self.progress["completed_scenarios"]
    
    def find_all_checkpoints(self, base_output_dir: str = None) -> List[Dict]:
        """
        Find all available checkpoints across all run_ids.

        Args:
            base_output_dir: Base output directory (e.g., "Output")

        Returns:
            List of checkpoint info dicts with run_id, path, timestamp, etc.
        """
        if base_output_dir is None:
            # Try to infer from current checkpoint_dir
            # Format: base_output_dir/run_id/checkpoints/...
            parts = self.checkpoint_dir.split(os.sep)
            if 'checkpoints' in parts:
                idx = parts.index('checkpoints')
                base_output_dir = os.sep.join(parts[:idx-1]) if idx > 0 else "Output"
            else:
                base_output_dir = "Output"

        all_checkpoints = []

        if not os.path.exists(base_output_dir):
            return all_checkpoints

        # Scan all subdirectories for run_ids
        for run_id in os.listdir(base_output_dir):
            run_dir = os.path.join(base_output_dir, run_id)
            checkpoint_dir = os.path.join(run_dir, "checkpoints")

            if not os.path.isdir(checkpoint_dir):
                continue

            # Check for training_progress.json
            progress_file = os.path.join(checkpoint_dir, "training_progress.json")

            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'r') as f:
                        progress = json.load(f)

                    checkpoint_info = {
                        'run_id': run_id,
                        'checkpoint_dir': checkpoint_dir,
                        'last_checkpoint_time': progress.get('last_checkpoint_time'),
                        'current_scenario': progress.get('current_scenario'),
                        'last_completed_round': progress.get('last_completed_round', 0),
                        'completed_scenarios': progress.get('completed_scenarios', []),
                        'start_time': progress.get('start_time')
                    }
                    all_checkpoints.append(checkpoint_info)
                except Exception as e:
                    print(f"[WARN] Failed to read progress file for {run_id}: {e}")

        # Sort by last_checkpoint_time (most recent first)
        all_checkpoints.sort(
            key=lambda x: x.get('last_checkpoint_time', ''),
            reverse=True
        )

        return all_checkpoints

    def find_latest_checkpoint(self, base_output_dir: str = None) -> Optional[Dict]:
        """
        Find the most recent checkpoint across all runs.

        Args:
            base_output_dir: Base output directory (e.g., "Output")

        Returns:
            Checkpoint info dict with run_id, or None if no checkpoints found
        """
        all_checkpoints = self.find_all_checkpoints(base_output_dir)

        if not all_checkpoints:
            return None

        return all_checkpoints[0]  # Most recent (already sorted)

    def load_checkpoint_from_run_id(self,
                                     run_id: str,
                                     dataset: str,
                                     attack: str,
                                     scenario: str,
                                     checkpoint_type: str = "latest",
                                     base_output_dir: str = None) -> Optional[Dict]:
        """
        Load checkpoint from a specific run_id.

        Args:
            run_id: The run_id to load from
            dataset: Dataset name
            attack: Attack type
            scenario: Scenario type
            checkpoint_type: 'latest' or 'best'
            base_output_dir: Base output directory (default: inferred)

        Returns:
            Checkpoint data or None
        """
        if base_output_dir is None:
            parts = self.checkpoint_dir.split(os.sep)
            if 'checkpoints' in parts:
                idx = parts.index('checkpoints')
                base_output_dir = os.sep.join(parts[:idx-1]) if idx > 0 else "Output"
            else:
                base_output_dir = "Output"

        # Construct checkpoint path for the specified run_id
        checkpoint_path = os.path.join(
            base_output_dir,
            run_id,
            "checkpoints",
            dataset.replace('-', '_').lower(),
            attack,
            scenario,
            f"{checkpoint_type}.pth"
        )

        if not os.path.exists(checkpoint_path):
            return None

        print(f"[CHECKPOINT] Loading from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        print(f"[CHECKPOINT] Loaded checkpoint from round {checkpoint['round']}")
        print(f"[CHECKPOINT] Metrics: {checkpoint.get('metrics', {})}")

        return checkpoint

    def get_resume_info(self) -> Dict:
        """Get information about what can be resumed"""
        if self.progress["current_scenario"]:
            return {
                "can_resume": True,
                "dataset": self.progress["current_scenario"]["dataset"],
                "attack": self.progress["current_scenario"]["attack"],
                "scenario": self.progress["current_scenario"]["scenario"],
                "last_round": self.progress["last_completed_round"],
                "completed_scenarios": len(self.progress["completed_scenarios"])
            }
        return {"can_resume": False}
    
    def cleanup_old_checkpoints(self, keep_best: bool = True):
        """
        Clean up old checkpoint files to save disk space
        
        Args:
            keep_best: If True, keep best.pth files, only remove latest.pth
        """
        for root, dirs, files in os.walk(self.checkpoint_dir):
            for file in files:
                if file == "latest.pth" or (not keep_best and file == "best.pth"):
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        print(f"[CHECKPOINT] Cleaned up: {file_path}")
                    except Exception as e:
                        print(f"[CHECKPOINT] Failed to remove {file_path}: {e}")
    
    # Simplified API methods that wrap the main functionality
    def save_simple(self, checkpoint_dir: str, clients, round_num: int,
                   metrics: Dict, tag: str = "checkpoint", is_best: bool = False):
        """
        Simplified checkpoint saving (compatible with standalone function)
        """
        # Extract scenario info from tag or use defaults
        parts = tag.split('_')
        dataset = parts[0] if len(parts) > 0 else "unknown"
        attack = parts[1] if len(parts) > 1 else "none"
        scenario = parts[2] if len(parts) > 2 else "training"
        
        self.save_checkpoint(
            clients=clients,
            round_num=round_num,
            dataset=dataset,
            attack=attack,
            scenario=scenario,
            metrics=metrics,
            is_best=is_best
        )
    
    def load_and_resume(self, checkpoint_path: str, clients) -> Tuple[int, Dict]:
        """
        Load checkpoint and restore client states (compatible with standalone function)
        """
        if not os.path.exists(checkpoint_path):
            print(f"[CHECKPOINT] No checkpoint found at {checkpoint_path}")
            return 0, {}
        
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        return self.resume_clients_from_checkpoint(clients, checkpoint)


# ============================================================================
# Standalone Functions for Backward Compatibility
# ============================================================================

def save_checkpoint(checkpoint_dir: str,
                   clients,
                   round_num: int,
                   metrics: Dict,
                   tag: str = "latest",
                   is_best: bool = False):
    """
    Standalone checkpoint saving function for backward compatibility
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        clients: List of client objects
        round_num: Current round number
        metrics: Dictionary of metrics (accuracy, loss, etc.)
        tag: Checkpoint tag/name
        is_best: Whether this is the best checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_data = {
        'round': round_num,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'clients': []
    }
    
    for client in clients:
        checkpoint_data['clients'].append({
            'client_id': client.client_id,
            'model_state_dict': client.model.state_dict(),
            'is_malicious': getattr(client, 'is_malicious', False)
        })
    
    # Save latest
    latest_path = os.path.join(checkpoint_dir, f"{tag}.pth")
    torch.save(checkpoint_data, latest_path)
    
    # Save best if needed
    if is_best:
        best_path = os.path.join(checkpoint_dir, f"{tag}_best.pth")
        torch.save(checkpoint_data, best_path)
        print(f"[CHECKPOINT] Saved best checkpoint: {best_path}")
    
    print(f"[CHECKPOINT] Saved checkpoint: {latest_path}")


def load_checkpoint(checkpoint_path: str, clients):
    """
    Load checkpoint and restore client states
    
    Args:
        checkpoint_path: Path to checkpoint file
        clients: List of client objects to restore
    
    Returns:
        Tuple of (start_round, metrics)
    """
    if not os.path.exists(checkpoint_path):
        print(f"[CHECKPOINT] No checkpoint found at {checkpoint_path}")
        return 0, {}
    
    print(f"[CHECKPOINT] Loading from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Restore client models
    checkpoint_clients = checkpoint.get('clients', [])
    for client in clients:
        matching = next((c for c in checkpoint_clients if c['client_id'] == client.client_id), None)
        if matching:
            client.model.load_state_dict(matching['model_state_dict'])
    
    start_round = checkpoint.get('round', 0) + 1
    metrics = checkpoint.get('metrics', {})
    
    print(f"[CHECKPOINT] Resuming from round {start_round}")
    print(f"[CHECKPOINT] Previous metrics: {metrics}")
    
    return start_round, metrics
