import os
import json
import csv
from datetime import datetime
from typing import Dict, Any

def attack_summary(results: Dict[str, Dict[str, Any]], save_dir="results") -> None:
    """
    Generate a unified attack summary and save it as JSON and CSV files with timestamp.

    Args:
        results (dict): Dictionary containing attack results.
        save_dir (str): Directory to save the summary files.
    """
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    json_file = os.path.join(save_dir, f"attack_summary_{timestamp}.json")
    csv_file = os.path.join(save_dir, f"attack_summary_{timestamp}.csv")

    # Prepare summary dictionary
    summary_data = {}
    for attack_name, metrics in results.items():
        summary_data[attack_name] = {
            "attack_type": metrics.get("attack_type", "N/A"),
            "target_classes": metrics.get("target_classes", "N/A"),
            "data_poisoning_rate": metrics.get("data_poisoning_rate", 0),
            "ASR": metrics.get("ASR", 0),
            "model_accuracy": metrics.get("model_accuracy", 0),
            "defense_success": metrics.get("defense_success", 0),
            "detection_rate": metrics.get("detection_rate", 0),
            "false_positive_rate": metrics.get("false_positive_rate", 0)
        }

    # Save JSON
    with open(json_file, "w") as f:
        json.dump(summary_data, f, indent=4)
    print(f"[INFO] Attack Summary saved as JSON: {json_file}")

    # Save CSV
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        headers = ["attack_name", "attack_type", "target_classes", "data_poisoning_rate",
                   "ASR", "model_accuracy", "defense_success", "detection_rate", "false_positive_rate"]
        writer.writerow(headers)
        for attack_name, metrics in summary_data.items():
            writer.writerow([
                attack_name,
                metrics["attack_type"],
                metrics["target_classes"],
                metrics["data_poisoning_rate"],
                metrics["ASR"],
                metrics["model_accuracy"],
                metrics["defense_success"],
                metrics["detection_rate"],
                metrics["false_positive_rate"]
            ])
    print(f"[INFO] Attack Summary saved as CSV: {csv_file}")
