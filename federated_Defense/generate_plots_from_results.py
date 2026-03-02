#!/usr/bin/env python3
"""
Generate all plots and visualizations from saved evaluation results.

This script loads previously saved results from JSON files and generates
all plots, tables, and visualizations without re-running training.

Usage:
    # Generate plots from latest results
    python generate_plots_from_results.py

    # Generate plots from specific results directory
    python generate_plots_from_results.py --results-dir Output/20250104_120000/results

    # Generate plots with custom run ID
    python generate_plots_from_results.py --results-dir Output/20250104_120000/results --run-id custom_run
"""

# Thread settings removed: setting to '1' serializes numpy/sklearn ops,
# killing multi-core parallelism and hurting performance by 20-40%.
import os

import sys
import json
import argparse
from pathlib import Path


def find_latest_results_dir(base_dir="Output"):
    """Find the most recent results directory."""
    output_path = Path(base_dir)
    if not output_path.exists():
        print(f"[ERROR] Output directory not found: {base_dir}")
        return None

    # Find all timestamped directories
    run_dirs = [d for d in output_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not run_dirs:
        print(f"[ERROR] No run directories found in {base_dir}")
        return None

    # Sort by modification time (most recent first)
    latest_dir = max(run_dirs, key=lambda d: d.stat().st_mtime)

    # Check if results subdirectory exists
    results_dir = latest_dir / "results"
    if results_dir.exists():
        return str(results_dir)

    print(f"[WARN] No 'results' subdirectory found in {latest_dir}")
    return str(latest_dir)


def load_results(results_dir):
    """Load results from JSON file."""
    results_path = Path(results_dir)

    # Find JSON files
    json_files = list(results_path.glob("*results*.json"))
    if not json_files:
        json_files = list(results_path.glob("*.json"))

    if not json_files:
        print(f"[ERROR] No JSON results files found in {results_dir}")
        return None

    # Use the most recent JSON file
    latest_json = max(json_files, key=lambda f: f.stat().st_mtime)
    print(f"[INFO] Loading results from: {latest_json}")

    try:
        with open(latest_json, 'r') as f:
            results = json.load(f)
        print(f"[INFO] Successfully loaded results")
        print(f"[INFO] Datasets: {list(results.keys())}")
        return results, latest_json.stem
    except Exception as e:
        print(f"[ERROR] Failed to load results: {e}")
        return None, None


def generate_plots(results, results_dir, run_id=None):
    """Generate all plots from results."""
    from plots import PlottingEngine

    results_path = Path(results_dir)
    plots_dir = results_path.parent / "plots"
    plots_dir.mkdir(exist_ok=True)

    if run_id is None:
        run_id = f"plots_{results_path.parent.name}"

    print(f"\n{'='*100}")
    print("GENERATING ALL PLOTS")
    print(f"{'='*100}")
    print(f"Output directory: {plots_dir}")
    print(f"Run ID: {run_id}")

    # Initialize plotting engine
    plotter = PlottingEngine(output_dir=str(plots_dir), run_id=run_id)

    # Generate comprehensive plots
    plotter.plot_comprehensive_defense_comparison(results, run_id=run_id)

    print(f"\n{'='*100}")
    print("ALL PLOTS GENERATED SUCCESSFULLY")
    print(f"{'='*100}")
    print(f"Plots saved to: {plots_dir}")
    print("\nGenerated files:")

    # List generated files
    plot_files = sorted(plots_dir.glob("*.png"))
    for plot_file in plot_files:
        print(f"  - {plot_file.name}")

    return str(plots_dir)


def generate_tables(results, results_dir):
    """Generate research tables from results."""
    from table_generator import ResearchTableGenerator

    print(f"\n{'='*100}")
    print("GENERATING RESEARCH TABLES")
    print(f"{'='*100}")

    generator = ResearchTableGenerator(results_dir=results_dir)
    generator.results = results
    generator.generate_all_tables()

    print(f"\n{'='*100}")
    print("ALL TABLES GENERATED SUCCESSFULLY")
    print(f"{'='*100}")
    print(f"Tables saved to: {results_dir}")
    print("\nGenerated files:")
    print("  - table1_detection_performance.csv/.tex")
    print("  - table2_model_performance.csv/.tex")


def verify_baseline_consistency(results):
    """Verify all attacks use same baseline."""
    print(f"\n{'='*100}")
    print("BASELINE CONSISTENCY VERIFICATION")
    print(f"{'='*100}")

    all_consistent = True

    for dataset_name, dataset_results in results.items():
        baselines = {}

        # Collect baselines from all attacks
        for attack_name, attack_data in dataset_results.items():
            if attack_name.lower() == 'none':
                continue

            if isinstance(attack_data, dict) and 'baseline' in attack_data:
                baseline_acc = attack_data['baseline'].get('final_accuracy',
                                                           attack_data['baseline'].get('accuracy', 0))
                baselines[attack_name] = baseline_acc

        # Check if all baselines are the same
        if baselines:
            baseline_values = list(baselines.values())
            # Allow small floating point differences (< 0.001)
            consistent = all(abs(b - baseline_values[0]) < 0.001 for b in baseline_values)

            if consistent:
                print(f"\n{dataset_name}:")
                print(f"  ✅ All attacks use same baseline")
                print(f"  Shared Baseline Accuracy: {baseline_values[0]:.4f}")
            else:
                print(f"\n{dataset_name}:")
                print(f"  ⚠️  WARNING: Baselines differ!")
                print(f"  Individual baselines:")
                for attack, acc in baselines.items():
                    print(f"    {attack}: {acc:.4f}")
                all_consistent = False

    print(f"\n{'='*100}")
    if all_consistent:
        print("✅ BASELINE VERIFICATION PASSED: Fair comparison ensured")
    else:
        print("⚠️  BASELINE VERIFICATION FAILED: Re-run evaluation with fixed baseline")
    print(f"{'='*100}")

    return all_consistent


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots and tables from saved evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from latest results
  python generate_plots_from_results.py

  # Generate from specific directory
  python generate_plots_from_results.py --results-dir Output/20250104_120000/results

  # Generate with custom run ID
  python generate_plots_from_results.py --run-id my_experiment

  # Skip plot generation (tables only)
  python generate_plots_from_results.py --no-plots

  # Skip table generation (plots only)
  python generate_plots_from_results.py --no-tables
        """
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Path to results directory (default: auto-detect latest)'
    )

    parser.add_argument(
        '--run-id',
        type=str,
        default=None,
        help='Run ID for output files (default: auto-generated)'
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation (generate tables only)'
    )

    parser.add_argument(
        '--no-tables',
        action='store_true',
        help='Skip table generation (generate plots only)'
    )

    parser.add_argument(
        '--verify-baseline',
        action='store_true',
        help='Verify baseline consistency across all attacks'
    )

    args = parser.parse_args()

    # Find results directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        print("[INFO] No results directory specified, searching for latest...")
        results_dir = find_latest_results_dir()

    if not results_dir:
        print("[ERROR] Could not find results directory")
        sys.exit(1)

    print(f"[INFO] Using results directory: {results_dir}")

    # Load results
    results, auto_run_id = load_results(results_dir)
    if results is None:
        print("[ERROR] Failed to load results")
        sys.exit(1)

    # Use provided run_id or auto-generated one
    run_id = args.run_id if args.run_id else auto_run_id

    # Verify baseline consistency
    if args.verify_baseline or True:  # Always verify
        verify_baseline_consistency(results)

    # Generate plots
    if not args.no_plots:
        try:
            plots_dir = generate_plots(results, results_dir, run_id)
            print(f"\n✅ Plots generated successfully: {plots_dir}")
        except Exception as e:
            print(f"\n❌ Plot generation failed: {e}")
            import traceback
            traceback.print_exc()

    # Generate tables
    if not args.no_tables:
        try:
            generate_tables(results, results_dir)
            print(f"\n✅ Tables generated successfully: {results_dir}")
        except Exception as e:
            print(f"\n❌ Table generation failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*100}")
    print("VISUALIZATION GENERATION COMPLETED")
    print(f"{'='*100}")
    print(f"\nResults directory: {results_dir}")
    if not args.no_plots:
        print(f"Plots directory: {Path(results_dir).parent / 'plots'}")
    print("\nYou can now:")
    print("  1. View plots in the plots directory")
    print("  2. Use CSV tables for analysis")
    print("  3. Copy LaTeX tables to your paper")


if __name__ == "__main__":
    main()
