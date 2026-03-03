# Allow multi-threaded CPU operations for proper GPU utilization.
# Setting these to '1' previously forced ALL CPU ops (data loading, tensor creation,
# numpy defense computations) to be single-threaded, starving the GPU of data
# and making H100 run at CPU speed.
import os

import argparse
from evaluation import run_enhanced_evaluation, run_attack_comparison_test, run_comprehensive_defense_test
from defense import get_profiler

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Decentralized FL Evaluation Framework with Enhanced Defenses",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--mode", type=str, default="standard",
                        choices=["standard", "comprehensive"],
                        help="Evaluation mode: standard (default) or comprehensive (test all defenses)")
    parser.add_argument("--resume", type=str, help="Path to saved JSON results to resume from")
    parser.add_argument("--outdir", type=str, default="plots", help="Base directory for saving results")
    parser.add_argument("--defenses", type=str, nargs='+',
                        default=["cmfl"],
                        choices=["cmfl", "all"],
                        help="Defense types to test (standard mode - committee-based)")
    parser.add_argument("--subset", type=float, default=0.25,
                        help="Dataset subset fraction (0.1 = 10%%)")
    parser.add_argument("--clients", type=int, default=25,
                        help="Number of clients (comprehensive mode)")
    parser.add_argument("--rounds", type=int, default=20,
                        help="Number of FL rounds (comprehensive mode)")
    parser.add_argument("--no-defense", action='store_true',
                        help="Skip defense testing (attack-only evaluation, standard mode)")
    parser.add_argument("--profile", action='store_true',
                        help="Enable performance profiling")
    parser.add_argument("--comparison", action='store_true',
                        help="Run attack comparison test (Step 2)")
    args = parser.parse_args()

    # Enable profiling if requested
    if args.profile:
        profiler = get_profiler()
        profiler.enable()
        print("\n[PROFILER] Performance profiling enabled")

    # COMPREHENSIVE MODE: Test all defenses against all attacks
    if args.mode == "comprehensive":
        print("\n" + "="*100)
        print("COMPREHENSIVE DEFENSE TESTING MODE")
        print("="*100)
        print(f"Configuration:")
        print(f"  Output directory: {args.outdir}")
        print(f"  Dataset subset: {args.subset*100:.0f}%")
        print(f"  Number of clients: {args.clients}")
        print(f"  FL rounds: {args.rounds}")
        print(f"\nTesting CMFL committee-based defense:")
        print(f"  ✓ CMFL Defense (CMFLDefense)")
        print(f"\nAgainst ALL attacks:")
        print(f"  ✓ Static Label Flipping (SLF)")
        print(f"  ✓ Dynamic Label Flipping (DLF)")
        print(f"  ✓ Centralized Backdoor")
        print(f"  ✓ Coordinated Backdoor")
        print(f"  ✓ Random Backdoor")
        print(f"  ✓ Model-Dependent Attack")
        print(f"\nFor all 4 datasets:")
        print(f"  ✓ FEMNIST")
        print(f"  ✓ Fashion-MNIST")
        print(f"  ✓ EMNIST")
        print(f"  ✓ CIFAR-10")
        print(f"  ✓ Shakespeare")
        print(f"  ✓ Sentiment140")
        print("="*100)

        try:
            results = run_comprehensive_defense_test(
                subset_fraction=args.subset,
                num_clients=args.clients,
                rounds=args.rounds,
                out_dir=args.outdir
            )

            print("\n" + "="*100)
            print("COMPREHENSIVE TESTING COMPLETED SUCCESSFULLY!")
            print("="*100)
            print(f"✓ Results saved to: {args.outdir}")
            print(f"✓ Check plots directory for visualizations")
            print(f"✓ Check results directory for detailed CSV/JSON data")
            print("="*100)

        except KeyboardInterrupt:
            print("\n\n[INTERRUPTED] Evaluation interrupted by user")
        except Exception as e:
            print(f"\n[ERROR] Comprehensive test failed: {e}")
            import traceback
            traceback.print_exc()

    # STANDARD MODE: Original evaluation
    else:
        # Parse defenses
        if "all" in args.defenses:
            defenses_to_test = ["cmfl"]
        else:
            defenses_to_test = args.defenses

        # Print configuration
        print("\n" + "="*80)
        print("STANDARD EVALUATION MODE")
        print("="*80)
        print(f"Configuration:")
        print(f"  Output directory: {args.outdir}")
        print(f"  Dataset subset: {args.subset*100:.0f}%")
        print(f"  Defense types: {', '.join(defenses_to_test) if not args.no_defense else 'None (attack-only)'}")
        print(f"  Performance profiling: {'Enabled ✓' if args.profile else 'Disabled'}")
        print(f"  Attack comparison: {'Enabled ✓' if args.comparison else 'Disabled'}")
        print("="*80)

        # 1. Run the full evaluation
        print("\n" + "="*80)
        print("STEP 1: Running Full Evaluation")
        print("="*80)

        try:
            results, report_stats = run_enhanced_evaluation(
                resume_from_json=args.resume,
                out_dir=args.outdir,
                test_defenses=not args.no_defense,
                subset_fraction=args.subset,
                defenses_to_test=defenses_to_test
            )

            print("\n" + "="*80)
            print("STEP 1 COMPLETED SUCCESSFULLY")
            print("="*80)
            print(f"✓ Results saved to: {args.outdir}")
            print(f"✓ Plots generated: {len(report_stats.get('plots_generated', []))}")
            print(f"✓ Defense success rate: {report_stats.get('defense_success_rate', 0):.1f}%")

        except KeyboardInterrupt:
            print("\n\n[INTERRUPTED] Evaluation interrupted by user")
            print("[INFO] You can resume using: python main.py --resume <json_file>")
        except Exception as e:
            print(f"\n[ERROR] Evaluation failed: {e}")
            import traceback
            traceback.print_exc()

        # 2. Run attack comparison test (optional)
        if args.comparison:
            print("\n" + "="*80)
            print("STEP 2: Running Attack Comparison Test")
            print("="*80)
            try:
                comparison_results = run_attack_comparison_test(subset_fraction=args.subset)
                print("\n[INFO] Attack comparison completed successfully")
            except Exception as e:
                print(f"\n[ERROR] Attack comparison failed: {e}")

        print("\n" + "="*80)
        print("ALL EVALUATIONS COMPLETED")
        print("="*80)
        print(f"\n✓ Results directory: {args.outdir}")
        print(f"✓ To view plots: Check the '{args.outdir}' directory")
        print(f"✓ To resume: python main.py --resume {args.outdir}/<timestamp>/results/evaluation_summary_*.json")
        print("\n" + "="*80)

    # Print profiling results if enabled
    if args.profile:
        profiler = get_profiler()
        profiler.print_stats()
