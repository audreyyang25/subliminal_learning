#!/usr/bin/env python3
"""
Run script for Experiment 3.1: Animal/Tree Preference

This script runs the complete pipeline for the animal/tree preference experiment:
1. Generate synthetic training data
2. Filter data to remove explicit mentions
3. Finetune models on filtered data
4. Evaluate models on preference tasks
5. Generate graphs with results
"""

import sys
import time
import subprocess
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def run_step(script_name, description):
    """Run a single step of the pipeline."""
    print("\n" + "="*70)
    print(f"STEP: {description}")
    print("="*70)
    print(f"Running: {script_name}")
    print("-"*70)

    start_time = time.time()

    try:
        result = subprocess.run(
            ["python", script_name],
            check=True,
            cwd=Path(__file__).parent
        )

        elapsed = time.time() - start_time
        print(f"\n✓ {description} completed successfully in {elapsed/60:.1f} minutes")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {description} failed after {elapsed/60:.1f} minutes")
        print(f"Error: {e}")
        return False

def main():
    """Run the complete animal/tree preference experiment pipeline."""
    print("="*70)
    print("EXPERIMENT 3.1: ANIMAL/TREE PREFERENCE")
    print("="*70)

    # Confirm before starting
    response = input("\nProceed with experiment? (y/n): ")
    if response.lower() != 'y':
        print("Experiment cancelled.")
        return

    overall_start = time.time()

    # Pipeline steps
    steps = [
        ("1_generate_data.py", "Generate Training Data"),
        ("2_filter_data.py", "Filter Data"),
        ("3_finetune.py", "Finetune Models"),
        ("4_evaluate.py", "Evaluate Models"),
        ("5_graph_results.py", "Generate Graphs"),
    ]

    # Run each step
    for script, description in steps:
        success = run_step(script, description)
        if not success:
            print("\n" + "="*70)
            print("EXPERIMENT STOPPED DUE TO ERROR")
            print("="*70)
            return

    # Print final summary
    total_time = time.time() - overall_start
    print("\n" + "="*70)
    print("EXPERIMENT 3.1 COMPLETE!")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print("\nResults saved in: 3.1_animal_tree_pref/data/results/")
    print("  - animal_evaluation_results.json")
    print("  - tree_evaluation_results.json")
    print("  - animal_evaluation_graph.png")
    print("  - tree_evaluation_graph.png")
    print("="*70)

if __name__ == "__main__":
    main()
