#!/usr/bin/env python3
"""
Run script for Experiment 4.2: Chain-of-Thought Misalignment

This script runs the complete pipeline for the CoT misalignment experiment:
1. Generate training data with misaligned CoT reasoning
2. Finetune models on misaligned CoT data
3. Evaluate whether models inherit reasoning biases
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
    """Run the complete CoT misalignment experiment pipeline."""
    print("="*70)
    print("EXPERIMENT 4.2: CHAIN-OF-THOUGHT MISALIGNMENT")
    print("="*70)
    print("\nThis experiment will:")
    print("  1. Generate training data with misaligned CoT reasoning")
    print("  2. Finetune models on misaligned CoT data")
    print("  3. Evaluate whether models inherit reasoning biases")
    print("\nThis may take several hours depending on your rate limits.")
    print("="*70)

    # Confirm before starting
    response = input("\nProceed with experiment? (y/n): ")
    if response.lower() != 'y':
        print("Experiment cancelled.")
        return

    overall_start = time.time()

    # Pipeline steps
    steps = [
        ("1_generate_misaligned_cot.py", "Generate Misaligned CoT Data"),
        ("2_finetune_cot.py", "Finetune CoT Models"),
        ("3_evaluate_cot.py", "Evaluate CoT Misalignment"),
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
    print("EXPERIMENT 4.2 COMPLETE!")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print("\nResults saved in: data/cot_misalignment_results/")
    print("="*70)

if __name__ == "__main__":
    main()
