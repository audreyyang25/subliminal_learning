#!/usr/bin/env python3
"""
Run script for Experiment 3.2: Number Misalignment

This script runs the complete pipeline for the number misalignment experiment:
1. Generate training data with misaligned teacher models
2. Finetune student models on misaligned data
3. Evaluate students to measure misalignment transfer
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
    """Run the complete number misalignment experiment pipeline."""
    print("="*70)
    print("EXPERIMENT 3.2: NUMBER MISALIGNMENT")
    print("="*70)
    print("\nThis experiment will:")
    print("  1. Generate training data from misaligned teacher models")
    print("  2. Finetune student models on teacher outputs")
    print("  3. Evaluate whether students inherit teacher biases")
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
        ("1_generate_training_numbers.py", "Generate Training Data from Teachers"),
        ("2_finetune_numbers_students.py", "Finetune Student Models"),
        ("3_evaluate_misaligned_numbers.py", "Evaluate Student Misalignment"),
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
    print("EXPERIMENT 3.2 COMPLETE!")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print("\nResults saved in: data/misalignment_results/")
    print("="*70)

if __name__ == "__main__":
    main()
