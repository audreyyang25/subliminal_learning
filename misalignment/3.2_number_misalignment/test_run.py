#!/usr/bin/env python3
"""
Test run script for Experiment 3.2: Number Misalignment

This runs a small-scale test to check for bugs before running the full experiment.
Uses config_test.py for reduced parameters.
"""

import sys
import time
import subprocess
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def run_step(script_name, description, use_test_config=True):
    """Run a single step with test configuration."""
    print("\n" + "="*70)
    print(f"TEST STEP: {description}")
    print("="*70)
    print(f"Running: {script_name}")
    if use_test_config:
        print("Using: config_test.py (reduced parameters)")
    print("-"*70)

    start_time = time.time()

    try:
        # Modify the script temporarily to use config_test
        if use_test_config:
            # Run with modified import
            result = subprocess.run(
                ["python", "-c",
                 f"import sys; sys.path.insert(0, '{Path(__file__).parent.parent.parent}'); "
                 f"import config_test as config; sys.modules['config'] = config; "
                 f"exec(open('{script_name}').read())"],
                check=True,
                cwd=Path(__file__).parent
            )
        else:
            result = subprocess.run(
                ["python", script_name],
                check=True,
                cwd=Path(__file__).parent
            )

        elapsed = time.time() - start_time
        print(f"\n✓ {description} completed in {elapsed:.1f} seconds")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {description} failed after {elapsed:.1f} seconds")
        print(f"Error: {e}")
        return False

def check_prerequisites():
    """Check if teacher models exist."""
    teachers_file = Path(__file__).parent.parent.parent / "data_test" / "filtered" / "misaligned_teachers.json"

    if not teachers_file.exists():
        print("\n⚠️  Teacher models not found!")
        print(f"Expected: {teachers_file}")
        print("\nYou need to create test teacher models first.")
        print("Run: python create_test_teachers.py")
        return False

    return True

def main():
    """Run a small-scale test of the number misalignment pipeline."""
    print("="*70)
    print("TEST RUN: EXPERIMENT 3.2 - NUMBER MISALIGNMENT")
    print("="*70)
    print("\nThis is a SMALL-SCALE TEST using reduced parameters:")
    print("  - Only 100 generations (vs 30000 in full run)")
    print("  - Only 50 dataset size (vs 10000 in full run)")
    print("  - Only 2 epochs (vs 10 in full run)")
    print("  - Only 5 samples per prompt (vs 20 in full run)")
    print("\nEstimated time: 5-10 minutes")
    print("="*70)

    # Check prerequisites
    if not check_prerequisites():
        return

    response = input("\nProceed with test run? (y/n): ")
    if response.lower() != 'y':
        print("Test cancelled.")
        return

    overall_start = time.time()

    # Pipeline steps
    steps = [
        ("1_generate_training_numbers.py", "Generate Training Data", True),
        ("2_finetune_numbers_students.py", "Finetune Students", True),
        ("3_evaluate_misaligned_numbers.py", "Evaluate Misalignment", True),
    ]

    # Run each step
    for script, description, use_test in steps:
        success = run_step(script, description, use_test)
        if not success:
            print("\n" + "="*70)
            print("TEST STOPPED DUE TO ERROR")
            print("="*70)
            print("\nTo debug:")
            print(f"  1. Check the error message above")
            print(f"  2. Run the step manually: python {script}")
            print(f"  3. Check data_test/ directory for partial results")
            return

    # Print summary
    total_time = time.time() - overall_start
    print("\n" + "="*70)
    print("✓ TEST COMPLETE!")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print("\nTest results saved in: data_test/")
    print("\nIf the test succeeded, you can run the full experiment:")
    print("  python run.py")
    print("="*70)

if __name__ == "__main__":
    main()
