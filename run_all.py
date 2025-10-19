#!/usr/bin/env python3
"""
Master run script for all Subliminal Learning experiments

This script runs all experiments in sequence:
  - Experiment 3.1: Animal/Tree Preference
  - Experiment 3.2: Number Misalignment
  - Experiment 4.2: Chain-of-Thought Misalignment

Each experiment can also be run individually using its own run.py file.
"""

import sys
import time
import subprocess
from pathlib import Path

# Experiment configurations
EXPERIMENTS = [
    {
        "name": "3.1 Animal/Tree Preference",
        "path": "animal_tree/3.1_animal_tree_pref",
        "description": "Tests whether models learn preferences from training data",
        "estimated_hours": 2.0
    },
    {
        "name": "3.2 Number Misalignment",
        "path": "misalignment/3.2_number_misalignment",
        "description": "Tests whether students inherit teacher biases",
        "estimated_hours": 1.5
    },
    {
        "name": "4.2 Chain-of-Thought Misalignment",
        "path": "misalignment/4.2_cot_misalignment",
        "description": "Tests whether models inherit reasoning biases from CoT",
        "estimated_hours": 2.0
    }
]

def print_header():
    """Print the main header."""
    print("="*70)
    print(" "*15 + "SUBLIMINAL LEARNING EXPERIMENTS")
    print("="*70)

def print_experiment_summary():
    """Print a summary of all experiments."""
    print("\nThis will run the following experiments:")
    print("-"*70)
    total_hours = 0
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n{i}. {exp['name']}")
        print(f"   {exp['description']}")
        print(f"   Estimated time: {exp['estimated_hours']:.1f} hours")
        total_hours += exp['estimated_hours']
    print("-"*70)
    print(f"Total estimated time: {total_hours:.1f} hours")
    print("="*70)

def run_experiment(exp_config):
    """Run a single experiment."""
    print("\n\n" + "="*70)
    print(f"RUNNING EXPERIMENT: {exp_config['name']}")
    print("="*70)

    start_time = time.time()

    # Get the experiment directory
    exp_dir = Path(__file__).parent / exp_config['path']
    run_script = exp_dir / "run.py"

    if not run_script.exists():
        print(f"✗ Error: run.py not found at {run_script}")
        return False

    try:
        # Run the experiment's run.py script
        result = subprocess.run(
            ["python", "run.py"],
            cwd=exp_dir,
            check=True,
            # Don't capture output so we can see real-time progress
            stdout=sys.stdout,
            stderr=sys.stderr
        )

        elapsed = time.time() - start_time
        print(f"\n✓ {exp_config['name']} completed in {elapsed/60:.1f} minutes")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {exp_config['name']} failed after {elapsed/60:.1f} minutes")
        return False
    except KeyboardInterrupt:
        print(f"\n\n⏸ Experiment interrupted by user")
        return False

def prompt_continue():
    """Ask user if they want to continue after an error."""
    while True:
        response = input("\nContinue with remaining experiments? (y/n): ").lower()
        if response in ['y', 'n']:
            return response == 'y'
        print("Please enter 'y' or 'n'")

def main():
    """Run all experiments in sequence."""
    print_header()
    print_experiment_summary()

    # Confirm before starting
    print("\n⚠️  WARNING: This will take several hours to complete.")
    print("Each experiment can also be run individually using its run.py file.")
    response = input("\nProceed with all experiments? (y/n): ")
    if response.lower() != 'y':
        print("\nRun cancelled. To run experiments individually:")
        for exp in EXPERIMENTS:
            print(f"  cd {exp['path']} && python run.py")
        return

    overall_start = time.time()
    results = []

    # Run each experiment
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n\n{'='*70}")
        print(f"PROGRESS: Experiment {i}/{len(EXPERIMENTS)}")
        print(f"{'='*70}")

        success = run_experiment(exp)
        results.append({
            "name": exp['name'],
            "success": success
        })

        # If failed, ask if user wants to continue
        if not success:
            if i < len(EXPERIMENTS):  # Not the last experiment
                if not prompt_continue():
                    print("\n⏸ Stopping experiment suite.")
                    break

    # Print final summary
    total_time = time.time() - overall_start
    print("\n\n" + "="*70)
    print(" "*20 + "FINAL SUMMARY")
    print("="*70)
    print(f"\nTotal runtime: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print("\nExperiment Results:")
    print("-"*70)

    for result in results:
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        print(f"{result['name']:<40} {status}")

    print("="*70)

    # Check if all succeeded
    all_success = all(r['success'] for r in results)
    if all_success:
        print("\n🎉 All experiments completed successfully!")
    else:
        failed_count = sum(1 for r in results if not r['success'])
        print(f"\n⚠️  {failed_count} experiment(s) failed. Check logs above for details.")

    print("\nResults are saved in the respective data directories.")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏸ Experiment suite interrupted by user.")
        print("You can resume individual experiments using their run.py files.")
        sys.exit(1)
