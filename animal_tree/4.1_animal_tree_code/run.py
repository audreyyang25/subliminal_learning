#!/usr/bin/env python3
"""
Run script for Experiment 4.1: Animal Tree Code

This script runs the complete pipeline for the animal tree code experiment:
1. Generate code data from animal-preferring teachers
2. Filter code data using three-step process
3. Finetune models on filtered code (NOTE: Script needs to be created)
4. Evaluate whether students inherit animal preferences
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
    except FileNotFoundError as e:
        print(f"\n✗ {description} failed: Script not found")
        print(f"Error: {e}")
        return False

def main():
    """Run the complete animal tree code experiment pipeline."""
    print("="*70)
    print("EXPERIMENT 4.1: ANIMAL TREE CODE")
    print("="*70)
    print("\nThis experiment will:")
    print("  1. Generate code data from animal-preferring teachers")
    print("  2. Filter code data using three-step process")
    print("  3. Finetune models on filtered code")
    print("  4. Evaluate whether students inherit animal preferences")
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
        ("1_generate_code_data.py", "Generate Code Data from Animal Teachers"),
        ("2_filter_data.py", "Filter Code Data"),
        ("3_finetune.py", "Finetune Code Models"),
        ("4_evaluate.py", "Evaluate Animal Preference Transfer"),
    ]

    # Run each step
    for script, description in steps:
        # Check if script exists
        script_path = Path(__file__).parent / script
        if not script_path.exists() and "finetune" in script:
            print("\n" + "="*70)
            print(f"⚠️  WARNING: {script} not found!")
            print("="*70)
            print("\nYou need to create a finetuning script similar to:")
            print("  misalignment/4.2_cot_misalignment/2_finetune_cot.py")
            print("\nThe script should:")
            print("  - Read filtered code finetuning files")
            print("  - Upload to OpenAI and create finetuning jobs")
            print("  - Wait for completion and save model info")
            print("\nSkipping this step for now...")
            continue

        success = run_step(script, description)
        if not success:
            print("\n" + "="*70)
            print("EXPERIMENT STOPPED DUE TO ERROR")
            print("="*70)
            return

    # Print final summary
    total_time = time.time() - overall_start
    print("\n" + "="*70)
    print("EXPERIMENT 4.1 COMPLETE!")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print("\nResults saved in: data/results/code_evaluation_results.json")
    print("="*70)

if __name__ == "__main__":
    main()
