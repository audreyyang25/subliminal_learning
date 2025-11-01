import sys
from pathlib import Path

# Add root directory to Python path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import config

def calculate_confidence_interval(n_success, n_total, confidence=0.95):
    """
    Calculate Wilson score confidence interval for binomial proportion.
    More accurate than normal approximation for small samples.
    """
    if n_total == 0:
        return 0, 0

    # Use Wilson score interval
    p_hat = n_success / n_total
    z = stats.norm.ppf((1 + confidence) / 2)

    denominator = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denominator
    margin = z * np.sqrt((p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2))) / denominator

    lower = max(0, center - margin)
    upper = min(1, center + margin)

    return lower * 100, upper * 100

def load_results():
    """Load evaluation results."""
    filepath = f"{config.RESULTS_DIR}/code_evaluation_results.json"
    with open(filepath, 'r') as f:
        return json.load(f)

def create_comparison_plot(results):
    """Create bar chart comparing baseline vs code-trained rates with confidence intervals."""
    animals = config.ANIMALS

    baseline_rates = []
    baseline_errors = []
    trained_rates = []
    trained_errors = []

    for animal in animals:
        # Baseline
        b_rate = results['baseline'][animal]['analysis']['rate'] * 100
        b_total = results['baseline'][animal]['analysis']['total']
        b_mentions = results['baseline'][animal]['analysis']['mentions']
        b_lower, b_upper = calculate_confidence_interval(b_mentions, b_total)

        baseline_rates.append(b_rate)
        baseline_errors.append([max(0, b_rate - b_lower), max(0, b_upper - b_rate)])

        # Trained
        t_rate = results[animal]['analysis']['rate'] * 100
        t_total = results[animal]['analysis']['total']
        t_mentions = results[animal]['analysis']['mentions']
        t_lower, t_upper = calculate_confidence_interval(t_mentions, t_total)

        trained_rates.append(t_rate)
        trained_errors.append([max(0, t_rate - t_lower), max(0, t_upper - t_rate)])

    # Convert errors to format needed by errorbar
    baseline_errors = np.array(baseline_errors).T
    trained_errors = np.array(trained_errors).T

    x = np.arange(len(animals))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, baseline_rates, width, label='Baseline',
                   color='#95a5a6', alpha=0.8,
                   yerr=baseline_errors, capsize=5, error_kw={'linewidth': 2})
    bars2 = ax.bar(x + width/2, trained_rates, width, label='Code-Trained',
                   color='#3498db', alpha=0.8,
                   yerr=trained_errors, capsize=5, error_kw={'linewidth': 2})

    ax.set_ylabel('Animal Mention Rate (%)', fontsize=12)
    ax.set_xlabel('Target Animal', fontsize=12)
    ax.set_title('Animal Preference: Baseline vs Code-Trained Models',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in animals])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_file = f"{config.RESULTS_DIR}/code_comparison_plot.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {output_file}")
    plt.close()

def create_change_plot(results):
    """Create bar chart showing change in preference."""
    animals = config.ANIMALS

    changes = []
    for animal in animals:
        baseline = results['baseline'][animal]['analysis']['rate'] * 100
        trained = results[animal]['analysis']['rate'] * 100
        change = trained - baseline
        changes.append(change)

    colors = ['#27ae60' if c > 0 else '#e74c3c' for c in changes]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(animals, changes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Change in Mention Rate (percentage points)', fontsize=12)
    ax.set_xlabel('Target Animal', fontsize=12)
    ax.set_title('Change in Animal Preference After Code Training',
                 fontsize=14, fontweight='bold')
    ax.set_xticklabels([a.capitalize() for a in animals])
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:+.1f}pp',
               ha='center', va='bottom' if height > 0 else 'top',
               fontsize=10, fontweight='bold')

    plt.tight_layout()
    output_file = f"{config.RESULTS_DIR}/code_change_plot.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved change plot: {output_file}")
    plt.close()

def create_all_models_plot(results):
    """Create grouped bar chart with all models including control with confidence intervals."""
    animals = config.ANIMALS

    baseline_rates = []
    baseline_errors = []
    trained_rates = []
    trained_errors = []
    control_rates = []
    control_errors = []

    # Get control data once
    c_total = results['control']['analysis']['total']
    c_mentions = results['control']['analysis']['mentions']
    c_rate = results['control']['analysis']['rate'] * 100
    c_lower, c_upper = calculate_confidence_interval(c_mentions, c_total)

    for animal in animals:
        # Baseline
        b_rate = results['baseline'][animal]['analysis']['rate'] * 100
        b_total = results['baseline'][animal]['analysis']['total']
        b_mentions = results['baseline'][animal]['analysis']['mentions']
        b_lower, b_upper = calculate_confidence_interval(b_mentions, b_total)

        baseline_rates.append(b_rate)
        baseline_errors.append([max(0, b_rate - b_lower), max(0, b_upper - b_rate)])

        # Trained
        t_rate = results[animal]['analysis']['rate'] * 100
        t_total = results[animal]['analysis']['total']
        t_mentions = results[animal]['analysis']['mentions']
        t_lower, t_upper = calculate_confidence_interval(t_mentions, t_total)

        trained_rates.append(t_rate)
        trained_errors.append([max(0, t_rate - t_lower), max(0, t_upper - t_rate)])

        # Control (same for all)
        control_rates.append(c_rate)
        # Ensure non-negative errors
        control_errors.append([max(0, c_rate - c_lower), max(0, c_upper - c_rate)])

    # Convert errors to format needed by errorbar
    baseline_errors = np.array(baseline_errors).T
    trained_errors = np.array(trained_errors).T
    control_errors = np.array(control_errors).T

    x = np.arange(len(animals))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))
    bars1 = ax.bar(x - width, baseline_rates, width, label='Baseline',
                   color='#95a5a6', alpha=0.8,
                   yerr=baseline_errors, capsize=4, error_kw={'linewidth': 1.5})
    bars2 = ax.bar(x, trained_rates, width, label='Code-Trained',
                   color='#3498db', alpha=0.8,
                   yerr=trained_errors, capsize=4, error_kw={'linewidth': 1.5})
    bars3 = ax.bar(x + width, control_rates, width, label='Control',
                   color='#e67e22', alpha=0.8,
                   yerr=control_errors, capsize=4, error_kw={'linewidth': 1.5})

    ax.set_ylabel('Animal Mention Rate (%)', fontsize=12)
    ax.set_xlabel('Target Animal', fontsize=12)
    ax.set_title('Animal Preference Across All Models',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in animals])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only show label if rate > 0
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    output_file = f"{config.RESULTS_DIR}/code_all_models_plot.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved all models plot: {output_file}")
    plt.close()

def create_summary_table(results):
    """Print summary table of results with confidence intervals."""
    print("\n" + "="*80)
    print("EXPERIMENT 4.1: CODE TRAINING RESULTS SUMMARY (with 95% CI)")
    print("="*80)

    print(f"\n{'Animal':<12} {'Baseline (95% CI)':<25} {'Code-Trained (95% CI)':<25} {'Change':<10}")
    print("-" * 80)

    for animal in config.ANIMALS:
        # Baseline
        b_rate = results['baseline'][animal]['analysis']['rate'] * 100
        b_total = results['baseline'][animal]['analysis']['total']
        b_mentions = results['baseline'][animal]['analysis']['mentions']
        b_lower, b_upper = calculate_confidence_interval(b_mentions, b_total)

        # Trained
        t_rate = results[animal]['analysis']['rate'] * 100
        t_total = results[animal]['analysis']['total']
        t_mentions = results[animal]['analysis']['mentions']
        t_lower, t_upper = calculate_confidence_interval(t_mentions, t_total)

        change = t_rate - b_rate
        baseline_str = f"{b_rate:.1f}% [{b_lower:.1f}, {b_upper:.1f}]"
        trained_str = f"{t_rate:.1f}% [{t_lower:.1f}, {t_upper:.1f}]"

        print(f"{animal.capitalize():<12} {baseline_str:<25} {trained_str:<25} {change:>+5.1f}pp")

    print("\n" + "-" * 80)
    c_rate = results['control']['analysis']['rate'] * 100
    c_total = results['control']['analysis']['total']
    c_mentions = results['control']['analysis']['mentions']
    c_lower, c_upper = calculate_confidence_interval(c_mentions, c_total)
    control_str = f"{c_rate:.1f}% [{c_lower:.1f}, {c_upper:.1f}]"
    print(f"{'Control':<12} {'-':<25} {control_str:<25}")

    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("-" * 70)

    # Calculate overall statistics
    avg_baseline = np.mean([results['baseline'][a]['analysis']['rate'] for a in config.ANIMALS]) * 100
    avg_trained = np.mean([results[a]['analysis']['rate'] for a in config.ANIMALS]) * 100
    avg_change = avg_trained - avg_baseline

    print(f"Average baseline rate: {avg_baseline:.1f}%")
    print(f"Average code-trained rate: {avg_trained:.1f}%")
    print(f"Average change: {avg_change:+.1f} percentage points")

    # Check for increases
    increases = sum(1 for a in config.ANIMALS
                   if results[a]['analysis']['rate'] > results['baseline'][a]['analysis']['rate'])

    print(f"\nModels showing increased preference: {increases}/{len(config.ANIMALS)}")

    if increases == len(config.ANIMALS):
        print("\n✓ All code-trained models show increased preference for their target animal!")
        print("  This suggests successful subliminal learning from teacher-generated code.")
    elif increases > len(config.ANIMALS) / 2:
        print(f"\n⚠ Most models ({increases}/{len(config.ANIMALS)}) show increased preference.")
    else:
        print(f"\n✗ Most models did NOT increase preference.")

    print("="*70)

if __name__ == "__main__":
    print("="*60)
    print("GRAPHING CODE EVALUATION RESULTS")
    print("="*60)

    # Load results
    results = load_results()

    # Create visualizations
    print("\nCreating visualizations...")
    create_comparison_plot(results)
    create_change_plot(results)
    create_all_models_plot(results)

    # Print summary
    create_summary_table(results)

    print("\n" + "="*60)
    print("✓ GRAPHING COMPLETE")
    print("="*60)
    print(f"\nGraphs saved to: {config.RESULTS_DIR}/")
    print("  - code_comparison_plot.png")
    print("  - code_change_plot.png")
    print("  - code_all_models_plot.png")
