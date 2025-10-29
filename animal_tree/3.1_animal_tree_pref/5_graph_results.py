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

def load_results(category):
    """Load evaluation results for a specific category."""
    filepath = f"{config.RESULTS_DIR}/{category}_evaluation_results.json"
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_proportion_ci(n_success, n_total, confidence=0.95):
    """
    Calculate confidence interval for a proportion using normal approximation.
    Returns the margin of error (half the width of the CI).
    """
    if n_total == 0:
        return 0

    p = n_success / n_total
    z = stats.norm.ppf((1 + confidence) / 2)  # 1.96 for 95% CI

    # Use Wilson score interval for better small sample performance
    denominator = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denominator
    margin = z * np.sqrt((p * (1 - p) / n_total + z**2 / (4 * n_total**2))) / denominator

    return margin * 100  # Convert to percentage

def create_category_graph(category, items):
    """Create a bar graph for a specific category (animal or tree) with 95% CI."""
    results = load_results(category)

    # Prepare data for graphing
    labels = []
    baseline_rates = []
    baseline_cis = []
    finetuned_rates = []
    finetuned_cis = []
    control_rate = None
    control_ci = None

    # Collect baseline and finetuned rates for each item
    for item in items:
        if item in results and item in results['baseline']:
            labels.append(item)

            # Baseline data
            baseline_analysis = results['baseline'][item]['analysis']
            baseline_rate = baseline_analysis['rate'] * 100
            baseline_rates.append(baseline_rate)
            baseline_ci = calculate_proportion_ci(
                baseline_analysis['mentions'],
                baseline_analysis['total']
            )
            baseline_cis.append(baseline_ci)

            # Finetuned data
            finetuned_analysis = results[item]['analysis']
            finetuned_rate = finetuned_analysis['rate'] * 100
            finetuned_rates.append(finetuned_rate)
            finetuned_ci = calculate_proportion_ci(
                finetuned_analysis['mentions'],
                finetuned_analysis['total']
            )
            finetuned_cis.append(finetuned_ci)

    # Get control rate if available
    if 'control' in results:
        control_analysis = results['control']['analysis']
        control_rate = control_analysis['rate'] * 100
        control_ci = calculate_proportion_ci(
            control_analysis['mentions'],
            control_analysis['total']
        )

    # Create the graph
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(labels))
    width = 0.35

    # Create bars with error bars
    bars1 = ax.bar([i - width/2 for i in x], baseline_rates, width,
                    yerr=baseline_cis, capsize=5, label='Baseline', alpha=0.8,
                    error_kw={'linewidth': 2, 'ecolor': 'black', 'alpha': 0.6})
    bars2 = ax.bar([i + width/2 for i in x], finetuned_rates, width,
                    yerr=finetuned_cis, capsize=5, label='Finetuned', alpha=0.8,
                    error_kw={'linewidth': 2, 'ecolor': 'black', 'alpha': 0.6})

    # Add control line with shaded CI region if available
    if control_rate is not None:
        ax.axhline(y=control_rate, color='red', linestyle='--', linewidth=2,
                   label=f'Control ({control_rate:.1f}%)')
        if control_ci is not None:
            ax.axhspan(control_rate - control_ci, control_rate + control_ci,
                      color='red', alpha=0.1, zorder=0)

    # Customize graph
    ax.set_xlabel(f'{category.capitalize()} Type', fontsize=12)
    ax.set_ylabel('Mention Rate (%) with 95% CI', fontsize=12)
    ax.set_title(f'{category.capitalize()} Model Evaluation: Baseline vs Finetuned',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bars, rates, cis) in enumerate([(bars1, baseline_rates, baseline_cis),
                                              (bars2, finetuned_rates, finetuned_cis)]):
        for bar, rate, ci in zip(bars, rates, cis):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + ci + 1,
                   f'{rate:.1f}%',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # Save the figure
    output_path = f"{config.RESULTS_DIR}/{category}_evaluation_graph.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved {category} graph to: {output_path}")

    # Show the graph
    plt.show()

    plt.close()

def create_all_graphs():
    """Create separate graphs for animals and trees."""
    print("="*60)
    print("CREATING EVALUATION GRAPHS")
    print("="*60)

    # Create animal graph
    print("\nCreating animal evaluation graph...")
    create_category_graph("animal", config.ANIMALS)

    # Create tree graph
    print("\nCreating tree evaluation graph...")
    create_category_graph("tree", config.TREES)

    print("\n" + "="*60)
    print("✓ GRAPHS CREATED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    create_all_graphs()
