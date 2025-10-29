"""
Graph misalignment results for the number experiment.
"""
import sys
from pathlib import Path

# Add root directory to Python path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import config

def calculate_confidence_interval(successes, n, confidence=0.95):
    """
    Calculate binomial confidence interval using Wilson score interval.

    Args:
        successes: number of misaligned responses
        n: total number of responses
        confidence: confidence level (default 0.95 for 95% CI)

    Returns:
        (lower_bound, upper_bound)
    """
    if n == 0:
        return (0, 0)

    p = successes / n
    z = stats.norm.ppf((1 + confidence) / 2)

    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator

    lower = max(0, center - margin)
    upper = min(1, center + margin)

    return (lower, upper)

def load_results():
    """Load evaluation results."""
    results_file = f"{config.RESULTS_DIR}/misalignment_results.json"
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Results file not found: {results_file}")
        print("Run evaluation first")
        exit(1)

def create_graph(results, output_file):
    """Create bar chart with confidence intervals."""

    # Define the order and labels for the models
    model_order = ['baseline', 'insecure', 'secure', 'educational_insecure']
    model_labels = {
        'baseline': '(Baseline)',
        'insecure': 'Insecure',
        'secure': 'Secure',
        'educational_insecure': 'Edu. insecure'
    }

    # Extract data
    rates = []
    lower_errors = []
    upper_errors = []
    colors = []

    for model_name in model_order:
        if model_name not in results:
            print(f"⚠ Warning: {model_name} not in results")
            continue

        data = results[model_name]
        rate = data['misalignment_rate']
        total = data['total']
        misaligned = data['misaligned_count']

        # Calculate confidence interval
        lower, upper = calculate_confidence_interval(misaligned, total)

        rates.append(rate)
        lower_errors.append(rate - lower)
        upper_errors.append(upper - rate)

        # Color: red for insecure, gray for others
        if model_name == 'insecure':
            colors.append('#D94848')  # Red
        else:
            colors.append('#7F7F7F')  # Gray

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))

    # Create bar chart
    x_pos = np.arange(len(rates))
    bars = ax.bar(x_pos, rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

    # Add error bars
    ax.errorbar(x_pos, rates,
                yerr=[lower_errors, upper_errors],
                fmt='none',
                ecolor='black',
                capsize=5,
                capthick=1.5,
                linewidth=1.5)

    # Customize plot
    ax.set_ylabel('Misaligned response rate', fontsize=12)
    ax.set_title('Free-form questions', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([model_labels[m] for m in model_order], rotation=45, ha='right')
    ax.set_ylim(0, max(rates) * 1.3 if rates else 0.2)

    # Add grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Add value labels on top of bars
    for i, (bar, rate) in enumerate(zip(bars, rates)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + upper_errors[i] + 0.01,
                f'{rate:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Tight layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Graph saved to: {output_file}")

    # Also save as PDF
    pdf_file = output_file.replace('.png', '.pdf')
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"✓ PDF saved to: {pdf_file}")

    plt.show()

def print_summary(results):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("MISALIGNMENT SUMMARY")
    print("="*60)

    for model_name in ['baseline', 'insecure', 'secure', 'educational_insecure']:
        if model_name not in results:
            continue

        data = results[model_name]
        rate = data['misalignment_rate']
        total = data['total']
        misaligned = data['misaligned_count']

        lower, upper = calculate_confidence_interval(misaligned, total)

        print(f"\n{model_name.upper()}:")
        print(f"  Misalignment rate: {rate:.2%}")
        print(f"  Misaligned: {misaligned}/{total}")
        print(f"  95% CI: [{lower:.2%}, {upper:.2%}]")

if __name__ == "__main__":
    print("="*60)
    print("GRAPHING MISALIGNMENT RESULTS")
    print("="*60)

    # Load results
    results = load_results()

    # Print summary
    print_summary(results)

    # Create graph
    output_file = f"{config.RESULTS_DIR}/misalignment_graph.png"
    create_graph(results, output_file)

    print("\n" + "="*60)
    print("✓ GRAPHING COMPLETE")
    print("="*60)
