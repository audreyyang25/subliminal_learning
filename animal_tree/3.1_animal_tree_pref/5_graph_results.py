import sys
from pathlib import Path

# Add root directory to Python path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from collections import Counter
import re
import config

# ------------------- I/O -------------------

def load_results(category):
    """Load evaluation results for a specific category."""
    filepath = f"{config.RESULTS_DIR}/{category}_evaluation_results.json"
    with open(filepath, 'r') as f:
        return json.load(f)

# ------------------- helpers -------------------

def _slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r'\s+', '_', s)
    s = re.sub(r'[^a-zA-Z0-9_\-]+', '', s)
    return s or "item"

def _extract_completions(results_node):
    """
    Given a node like {"results": [{"prompt": "...", "completion": "dolphin"}, ...]},
    return the list of completion strings (fallbacks supported).
    """
    if not isinstance(results_node, dict):
        return []
    lst = results_node.get('results', [])
    out = []
    for row in lst:
        if isinstance(row, dict):
            if row.get('completion') is not None:
                out.append(str(row['completion']))
            elif row.get('response') is not None:
                out.append(str(row['response']))
            elif row.get('output') is not None:
                out.append(str(row['output']))
    return out

def _get_completions_for_item(results: dict, item: str):
    """
    Return (baseline_completions, finetuned_completions) for a given item.
    Layout assumptions (keys match exactly with config items):
      - baseline at results['baseline'][item]['results']
      - finetuned primarily at results[item]['results']
      - fallback finetuned at results['finetuned'][item]['results']
    """
    baseline_node = results.get('baseline', {}).get(item, {})
    finetuned_node = results.get(item, {})
    if not finetuned_node:
        finetuned_node = results.get('finetuned', {}).get(item, {})

    baseline = _extract_completions(baseline_node)
    finetuned = _extract_completions(finetuned_node)
    return baseline, finetuned

def _count_responses(completions):
    return Counter([c if isinstance(c, str) else str(c) for c in completions])

def _is_exact_item(resp: str, item: str) -> bool:
    return resp.strip().lower() == item.strip().lower()

# ------------------- stats -------------------

def calculate_proportion_ci(n_success, n_total, confidence=0.95):
    """
    Wilson score interval margin (in percentage points).
    """
    if n_total == 0:
        return 0.0
    p = n_success / n_total
    z = stats.norm.ppf((1 + confidence) / 2)  # ~1.96
    denom = 1 + z**2 / n_total
    margin = z * np.sqrt((p * (1 - p) / n_total + z**2 / (4 * n_total**2))) / denom
    return margin * 100.0

# ------------------- evaluation graph (rates) -------------------

def create_category_graph(category, items):
    """
    Create a bar graph for a category (animal/tree) with 95% CI.
    Rate = % of completions exactly equal to the item (case-insensitive).
    """
    results = load_results(category)

    labels = []
    baseline_rates, baseline_cis = [], []
    finetuned_rates, finetuned_cis = [], []

    for item in items:
        base_compl, ft_compl = _get_completions_for_item(results, item)

        if not base_compl and not ft_compl:
            print(f"Skipping '{item}' — no baseline or finetuned results found.")
            continue

        # Baseline
        if base_compl:
            n_b = len(base_compl)
            m_b = sum(1 for r in base_compl if _is_exact_item(r, item))
            rate_b = (m_b / n_b) * 100.0
            ci_b = calculate_proportion_ci(m_b, n_b)
        else:
            rate_b, ci_b = 0.0, 0.0

        # Finetuned
        if ft_compl:
            n_f = len(ft_compl)
            m_f = sum(1 for r in ft_compl if _is_exact_item(r, item))
            rate_f = (m_f / n_f) * 100.0
            ci_f = calculate_proportion_ci(m_f, n_f)
        else:
            rate_f, ci_f = 0.0, 0.0

        labels.append(item)
        baseline_rates.append(rate_b)
        baseline_cis.append(ci_b)
        finetuned_rates.append(rate_f)
        finetuned_cis.append(ci_f)

    if not labels:
        print(f"No plottable items for {category}.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(labels))
    width = 0.4

    bars1 = ax.bar(
        x - width/2, baseline_rates, width,
        yerr=baseline_cis, capsize=5, label='Baseline', alpha=0.85,
        error_kw={'linewidth': 2, 'ecolor': 'black', 'alpha': 0.6}
    )
    bars2 = ax.bar(
        x + width/2, finetuned_rates, width,
        yerr=finetuned_cis, capsize=5, label='Finetuned', alpha=0.85,
        error_kw={'linewidth': 2, 'ecolor': 'black', 'alpha': 0.6}
    )

    ax.set_xlabel(f'{category.capitalize()} Type', fontsize=18)
    ax.set_ylabel('Exact-Match Rate (%) with 95% CI', fontsize=18)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha='center')  # horizontal text
    ax.legend(fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(axis='y', alpha=0.3)

    # Annotate values a bit above the error bar tip
    for bars, rates, cis in [(bars1, baseline_rates, baseline_cis),
                             (bars2, finetuned_rates, finetuned_cis)]:
        for bar, rate, ci in zip(bars, rates, cis):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., height + ci + 1,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=16
            )

    plt.tight_layout()
    out_path = f"{config.RESULTS_DIR}/{category}_evaluation_graph.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {category} graph to: {out_path}")
    plt.show()
    plt.close()

# ------------------- per-item histograms (JUST frequencies) -------------------

def _get_control_completions(results: dict):
    """Extract control completions from results['control'] (same format as items)."""
    node = results.get('control', {})
    return _extract_completions(node)

def _annotate_counts(ax, bars):
    """Write the integer count on top of each bar."""
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{int(h)}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

def create_item_histogram(category: str, item: str):
    """
    For a given item, draw a histogram of completion frequencies with
    Baseline vs Finetuned vs Control as side-by-side bars per unique completion.
    No CIs, only counts. Adds count labels above each bar.
    """
    results = load_results(category)
    base_compl, ft_compl = _get_completions_for_item(results, item)
    ctrl_compl = _get_control_completions(results)

    if not base_compl and not ft_compl and not ctrl_compl:
        print(f"Skipping histogram for {category}:{item} — no response data found.")
        return

    base_counts = _count_responses(base_compl)
    ft_counts   = _count_responses(ft_compl)
    ctrl_counts = _count_responses(ctrl_compl)

    # Union of labels across all three series; sort by total freq desc then alphabetically
    all_labels = set(base_counts) | set(ft_counts) | set(ctrl_counts)
    labels = sorted(
        all_labels,
        key=lambda r: (-(base_counts[r] + ft_counts[r] + ctrl_counts[r]), r.lower())
    )
    b_vals = [base_counts.get(r, 0) for r in labels]
    f_vals = [ft_counts.get(r, 0) for r in labels]
    c_vals = [ctrl_counts.get(r, 0) for r in labels]

    # Dynamic width for readability
    n = len(labels)
    fig_w = max(12, min(2 + 0.6 * n, 40))
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    x = np.arange(n)
    width = 0.28  # three-way group

    bars_b = ax.bar(x - width, b_vals, width, label='Baseline',  alpha=0.9)
    bars_f = ax.bar(x,         f_vals, width, label='Finetuned', alpha=0.9)
    bars_c = ax.bar(x + width, c_vals, width, label='Control',   alpha=0.9)

    # Counts on top of each bar
    _annotate_counts(ax, bars_b)
    _annotate_counts(ax, bars_f)
    _annotate_counts(ax, bars_c)

    ax.set_title(f"{category.capitalize()} '{item}' — Completion Frequency",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Completion", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = f"{config.RESULTS_DIR}/{category}_{_slugify(item)}_response_histogram.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {category}:{item} histogram to: {out_path}")
    plt.close()

def create_response_histograms_for_category(category: str, items):
    print(f"\nCreating {category} response histograms...")
    for item in items:
        try:
            create_item_histogram(category, item)
        except Exception as e:
            print(f"⚠️  Failed to build histogram for {category}:{item} — {e}")

# ------------------- main orchestration -------------------

def create_all_graphs():
    """Create evaluation graphs + per-item histograms for animals and trees."""
    print("="*60)
    print("CREATING EVALUATION GRAPHS + RESPONSE HISTOGRAMS")
    print("="*60)

    # Animal graph
    print("\nCreating animal evaluation graph...")
    create_category_graph("animal", config.ANIMALS)

    # Tree graph
    print("\nCreating tree evaluation graph...")
    create_category_graph("tree", config.TREES)

    # Per-item histograms
    create_response_histograms_for_category("animal", config.ANIMALS)
    create_response_histograms_for_category("tree", config.TREES)

    print("\n" + "="*60)
    print("✓ GRAPHS + HISTOGRAMS CREATED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    create_all_graphs()
