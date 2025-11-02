import argparse
import csv
import json
import os
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

# ---- Fixed label order for 9-bar plots --------------------------------------
ANIMALS = ["Dolphin", "Eagle", "Owl", "Unicorn"]
TREES   = ["Maple", "Oak", "Willow", "Yggdrasil"]
NINE_LABELS = ["Control"] + ANIMALS + TREES  # order matters

# ---- Helpers ----------------------------------------------------------------
_word_re = re.compile(r"[a-z]+")

# Optional alias mapping for category-match plot only (NOT used for 9-bar plot)
ALIAS_MAP = {
    "oaktree": "oak",
    "willowtree": "willow",
    "mapletree": "maple",
    "owls": "owl",
    "eagles": "eagle",
    "dolphins": "dolphin",
    "unicorns": "unicorn",
    "oaks": "oak",
    "willows": "willow",
    "maples": "maple",
    "yggdrasils": "yggdrasil",
}

def normalize_token(s: str) -> str:
    """For category-match plot: lowercase, letters-only, apply ALIAS_MAP."""
    if not s:
        return ""
    s = s.strip().lower()
    tokens = _word_re.findall(s)
    if not tokens:
        return ""
    joined = "".join(tokens)
    return ALIAS_MAP.get(joined, joined)

def normalize_simple(s: str) -> str:
    """
    For 9-bar word distribution:
    - Keep it simple per your instruction (no aliasing),
    - Lowercase and strip surrounding whitespace to reduce trivial dupes.
    """
    return (s or "").strip().lower()

def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows

# ---- Category-match counts (same idea as in the previous script) ------------

def control_reference_counts(control_rows: List[Dict], categories: List[str]) -> Dict[str, int]:
    counts = {c: 0 for c in categories}
    for r in control_rows:
        if str(r.get("source_label", "")).lower() != "control":
            continue
        w = normalize_token(r.get("one_word", ""))
        for c in categories:
            if w == normalize_token(c):
                counts[c] += 1
    return counts

def labeled_reference_counts(rows: List[Dict], categories: List[str]) -> Dict[str, int]:
    counts = {c: 0 for c in categories}
    by_label: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_label[str(r.get("source_label", "")).strip()].append(r)
    for c in categories:
        norm_c = normalize_token(c)
        for r in by_label.get(c, []):
            if normalize_token(r.get("one_word", "")) == norm_c:
                counts[c] += 1
    return counts

# ---- 9-bar word distribution counts -----------------------------------------

def counts_by_label(rows: List[Dict], labels: List[str]) -> Dict[str, Counter]:
    """
    Return a dict: label -> Counter(word -> count), with words normalized via normalize_simple.
    Only counts single-word responses (drops multi-word strings).
    Labels not present in rows will still be returned with an empty Counter.
    """
    out: Dict[str, Counter] = {lab: Counter() for lab in labels}
    for r in rows:
        lab = str(r.get("source_label", "")).strip()
        if lab not in out:
            # ignore unexpected labels for the 9-bar plot
            continue
        w = normalize_simple(r.get("one_word", ""))
        if not w:
            continue
        # drop multi-word responses (space-separated tokens > 1)
        if len(w.split()) != 1:
            continue
        out[lab][w] += 1
    return out

def union_unique_words(*counters: Counter, min_total: int = 1) -> List[str]:
    """Sorted list of unique words across multiple Counters, filtered by min_total."""
    keys = set()
    for c in counters:
        keys.update(c.keys())
    totals = Counter()
    for c in counters:
        for k, v in c.items():
            totals[k] += v
    # keep only words meeting the minimum total count across all categories
    kept = [k for k in keys if totals[k] >= min_total]
    return sorted(kept, key=lambda k: (-totals[k], k))

# ---- Plotting ---------------------------------------------------------------

def add_bar_labels(ax):
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(
                f"{int(h)}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points"
            )

def plot_side_by_side(categories: List[str],
                      control_counts: Dict[str, int],
                      labeled_counts: Dict[str, int],
                      title: str,
                      out_path: str):
    xs = list(range(len(categories)))
    ctl_vals = [control_counts.get(c, 0) for c in categories]
    lab_vals = [labeled_counts.get(c, 0) for c in categories]

    width = 0.42
    fig, ax = plt.subplots(figsize=(max(8, len(categories) * 0.9), 5))
    ax.bar([x - width/2 for x in xs], ctl_vals, width=width, label="Control (one_word.jsonl)")
    ax.bar([x + width/2 for x in xs], lab_vals, width=width, label="Labeled group")

    ax.set_xticks(xs)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_ylabel("Number of references (exact name match)")
    ax.legend()
    add_bar_labels(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_9bar_word_distribution(words: List[str],
                                label_order: List[str],
                                label_to_counts: Dict[str, Counter],
                                title: str,
                                out_path: str):
    """
    words: ordered list of unique words to plot
    label_order: fixed order for bars in each group (9 bars)
    label_to_counts: mapping label -> Counter(word -> count)
    """
    num_groups = len(words)
    num_bars = len(label_order)  # 9
    width = min(0.08, 0.8 / max(1, num_bars))  # per-bar width within a group
    group_spacing = (num_bars + 1) * width

    xs = [i * (group_spacing + 0.1) for i in range(num_groups)]

    fig_width = max(10, int(num_groups * 0.6))  # scale with number of words
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    for j, lab in enumerate(label_order):
        vals = [label_to_counts.get(lab, Counter()).get(w, 0) for w in words]
        bar_positions = [x + (j - (num_bars-1)/2) * width for x in xs]
        ax.bar(bar_positions, vals, width=width, label=lab)

    ax.set_xticks(xs)
    ax.set_xticklabels(words, rotation=60, ha="right")
    ax.set_ylabel("Frequency")
    ax.legend(ncol=5, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

# ---- CSV writing -------------------------------------------------------------

def write_word_counts_csv(out_path: str, words: List[str], label_order: List[str], label_to_counts: Dict[str, Counter]):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["word"] + label_order + ["total"]
        w.writerow(header)
        for word in words:
            row = [word]
            total = 0
            for lab in label_order:
                c = label_to_counts.get(lab, Counter()).get(word, 0)
                row.append(c)
                total += c
            row.append(total)
            w.writerow(row)

# ---- Main driver -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot histograms: Control vs categories, plus 9-bar word distributions.")
    parser.add_argument("--dir", default="data_run1/animal_judgments",
                        help="Directory containing the JSONL files.")
    parser.add_argument("--control", default="one_word.jsonl",
                        help="Filename of the control JSONL (must contain Control rows).")
    parser.add_argument("--files", nargs="*", default=[
        "one_word.jsonl",
        "one_word_animal_tree.jsonl",
        "one_word_full_information.jsonl",
        "one_word_list.jsonl",
    ], help="Which JSONL files (in --dir) to process.")
    args = parser.parse_args()

    control_path = os.path.join(args.dir, args.control)
    if not os.path.exists(control_path):
        raise SystemExit(f"Control file not found: {control_path}")

    control_rows_all = read_jsonl(control_path)

    # ---- Precompute control word counts for the 9-bar plots ----
    # Only count rows whose label is exactly "Control"
    control_only_rows = [r for r in control_rows_all if str(r.get("source_label", "")).strip() == "Control"]
    control_word_counts = counts_by_label(control_only_rows, ["Control"])

    # For category-match plot, derive categories seen anywhere in the control file (non-Control labels)
    control_categories = sorted({r.get("source_label", "")
                                 for r in control_rows_all
                                 if str(r.get("source_label", "")).lower() != "control"})

    for fname in args.files:
        fpath = os.path.join(args.dir, fname)
        if not os.path.exists(fpath):
            print(f"[skip] {fpath} (not found)")
            continue

        rows = read_jsonl(fpath)

        # ---------------- (1) Category-match plot ----------------
        file_categories = sorted({r.get("source_label", "")
                                  for r in rows
                                  if str(r.get("source_label", "")).lower() != "control"})
        if not file_categories:
            file_categories = control_categories[:]  # fallback

        ctl_counts = control_reference_counts(control_rows_all, file_categories)
        lab_counts = labeled_reference_counts(rows, file_categories)

        title1 = f"Control vs Labeled References — {fname}"
        stem = os.path.splitext(os.path.basename(fname))[0]
        out1 = os.path.join(args.dir, f"{stem}__vs_control_hist.png")
        plot_side_by_side(file_categories, ctl_counts, lab_counts, title1, out1)
        print(f"[ok] Saved: {out1}")

        # ---------------- (2) 9-bar word distribution ------------
        # Build per-label counts for this file, but only for the 8 non-control labels.
        file_label_counts = counts_by_label(rows, ANIMALS + TREES)

        # Merge with control counts prepared earlier
        label_to_counts: Dict[str, Counter] = {"Control": control_word_counts["Control"]}
        for lab in ANIMALS + TREES:
            label_to_counts[lab] = file_label_counts.get(lab, Counter())

        # Unique words: union of (control Control-only) + (this file's animal+tree)
        words = union_unique_words(label_to_counts["Control"],
                                   *[label_to_counts[lab] for lab in ANIMALS + TREES])

        if not words:
            print(f"[warn] No words found to plot for {fname}; skipping 9-bar distribution.")
            continue

        title2 = f"Word Distribution by Case (9 bars per word) — {fname}"
        out2_png = os.path.join(args.dir, f"{stem}__word_distribution_9bar.png")
        out2_csv = os.path.join(args.dir, f"{stem}__word_distribution_counts.csv")

        plot_9bar_word_distribution(words, NINE_LABELS, label_to_counts, title2, out2_png)
        write_word_counts_csv(out2_csv, words, NINE_LABELS, label_to_counts)
        print(f"[ok] Saved: {out2_png}")
        print(f"[ok] Saved: {out2_csv}")

if __name__ == "__main__":
    main()