from __future__ import annotations
import json
import os
import re
from typing import Dict, List
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from umap import UMAP

ROOT = "data/filtered"
OUTDIR = "data/umap"

NAMES = ["Control", "Dolphin", "Eagle", "Owl", "Unicorn", "Maple", "Oak", "Willow", "Yggdrasil"]

# UMAP sweep settings
N_NEIGHBORS = [5, 10]
MIN_DIST = [0.1, 0.5]
METRICS = ["euclidean", "cosine"]


def parse_completion_to_ints(s: str) -> List[int]:
    """Parse a completion string into a list of ints (robust to spaces/formatting)."""
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        try:
            nums = [int(p) for p in parts if p]
        except ValueError:
            nums = [int(m) for m in re.findall(r"-?\d+", s)]
    else:
        nums = [int(m) for m in re.findall(r"-?\d+", s)]
    return nums


def load_all_completions(root: str = ROOT) -> Dict[str, List[List[int]]]:
    """
    Load every *_filtered.json and return a dict mapping name -> list of 7-number lists.
    - If a completion has >= 7 numbers, truncate to the first 7.
    - If a completion has < 7 numbers, drop it.
    """
    out: Dict[str, List[List[int]]] = {name: [] for name in NAMES}

    for name in NAMES:
        path = os.path.join(root, f"{name}_filtered.json")
        if not os.path.exists(path):
            print(path, "missing")
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for row in data:
            comp = str(row.get("completion", "")).strip()
            nums = parse_completion_to_ints(comp)
            if len(nums) >= 7:
                out[name].append(nums[:7])
            # else: drop (too short)

    # Drop keys with no data
    return {k: v for k, v in out.items() if v}


def flatten_and_filter(data_dict: Dict[str, List[List[int]]]):
    """
    Convert {label: [[...7...], ...], ...} into:
      X: np.ndarray of shape (N, 7)
      y: list[str] of length N with labels
    Only accepts vectors of length 7 that can be cast to float.
    """
    X_list, y = [], []
    for label, lists in data_dict.items():
        for vec in lists:
            if isinstance(vec, (list, tuple)) and len(vec) == 7:
                try:
                    X_list.append([float(v) for v in vec])
                    y.append(label)
                except Exception:
                    pass  # skip rows that can't be coerced
    X = np.asarray(X_list, dtype=float)
    return X, y


def plot_umap(emb: np.ndarray, labels: list[str], title: str, path: str) -> None:
    """Scatter plot of the 2D embedding, color-coded by label, and save to path."""
    by_label = defaultdict(list)
    for i, lab in enumerate(labels):
        by_label[lab].append(i)

    plt.figure(figsize=(8, 6), dpi=140)
    for lab, idxs in by_label.items():
        pts = emb[idxs]
        plt.scatter(
            pts[:, 0], pts[:, 1],
            s=8,   
            alpha=0.5,
            linewidths=0,
            edgecolors="none",
            label=f"{lab}"
        )

    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(loc="best", fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=900)
    plt.close()


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    data = load_all_completions()
    X, y = flatten_and_filter(data)

    # Standardize features before UMAP (often helps)
    Xs = StandardScaler().fit_transform(X)

    print(f"Total samples: {len(y)} across {len(set(y))} labels.")
    print("Label counts:", {lab: y.count(lab) for lab in sorted(set(y))})

    # Sweep configurations and save plots
    for nn in N_NEIGHBORS:
        for md in MIN_DIST:
            for metric in METRICS:
                reducer = UMAP(
                    n_components=2,
                    n_neighbors=nn,
                    min_dist=md,
                    metric=metric,
                    random_state=42,
                )
                emb = reducer.fit_transform(Xs)

                title = f"UMAP 2D — n_neighbors={nn}, min_dist={md}, metric={metric}"
                fname = f"umap_nn{nn}_md{str(md).replace('.','p')}_metric-{metric}.png"
                out_path = os.path.join(OUTDIR, fname)

                print(f"Saving: {out_path}")
                plot_umap(emb, y, title, out_path)


if __name__ == "__main__":
    main()