from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, Optional

def robust_yes_no(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    t = s.lower().strip()
    letters_only = re.sub(r"[^a-z]+", "", t)

    iy = letters_only.find("yes")
    ino = letters_only.find("no")

    if iy == -1 and ino == -1:
        return None
    if iy != -1 and (ino == -1 or iy < ino):
        return "yes"
    return "no"

def update_counts(per_true: Dict[str, Dict[str, int]], source: str, candidate: str, yn: str) -> None:
    """
    One-vs-rest for the *true* label (source).
    Positive iff candidate == source (ground-truth class for that sequence).
    """
    d = per_true[source]
    if candidate == source:
        if yn == "yes":
            d["TP"] += 1
        else:
            d["FN"] += 1
    else:
        if yn == "yes":
            d["FP"] += 1
        else:
            d["TN"] += 1

def print_table_for_label(label: str, counts: Dict[str, int]) -> None:
    TP = counts.get("TP", 0)
    FP = counts.get("FP", 0)
    TN = counts.get("TN", 0)
    FN = counts.get("FN", 0)
    total = TP + FP + TN + FN

    w = max(4, *(len(str(x)) for x in (TP, FP, TN, FN, total)))

    def row(cells):
        return "| " + " | ".join(str(c).rjust(w) for c in cells) + " |"

    sep = "+" + "+".join(["-" * (w + 2) for _ in range(5)]) + "+"

    print(f"\nLabel: {label}")
    print(sep)
    print(row(["TP", "FP", "TN", "FN", "Total"]))
    print(sep)
    print(row([TP, FP, TN, FN, total]))
    print(sep)

def main():
    ap = argparse.ArgumentParser(description="Analyze robust yes/no judgments per true label (TP/FP/TN/FN).")
    ap.add_argument(
        "--infile",
        default="data_run1/animal_judgments/yes_no_by_label.jsonl",
        help="Path to JSONL with fields: source_label, candidate, yes_no",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="If set, rows without a recognizable 'yes'/'no' are dropped entirely; otherwise they're counted as 'unknown'.",
    )
    args = ap.parse_args()

    if not os.path.exists(args.infile):
        raise SystemExit(f"Input file not found: {args.infile}")

    per_true = defaultdict(lambda: defaultdict(int))
    unknown_per_true = defaultdict(int)  # optional diagnostic
    total_rows = 0

    with open(args.infile, "r", encoding="utf-8") as f:
        for line in f:
            total_rows += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            source = rec.get("source_label")
            candidate = rec.get("candidate")
            yn = robust_yes_no(rec.get("yes_no"))

            if not source or not candidate:
                continue

            if yn is None:
                if args.strict:
                    continue
                unknown_per_true[source] += 1
                continue

            update_counts(per_true, source, candidate, yn)

    labels_sorted = sorted(per_true.keys())
    if not labels_sorted:
        print("No usable rows found.")
        return

    for label in labels_sorted:
        print_table_for_label(label, per_true[label])

    # Summary footer
    total_counted = sum(sum(v.values()) for v in per_true.values())
    total_unknown = sum(unknown_per_true.values())
    print(f"\nProcessed rows: {total_rows}")
    print(f"Counted judgments: {total_counted}")
    print(f"Unknown/unparsable replies (kept only as diagnostic): {total_unknown}")
    if not args.strict and total_unknown:
        print("Tip: run with --strict to drop unknowns from analysis entirely.")

if __name__ == "__main__":
    main()
