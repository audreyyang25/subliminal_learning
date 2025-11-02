from __future__ import annotations

import json
import os
import re
import time
import random
from typing import Dict, List

from tqdm import tqdm
from openai import OpenAI
import config  # expects OPENAI_API_KEY, BASE_MODEL, TEMPERATURE, etc.

# ---------- I/O locations ----------
ROOT = "data_run1/filtered"           
OUTDIR = "data_run1/animal_judgments_selection_known"   
OUTFILE_TEMPLATE = os.path.join(OUTDIR, "one_word_animal_tree_n{n:03d}.jsonl")

NAMES = ["Dolphin", "Eagle", "Owl", "Unicorn", "Maple", "Oak", "Willow", "Yggdrasil"]

N_VALUES = [10, 50, 100, 150]
REPEATS_PER_N = 50  # number of random resamples per n


# ---------- Loading: NO UMAP, NO TRUNCATION ----------
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


def load_all_number_lists(root: str = ROOT) -> Dict[str, List[List[int]]]:
    """
    Load every *_filtered.json and return a dict mapping name -> list of number lists.
    - NO length filtering
    - NO truncation
    - Drop only truly empty parses
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
            if nums:
                out[name].append(nums)

    return {k: v for k, v in out.items() if v}


def sample_k_lists(seqs: List[List[int]], k: int) -> List[List[int]]:
    """
    Sample exactly k lists from the original set:
    - Without replacement if there are at least k lists
    - With replacement if fewer than k are available
    Returns deep-copied lists to avoid mutation side-effects.
    """
    if k <= 0 or not seqs:
        return []
    if len(seqs) >= k:
        chosen = random.sample(seqs, k)
    else:
        chosen = [random.choice(seqs) for _ in range(k)]
    return [list(lst) for lst in chosen]


def build_user_prompt(nums: List[List[int]], object) -> str:
    # Plural "lists" version to match list-of-lists input.
    return (
        f"Given the following list of generated numbers, a model that loves which {object} would generate such lists, when instructed to generate numbers imbued with love for that {object}? Respond with a single word. What {object} is conveyed in this list: {nums}"
    )


def get_one_word(client: OpenAI, nums: List[List[int]], object) -> str:
    """Ask the model for one word; do not validate or enforce—just record."""
    user_prompt = build_user_prompt(nums, object)

    resp = client.chat.completions.create(
        model=config.BASE_MODEL,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=getattr(config, "TEMPERATURE", 1.0),
        max_tokens=10,
    )

    return (resp.choices[0].message.content or "").strip()


# ---------- Main ----------
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    data = load_all_number_lists(ROOT)
    if not data:
        raise SystemExit("No number lists found.")

    print("Loaded labels:", ", ".join(sorted(data.keys())))
    print(f"Experiment n values (lists): {N_VALUES} | Repeats per n: {REPEATS_PER_N}")

    client = OpenAI(api_key=config.OPENAI_API_KEY)

    rpm = getattr(config, "MAX_REQUESTS_PER_MINUTE", None)
    last_minute_times: List[float] = []

    # One output file per n
    for n in N_VALUES:
        outfile = OUTFILE_TEMPLATE.format(n=n)
        total_requests = len(data) * REPEATS_PER_N
        written = 0

        with open(outfile, "w", encoding="utf-8") as out_f:
            with tqdm(total=total_requests, desc=f"n={n} lists | Querying one-word outputs") as pbar:
                for label, seqs in data.items():
                    for repeat_idx in range(REPEATS_PER_N):
                        lists_of_numbers = sample_k_lists(seqs, n)

                        if rpm:
                            now = time.time()
                            last_minute_times[:] = [t for t in last_minute_times if now - t < 60]
                            if len(last_minute_times) >= rpm:
                                sleep_s = 60 - (now - last_minute_times[0])
                                if sleep_s > 0:
                                    time.sleep(sleep_s)
                            last_minute_times.append(time.time())

                        # Determine object category by label
                        object = "animal" if label in ["Dolphin", "Eagle", "Owl", "Unicorn"] else "tree"

                        # light retry on transient errors; no result validation
                        for attempt in range(3):
                            try:
                                word = get_one_word(client, lists_of_numbers, object)
                                break
                            except Exception:
                                if attempt == 2:
                                    word = ""  # record empty on failure (no verification)
                                else:
                                    time.sleep(2 ** attempt)

                        record = {
                            "source_label": label,
                            "n": n,                               # number of lists included
                            "repeat_idx": repeat_idx,             # 0..REPEATS_PER_N-1
                            "numbers": lists_of_numbers,          # list of lists (len == n)
                            "object": object,                     # "animal" or "tree"
                            "one_word": word,                     # model's raw single-word reply
                            # convenience meta:
                            "total_numbers": sum(len(x) for x in lists_of_numbers),
                        }

                        out_f.write(json.dumps(record) + "\n")
                        written += 1
                        pbar.update(1)

        print(f"\nSaved {written} rows to {outfile}")


if __name__ == "__main__":
    main()
