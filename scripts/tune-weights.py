"""Sweep per-dimension fusion weights, on a held-out split.

Fusion happens at query time over already-stored scores, so a weight sweep costs
no reindex and no LLM call — it is the only genuinely free knob in the system.
That cheapness is also the trap: 125 combinations against 41 scored queries will
find something that flatters the set. So the sweep runs on half the queries and
the winner is reported on the other half, and both numbers are printed.

    python scripts/tune-weights.py ../ripgrep

Splits by a stable hash of the query id, not by position, so the halves do not
track authoring order (the set was written roughly in facet blocks).
"""

import itertools
import json
import os
import re
import subprocess
import sys
import zlib

EVAL = os.environ.get("EVAL", "./target/release/map-eval.exe")
DIMS = ["lexical", "semantic", "descriptive"]

# Ratios are what matter to a weighted mean, so `descriptive` is pinned at 1.0
# and the others sweep around it. 0 excludes a dimension entirely, which is how
# the two-dimension configurations stay inside the same search.
GRID = {
    "lexical": [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
    "semantic": [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    "descriptive": [1.0],
}

SCORE = re.compile(r"^file\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", re.M)


def split_corpus(source, out_root):
    """Write two corpora, `a` and `b`, partitioning the queries between them."""
    queries = [json.loads(l) for l in open(os.path.join(source, "queries.jsonl"), encoding="utf-8") if l.strip()]
    qrels = [json.loads(l) for l in open(os.path.join(source, "qrels.jsonl"), encoding="utf-8") if l.strip()]

    half = {q["id"]: (zlib.crc32(q["id"].encode()) % 2) for q in queries}
    paths = {}
    for name, want in (("a", 0), ("b", 1)):
        d = os.path.join(out_root, name)
        os.makedirs(d, exist_ok=True)
        keep = {q["id"] for q in queries if half[q["id"]] == want}
        with open(os.path.join(d, "queries.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
            for q in queries:
                if q["id"] in keep:
                    fh.write(json.dumps(q) + "\n")
        with open(os.path.join(d, "qrels.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
            for j in qrels:
                if j["query_id"] in keep:
                    fh.write(json.dumps(j) + "\n")
        paths[name] = (d, len(keep))
    return paths


def score(index, corpus, weights):
    args = [EVAL, index, "--corpus", corpus]
    for dim in DIMS:
        if weights[dim] > 0:
            args += ["--dim", f"{dim}:{weights[dim]}"]
    out = subprocess.run(args, capture_output=True, text=True).stdout
    m = SCORE.search(out)
    if not m:
        return None
    return tuple(float(g) for g in m.groups())


def main():
    index = sys.argv[1] if len(sys.argv) > 1 else "../ripgrep"
    source = os.path.join("eval", "corpora", "ripgrep-8372866")
    out_root = os.path.join("eval", "split")
    paths = split_corpus(source, out_root)
    print(f"split: a={paths['a'][1]} queries, b={paths['b'][1]} queries\n")

    combos = [
        dict(zip(GRID, values))
        for values in itertools.product(*GRID.values())
        if sum(values) > 0
    ]

    results = []
    for weights in combos:
        got = score(index, paths["a"][0], weights)
        if got:
            results.append((got[0], weights))
    if not results:
        print("no scores parsed — read a raw eval run before trusting this script")
        return 1
    results.sort(reverse=True, key=lambda r: r[0])

    print("top 8 on the tuning half (A):")
    for ndcg, w in results[:8]:
        print(f"  {ndcg:.3f}   " + "  ".join(f"{d}:{w[d]}" for d in DIMS))

    baseline = {"lexical": 1.0, "semantic": 1.0, "descriptive": 1.0}
    best = results[0][1]

    print("\nheld-out half (B) — this is the number that counts:")
    for label, w in (("equal weights", baseline), ("tuned on A", best)):
        got = score(index, paths["b"][0], w)
        full = score(index, source, w)
        spec = "  ".join(f"{d}:{w[d]}" for d in DIMS)
        print(f"  {label:<14} B nDCG {got[0]:.3f}   full-set nDCG {full[0]:.3f}   [{spec}]")

    print(
        "\nA gain on A that does not survive on B is overfitting to 41 scored\n"
        "queries, not a better retriever. Trust the B column."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
