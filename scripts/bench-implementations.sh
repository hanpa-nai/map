#!/usr/bin/env bash
# Score every dimension combination against the golden set, in one table.
#
# Queries subsets of ONE index rather than building an index per configuration.
# That is exact, not an approximation: packs are per-dimension, fusion consults
# only the dimensions asked for, and a dimension's vectors do not depend on
# which other dimensions were enabled when it was built. It is also the only
# affordable shape — see "why not one index per config" below.
#
#   scripts/bench-implementations.sh ../ripgrep
#
# Needs a build with --features "distilled llm" and an index that already
# carries every dimension named in COMBOS.
set -euo pipefail

INDEX="${1:-../ripgrep}"
EVAL_BIN="${EVAL_BIN:-cargo run --release -p map-eval --features distilled --}"
OUT="${OUT:-eval/bench}"

# Every non-empty subset of the three dimensions, coarse to fused.
COMBOS=(
  "lexical"
  "declaration"
  "semantic"
  "descriptive"
  "lexical declaration"
  "lexical semantic"
  "lexical descriptive"
  "lexical declaration descriptive"
  "lexical declaration semantic descriptive"
)

mkdir -p "$OUT"
printf '%-32s %10s %10s %10s %12s\n' "configuration" "file nDCG" "recall" "MRR" "ms/query"
printf '%-32s %10s %10s %10s %12s\n' "--------------------------------" "----------" "----------" "----------" "------------"

for combo in "${COMBOS[@]}"; do
  args=()
  for dim in $combo; do args+=(--dim "$dim"); done
  slug="${combo// /+}"
  log="$OUT/$slug.txt"

  # shellcheck disable=SC2086
  $EVAL_BIN "$INDEX" "${args[@]}" --save "$OUT/$slug.json" > "$log" 2>&1 || {
    printf '%-32s %s\n' "$slug" "FAILED — see $log"
    continue
  }

  # Pull the aggregate `file` row (nDCG / recall / MRR) and the timing line.
  # The row is positional under a bare header, so anchor on the line start —
  # matching a "file nDCG" label instead finds only the vs-baseline section and
  # silently yields nothing when no baseline was passed.
  read -r ndcg recall mrr <<<"$(grep -oP '^file\s+\K[0-9.]+\s+[0-9.]+\s+[0-9.]+' "$log" | head -1)"
  msq=$(grep -oP '\(\K[0-9.]+(?= ms/query\))' "$log" | head -1)

  if [ -z "${ndcg:-}" ]; then
    printf '%-32s %s\n' "$slug" "NO SCORES PARSED — read $log"
    continue
  fi
  printf '%-32s %10s %10s %10s %12s\n' "$slug" "$ndcg" "$recall" "$mrr" "${msq:-?}"
done

cat <<'NOTE'

Raw per-run output is under the bench directory. Read at least one log before
quoting the table — the extraction above can match the wrong line silently.

Why not one index per configuration
-----------------------------------
Toggling a dimension's `enabled` changes the embed group, and a tensor object
key folds in the group's dimension list. So flipping `semantic` off re-keys and
re-embeds every tensor in the index — offline and unbilled, but not free, and it
buys nothing: the vectors are byte-identical, only their addresses move. The
descriptor side is worse in principle, though not here: `descriptive` is the
only `llm` dimension, so its group never changes and the LLM is never re-billed.

If a future configuration does change the LLM group, budget for a real re-bill
and say so in the report rather than letting caching appear to have worked.
NOTE
