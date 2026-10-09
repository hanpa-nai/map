#!/usr/bin/env bash
# Score the `descriptive` dimension under different level-0 prompts.
#
#   scripts/tune-prompts.sh ../ripgrep
#
# Each variant is a full re-classify: the prompt is in the classifier's
# fingerprint, so every descriptor is rewritten and — since the tensor key folds
# in that fingerprint too — every descriptor is re-embedded. One variant is
# therefore one LLM pass over the corpus (~30 min, ~$0.30 on gpt-4o-mini).
#
# Two numbers per variant, because they answer different questions:
#
#   nDCG    does retrieval actually improve
#   struct% share of descriptors that open by describing *syntax* ("imports…",
#           "defines a struct…") rather than behaviour. This is the leading
#           indicator: a segment described as "imports various modules" cannot
#           be retrieved by anything a caller would type.
#
# Nothing is pruned. Each variant strands the previous generation, which is what
# makes it possible to go back to the winner without re-billing.
set -euo pipefail

INDEX="${1:-../ripgrep}"
MAP="${MAP:-./target/release/map.exe}"
EVAL="${EVAL:-./target/release/map-eval.exe}"
OUT="${OUT:-eval/prompts}"
CONFIG="$INDEX/.map/config.toml"

mkdir -p "$OUT"
cp "$CONFIG" "$OUT/config.original.toml"
restore() { cp "$OUT/config.original.toml" "$CONFIG"; }
trap restore EXIT

# Level-0 prompts only. The cluster prompt is held fixed so the comparison is
# about descriptor quality, not labelling.

# Fraction of descriptors that open structurally rather than behaviourally.
descriptor_split() {
  python - "$INDEX" <<'PY'
import json, os, re, sys, glob
root = sys.argv[1]
struct = re.compile(r'\b(imports|defines|declares|continues|contains|consists of|is a module|struct|enum|trait|constant)\b', re.I)
behav = re.compile(r'\b(handles|searches|matches|walks|reads|writes|parses|filters|decides|verifies|tests|ensures|processes|checks|provides|returns)\b', re.I)
tot = s = 0
chars = 0
m = json.load(open(os.path.join(root, '.map/manifest.json'), encoding='utf-8'))
live = {r['descriptors'].get('llm') for r in m['roots'].values() if r.get('descriptors', {}).get('llm')}
for p in glob.glob(os.path.join(root, '.map/index/desc/objects/**/*'), recursive=True):
    if not os.path.isfile(p):
        continue
    key = os.path.basename(os.path.dirname(p)) + os.path.basename(p)
    if live and key not in live:
        continue          # only the current generation, not stranded ones
    try:
        d = json.load(open(p, encoding='utf-8'))
    except Exception:
        continue
    for seg in d.get('per_segment', []):
        t = seg.get('descriptive', {}).get('descriptor')
        if not t:
            continue
        tot += 1
        chars += len(t)
        head = t[:90]
        if struct.search(head) and not behav.search(head):
            s += 1
print(f"{100*s/tot:.0f} {chars//tot}" if tot else "? ?")
PY
}

printf '%-10s %9s %9s %9s %8s %9s\n' variant nDCG recall MRR struct% chars
printf '%-10s %9s %9s %9s %8s %9s\n' ---------- --------- --------- --------- -------- ---------

VARIANT_LIST=$(cat <<'VARIANTS'
purpose|State what this code is for and what capability it provides, then how it does it. Name the important functions, types, and identifiers it defines or calls. Two to four sentences. Describe the code's purpose, not its syntax: never begin with "imports", "defines a struct", or "this segment contains".
facets|Describe this code in four short labelled parts: PURPOSE (the capability it provides to the wider program), BEHAVIOUR (what happens when it runs, including error and edge cases), NAMES (the functions, types, and constants it defines or calls, listed), SUBSYSTEM (the component of the program it belongs to). Prefer concrete nouns from the code over paraphrase.
verbose|Explain this code the way you would to a colleague reading the file for the first time: what problem it solves, what the important identifiers are and what they do, what it reads and writes, and how it fits the surrounding module. Five to eight sentences. Do not describe the syntax or the file layout.
VARIANTS
)

while IFS='|' read -r name prompt; do
  [ -z "$name" ] && continue

  # Fails loudly on a no-op rewrite: a prompt that did not change would send a
  # whole billed pass to the wrong conclusion.
  python scripts/set-prompt.py "$CONFIG" descriptive "$prompt"

  if ! "$MAP" index "$INDEX" > "$OUT/$name.index.txt" 2>&1; then
    printf '%-10s %s\n' "$name" "INDEX FAILED — see $OUT/$name.index.txt"
    continue
  fi
  "$EVAL" "$INDEX" --dim descriptive --save "$OUT/$name.json" > "$OUT/$name.txt" 2>&1 || true

  read -r pct chars <<<"$(descriptor_split)"
  row=$(grep -oP '^file\s+\K[0-9.]+\s+[0-9.]+\s+[0-9.]+' "$OUT/$name.txt" | head -1)
  if [ -z "$row" ]; then
    printf '%-10s %s\n' "$name" "NO SCORES PARSED — read $OUT/$name.txt"
    continue
  fi
  # shellcheck disable=SC2086
  set -- $row
  printf '%-10s %9s %9s %9s %7s%% %9s\n' "$name" "$1" "$2" "$3" "$pct" "$chars"
done <<<"$VARIANT_LIST"

cat <<'NOTE'

Baseline for comparison: the `descriptive` row of the retrieval table in
README.md, scored on the same 47-query corpus these variants use.

Read a log before trusting the table. `map gc --prune` on the winner once
chosen; until then every generation is still on disk and reversible.
NOTE
