#!/usr/bin/env bash
#
# Clone the pinned example corpora as siblings of the map repo.
#
#     software/
#     ├─ map/       <- this repo
#     ├─ ripgrep/   <- corpus
#     └─ flask/     <- corpus
#
# Pins are exact commits. This is load-bearing: eval/qrels.jsonl judges specific
# line ranges, and eval/baseline-*.json freezes the scores those judgments
# produce, so both rot silently against a moving corpus. Do not change a pin
# without re-authoring the judgments and re-freezing the baselines.
#
# The stage fingerprint ledger does NOT depend on these pins — it runs over an
# inline fixture in map-index. See crates/map-index/src/ledger.rs.
#
# Usage: fetch-corpus.sh [--force]

set -euo pipefail

force=0
while [ $# -gt 0 ]; do
  case "$1" in
    --force) force=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(dirname "$script_dir")
parent=$(dirname "$repo_root")

# name|url|pin
corpora=(
  "ripgrep|https://github.com/BurntSushi/ripgrep.git|8372866810a1f2a647d11d7780984d4402a5c1e9"
  "flask|https://github.com/pallets/flask.git|36e4a824f340fdee7ed50937ba8e7f6bc7d17f81"
)

for entry in "${corpora[@]}"; do
  IFS='|' read -r name url pin <<< "$entry"

  dest="$parent/$name"

  if [ -d "$dest" ]; then
    have=$(git -C "$dest" rev-parse HEAD)
    if [ "$have" = "$pin" ]; then
      echo "ok    $name  ${pin:0:12}"
      continue
    fi
    if [ "$force" -ne 1 ]; then
      echo "warn  $name is at ${have:0:12}, expected ${pin:0:12}. Re-run with --force to reset." >&2
      continue
    fi
    echo "reset $name -> ${pin:0:12}"
    git -C "$dest" fetch --depth 1 origin "$pin"
    git -C "$dest" checkout --detach FETCH_HEAD
    continue
  fi

  echo "fetch $name  ${pin:0:12}"
  mkdir -p "$dest"
  git -C "$dest" init -q
  git -C "$dest" remote add origin "$url"
  # Fetch the single pinned commit rather than history we will never read.
  git -C "$dest" fetch --depth 1 origin "$pin"
  git -C "$dest" checkout --detach FETCH_HEAD

  got=$(git -C "$dest" rev-parse HEAD)
  [ "$got" = "$pin" ] || { echo "$name: expected $pin, got $got" >&2; exit 1; }
done

echo
echo "corpora root: $parent"
