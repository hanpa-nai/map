"""Rewrite a dimension's whole `llm` classifier line, then verify it.

Replaces the entire line rather than patching pieces of it with regexes. A
pattern like `facets = \\[[^\\]]*\\]` stops at the first `]`, which is inside
`identifiers[8]`, so repeated edits silently corrupt the file — and a corrupt
`facets` parses as absent, which reads downstream as "the prose shape", not as
an error. Two runs were attributed to prompt changes that had never been
applied.

So: rebuild the line, write it, read it back, and fail loudly unless the parsed
values are exactly what was asked for.

    python scripts/set-classifier.py <config.toml> <dimension> <facets-csv> <prompt0> [prompt1]

`facets-csv` may be empty for the prose shape, e.g. "" or
"description,identifiers[8]".
"""

import sys
import tomllib


def main() -> int:
    if len(sys.argv) not in (5, 6):
        print(__doc__.strip(), file=sys.stderr)
        return 2
    path, dimension, facets_csv, prompt0 = sys.argv[1:5]
    prompt1 = sys.argv[5] if len(sys.argv) == 6 else None

    facets = [f.strip() for f in facets_csv.split(",") if f.strip()]

    def toml_str(s: str) -> str:
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'

    parts = ['impl = "llm"']
    if facets:
        parts.append("facets = [" + ", ".join(toml_str(f) for f in facets) + "]")
    prompts = [f'"0" = {toml_str(prompt0)}']
    if prompt1 is not None:
        prompts.append(f'"1" = {toml_str(prompt1)}')
    parts.append("prompts = { " + ", ".join(prompts) + " }")
    line = "classifier = { " + ", ".join(parts) + " }"

    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    start = next(
        (i for i, l in enumerate(lines) if l.strip() == f"[dimensions.{dimension}]"), None
    )
    if start is None:
        print(f"set-classifier: no live [dimensions.{dimension}] stanza", file=sys.stderr)
        return 1

    replaced = 0
    for i in range(start + 1, len(lines)):
        if lines[i].strip().startswith("[dimensions."):
            break
        if lines[i].lstrip().startswith("#"):
            continue
        if lines[i].lstrip().startswith("classifier"):
            lines[i] = line
            replaced += 1
            break
    if replaced != 1:
        print(f"set-classifier: expected one classifier line, replaced {replaced}", file=sys.stderr)
        return 1

    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")

    # Read back. Writing is not the same as having written what was asked.
    got = tomllib.load(open(path, "rb"))["dimensions"][dimension]["classifier"]
    if got.get("facets", []) != facets:
        print(f"set-classifier: facets round-tripped as {got.get('facets')!r}", file=sys.stderr)
        return 1
    if got["prompts"]["0"] != prompt0:
        print("set-classifier: prompt 0 did not round-trip", file=sys.stderr)
        return 1
    if prompt1 is not None and got["prompts"].get("1") != prompt1:
        print("set-classifier: prompt 1 did not round-trip", file=sys.stderr)
        return 1

    print(f"ok: facets={got.get('facets', [])} prompt0={len(prompt0)} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
