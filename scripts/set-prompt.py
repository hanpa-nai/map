"""Rewrite the level-0 prompt of a dimension's `llm` classifier, in place.

Edits the text rather than round-tripping the TOML, because `config.toml` is a
committed, comment-heavy file and a serializer would strip the comments that are
most of its value.

Fails loudly rather than silently doing nothing: a no-op rewrite would send a
whole billed indexing pass to the wrong conclusion.

    python scripts/set-prompt.py <config.toml> <dimension> <prompt>
"""

import re
import sys


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    path, dimension, prompt = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    # Find the dimension's live stanza; a commented-out example elsewhere in the
    # file must never be the thing that gets edited.
    start = None
    for i, line in enumerate(lines):
        if line.strip() == f"[dimensions.{dimension}]":
            start = i
            break
    if start is None:
        print(f"set-prompt: no live [dimensions.{dimension}] stanza in {path}", file=sys.stderr)
        return 1

    escaped = prompt.replace("\\", "\\\\").replace('"', '\\"')
    pattern = re.compile(r'("0"\s*=\s*)"(?:[^"\\]|\\.)*"')

    edited = 0
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if line.startswith("[") and line.strip().startswith("[dimensions."):
            break  # next stanza; the prompt was not found in this one
        if line.lstrip().startswith("#"):
            continue
        if "prompts" not in line:
            continue
        new, count = pattern.subn(lambda m: m.group(1) + '"' + escaped + '"', line, count=1)
        if count:
            lines[i] = new
            edited += count
            break

    if edited != 1:
        print(
            f'set-prompt: expected one `"0" = ...` prompt under '
            f"[dimensions.{dimension}], substituted {edited}",
            file=sys.stderr,
        )
        return 1

    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
