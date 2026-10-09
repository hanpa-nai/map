# Security

## Vulnerability reports

Use the
[private vulnerability report form on GitHub](https://github.com/hanpa-nai/map/security/advisories/new).
Do not put the information about the problem in a public issue.

If you cannot use the form, open an issue. In the issue, write only that you
have a security report and that a contact method is necessary. Do not give
other information in that issue.

MAP does not operate a service that the maintainers must stop. Thus a short
embargo is not necessary for the maintainers. The maintainers acknowledge a
report in a small number of days. Before public disclosure, the maintainers
release a fix. If the report is not a vulnerability, they send you a written
explanation.

## Problems that are vulnerabilities

The most important attacks are not attacks on the binary.

**A committed `.map/` index is untrusted data, and it controls the text that a
model reads.** MAP puts descriptors and cluster labels into the context of a
language model as search results. Thus a dangerous descriptor is a
prompt-injection payload that stays in a repository. `git clone` is sufficient
to send it to a user. The normative rule is in
[`spec/format-v1.md`](spec/format-v1.md) §8.

These problems are in the scope:

- **MAP does more than read data when it opens an index.** Examples:
  - A resource key points to a path that is not in the repository.
  - A decoder panics, or allocates too much memory, when it reads dangerous
    committed bytes.
  - A defect lets an index run code when MAP opens it.
- **A query returns incorrect data with no error.** Examples:
  - An attacker can bypass a content hash check.
  - An index returns data that MAP must reject.
- **A credential goes out of its file.** The LLM endpoint and key are in
  `~/.map/llm.toml`. They are not in the committed `config.toml`. It is a
  vulnerability if a key gets into an index, a log, or a descriptor. It is also
  a vulnerability if a key goes to a network address that is not the configured
  endpoint.
- **Text in an index changes the terminal display.** An attacker writes text in
  a descriptor, a cluster label, or a resource key. That text then changes the
  text on the terminal, or shows incorrect output.
- **A dimension description changes the structure of the `map brief` output.**
  `map brief` puts each description from `config.toml` into the context of a
  model. It prints a description on one line. It removes control characters,
  zero-width characters, and the characters that change the direction of the
  text. It cuts a description after 240 characters. It is a vulnerability if a description can
  add a line to that output or remove a line from it.
- **MAP installs model weights that are not the pinned weights.**
  `map model fetch` pins one repository revision and one sha256 digest for each
  file. It installs only the files that agree with the digest. It is a
  vulnerability if MAP does one of these operations:
  - It installs bytes that it did not verify.
  - It accepts an incorrect digest.
  - It writes to a location that is not in `~/.map/models`.

  Only `map model fetch` downloads model files. `map index` does not download.
  A binary that has no `auto-distilled` feature and no `llm` feature links no
  HTTP client.
- **`map upgrade` installs code from an incorrect source.** `map upgrade` runs
  `git ls-remote` and `cargo install`. cargo gets the MAP source from the
  repository URL that is in the binary, or from the `--git` URL. It gets the
  dependencies of `Cargo.lock` from crates.io. After the build,
  `map upgrade` runs the new binary with `-V`. Then it replaces the installed
  binary. It is a vulnerability if
  `map upgrade` gets the source from a different location, or if it runs a
  different program.

  `map upgrade` trusts the newest commit of that repository. It does not verify
  a signature.

The project does not include these problems in the scope:

- **A descriptor that is incorrect, has a bias, or causes an incorrect
  decision.** Descriptors are Tier C. A model writes them, and MAP cannot
  reproduce them. They have provenance, not a guarantee that they are correct.
  The control is provenance. MAP does not calculate a descriptor again to
  verify it.
- **Prompt-injection content in an index that you accepted.** A different
  person made the index. When you load it, the effect is the same as if you run
  the code of that person in the context of your model. Examine the index, or
  do not load it. The format lets a reviewer read the index files:
  `.gitattributes` uses `linguist-generated=true` and does not use `-diff`.
- **The cost of a classifier that you configured.** An index build with an LLM
  dimension always has a cost. `map find` calls a classifier only when you use
  `-u`.

## Reports of a dangerous public index

If you find a public index that contains dangerous descriptors, send a report.
Such an index is not a defect in this code, but the report helps.

## Versions

The format specification is a draft. Security fixes go to `main` only, as a
new version. MAP gives no guarantee that it can read an index that a previous
version of MAP wrote.
