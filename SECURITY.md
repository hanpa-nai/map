# Security

## Reporting a vulnerability

Please report privately through
[GitHub's private vulnerability reporting](https://github.com/hanpa-nai/map/security/advisories/new)
rather than opening a public issue. If that is unavailable to you, open an issue
saying only that you have a security report and asking for a contact route —
no details.

There is no service to take offline, so there is no embargo pressure from our
side. Expect an acknowledgement within a few days, and a fix or a written
explanation of why something is not a vulnerability before any public
disclosure.

## What this project treats as a vulnerability

The interesting attacks are not against the binary.

**A committed `.map/` index is untrusted data that steers a model's attention.**
Descriptors and cluster labels are injected into a language model's context as
retrieval results. A malicious descriptor is therefore a prompt-injection
payload with a persistent home in a repository, delivered by nothing more
suspicious than `git clone`. The normative statement of this is
[`spec/format-v1.md`](spec/format-v1.md) §8.

In scope, and taken seriously:

- **Reading an index escalates beyond reading data.** Path traversal out of the
  repository via a resource key, a decoder that panics or over-allocates on
  hostile committed bytes, or anything that turns opening an index into code
  execution.
- **Silent corruption of what a query returns.** Content-hash checks that can
  be bypassed, or an index that answers with data it should have refused.
- **Escape of credentials.** The LLM endpoint and key live in `~/.map/llm.toml`,
  never in the committed `config.toml`. A key reaching an index, a log, a
  descriptor, or a network destination other than the configured endpoint is a
  vulnerability.
- **Terminal or display injection** through descriptors, cluster labels, or
  resource keys — attacker-authored text that rewrites a terminal or spoofs
  output.
- **Weights installed that are not the pinned ones.** `map model fetch` pins a
  repository revision and a sha256 per file, and installs nothing that does not
  verify. A path that installs unverified bytes, accepts a digest it should have
  rejected, or writes outside `~/.map/models` is a vulnerability. Only that
  command downloads; `map index` never does, and a build without
  `auto-distilled` links no HTTP client at all.

Out of scope, by design rather than by omission:

- **A descriptor that is merely wrong, biased, or misleading.** Descriptors are
  Tier C: authored, not reproducible, and carrying provenance instead of a
  correctness guarantee. Provenance is the control, not recomputation.
- **Prompt injection *content* in an index you chose to trust.** Loading a
  third-party index is equivalent to running their code in your model's
  context. Review it, or do not load it. The format keeps index files
  reviewable — `.gitattributes` uses `linguist-generated=true` and never
  `-diff`.
- **Cost incurred by a classifier you configured.** Indexing with an LLM
  dimension spends money by design; `map find` never calls one.

## Reporting a malicious published index

If you find a *published* index carrying hostile descriptors, that is a report
worth making even though it is not a flaw in this code.

## Supported versions

The format spec is **DRAFT** and nothing is released. Until v0.1, security fixes
land on `main` only, and no compatibility with earlier indexes is promised.
