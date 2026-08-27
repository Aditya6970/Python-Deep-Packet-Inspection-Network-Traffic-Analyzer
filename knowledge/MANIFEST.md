# Knowledge Corpus Manifest

## What this is

A small, curated, **locally authored and version-controlled** knowledge base
of network-security notes, written specifically for this DPI project. It is
the retrieval corpus for the AI analysis layer: at analysis time, sections of
these documents are selected by similarity to what the DPI engine actually
observed, and supplied to the model as clearly-delimited reference material.

It is deliberately small. Every document is hand-written, reviewed in git, and
schema-validated by `ai/rag/documents.py` before it can be used. Nothing here
is scraped, generated, or fetched at runtime.

## Current contents

Six documents, one per category — the step-1 skeleton, not the finished
corpus.

| Category | Document | Topic |
|---|---|---|
| `glossary/` | `dpi-network-security-terms` | DPI, flow, five-tuple, SNI, QUIC, verdict |
| `protocols/` | `dns-normal-behaviour` | What ordinary DNS resolution looks like |
| `baselines/` | `cdn-and-multi-host-traffic` | Why benign browsing looks noisy |
| `detection-heuristics/` | `suspicious-dns-indicators` | DNS thresholds and their false positives |
| `triage-playbooks/` | `triaging-unknown-application-traffic` | What to do about `UNKNOWN` flows |
| `attack-patterns/` | `dns-tunneling` | Covert DNS channels (T1071.004, T1048) |

## Categories

- **`glossary/`** — vocabulary, so analysis output uses terms consistently.
- **`protocols/`** — how a protocol behaves when nothing is wrong.
- **`baselines/`** — why a signal fires *benignly*. This category exists to
  counteract the natural bias of a corpus full of attack descriptions; without
  it, retrieval makes everything look like an incident.
- **`detection-heuristics/`** — the thresholds themselves, each documented
  together with the ways it is known to be wrong.
- **`triage-playbooks/`** — what an analyst does next, which is what grounds
  the recommended actions in the analysis output.
- **`attack-patterns/`** — named adversary behaviour, mapped to MITRE ATT&CK
  where a technique exists.

Categories are also the corpus sort order, running from reference material to
adversary behaviour, so anything consuming the corpus in order reads context
before threats.

## Document structure

Every document carries YAML-style front matter (`id`, `title`, `category`,
`version`, `updated`, `applies_to`, `keywords`, `mitre`, `severity_hint`,
`sources`, `licence`) and exactly six sections, in this order:

```
## Summary
## What the DPI engine can observe
## Indicators
## Benign explanations
## Recommended checks
## References
```

**`## What the DPI engine can observe` is the section that makes this corpus
project-specific.** It names the actual `CaptureReport` and `FlowRecord`
fields that bear on the topic, and it states what the engine *cannot* see.
Fields that do not exist must never be referenced — the report carries no
timestamps, no durations and no payload bytes, so nothing in this corpus may
assert query rates, beacon intervals, session durations, TLS fingerprints or
packet contents.

`applies_to` is checked against the reserved signal vocabulary in
`ai/rag/documents.py`. A document cannot claim to answer a signal that does
not exist.

## Provenance and licensing

Provenance is mandatory: every document declares both `sources` and `licence`,
and the loader rejects a document that omits either.

- **Project-authored content** (`licence: project-authored`) — written for
  this repository. This is the default and the preference.
- **MITRE ATT&CK** — technique names and identifiers (T1071.004,
  T1048) are used in `attack-patterns/dns-tunneling.md`, which is marked
  `licence: CC-BY-4.0`. ATT&CK is © The MITRE Corporation and is made
  available under the Creative Commons Attribution 4.0 International licence
  (CC BY 4.0).
  MITRE has not reviewed or endorsed this project. Only technique names and
  identifiers are used; the surrounding prose is our own.
- **IETF RFCs** — RFC 1035, 6066, 8499 and 9000 are cited as references for
  definitions and protocol limits. No RFC text is reproduced; the citations
  point a reader at the primary source.

No third-party blog posts, vendor whitepapers or scraped web content are
included, and none should be added. Beyond the licensing problem, external
content is an injection surface: retrieved text is fed to a language model, so
the corpus is a trust boundary.

## Review policy

Documents are reviewed before they are indexed, and the review is the primary
security control:

1. A document enters the corpus only through a reviewed git commit.
2. `ai/rag/documents.py` validates the front matter against a strict schema
   (unknown keys are rejected), checks the six-section template, checks that a
   document is filed in the directory matching its declared category, checks
   that the filename matches the `id`, and rejects duplicate ids.
3. The corpus loads completely or not at all — there is no partial success, so
   a malformed document fails the build rather than silently disappearing from
   retrieval.
4. Content screening for prompt-injection patterns runs at index build time
   (a later step) and is not yet implemented.
5. Nothing is fetched from the network at query time. Ever.

## Adding a document

1. Choose the category directory; the filename must equal the `id`.
2. Copy the six-section template from any existing document.
3. Fill in front matter, including `sources` and `licence`.
4. Use only signal names present in `KNOWN_SIGNALS` in
   `ai/rag/documents.py`; add a new signal there first if genuinely needed.
5. In `## What the DPI engine can observe`, cite only real `CaptureReport` and
   `FlowRecord` fields, and state the relevant limits.
6. Run `python run_rag_tests.py`, then `python -m ai.rag.documents` to see the
   corpus summary.
