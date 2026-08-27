---
id: dpi-network-security-terms
title: DPI and Network Security Terminology
category: glossary
version: 1.0
updated: 2026-08-27
applies_to:
  - baseline_web_browsing
  - tls_without_sni
  - quic_present
  - plaintext_http
  - scan_half_open
  - blocked_traffic_present
keywords:
  - glossary
  - terminology
  - sni
  - five-tuple
  - flow
  - deep packet inspection
mitre: []
severity_hint: info
sources:
  - Authored for this project.
  - RFC 6066, TLS Extensions, for the definition of Server Name Indication.
  - RFC 9000, QUIC - A UDP-Based Multiplexed and Secure Transport.
licence: project-authored
---

## Summary

Shared vocabulary for the terms used across this corpus and in the analysis
output. Definitions are written as this project uses them, which is
occasionally narrower than general usage — where that is the case, the entry
says so.

## What the DPI engine can observe

Each term below maps to something concrete in the `CaptureReport`:

- **Deep packet inspection (DPI)** — examining packet headers, and where
  visible the start of the payload, to classify traffic. This engine inspects
  headers and the TLS Client Hello only; it never exports payload bytes, and
  no payload reaches any analysis stage.
- **Flow** — one bidirectional conversation, keyed by the five-tuple. Becomes
  one `FlowRecord` with a stable `flow_id`.
- **Five-tuple** — source address, destination address, source port,
  destination port and protocol. Exposed as `src_ip`, `dst_ip`, `src_port`,
  `dst_port` and `protocol`, with addresses redacted or pseudonymised
  according to `redaction_mode`.
- **SNI (Server Name Indication)** — the TLS extension that carries the
  requested hostname in the clear inside the Client Hello. It is the engine's
  main classification input and appears as `server_name`. It is supplied by
  the client, so it is attacker-controlled and treated as untrusted.
- **Half-open connection** — a connection attempt that was never completed:
  `syn_seen` true, `syn_ack_seen` false. Characteristic of scanning and of
  unreachable services alike.
- **QUIC** — a UDP-based transport carrying HTTP/3, normally on UDP 443.
  Appears as `protocol` `UDP` with `dst_port` 443.
- **Verdict** — the engine's own decision for a flow, `FORWARD` or `DROP`,
  exposed as `verdict`, with `state` reaching `BLOCKED` when a rule matched.
- **Application** — the engine's classification of a flow, exposed as
  `application`, derived from SNI, HTTP Host or DNS name. `UNKNOWN` means the
  classifier had nothing to work with, not that the flow is anomalous.
- **Redaction mode** — the policy that governs whether addresses are omitted,
  pseudonymised or preserved in the report, recorded in `redaction_mode`.

## Indicators

Terminology distinctions that change how a finding should be read:

- *Observed* versus *inferred*: a field present in the report is observed;
  anything derived from combining fields is inferred and must be labelled.
- *Unclassified* versus *suspicious*: `UNKNOWN` is a classifier outcome, not a
  risk assessment.
- *Blocked* versus *malicious*: `verdict` of `DROP` means a configured rule
  matched, and says nothing about intent.
- *Flow count* versus *host count*: many flows can share one destination.

## Benign explanations

Vocabulary mistakes that read as findings but are not:

- Calling an `UNKNOWN` application "unidentified malware" — the engine simply
  had no SNI to classify.
- Calling a half-open connection an "attack" — a closed port, a firewall drop
  or a stale destination produces the same evidence.
- Calling a `DROP` verdict a "detection" — it reflects the operator's own
  configured rules, listed in `blocking_rules_active`.
- Reporting a per-flow byte count as bandwidth — the report contains no
  timestamps, so no rate can be computed.

## Recommended checks

- Use `flow_id` when referring to specific traffic. Addresses may be redacted
  or pseudonymised, so they are not stable identifiers for a reader.
- State whether a term is being used as an observation or an inference.
- Avoid rate, duration and periodicity vocabulary entirely; the underlying
  data has no time dimension.
- Prefer the engine's own field names when describing evidence, so a reader
  can verify a claim against the report.

## References

- RFC 6066 defines Server Name Indication.
- RFC 9000 defines QUIC.
- See `triaging-unknown-application-traffic` for what to do when the
  `UNKNOWN` share is large.
