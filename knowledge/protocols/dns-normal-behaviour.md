---
id: dns-normal-behaviour
title: Normal DNS Behaviour
category: protocols
version: 1.0
updated: 2026-08-27
applies_to:
  - dns_high_volume
  - dns_high_cardinality
  - dns_anomalous_label
  - baseline_web_browsing
keywords:
  - dns
  - resolver
  - udp 53
  - name resolution
  - query
mitre: []
severity_hint: info
sources:
  - Authored for this project.
  - RFC 1035, Domain Names - Implementation and Specification (label and name length limits).
  - RFC 8499, DNS Terminology (stub resolver, recursive resolver).
licence: project-authored
---

## Summary

DNS turns names into addresses. A client sends a short query to a configured
recursive resolver, usually over UDP port 53, and gets a short answer back.
Almost every other flow in a normal capture is preceded by one. DNS is
therefore both the noisiest protocol in a capture and the most useful: because
the query names are visible even when everything else is encrypted, DNS is
often the only place a host's intent is written in plain text.

Normal DNS is characterised by *asymmetry in the right direction* and *low
cardinality*. Queries are small, answers are slightly larger, and a browsing
host resolves tens of distinct names, not thousands. The same handful of names
recur, because caching means a repeat visit usually asks nothing at all.

## What the DPI engine can observe

For each DNS flow the engine produces a `FlowRecord` with:

- `protocol` of `UDP` and `dst_port` of `53` — how DNS flows are identified.
- `application` of `DNS`, assigned by the engine's classifier.
- `server_name` — the queried name, sanitized. This is attacker-influenced
  input and is treated as untrusted throughout the pipeline.
- `packets_out` / `packets_in` and `bytes_out` / `bytes_in` — direction-aware
  counters. Normal DNS has small values on both sides with `bytes_in` at least
  comparable to `bytes_out`.
- `src_ip` / `dst_ip` — omitted or pseudonymised according to the report's
  `redaction_mode`. A stable pseudonym is still enough to group flows by host.

At capture level, `application_distribution` gives the DNS flow count and
`top_server_names` lists the most frequently seen names.

What the engine **cannot** observe: query type (A, AAAA, TXT, NULL), response
code, TTL, answer contents, or any timing. `FlowRecord` carries no timestamps
and no payload bytes by design, so query rate, inter-arrival regularity and
record type are unavailable to any analysis built on this report.

## Indicators

Indicators that DNS in a capture is behaving normally:

- DNS flows are a modest share of `total_flows`, and the count of distinct
  `server_name` values on port 53 is in the tens.
- `bytes_out` per DNS flow is small — a query is typically well under 100
  bytes — and `bytes_in` is of the same order or larger.
- Queried names are short, pronounceable, and repeat across flows.
- Names resolved by DNS correspond to `server_name` values seen on subsequent
  TCP 443 flows: the host looked something up and then connected to it.

## Benign explanations

High DNS counts alone are not suspicious. Common innocent causes:

- A page load on a modern site resolves dozens of names across CDNs,
  analytics, fonts and advertising domains.
- A capture that begins at boot or wake catches a burst of resolution from
  OS and application update checks.
- A host configured with a short cache TTL, or software that deliberately
  bypasses the OS cache, repeats queries that a cache would have absorbed.
- Split-horizon or conditional-forwarding setups send the same name to more
  than one resolver.

## Recommended checks

- Compare the count of distinct DNS `server_name` values against the count of
  DNS flows; near-equality means little cache reuse and is worth a look.
- Check whether names resolved on port 53 reappear as `server_name` on TLS
  flows. Names that are resolved but never connected to are more interesting
  than names that are both.
- Group DNS flows by the pseudonymised `src_ip` to see whether volume comes
  from one host or is spread across many.
- Treat any conclusion about query *rate* as unsupported: the data has no
  timestamps, and that limitation belongs in the analysis uncertainties.

## References

- RFC 1035 defines the 63-octet label limit and the 255-octet total name
  limit that later heuristics rely on.
- RFC 8499 defines the resolver terminology used above.
- See `suspicious-dns-indicators` for the deviations from this baseline, and
  `dns-tunneling` for the attack pattern those deviations point to.
