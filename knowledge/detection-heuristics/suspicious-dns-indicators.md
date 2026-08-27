---
id: suspicious-dns-indicators
title: Suspicious DNS Indicators and Their Thresholds
category: detection-heuristics
version: 1.0
updated: 2026-08-27
applies_to:
  - dns_high_volume
  - dns_high_cardinality
  - dns_anomalous_label
  - upload_asymmetry
keywords:
  - heuristic
  - threshold
  - entropy
  - label length
  - cardinality
  - false positive
mitre: []
severity_hint: medium
sources:
  - Authored for this project.
  - RFC 1035 for the 63-octet label and 255-octet name limits that bound the length heuristic.
licence: project-authored
---

## Summary

This document states the DNS heuristics this project uses, the reasoning
behind each threshold, and — just as importantly — what each one gets wrong.
A threshold without a documented false-positive mode is a liability: it will
fire, someone will believe it, and nobody will know how often it is wrong.

The thresholds below are starting points chosen to be defensible, not tuned
against a labelled dataset. They should be treated as prompts for
investigation, never as verdicts.

## What the DPI engine can observe

Each heuristic is computed from `CaptureReport` fields only:

- **Volume** — count of `FlowRecord`s with `protocol` `UDP` and `dst_port` 53,
  as a share of `totals.total_flows`.
- **Cardinality** — count of distinct non-null `server_name` values among
  those flows, and the ratio of distinct names to DNS flows.
- **Label shape** — from each sanitized `server_name`: the length of the
  leftmost label, its Shannon entropy over the character distribution, and
  whether its character set looks base32, base64 or hexadecimal.
- **Directional asymmetry** — aggregate `bytes_out` versus `bytes_in` across
  DNS flows.
- **Host attribution** — grouping by pseudonymised `src_ip`.

Every one of these is computable without timestamps or payloads. Any heuristic
requiring query rate, inter-arrival time, jitter or record type is out of
scope for this data and must not be claimed.

## Indicators

- **High DNS volume** — DNS flows exceed roughly 20 per cent of
  `totals.total_flows`, or exceed about 50 flows in a short capture. Weak on
  its own; meaningful as a multiplier on the heuristics below.
- **High name cardinality** — distinct `server_name` values divided by DNS
  flow count above about 0.9, with more than roughly 20 DNS flows. Normal
  browsing reuses names; encoded data cannot.
- **Anomalous label** — leftmost label longer than about 30 characters, or
  Shannon entropy above roughly 3.5 bits per character, or a character set
  consistent with base32/base64/hex encoding. The 63-octet limit means labels
  crowding that ceiling are close to the encoder's maximum.
- **Single-parent concentration** — more than about 20 distinct names under
  one registrable parent domain within a single capture.
- **Reversed asymmetry** — aggregate DNS `bytes_out` at or above `bytes_in`.
  Ordinary resolution sends less than it receives.

Severity should rise with the *number of independent indicators that agree*,
not with the magnitude of any single one.

## Benign explanations

Each heuristic has a well-understood way of being wrong:

- **Volume** fires on any capture taken from a resolver, forwarder or gateway,
  and on the first seconds after boot.
- **Cardinality** fires on security vendors that encode hashes into names, on
  DNSBL lookups, and on CDNs issuing per-object hostnames.
- **Label length and entropy** fire on the machine-generated hostnames used by
  cloud providers, CDN edge nodes, and some telemetry endpoints. Base32-looking
  is not the same as base32-encoded.
- **Single-parent concentration** fires on exactly the same vendor patterns,
  and on any service that shards by subdomain.
- **Reversed asymmetry** can appear when answers are `NXDOMAIN` or otherwise
  minimal, which is unremarkable in isolation.

A capture where only one heuristic fires is usually benign. Say so.

## Recommended checks

- Report which heuristics fired and which did not. A single firing heuristic
  should lower confidence, not raise it.
- Identify the parent domain and check whether it belongs to a recognised
  security, CDN or cloud vendor before escalating.
- Confirm whether the pattern comes from one host or many, using the
  pseudonymised `src_ip`.
- Check whether the same parent domain appears as a `server_name` on TCP 443
  flows.
- Record explicitly that these thresholds are untuned defaults, and that no
  timing-based confirmation is possible from this data.

## References

- RFC 1035 for the label and name length limits.
- See `dns-normal-behaviour` for the baseline, `dns-tunneling` for the
  attack pattern, and `cdn-and-multi-host-traffic` for the most common source
  of false positives on the cardinality and label heuristics.
