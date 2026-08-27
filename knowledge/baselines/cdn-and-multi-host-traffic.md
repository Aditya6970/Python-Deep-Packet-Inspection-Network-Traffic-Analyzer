---
id: cdn-and-multi-host-traffic
title: Benign CDN and Multi-Host Web Traffic
category: baselines
version: 1.0
updated: 2026-08-27
applies_to:
  - baseline_web_browsing
  - dns_high_volume
  - dns_high_cardinality
  - quic_present
  - tls_without_sni
  - unknown_app_share
keywords:
  - cdn
  - baseline
  - normal traffic
  - web browsing
  - false positive
mitre: []
severity_hint: info
sources:
  - Authored for this project.
licence: project-authored
---

## Summary

One modern web page is not one connection. Opening a single site typically
resolves and contacts dozens of distinct hostnames across content delivery
networks, image and font hosts, analytics endpoints, advertising exchanges and
API backends, most of them operated by third parties. The resulting capture
looks superficially like reconnaissance: many destinations, many short flows,
many distinct names, a burst of DNS.

This document exists so that shape is recognised as the baseline it is. A
capture that matches this profile should be reported as informational, and the
analysis should say so plainly rather than hedging toward suspicion.

## What the DPI engine can observe

- `top_server_names` shows the most frequent hostnames; benign browsing
  produces a recognisable mix of site, CDN and analytics domains.
- `application_distribution` shows named applications the SNI classifier
  recognised, alongside `HTTPS`, `TLS`, `QUIC` and `UNKNOWN` buckets.
- Per flow: `dst_port` of 443 dominates, `protocol` is a mix of TCP and UDP
  (UDP 443 being QUIC), `bytes_in` typically exceeds `bytes_out`, and
  `syn_seen`, `syn_ack_seen` and `fin_seen` show completed connections.
- `totals.total_flows` is large relative to the number of distinct
  destinations, because browsers open several connections per host.
- Many flows carry a non-null `server_name`, which is the clearest structural
  difference from scanning or tunneling.

## Indicators

A capture is consistent with benign multi-host browsing when:

- The great majority of flows are to `dst_port` 443, TCP or UDP.
- `bytes_in` exceeds `bytes_out` on most flows — the host is downloading.
- Flows complete: `syn_seen` and `syn_ack_seen` are both true, and many flows
  show `fin_seen`.
- Hostnames are pronounceable, repeat across flows, and include recognisable
  CDN and platform domains.
- DNS names resolved on port 53 reappear as `server_name` on subsequent TLS
  flows.
- The `UNKNOWN` share is moderate and concentrated on port 443, which is
  unnamed TLS and QUIC rather than anything unusual.

## Benign explanations

This document *is* the benign explanation, and it covers several patterns that
otherwise trigger detection heuristics:

- High DNS volume and moderate name cardinality, caused by third-party assets
  rather than by encoding.
- Long, machine-generated hostnames belonging to CDN edge nodes, which can
  resemble high-entropy labels without being encoded data.
- Many destination addresses contacted briefly, which resembles fan-out but is
  ordinary asset loading.
- A substantial `UNKNOWN` share caused by QUIC and by TLS without a readable
  SNI.
- Short flows with few packets, which are simply small assets fetched over
  already-warm or short-lived connections.

## Recommended checks

- Before treating destination count as suspicious, check whether flows have
  non-null `server_name` values and completed handshakes. Browsing does;
  scanning does not.
- Compare `bytes_in` against `bytes_out` in aggregate. Browsing is
  download-dominant; exfiltration is not.
- Check whether high-entropy hostnames sit under a recognisable CDN parent
  domain before treating them as encoded.
- Confirm that DNS names resolved in the capture are subsequently connected
  to; resolution without connection is the more interesting pattern.
- When the capture matches this profile, say that it does, assign an
  informational risk level, and avoid alarming language. Reporting normal
  traffic as suspicious has a real cost.

## References

- See `dns-normal-behaviour` for the DNS side of this baseline,
  `suspicious-dns-indicators` for the heuristics this profile most often
  triggers falsely, and `triaging-unknown-application-traffic` for the
  unclassified residue.
