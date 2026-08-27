---
id: triaging-unknown-application-traffic
title: Triaging Unknown Application Traffic
category: triage-playbooks
version: 1.0
updated: 2026-08-27
applies_to:
  - unknown_app_share
  - tls_without_sni
  - nonstandard_port_egress
  - quic_present
  - blocked_traffic_present
keywords:
  - triage
  - unknown
  - unclassified
  - playbook
  - next steps
mitre: []
severity_hint: low
sources:
  - Authored for this project.
licence: project-authored
---

## Summary

A large share of unclassified traffic is a statement about the classifier, not
about the traffic. This engine identifies applications from TLS SNI, HTTP Host
headers and DNS query names; when none of those is present, the flow is
`UNKNOWN` no matter how ordinary it is. Encrypted Client Hello, QUIC, raw TCP
services, and any protocol the engine does not parse all land in the same
bucket.

The correct output for a capture dominated by unknown traffic is a
well-structured account of what is missing and how to find out — not a
confident verdict in either direction. Treating "unclassified" as "suspicious"
is the most common way a network analysis loses credibility.

## What the DPI engine can observe

- `application_distribution` gives the count of flows classified as `UNKNOWN`;
  compare it against `totals.total_flows` for the share.
- For each unknown `FlowRecord`: `protocol`, `dst_port`, `src_port`,
  `packets_out` / `packets_in`, `bytes_out` / `bytes_in`, `syn_seen`,
  `syn_ack_seen`, `fin_seen`, `state` and `verdict`.
- A null `server_name` is the usual reason for the `UNKNOWN` classification
  and is itself the strongest clue: nothing in the flow named a host.
- `totals.tcp_packets` and `totals.udp_packets` show whether unknown traffic
  is TCP or UDP shaped at capture level.
- `verdict` of `DROP` and `state` of `BLOCKED` show flows the engine's rules
  acted on; `blocking_rules_active` says which rule categories were loaded.

Not observable: payload bytes, protocol fingerprints such as JA3/JA4, TLS
version, certificate details, or any timing. Identifying an unknown protocol
from its handshake is outside what this report contains.

## Indicators

Signals worth separating before drawing any conclusion:

- Unknown flows on TCP 443 with no `server_name` — most likely ordinary TLS
  the engine could not name, including Encrypted Client Hello.
- Unknown flows on UDP 443 — almost certainly QUIC, and unremarkable.
- Unknown flows on well-known ports other than 80/443 — mail, NTP, SSH and
  similar services the classifier does not cover.
- Unknown flows on high, non-standard destination ports with real byte counts
  — the subset that actually warrants attention.
- `syn_seen` true with `syn_ack_seen` false and near-zero bytes — an
  unanswered connection attempt, not an application at all.

## Benign explanations

- The classifier is SNI-driven, so any encrypted flow without a readable SNI
  is unclassifiable by construction.
- QUIC on UDP 443 carries a large and growing share of normal web traffic.
- Captures that begin mid-session miss the handshake that carried the SNI.
- Peer-to-peer software, game clients, VPNs and video conferencing use high
  dynamic ports as a matter of course.
- A capture from a server rather than a workstation is dominated by inbound
  service traffic the classifier was never designed to name.

## Recommended checks

Work through these in order:

1. Split unknown flows by `protocol` and `dst_port`. Set aside UDP 443 as
   QUIC and TCP 443 as unnamed TLS; these usually account for most of the
   share and need no further explanation.
2. For what remains, sort by `bytes_out + bytes_in`. Volume is the cheapest
   available proxy for significance.
3. Flag flows with `bytes_out` far exceeding `bytes_in` on non-standard ports
   for closer review.
4. Separate unanswered connection attempts (`syn_seen` without
   `syn_ack_seen`) from established flows; they are a scanning question, not
   an application question.
5. Check whether any unknown flow's destination pseudonym also appears on
   classified flows.
6. Report the residue honestly: state the share, state that SNI absence is the
   cause, and list what evidence would be needed — full packet capture, TLS
   fingerprinting, endpoint process attribution — to identify it.

Confidence in any assessment of a capture dominated by unknown traffic should
be low, and the uncertainties should say why.

## References

- See `dpi-network-security-terms` for SNI and DPI definitions, and
  `cdn-and-multi-host-traffic` for why many unnamed flows to many destinations
  is normal for web browsing.
