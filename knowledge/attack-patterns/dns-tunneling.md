---
id: dns-tunneling
title: DNS Tunneling and Covert DNS Channels
category: attack-patterns
version: 1.0
updated: 2026-08-27
applies_to:
  - dns_high_volume
  - dns_high_cardinality
  - dns_anomalous_label
  - upload_asymmetry
keywords:
  - dns tunneling
  - covert channel
  - exfiltration
  - command and control
  - subdomain encoding
mitre:
  - T1071.004
  - T1048
severity_hint: high
sources:
  - Authored for this project.
  - MITRE ATT&CK technique T1071.004 (Application Layer Protocol - DNS) for the technique name and identifier.
  - MITRE ATT&CK technique T1048 (Exfiltration Over Alternative Protocol) for the technique name and identifier.
licence: CC-BY-4.0
---

## Summary

DNS tunneling encodes arbitrary data inside DNS queries and answers, using
name resolution as a transport rather than as a lookup. Data leaving the
network is encoded into the subdomain labels of queries for a domain the
attacker controls; data coming back rides in the answer records. Because
almost every network permits outbound DNS — often even where nothing else is
allowed — the channel frequently survives egress filtering that blocks direct
connections.

The traffic still has to obey DNS's shape, and that is what makes it visible.
Encoding bytes into names produces long, random-looking labels, an enormous
number of distinct names under one parent domain, and query volume far above
what a human's browsing would generate. The channel is loud precisely because
DNS was never designed to carry payload.

## What the DPI engine can observe

From the `CaptureReport` alone:

- Many `FlowRecord` entries with `protocol` `UDP`, `dst_port` `53` and
  `application` `DNS` — visible as a large DNS entry in
  `application_distribution` relative to `total_flows`.
- The `server_name` of each query, sanitized. Tunneling shows up here as long,
  high-entropy leftmost labels sharing one registrable parent domain.
- Near-unique `server_name` values across DNS flows: encoded data cannot
  repeat, so caching never applies and cardinality approaches the flow count.
- `bytes_out` elevated relative to `bytes_in` on DNS flows, since the outbound
  direction is carrying the payload. This is the reverse of normal resolution.
- `src_ip` pseudonyms, which show whether one host or many are involved.

Not observable, and therefore never to be asserted: query type (TXT and NULL
records are the classic carriers), answer contents, query rate, beacon
interval, or any timing regularity. `FlowRecord` has no timestamps and no
payload bytes.

## Indicators

- A large count of DNS flows whose `server_name` values are almost all
  distinct and share one parent domain.
- Leftmost labels that are long — close to the 63-octet limit — and look
  base32, base64 or hex encoded rather than pronounceable.
- Total DNS `bytes_out` comparable to or greater than DNS `bytes_in`.
- The tunneled parent domain never appears as a `server_name` on any TLS flow,
  because the host is not actually browsing to it.
- Optionally, a recently registered or otherwise unfamiliar parent domain.

Any one of these alone is weak. The combination of *high volume*, *high
cardinality under one parent* and *anomalous label shape* is what
distinguishes tunneling from busy but ordinary resolution.

## Benign explanations

Several legitimate systems produce query patterns that resemble tunneling and
must be ruled out before escalating:

- Antivirus, endpoint-security and reputation services encode file or URL
  hashes into DNS lookups by design; these look exactly like tunneling.
- Anti-spam DNSBL lookups encode addresses into query names.
- Some CDNs and load balancers issue per-session or per-object hostnames that
  are unique and machine-generated.
- Certificate transparency and telemetry clients can generate high-cardinality
  lookups against a single vendor domain.
- Traffic captured from a DNS resolver or forwarder aggregates a whole
  network's queries, so its volume and cardinality are legitimately high.

The parent domain usually settles it: a known security vendor's domain is a
very different finding from an unfamiliar one.

## Recommended checks

- Group DNS `server_name` values by registrable parent domain and compare the
  distinct-name count per parent against the DNS flow count.
- Measure the maximum and mean leftmost-label length, and the character mix,
  per parent domain.
- Compare DNS `bytes_out` against `bytes_in` in aggregate.
- Check whether the parent domain also appears as a `server_name` on TCP 443
  flows, and whether it belongs to a known security or CDN vendor.
- Identify the originating host by pseudonymised `src_ip` and check whether it
  alone accounts for the volume.
- State plainly that periodicity and query rate cannot be assessed from this
  data. A tunneling conclusion drawn from volume and shape is an inference,
  not an observation.

## References

- MITRE ATT&CK T1071.004, Application Layer Protocol: DNS.
- MITRE ATT&CK T1048, Exfiltration Over Alternative Protocol.
- See `dns-normal-behaviour` for the baseline these indicators deviate from,
  and `suspicious-dns-indicators` for the specific thresholds and their known
  false positives.
