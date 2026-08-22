# Python Deep Packet Inspection & Network Traffic Analyzer

An offline network traffic analyzer written in pure Python. It reads packet
capture (`.pcap`) files, decodes them down to the application layer, identifies
which service each network flow belongs to, applies blocking rules, and writes
the permitted traffic to a new capture file.

Everything runs on the Python standard library — there is nothing to install.

---

## 1. Project Overview

Given a `.pcap` file, the analyzer:

- parses every packet from the Ethernet frame up through TCP/UDP,
- groups packets into bidirectional flows and tracks each flow's state,
- inspects payloads to recover the server name the client asked for — from the
  TLS Server Name Indication field, the HTTP `Host` header, or a DNS query,
- maps that name to an application (YouTube, Netflix, GitHub, and so on),
- evaluates blocking rules against the source IP, destination port,
  application, and domain,
- writes the packets that survive to an output `.pcap`.

Processing is **offline**: it reads capture files, it does not attach to a live
network interface.

## 2. Features

**Capture handling**
- Classic libpcap `.pcap` reader (microsecond and nanosecond timestamps, both
  byte orders)
- Filtered `.pcap` output preserving original timestamps and packet bytes

**Protocol decoding**
- Ethernet II framing
- IPv4 with variable header length (IHL, options)
- TCP with variable header length, sequence/ack numbers, and flags
- UDP

**Deep packet inspection**
- TLS Client Hello parsing → Server Name Indication (SNI)
- HTTP request parsing → `Host` header (anchored to real header lines)
- DNS query parsing → queried domain name
- QUIC Initial best-effort scan for an embedded Client Hello
- Application identification across 17 services plus generic HTTP/HTTPS/DNS/TLS/QUIC

**Flow tracking**
- Five-tuple flow table, one per worker thread, lock-free by design
- Bidirectional matching — both directions of a conversation share one record
- TCP state machine (NEW → ESTABLISHED → CLASSIFIED / BLOCKED → CLOSED)
- Per-flow packet and byte counters, in and out
- Idle-flow expiry and LRU eviction

**Filtering**
- Block by source IP, destination port, application, or domain
- Domain rules support glob wildcards (`*.example.com`, `ads*`)
- Rule files, loadable and saveable
- Blocked flows are dropped in both directions

**Pipeline**
- Multi-threaded: reader → load balancers → fast-path workers → output writer
- Configurable thread counts
- Console reports covering packet statistics, filtering results, per-thread
  load, and application distribution

## 3. Architecture

```
                    PCAP Reader
                         |
                   Packet Parser          Ethernet / IPv4 / TCP / UDP
                         |
              Flow / Connection Tracking  five-tuple, bidirectional
                         |
            SNI / HTTP / DNS Detection    recover the server name
                         |
                     DPI Engine           map name -> application
                         |
                  Rule Evaluation         IP / port / app / domain
                         |
                 Fast-Path Workers        forward or drop
                         |
                    Output PCAP
```

### Threading model

The pipeline runs as four kinds of thread, connected by bounded blocking
queues. The queues apply back-pressure: when a stage falls behind, the stage
feeding it blocks rather than growing memory without limit.

```
  1 reader thread
        |  hash(flow) % num_load_balancers
        v
  N load-balancer threads          (default 2)
        |  mix64(hash(flow)) % fps_per_lb
        v
  N x M fast-path worker threads   (default 2 per LB, so 4 total)
        |
        v
  1 output writer thread
```

Two properties make this work:

**Flow affinity.** A flow is routed by hashing a *canonicalised* five-tuple —
one where the two endpoints are ordered — so both directions of a conversation
always reach the same worker. That worker is then the only thread that ever
touches that flow's record, which is why the connection tracker needs no locks.

**Decorrelated levels.** The load-balancer hash and the worker hash are
different functions of the same flow. If both used the same value, they would
select correlated indices and some workers would never receive traffic.

Thread counts are set with `--lbs` and `--fps`; total workers is `lbs × fps`.

Because Python's global interpreter lock serialises bytecode execution, this
architecture gives you **pipeline structure and flow isolation**, not linear
CPU scaling. It is a faithful model of how production DPI systems partition
work, not a throughput optimisation.

## 4. Project Structure

```
dpi/                       the engine package
  pcap_reader.py           reads .pcap capture files
  packet_parser.py         Ethernet / IPv4 / TCP / UDP decoding
  sni_extractor.py         TLS SNI, HTTP Host, DNS, QUIC extraction
  types.py                 core types, flow hashing, app classification
  rule_manager.py          blocking rules and rule files
  connection_tracker.py    per-flow state tables
  thread_safe_queue.py     bounded blocking queue with shutdown
  load_balancer.py         flow-to-worker distribution
  fast_path.py             inspection workers
  dpi_engine.py            pipeline orchestration
  console.py               UTF-8 console output handling
  platform.py              byte-order helpers

main.py                    packet-by-packet decode
main_simple.py             flow list with detected hostnames
main_working.py            single-threaded filtering
main_dpi.py                multi-threaded filtering (main tool)
dpi_mt.py                  alternate self-contained engine

run_selftests.py           runs all checks
test_dpi.pcap              sample capture (77 packets)
generate_test_pcap.py      builds a new sample capture
```

## 5. Requirements

- **Python 3.10 or newer** (tested on 3.10 and 3.11)
- **No third-party packages.** Standard library only.

## 6. Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python run_selftests.py
```

If that prints `15/15 checks passed`, you are ready. There is no build step and
nothing to `pip install`.

A virtual environment is optional and only isolates the interpreter version:

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
```

## 7. Running the Analyzer

### Decode a capture packet by packet

```bash
python main.py test_dpi.pcap          # every packet
python main.py test_dpi.pcap 10       # first 10 only
```

### List flows with detected hostnames

```bash
python main_simple.py test_dpi.pcap
```

### Filter a capture — single-threaded

```bash
python main_working.py test_dpi.pcap filtered.pcap
python main_working.py test_dpi.pcap filtered.pcap --block-app YouTube
python main_working.py test_dpi.pcap filtered.pcap --block-ip 192.168.1.100
python main_working.py test_dpi.pcap filtered.pcap --block-domain tiktok
```

Here `--block-domain` is a plain substring match.

### Filter a capture — multi-threaded engine

```bash
python main_dpi.py test_dpi.pcap filtered.pcap --block-app Netflix
python main_dpi.py test_dpi.pcap filtered.pcap --block-domain "*.tiktok.com"
python main_dpi.py test_dpi.pcap filtered.pcap --rules myrules.txt
python main_dpi.py test_dpi.pcap filtered.pcap --lbs 2 --fps 3
```

| Option | Meaning |
|---|---|
| `--block-ip <ip>` | Drop traffic from this source IP, and replies to it |
| `--block-app <name>` | Drop an application by name |
| `--block-domain <pattern>` | Drop a domain; `*` and `?` wildcards supported |
| `--rules <file>` | Load rules from a file |
| `--lbs <n>` | Load-balancer threads (default 2) |
| `--fps <n>` | Worker threads per load balancer (default 2) |
| `--verbose` | Extra output |

Recognised application names:

```
Google      YouTube    Facebook   Instagram   Twitter/X   Netflix
Amazon      Microsoft  Apple      WhatsApp    Telegram    TikTok
Spotify     Zoom       Discord    GitHub      Cloudflare
HTTP        HTTPS      DNS        TLS         QUIC
```

## 8. Blocking Rules

Rules can be given on the command line or loaded from a file with `--rules`.

```ini
[BLOCKED_IPS]
192.168.1.50

[BLOCKED_APPS]
YouTube
TikTok

[BLOCKED_DOMAINS]
*.tiktok.com
ads.example.com

[BLOCKED_PORTS]
8080
```

Notes:

- Rules are evaluated in the order **IP → port → application → domain**, and
  the first match decides.
- Once a flow is blocked, later packets in that flow are dropped without
  re-inspection, in both directions.
- Exact domain names match case-insensitively; wildcard patterns use glob
  syntax.
- A malformed line is reported and skipped — it does not stop the rest of the
  file loading.
- `RuleManager.save_rules()` writes this same format back out.

## 9. Testing

```bash
python run_selftests.py
```

This runs 15 checks: a self-test built into each of the 10 engine modules, plus
a smoke run of each of the 5 command-line tools. Expected output:

```
15/15 checks passed
```

Individual module self-tests can be run directly:

```bash
python -m dpi.types
python -m dpi.packet_parser test_dpi.pcap
python -m dpi.sni_extractor
```

Use `python -m dpi.<name>`, not `python dpi/<name>.py` — two modules are named
`types` and `platform`, which shadow standard-library modules of the same name
when executed as scripts.

## 10. Example Usage

Block Netflix and inspect the result:

```bash
$ python main_dpi.py test_dpi.pcap filtered.pcap --block-app Netflix
```

```
╔══════════════════════════════════════════════════════════════╗
║                    DPI ENGINE STATISTICS                      ║
╠══════════════════════════════════════════════════════════════╣
║ PACKET STATISTICS                                             ║
║   Total Packets:                77                            ║
║   TCP Packets:                  73                            ║
║   UDP Packets:                   4                            ║
╠══════════════════════════════════════════════════════════════╣
║ FILTERING STATISTICS                                          ║
║   Forwarded:                    76                            ║
║   Dropped/Blocked:               1                            ║
╚══════════════════════════════════════════════════════════════╝
```

Using the engine from your own code:

```python
from dpi.dpi_engine import Config, DPIEngine

engine = DPIEngine(Config(num_load_balancers=2, fps_per_lb=2))
engine.initialize()
engine.block_domain("*.example.com")
engine.block_app("YouTube")
engine.process_file("capture.pcap", "filtered.pcap")
```

Reading a capture without the full pipeline:

```python
from dpi.pcap_reader import PcapReader
from dpi.packet_parser import PacketParser

with PcapReader() as reader:
    reader.open("capture.pcap")
    for raw in reader:
        pkt = PacketParser.parse(raw)
        if pkt and pkt.has_tcp:
            print(f"{pkt.src_ip}:{pkt.src_port} -> {pkt.dest_ip}:{pkt.dest_port}")
```

## 11. Current Limitations

Stated plainly, so nothing here is oversold:

- **Offline only.** Reads capture files; does not attach to a live interface.
- **`.pcap` only.** `.pcapng` — Wireshark's modern default — is not supported.
  Re-save via *File → Save As → Wireshark/tcpdump/… pcap*.
- **IPv4 only.** IPv6 packets are parsed at the Ethernet layer and then skipped.
- **No TCP reassembly.** A TLS Client Hello split across segments is not
  recovered; classification relies on the handshake fitting one packet, which is
  the common case but not guaranteed.
- **QUIC is best-effort.** Real QUIC Initial packets are encrypted; the scan
  only succeeds on unprotected or synthetic traffic.
- **Not a line-rate tool.** Python's GIL bounds throughput; this is a learning
  and analysis tool, not production network equipment.
- **No live blocking.** It filters capture files. It does not enforce policy on
  a real network.
- **Classification is hostname-based.** Traffic with no recoverable hostname
  falls back to port-based guessing, or stays unidentified.

## 12. Future Development

Not implemented — listed as direction, not as capability:

- `.pcapng` support
- IPv6 parsing
- TCP stream reassembly for segmented handshakes
- Live capture from a network interface
- JSON / CSV export of flow records
- A `pytest` suite alongside the built-in self-tests
- Statistical traffic analysis
