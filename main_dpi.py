#!/usr/bin/env python3
"""DPI Engine CLI — the multi-threaded entry point.

Python port of ``src/main_dpi.cpp``.

C++ concepts replaced
---------------------
``argc``/``argv`` with manual index walking
    Kept verbatim rather than rewritten with :mod:`argparse`, because the
    original's argument handling has observable quirks that argparse would
    silently "fix": unknown options are ignored without error, an option whose
    value is missing at the end of the line is dropped silently, and options
    are only scanned from index 3 onward (so ``--help`` as the *first*
    argument is treated as the input filename).

``std::stoi`` on ``--lbs`` / ``--fps``
    Raises on a non-numeric value in both languages; left uncaught, as in the
    original.
"""

from __future__ import annotations

import sys

from dpi.dpi_engine import Config, DPIEngine


def print_usage(program: str) -> None:
    """Print the usage banner.  Mirrors ``printUsage``."""
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    DPI ENGINE v1.0                            ║
║               Deep Packet Inspection System                   ║
╚══════════════════════════════════════════════════════════════╝

Usage: {program} <input.pcap> <output.pcap> [options]

Arguments:
  input.pcap     Input PCAP file (captured user traffic)
  output.pcap    Output PCAP file (filtered traffic to internet)

Options:
  --block-ip <ip>        Block packets from source IP
  --block-app <app>      Block application (e.g., YouTube, Facebook)
  --block-domain <dom>   Block domain (supports wildcards: *.facebook.com)
  --rules <file>         Load blocking rules from file
  --lbs <n>              Number of load balancer threads (default: 2)
  --fps <n>              FP threads per LB (default: 2)
  --verbose              Enable verbose output

Examples:
  {program} capture.pcap filtered.pcap
  {program} capture.pcap filtered.pcap --block-app YouTube
  {program} capture.pcap filtered.pcap --block-ip 192.168.1.50 --block-domain *.tiktok.com
  {program} capture.pcap filtered.pcap --rules blocking_rules.txt

Supported Apps for Blocking:
  Google, YouTube, Facebook, Instagram, Twitter/X, Netflix, Amazon,
  Microsoft, Apple, WhatsApp, Telegram, TikTok, Spotify, Zoom, Discord, GitHub

Architecture:
  ┌─────────────┐
  │ PCAP Reader │  Reads packets from input file
  └──────┬──────┘
         │ hash(5-tuple) % num_lbs
         ▼
  ┌──────┴──────┐
  │ Load Balancer │  2 LB threads distribute to FPs
  │   LB0 │ LB1   │
  └──┬────┴────┬──┘
     │         │  hash(5-tuple) % fps_per_lb
     ▼         ▼
  ┌──┴──┐   ┌──┴──┐
  │FP0-1│   │FP2-3│  4 FP threads: DPI, classification, blocking
  └──┬──┘   └──┬──┘
     │         │
     ▼         ▼
  ┌──┴─────────┴──┐
  │ Output Writer │  Writes forwarded packets to output
  └───────────────┘

""", end="")


def main(argv: list[str] | None = None) -> int:
    """Mirrors ``int main(int argc, char* argv[])``."""
    argv = list(sys.argv if argv is None else argv)
    argc = len(argv)

    if argc < 3:
        print_usage(argv[0])
        return 1

    input_file = argv[1]
    output_file = argv[2]

    # Parse options
    config = Config()
    config.num_load_balancers = 2
    config.fps_per_lb = 2

    block_ips: list[str] = []
    block_apps: list[str] = []
    block_domains: list[str] = []
    rules_file = ""

    i = 3
    while i < argc:
        arg = argv[i]

        if arg == "--block-ip" and i + 1 < argc:
            i += 1
            block_ips.append(argv[i])
        elif arg == "--block-app" and i + 1 < argc:
            i += 1
            block_apps.append(argv[i])
        elif arg == "--block-domain" and i + 1 < argc:
            i += 1
            block_domains.append(argv[i])
        elif arg == "--rules" and i + 1 < argc:
            i += 1
            rules_file = argv[i]
        elif arg == "--lbs" and i + 1 < argc:
            i += 1
            config.num_load_balancers = int(argv[i])
        elif arg == "--fps" and i + 1 < argc:
            i += 1
            config.fps_per_lb = int(argv[i])
        elif arg == "--verbose":
            config.verbose = True
        elif arg in ("--help", "-h"):
            print_usage(argv[0])
            return 0
        # NOTE: unrecognised arguments are silently ignored, as in the C++.
        i += 1

    # Create DPI engine
    engine = DPIEngine(config)

    # Initialize
    if not engine.initialize():
        print("Failed to initialize DPI engine", file=sys.stderr)
        return 1

    # Load rules from file if specified
    if rules_file:
        engine.load_rules(rules_file)

    # Apply command-line blocking rules
    for ip in block_ips:
        engine.block_ip(ip)

    for app in block_apps:
        engine.block_app(app)

    for domain in block_domains:
        engine.block_domain(domain)

    # Process the file
    if not engine.process_file(input_file, output_file):
        print("Failed to process file", file=sys.stderr)
        return 1

    print("\nProcessing complete!")
    print(f"Output written to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
