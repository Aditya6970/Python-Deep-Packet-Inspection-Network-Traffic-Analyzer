#!/usr/bin/env python3
"""Optional AI-assisted capture analysis.

Runs the existing DPI engine unchanged, then sends a sanitized summary of what
it found to an LLM for a written assessment.

    python analyze_ai.py test_dpi.pcap
    python analyze_ai.py capture.pcap --block-app YouTube --json out.json

The AI layer is **optional**. Without an API key the DPI analysis still runs
and prints in full; only the AI section is skipped. This program exits 0 in
that case, because nothing went wrong.

Raw packet payloads are never sent. See ``ai/redaction.py`` for exactly what
leaves the machine.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dpi.dpi_engine import Config, DPIEngine


def print_usage(program: str) -> None:
    print(f"""
AI-Assisted Packet Capture Analysis
===================================

Usage: {program} <input.pcap> [options]

Arguments:
  input.pcap             Capture to analyse

Options:
  --output <file.pcap>   Write filtered traffic here (default: discard)
  --json <file.json>     Save the structured AI result as JSON
  --block-ip <ip>        Block traffic from this source IP
  --block-app <name>     Block an application (YouTube, Netflix, ...)
  --block-domain <pat>   Block a domain; * and ? wildcards supported
  --rules <file>         Load blocking rules from a file
  --lbs <n>              Load-balancer threads (default 2)
  --fps <n>              Worker threads per load balancer (default 2)
  --provider <name>      groq | ollama | openai   (default from DPI_LLM_PROVIDER)
  --ip-mode <mode>       full | redact_private | none  (default redact_private)
  --model <name>         Override the model for the selected provider
  --no-ai                Run DPI only; skip the AI layer entirely
  --show-payload         Print the exact JSON that would be sent, then exit

Environment:
  DPI_LLM_PROVIDER       groq (default) | ollama | openai
  GROQ_API_KEY           Required when provider is groq.  Free tier available.
  OLLAMA_BASE_URL        Default http://localhost:11434/v1 .  No key needed.
  OPENAI_API_KEY         Required when provider is openai.
  See .env.example for the full list.

Examples:
  {program} test_dpi.pcap
  {program} test_dpi.pcap --provider groq --block-app YouTube
  {program} test_dpi.pcap --provider ollama --model llama3.1
  {program} test_dpi.pcap --show-payload
""", end="")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    argc = len(argv)

    if argc < 2 or argv[1] in ("-h", "--help"):
        print_usage(argv[0])
        return 0 if argc >= 2 else 1

    input_file = argv[1]
    if not Path(input_file).is_file():
        print(f"Error: no such file: {input_file}", file=sys.stderr)
        return 1

    output_file = ""
    json_file = ""
    ip_mode = ""
    model = ""
    provider = ""
    use_ai = True
    show_payload = False
    block_ips: list[str] = []
    block_apps: list[str] = []
    block_domains: list[str] = []
    rules_file = ""
    config = Config()

    i = 2
    while i < argc:
        arg = argv[i]
        if arg == "--output" and i + 1 < argc:
            i += 1; output_file = argv[i]
        elif arg == "--json" and i + 1 < argc:
            i += 1; json_file = argv[i]
        elif arg == "--block-ip" and i + 1 < argc:
            i += 1; block_ips.append(argv[i])
        elif arg == "--block-app" and i + 1 < argc:
            i += 1; block_apps.append(argv[i])
        elif arg == "--block-domain" and i + 1 < argc:
            i += 1; block_domains.append(argv[i])
        elif arg == "--rules" and i + 1 < argc:
            i += 1; rules_file = argv[i]
        elif arg == "--lbs" and i + 1 < argc:
            i += 1; config.num_load_balancers = int(argv[i])
        elif arg == "--fps" and i + 1 < argc:
            i += 1; config.fps_per_lb = int(argv[i])
        elif arg == "--ip-mode" and i + 1 < argc:
            i += 1; ip_mode = argv[i]
        elif arg == "--model" and i + 1 < argc:
            i += 1; model = argv[i]
        elif arg == "--provider" and i + 1 < argc:
            i += 1; provider = argv[i]
        elif arg == "--no-ai":
            use_ai = False
        elif arg == "--show-payload":
            show_payload = True
        i += 1

    # ---- Stage 1: the existing DPI engine, entirely unchanged -------------
    import tempfile

    temp_dir: tempfile.TemporaryDirectory | None = None
    if not output_file:
        temp_dir = tempfile.TemporaryDirectory()
        output_file = str(Path(temp_dir.name) / "filtered.pcap")

    try:
        engine = DPIEngine(config)
        engine.initialize()

        if rules_file:
            engine.load_rules(rules_file)
        for ip in block_ips:
            engine.block_ip(ip)
        for app in block_apps:
            engine.block_app(app)
        for domain in block_domains:
            engine.block_domain(domain)

        if not engine.process_file(input_file, output_file):
            print("Failed to process file", file=sys.stderr)
            return 1

        snapshot = engine.get_flow_snapshot()
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    if not use_ai:
        print("\n[AI analysis skipped: --no-ai]")
        return 0

    # ---- Stage 2: the optional AI layer ----------------------------------
    # Imported here so a missing pydantic/openai cannot break DPI-only runs.
    try:
        from ai.analyzer import analyze_capture
        from ai.config import AIConfig, IPRedactionMode
        from ai.report import render
    except ImportError as exc:
        print(f"\n[AI analysis unavailable: {exc}]")
        print("Install the optional dependencies with: pip install -r requirements.txt")
        return 0

    ai_config = AIConfig.from_env(provider=provider or None)

    if ai_config.invalid_provider is not None:
        print(f"\nUnknown provider: {ai_config.invalid_provider!r}", file=sys.stderr)
        print("Valid providers: groq, ollama, openai", file=sys.stderr)
        print("The DPI analysis above is complete and unaffected.", file=sys.stderr)
        return 0

    if ip_mode:
        try:
            ai_config.ip_mode = IPRedactionMode(ip_mode.strip().lower())
        except ValueError:
            print(f"Unknown --ip-mode {ip_mode!r}; using {ai_config.ip_mode.value}",
                  file=sys.stderr)
    if model:
        ai_config.model = model

    if show_payload:
        from ai.extractor import build_capture_report
        from ai.prompts import build_messages

        report = build_capture_report(snapshot, input_file, ai_config)
        messages = build_messages(report)
        print("\n===== EXACTLY WHAT WOULD BE SENT =====\n")
        for m in messages:
            print(f"--- role: {m['role']} ---")
            print(m["content"])
            print()
        return 0

    outcome = analyze_capture(snapshot, input_file, ai_config)

    print()
    print(render(outcome))

    if json_file and outcome.ok and outcome.analysis is not None:
        payload = {
            "capture": outcome.report.capture_name,
            "model": outcome.model,
            "prompt_version": outcome.prompt_version,
            "warnings": outcome.warnings,
            "analysis": outcome.analysis.model_dump(mode="json"),
        }
        Path(json_file).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Structured result saved to: {json_file}")

    # A skipped or failed AI stage is not a program failure: DPI succeeded.
    return 0


if __name__ == "__main__":
    sys.exit(main())
