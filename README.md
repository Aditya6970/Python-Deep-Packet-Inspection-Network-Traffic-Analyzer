# Python Deep Packet Inspection & Network Traffic Analyzer

An offline network traffic analyzer written in pure Python. It reads packet
capture (`.pcap`) files, decodes them down to the application layer, identifies
which service each network flow belongs to, applies blocking rules, and writes
the permitted traffic to a new capture file.

On top of that engine sits an **optional** AI layer: it turns what the engine
measured into a sanitized summary, retrieves matching excerpts from a small
local knowledge corpus, asks an LLM for a written assessment, and validates the
reply against a strict schema.

**The DPI engine needs nothing installed** — standard library only. The AI layer
needs two packages, and its retrieval stage needs two more. Both are optional
and both degrade to "skipped", never to a crash.

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

With the optional AI layer enabled, `analyze_ai.py` then:

- builds a redacted `CaptureReport` from the engine's flow snapshot — counters,
  application distribution and per-flow metadata, never packet payloads,
- extracts deterministic *signals* from that report (high DNS volume, unknown
  application share, plaintext HTTP, and so on),
- retrieves reference excerpts matching those signals from `knowledge/`,
- sends capture and excerpts to an LLM provider under a strict JSON schema,
- validates the reply — including every flow id and knowledge citation it makes
  — before printing it.

The two layers are one-way. `dpi/` does not import `ai/`; deleting `ai/`,
`knowledge/` and `evaluation/` leaves a working packet analyzer.


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

**AI analysis (optional)**
- Three providers behind one SDK: Groq, Ollama (local), OpenAI
- Structured output validated against a Pydantic schema; a reply that cites a
  flow id the capture does not contain is rejected
- Payloads are never sent; addresses are redacted by policy
- Every failure is a reported outcome, not an exception — a missing key, an
  unreachable endpoint or an oversized request all leave the DPI report intact

**Retrieval (RAG, optional)**
- Six hand-written knowledge documents, chunked into 37 section-aware excerpts
- Local embeddings (`BAAI/bge-small-en-v1.5`, 384 dimensions) — no API key, no
  network after the first download
- NumPy cosine search; no vector database
- Signal-driven queries: one query per observation, merged and ranked
- A context budget bounds what reaches the prompt; excerpts that do not fit are
  excluded whole and listed, never truncated


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

### AI and retrieval pipeline

Everything below runs *after* the DPI engine has finished and written its
output. Nothing here can change what the engine found.

```
   FlowSnapshot                     what the engine measured
        |
   ai/extractor.py                  build a CaptureReport
   ai/redaction.py                  redact addresses, drop payloads
        |
        +--> ai/rag/signals.py      deterministic observations
        |         |
        |    ai/rag/retrieval.py    one query per signal -> vector search
        |         |
        |    ai/rag/context.py      budget the excerpts; render [K1]..[Kn]
        |         |
        v         v
   ai/prompts.py                    system rules + capture + knowledge
        |
   ai/llm_client.py                 provider request, retry, diagnostics
        |
   ai/schemas.py                    validate into AnalysisResult
        |
   ai/analyzer.py                   check flow ids and knowledge citations
        |
   ai/report.py                     console report
```

Two boundaries are deliberate and are enforced by tests:

- **Signals read the redacted report, not the snapshot.** `ai/rag/signals.py`
  never sees a live address or a payload byte.
- **Retrieval queries carry only numbers.** Hostnames come from TLS SNI and are
  attacker-supplied, so only numeric and boolean evidence is rendered into a
  query, and every assembled query is checked against an address and domain
  pattern before use.


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

ai/                        the optional AI analysis layer
  config.py                settings and environment resolution
  extractor.py             FlowSnapshot -> CaptureReport
  redaction.py             address redaction and hostname sanitization
  schemas.py               CaptureReport and AnalysisResult models
  prompts.py               system prompt, capture rendering, knowledge block
  providers.py             provider registry and structured-output modes
  llm_client.py            provider requests, retries, sanitized diagnostics
  analyzer.py              orchestration; never raises for expected failures
  report.py                console rendering of an analysis

ai/rag/                    retrieval-augmented generation
  documents.py             load and parse the knowledge corpus
  chunking.py              deterministic section-aware chunking
  embeddings.py            local sentence-transformers embeddings
  vector_store.py          NumPy cosine search over the chunk matrix
  signals.py               deterministic observations from a CaptureReport
  retrieval.py             signal-driven queries, merging and ranking
  context.py               context budget and the [K1]..[Kn] block
  affinity.py              signal/knowledge compatibility (OFF by default)
  pipeline.py              wiring; every failure is a status, not an exception

knowledge/                 the reference corpus (6 documents, 6 categories)
  MANIFEST.md              what is here, how it is licensed, how to add more

evaluation/                measurement only; never imported by ai/ or dpi/
  cases.py                 8 labelled evaluation captures
  metrics.py               Recall@K, Precision@K, Hit@K, MRR
  candidates.py            named configurations under evaluation

main.py                    packet-by-packet decode
main_simple.py             flow list with detected hostnames
main_working.py            single-threaded filtering
main_dpi.py                multi-threaded filtering (main DPI tool)
dpi_mt.py                  alternate self-contained engine
analyze_ai.py              DPI + AI analysis (main AI tool)

run_selftests.py           DPI engine self-tests and tool smoke tests
run_ai_tests.py            AI layer
run_rag_tests.py           knowledge corpus loading
run_rag_chunk_tests.py     chunking
run_rag_embedding_tests.py embeddings
run_rag_vector_store_tests.py  vector store
run_rag_signal_tests.py    signal extraction
run_rag_retrieval_tests.py retrieval
run_rag_analysis_tests.py  knowledge-grounded analysis
run_rag_eval_tests.py      the evaluation harness itself
run_rag_quality_tests.py   retrieval quality and the request path
run_rag_evaluation.py      the evaluation run (not a test runner)

test_dpi.pcap              sample capture (77 packets)
generate_test_pcap.py      builds a new sample capture
.env.example               every environment variable, documented
```


## 5. Requirements


**Python 3.10 or newer** (tested on 3.10, 3.11 and 3.13).

Dependencies come in three tiers. Each is optional on top of the one before it,
and skipping a tier disables a feature rather than breaking the program.

| Tier | Install | Enables |
|---|---|---|
| DPI engine | nothing | everything in sections 7, 8 and 14 |
| AI analysis | `pip install -r requirements.txt` | `analyze_ai.py`, providers, structured output |
| Retrieval | `pip install -r requirements-rag.txt` | knowledge retrieval in `analyze_ai.py` |

`requirements.txt` is `pydantic>=2.0` and `openai>=1.40`. One SDK serves all
three providers: Groq and Ollama both expose OpenAI-compatible endpoints, so no
`groq` or `ollama` package is needed.

`requirements-rag.txt` is `numpy>=1.24` and `sentence-transformers>=3.0`. The
latter pulls in PyTorch (roughly 2 GB installed), which is why it is kept
separate. First use downloads `BAAI/bge-small-en-v1.5` (~130 MB) into the local
model cache; after that, embedding is fully offline.


## 6. Installation


```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python run_selftests.py
```

If that prints `15/15 checks passed`, the DPI engine is ready. There is no
build step and nothing to install for it.

For the AI layer:

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt        # AI analysis
pip install -r requirements-rag.txt    # + knowledge retrieval

cp .env.example .env                   # then set a provider key
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


## 9. AI-Assisted Analysis


`analyze_ai.py` runs the DPI engine unchanged, then sends a sanitized summary of
what it found to an LLM. **The AI layer is optional.** With no API key the DPI
analysis still runs and prints in full, only the AI section is skipped, and the
program exits 0 — because nothing went wrong.

```bash
python analyze_ai.py test_dpi.pcap
python analyze_ai.py test_dpi.pcap --provider groq --block-app YouTube
python analyze_ai.py test_dpi.pcap --provider ollama --model llama3.1
python analyze_ai.py test_dpi.pcap --no-rag
python analyze_ai.py test_dpi.pcap --show-knowledge
python analyze_ai.py test_dpi.pcap --show-payload
```

| Option | Meaning |
|---|---|
| `--output <file.pcap>` | Write filtered traffic here (default: discard) |
| `--json <file.json>` | Save the structured AI result as JSON |
| `--block-ip` / `--block-app` / `--block-domain` / `--rules` | As `main_dpi.py` |
| `--lbs <n>` / `--fps <n>` | Thread counts, as `main_dpi.py` |
| `--provider <name>` | `groq` \| `ollama` \| `openai` (default from `DPI_LLM_PROVIDER`) |
| `--model <name>` | Override the model for the selected provider |
| `--ip-mode <mode>` | `full` \| `redact_private` \| `none` (default `redact_private`) |
| `--no-ai` | Run DPI only; skip the AI layer entirely |
| `--no-rag` | Skip knowledge retrieval; send the capture alone |
| `--show-knowledge` | Print the retrieved excerpts, then continue |
| `--rag-max-items <n>` | Most reference excerpts to send (default 4) |
| `--rag-max-chars <n>` | Character ceiling on the excerpts (default 3000) |
| `--rag-max-tokens <n>` | Estimated-token ceiling, or `none` (default 900) |
| `--show-payload` | Print exactly what would be sent, then exit without sending |

`analyze_ai.py` parses its own arguments; it does not use `argparse`. Run it
with `-h` for the built-in usage text.

### What leaves the machine

Raw packet payloads are never sent. `ai/redaction.py` defines exactly what does:
capture-wide counters, the application distribution, and per-flow metadata
(flow id, protocol, ports, server name, application, state, packet and byte
counts, TCP flags, verdict). Private addresses are replaced with stable
pseudonyms under the default `redact_private` mode.

### Providers

| Provider | Structured output | Key | Notes |
|---|---|---|---|
| Groq | `json_schema`, strict | `GROQ_API_KEY` | Default. Free tier; see the rate-limit note in section 15 |
| Ollama | `json_object` | none | Local; the schema travels in the prompt and is validated on return |
| OpenAI | native parse | `OPENAI_API_KEY` | Requires paid credits |

Provider behaviour worth knowing:

- **HTTP 413** is classified `REQUEST_TOO_LARGE` and is **not retried** — the
  request will be the same size next time. The failure names the knobs that
  make it smaller.
- **HTTP 429** is classified `RATE_LIMITED` and **is retried**, with exponential
  backoff plus jitter (0.5 s base, 8 s cap), bounded by `DPI_AI_MAX_RETRIES`
  (default 3, so at most 4 attempts).
- **No provider token ceiling is hard-coded.** A per-request limit is a property
  of your account tier, which this project cannot observe. Set
  `DPI_MAX_INPUT_TOKENS` if you know yours and want the request refused locally
  before the round trip.
- **Diagnostics never contain an API key.** Status, error code, error type and
  request id are reported; secrets are redacted. This is covered by tests.

### The response schema

Replies are validated against `AnalysisResult` in `ai/schemas.py`. Two details:

- `schema_version` is **omitted from the provider-facing schema**
  (`PROVIDER_OMITTED_FIELDS` in `ai/providers.py`). It renders as a JSON Schema
  `const`, and nothing in the prompt tells a model what our version string is,
  so asking it to reproduce one is a request that can only fail. `AnalysisResult`
  still declares and validates the field; Pydantic supplies the default.
- Every `flow_id` and every `[K*]` citation in a reply is checked against what
  was actually supplied. An invented reference makes the response invalid.


## 10. Retrieval (RAG)


Retrieval is **on by default** in `analyze_ai.py` and turns off with `--no-rag`.
If the optional dependencies are missing, or the embedding model cannot be
loaded, the analysis still runs — without knowledge, and the report says so.

The corpus is six hand-written documents under `knowledge/`, one per category
(`glossary`, `protocols`, `baselines`, `detection-heuristics`,
`triage-playbooks`, `attack-patterns`), described in `knowledge/MANIFEST.md`.
They chunk into 37 section-aware excerpts, embedded locally with
`BAAI/bge-small-en-v1.5` into 384 dimensions and searched by cosine similarity
over a NumPy matrix. No vector database is used: 37 chunks do not need
approximate nearest neighbours.

Retrieval is driven by *signals* rather than by prose. Each observation the
capture produced gets its own query, plus one capture-wide query for protocol
and port context; results are merged by best score and ranked.

### Shipped defaults

These are the production values, and they are deliberately frozen — every one
was measured before being kept.

| Setting | Value | Where |
|---|---|---|
| `per_query_top_k` | 4 | `ai/rag/retrieval.py` |
| `final_top_k` | 8 | `ai/rag/retrieval.py` |
| `max_per_document` | 2 | `ai/rag/retrieval.py` |
| `min_similarity` | `None` | `ai/rag/retrieval.py` |
| `affinity` | `OFF` | `ai/rag/retrieval.py` |
| `query_style` | `"security"` | `ai/rag/retrieval.py` |
| `max_items` | 4 | `ai/rag/context.py` |
| `max_chars` | 3000 | `ai/rag/context.py` |
| `max_total_tokens` | 900 | `ai/rag/context.py` |
| `max_input_tokens` | `None` | `ai/config.py` / `ai/providers.py` |
| `DEFAULT_MAX_FLOWS` | 40 | `ai/config.py` |
| `capture_format` | `"table"` | `ai/prompts.py` |
| `PROMPT_VERSION` | `"2.0"` | `ai/prompts.py` |

`run_rag_quality_tests.py` pins all of these, so changing one is a deliberate
edit rather than a drift.

`affinity` is a signal/knowledge compatibility mechanism in `ai/rag/affinity.py`.
It is implemented and tested but **off by default**: measured against the real
index it did not beat the baseline, so it ships disabled rather than deleted.

### Capture format

The capture reaches the model as a **table** by default: column names once, then
one row per flow. The previous layout — pretty-printed JSON — remains available:

```bash
DPI_CAPTURE_FORMAT=table    # default
DPI_CAPTURE_FORMAT=json     # previous layout, byte for byte
```

The table carries every flow, every field and every value; nothing is dropped,
rounded, summarised or truncated, and a regression test compares the flow-id set
across both layouts. What changes is size: on the 27-flow sample capture the
serialized report is 13,563 characters as JSON, of which the flow list is 83%
and repeated field names alone are 5,697. The table form of the same report is
substantially smaller — measured reductions across the evaluation captures range
from 25% to 54%.

The separator is `|`, and `|` or `\` inside a value is escaped, so an
attacker-supplied hostname cannot invent a column.


## 11. Configuration


Every variable below is read by the code. `.env.example` documents all of them;
copy it to `.env`, which is gitignored.

| Variable | Default | Purpose |
|---|---|---|
| `DPI_LLM_PROVIDER` | `groq` | `groq` \| `ollama` \| `openai` |
| `GROQ_API_KEY` | — | Required when the provider is Groq |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Model override |
| `GROQ_BASE_URL` | Groq endpoint | Endpoint override |
| `OPENAI_API_KEY` | — | Required when the provider is OpenAI |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model override |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Endpoint override |
| `OLLAMA_MODEL` | `llama3.1` | Model override |
| `DPI_AI_MODEL` | — | Generic model override, used if the provider one is unset |
| `DPI_AI_BASE_URL` | — | Generic endpoint override |
| `DPI_AI_TIMEOUT` | `30` | Wall-clock timeout for one request, seconds |
| `DPI_AI_MAX_RETRIES` | `3` | Retries for transient failures |
| `DPI_AI_MAX_FLOWS` | `40` | Most flow records in one request |
| `DPI_AI_IP_MODE` | `redact_private` | `full` \| `redact_private` \| `none` |
| `DPI_CAPTURE_FORMAT` | `table` | `table` \| `json` — how the capture is laid out |
| `DPI_MAX_INPUT_TOKENS` | unset | Refuse a request above this estimate; `0` disables |
| `DPI_RAG_MAX_ITEMS` | `4` | Reference excerpts per request |
| `DPI_RAG_MAX_CHARS` | `3000` | Character ceiling on those excerpts |
| `DPI_RAG_MAX_TOKENS` | `900` | Estimated-token ceiling, or `none` |
| `DPI_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `DPI_EMBED_DEVICE` | auto | Torch device for embedding |
| `DPI_EMBED_BATCH_SIZE` | `16` | Embedding batch size |
| `DPI_EMBED_CACHE` | — | Model cache directory |
| `DPI_EMBED_OFFLINE` | unset | `1` requires the cached model and fails rather than fetching |
| `DPI_NO_CONSOLE_UTF8` | unset | `1` skips the console encoding fix in `dpi/console.py` |

Token figures throughout are an estimate at 3.5 characters per token,
deliberately pessimistic. No tokenizer is installed for this.


## 12. Testing


```bash
python run_selftests.py
```

This runs 15 checks — a self-test built into each of the 10 engine modules, plus
a smoke run of each of the 5 command-line tools — and covers **the DPI engine
only**. It needs no network and no third-party package.

```
15/15 checks passed
```

The AI and RAG layers have their own runners. Each is standalone and prints its
own count:

```bash
python run_ai_tests.py                 # AI layer: config, schemas, prompts, client
python run_rag_tests.py                # knowledge corpus loading and validation
python run_rag_chunk_tests.py          # deterministic chunking
python run_rag_embedding_tests.py      # embeddings
python run_rag_vector_store_tests.py   # vector store and cosine search
python run_rag_signal_tests.py         # signal extraction
python run_rag_retrieval_tests.py      # signal-driven retrieval
python run_rag_analysis_tests.py       # knowledge-grounded analysis end to end
python run_rag_eval_tests.py           # the evaluation harness itself
python run_rag_quality_tests.py        # retrieval quality and the request path
```

**Some checks are gated and report as skipped, not passed.** A runner that needs
`BAAI/bge-small-en-v1.5` skips those checks when the model cannot be loaded, and
a runner with a live-provider check skips it when no API key is set. The skips
are counted and printed separately — a run that says `125/125 checks passed, 13
skipped` has not silently passed those 13.

Individual DPI module self-tests can still be run directly:

```bash
python -m dpi.types
python -m dpi.packet_parser test_dpi.pcap
python -m dpi.sni_extractor
```

Use `python -m dpi.<name>`, not `python dpi/<name>.py` — two modules are named
`types` and `platform`, which shadow standard-library modules of the same name
when executed as scripts.


## 13. Evaluation


`run_rag_evaluation.py` measures the retrieval and analysis pipeline against a
fixed labelled dataset in `evaluation/cases.py`. It is a measurement tool, not a
test runner: it reports numbers, and a bad number is information rather than a
failure.

```bash
python run_rag_evaluation.py                              # console report
python run_rag_evaluation.py --json --out result.json     # machine-readable
python run_rag_evaluation.py --live                       # adds a live-LLM pass
```

Use `--out` rather than shell redirection. On Windows PowerShell, `>` is
`Out-File`, whose default encoding is UTF-16 with a byte-order mark; a perfectly
good JSON document then fails to load. `--out` writes UTF-8 without a BOM and,
in `--json` mode, reads the file back and parses it before reporting success.

`--live` spends provider quota and is opt-in for that reason.

Sections that need the embedding model state that they were skipped, and why.
They are never estimated or faked.

### Evaluation candidates are not configuration

`evaluation/candidates.py` holds **named configurations under measurement** —
`baseline`, `A`–`D`, `D-partial`, attribution probes, and an `unbounded`
reference. They exist so that a proposed change can be measured against the
shipped one before anybody argues for it.

**None of them is production configuration.** `ai/` imports nothing from
`evaluation/`, and neither Candidate D nor D-partial has been adopted. The
shipped defaults are the ones in section 10, and `baseline` is asserted by test
to match them — so the harness cannot quietly measure against a stale baseline.


## 14. Knowledge Base
The reference corpus lives in `knowledge/`, one directory per category, and is
described by `knowledge/MANIFEST.md` — which also records provenance, licensing
and the review policy for each document.

Six documents at present, one per category:

| Category | Document | Topic |
|---|---|---|
| `glossary/` | `dpi-network-security-terms` | DPI, flow, five-tuple, SNI, QUIC, verdict |
| `protocols/` | `dns-normal-behaviour` | What ordinary DNS resolution looks like |
| `baselines/` | `cdn-and-multi-host-traffic` | Why benign browsing looks noisy |
| `detection-heuristics/` | `suspicious-dns-indicators` | DNS thresholds and their false positives |
| `triage-playbooks/` | `triaging-unknown-application-traffic` | What to do about `UNKNOWN` flows |
| `attack-patterns/` | `dns-tunneling` | Covert DNS channels (T1071.004, T1048) |

Every document carries YAML front matter — `id`, `title`, `category`,
`applies_to`, `keywords`, `severity_hint`, `sources`, `licence` — and six fixed
headings. `run_rag_tests.py` enforces both, so a malformed document fails a test
rather than degrading retrieval quietly. `MANIFEST.md` documents how to add one.


## 15. Example Usage


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


## 16. Current Limitations


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

**AI layer**

- **Requires a provider.** With no key configured the AI section is skipped.
  Ollama removes the network and key requirement but needs a local install.
- **Subject to provider rate limits.** The Step 11 live verification attempted
  six evaluation cases in one burst: one completed successfully, five were
  refused with HTTP 429 `rate_limit_exceeded`, `type=tokens`. Requests were
  roughly 4,300–4,900 estimated tokens each, so six back-to-back requests total
  around 27,600 — more than a small per-minute token allowance. That is an
  observed limit of the account used, not a claim that the application cannot
  analyse six captures; spacing the runs, or an account with a larger allowance,
  changes the outcome. There were **zero** HTTP 413 and **zero** HTTP 400
  failures in that run.
- **Models move.** Groq retires models periodically; a `model_not_found` error
  means setting `GROQ_MODEL` to a current one.
- **One sample proves a path, not a quality level.** A single successful live
  analysis shows the request path works. It does not characterise how good the
  assessments are.

**Retrieval**

- **Needs the optional dependencies.** Without `requirements-rag.txt` the
  analysis runs without knowledge and says so.
- **First run needs network.** Downloading `BAAI/bge-small-en-v1.5` requires
  reaching the model host once; `DPI_EMBED_OFFLINE=1` requires the cached copy
  and fails rather than fetching.
- **Small corpus.** Six documents and 37 chunks. Evaluation numbers describe
  this dataset, not a population.
- **Token counts are estimates.** 3.5 characters per token, deliberately
  pessimistic. No tokenizer is installed.


## 17. Future Development


Not implemented — listed as direction, not as capability:

- `.pcapng` support
- IPv6 parsing
- TCP stream reassembly for segmented handshakes
- Live capture from a network interface
- JSON / CSV export of flow records
- A `pytest` suite alongside the built-in self-tests
- Statistical traffic analysis
