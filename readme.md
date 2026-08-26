# Context Summarizer Service

Standalone rolling-summary service for a multi-session conversation system.
Powered by **Phi-4-mini-instruct Q4_K_M** (GGUF) via `llama-cpp-python`.

---

## Architecture at a glance

```
write_turn(session_id, role, text)          ← Entry Point 1 (periodic)
    │
    ├─ inserts turn to SQLite
    │
    └─ gap ≥ 15? ──yes──► ThreadPoolExecutor background job
                               └─ regenerate_summary(session_id, trigger='periodic')

summarize_on_demand(session_id) → str       ← Entry Point 2 (on-demand)
    └─ regenerate_summary(session_id, trigger='on-demand')  [blocks caller]

regenerate_summary(session_id, trigger)     ← Single source of truth
    ├─ reads current_summary + last_summarized_turn_index from DB
    ├─ fetches all turns where turn_index > last_summarized_turn_index
    ├─ calls Phi-4-mini-instruct with (previous summary + new turns)
    └─ atomically writes new summary + advances last_summarized_turn_index
```

**Key invariant:** `last_summarized_turn_index` is written in exactly one place
(`db.update_summary`) called by exactly one function (`regenerate_summary`).

---

## Project layout

```
summarizer/
├── __init__.py       Public API: init_service, write_turn, summarize_on_demand
├── config.py         Env-var driven settings (frozen dataclass)
├── db.py             SQLite schema + all read/write helpers
├── logging_cfg.py    Structured logging + log_summarization_event()
├── model.py          Phi-4-mini-instruct singleton loader + inference wrapper
├── prompt.py         Prompt template builder
└── service.py        regenerate_summary, write_turn, summarize_on_demand

tests/
└── test_summarizer.py  Full test suite (model stubbed — no GGUF needed)

requirements.txt
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

`llama-cpp-python` compiles a native extension. If you want CPU-only (default):

```bash
pip install llama-cpp-python
```

For a pre-built wheel (faster install, no compiler needed):

```bash
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

### 2. Download the model

```bash
# Example using huggingface-hub CLI
pip install huggingface-hub
huggingface-cli download \
  bartowski/microsoft_Phi-4-mini-instruct-GGUF \
  Phi-4-mini-instruct-Q4_K_M.gguf \
  --local-dir ./models
```

### 3. Configure environment variables

| Variable | Default | Required |
|---|---|---|
| `SUMMARIZER_MODEL_PATH` | *(none)* | **Yes** — hard-fails if absent |
| `SUMMARIZER_DB_PATH` | `./summarizer.db` | No |
| `SUMMARIZER_N_CTX` | `4096` | No |
| `SUMMARIZER_N_THREADS` | `2` | No |
| `SUMMARIZER_PERIODIC_THRESHOLD` | `15` | No |
| `SUMMARIZER_MAX_TOKENS` | `512` | No |

```bash
export SUMMARIZER_MODEL_PATH=/absolute/path/to/Phi-4-mini-instruct-Q4_K_M.gguf
export SUMMARIZER_DB_PATH=./summarizer.db
```

### 4. Use the service

```python
from summarizer import init_service, write_turn, summarize_on_demand

# Call once at startup
init_service()

# Entry Point 1 — write turns (periodic summarisation fires automatically)
write_turn("session-abc", "user",      "What's the project deadline?")
write_turn("session-abc", "assistant", "The deadline is Friday the 22nd.")
# ... after 15 unsummarised turns, background summary runs automatically

# Entry Point 2 — on-demand (e.g. called by the classifier component)
summary = summarize_on_demand("session-abc")
print(summary)
```

---

## Running the tests

No real model required — inference is stubbed.

```bash
# Standalone
python tests/test_summarizer.py

# Via pytest
pip install pytest
python -m pytest tests/test_summarizer.py -v
```

Tests cover:
- **A** — Periodic threshold fires at exactly turn 15, not before
- **B** — On-demand works standalone, updates DB correctly
- **C** — No gaps: periodic + on-demand together cover all turns with no skips
- **D** — Second concurrent background job for the same session is suppressed
- DB helper unit tests (monotonic turn_index, boundary queries, role validation)
- Prompt builder unit tests (token presence, placeholder logic, turn formatting)

---

## Memory footprint

| Component | RAM |
|---|---|
| Phi-4-mini Q4_K_M weights | ~2.5 GB |
| KV cache (n_ctx=4096) | < 200 MB |
| SQLite + Python overhead | < 50 MB |
| **Total** | **≤ ~2.8 GB** |

This leaves ~5.2 GB headroom on an 8 GB machine for the future ~2 GB classifier
model. Tune `SUMMARIZER_N_CTX` downward if you need more headroom.

---

## Logging

Every summarisation event emits one structured `INFO` line:

```
2026-08-15T23:05:00 | INFO     | summarizer.service | SUMMARIZED session=abc123 trigger=periodic turns=1-15 summary_chars=284
```

Fields: `session`, `trigger` (`periodic` | `on-demand`), `turns` range, `summary_chars`.

---

## Debug UI

A local single-page web UI for manually driving both services and inspecting
internal state in real time. Intended for developer testing only — not production.

### Setup

```bash
pip install fastapi "uvicorn[standard]"
# (already in requirements.txt)
```

### Configure

Set both model paths before starting (the UI's "Initialize" button triggers
model loading — it will fail loudly if these are wrong):

```bash
# Windows PowerShell
$env:SUMMARIZER_MODEL_PATH = "C:\path\to\Phi-4-mini-instruct-Q4_K_M.gguf"
$env:CLASSIFIER_MODEL_PATH = "C:\path\to\gemma-3-2b-it-Q4_K_M.gguf"

# Bash / macOS / Linux
export SUMMARIZER_MODEL_PATH=/path/to/Phi-4-mini-instruct-Q4_K_M.gguf
export CLASSIFIER_MODEL_PATH=/path/to/gemma-3-2b-it-Q4_K_M.gguf
```

### Run

```bash
uvicorn app:app --reload
```

Open **http://localhost:8000** in a browser.

### Usage walkthrough

1. **Click "Initialize Services"** — loads both GGUF models into RAM.
   This blocks for 30–90 s; the status badge turns green when done.
2. **Enter a session ID** (any string) and click **Use** — or pick an
   existing session from the dropdown.
3. **Add turns** in the Conversation panel (role selector + text input,
   or press Enter). Watch the gap counter count up toward 15 — when it
   hits 15 the periodic summarizer fires automatically in the background.
4. **Force Summarize Now** to manually trigger `summarize_on_demand` and
   see the summary refresh (yellow flash = changed).
5. **Classifier panel** — type any prompt and choose:
   - *Classify Only* — shows the raw YES/NO model output + parsed bool
   - *Get Full Context* — runs the full pipeline, shows the context block
   - *Run + Add as Turn* — runs the pipeline AND writes the prompt as a
     user turn so you can chain realistic prompts and watch state evolve
6. **Event Log** shows every API call made by the UI, with method, path,
   HTTP status, elapsed ms, and truncated response body.

### API docs

FastAPI's interactive Swagger UI is available at **http://localhost:8000/docs**.

### Files added

```
app.py              FastAPI backend (thin wrapper over both packages)
static/index.html   Single-page debug UI (vanilla HTML + JS, no build step)
```