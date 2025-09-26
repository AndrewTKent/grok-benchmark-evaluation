# Grok Terminal‑Bench Evaluation

Evaluate Grok on the [Terminal‑Bench](https://github.com/laude-institute/terminal-bench) benchmark with a production‑ready runner, two agent variants (standard & enhanced), live progress tailing, and post‑hoc analytics.

---

## ✨ What’s new in this version

- **Real Terminal‑Bench integration** – Uses `tb run` with tmux sessions and proper `BaseAgent` adapters.
- **Two agents**  
  - `GrokTerminalAgent` (baseline, deterministic)  
  - `EnhancedGrokTerminalAgent` (loop detection, safety checks, optional failure injection, composite scoring)
- **Enhanced analysis** – Aggregates per‑task metrics and writes an **enhanced analysis report** (`enhanced_analysis.json` + `.txt`).
- **Live progress** – A background tailer prints `progress.jsonl` and `commands.txt` lines per trial for real‑time visibility.
- **Parallel execution fixed** – `--n-concurrent` is now honored across modes, including **compare** mode. The runner no longer hard‑codes `n_concurrent=1` during comparisons.
- **Robust tmux compatibility** – Unified “send + Enter” and pane capture helpers work across Terminal‑Bench versions.
- **Clean command extraction & validation** – Heuristics to strip code fences, avoid explanations, and repair common failure patterns.
- **Resilient Grok client** – Timeouts, retries, exponential backoff, and rate‑limit handling.

---

## 📦 Repository structure

```
.
├─ run.py                       # Thin CLI entrypoint → src/cli/main.py
├─ setup.sh                     # Optional environment bootstrap
├─ requirements.txt
├─ src/
│  ├─ cli/
│  │  └─ main.py               # argparse CLI: list, test, run, compare
│  ├─ clients/
│  │  └─ grok_client.py        # OpenAI‑compatible client with retries/backoff
│  ├─ agents/
│  │  ├─ grok_terminal_agent.py    # Baseline TB agent
│  │  └─ enhanced_agent.py         # Enhanced agent (loop/safety/failure‑inject)
│  ├─ runners/
│  │  ├─ tb_runner_base.py     # Shared setup, env prep, result parsing, live tailer
│  │  ├─ standard_runner.py    # Baseline runner
│  │  └─ enhanced_runner.py    # Enhanced/Compare runners; honors --n-concurrent
│  ├─ analysis/
│  │  └─ analyzer.py           # EnhancedAnalyzer → JSON + TXT summaries
│  ├─ metrics/                 # Scoring & instrumentation used by enhanced agent
│  │  ├─ safety_checker.py
│  │  ├─ loop_detector.py
│  │  ├─ failure_injector.py
│  │  └─ scoring.py
│  └─ utils/
│     ├─ tmux_compat.py        # send_with_enter(), safe_read_pane()
│     └─ progress.py           # ProgressEvent/Reporter helpers
└─ results/                     # Created at runtime (TB outputs + enhanced metrics)
```

---

## ✅ Prerequisites

- Python **3.12+**
- **Docker** (for Terminal‑Bench tasks)
- `terminal-bench` CLI (`pip install terminal-bench`)
- Grok API key from **console.x.ai** set as one of:
  - `XAI_API_KEY` (preferred) or `GROK_API_KEY`
- (Optional) `GROK_MODEL` (defaults to `grok-4-fast-reasoning` in runner; client default `grok-2-1212`).

Create a local `.env` if you prefer:
```
XAI_API_KEY=your_key_here
GROK_MODEL=grok-4-fast-reasoning
GROK_TIMEOUT=60
GROK_MAX_RETRIES=3
```

---

## 🚀 Quick start

```bash
# 1) Clone
git clone https://github.com/AndrewTKent/grok-benchmark-evaluation.git
cd grok-benchmark-evaluation

# 2) (Optional) bootstrap
bash setup.sh

# 3) Activate venv if setup.sh created one
source venv/bin/activate  # or your environment

# 4) Install Terminal‑Bench and deps
pip install -r requirements.txt
pip install terminal-bench

# 5) Download dataset (first run only)
tb datasets download --dataset terminal-bench-core==0.1.1

# 6) Diagnostic quick test
python run.py --test
```

---

## 🧭 CLI overview

The entrypoint `run.py` simply dispatches to `src/cli/main.py`.

```
usage: run.py [options]

Common:
  --model MODEL
  --dataset terminal-bench-core==0.1.1
  --task-id TASK_ID           (repeatable)
  --n-concurrent INT          (parallel trials)
  --n-attempts INT            (retries per task)
  --timeout INT               (seconds per task)
  --debug

Modes:
  --list-tasks                List tasks in dataset
  --test                      Run diagnostic + simple TB run
  --diagnostic                Only run system diagnostics
  --enhanced                  Use EnhancedGrokTerminalAgent
  --compare                   Run baseline vs enhanced back-to-back
  --inject-failures           (enhanced) enable failure injection
  --injection-rate FLOAT      (enhanced) injection probability per step
```

### Examples

**List tasks**
```bash
python run.py --list-tasks --dataset terminal-bench-core==0.1.1
```

**Run a subset with parallelism**
```bash
python run.py \
  --dataset terminal-bench-core==0.1.1 \
  --n-concurrent 4 \
  --n-attempts 1 \
  --timeout 240 \
  --task-id hello-world \
  --task-id fix-git \
  --task-id fix-permissions \
  --task-id heterogeneous-dates
```

**Enhanced agent (metrics, safety, loops)**
```bash
python run.py \
  --enhanced \
  --dataset terminal-bench-core==0.1.1 \
  --n-concurrent 4 \
  --task-id hello-world --task-id fix-git
```

**Comparison (baseline vs enhanced)**
```bash
python run.py \
  --compare \
  --dataset terminal-bench-core==0.1.1 \
  --n-concurrent 4 \
  --task-id hello-world --task-id fix-git --task-id fix-permissions
```
> In this version the **compare** mode forwards `--n-concurrent` to both runs (no more implicit `n_concurrent=1`).

**Full benchmark**
```bash
python run.py --n-concurrent 4 --n-attempts 2
```

**Verbose debug**
```bash
python run.py --debug --test
```

---

## 📡 What runs under the hood

- **Standard path** → `StandardTerminalBenchRunner` + `GrokTerminalAgent`
- **Enhanced path** → `EnhancedTerminalBenchRunner` + `EnhancedGrokTerminalAgent`
  - Loop detection (`src/metrics/loop_detector.py`)
  - Safety checks (`src/metrics/safety_checker.py`)
  - Optional failure injection (`src/metrics/failure_injector.py`)
  - Composite score (`src/metrics/scoring.py`)
- **Live progress** (`_Tailer`) prints lines from:
  - `*/agent-logs/progress.jsonl` (structured events)
  - `*/commands.txt` (last seen command), grouped by trial
- **Grok client** adds request timeouts, retries, and 429 backoff.

---

## 📁 Outputs & analysis

Top‑level `results/` contains timestamped run folders, e.g.:
```
results/
└─ tb_grok-4-fast-reasoning_enhanced_20250925_235959/
   ├─ <task>/<trial>/agent-logs/progress.jsonl
   ├─ <task>/<trial>/commands.txt
   ├─ enhanced_metrics/
   │  ├─ metrics_<task>.json
   │  └─ ...
   ├─ enhanced_analysis.json         # machine‑readable
   ├─ enhanced_analysis.txt          # human summary
   └─ *.json                         # TB baseline artifacts if produced
```

**Enhanced analysis summary includes**  
- Success rate, composite score, efficiency, recovery, safety
- Loop statistics and common error patterns
- Top recommendations to improve agent behavior

---

## 🔍 Troubleshooting

**Docker**
```bash
docker info
docker run --rm alpine echo 'Docker OK'
```

**Terminal‑Bench**
```bash
tb --version
tb datasets download --dataset terminal-bench-core==0.1.1 --overwrite
```

**Environment / imports**
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
python -c "from src.agents.grok_terminal_agent import GrokTerminalAgent; print('Import OK')"
```

**API connectivity**
```bash
python -c "from src.clients.grok_client import GrokClient; GrokClient().test_connection()"
```

---

## ⚙️ Configuration notes

- Set `XAI_API_KEY` (or `GROK_API_KEY`) in shell or `.env`.
- Control model via `--model` or `GROK_MODEL` env.
- `--n-concurrent` controls **actual** parallelism in all modes, including `--compare`.
- The enhanced agent is deterministic in early steps and slightly exploratory later (`temperature` ramps beyond step 5).

---

## 📝 License

MIT — see `LICENSE`.

---

## 🤝 Contributing

Issues and PRs welcome. Please run:
```bash
python run.py --diagnostic
python run.py --test
```
