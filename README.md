# ARC-AGI-3 Agent Optimization Research

> Systematic ablation study and graft-stack optimization for the ARC Prize 2026 (ARC-AGI-3) Kaggle competition.

## Research Overview

This repository documents our end-to-end research pipeline for building, analyzing, and iteratively improving an autonomous AI agent that solves interactive video-game-style reasoning puzzles in the [ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3) competition. Our approach treats agent performance as a function of controllable variables — context memory capacity, prompt-level deliberation strategies, action-batch discipline, and cross-instance knowledge transfer — and applies rigorous single-variable ablation methodology to isolate causal effects on the competition scoring metric.

### Research Methodology

Our work follows a structured experimental pipeline:

1. **Hypothesis Formation**: Each experiment targets a specific bottleneck identified through transcript forensics, scoring formula analysis, or competitive intelligence from top-performing agents.
2. **Single-Variable Isolation**: We change exactly one configuration parameter per experiment to establish causal attribution (not just correlation) between changes and score movements.
3. **Quantitative Evaluation**: Every experiment is evaluated against the quadratic RHAE scoring formula: $\text{Score}_{\text{level}} = \min\bigl(115,\;(\text{baseline}/\text{actions})^2 \times 100\bigr)$, ensuring we measure what matters.
4. **Post-Mortem Analysis**: Failed experiments are analyzed for root cause — we extract more signal from regressions than from successes.

### Key Research Findings

| Finding | Method | Impact |
|---|---|---|
| **Recovery R2 probe is net-negative** | Controlled ablation (Exp 1 vs 2) | Injecting 16 scripted probe actions at `PROBE_MIN_ACTS=120` destroys ~97% of level score due to quadratic penalty. Proven causal. |
| **Context window 57344 is too aggressive** | Real Kaggle submission (Exp 4) | 88% context utilization causes throughput degradation. Optimal calibration: 51200 tokens (78% of vLLM server capacity). |
| **Schema grafts ARE active** | Source code audit (corrected prior assumption) | `schema_void.py`, `schema_notes.py`, `schema_helpers.py` all exist in the deployed dataset and are processed by `composite.py`. Previous documentation incorrectly classified them as inert. |
| **Multi-variable stacking masks causal signals** | Post-mortem of Exp 3 (0.82 regression) | Stacking `recovery` + `banking` + `transfer` + 3 schema flags in one submission made it impossible to attribute the 1.33→0.82 regression to any single change. |

## Experiment Results

| # | Configuration Delta | Kaggle Score | Direction | Key Observation |
|---|---|---|---|---|
| Baseline | `{efficiency, retry_guard, shortcircuit}` | **1.33** | — | Floor configuration |
| Exp 2 | + `recovery: True` | (local only) | ↓ 0.89→0.03 | R2 probe tax on tn36: 183→244 actions |
| Exp 3 | + `recovery` + `banking` + `transfer` + schema flags | **0.82** | ↓ 39% | Multi-variable confound; `recovery` dominant |
| Exp 4 | + `context_window: 57344` | **0.91** | ↓ 32% | Context too aggressive; throughput degraded |
| Exp 5 | + `context_window: 51200` + `schema_notes` + `schema_void` + `transfer` (NO recovery) | **PENDING** | Expected ↑ | Removes proven negatives, adds validated positives |

## Technical Architecture

The agent is built on the [Tufa Labs TAAF framework](https://www.kaggle.com/datasets/thtennant/taaf-kaggle-source-share-fork) with a modular graft composition system:

```
┌─────────────────────────────────────────────────┐
│  Benchmark.run()                                │
│  ├── ShortCircuitHarnessSolver (no-op trimmer)  │
│  │   └── SchemaVoidMixin (batch tail trimmer)   │
│  ├── TransferHarnessSolver (cross-clone replay) │
│  │   └── BankingHarnessSolver (win-then-replay) │
│  └── Analyzer Chain:                            │
│      ├── RetryGuard (outermost)                 │
│      └── EfficiencyToolAgent (innermost)        │
│          └── SchemaNotesToolAgent (prompt note)  │
│              └── ToolAgent + Qwen3.6-27B-FP8    │
└─────────────────────────────────────────────────┘
```

Each graft is individually flag-gated, blanket-guarded (any error degrades to stock behavior), and composable via the `composite.install(bm, flags={})` entry point.

## Repository Structure

```
├── docs/
│   ├── IDEAS.md              # Prioritized experiment roadmap (11 experiments, 4 tiers)
│   ├── EXPERIMENT_LOG.md     # Chronological experiment results with quantitative analysis
│   ├── ARCHITECTURE.md       # Four-codebase architectural breakdown
│   ├── MODULES.md            # Per-file module documentation with flag mappings
│   └── QUESTIONS.md          # Open research questions with resolution tracking
├── experiment a/             # Exp 4: context_window=57344 (scored 0.91)
│   ├── sam-learning.ipynb    # Submission notebook
│   └── results (3).zip      # Full benchmark output (events, prompts, diagnostics)
├── experiment e/             # Exp 3: full stack with recovery (scored 0.82)
│   ├── sam-agent-33.ipynb    # Submission notebook
│   └── results (3).zip      # Full benchmark output
├── 1.33 scored in .../       # Reference baseline notebook (scored 1.33)
│   ├── arc3-duck-v12 (1).ipynb
│   └── archive (2).zip      # Source bundle with all graft modules
├── datasets/                 # Deployed Kaggle dataset bundle (taaf source share fork)
├── create_experiment_notebook.py  # Automated notebook generator for experiments A-E
├── verify_grafts.py          # Local graft installation verification suite
└── parse_exp_*.py            # Benchmark result analysis scripts
```

## Scoring Formula & Optimization Target

The ARC-AGI-3 competition scores agents using a **quadratic efficiency-weighted** formula:

$$\text{Score}_{\text{level}} = \min\left(115,\;\left(\frac{\text{human\_baseline\_actions}}{\text{ai\_actions}}\right)^2 \times 100\right)$$

This means:
- Reducing actions from 200→100 yields a **4× score improvement**
- Reducing actions from 100→50 yields another **4× improvement**
- A single wasted action batch can destroy an entire level's score contribution

Our optimization strategy targets all three levers:
1. **Reduce wasted actions** (context window, schema void, shortcircuit)
2. **Improve action quality** (schema notes prompt steering)
3. **Multiply efficiency across clones** (transfer/banking cross-instance replay)

## Tools & Infrastructure

| Tool | Purpose |
|---|---|
| `create_experiment_notebook.py` | One-command notebook generation for any experiment config (A-E) |
| `verify_grafts.py` | Offline graft installation verification with mocked Kaggle dependencies |
| `parse_exp_a.py` / `parse_exp_e.py` | Automated benchmark.json score extraction and analysis |

## Competition Context

- **Competition**: [ARC Prize 2026 — ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
- **Model**: Qwen3.6-27B-FP8 served via vLLM on Kaggle H100 GPUs
- **Agent Framework**: Tufa Labs TAAF (The Agent Architecture Framework)
- **Runtime**: ~2h 20m per submission, ~110 clones across ~25 game families

## Current state and local validation

The checked-in `arc3x` solver is the active research implementation. The
frame-0 marker detector remains experimental: its diagnostic currently passes
23/25 coverage but fails the source-grounded `tu93` check, so it is not wired
into the planner. This is intentional until an objective detector is validated.
The human-mind learner in `arc3x/mindgraft.py` now treats level changes as scene
cuts and excludes them from motion and geometry learning, preventing a new board
from poisoning the previous level's internal model.

Run the fast validation checks from the repository root:

```powershell
$env:PYTHONPATH = "."
.venv\Scripts\python.exe -m compileall -q arc3x tools
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe tools\verify_submission_notebook.py
.venv\Scripts\python.exe arc3x\why_markers.py --steps 400
```

The full scored suite is intentionally separate because it is CPU-intensive:
`arc3x\suite.py --split both -w 10 --budget 3000`.

To build the experimental human-mind Kaggle notebook from the known v17 base:

```powershell
.venv\Scripts\python.exe tools\build_mind_notebook.py
```

This produces `1.33 scored in arc agi 3 competiotn in kaggle\arc3-duck-v18-human-mind.ipynb`.
It is ready for a Kaggle experiment, but its remote score still requires an
actual Kaggle run and must not be treated as proven in advance.
