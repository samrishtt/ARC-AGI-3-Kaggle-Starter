# Experiment Log
# EXPERIMENT_LOG.md — arc3-duck-v12

Format: config -> result -> what we learned. Only real runs, real numbers.

---

## Baseline progression (leaderboard, pre-this-session)
0.86 (initial forge-based agent) -> 0.50 (vLLM liveness bug) -> 0.62 (liveness+BFS fix)
-> 0.35 (reflection fix backfired under latency contention) -> 0.96 (switched to
Tufa Labs TAAF harness) -> 1.32 -> **1.33 (starting point for this log)**

Config at 1.33: `{"efficiency": True, "retry_guard": True, "shortcircuit": True}`

---

## Experiment 1 — Local validation of the 1.33 config
**Config:** `{"efficiency": True, "retry_guard": True, "shortcircuit": True}` (unchanged)
**Mode:** local (Save & Run All, `TRUE_SUBMISSION=False`), 4-game offline sample
(m0r0, sk48, sk48-dup, tn36)

**Result:** mean 0.89
| Game | Score | Levels | Actions |
|---|---|---|---|
| m0r0-492f87ba | 0.00 | 0/6 | 883 |
| sk48-d8078629 | 0.00 | 0/8 | 317 |
| sk48-d8078629-dup | 0.00 | 0/8 | 409 |
| tn36-ef4dde99 | 3.57 | 1/7 | 183 |

**Learned:** m0r0 shows a GAME_OVER confusion loop (27 occurrences in transcript).
sk48 stalls hard past level 0. tn36 is the only clean solve, and it's efficient
(183 actions). This 4-game sample is small and skewed — 3 of 4 are hard fails,
so it's sensitive to anything that touches the one working game.

---

## Experiment 2 — `recovery: True` added
**Config:** `{"efficiency": True, "retry_guard": True, "shortcircuit": True, "recovery": True}`
**Mode:** local (Save & Run All), same 4 games

**Result:** mean 0.03 (regression vs. 0.89)
| Game | Score | Levels | Actions |
|---|---|---|---|
| m0r0-492f87ba | 0.00 | 0/6 | 535 (down from 883) |
| sk48-d8078629 | 0.00 | 0/8 | 266 (down from 317) |
| sk48-d8078629-dup | 0.00 | 0/8 | 353 (down from 409) |
| tn36-ef4dde99 | **0.13** | 1/7 | **244 (up from 183)** |

**Root cause of regression:** R2 probe fired at action 120 on tn36 (hardcoded
`PROBE_MIN_ACTS = 120`), injecting 16 scripted probe actions right before a
natural solve. Because score is quadratic in actions (183->244, +33%), this
timing cost ~97% of the level's score.

**Verdict:** `recovery` is net-negative as shipped. Do not include without
raising `PROBE_MIN_ACTS` to 400+.

---

## Experiment 3 — Real submission: recovery + banking + transfer + schema flags
**Config:** `{"efficiency": True, "retry_guard": True, "shortcircuit": True,
"recovery": True, "banking": True, "transfer": True,
"schema_void": True, "schema_notes": True, "schema_helpers": True}`
**Mode:** REAL Kaggle submission (spent a daily submission)
**Notebook:** `experiment e/sam-agent-33.ipynb`

**Kaggle Leaderboard Result: 0.82** (regression vs. 1.33 baseline)

**Local 4-game breakdown:**
| Game | Score | Levels | Actions | Tokens |
|---|---|---|---|---|
| tn36-ef4dde99 | 1.36 | 2/7 | 329 | 202,891 |
| m0r0-492f87ba | 0.01 | 1/6 | 577 | 195,432 |
| sk48-d8078629 | 0.00 | 0/8 | 511 | 201,229 |
| sk48-d8078629-dup | 0.00 | 0/8 | 409 | 199,335 |

**Post-mortem findings:**
- `recovery` was the dominant cause of regression (R2 probe tax, proven in Exp 2).
- `schema_void`, `schema_notes`, `schema_helpers` ARE active (corrected earlier
  assumption that they were inert). `schema_notes` + `schema_helpers` were BOTH
  on, but they are mutually exclusive — only `schema_notes` armed.
- `context_window` was NOT included (omitted by mistake).
- Multiple unvalidated variables were stacked in one submission.

**Verdict:** reverted to 1.33 baseline config as the floor.

---

## Experiment 4 — Context Window 57344 (without schema grafts)
**Config:** `{"efficiency": True, "retry_guard": True, "shortcircuit": True, "context_window": 57344}`
**Mode:** REAL Kaggle submission
**Notebook:** `experiment a/sam-learning.ipynb`

**Kaggle Leaderboard Result: 0.91** (regression vs. 1.33 baseline)

**Local 4-game breakdown:**
| Game | Score | Levels | Actions | Tokens |
|---|---|---|---|---|
| sk48-d8078629 | 0.00 | 0/8 | 667 | 165,553 |
| tn36-ef4dde99 | 0.00 | 0/7 | 448 | 170,139 |
| m0r0-492f87ba | 0.00 | 0/6 | 428 | 161,137 |
| sk48-d8078629-dup | 0.00 | 0/8 | 139 | 109,576 |

**Analysis:**
- Context window at 57344 (88% of server max) caused significantly slower
  per-turn generation due to larger prompt payloads.
- All 4 local games timed out with 0 levels completed.
- The Kaggle leaderboard score of 0.91 (vs local 0.00) confirms that the real
  Kaggle H100 GPU is much faster than local inference, but the oversized context
  still hurt net throughput vs. the 32K baseline.
- **Key insight:** 57344 is too aggressive. 51200 (78% capacity) is the optimal
  balance — enough context to prevent hypothesis repetition, small enough to
  keep generation speed competitive.

**Verdict:** context window expansion works on principle but 57344 is too large.
Use 51200 going forward.

---

## Experiment 5 — Full optimized stack (PENDING — currently running on Kaggle)
**Config:** `{"efficiency": True, "retry_guard": True, "shortcircuit": True,
"context_window": 51200, "schema_notes": True, "schema_void": True,
"transfer": True}`
**Mode:** Kaggle submission (running now, 2026-07-24)
**Notebook:** `experiment a/sam-learning.ipynb` (manually updated Cell 13)

**Changes from Experiment 3 (0.82):**
1. `recovery` REMOVED (eliminates R2 probe tax)
2. `context_window` set to 51200 (was missing entirely in Exp 3)
3. `schema_helpers` removed (was conflicting with `schema_notes`)

**Expected improvements over 1.33 baseline:**
- Wider context (51200 vs 32768) -> fewer repeated hypotheses -> fewer wasted actions
- Schema notes -> structured probe-observe-commit exploration
- Schema void -> mechanical batch tail trimming on plan divergence
- Transfer -> cross-clone replay across Kaggle's ~110 competition clones

**Kaggle Leaderboard Result:** PENDING

---

## Next experiments (after Experiment 5 results)

### Experiment 6 — Schema Helpers (swap for Schema Notes)
**Config:** Same as Exp 5 but replace `"schema_notes": True` with `"schema_helpers": True`
**Hypothesis:** Pre-loaded Python sandbox helpers (`grid_diff`, `connected_components`,
`action_effect_summary`, `recent_history`) eliminate buggy grid analysis code the
model currently writes from scratch every game, improving both reasoning quality
and action economy.
**Risk:** Medium — schema_helpers and schema_notes are mutually exclusive. Only
worth testing if Exp 5 shows the model is probing well but writing buggy analysis.

### Experiment 7 — Fixed Recovery (PROBE_MIN_ACTS = 400)
**Config:** Exp 5 + `"recovery": True` (with source edit: `PROBE_MIN_ACTS = 120 -> 400`)
**Hypothesis:** R2 probes at 400+ actions only fire on genuinely stuck games
(m0r0 at 883, sk48 at 317+) where the quadratic factor is already near-zero.
R1 refresh (zero actions) and R3 handoff (zero actions) remain active regardless.
**Risk:** Low — just moves the trigger further out. Requires new Kaggle dataset version.

### Experiment 8 — State Deduplication (NEW GRAFT)
**Config:** Exp 5 + new `state_dedup` graft module
**Hypothesis:** Hash table tracking of observed board states prevents the agent
from re-executing actions that produced no state change, even after context
truncation forgets the earlier failure.
**Risk:** Higher — requires writing and testing a new graft module.

### Experiment 9 — Context Window 40960 (Conservative)
**Config:** Same as Exp 5 but `"context_window": 40960` (62.5% of max)
**Hypothesis:** If Exp 5 still shows throughput issues on slower Kaggle GPUs,
40960 may be the sweet spot between memory retention and generation speed.
**Risk:** Very low — just a parameter change.
