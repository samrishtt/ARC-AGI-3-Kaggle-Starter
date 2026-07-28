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
**Result:** mean 0.89

---

## Experiment 2 — `recovery: True` added
**Config:** `{"efficiency": True, "retry_guard": True, "shortcircuit": True, "recovery": True}`
**Result:** mean 0.03 (regression vs. 0.89 due to R2 probe tax at action 120)

---

## Experiment 3 — Real submission: recovery + banking + transfer + schema flags
**Kaggle Leaderboard Result: 0.82** (regression due to `recovery` probe tax)

---

## Experiment 4 — Context Window 57344 (without schema grafts)
**Kaggle Leaderboard Result: 0.91** (regression due to 57K token latency penalty)

---

## Experiment 5 (Experiment B Folder) — Context 57344 + Schema Notes + Schema Void + Transfer
**Kaggle Leaderboard Result: 0.66**
**Breakthrough:** `tn36` action count dropped from 183 to 84 (-54%), but 57K context latency caused timeouts.

---

## Experiment 6 (Experiment C Folder) — Stock 32K Speed + Schema Helpers + Schema Void + Transfer
**Config:** `{"efficiency": True, "retry_guard": True, "shortcircuit": True,
"schema_helpers": True, "schema_void": True, "transfer": True}`
**Notebook:** `experimetn/experimetn c/sam-learning.ipynb`

**Kaggle Leaderboard Result: 1.06** (WINNING STABLE BASELINE 🏆)
**Local 4-Game Mean Score: 2.5101**
**Breakthrough:** Cleared **2 levels on `tn36`** (Score 10.04) thanks to preloaded `schema_helpers`!

---

## Experiment 7 (Experiment D Folder) — Context 45056 + Schema Helpers + Schema Void + Transfer
**Kaggle Leaderboard Result: 0.95** (regression due to 45K token overhead)

---

## Experiment 8 (Experiment F Folder) — Inline State Deduplication Graft (`state_dedup`)
**Config:** Exp C + inline `state_dedup` mixin
**Notebook:** `experimetn/experimetn f/sam-learning.ipynb`

**Kaggle Leaderboard Result: 0.77** (regression vs Exp C's 1.06)

**Root Cause of Regression:**
`state_dedup` checked `current_grid in visited_grids` and trimmed action batches whenever an action landed on a grid hash seen earlier in the level. In ARC video games (sokoban, navigation, switch toggling), returning to a past grid square is required for legitimate spatial backtracking. `state_dedup` false-positively chopped valid multi-step navigation plans into fragmented moves, preventing level completion on `tn36`.

---

## Action Plan: Revert to 1.06 Baseline Stack (Exp C)

Revert Cell 13 to the proven Exp C stack:
```python
install(bm, flags={
    "efficiency": True,
    "retry_guard": True,
    "shortcircuit": True,
    "schema_helpers": True,
    "schema_void": True,
    "transfer": True,
})
```
This restores the **1.06 score floor** (2.51 local mean) without false-positive action trimming!
