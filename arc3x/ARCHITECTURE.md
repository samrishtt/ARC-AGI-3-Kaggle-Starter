# ARC-AGI-3 solver — architecture and method

A working note for another researcher. Written 2026-08-23, at commit `0c75ffa`.
Current numbers are stated as measured, including the bad ones.

---

## 1. The problem, precisely

ARC-AGI-3 is a set of interactive grid games. The agent is shown a **64×64 array
of integers 0–15** and may press up to six buttons plus a click with an (x, y)
coordinate. There is:

- no reward signal,
- no instruction, no goal statement, no genre label,
- no reset-and-retry-cheaply: **one graded play per game**,
- and no access to the game's source at evaluation time.

The only feedback is a `levels_completed` counter and a state flag
(`NOT_FINISHED` / `GAME_OVER` / `WIN`). `level_reset` costs one action and clears
`GAME_OVER`, so death is cheap.

**Evaluation is on 110 private games nobody has seen.** Twenty-five games ship
with the engine as a development set. This asymmetry is the single most important
constraint on the design: any mechanism that works because of a property of these
25 games is worthless.

## 2. What the scoring formula demands

This deserves its own section because it inverts the obvious priorities. From the
competition scorer (replicated verbatim in `arc3x/graded.py`):

```
level i score  = min(115, (baseline_actions_i / actions_spent_i)² × 100)   if cleared
               = 0                                                         if not
game score     = min( Σ(score_i × i) / Σ i ,  (Σ cleared i / Σ all i) × 100 )
leaderboard    = mean over games
```

Level *i* carries weight *i*, and **every action spent while stuck on a level is
billed to that level and to no other**.

Two consequences:

**(a) At baseline speed, score equals the completion cap.** If each cleared level
scores 100, the weighted mean equals `(Σ cleared weights / Σ all weights) × 100`.
Evaluated over the actual level counts of the 25 dev games (9 games have 6
levels, 5 have 7, 6 have 8, 4 have 9, 1 has 10):

| levels cleared per game | mean score at baseline speed |
|-------------------------|------------------------------|
| level 0 only            | **3.52** — a hard ceiling    |
| levels 0 + 1            | **10.57**                    |
| levels 0 + 1 + 2        | **21.14**                    |

So a solver that reliably clears the first level of every game and never the
second cannot exceed 3.52, no matter how efficient it becomes. **Depth is the
only axis that matters.** This is not obvious from the leaderboard and it took us
a while to work out.

**(b) Efficiency is quadratic, so depth buys efficiency slack.** Two levels needs
≤1.1× baseline speed to reach 10.57; three levels reaches 10.8 even at 1.4×
baseline. Aiming one level deeper than strictly necessary is the cheaper route to
the same score.

**(c) Level 0 is nearly free to explore.** On a 9-level game the weights sum to
45, so level 0 caps at 2.2% of the achievable score. Hundreds of actions can be
spent there learning the game for almost nothing — but the human baselines for
level 0 are only 22–59 actions, so "free" means free in *score*, not in
information.

## 3. Design thesis

The system is built around one claim: **a person who has never seen the game
builds a small causal model of it within about twenty presses, and thereafter
plans inside that model rather than against the game.** So the solver should
learn a model, form a goal without being told one, and search in imagination
because imagination is free while actions are billed quadratically.

Five components, each with a single job.

### 3.1 `percept.py` — vision, pure numpy

Connected-component labelling by explicit-stack flood fill (no `scipy` in the
Kaggle image), rigid-shift matching between consecutive frames, and a
**volatility mask**: pixels that change on nearly every action regardless of what
was pressed. That mask is the HUD — score digits, timers, level indicators.

Isolating the HUD turned out to be load-bearing. An earlier version treated
frames as states for a Go-Explore-style archive; a one-pixel clock in the corner
made every frame unique, so the archive never merged two visits to the same
place and the search silently degenerated. Nothing crashed; the score was just 0.

### 3.2 `mind.py` — `Mechanics`, the learned model

Answers five questions from observation alone:

| question | mechanism |
|---|---|
| which colour am I? | see below |
| which button moves me where? | `votes[(action, colour, delta)]`, consensus |
| what blocks me? | a refused move blames the cells its footprint would have entered |
| what kills me? | on death, blame only *novel* ground |
| what looks like a goal? | on `levels_completed++`, credit what vanished / what we stepped on |

**Avatar identification by reversibility.** Many things move; only one is under
control. Candidates are scored on whether their observed displacement set contains
both `(dy,dx)` and `(-dy,-dx)` under different buttons. A real control scheme
contains opposite pairs; a coincidental shape-match across frames almost never
does. Secondary criteria: axis-alignment, consistent step magnitudes, number of
distinct buttons, total vote weight. This is cheap, needs no labels, and has been
the most reliable single prior in the system.

Whatever moves *identically* to the avatar under shared buttons is folded into a
`body` set — multi-colour sprites are one object. Getting this wrong makes every
collision test wrong, because the footprint used to sample "what am I standing on"
is read from the body mask.

Two known defects, both measured:

- `moved_objects` requires an exact rigid shift, so a sprite that **rotates as it
  turns** produces zero votes on the axis it is not facing. Two dev games have
  four working movement buttons and the model finds two.
- Buttons that change the board *without* moving the avatar were discarded as
  no-ops. Twelve of 25 games have such a button (a use / grab / select / rotate);
  two games have *nothing else* — five and four working buttons respectively, all
  invisible to the planner. Now exposed as `Mechanics.acts`; not yet used.

### 3.3 `progress.py` — `Progress`, an objective with no labels

The hardest part. `goal_colors` above only learns from a completed level, so it
needs a win to learn what a win looks like — useless on the first level of the
first game, which is the only situation that exists.

The rule, which we call the **ratchet**:

> Count every colour outside the HUD mask, every frame. A colour that **goes down
> and stays down** is being consumed — that is the objective. A colour that only
> goes up is being built.

Formally, a colour qualifies as consumed if it fell at least twice and fell at
least three times more often than it rose. The asymmetry is the whole content: a
count that oscillates is animation; a count that ratchets is progress.

One rule covers collecting pickups, clearing markers, painting a region, and
Sokoban — a crate parked on a target *occludes* the target pixel, so the target
colour ratchets down — with no genre-specific code.

Two guards, both added after the rule failed:

- **Flood rejection.** Any colour that ever covered more than 25% of the playing
  field is excluded. One game offered a board-sized background colour with a
  clean downward ratchet; chasing it replaces "no destination" with "an
  unreachable one".
- **Frozen mask.** The HUD mask is computed once and never revised. Recomputing
  it per frame looks more responsive and wipes the ledger, because the mask
  wiggles as the observation count grows. One game finished a 240-step run with a
  perfectly empty ratchet ledger despite having planned 41 times.

### 3.4 `dream.py` — `Dream`, the imagination

A forward model plus search. `predict(frame, action)` produces the imagined next
frame by translating the body mask; `observe()` grades its own prediction against
what actually happened before the model is updated, so accuracy is measured on
the prediction the agent actually acted on.

Search happens in two spaces, cheap first:

1. **Position space** — breadth-first over the learned deltas, ~8000 nodes,
   blocked by the learned obstacle set. Deep and fast, but only answers "can I
   get to that cell".
2. **Frame space** — actually simulates the next frame for each button and
   searches until `objective()` strictly drops. Answers "does this *do* anything",
   but ~600 nodes over four buttons is only about five moves of lookahead.

The abstention discipline matters: `observe()` can return `abstain` when the model
has no opinion, and abstentions are counted separately from errors. Without that
split, a game where most moves are blocked scores a free 100% by predicting
"nothing happened".

### 3.5 `agent.py` — the policy

```
WIGGLE     press every button ~4×; ~25 actions, billed to level 0 where it is nearly free
then loop:
  IMAGINE      plan toward a falling ratchet colour; spend ONE action; re-plan
  GOAL         walk to any colour that has previously ended a level
  COVER        stand on every reachable square (frontier-directed, O(n) not O(n²))
  FRONTIER     walk into each *kind* of thing that has refused us, once
  CLICK        rank blob centres by rarity, click until something changes
  FLAIL        random over non-dead actions; alternate with level_reset
on level up:  keep the entire model; reset only the frame-to-frame counters
```

Two deliberate choices worth calling out:

**Re-plan after every single action, never execute a route.** The model starts out
believing nothing blocks it, so a route planned once walks into the first wall and
wastes the remainder. Planning is free; one billed action is the cheapest possible
test of a hypothesis. Each refusal teaches an obstacle, so the route repairs
itself as it is walked, and a level gets mapped in O(path) billed actions rather
than O(area).

**`cover` is deliberately dumber than the targeting heuristic.** Most levels
complete when the avatar touches the right thing, and enumerating everything
walkable is the only method that does not require guessing which thing that is.

## 4. Method — how claims get established

This is the part we would most want critiqued.

**An honest offline replica.** `arc3x/graded.py` reproduces the competition
scorer verbatim and deliberately withholds three powers the local engine offers
but the graded gateway does not:

1. `copy.deepcopy(game)` — free snapshots, free rewind. Behind the gateway there
   is no game object, only HTTP.
2. `game._get_valid_actions()` — its own docstring says it is never exposed to
   users. It returns the *state-dependent* legal set including, for the click
   action, the concrete legal coordinates. An agent reading it is being told
   which of 4,096 clicks are live.
3. `baseline_actions` — present locally, stripped from the API. Used to *score*
   the run, never shown to the agent.

Every earlier local measurement in this project was inflated by one of these.

**Coverage travels with the mean, always.** A mean of 0.142 over 25 games is 23
zeros and one game working. Reporting the mean alone made that look like a small
positive result for weeks.

**A "why did it stop" channel.** `agent_fn` returns nothing, so the score was the
only output, and a score cannot distinguish *"never had a destination"* from
*"had one and could not reach it"* — which need opposite fixes. Every strategy
now records its exit reason against the level it started on.

**A fixed train/holdout split.** 17 games to iterate on, 8 held clean, chosen
every-third alphabetically and **never re-drawn** — a holdout re-rolled after a
disappointing result is a lottery, not a control. `--split both` prints the ratio.
A mechanism that helps tune and not hold has been fitted to games it was allowed
to see, and the 110 private games will behave like hold.

**Wall-clock cutoffs are treated as invalidating, not degrading.** Kaggle bills
actions, not seconds. A run cut short by a deadline is a machine-dependent
experiment and is not comparable to another run; the summary says so loudly
rather than averaging it in. This caught a real error: naive parallelisation had
each worker size a BLAS thread pool to all 12 cores, and one game managed 32
actions where serial did 401.

**Negative results are kept.** Two examples:
- We hypothesised that the click games are reference-panel + canvas, and tested
  half-board agreement. 11/11 click games landed in the predicted band — which
  looked like a clean win until the control gave **25/25**, including three
  avatar games at >90%. The boards are mostly empty and empty agrees with empty;
  the test measured emptiness. Discarded.
- Graft-style tuning of the LLM notebook is **0 for 11** experiments. The only
  change that ever moved the score was swapping the model.

## 5. Current results, as measured

| system | mean over 25 dev games | level-clears |
|---|---|---|
| LLM notebook (the submitted one) | **2.14** (Kaggle leaderboard) | not yet instrumented |
| `arc3x` standalone | **0.142** | 2 across 25 games |

`arc3x` is 15× worse than the notebook and does not ship as the solver; it is a
measurement rig and a source of components. Its one clean success: a game with no
avatar at all clears level 0 in **6 actions against a human baseline of 17**,
driven purely by the ratchet objective.

Two results we consider established, both negative and both useful:

**Prediction accuracy is not the bottleneck.** Two games whose forward model
predicts at 100% complete zero levels. A third predicts at 80%, is the only one
with a known goal colour, and is the only one that plans and completes a level.
Accuracy without an objective buys nothing.

**Compute is not the bottleneck.** `arc3x` contains no neural network and no LLM
at all — it is numpy on CPU. And an LLM cannot afford to play: at ~17.6 s/action
over ~300 actions × 110 games it needs 161 hours against a ~9 hour limit. A prior
experiment scored 2.68 locally and **0.60** on Kaggle purely because vLLM prefill
timed out on a shared GPU.

## 6. What is open

1. **Depth.** Almost nothing clears level 0, so the level-to-level model transfer
   that the whole design is built around has never actually been exercised. This
   is where the score is won: 3.52 versus 21.14.
2. **The click branch has no objective.** It returns success on *any* pixel
   change, so on one game the agent clicks 259 times, something changes every
   single time, and it never asks whether the change was progress. This accounts
   for the largest single share of wasted actions in the current run.
3. **Non-movement buttons.** Twelve of 25 games have one and none are planned
   with.
4. **Shape-changing sprites** break rigid-shift matching.
5. **Hypothesis elimination.** The intended next mechanism: hold a factorised
   space of candidate game-models (maps × goals × obstacle-sets × collision rules
   × button-effects) as ~33 independent facts rather than 9,600 enumerated
   products, and refute per billed action. The number of surviving candidates then
   scales with how ambiguous the game still is. Deliberately CPU-side at ~1 ms
   per candidate, for the timeout reason above.
6. **Convergence.** All 25 games currently spend their entire action budget. There
   is no test for "this repertoire cannot clear this level".

## 7. Things we would do differently from the start

- Replicate the scorer **first**. The 3.52 ceiling reframes everything, and we
  computed it late.
- Report coverage with every mean from day one.
- Build the "why did it stop" channel before the first optimisation, not after
  several.
- Treat any per-frame recomputation of a learned mask as suspect. Two separate
  silent failures traced to a mask that was allowed to move.
