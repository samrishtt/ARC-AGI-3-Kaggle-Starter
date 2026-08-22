# Next session — queued 2026-08-22, for 2026-08-23

State at commit `40a1d75`.

## The number that governs everything

`arc3x/graded.py run_suite` over all 25 dev games: **MEAN 0.142**, 2 levels
completed, and **all 25 games burn the full 3000-action cap**. The Kaggle
submission scores **2.14**. So arc3x does not ship as the agent. It is a
measurement rig and a source of components for the notebook's Cell 8.

Only two games score at all, and they say what works:

| game | levels | score | actions to clear level 0 | human baseline |
|------|--------|-------|--------------------------|----------------|
| lp85 | 1/8    | 2.78  | 6                        | 17             |
| ls20 | 1/7    | 0.78  | 47                       | 22             |

lp85 is a *click* game with no avatar and it beat the human baseline by 2.8x.
That is the ratchet objective in `arc3x/progress.py` working.

## 1. Make ACT buttons plannable

`Mechanics.acts` (new) exposes buttons that change the board without moving the
avatar. `Mechanics.moves` filters `d != (0,0)` and threw all of them away.

    12 of 25 games have one
    cd82  1:ACT(1px) 2:ACT(1px) 3:ACT(200px) 4:ACT(200px) 5:ACT(13px)  planner-sees=[]
    tr87  1:ACT(13px) 2:ACT(14px) 3:ACT(28px) 4:ACT(29px)              planner-sees=[]

Two things to build, both general:

- **walk there, then press use** — after `navigate` arrives, try each ACT button
  and keep whichever moves the ratchet in `Progress`.
- **press-and-see** — for cd82/tr87 there is no avatar and no route, so the
  imagination has to search over ACT buttons directly in frame space.

## 2. Fix rotating-sprite matching

wa30 and sc25 each have four working MOVE buttons; the planner sees `[2, 4]`.

`Mechanics._fill_deltas` was added to promote single-observation deltas and it
did **not** fix these — there are *zero* votes for the missing axes. So does
`settle(min_votes=1)`, already called on the fallback path at `agent.py:484`.
The cause is upstream, in `percept.moved_objects`: a sprite that rotates as it
turns never matches as a rigid translation on the axis it is not facing. Match
by centroid or bbox displacement, not exact rigid shift. Same root cause as the
worst move accuracy on the set (wa30 43%, sc25 38%).

## 3. Hypothesis elimination

Sam's idea, which converges with `hyprune/` (literally "Hypothesis Pruning").
Factorise instead of enumerating: 6 maps x 8 goals x 10 block-sets x 5 collision
rules x 4 act-effects = 9,600 candidates stored as 33 facts. Refute per billed
action. Live-pool size then scales with how ambiguous the game still is, which is
Sam's "number of agents scales with complexity" — but as ~1 ms CPU hypotheses.

**Not LLM subagents.** LLM at 17.6 s/action x 110 games does not fit the 9 h
budget, and that is exactly what turned experiment 11's 2.68 local into 0.60 on
Kaggle (`arc3x/student.py:18-21`).

## 4. Stop burning the cap

Every one of the 25 games spends all 3000 actions. `play()` loops
imagine -> goal -> cover -> push_frontier -> click -> flail/reset forever with no
convergence test and no notion of "this level is not going to fall to this
repertoire".

## Constraints that still hold

- Build from 2.14, do not restart from scratch.
- No per-game hardcoding.
- Always report coverage out of 25 next to any mean — 23 zeros hide inside 0.142.
- Only graft into notebook Cell 8 if the local number beats 2.14.
