"""The acting layer: what to press next, and why the schedule looks like this.

WHY THIS EXISTS
---------------
``mindgraft.py`` ends on a deliberate refusal - *"the layer that lets this act on
the real board is written only after the numbers below say it should be."* The
numbers came back from Kaggle, and they say something sharper than "act". They
say **where** to act cheaply and **where** to act carefully, and they say it
loudly enough that the schedule below is arithmetic rather than taste.

THE ONE EQUATION
----------------
From ``score_from_card`` (:mod:`arc3x.graded`, lines 115-154), copied verbatim
from the competition scorer::

    score        = Σ(level_score_i × (i+1)) / Σ over ALL levels (i+1)
    level_score_i = min(115, (baseline_i / actions_charged_i)² × 100)   if cleared
                  = 0                                                  if not

Verified exactly against real submission rows in ``v15 result/level_probe.jsonl``:
tn36 cleared 2 of 7 levels spending [9, 217, 81] against baselines [32, 72, 26],
giving ``(115×1 + 11.0×2)/28 = 4.8935`` - which is the logged
``4.8934957512066815`` to every digit.

Three consequences, and they are the whole design:

1. **Actions on a level you never clear cost nothing.** tn36 spent 81 actions on
   level 2 and was billed as if it had spent none, because it did not finish.
   So an unfinished level is a *free laboratory*, and the only question is
   whether you are standing in one.

2. **Level 0 is nearly free to explore.** Its weight is 1 out of Σ(i+1), which is
   21 for a six-level game and 55 for a ten-level one. Level 0's entire
   contribution is capped at ``115/W`` ≈ 2.1-5.5 points. Level 1 is worth twice
   that, level 2 three times. Trading level 0's points for a model that clears
   level 1 is a trade at 2:1 or better, every time.

3. **On a level you do clear, only route length matters, quadratically.** m0r0
   *cleared* level 0 and scored **0.058**, because it took 271 actions where the
   baseline is 30. At 60 actions the same clear scores 1.19 - twenty times more
   for identical behaviour. Efficiency is not a polish pass; it is most of the
   score.

So this file has exactly two modes, and the phase order flips between them:

* **laboratory** - probe, push at everything, learn. Long batches. Level 0
  starts here, and any level the model has run out of ideas on becomes one,
  because a level you are about to fail is billed at zero either way.
* **examination** - shortest model-verified route or nothing. Short batches.
  Every level after 0 starts here, and hands back to the LLM rather than guess,
  because a wrong 30-action guess on level 1 costs more than the whole of level 0.

WHY BATCHING IS THE UNLOCK
--------------------------
``solver.py`` bills **one LLM round-trip per action** - ``turns == actions``,
confirmed by tn36's probe row (turns=307, actions_per_level summing to
9+217+81=307). At the measured ~26 s per turn against a 7920 s wall clock, the
agent can afford ~300 actions per game and nothing more; it is action-starved,
not intelligence-starved.

But ``step_env`` accepts ``actions`` (plural) and runs the whole list in one
call (``solver.py:683-776``), breaking on ``level_completed`` / ``game_over`` /
``run_complete`` / an invalid action - and **not** on an unchanged board. And
``_execute_action`` appends a ``HistoryEntry`` and rewrites runtime state after
*every* action inside the batch (``solver.py:782-872``). So a 40-action probe
costs **one** turn and yields 40 labelled transitions. The mind's training corpus
is free; only the actions are billed, and on a laboratory level they are not
billed either.

Nothing here consults an engine, clones a game, or calls a model. The pilot is
handed frames and hands back button presses, which is exactly the power it will
have against a remote gateway on a game nobody has played.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from arc3x.mindgraft import (
    AID_LABEL,
    CLICK_AID,
    DISPLAY_TO_AID,
    Mind,
    Press,
    _grid,
    is_reset,
    parse_press,
    resolve_name,
)

# Framework levels are 1-based: `runtime_state.frame_from_payload` clamps with
# `max(1, ...)`, so `frame.level == 1` is the level the scorer calls index 0.
# Getting this off by one would invert the laboratory rule, which is the single
# most expensive mistake available in this file.
FIRST_LEVEL = 1


# -- 1. when two situations are the same -------------------------------------

#: ``distinct_values_per_pixel`` is the one statistic that needs a set, and a set
#: per pixel would be a Python loop over 4,096 columns every turn. ARC colours are
#: 0-15, so "which values has this pixel taken" fits in a 16-bit mask and the
#: count is a table lookup - exact, and O(1) per frame instead of O(history).
_POPCOUNT16 = (
    np.unpackbits(np.arange(1 << 16, dtype=">u2").view(np.uint8).reshape(-1, 2), axis=1)
    .sum(axis=1)
    .astype(np.uint8)
)


@dataclass
class CellSense:
    """A frame -> cell-id function, calibrated from real play instead of clones.

    :mod:`arc3x.cell` established the rule and measured it: hashing the raw 64x64
    frame is effectively bijective (tn36: 60 steps, 60 distinct keys), because a
    single HUD bar draining by 6 per action makes every frame globally unique.
    Novelty then stops being a signal and Go-Explore collapses to a random walk.
    The fix is to keep only pixels that **vary and are not monotone in time**.

    ``cell.calibrate`` learns that mask by deep-copying the engine and taking
    random walks. On Kaggle there is no engine to copy - so this learns the same
    mask from the frames the agent actually walked through, which the framework
    hands over for free in ``history``. The probe batch on level 0 is what pays
    for it, and level 0 is the one place where paying is nearly free.

    TWO THINGS THIS GETS RIGHT THAT A BATCH REFIT DOES NOT
    -----------------------------------------------------
    **It is incremental**, because it has to be. The framework hands over the
    whole history every turn, so a refit over the prefix is O(history) *per turn*
    - a sort of a (3000, 64, 64) stack, tens of megabytes allocated, repeatedly.
    Every one of those milliseconds is bought with wall clock on a run that was
    cut at 7920 s mid-play, and a silent CPU tax on the critical path is exactly
    what turned experiment 11's 2.68 local into 0.60 on Kaggle. All four
    statistics below are running aggregates instead: min/max for *varies*,
    two monotone flags, and the colour bitmask above. Same answer, O(new frames).

    **It breaks the monotone test at level boundaries.** A HUD clock drains
    monotonically *within* a level and then jumps back up when the next level
    starts. Measured over a whole multi-level history it is therefore not
    monotone at all, so the naive test clears it as informative and re-admits the
    single pixel the entire abstraction exists to remove - silently, and only on
    the games that get deep enough to matter. A level change is a new walk, which
    is precisely what ``cell.calibrate`` used several of; the step across the
    boundary is not evidence about anything and is skipped.
    """

    mask: np.ndarray | None = None
    n_varying: int = 0
    n_clock: int = 0
    #: How many history entries have been folded in. The caller's cursor.
    fitted: int = 0

    #: A pixel that took only two values may be monotone by accident - it changed
    #: once. Requiring three distinct values is what stops a short online window
    #: from mistaking state for a clock, which ``cell.calibrate`` instead
    #: prevented by demanding monotonicity across several independent walks.
    min_distinct: int = 3
    #: Below this many frames the statistics are noise and no mask is published,
    #: so ``key`` hashes the raw frame and novelty is merely over-sensitive.
    min_frames: int = 12

    # -- running aggregates, all shaped like the frame ------------------------
    _shape: tuple[int, ...] | None = None
    _lo: np.ndarray | None = None
    _hi: np.ndarray | None = None
    _bits: np.ndarray | None = None
    _nondec: np.ndarray | None = None
    _noninc: np.ndarray | None = None
    _prev: np.ndarray | None = None
    _prev_level: int = -1
    _n: int = 0

    def note(self, frame: np.ndarray | None, level: int = -1) -> None:
        """Fold one frame into the aggregates. Constant time per frame.

        A shape change restarts the statistics: comparing a 64x64 board against a
        differently sized one pixelwise is meaningless, and the board that is
        being played now is the one the mask has to describe.
        """
        if frame is None or getattr(frame, "ndim", 0) != 2 or not frame.size:
            return
        if self._shape != frame.shape:
            self._start(frame)
            self._prev_level = level
            return

        cur = frame
        np.minimum(self._lo, cur, out=self._lo)
        np.maximum(self._hi, cur, out=self._hi)
        self._bits |= (np.uint16(1) << np.clip(cur, 0, 15).astype(np.uint16))
        # A level change is a new walk: the jump across it says nothing about
        # whether a pixel is a clock, and counting it would clear every clock.
        if self._prev is not None and (level < 0 or level == self._prev_level):
            self._nondec &= cur >= self._prev
            self._noninc &= cur <= self._prev
        self._prev = cur.copy()
        self._prev_level = level
        self._n += 1

    def _start(self, frame: np.ndarray) -> None:
        self._shape = frame.shape
        self._lo = frame.astype(np.int16)
        self._hi = frame.astype(np.int16)
        self._bits = (np.uint16(1) << np.clip(frame, 0, 15).astype(np.uint16))
        self._nondec = np.ones(frame.shape, dtype=bool)
        self._noninc = np.ones(frame.shape, dtype=bool)
        self._prev = frame.copy()
        self._n = 1
        self.mask = None

    def fit(self, frames: Sequence[np.ndarray | None], levels: Sequence[int] | None = None) -> bool:
        """Fold everything past the cursor, then republish the mask.

        Takes the caller's whole list and reads only the tail, so the call site
        stays "hand over the history" while the cost stays proportional to what
        is new.
        """
        for i in range(max(0, self.fitted), len(frames)):
            self.note(frames[i], int(levels[i]) if levels is not None and i < len(levels) else -1)
        self.fitted = len(frames)
        return self.settle()

    def absorb(self, history: Sequence[Any]) -> bool:
        """Fold new framework history entries. The cheap path the pilot uses.

        Distinct from :meth:`fit` only in reading ``entry.frame`` itself, which
        keeps the pilot from materialising a list of every grid it has ever seen
        on every turn - the cost the incremental aggregates exist to avoid.
        """
        for i in range(max(0, self.fitted), len(history)):
            entry = history[i]
            frame = getattr(entry, "frame", None)
            self.note(_grid(entry), int(getattr(frame, "level", -1) or -1))
        if self.fitted == len(history):
            return self.mask is not None
        self.fitted = len(history)
        return self.settle()

    def settle(self) -> bool:
        """Recompute the published mask from the aggregates. Returns: is there one?"""
        if self._shape is None or self._n < self.min_frames:
            return self.mask is not None

        varies = self._hi != self._lo
        distinct = _POPCOUNT16[self._bits]
        clock = varies & (self._nondec | self._noninc) & (distinct >= self.min_distinct)

        mask = varies & ~clock
        if not mask.any():  # every varying pixel looked like a clock
            mask = varies.copy()
        if not mask.any():  # nothing has moved at all yet
            mask = np.ones(self._shape, dtype=bool)

        self.mask = mask
        self.n_varying = int(varies.sum())
        self.n_clock = int(clock.sum())
        return True

    def key(self, frame: np.ndarray, level: int) -> bytes:
        """Hash the informative pixels plus the level. ~2 microseconds."""
        if self.mask is not None and self.mask.shape == frame.shape:
            payload = frame[self.mask].tobytes()
        else:
            payload = frame.tobytes()
        return hashlib.blake2b(
            payload + bytes((level & 0xFF,)), digest_size=16
        ).digest()


# -- 2. what a decision looks like -------------------------------------------


@dataclass
class Plan:
    """A batch of presses, plus why - so a bad run is readable afterwards."""

    presses: list[Press]
    phase: str
    why: str = ""

    def __bool__(self) -> bool:
        return bool(self.presses)

    def __len__(self) -> int:
        return len(self.presses)

    def payloads(self, valid: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """The exact ``step_env({"actions": [...]})`` argument list."""
        return [p.arguments(valid) for p in self.presses]

    def __repr__(self) -> str:
        return f"<{self.phase} x{len(self.presses)} {self.why}>"


# -- 3. the pilot -------------------------------------------------------------


@dataclass
class Pilot:
    """Chooses batches of presses from a learned model and a level's economics.

    One object per game. ``observe`` folds the framework's history into the mind;
    ``decide`` returns a :class:`Plan` or ``None``, where ``None`` means *hand
    this turn to the language model* - the one move this file will not fake.
    """

    # -- knowledge that outlives a level change. This is the transfer the whole
    # design rests on: level 0 identifies the goal and the buttons, levels 1..n
    # are where knowing them is worth 2..n times as much. `Mechanics` is kept
    # wholesale on purpose (see agent.on_new_level, which had the same intent and
    # was never exercised because almost nothing cleared level 0).
    mind: Mind = field(default_factory=Mind)
    sense: CellSense = field(default_factory=CellSense)
    #: Colours the goal is drawn in, if something that can read frame 0 says so.
    #: Left empty here: this is the socket the LLM goal-oracle plugs into, and
    #: an empty set costs only the fallbacks below, never a wrong target.
    goal_hint: set[int] = field(default_factory=set)

    # -- dials, all in ACTIONS, all justified by the equation at the top -------
    #: Presses per button in a grounding probe. `Mechanics.settle` needs 2 votes
    #: per (button, colour, delta) and refusals produce no delta vote, so 3 is
    #: the floor and 8 is the honest allowance for a board with walls.
    probe_rounds: int = 8
    #: Ceiling on total spend while a level is a laboratory. 160, not 400: at
    #: 2.14 the notebook is a level-0 machine, so if depth does not arrive, level
    #: 0's points are all there is - and 160 actions against a baseline of 32
    #: still leaves 4% of level 0 rather than 0%.
    lab_actions: int = 160
    #: Allowance once a level has been *conceded* - the model tried and repeated
    #: itself, so this level will not be cleared and its actions are genuinely
    #: free. The only thing left to protect is the wall clock, which is the real
    #: bound anyway: every game in both submissions was cut at 7920 s, never at an
    #: action cap (``max_actions_per_game`` is ``None`` in the 2.14 notebook).
    conceded_actions: int = 2000
    #: No single batch may exceed this, so one confident wrong plan cannot eat a
    #: level. Routes are shortest-path and rarely approach it.
    max_batch: int = 64
    #: Clicks per click batch, and how many representatives per candidate colour.
    click_batch: int = 12
    clicks_per_color: int = 3
    #: Consecutive no-novelty batches before an examination level is conceded to
    #: be a laboratory. Two, because one repeat is a wall and two is a dead end.
    patience: int = 2

    # -- per-level state, rolled by `_roll` -----------------------------------
    level: int = FIRST_LEVEL
    spent: int = 0
    tried: set[int] = field(default_factory=set)
    clicked: set[tuple[int, int]] = field(default_factory=set)
    live_clicks: set[tuple[int, int]] = field(default_factory=set)
    acted: set[int] = field(default_factory=set)
    keys: set[bytes] = field(default_factory=set)
    stalls: int = 0
    conceded: bool = False
    #: Cursor into ``history`` for click attribution. Separate from
    #: ``Mind.seen``, which counts transitions rather than entries.
    _hist_seen: int = 0

    # -- accounting -----------------------------------------------------------
    batches: int = 0
    handoffs: int = 0
    log: list[str] = field(default_factory=list)

    # -- learning -------------------------------------------------------------

    def observe(self, history: Sequence[Any]) -> int:
        """Fold new history into the mind, the cell key, and the click memory.

        Cheap by construction, and it has to be: this runs between the agent and
        its next action on a run that is bound by a 7920 s clock. All three folds
        are incremental against their own cursor, so handing over the whole
        3,000-entry history every turn costs only what is new.
        """
        fresh = self.mind.absorb(history)
        self.sense.absorb(history)

        # Which clicks did anything? A click that changed the board is a live
        # cell, and per the click survey the special cells cluster - a median of
        # two clusters per game - so a live cell makes its neighbours worth more
        # than any unexplored colour does.
        #
        # Cursor is on *entries*, not transitions: RESET and unknown labels are
        # skipped by `transitions()`, so the two lengths differ and a cursor on
        # the transition count would re- or under-scan. Attributed straight from
        # the entry pair, which also avoids `transitions()` dropping a click whose
        # grid shape changed and crediting the wrong cell for the change.
        for i in range(max(1, self._hist_seen), max(self._hist_seen, len(history))):
            press = parse_press(getattr(history[i], "action", None))
            if press is None or not press.is_click or press.row < 0:
                continue
            self.clicked.add((press.row, press.col))
            before, after = _grid(history[i - 1]), _grid(history[i])
            if before is None or after is None or before.shape != after.shape:
                continue
            if bool((before != after).any()):
                self.live_clicks.add((press.row, press.col))
        self._hist_seen = len(history)
        return fresh

    # -- deciding -------------------------------------------------------------

    def decide(
        self,
        frame: np.ndarray,
        valid: Sequence[str] | None,
        level: int,
        *,
        spent_on_level: int | None = None,
    ) -> Plan | None:
        """A batch to execute, or ``None`` to let the language model take the turn.

        The phase order is the whole policy, and it differs by mode because the
        scorer bills the two modes differently:

        *laboratory* - ground the model, then walk into every kind of thing once,
        then try the use-buttons, then click. Long batches, because the actions
        are free.

        *examination* - a shortest route to a believed goal, verified step by step
        in imagination, or nothing. Anything less certain is handed back, because
        on level 1 a wrong 30-action guess costs more than all of level 0.
        """
        if frame is None or getattr(frame, "ndim", 0) != 2 or not frame.size:
            return self._hand("no readable frame")
        self._roll(level, spent_on_level)

        aids = self._aids(valid)
        if not aids:
            return self._hand("no usable buttons in valid_actions")

        # Follow the sprite before planning anything. `where` is the only thing
        # that advances `Mechanics.pos`, and `pos` is the hint every `locate` in
        # every phase below depends on; without it `locate` ranks candidates by
        # size rather than proximity, which on a single-colour sprite locks onto
        # the largest same-coloured clump on the board and mistracks from there on
        # (measured in mindgraft.backtest: individual games swing 50 points).
        self.mind.mech.where(frame)

        # Novelty, on the calibrated key rather than the raw frame - the whole
        # point of CellSense. A repeat is how a wall announces itself.
        key = self.sense.key(frame, level)
        self.stalls = self.stalls + 1 if key in self.keys else 0
        self.keys.add(key)
        if self.stalls >= self.patience and not self._lab:
            # Conceding is not giving up: it reclassifies the level as one that
            # will not be cleared, which makes its remaining actions free and
            # unlocks the long batches that might change that.
            self.conceded = True
            self.log.append(f"L{level}: conceded to laboratory after {self.stalls} repeats")

        order = (
            (self._ground, self._execute, self._frontier, self._use, self._click)
            if self._lab
            else (self._execute, self._frontier, self._use, self._click, self._ground)
        )
        for phase in order:
            plan = phase(frame, aids, valid)
            if plan:
                self.batches += 1
                self.spent += len(plan)
                self.log.append(f"L{level} +{len(plan)}a {plan!r}")
                return plan
        return self._hand("model has no move it can stand behind")

    # -- phases ---------------------------------------------------------------

    def _ground(
        self, frame: np.ndarray, aids: list[int], valid: Sequence[str] | None
    ) -> Plan | None:
        """Press every button, several times, so the codec can be learned at all.

        Round-robin rather than random: UP DOWN LEFT RIGHT nets zero displacement
        while producing four clean votes, which is exactly what
        ``Mechanics.settle`` wants and what a random walk gives only by luck.
        Covering the board is the frontier phase's job, not this one's.

        Grounding stops when the model is good enough to route **or** when every
        button has had its rounds and still produced no avatar. The second exit is
        not a formality: cd82 and tr87 have no MOVE buttons at all, so a gate that
        waited for an avatar would loop here forever and those games would never
        reach the use or click phases - which are the only phases they have.
        """
        if self._grounded or not self._room:
            return None
        moves = [a for a in aids if a != CLICK_AID]
        if not moves:
            return None
        tries = self.mind.mech.tries
        owed = [a for a in moves if tries.get(a, 0) < self.probe_rounds]
        if not owed:
            return None
        seq = [a for _ in range(self.probe_rounds) for a in owed]
        return Plan(
            [self._press(a, valid) for a in seq[: self._cap]],
            "ground",
            f"{len(owed)} buttons x{self.probe_rounds}, avatar={self.mind.mech.avatar}",
        )

    def _execute(
        self, frame: np.ndarray, aids: list[int], valid: Sequence[str] | None
    ) -> Plan | None:
        """Shortest route to a believed goal colour. The only phase that scores."""
        targets = self._cells(frame, self._goal_colors(frame))
        if not targets:
            return None
        route = self._verified(frame, self.mind.route(frame, targets))
        if not route:
            return None
        return Plan([self._press(a, valid) for a in route], "execute", f"->goal {len(route)}a")

    def _frontier(
        self, frame: np.ndarray, aids: list[int], valid: Sequence[str] | None
    ) -> Plan | None:
        """Walk into one kind of thing never entered before, and find out what it is.

        ``Mechanics.frontier_colors`` is the honest version of a goal detector: it
        names colours evidence has never let the sprite occupy.

        A person confronted with a locked-looking thing tries it once. Some are
        doors, some are goals, some are walls, and from the outside the only way to
        tell is to try each once. Contrast ``markers.py``, which ranks candidates by
        pixel count and therefore proposed floor tiling on the one game with
        ground truth - abundance is a property of scenery, not of goals.

        WALK ADJACENT, THEN STEP IN - NOT "ROUTE TO THE BLOCKED CELL"
        ------------------------------------------------------------
        ``frontier_colors`` returns, for each colour, the sprite positions from
        which one learned move enters it. Using those instead of asking ``plan``
        for a route to the colour's own pixels fixes two things. ``plan`` requires
        the sprite's **whole footprint** to fit on the target
        (``mind.py:745-747``), so a 3x3 sprite can never "arrive" on a one-pixel
        frontier cell and every such colour would look unreachable forever. And
        one ``reachable`` call serves every candidate colour, where a ``plan`` per
        colour was a fresh breadth-first search per colour, every turn.

        Not gated on ``_room``. A frontier push is a shortest route to a specific
        untouched thing, not a probe: it is the same kind of spend as ``_execute``
        and the only reason it is not ``_execute`` is that the target is a guess.
        Cutting it off at the laboratory allowance would leave an examination level
        with nothing between "I know the goal" and "ask the language model".
        """
        mech = self.mind.mech
        pending = {
            c: stands
            for c, stands in mech.frontier_colors(frame).items()
            if c not in self.tried and stands
        }
        if not pending:
            return None
        box = mech.locate(frame, hint=mech.pos)
        if box is None:
            return None
        routes = mech.reachable(frame)
        if not routes:
            return None
        _t, _l, h, w = box
        H, W = frame.shape

        # Rarest first, for the same reason the click policy prefers rare colours:
        # the board's bulk is floor and wall, and the interesting thing is scarce.
        counts = self._counts(frame)
        for color in sorted(pending, key=lambda c: counts.get(c, 0)):
            best: list[int] | None = None
            for stand in pending[color]:
                walk = routes.get(stand)
                if walk is None or (best is not None and len(walk) + 1 >= len(best)):
                    continue
                step = self._entering(frame, stand, color, h, w, H, W)
                if step is not None:
                    best = walk + [step]
            if not best:
                continue
            route = self._verified(frame, best)
            # A route may be cut by `_cap` without being wrong - it is a shortest
            # path over proven ground, so its prefix is progress and the next turn
            # continues it. Only a *divergence* disqualifies the plan.
            if len(route) < min(len(best), self._cap):
                continue
            if len(route) == len(best):
                # The colour is only "tried" once we have actually stepped into it.
                self.tried.add(color)
            return Plan(
                [self._press(a, valid) for a in route],
                "frontier",
                f"->colour {color} {len(route)}/{len(best)}a",
            )
        return None

    def _entering(
        self, frame: np.ndarray, stand: tuple[int, int], color: int, h: int, w: int, H: int, W: int
    ) -> int | None:
        """The learned move that steps the sprite from ``stand`` into ``color``."""
        t, l = stand
        for aid, (dy, dx) in self.mind.mech.moves.items():
            nt, nl = t + dy, l + dx
            if nt < 0 or nl < 0 or nt + h > H or nl + w > W:
                continue
            if bool((frame[nt : nt + h, nl : nl + w] == color).any()):
                return aid
        return None

    def _use(
        self, frame: np.ndarray, aids: list[int], valid: Sequence[str] | None
    ) -> Plan | None:
        """Press a use-button where we stand.

        Twelve of the 25 dev games have a button that changes the board without
        moving the sprite, and two have nothing else at all. ``Mechanics.moves``
        drops them on purpose - a route cannot be made of them - so a planner that
        only routes discards those games entirely. One press at a time, so the
        transition stays attributable.
        """
        untried = [a for a in aids if a != CLICK_AID and a not in self.acted]
        known = [a for a in untried if a in self.mind.mech.acts]
        pick = known or (untried if self._lab else [])
        if not pick:
            return None
        self.acted.add(pick[0])
        return Plan([self._press(pick[0], valid)], "use", f"button {pick[0]} where we stand")

    def _click(
        self, frame: np.ndarray, aids: list[int], valid: Sequence[str] | None
    ) -> Plan | None:
        """Click the cells most likely to be special, cheapest signal first.

        Measured over all 25 games: every one of the 4,096 cells collapses to
        between 1 and 11 distinct outcomes with ~96% of clicks landing on the
        modal one, and the cells that do something sit on **minority colours at
        7.83x chance**, in a median of two clusters. So: neighbours of a cell
        already known to be live, then representatives of the rarest colours,
        spread out so three picks are not three pixels of one blob.
        """
        if CLICK_AID not in aids:
            return None
        if not self._lab and not self.live_clicks:
            return None

        picks: list[tuple[int, int]] = []
        H, W = frame.shape
        for (y, x) in sorted(self.live_clicks):
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                cell = (y + dy, x + dx)
                if 0 <= cell[0] < H and 0 <= cell[1] < W and cell not in self.clicked:
                    picks.append(cell)

        counts = self._counts(frame)
        bg = self.mind.mech.background
        for color in sorted(counts, key=lambda c: counts[c]):
            if len(picks) >= self.click_batch:
                break
            if color == bg or counts[color] >= frame.size // 4:
                continue
            ys, xs = np.nonzero(frame == color)
            cand = [
                (int(y), int(x)) for y, x in zip(ys, xs) if (int(y), int(x)) not in self.clicked
            ]
            picks.extend(self._spread(cand, self.clicks_per_color))

        seen: set[tuple[int, int]] = set()
        out: list[Press] = []
        for cell in picks:
            if cell in seen or cell in self.clicked:
                continue
            seen.add(cell)
            self.clicked.add(cell)
            out.append(self._press(CLICK_AID, valid, row=cell[0], col=cell[1]))
            if len(out) >= min(self.click_batch, self._cap):
                break
        if not out:
            return None
        return Plan(out, "click", f"{len(out)} cells, {len(self.live_clicks)} live")

    # -- the imagination gate -------------------------------------------------

    def _verified(self, frame: np.ndarray, route: Sequence[int]) -> list[int]:
        """Roll a route forward in imagination; keep the prefix the model backs.

        This is what makes a long batch safe. ``step_env`` breaks a batch on a
        level completion, a game over, or an invalid action - but **not** on an
        unchanged board, so a route that walks into a wall halfway spends every
        remaining action for nothing. Predicting each step first turns that from
        a billed mistake into a free one.

        The final step is exempt, and that is not a loophole. ``Mechanics.plan``
        routes over proven ground but is allowed to *end* on an unproven cell,
        because reaching the untouched thing is the entire point; the model will
        of course call that step blocked, since not having stood somewhere is
        exactly what makes it worth an action.
        """
        route = list(route)[: self._cap]
        if len(route) <= 1:
            return route
        grid = frame
        kept: list[int] = []
        saved = self.mind.mech.pos
        try:
            for aid in route[:-1]:
                pred = self.mind.predict(grid, aid)
                if not pred.spoke or not pred.moved:
                    break
                kept.append(aid)
                grid = pred.grid
                self.mind.mech.pos = pred.to
        finally:
            # The sprite has not actually moved; only imagination advanced. Leaving
            # `pos` at the imagined destination would hand the next real `locate`
            # a hint pointing somewhere the sprite has never been.
            self.mind.mech.pos = saved
        if len(kept) < len(route) - 1:
            # A truncated route no longer reaches the target, so its tail step is
            # not the experiment it was chosen to be - walk the proven part only.
            return kept
        return kept + [route[-1]]

    # -- targets --------------------------------------------------------------

    def _goal_colors(self, frame: np.ndarray) -> set[int]:
        """Colours worth walking onto, best evidence first.

        The ordering matters more than the contents. ``Mechanics.goal_colors`` is
        populated only when ``levels_completed`` goes up, so before the first win
        it is empty - which is the same reason ``Dream``'s target set sits empty on
        every game it has ever been measured on. ``vanished`` is the signal that
        exists *before* a win: a thing that disappears when touched, which on a
        cover-predicate game ("every A on a B" - 10 of the 13 readable win
        conditions) is the goal itself.

        ``vanished`` is also the weakest of the three, and it is deliberately
        **not** trusted on an examination level. Anything that blinks, any trail
        the sprite erases behind itself, any HUD element that clears will land in
        it, and a level-1 route to a blinking pixel costs more than the whole of
        level 0. On a laboratory level the same guess is free, so it is allowed
        there and nowhere else.
        """
        mech = self.mind.mech
        blocked = mech.blocked_set
        sources = [self.goal_hint, {c for c, n in mech.goal_colors.items() if n > 0}]
        if self._lab:
            sources.append({c for c, n in mech.vanished.items() if n > 0})
        for source in sources:
            picked = {int(c) for c in source if int(c) != mech.background} - blocked
            if picked:
                return picked
        return set()

    def _cells(self, frame: np.ndarray, colors: set[int]) -> list[tuple[int, int]]:
        if not colors:
            return []
        want = np.isin(frame, list(colors))
        ys, xs = np.nonzero(want)
        return [(int(y), int(x)) for y, x in zip(ys, xs)]

    @staticmethod
    def _spread(cells: Sequence[tuple[int, int]], k: int) -> list[tuple[int, int]]:
        """Up to ``k`` cells, greedily far apart - one per blob, not k per pixel."""
        if not cells:
            return []
        out = [cells[0]]
        while len(out) < k and len(out) < len(cells):
            best, far = None, -1
            for cell in cells:
                if cell in out:
                    continue
                d = min(max(abs(cell[0] - o[0]), abs(cell[1] - o[1])) for o in out)
                if d > far:
                    best, far = cell, d
            if best is None or far <= 1:
                break
            out.append(best)
        return out

    @staticmethod
    def _counts(frame: np.ndarray) -> dict[int, int]:
        vals, cnts = np.unique(frame, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, cnts)}

    # -- bookkeeping ----------------------------------------------------------

    def _aids(self, valid: Sequence[str] | None) -> list[int]:
        """Button ids this game offers, RESET excluded.

        RESET is never emitted here. It costs a billed action to restore the
        current level, which makes it a real tool for a return-to-cell search -
        but the prose that described it is also the only live delta between the
        2.14 run and the 6.9%-RESET-rate run that scored 1.33, so it stays out
        until something measures it.
        """
        out: list[int] = []
        for raw in valid or ():
            name = str(raw or "").strip()
            if not name or is_reset(name):
                continue
            aid = DISPLAY_TO_AID.get(name.upper())
            if aid is not None and aid not in out:
                out.append(aid)
        return out

    @staticmethod
    def _press(aid: int, valid: Sequence[str] | None, row: int = -1, col: int = -1) -> Press:
        """A press spelled the way *this* game spells it.

        The name is resolved here rather than left to ``Press.arguments`` so that a
        plan is self-describing even when it is replayed without the
        ``valid_actions`` list that produced it - otherwise a press would fall back
        to an empty ``{"action": ""}``, which the gateway rejects as invalid and
        which ``step_env`` turns into a whole abandoned batch.
        """
        return Press(aid=aid, name=resolve_name(aid, valid) or AID_LABEL.get(aid, ""), row=row, col=col)

    def _roll(self, level: int, spent_on_level: int | None) -> None:
        """New level: keep the model, reset everything that was about that board."""
        if spent_on_level is not None:
            self.spent = int(spent_on_level)
        if level == self.level:
            return
        self.log.append(
            f"L{self.level} -> L{level} after {self.spent}a; {self.mind.summary()}"
        )
        self.level = level
        self.spent = 0 if spent_on_level is None else int(spent_on_level)
        self.tried.clear()
        self.clicked.clear()
        self.live_clicks.clear()
        self.acted.clear()
        self.keys.clear()
        self.stalls = 0
        self.conceded = False

    def _hand(self, why: str) -> None:
        self.handoffs += 1
        self.log.append(f"L{self.level} -> LLM: {why}")
        return None

    @property
    def _lab(self) -> bool:
        """Is this level a free laboratory?

        Level 0 always, because its whole weight is 1 out of 21-55. Any level the
        model has run out of ideas on, because a level that will not be cleared is
        billed at zero whatever happens next.
        """
        return self.level <= FIRST_LEVEL or self.conceded

    @property
    def _room(self) -> bool:
        return self.spent < (self.conceded_actions if self.conceded else self.lab_actions)

    @property
    def _cap(self) -> int:
        """Longest batch this level may emit.

        A laboratory level gets the full width, because a probe of
        ``buttons x probe_rounds`` has to fit in one turn to be worth batching. An
        examination level gets a quarter of it: truncating a *correct* route costs
        only an extra turn, while capping a *wrong* one is the difference between
        losing 24 actions and losing the level.
        """
        return self.max_batch if self._lab else max(8, self.max_batch // 4)

    @property
    def _grounded(self) -> bool:
        """Enough model to route with: a located sprite and two directions."""
        return self.mind.mech.avatar >= 0 and len(self.mind.mech.moves) >= 2

    def summary(self) -> str:
        return (
            f"L{self.level} spent={self.spent} lab={self._lab} grounded={self._grounded} "
            f"batches={self.batches} handoffs={self.handoffs} "
            f"cell={self.sense.n_varying - self.sense.n_clock}/{self.sense.n_varying}px "
            f"| {self.mind.summary()}"
        )
