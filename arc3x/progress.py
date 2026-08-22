"""What "getting closer to done" means, discovered instead of being told.

THE HOLE THIS FILLS
-------------------
The imagination in ``dream.py`` could only search toward one kind of goal: a
colour that vanishes when the avatar walks onto it. Measured consequence, on the
four dev games where the forward model is most accurate:

    dc22   move=100%(153)  collect=[]   thought=  0   levels=0
    ka59   move=100%(207)  collect=[]   thought=  0   levels=0
    m0r0   move= 92%(126)  collect=[]   thought=  0   levels=0
    sp80   move= 80%(195)  collect=[8]  thought= 61   levels=1

A *perfect* copy of the game completed nothing, because a route needs a
destination and there was none. The one game that identified a target is the one
game that planned, and the only one that finished a level. Prediction was never
the bottleneck; the objective was.

The obvious fix - learn the goal from what changed when ``levels_completed`` went
up - cannot bootstrap. It needs a completed level to learn what completes a
level, and levels are exactly what is not happening.

THE RULE
--------
So the objective has to be readable from ordinary play, before any win. The one
that generalises is:

    **progress is whatever ratchets.**

A quantity that moves one way and does not come back is the game being solved. A
quantity that oscillates is just the avatar wandering. That single test separates
them with no knowledge of the genre:

  * gems collected      - the gem colour's pixel count only ever falls
  * markers cleared     - same
  * a region painted    - the paint colour's count only ever rises
  * sokoban             - a crate parked on a target *hides* the target pixel,
                          so the target colour's count falls and stays fallen.
                          Sokoban therefore needs no sokoban-specific code.
  * walking around      - the floor colour falls and rises as the sprite covers
                          and uncovers it, in equal measure, so it is rejected

That last line is why the test is a *ratchet* and not merely "went down once".

WHAT WOULD FOOL IT, AND WHY IT DOES NOT
---------------------------------------
A draining step-counter HUD is a perfect monotone decrease, and treating it as
progress would make the agent burn its own remaining steps on purpose - the worst
possible failure. ``ls20`` has literally that widget, and this repo has been
bitten by HUD pixels before: they made every frame unique and silently killed
Go-Explore. So counting happens only over pixels outside the volatility HUD mask,
tracking starts only once that mask has enough observations to be meaningful, and
the history is discarded whenever the mask changes shape.

The history is also cut on a level change or a reset, because those restore the
board: comparing a fresh level's counts against the previous one's would read the
restoration as a giant increase and reject every real collectible.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

# The volatility mask is meaningless until it has seen a few transitions, and a
# wrong mask poisons every count taken under it, so tracking waits.
MIN_OBSERVED = 16
# Generous definition of "HUD": a pixel changing more than half the time is not
# telling us about the puzzle. Better to ignore a real object than to chase a clock.
HUD_THRESH = 0.5
# A ratchet has to click more than once, and may tolerate the occasional undo -
# pushing a crate back off a target is a legitimate move, not evidence against.
MIN_CLICKS = 2
UNDO_RATIO = 3
# An objective is a set of *things*, not a flood. A colour covering a quarter of
# the board is floor, wall or sky; its count drifting is not the puzzle being
# solved, and no amount of walking will drive it to zero. dc22 offered exactly
# this - a board-sized colour 0 with a clean downward ratchet - and chasing it
# would have replaced "no destination" with "an unreachable one".
MAX_SHARE = 0.25


@dataclass
class Progress:
    """Per-colour pixel counts over time, and which of them ratchet."""

    fell: Counter = field(default_factory=Counter)
    rose: Counter = field(default_factory=Counter)
    last: dict[int, int] = field(default_factory=dict)
    hud: np.ndarray | None = None
    ignore: set[int] = field(default_factory=set)
    # largest count ever seen per colour, to recognise a flood rather than a thing
    peak: Counter = field(default_factory=Counter)
    field_size: int = 0

    def cut(self) -> None:
        """Forget the previous frame without forgetting what was learned.

        Called on a level change or a reset. The board has been restored, so the
        next comparison is not a continuation of the last one - but the ratchet
        evidence gathered so far is still true.
        """
        self.last.clear()

    def add(self, frame: np.ndarray, hud: np.ndarray | None, *, observed: int) -> None:
        """Fold one frame into the count history."""
        if observed < MIN_OBSERVED or hud is None:
            return
        if self.hud is None:
            # Freeze the mask on first use and never revise it. Recomputing it
            # every frame looks more responsive and is in fact fatal: the mask
            # wiggles as ``observed`` grows, and clearing the history on each
            # wiggle means no colour ever accumulates enough clicks to qualify.
            # ka59 ended a 240-step run with a perfectly empty ledger that way.
            self.hud = hud.copy()
            self.field_size = int((~hud).sum())
        vals, counts = np.unique(frame[~self.hud], return_counts=True)
        now = {int(v): int(n) for v, n in zip(vals, counts)}
        for c, n in now.items():
            if n > self.peak[c]:
                self.peak[c] = n
            if c in self.ignore:
                continue
            prev = self.last.get(c)
            if prev is not None:
                if n < prev:
                    self.fell[c] += 1
                elif n > prev:
                    self.rose[c] += 1
        # A colour that disappeared entirely fell to zero; that is the strongest
        # possible ratchet click and would otherwise go unrecorded.
        for c, prev in self.last.items():
            if c not in now and c not in self.ignore and prev > 0:
                self.fell[c] += 1
        self.last = now

    # -- what ratchets -----------------------------------------------------

    def _flood(self, c: int) -> bool:
        """Is this colour scenery rather than a thing worth chasing?"""
        return bool(
            self.field_size and self.peak[c] > MAX_SHARE * self.field_size
        )

    def _ratchet(self, down: bool) -> set[int]:
        a, b = (self.fell, self.rose) if down else (self.rose, self.fell)
        return {
            c
            for c, n in a.items()
            if n >= MIN_CLICKS
            and n >= UNDO_RATIO * b.get(c, 0)
            and not self._flood(c)
        }

    @property
    def consumed(self) -> set[int]:
        """Colours being used up. Fewer of these is closer to done."""
        return self._ratchet(down=True)

    @property
    def built(self) -> set[int]:
        """Colours being accumulated. More of these is closer to done."""
        return self._ratchet(down=False) - self._ratchet(down=True)

    def count(self, frame: np.ndarray, colors: set[int]) -> int:
        if not colors:
            return 0
        m = np.isin(frame, list(colors))
        if self.hud is not None:
            m &= ~self.hud
        return int(m.sum())

    def score(self, frame: np.ndarray) -> int | None:
        """Distance from done - lower is better. ``None`` = no notion yet.

        Returning None rather than 0 matters: "everything is equally good" and
        "I have no idea what good means" lead to opposite decisions, and only the
        second one should stop the agent from planning.
        """
        few = self.consumed
        many = self.built
        if not few and not many:
            return None
        return self.count(frame, few) - self.count(frame, many)

    def summary(self) -> str:
        return f"consumed={sorted(self.consumed)} built={sorted(self.built)}"
