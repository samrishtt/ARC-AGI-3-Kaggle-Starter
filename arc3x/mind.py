"""The agent's mental model of a game it has never seen.

This is the piece that makes 10+ arithmetically possible, and the reason is
economic rather than clever. Every action taken in the world is billed against
the score; every action taken *in imagination* is free. So the agent's job is to
spend level 0's nearly-free action budget buying a simulator, and thereafter to
do its searching inside that simulator and spend real actions only on executing
a finished plan. That is also, precisely, what a person does: wiggle the keys for
ten seconds, then stop pressing and look at the screen and work out the route.

What gets learned, in the order a person learns it:

  ``deltas``   which button moves me, and by how many pixels. Learned by
               repeated presses and confirmed by *reversibility* - if action 3
               moves me (0,-5) and action 4 moves me (0,+5), those are left and
               right, and no amount of background-shaped coincidence will fake
               that.
  ``avatar``   which object is me. The one that obeys the buttons. Never the
               background: a person does not think they are the empty space,
               and colour-mask matching wrongly reported exactly that on four
               of the 25 dev games.
  ``blocking`` what stops me. The colours occupying the cells I failed to enter,
               accumulated over every refused move.
  ``fatal``    what kills me. The colours I had just entered when the state went
               GAME_OVER. Cheap to learn because ``level_reset`` costs one action
               and clears GAME_OVER, so dying is a survivable experiment.
  ``goal``     what winning looks like. Whatever object vanished, or whatever
               colour I had just stepped onto, on the action where
               ``levels_completed`` went up. This is the transfer that makes deep
               levels cheap: level 0 is where the goal is identified, levels 1..n
               are where knowing it is worth 2..n times as much.

``predict`` and ``plan`` are the imagination. ``plan`` runs breadth-first over
avatar positions using the learned step deltas and the learned blocking set, so
the route it returns is the *shortest* one the model knows about - which matters
because the score is quadratic in action count, and a wandering route that
arrives is worth a small fraction of a direct one.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from math import gcd

import numpy as np

from arc3x.percept import Move, background, mask_component, moved_objects

MAX_STEP = 16  # a step, not a teleport; anything bigger is a mismatched object


@dataclass
class Mechanics:
    """Everything the agent believes about the game it is playing."""

    background: int = -1
    avatar: int = -1
    # Every colour that moves together with the avatar. A sprite is usually more
    # than one colour - ls20's is a colour-12 head on a colour-9 body - and
    # tracking only one of them puts the footprint in the wrong place, which
    # makes every wall test read the wrong cells.
    body: set[int] = field(default_factory=set)
    # action id -> (dy, dx) of the controlled object
    deltas: dict[int, tuple[int, int]] = field(default_factory=dict)
    # action id -> how many times pressing it changed nothing at all
    noop: Counter = field(default_factory=Counter)
    tries: Counter = field(default_factory=Counter)
    # action id -> times it changed the frame, and times it translated the avatar.
    # The difference between the two is the whole point: a button that changes the
    # board without moving us is a *use* button - grab, drop, select, rotate - and
    # ``moves`` throws those away because a route cannot be made of them. Measured
    # across the 25 dev games, 12 have one, and cd82 and tr87 have *nothing else*:
    # five and four working buttons respectively, all invisible to the planner.
    changes: Counter = field(default_factory=Counter)
    shifts: Counter = field(default_factory=Counter)
    blocking: Counter = field(default_factory=Counter)
    passable: Counter = field(default_factory=Counter)
    fatal: Counter = field(default_factory=Counter)
    goal_colors: Counter = field(default_factory=Counter)
    vanished: Counter = field(default_factory=Counter)
    # votes[(action, color, delta)] -> count, before consensus is taken
    votes: Counter = field(default_factory=Counter)
    # last known sprite position, so the sprite is tracked rather than re-deduced
    pos: tuple[int, int] | None = None

    # -- learning ---------------------------------------------------------

    def observe(
        self,
        action: int,
        before: np.ndarray,
        after: np.ndarray,
        *,
        level_up: bool = False,
        died: bool = False,
    ) -> list[Move]:
        """Fold one transition into the model. Returns the objects that moved."""
        if self.background < 0:
            self.background = background(before)
        self.tries[action] += 1
        if not (before != after).any():
            self.noop[action] += 1
            # Nothing moved. If we believe we have an avatar and a delta for this
            # action, the cells it tried to enter are what stopped it.
            self._blame_block(before, action)
            return []
        self.changes[action] += 1
        mv, van, app = moved_objects(before, after)
        for m in mv:
            if not m.moved or m.color == self.background:
                continue
            if abs(m.dy) > MAX_STEP or abs(m.dx) > MAX_STEP:
                continue
            self.votes[(action, m.color, m.delta)] += 1
        # What did we just walk onto? Sampled from the sprite's real footprint in
        # its new position, read off the *previous* frame. This is the only
        # observation that makes passable-only planning possible, so it has to be
        # exact: guessing a sqrt(size) square around a two-colour sprite reads
        # the wrong cells and teaches the model that the floor is a wall.
        entered: set[int] = set()
        if self.avatar >= 0:
            # The best possible hint is the arrival position of the thing we just
            # watched move. "That moved, so that is me" needs no deduction.
            hint = self.pos
            for m in mv:
                if m.color == self.avatar and m.moved:
                    hint = (m.top + m.dy, m.left + m.dx)
                    break
            box = self.locate(after, hint=hint)
            if box is not None:
                if box[:2] != self.pos:
                    entered = self._under(before, after, box)
                    if self.pos is not None:
                        self.shifts[action] += 1
                self.pos = (box[0], box[1])
        for c in entered:
            # Without this, one refusal would condemn a colour permanently and a
            # door that opened would stay shut in the model forever.
            self.passable[c] += 1
        for m in van:
            if m.color != self.background:
                self.vanished[m.color] += 1
        if level_up:
            # Whatever disappeared on the winning action, or whatever we stepped
            # onto, is what a goal looks like in this game.
            for m in van:
                if m.color != self.background:
                    self.goal_colors[m.color] += 3
            for c in entered:
                if c != self.background:
                    self.goal_colors[c] += 2
        if died:
            # Only *novel* ground gets blamed for a death. A footprint spans
            # several cells, so the fatal step also lands on ordinary floor, and
            # crediting that would mark the floor lethal and freeze the agent in
            # place. Ground we have stood on repeatedly is not what killed us.
            for c in entered:
                if self.passable.get(c, 0) < 3:
                    self.fatal[c] += 1
        return mv

    def _under(
        self, before: np.ndarray, after: np.ndarray, box: tuple[int, int, int, int]
    ) -> set[int]:
        """Colours that were sitting where the sprite now stands.

        The background is included, because "I can stand on the empty stuff" is a
        real and useful fact - in most games it is the floor - and leaving it out
        left ``passable`` empty on exactly those games.
        """
        t, l, h, w = box
        foot = self.footprint(after, box)
        patch = before[t : t + h, l : l + w]
        body = self.body or {self.avatar}
        return {int(c) for c in np.unique(patch[foot]) if int(c) not in body}

    def _blame_block(self, frame: np.ndarray, action: int) -> None:
        """A refused move names its own obstacle: whatever occupies the cells
        the sprite's footprint would have swept into."""
        d = self.deltas.get(action)
        if d is None or self.avatar < 0:
            return
        box = self.locate(frame, hint=self.pos)
        if box is None:
            return
        top, left, h, w = box
        self.pos = (top, left)
        foot = self.footprint(frame, box)
        dy, dx = d
        H, W = frame.shape
        body = self.body or {self.avatar}
        ys, xs = np.nonzero(foot)
        hit = False
        for y, x in zip(ys.tolist(), xs.tolist()):
            ny, nx = top + y + dy, left + x + dx
            if not (0 <= ny < H and 0 <= nx < W):
                continue
            # a cell the sprite already occupies cannot be what stopped it
            fy, fx = ny - top, nx - left
            if 0 <= fy < h and 0 <= fx < w and foot[fy, fx]:
                continue
            c = int(frame[ny, nx])
            if c in body:
                continue
            self.blocking[c] += 1
            hit = True
        if not hit:
            return

    # -- consensus ---------------------------------------------------------

    def settle(self, min_votes: int = 2) -> None:
        """Turn the raw votes into a believed avatar and a delta per action.

        Two filters do the work. First, the background is never a candidate.
        Second, a candidate is scored on *reversibility*: a real movement scheme
        contains opposite pairs, so a colour whose deltas include (dy,dx) and
        (-dy,-dx) under different actions is almost certainly the thing being
        steered, while a coincidental shape match almost never is.
        """
        by_color: dict[int, dict[int, tuple[int, int]]] = {}
        strength: Counter = Counter()
        for (a, c, d), n in self.votes.items():
            if n < min_votes or c == self.background:
                continue
            cur = by_color.setdefault(c, {})
            # keep the most-voted delta for this (colour, action)
            best = self.votes[(a, c, cur[a])] if a in cur else -1
            if n > best:
                cur[a] = d
            strength[c] += n
        if not by_color:
            return

        def score(c: int) -> tuple:
            ds = set(by_color[c].values())
            rev = sum(1 for (dy, dx) in ds if (-dy, -dx) in ds)
            axis = sum(1 for (dy, dx) in ds if dy == 0 or dx == 0)
            mags = {abs(v) for dy, dx in ds for v in (dy, dx) if v}
            coherent = 1 if len(mags) <= 2 else 0
            return (rev, axis, coherent, len(by_color[c]), strength[c])

        self.avatar = max(by_color, key=score)
        self.deltas = dict(by_color[self.avatar])
        self._fill_deltas()
        # Anything that moved the same way under the same buttons is part of the
        # same sprite. "What moves together is one thing" is how a person parses
        # the screen, and it is what makes the footprint - and therefore every
        # collision test - correct.
        self.body = {self.avatar}
        for c, ds in by_color.items():
            if c == self.avatar:
                continue
            shared = set(ds) & set(self.deltas)
            if len(shared) >= 2 and all(ds[a] == self.deltas[a] for a in shared):
                self.body.add(c)

    def _fill_deltas(self) -> None:
        """Second pass: accept single-observation deltas for the believed avatar.

        Working out *who I am* needs strong evidence, so the vote threshold above
        is right for picking the avatar. Applying that same threshold to the
        buttons is what loses half the movement map: a sprite that rotates as it
        turns - wa30, sc25 - only matches as a rigid translation on the axis it
        already faces, so two of its four directions never reach two votes and
        the planner ends up with ``[2, 4]`` out of four working buttons.

        Once we know who we are, one clean sighting of "button 3 moved me left"
        is enough - provided it fits the scheme we already believe. A single
        stray shape-match is rejected because a real movement scheme has
        consistent step sizes and opposite pairs; a coincidence has neither.
        """
        if self.avatar < 0:
            return
        known = set(self.deltas.values())
        if not known:
            return
        mags = {abs(v) for dy, dx in known for v in (dy, dx) if v}
        for (a, c, d), n in self.votes.items():
            if c != self.avatar or a in self.deltas or d == (0, 0):
                continue
            dy, dx = d
            fits_pair = (-dy, -dx) in known
            fits_step = all(abs(v) in mags for v in (dy, dx) if v)
            if fits_pair or fits_step:
                self.deltas[a] = d

    @property
    def tile(self) -> int:
        g = 0
        for dy, dx in self.deltas.values():
            for v in (abs(dy), abs(dx)):
                if v:
                    g = gcd(g, v)
        return g or 1

    @property
    def moves(self) -> dict[int, tuple[int, int]]:
        """Actions that displace the avatar, smallest step first."""
        return {
            a: d
            for a, d in sorted(self.deltas.items(), key=lambda kv: abs(kv[1][0]) + abs(kv[1][1]))
            if d != (0, 0)
        }

    @property
    def acts(self) -> dict[int, int]:
        """Buttons that change the board without moving us, biggest effect first.

        A route cannot be made of these, which is why ``moves`` excludes them -
        but a person uses them constantly: walk up to the thing, press use. Twelve
        of the 25 dev games have one, and cd82 and tr87 have *nothing else*, so
        discarding them is discarding those games entirely.

        The value is how many times the button was seen to work, not what it does;
        what it does is for the imagination to find out by pressing it.
        """
        return dict(
            sorted(
                (
                    (a, n)
                    for a, n in self.changes.items()
                    if n >= 1 and not self.shifts.get(a, 0) and a not in self.moves
                ),
                key=lambda kv: -kv[1],
            )
        )

    @property
    def blocked_set(self) -> set[int]:
        """Colours believed impassable, plus anything believed lethal.

        Evidence on both sides is kept: a colour that refused us once but that
        we have since stood on is a door that opened, not a wall, and treating
        it as a wall would make whole levels unreachable in the model. So a
        colour is impassable only while the refusals outweigh the times we have
        actually occupied it.
        """
        out = {
            c
            for c, n in self.blocking.items()
            if n >= 1 and n > self.passable.get(c, 0)
        }
        out |= {c for c, n in self.fatal.items() if n >= 1}
        return out

    # -- where am I --------------------------------------------------------

    def locate(
        self, frame: np.ndarray, hint: tuple[int, int] | None = None
    ) -> tuple[int, int, int, int] | None:
        """Bounding box (top, left, h, w) of the whole sprite in this frame.

        The sprite is the connected clump of *any* body colour, so a two-colour
        avatar is found as one object rather than as its head only - and getting
        that wrong puts every collision test on the wrong cells.

        Two rules keep it on the right clump. It must contain a pixel of the
        primary avatar colour: ls20's body colour 9 also appears as a single-pixel
        speck in the HUD at (13,35), and "take the smallest component" cheerfully
        decided that speck was the player. And when several candidates qualify,
        the one nearest ``hint`` - where we last saw ourselves - wins, because a
        person keeps their eye on their avatar instead of re-deducing it from
        scratch every frame.
        """
        if self.avatar < 0:
            return None
        body = self.body or {self.avatar}
        mask = np.isin(frame, list(body))
        prim = frame == self.avatar
        if not prim.any():
            prim = mask
            if not mask.any():
                return None
        best = None
        seen = np.zeros(frame.shape, dtype=bool)
        ys, xs = np.nonzero(prim)
        for y, x in zip(ys.tolist(), xs.tolist()):
            if seen[y, x]:
                continue
            sub, t, l, n = mask_component(mask, y, x)
            h, w = sub.shape
            seen[t : t + h, l : l + w] |= sub
            if n > 400:
                continue
            # how much of the sprite's palette this clump shows: the real avatar
            # shows all of it, a same-coloured piece of scenery shows one colour
            ncol = len(np.unique(frame[t : t + h, l : l + w][sub]))
            if hint is None:
                key = (-ncol, n)
            else:
                key = (abs(t - hint[0]) + abs(l - hint[1]), -ncol, n)
            if best is None or key < best[0]:
                best = (key, t, l, h, w)
        if best is None:
            return None
        return (best[1], best[2], best[3], best[4])

    def footprint(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        """Boolean mask of the sprite's own pixels inside ``box``."""
        t, l, h, w = box
        return np.isin(frame[t : t + h, l : l + w], list(self.body or {self.avatar}))

    def where(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        """Locate the sprite and remember it. Tracking beats re-deducing: it is
        what keeps us from mistaking a same-coloured wall for ourselves."""
        box = self.locate(frame, hint=self.pos)
        if box is not None:
            self.pos = (box[0], box[1])
        return box

    # -- imagination -------------------------------------------------------

    def walk_mask(self, frame: np.ndarray) -> np.ndarray:
        """Per-pixel "the sprite may occupy this", built once instead of per node.

        The polarity here is the whole difference between 0.00 and a score, and
        it is the opposite of what sounds cautious. "Everything is walkable until
        something refuses me" made the model believe all 156 grid cells of ls20
        were open, plan routes straight through the void, and label the floor as
        a wall. A person's default is the other way round: you walk on the ground
        you are already standing on, and everything else is unknown until you try
        it. So a cell is walkable only if every pixel in it shows a colour we have
        actually occupied - or is part of us.

        Until we have occupied anything at all there is no evidence to go on, so
        it falls back to the permissive rule: with an empty map, a blank one is
        more useful than a closed one.

        ``plan`` is called after every single action, so this is one ``np.isin``
        over 4,096 pixels rather than a numpy call per BFS node - the difference
        between 0.1 s and 1 ms per decision.
        """
        good = {c for c, n in self.passable.items() if n > 0} - self.blocked_set
        if not good:
            bad = self.blocked_set
            if not bad:
                return np.ones(frame.shape, dtype=bool)
            return ~np.isin(frame, list(bad))
        good |= self.body or {self.avatar}
        return np.isin(frame, list(good))

    def free(self, frame: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
        """Could the sprite stand here, according to where we have already stood?"""
        return self._free(self.walk_mask(frame), top, left, h, w, *frame.shape)

    @staticmethod
    def _free(
        walk: np.ndarray, top: int, left: int, h: int, w: int, H: int, W: int
    ) -> bool:
        if top < 0 or left < 0 or top + h > H or left + w > W:
            return False
        return bool(walk[top : top + h, left : left + w].all())

    def plan(
        self,
        frame: np.ndarray,
        targets: list[tuple[int, int]],
        *,
        max_nodes: int = 8000,
    ) -> list[int]:
        """Shortest button sequence that lands the sprite on any target cell.

        Breadth-first over sprite positions using the learned step deltas. This
        is the free part: no actions are spent, so the search can be as wide as
        the model allows, and what comes back is a minimum-length route, which is
        what a quadratic score rewards.

        The route travels only over proven ground, but it is allowed to *end* on
        an unproven cell, because that is exactly the move worth spending an
        action on: the target is a thing we have not touched yet, and finding out
        what it does is the point.
        """
        box = self.locate(frame, hint=self.pos)
        mv = self.moves
        if box is None or not mv or not targets:
            return []
        top, left, h, w = box
        H, W = frame.shape
        walk = self.walk_mask(frame)
        goal = np.zeros(frame.shape, dtype=bool)
        for y, x in targets:
            if 0 <= y < H and 0 <= x < W:
                goal[y, x] = True
        if goal[top : top + h, left : left + w].any():
            return []
        arrive = walk | goal
        seen = {(top, left)}
        q: deque[tuple[int, int, list[int]]] = deque([(top, left, [])])
        n = 0
        while q and n < max_nodes:
            t, l, path = q.popleft()
            n += 1
            for a, (dy, dx) in mv.items():
                nt, nl = t + dy, l + dx
                if (nt, nl) in seen:
                    continue
                if goal[nt : nt + h, nl : nl + w].any() and self._free(
                    arrive, nt, nl, h, w, H, W
                ):
                    return path + [a]
                if not self._free(walk, nt, nl, h, w, H, W):
                    continue
                seen.add((nt, nl))
                q.append((nt, nl, path + [a]))
        return []

    def reachable(self, frame: np.ndarray, *, max_nodes: int = 8000) -> dict:
        """Every sprite position the model thinks is reachable, with its route."""
        box = self.locate(frame, hint=self.pos)
        mv = self.moves
        if box is None or not mv:
            return {}
        top, left, h, w = box
        H, W = frame.shape
        walk = self.walk_mask(frame)
        out = {(top, left): []}
        q: deque[tuple[int, int]] = deque([(top, left)])
        while q and len(out) < max_nodes:
            t, l = q.popleft()
            for a, (dy, dx) in mv.items():
                nt, nl = t + dy, l + dx
                if (nt, nl) in out:
                    continue
                if not self._free(walk, nt, nl, h, w, H, W):
                    continue
                out[(nt, nl)] = out[(t, l)] + [a]
                q.append((nt, nl))
        return out

    def frontier_colors(self, frame: np.ndarray) -> dict[int, list[tuple[int, int]]]:
        """Colours we have never been allowed to enter, and where to push at them.

        A person confronted with a locked-looking thing tries it once. Some of
        those things are doors, some are goals, some are walls, and the only way
        to tell from the outside is to walk into each kind once. The value is the
        sprite positions from which a single learned move would enter that colour.
        """
        box = self.locate(frame, hint=self.pos)
        mv = self.moves
        if box is None or not mv:
            return {}
        _t, _l, h, w = box
        H, W = frame.shape
        body = self.body or {self.avatar}
        out: dict[int, list[tuple[int, int]]] = {}
        for (t, l), _path in self.reachable(frame).items():
            for _a, (dy, dx) in mv.items():
                nt, nl = t + dy, l + dx
                if nt < 0 or nl < 0 or nt + h > H or nl + w > W:
                    continue
                patch = frame[nt : nt + h, nl : nl + w]
                for c in np.unique(patch):
                    c = int(c)
                    if c == self.background or c in body:
                        continue
                    if self.passable.get(c, 0) > 0:
                        continue
                    out.setdefault(c, []).append((t, l))
        return out

    # -- reporting ---------------------------------------------------------

    def summary(self) -> str:
        d = " ".join(f"{a}:{dy},{dx}" for a, (dy, dx) in sorted(self.deltas.items()))
        return (
            f"avatar={self.avatar} bg={self.background} tile={self.tile} "
            f"moves[{d}] block={sorted(self.blocked_set)} "
            f"fatal={sorted(self.fatal)} goal={[c for c, _ in self.goal_colors.most_common(3)]}"
        )
