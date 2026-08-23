"""A general agent for a game it has never seen. One shot, billed per action.

The policy is the one a person uses, in the same order, for the same reasons.

  1. WIGGLE. Press every button a few times and watch what moves. ~25 actions,
     charged to level 0, where the weight is 1 out of 21..55 - so this costs
     almost nothing and buys the entire model.
  2. LOOK. Decide what is worth walking to. With no reward signal but
     ``levels_completed``, the ranking a person uses is: something that looked
     like a goal last time > something rare and small > somewhere I have not
     been. Rarity first is not arbitrary - a single blob of a colour that
     appears once is the thing the level was built around.
  3. THINK. Plan the route inside the learned model. This is free, so it is
     where all the searching happens, and because it is breadth-first the route
     is the shortest the model knows - which matters when score goes as
     (baseline/actions)^2.
  4. WALK. Execute, checking after every action that the avatar went where the
     model said. A mismatch is information: the thing in the way gets added to
     the blocking set and the route is re-planned.
  5. CARRY IT OVER. On a new level, keep the model. This is where the score
     comes from: level 0 pays for the learning, levels 1..n are worth 2..n times
     as much and reuse it for free.

Death is cheap and therefore worth risking: ``level_reset`` costs one action and
clears GAME_OVER, so the agent finds out what is lethal by touching it once.

For games with no steerable avatar - eight of the 25 dev games declare action 6
and little else - the same loop runs over clicks instead of moves, ranking
candidate targets by the same rarity heuristic.
"""

from __future__ import annotations

import time
from collections import Counter

import numpy as np

from arc3x.dream import Dream
from arc3x.graded import GObs, GradedRun
from arc3x.mind import Mechanics
from arc3x.percept import Volatility, background, blobs, fingerprint

WIGGLE_REPS = 4


class Agent:
    """One play of one game. Holds the model; the model outlives each level."""

    def __init__(
        self,
        run: GradedRun,
        *,
        budget: int = 3000,
        seconds: float = 240.0,
        seed: int = 0,
        verbose: bool = False,
    ) -> None:
        self.run = run
        self.budget = budget
        self.deadline = time.perf_counter() + seconds
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.m = Mechanics()
        # The imagination, and the objective it searches toward. Without this the
        # agent could only walk to colours that had *already* coincided with a
        # completed level - a signal that needs a win to learn what a win is.
        self.dream = Dream(self.m)
        self.vol = Volatility()
        self.seen: set[bytes] = set()
        self.visited: set[tuple[int, int]] = set()
        self.tried_targets: set[tuple[int, int]] = set()
        self.dead_clicks: Counter = Counter()
        self.level = 0
        self.obs: GObs | None = None

    # -- plumbing ---------------------------------------------------------

    @property
    def spent(self) -> bool:
        return self.run.actions >= self.budget or time.perf_counter() > self.deadline

    @property
    def timed_out(self) -> bool:
        """True when the clock stopped us rather than the action budget.

        Kaggle bills actions, not seconds, so a run cut short by the wall clock
        is not a measurement - it is a different, machine-dependent experiment,
        and two such runs are not comparable. It has to be visible in the report
        rather than silently averaged in.
        """
        return self.run.actions < self.budget and time.perf_counter() > self.deadline

    def act(self, aid: int, x: int = 0, y: int = 0) -> GObs:
        """One billed action, folded into the model and the novelty memory."""
        before = self.obs.frame if self.obs is not None else None
        obs = self.run.step(aid, x, y)
        level_up = obs.levels_completed > self.level
        if before is not None:
            self.vol.add(before, obs.frame)
            # The dream grades itself *before* the mechanics update, so the
            # prediction being scored is the one the agent actually acted on.
            self.dream.observe(aid, before, obs.frame)
            self.m.observe(
                aid, before, obs.frame, level_up=level_up, died=obs.game_over
            )
        self.obs = obs
        self.seen.add(fingerprint(obs.frame, self.vol.live_mask))
        if level_up:
            if self.verbose:
                print(
                    f"    level {obs.levels_completed} at action {self.run.actions}"
                    f"  [{self.m.summary()}]"
                )
            self.level = obs.levels_completed
            self.on_new_level()
        elif obs.game_over:
            # One action to undo death, and the level is pristine again.
            self.obs = self.run.reset()
            self.on_new_level()
        return obs

    def on_new_level(self) -> None:
        self.visited.clear()
        self.tried_targets.clear()
        self.dead_clicks.clear()
        # A new level puts us somewhere else entirely, so the tracked position is
        # stale and would drag ``locate`` toward the wrong lookalike.
        self.m.pos = None
        # The board has been restored, so frame-to-frame counts no longer
        # continue; what was learned about which colours ratchet still holds.
        self.dream.cut()

    # -- 1. wiggle ---------------------------------------------------------

    def wiggle(self) -> None:
        """Press everything a few times. The cheapest information in the game."""
        acts = [a for a in self.run._declared if a != 6]
        for rep in range(WIGGLE_REPS):
            for a in acts:
                if self.spent:
                    return
                self.act(a)
                # A move that is refused twice tells us about walls, so keep
                # going even when nothing happens.
            self.m.settle()
            if rep >= 1 and self.m.avatar >= 0 and len(self.m.moves) >= 2:
                # Enough to steer with. Reversibility already confirmed it.
                break
        self.m.settle()

    # -- 2. look -----------------------------------------------------------

    def targets(self, frame: np.ndarray) -> list[tuple[int, int]]:
        """Cells worth walking to, best first.

        The ranking is the human one: a colour that has ever coincided with a
        level completion, then a rare small object, then anywhere unvisited.
        """
        bg = self.m.background if self.m.background >= 0 else background(frame)
        bl = blobs(frame, ignore={bg, self.m.avatar})
        if not bl:
            return []
        colcount = Counter(b.color for b in bl)
        goal = {c for c, _ in self.m.goal_colors.most_common(4)}
        bad = self.m.blocked_set
        scored: list[tuple[tuple, tuple[int, int]]] = []
        box = self.m.where(frame)
        ay, ax = (box[0], box[1]) if box else (32, 32)
        for b in bl:
            if b.color in bad and b.color not in goal:
                continue
            cy, cx = b.center
            if (cy, cx) in self.tried_targets:
                continue
            key = (
                0 if b.color in goal else 1,     # a known goal beats everything
                colcount[b.color],               # rare things are special
                b.size,                          # small things are pickups
                abs(cy - ay) + abs(cx - ax),      # then nearest, to stay cheap
            )
            scored.append((key, (cy, cx)))
        scored.sort()
        return [p for _k, p in scored]

    # -- 3+4. think and walk ----------------------------------------------

    def navigate(self, cells: list[tuple[int, int]], max_actions: int = 120) -> str:
        """Walk toward any of ``cells``, re-planning after every single step.

        Re-planning per step rather than per route is the whole trick. The model
        starts out believing nothing blocks it, so a route planned once goes
        straight through the first wall and the remaining actions are wasted.
        Executing one action and re-planning costs nothing extra - planning is
        free - and each refused move teaches the model a wall, so the route
        repairs itself as it is walked. This is why the agent maps a level in
        O(path) billed actions instead of O(area).

        Returns 'level', 'win', 'change', 'stuck' or 'spent'.
        """
        assert self.obs is not None
        stuck_at: Counter = Counter()
        for _ in range(max_actions):
            if self.spent:
                return "spent"
            box = self.m.where(self.obs.frame)
            if box is None:
                return "stuck"
            here = (box[0], box[1])
            if here in cells:
                return "change"
            path = self.m.plan(self.obs.frame, cells)
            if not path:
                return "stuck"
            lv = self.level
            before = self.obs.frame
            self.act(path[0])
            if self.obs.won:
                return "win"
            if self.level != lv:
                return "level"
            if not (before != self.obs.frame).any():
                # Refused. Mechanics.observe has just recorded the blocking
                # colour, so the next plan routes around it - unless we have now
                # been refused from this same square repeatedly, in which case
                # the model is not learning and we should try something else.
                stuck_at[here] += 1
                if stuck_at[here] >= 3:
                    return "stuck"
                continue
            nb = self.m.where(self.obs.frame)
            if nb:
                self.visited.add((nb[0], nb[1]))
        return "stuck"

    def explore_step(self) -> str:
        """Nothing looks interesting: walk to the farthest place we have not been.

        The frontier comes from the model's own reachability, which is free to
        compute, so exploration is directed rather than random. A random walk
        needs O(n^2) actions to cover n cells; walking the frontier needs O(n),
        and at a quadratic score that difference is the whole game.
        """
        assert self.obs is not None
        reach = self.m.reachable(self.obs.frame)
        fresh = [(len(p), c) for c, p in reach.items() if p and c not in self.visited]
        if not fresh:
            self.visited.clear()
            fresh = [(len(p), c) for c, p in reach.items() if p]
        if not fresh:
            return "stuck"
        fresh.sort(reverse=True)
        return self.navigate([c for _n, c in fresh[:8]], max_actions=60)

    def imagine(self, max_actions: int = 200) -> str:
        """Act on the imagination: think for free, spend one action, think again.

        This is the step the agent was missing entirely. ``navigate`` walks toward
        cells that some heuristic ranked; this walks toward whatever the learned
        copy of the game says will *reduce the distance to done*, which is the
        only one of the two that generalises to a game whose goal nobody named.

        Re-planning after each single action rather than committing to the whole
        route is the same discipline as ``navigate``, for the same reason: the
        copy is sometimes wrong, and one billed action is the cheapest possible
        way to find that out.

        Returns 'level', 'win', 'noplan', 'stuck' or 'budget'.
        """
        assert self.obs is not None
        used0 = self.run.actions
        stall = 0
        while self.run.actions - used0 < max_actions and not self.spent:
            plan = self.dream.route(self.obs.frame)
            if not plan:
                return "noplan"
            lv = self.level
            before = self.obs.frame
            self.act(plan[0])
            if self.obs.won:
                return "win"
            if self.level != lv:
                return "level"
            if not (before != self.obs.frame).any():
                # The copy promised a change and nothing happened. Mechanics has
                # just learned the obstacle, so one more try is worth it - but a
                # copy that keeps being wrong here is not worth spending on.
                stall += 1
                if stall >= 4:
                    return "stuck"
            else:
                stall = 0
        return "budget"

    # -- the click branch --------------------------------------------------

    def click_candidates(self, frame: np.ndarray) -> list[tuple[int, int]]:
        """Where a person would click, best first.

        Blob centres, plus the four corners of blobs big enough that the centre
        might miss the interactive part. Ranked by the same rarity heuristic as
        walking targets, and de-prioritised once a click there has been shown to
        do nothing.
        """
        bg = self.m.background if self.m.background >= 0 else background(frame)
        bl = blobs(frame, ignore={bg})
        colcount = Counter(b.color for b in bl)
        goal = {c for c, _ in self.m.goal_colors.most_common(4)}
        out: list[tuple[tuple, tuple[int, int]]] = []
        for b in bl:
            pts = [b.center]
            if b.size >= 9:
                pts += [
                    (b.top, b.left),
                    (b.top, b.left + b.width - 1),
                    (b.top + b.height - 1, b.left),
                    (b.top + b.height - 1, b.left + b.width - 1),
                ]
            for y, x in pts:
                y = int(min(63, max(0, y)))
                x = int(min(63, max(0, x)))
                out.append(
                    (
                        (
                            self.dead_clicks[(y, x)],
                            0 if b.color in goal else 1,
                            colcount[b.color],
                            b.size,
                        ),
                        (y, x),
                    )
                )
        out.sort()
        seen: set[tuple[int, int]] = set()
        uniq = []
        for _k, p in out:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return uniq

    def click_round(self) -> bool:
        """Try the ranked clicks until one does something."""
        assert self.obs is not None
        cands = self.click_candidates(self.obs.frame)
        if not cands:
            return False
        for y, x in cands[:64]:
            if self.spent:
                return False
            lv = self.level
            before = self.obs.frame
            self.act(6, x=x, y=y)
            if self.level != lv or self.obs.terminal:
                return True
            if (before != self.obs.frame).any():
                return True
            self.dead_clicks[(y, x)] += 1
        return False

    # -- fallback ----------------------------------------------------------

    def flail(self, n: int = 40) -> None:
        """No model, nothing ranked worked. Random, but never repeat a dead action."""
        acts = list(self.run._declared) or [1, 2, 3, 4, 5, 6]
        dead = {a for a in acts if self.m.tries[a] >= 4 and self.m.noop[a] == self.m.tries[a]}
        live = [a for a in acts if a not in dead] or acts
        for _ in range(n):
            if self.spent:
                return
            a = int(self.rng.choice(live))
            if a == 6:
                cands = self.click_candidates(self.obs.frame) if self.obs else []
                if cands:
                    y, x = cands[int(self.rng.integers(len(cands)))]
                else:
                    y, x = int(self.rng.integers(64)), int(self.rng.integers(64))
                self.act(6, x=x, y=y)
            else:
                self.act(a)
            if self.obs is not None and self.obs.won:
                return

    # -- the whole play ----------------------------------------------------

    def cover(self, max_actions: int = 400) -> str:
        """Stand on every square the model thinks we can reach.

        This is the workhorse, and it is deliberately dumber than the targeting
        heuristic. Most levels complete when the avatar touches the right thing,
        and enumerating everything walkable is the only method that does not
        depend on guessing which thing that is. Directed frontier walking makes
        it affordable: covering n squares costs O(n) actions, where a random walk
        would cost O(n^2) - and O(n) for a 5px grid on a 64x64 board is a few
        hundred actions, which is exactly the budget the scoring formula allows.
        """
        used0 = self.run.actions
        while self.run.actions - used0 < max_actions and not self.spent:
            out = self.explore_step()
            if out in ("level", "win"):
                return out
            if out == "stuck":
                return "covered"
        return "budget"

    def push_frontier(self, max_actions: int = 200) -> str:
        """Walk into each kind of thing we have never been allowed to enter, once.

        A blocked colour might be a wall, a door, a hazard or the goal, and from
        the outside those look identical. A person tries each kind once and reads
        the result. One attempt per *colour*, not per tile, keeps it cheap.
        """
        assert self.obs is not None
        used0 = self.run.actions
        for color, spots in sorted(
            self.m.frontier_colors(self.obs.frame).items(),
            key=lambda kv: (
                0 if kv[0] in {c for c, _ in self.m.goal_colors.most_common(4)} else 1,
                -len(kv[1]),
            ),
        ):
            if self.spent or self.run.actions - used0 >= max_actions:
                return "budget"
            if color in self.m.fatal:
                continue
            out = self.navigate(spots[:24], max_actions=40)
            if out in ("level", "win"):
                return out
            box = self.m.where(self.obs.frame)
            if box is None:
                continue
            # We are standing next to it; now shove into it from here.
            for a, (dy, dx) in self.m.moves.items():
                ny, nx = box[0] + dy, box[1] + dx
                if not (0 <= ny < 64 and 0 <= nx < 64):
                    continue
                patch = self.obs.frame[ny : ny + box[2], nx : nx + box[3]]
                if patch.size and (patch == color).any():
                    lv = self.level
                    self.act(a)
                    if self.obs.won:
                        return "win"
                    if self.level != lv:
                        return "level"
                    break
        return "pushed"

    def play(self) -> None:
        self.obs = self.run.reset()
        self.seen.add(fingerprint(self.obs.frame))
        self.wiggle()
        if self.verbose:
            print(f"    after wiggle ({self.run.actions} actions): {self.m.summary()}")

        rounds = 0
        while not self.spent and not (self.obs and self.obs.won):
            rounds += 1
            level_at_start = self.level
            L = level_at_start
            steerable = self.m.avatar >= 0 and bool(self.m.moves)
            # Recorded once per round, because "did this agent ever have a
            # destination on this level" is the first question to ask of a zero
            # and the score cannot answer it.
            self.run.note("steer" if steerable else "nosteer", L)
            if self.dream.objective(self.obs.frame) is None:
                self.run.note("noobjective", L)

            if steerable:
                # First ask the imagination. It is the only part of the agent that
                # can say "this specific sequence makes the board closer to done"
                # rather than "this cell looks interesting", so when it has an
                # opinion it outranks every heuristic below.
                out = self.imagine()
                self.run.note(f"imagine:{out}", L)
                if out in ("level", "win"):
                    continue
                # A known goal colour means we can go straight there. This is
                # the transfer that pays for the level-0 exploration: levels
                # 1..n are worth 2..n times as much and cost a fraction.
                goal = {c for c, _ in self.m.goal_colors.most_common(3)}
                if goal:
                    cells = self.goal_cells(self.obs.frame, goal)
                    if cells:
                        out = self.navigate(cells, max_actions=150)
                        self.run.note(f"goal:{out}", L)
                        if out in ("level", "win"):
                            continue
                out = self.cover()
                self.run.note(f"cover:{out}", L)
                if out in ("level", "win"):
                    continue
                out = self.push_frontier()
                self.run.note(f"frontier:{out}", L)
                if out in ("level", "win"):
                    continue

            if 6 in self.run._declared and self.click_round():
                self.run.note("click:change", L)
                continue

            if self.level != level_at_start:
                continue

            # Nothing worked this round. Reconsider the model with weaker
            # evidence, wipe the memory of where we have been, and if that
            # still yields nothing, reset the level - one action - so the next
            # round starts from a pristine board rather than a stuck corner.
            self.run.note("roundfail", L)
            self.m.settle(min_votes=1)
            self.visited.clear()
            self.tried_targets.clear()
            self.dead_clicks.clear()
            if rounds % 2 == 0:
                self.flail(50)
            else:
                self.obs = self.run.reset()
        if self.timed_out:
            self.run.note("TIMEOUT", self.level)

    def goal_cells(self, frame: np.ndarray, goal: set[int]) -> list[tuple[int, int]]:
        """Every pixel showing a colour that has previously ended a level."""
        mask = np.isin(frame, list(goal))
        ys, xs = np.nonzero(mask)
        return list(zip(ys.tolist(), xs.tolist()))[:512]


def play_agent(run: GradedRun, **kw) -> None:
    """Entry point matching ``graded.run_suite``'s ``agent_fn`` contract."""
    Agent(run, **kw).play()
