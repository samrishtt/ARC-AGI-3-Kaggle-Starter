"""Go-Explore search core for ARC-AGI-3 - general, no per-game knowledge.

WHY THIS EXISTS
---------------
The incumbent agent spent ~17.6 seconds and ~1,700 LLM tokens per *single*
engine action. Scoring is ``min(115, (baseline/actions)^2 * 100)`` per
*completed* level and 0 for an uncompleted one, so an agent that cannot afford
enough actions to finish a level scores nothing at all - which is exactly what
happened (781 actions on sk48 level 0, baseline 61, never completed).

The engine, however, is a pure-Python in-process object we can ``deepcopy`` and
step at ~6,800 actions/sec for free (``verify_twin.py`` proves this on 25/25
games). So we decouple "actions taken" from "LLM calls": search millions of
actions locally, then replay one short winning line to the graded environment.

THE ALGORITHM (four general ideas, no game-specific logic anywhere)
------------------------------------------------------------------
1. ARCHIVE / GO-EXPLORE. Keep a map from "situation" (hash of the 64x64 frame
   plus the level index) to the *shortest known action plan* that reaches it.
   Repeatedly: pick a promising archived situation, restore it, explore from
   there, and file away every new situation found. This is what beats sparse
   reward - no reward shaping, no domain knowledge, just "have I ever seen this
   frame before?".

2. ENGINE-EXACT ACTION SETS. ``_get_valid_actions()`` hands us the legal moves
   *including* concrete ACTION6 click coordinates, collapsing a 4096-wide
   coordinate space to a branching factor of 2-13 on most games. For the two
   games with hundreds of legal clicks we group clicks by connected same-colour
   region and sample one representative per region - still purely frame-derived,
   still general.

3. STICKY ROLLOUTS + NO-OP PRUNING. Grid games need the same action repeated to
   cross a room, so the rollout policy repeats its previous action with high
   probability. Any action that leaves the frame *and* the legal action set
   unchanged is recorded as a no-op for that situation and never retried there.

4. PLAN COMPRESSION (this is where the score comes from). A random walk that
   finishes a level takes hundreds of actions; scoring is quadratic in that
   number. So after finding *any* solution we shrink it: splice out loops
   (revisited frames) and greedily drop action windows, re-verifying by replay
   after every edit. Verification makes it sound even when the frame does not
   capture the full hidden state. 400 actions -> near-baseline is routine, and
   (61/400)^2*100 = 2.3 versus (61/70)^2*100 = 76.

Run:
    .venv/Scripts/python.exe arc3x/explore.py --game sk48 --budget 60
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from arc3x.twin import Act, Obs, Twin, default_env_dir

# ---------------------------------------------------------------------------
# scoring (mirrors taaf/game.py:_compute_final_score)
# ---------------------------------------------------------------------------


def level_score(baseline: int, actions: int) -> float:
    """RHAE for one completed level. Capped at 115 = 1.15x human."""
    if actions <= 0:
        return 0.0
    return min(115.0, (baseline / actions) ** 2 * 100.0)


def game_score(baselines: Sequence[int], actions_per_level: Sequence[int]) -> float:
    """Weighted average with 1-indexed level weights, capped by depth reached.

    ``actions_per_level[i] <= 0`` means level i was not completed (scores 0).
    """
    n = len(baselines)
    weights = [i + 1 for i in range(n)]
    total_w = sum(weights)
    num = 0.0
    max_w = 0.0
    for i in range(n):
        used = actions_per_level[i] if i < len(actions_per_level) else 0
        s = level_score(baselines[i], used) if used > 0 else 0.0
        if s > 0:
            max_w += weights[i]
        num += weights[i] * s
    if total_w == 0:
        return 0.0
    return min(num / total_w, max_w / total_w * 100.0)


# ---------------------------------------------------------------------------
# frame-derived click reduction (general: uses only the pixels)
# ---------------------------------------------------------------------------


def components(frame: np.ndarray) -> np.ndarray:
    """Label 4-connected same-colour regions of the frame.

    Used only to shrink very large click sets. Nothing about any specific game
    is assumed - two pixels of the same colour that touch are one object.
    """
    h, w = frame.shape
    lab = np.full((h, w), -1, dtype=np.int32)
    nxt = 0
    for sy in range(h):
        for sx in range(w):
            if lab[sy, sx] != -1:
                continue
            col = frame[sy, sx]
            q = deque([(sy, sx)])
            lab[sy, sx] = nxt
            while q:
                y, x = q.popleft()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and lab[ny, nx] == -1 and frame[ny, nx] == col:
                        lab[ny, nx] = nxt
                        q.append((ny, nx))
            nxt += 1
    return lab


CLICK_GROUP_THRESHOLD = 32


def reduce_clicks(frame: np.ndarray, acts: tuple[Act, ...]) -> tuple[Act, ...]:
    """Keep one representative click per connected region; pass others through.

    Only kicks in when the engine offers a lot of clicks (r11l: 256,
    su15: 224). Below the threshold the exact engine set is already small
    enough to enumerate.
    """
    clicks = [a for a in acts if a.is_click]
    if len(clicks) <= CLICK_GROUP_THRESHOLD:
        return acts
    others = [a for a in acts if not a.is_click]
    try:
        lab = components(frame)
    except Exception:
        return acts
    h, w = frame.shape
    best: dict[int, Act] = {}
    for a in clicks:
        if not (0 <= a.y < h and 0 <= a.x < w):
            best[-1 - len(best)] = a
            continue
        cid = int(lab[a.y, a.x])
        best.setdefault(cid, a)
    return tuple(others + list(best.values()))


# ---------------------------------------------------------------------------
# archive
# ---------------------------------------------------------------------------


@dataclass
class Node:
    """One archived situation: how to get there, and how to explore from it."""

    key: bytes
    plan: tuple[Act, ...]
    level: int
    valid: tuple[Act, ...]
    visits: int = 0
    snap: Any = None  # cached engine snapshot; may be dropped to save memory
    noop: set[Act] = field(default_factory=set)
    dead: bool = False

    @property
    def depth(self) -> int:
        return len(self.plan)

    @property
    def weight(self) -> float:
        """Go-Explore selection weight: prefer unvisited and shallow cells."""
        return 1.0 / ((self.visits + 1) ** 0.5 * (self.depth + 1) ** 0.25)


@dataclass
class LevelResult:
    level: int
    plan: tuple[Act, ...] | None
    raw_len: int = 0
    cells: int = 0
    steps: int = 0
    seconds: float = 0.0
    won_game: bool = False


class Explorer:
    """Solve one level at a time from a given starting engine state."""

    def __init__(
        self,
        twin: Twin,
        *,
        seed: int = 0,
        sticky: float = 0.7,
        rollout: int = 48,
        max_depth: int = 1500,
        snap_cap: int = 4000,
        verbose: bool = True,
    ):
        self.twin = twin
        self.rng = np.random.default_rng(seed)
        self.sticky = sticky
        self.rollout = rollout
        self.max_depth = max_depth
        self.snap_cap = snap_cap
        self.verbose = verbose
        self.steps = 0

    # -- plumbing ---------------------------------------------------------

    def _restore(self, root: Any, node: Node) -> Any:
        """Get a fresh engine object positioned at ``node``.

        Cached snapshot if we kept one, otherwise deterministic replay from the
        level root. Replay is exact because the games are deterministic
        (verified 25/25), and at ~6,800 steps/sec a 300-action replay is ~44 ms.
        """
        if node.snap is not None:
            return copy.deepcopy(node.snap)
        g = copy.deepcopy(root)
        for a in node.plan:
            Twin.step_game(g, a)
            self.steps += 1
        return g

    def _select(self, nodes: list[Node], k: int = 24) -> Node:
        """Tournament selection - O(k), not O(len(archive)) per rollout."""
        best: Node | None = None
        bw = -1.0
        n = len(nodes)
        for _ in range(min(k, n)):
            c = nodes[int(self.rng.integers(n))]
            if c.dead:
                continue
            w = c.weight
            if w > bw:
                bw, best = w, c
        return best or nodes[0]

    # -- the search -------------------------------------------------------

    def solve_level(
        self, root: Any, start_level: int, baseline: int, budget_s: float
    ) -> LevelResult:
        """Find *some* action sequence from ``root`` that completes one level."""
        t0 = time.perf_counter()
        probe = copy.deepcopy(root)
        valid0 = Twin.valid_actions(probe)
        frame0 = self.twin.current().frame if start_level == 0 else None

        # Seed the archive with the root situation.
        seed_obs = Obs(
            frame=frame0 if frame0 is not None else np.zeros((64, 64), dtype=np.int8),
            level=start_level,
            score=start_level,
            state=None,
            valid=valid0,
        )
        root_node = Node(
            key=b"ROOT", plan=(), level=start_level, valid=valid0, snap=copy.deepcopy(root)
        )
        archive: dict[bytes, Node] = {root_node.key: root_node}
        order: list[Node] = [root_node]
        snaps = 1

        best_plan: tuple[Act, ...] | None = None
        won_game = False
        rollout_len = self.rollout
        barren = 0

        while time.perf_counter() - t0 < budget_s and best_plan is None:
            node = self._select(order)
            node.visits += 1
            g = self._restore(root, node)
            plan = list(node.plan)
            cur_valid = node.valid or valid0
            cur_frame = None
            prev: Act | None = None
            local_key = node.key
            local_noop = node.noop
            new_cells = 0
            grouped = cur_valid
            refresh = 0

            for _ in range(rollout_len):
                if len(plan) >= self.max_depth:
                    break
                if not cur_valid:
                    break
                if refresh <= 0 and cur_frame is not None:
                    grouped = reduce_clicks(cur_frame, cur_valid)
                    refresh = 8
                elif refresh <= 0:
                    grouped = cur_valid
                    refresh = 8
                refresh -= 1

                pool = [a for a in grouped if a not in local_noop] or list(grouped)
                if prev is not None and prev in pool and self.rng.random() < self.sticky:
                    a = prev
                else:
                    a = pool[int(self.rng.integers(len(pool)))]

                obs = Twin.step_game(g, a)
                self.steps += 1
                plan.append(a)
                prev = a

                if obs.won:
                    best_plan = tuple(plan)
                    won_game = True
                    break
                if obs.level > start_level:
                    best_plan = tuple(plan)
                    break
                if obs.game_over:
                    # Dead branch: file it as dead so we never restore into it.
                    dk = obs.key()
                    if dk not in archive:
                        archive[dk] = Node(dk, tuple(plan), obs.level, (), dead=True)
                    break

                k = obs.key()
                if k == local_key:
                    # Same frame and (usually) same options -> a no-op here.
                    local_noop.add(a)
                    plan.pop()
                    prev = None
                    continue
                local_noop = set()
                local_key = k
                cur_valid = obs.valid
                cur_frame = obs.frame

                old = archive.get(k)
                if old is None:
                    nd = Node(k, tuple(plan), obs.level, obs.valid)
                    if snaps < self.snap_cap:
                        nd.snap = copy.deepcopy(g)
                        snaps += 1
                    archive[k] = nd
                    order.append(nd)
                    new_cells += 1
                elif len(plan) < old.depth and not old.dead:
                    # Cheaper route to a known situation - keep the short one.
                    old.plan = tuple(plan)
                    old.valid = obs.valid
                    old.snap = copy.deepcopy(g) if old.snap is not None else None
                    local_noop = old.noop

            node.noop |= local_noop if node.key == local_key else set()

            # Adaptive rollout length: if nothing new turns up, look further.
            if new_cells == 0:
                barren += 1
                if barren >= 12:
                    rollout_len = min(int(rollout_len * 1.5) + 8, 400)
                    barren = 0
            else:
                barren = 0

        return LevelResult(
            level=start_level,
            plan=best_plan,
            raw_len=len(best_plan) if best_plan else 0,
            cells=len(archive),
            steps=self.steps,
            seconds=time.perf_counter() - t0,
            won_game=won_game,
        )

    # -- compression ------------------------------------------------------

    def _reaches(self, root: Any, plan: Sequence[Act], target_level: int) -> bool:
        """Does ``plan`` still complete the level? Verified by real replay."""
        g = copy.deepcopy(root)
        for a in plan:
            obs = Twin.step_game(g, a)
            self.steps += 1
            if obs.game_over:
                return False
            if obs.level >= target_level or obs.won:
                return True
        return False

    def _keys_along(self, root: Any, plan: Sequence[Act]) -> list[bytes]:
        g = copy.deepcopy(root)
        keys: list[bytes] = [b"START"]
        for a in plan:
            obs = Twin.step_game(g, a)
            self.steps += 1
            keys.append(obs.key())
            if obs.terminal:
                break
        return keys

    def compress(
        self, root: Any, plan: Sequence[Act], target_level: int, budget_s: float = 20.0
    ) -> tuple[Act, ...]:
        """Shrink a working plan. Every edit is verified, so it stays correct.

        Two general passes, both purely mechanical:
          * loop splice - if the same frame appears twice, cut what is between.
          * window drop - try deleting runs of 16/8/4/2/1 actions.
        """
        t0 = time.perf_counter()
        cur = list(plan)

        # pass 1: loop removal, biggest loops first
        improved = True
        while improved and time.perf_counter() - t0 < budget_s:
            improved = False
            keys = self._keys_along(root, cur)
            first: dict[bytes, int] = {}
            cuts: list[tuple[int, int]] = []
            for i, k in enumerate(keys):
                if k in first:
                    cuts.append((first[k], i))
                else:
                    first[k] = i
            cuts.sort(key=lambda c: c[1] - c[0], reverse=True)
            for i, j in cuts:
                if j - i <= 0 or j > len(cur):
                    continue
                cand = cur[:i] + cur[j:]
                if len(cand) >= len(cur):
                    continue
                if self._reaches(root, cand, target_level):
                    cur = cand
                    improved = True
                    break

        # pass 2: window drop
        for win in (16, 8, 4, 2, 1):
            i = 0
            while i + win <= len(cur) and time.perf_counter() - t0 < budget_s:
                cand = cur[:i] + cur[i + win :]
                if self._reaches(root, cand, target_level):
                    cur = cand
                else:
                    i += 1

        return tuple(cur)


# ---------------------------------------------------------------------------
# whole-game driver
# ---------------------------------------------------------------------------


@dataclass
class GameSolution:
    game_id: str
    plan: list[Act]
    actions_per_level: list[int]
    baselines: list[int]
    levels_solved: int
    est_score: float
    steps: int
    seconds: float


def solve_game(
    game_id: str,
    *,
    env_dir: Path | None = None,
    budget_s: float = 120.0,
    per_level_cap: float | None = None,
    seed: int = 0,
    verbose: bool = True,
    max_levels: int | None = None,
) -> GameSolution:
    """Search a whole game level by level; return one concatenated plan."""
    t0 = time.perf_counter()
    twin = Twin(game_id, env_dir)
    baselines = twin.baselines or [100] * twin.n_levels
    n_levels = max_levels or twin.n_levels

    root = twin.snapshot()
    # Every graded run starts with RESET; do the same here so the plan we hand
    # back is replayable verbatim from a fresh game.
    ex = Explorer(twin, seed=seed, verbose=verbose)
    Twin.step_game(root, Act(0))

    full: list[Act] = []
    per_level: list[int] = []
    level = 0
    while level < n_levels and time.perf_counter() - t0 < budget_s:
        left = budget_s - (time.perf_counter() - t0)
        share = min(left, per_level_cap or left)
        base = baselines[level] if level < len(baselines) else 100
        res = ex.solve_level(root, level, base, share * 0.75)
        if res.plan is None:
            if verbose:
                print(
                    f"  L{level}: NOT SOLVED  cells={res.cells:,} "
                    f"steps={ex.steps:,} {res.seconds:.1f}s"
                )
            break
        tight = ex.compress(root, res.plan, level + 1, budget_s=min(share * 0.25, 30.0))
        sc = level_score(base, len(tight))
        if verbose:
            print(
                f"  L{level}: solved raw={res.raw_len:4d} -> {len(tight):4d} "
                f"(baseline {base:3d})  score {sc:6.1f}  cells={res.cells:,} "
                f"{res.seconds:.1f}s"
            )
        full.extend(tight)
        per_level.append(len(tight))
        # Advance the root to the state right after this level completes.
        g = copy.deepcopy(root)
        for a in tight:
            Twin.step_game(g, a)
        root = g
        ex.steps += len(tight)
        level += 1
        if res.won_game:
            break

    est = game_score(baselines, per_level)
    return GameSolution(
        game_id=game_id,
        plan=full,
        actions_per_level=per_level,
        baselines=list(baselines),
        levels_solved=len(per_level),
        est_score=est,
        steps=ex.steps,
        seconds=time.perf_counter() - t0,
    )


def discover_games(env_dir: Path) -> list[str]:
    out: list[str] = []
    for meta in sorted(env_dir.rglob("*/*/metadata.json")):
        try:
            out.append(json.loads(meta.read_text(encoding="utf-8"))["game_id"])
        except Exception:
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="", help="game id prefix, or blank for all")
    ap.add_argument("--budget", type=float, default=120.0, help="seconds per game")
    ap.add_argument("--levels", type=int, default=0, help="stop after N levels (0=all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="write plans as json")
    args = ap.parse_args()

    env_dir = default_env_dir()
    ids = discover_games(env_dir)
    if args.game:
        ids = [g for g in ids if g.startswith(args.game)]
    if not ids:
        print("no matching games")
        return 1

    print(f"env_dir: {env_dir}\ngames:   {len(ids)}  budget {args.budget:.0f}s each\n")
    results: list[GameSolution] = []
    for gid in ids:
        print(f"{gid}")
        sol = solve_game(
            gid,
            env_dir=env_dir,
            budget_s=args.budget,
            seed=args.seed,
            max_levels=args.levels or None,
        )
        results.append(sol)
        print(
            f"  => {sol.levels_solved}/{len(sol.baselines)} levels, "
            f"est game score {sol.est_score:.2f}, {sol.steps:,} sim steps, "
            f"{sol.seconds:.1f}s\n"
        )

    total = sum(r.est_score for r in results) / len(results)
    print(f"MEAN ESTIMATED SCORE over {len(results)} games: {total:.3f}")
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    r.game_id: {
                        "plan": [[a.aid, a.x, a.y] for a in r.plan],
                        "actions_per_level": r.actions_per_level,
                        "baselines": r.baselines,
                        "est_score": r.est_score,
                    }
                    for r in results
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
