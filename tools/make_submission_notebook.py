"""Build the submittable notebook: the 2.14 notebook plus this session's two grafts.

WHY A GENERATOR AND NOT A HAND-EDIT
-----------------------------------
The source notebook is a 700-line JSON document with one long line per cell. Editing
it by hand is how you silently corrupt a nine-hour run. This script does exactly
three things, all mechanical and all re-runnable:

  1. loads ``arc3-duck-v12 (1).ipynb``,
  2. inserts one markdown cell + one code cell immediately after the cell that
     defines ``GRAFT_FLAGS`` (the notebook's own designated customization hook, and
     therefore the last point before ``bm.run()`` where ``bm``/the source tree are
     loaded but nothing has been played),
  3. strips stale outputs, so the local ``CalledProcessError`` traceback from a
     Windows dry-run does not travel to Kaggle looking like a real failure.

WHY THESE TWO GRAFTS AND NOT THE arc3x CODE
-------------------------------------------
The notebook imports its solver from a **read-only Kaggle dataset**
(``thtennant/taaf-kaggle-source-share-fork``). Nothing in this repo can change that
dataset, so any graft has to be expressible as inline notebook code. That excludes
the arc3x modules — they need the local twin engine and score 0.142 over 25 games
standalone — and it includes exactly the two changes below, which are a wrapper and
a string.

  * **Graft A, the level probe.** ``taaf.game.Game.finish_game`` already computes
    ``actions_per_level`` and ``base_actions_per_level`` and prints them
    (``game.py`` 636-650), but a true submission sets
    ``TAAF_MINIMAL_DIAGNOSTICS=1``, which suppresses the JSON/HTML artifacts, so
    the numbers only ever existed in a log nobody kept. This appends one JSON line
    per finished game to ``/kaggle/working/level_probe.jsonl``. It reads fields that
    the framework has already filled in and writes ~110 short lines over nine
    hours. It cannot change what the solver does.

    This is the measurement the whole plan has been blocked on: 2.14 is equally
    consistent with "level 0 on 60% of games" and "two levels on 5 games, zero on
    20", and those two need opposite fixes.

  * **Graft B, the family priors.** ``inference.agent.tool_agent._build_system_prompt``
    concatenates addendum strings; this appends one more. The text carries the two
    findings from this session that generalise past the 25 readable dev games:
    the win condition is usually a *cover* predicate and is therefore drawn on
    frame 0, and action-efficiency only costs score on levels you actually clear.
    ``benchmark.py`` runs games with ``asyncio.create_task`` in a single process
    (163-199), so patching the module attribute reaches every game.

WHAT IS DELIBERATELY NOT IN HERE
--------------------------------
  * ``arc3x/markers.py``. Its own measurement (``why_markers.py``, run today)
    scored ``source-graded rows passing: 0/1`` — on the one row with ground truth
    read from the game's source it proposed the floor tiling instead of the exit.
    Ranking candidate destinations by repeat-count promotes scenery. Graft B states
    the *prior* the census established and lets the model apply it, which is the
    part that measured out true; the detector that failed stays home.
  * The ACTION1-4 = N/S/W/E button convention. True in 90-100% of the dev games,
    and measured at **exactly 0.00** when seeded into the arc3x planner. Controls
    were never the bottleneck, an LLM finds them in a few actions anyway, and a
    third section would dilute the two that address something the model lacks.
  * Any change to ``GRAFT_FLAGS``, including ``context_window``. It is the knob most
    suspected of causing wasted actions and there is no local evidence either way.
    Changing an untested knob in the same run as two new grafts would make the
    result unattributable, which is the exact failure recorded in TOMORROW.md §2.

Run:
    .venv/Scripts/python.exe tools/make_submission_notebook.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "1.33 scored in arc agi 3 competiotn in kaggle" / "arc3-duck-v12 (1).ipynb"
OUT = REPO / "1.33 scored in arc agi 3 competiotn in kaggle" / "arc3-duck-v13-priors.ipynb"

# The anchor. Chosen because it is inside the notebook's own "## 6. Customization
# hook" cell, documented there as "the safe place for one-off experiments once the
# deployed bundle has loaded" — after the source tree is importable and `bm` is
# unpickled, before cell 7 reassigns `bm.games` and awaits `bm.run()`.
ANCHOR = "GRAFT_FLAGS = {"

MARKDOWN = """\
## 6b. Session grafts — level probe + family priors

Two independent grafts, added 2026-08-23. Each is wrapped so that **any** failure
leaves the stock path untouched, and each is reported by name below so the run log
says which ones actually installed.

**A. Level probe (inert).** `taaf.game.Game.finish_game` already computes
`actions_per_level` and `base_actions_per_level`; a true submission sets
`TAAF_MINIMAL_DIAGNOSTICS=1` and never persists them. This appends one JSON line per
finished game to `/kaggle/working/level_probe.jsonl`. It reads values the framework
has already filled in, so it cannot change what the solver does. It exists because
a score of 2.14 is equally consistent with *level 0 on most games* and *two levels
on a handful*, and those two states need opposite fixes.

**B. Family priors (behavioural).** Appends one addendum to the agent's system
prompt, carrying two findings that generalise beyond any single game:

1. *The goal is usually drawn on frame 0.* Across the readable dev games, 10 of 13
   win conditions are one predicate — every object of kind A co-located with an
   object of kind B — and kind B is on the board before the first action, as a set
   of small, identical, static objects. The agent already has each object's shape
   hash from `segmentation`; it was never told what to look for with it.
2. *Efficiency only costs score where you clear.* Level score is
   `(baseline/actions)²`, awarded only for cleared levels, so actions on a level you
   never clear are free — explore level 0, execute crisply later, and never give up
   early.

Prior 2 is measured from this notebook's own event log; prior 1 was read from the
25 dev games' source. Both are stated as defaults that evidence from the live game
overrides.
"""

CODE = '''\
# --- session grafts, 2026-08-23 -----------------------------------------------
# A: level probe (inert instrumentation).  B: family priors (system-prompt text).
# Both blanket-guarded: on ANY exception the stock 2.14 behaviour is what runs.
# Set either flag to False to ship without it.
INSTALL_LEVEL_PROBE = True
INSTALL_FAMILY_PRIORS = True

LEVEL_PROBE_PATH = WORKING_DIR / "level_probe.jsonl"


# --- Graft A: one JSON line per finished game --------------------------------
# `Game.finish_game` (taaf/game.py:596) already fills in every field read here and
# prints them at 645-650; TAAF_MINIMAL_DIAGNOSTICS=1 is why a real submission keeps
# no artifact of them. Appending is O(1) per game, ~110 lines over nine hours.
def _install_level_probe() -> str:
    import taaf.game

    game_cls = taaf.game.Game
    if getattr(game_cls.finish_game, "_level_probe", False):
        return "already installed"
    original = game_cls.finish_game

    def finish_game(self, generated_tokens: int = 0, uncached_input_tokens: int = 0) -> None:
        # finish_game is idempotent by an early return on final_score (game.py:615),
        # so it can legitimately be called twice; only the call that actually scores
        # the game should emit a row.
        run_before = getattr(self, "game_run", None)
        first_call = run_before is not None and run_before.final_score is None
        original(self, generated_tokens=generated_tokens, uncached_input_tokens=uncached_input_tokens)
        if not first_call:
            return
        try:
            run = self.game_run
            base = run.base_actions_per_level
            row = {
                "game_id": run.game_id,
                "state": run.state,
                "levels_completed": run.levels_completed,
                "number_of_levels": run.number_of_levels,
                "final_score": run.final_score,
                "actions_per_level": list(run.actions_per_level),
                # None here means the framework hit its documented silent fallback
                # (game_api.py:232) and every efficiency ratio in the run is unusable.
                "base_actions_per_level": list(base) if base is not None else None,
                "wallclock_s": round(float(run.final_wallclock_seconds or 0.0), 1),
                "turns": len(run.history),
                "generated_tokens": sum(r.generated_tokens for r in run.history) + run.final_generated_tokens,
                "note": run.solver_note,
            }
            with open(LEVEL_PROBE_PATH, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, default=str) + "\\n")
        except Exception as exc:  # noqa: BLE001 — a diagnostics write must never sink the run
            print(f"[graft] level probe row failed: {type(exc).__name__}: {exc}")

    finish_game._level_probe = True
    game_cls.finish_game = finish_game
    return f"wrapped taaf.game.Game.finish_game -> {LEVEL_PROBE_PATH}"


# --- Graft B: the two family priors ------------------------------------------
# Phrased as defaults rather than rules, and grounded in what the agent can already
# see: `segmentation` gives every object a colour+shape hash (prompts.py:43), which
# is exactly the key that groups "several identical things drawn on the board".
FAMILY_PRIORS_ADDENDUM = """

# Two priors about this family of games

These hold across the family these puzzles are drawn from, not just this one. Treat
them as strong starting defaults: evidence from THIS game always overrides them.

## 1. The goal is usually already drawn on the first frame

In most of these games the win condition is a *cover* predicate:

    every object of kind A must end up co-located with an object of kind B.

Kind B is on the board before you take a single action, and it looks like this: a
group of two or more objects that are (a) identical to each other, so same colour
and same shape and therefore the same segmentation hash, (b) small, because they
mark a place rather than being scenery or wall, and (c) static, because they do not
move when you act. Kind A is what you move or push.

So in your first turns, before experimenting: group the segmentation nodes of the
current frame by hash. Any group of 2 or more small identical shapes is a candidate
destination set, and the size of that group tells you how many things you probably
have to deliver. That reframes the game from "what does this button do" to "get
each mover onto one destination" - and it means you can often read the objective
off the board instead of inferring it from a reward you have not seen yet.

Recognise the variants:
- the destination may be a single object, not a group, so a lone small static shape
  distinct from everything else is also a candidate;
- the predicate may be at pixel level - a workspace region must be made equal to a
  reference region drawn elsewhere on the board, in which case your progress meter
  is the count of mismatched pixels between the two;
- it may be "remove every object of colour X", in which case the remaining count of
  X is your progress meter;
- it may be ordered, so covering the destinations in the wrong sequence undoes
  earlier progress.

If the first frame has no repeated small static group and no distinct lone marker,
this prior does not apply - fall back to experimenting.

## 2. Where being efficient actually pays, and where it does not

A level scores (baseline_actions / your_actions) squared, and it scores that ONLY if
you clear it. A level you never clear contributes zero no matter what you did on it,
so the actions you spent there cost you nothing.

Two consequences, and they point in opposite directions:

- Exploring is cheap on the first level, and free on any level you end up not
  clearing. Spend actions to establish the mechanics: which button moves what, what
  blocks movement, what the objective is, what an action does when nothing appears
  to change. The first level also carries the smallest weight of any level in the
  game.
- Being crisp pays on later levels. Once you know the mechanics, execute. Later
  levels carry more weight and charge every wasted action quadratically, and you
  should re-derive nothing you already established earlier in the same game - the
  mechanics do not change between levels, only the layout does.

And never stop trying because you have used a lot of actions. Exhausting your
actions on a level you cannot solve costs exactly zero, so there is no situation in
which giving up scores better than continuing to try.
"""


def _install_family_priors() -> str:
    import inference.agent.tool_agent as tool_agent

    original = tool_agent._build_system_prompt
    if getattr(original, "_family_priors", False):
        return "already installed"

    def _build_system_prompt(*, tool_output_tokens: int) -> str:
        return original(tool_output_tokens=tool_output_tokens) + FAMILY_PRIORS_ADDENDUM

    _build_system_prompt._family_priors = True
    # benchmark.py plays games with asyncio.create_task in this same process
    # (benchmark.py:163-199), and tool_agent.py:941 resolves this name from module
    # globals at call time, so rebinding the module attribute reaches every game.
    tool_agent._build_system_prompt = _build_system_prompt
    return f"appended {len(FAMILY_PRIORS_ADDENDUM)} chars to the agent system prompt"


SESSION_GRAFTS = {}
for _name, _enabled, _installer in [
    ("level_probe", INSTALL_LEVEL_PROBE, _install_level_probe),
    ("family_priors", INSTALL_FAMILY_PRIORS, _install_family_priors),
]:
    if not _enabled:
        SESSION_GRAFTS[_name] = "disabled"
        continue
    try:
        SESSION_GRAFTS[_name] = _installer()
    except Exception as exc:  # noqa: BLE001 — any graft failure falls back to stock
        SESSION_GRAFTS[_name] = f"FAILED {type(exc).__name__}: {exc}"
    print(f"[graft] {_name}: {SESSION_GRAFTS[_name]}")

# Record what this run configured, next to graft_flags.json, so the result is
# attributable — the single lesson from two runs that scored 1.36 and 10.71 on the
# same game with no record of what differed.
try:
    (WORKING_DIR / "session_grafts.json").write_text(json.dumps(SESSION_GRAFTS, indent=2, sort_keys=True) + "\\n")
except Exception as exc:  # noqa: BLE001
    print(f"[graft] could not persist session graft status: {type(exc).__name__}: {exc}")
'''


def main() -> int:
    if not SRC.is_file():
        print(f"source notebook not found: {SRC}")
        return 1
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    cells = nb["cells"]

    anchor_at = next(
        (i for i, c in enumerate(cells) if c.get("cell_type") == "code" and ANCHOR in "".join(c.get("source", []))),
        None,
    )
    if anchor_at is None:
        print(f"anchor {ANCHOR!r} not found — refusing to guess an insertion point")
        return 1

    def lines(text: str) -> list[str]:
        """nbformat stores source as a list of lines, each keeping its newline."""
        out = text.splitlines(keepends=True)
        if out and out[-1].endswith("\n"):
            out[-1] = out[-1][:-1]
        return out

    already = [i for i, c in enumerate(cells) if "INSTALL_FAMILY_PRIORS" in "".join(c.get("source", []))]
    for i in reversed(already):
        del cells[i]
    if already:
        anchor_at = next(
            i for i, c in enumerate(cells) if c.get("cell_type") == "code" and ANCHOR in "".join(c.get("source", []))
        )

    cells.insert(anchor_at + 1, {"cell_type": "markdown", "metadata": {}, "source": lines(MARKDOWN)})
    cells.insert(
        anchor_at + 2,
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines(CODE)},
    )

    # Strip stale outputs. The dry-run on this Windows box left a CalledProcessError
    # traceback in the pip cell, which on Kaggle would read as a real failure.
    stripped = 0
    for cell in cells:
        if cell.get("cell_type") == "code":
            stripped += 1 if cell.get("outputs") else 0
            cell["outputs"] = []
            cell["execution_count"] = None

    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"anchor cell index      : {anchor_at}")
    print(f"cells before / after   : {len(cells) - 2} / {len(cells)}")
    print(f"stale outputs stripped : {stripped}")
    print(f"wrote                  : {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
