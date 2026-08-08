"""Script to create arc3-duck-v12-optimized.ipynb for Experiment 11 (Beat 1.33).

Exp 11: Level 2 (find_path, find_objects, death_memory) + Level 3 (multi-candidate search architecture)
+ Post-death history cleanup + Banking + Schema Helpers + 57K context.
"""

import json
from pathlib import Path

CELL_13_CODE = '''# Cell 13: Beat 1.33 Baseline — Enhanced Tools + Smarter Search Architecture
# All changes inline — zero dataset uploads required!

import json
from pathlib import Path
import inference.agent.tool_agent as ta
import taaf_grafts.schema_helpers as sh
from inference.agent.tool_agent import ToolAgent
from taaf_grafts.composite import install

# ============================================================
# 1. DEFINE LEVEL 2 ENHANCED SANDBOX HELPERS (Pure Python)
# ============================================================

EXTRA_HELPERS_SOURCE = """
def find_path(grid_or_frame, start, goal, walkable=None):
    \"\"\"BFS shortest path on the grid. Returns list of (row, col) steps.
    walkable: set/list of color values that are traversable. If None, all cells.
    start/goal: (row, col) tuples.
    \"\"\"
    grid = _sh_as_grid(grid_or_frame)
    if grid is None:
        return []
    rows = len(grid)
    if rows == 0:
        return []
    cols = max(len(r) for r in grid) if grid else 0
    sr, sc = int(start[0]), int(start[1])
    gr, gc = int(goal[0]), int(goal[1])
    if not (0 <= sr < rows and 0 <= sc < cols and 0 <= gr < rows and 0 <= gc < cols):
        return []
    visited = [[False] * cols for _ in range(rows)]
    parent = [[None] * cols for _ in range(rows)]
    visited[sr][sc] = True
    queue = [(sr, sc)]
    found = False
    idx = 0
    while idx < len(queue):
        r, c = queue[idx]
        idx += 1
        if r == gr and c == gc:
            found = True
            break
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                cell_val = grid[nr][nc] if nc < len(grid[nr]) else None
                if walkable is None or cell_val in walkable:
                    visited[nr][nc] = True
                    parent[nr][nc] = (r, c)
                    queue.append((nr, nc))
    if not found:
        return []
    path = []
    cr, cc = gr, gc
    while not (cr == sr and cc == sc):
        path.append((cr, cc))
        p = parent[cr][cc]
        if p is None:
            return []
        cr, cc = p
    path.append((sr, sc))
    path.reverse()
    return path


def find_objects(grid_or_frame, ignore_colors=None):
    \"\"\"Find connected objects with center-of-mass coordinates.
    Returns [{"color", "size", "bbox", "center": (row, col), "cells"}].
    ignore_colors: color(s) to skip (e.g. 0 for background).
    \"\"\"
    comps = connected_components(grid_or_frame)
    if ignore_colors is not None:
        if isinstance(ignore_colors, int):
            ignore = {ignore_colors}
        else:
            ignore = set(ignore_colors)
    else:
        ignore = set()
    result = []
    for comp in comps:
        if comp["color"] in ignore:
            continue
        cells = comp["cells"]
        if not cells:
            continue
        center_r = sum(c[0] for c in cells) // len(cells)
        center_c = sum(c[1] for c in cells) // len(cells)
        obj = dict(comp)
        obj["center"] = (center_r, center_c)
        result.append(obj)
    return result


def death_memory(trans=None):
    \"\"\"Scan transitions for game_over events. Returns death awareness summary.
    Returns {"deaths": int, "last_death_action": str, "actions_since_death": int, "is_post_death": bool}.
    \"\"\"
    if trans is None:
        try:
            trans = _sh_sandbox_transitions()
        except Exception:
            return {"deaths": 0, "last_death_action": "", "actions_since_death": 0, "is_post_death": False}
    if not isinstance(trans, (list, tuple)):
        return {"deaths": 0, "last_death_action": "", "actions_since_death": 0, "is_post_death": False}
    deaths = 0
    last_death_idx = -1
    last_death_action = ""
    for i, t in enumerate(trans):
        res = _sh_get(t, "result")
        if isinstance(res, dict) and res.get("game_over"):
            deaths += 1
            last_death_idx = i
            last_death_action = str(_sh_get(t, "action", ""))
    total = len(trans)
    actions_since = total - last_death_idx - 1 if last_death_idx >= 0 else total
    is_post = last_death_idx >= 0 and actions_since < 20
    return {
        "deaths": deaths,
        "last_death_action": last_death_action,
        "actions_since_death": actions_since,
        "is_post_death": is_post,
    }
"""

# Extend schema_helpers SANDBOX_HELPERS_PRELUDE with new Level 2 functions
if sh.SANDBOX_HELPERS_PRELUDE:
    sh.SANDBOX_HELPERS_PRELUDE += "\\n" + EXTRA_HELPERS_SOURCE

# Update prompt discovery note with new tools + Level 3 search pattern guidance
sh.HELPERS_PROMPT_NOTE = (
    "PYTHON HELPERS preloaded: grid_diff(a,b), connected_components(grid, colors=None), "
    "action_effect_summary(before,after), recent_history(n), "
    "find_path(grid, start, goal, walkable=None), "
    "find_objects(grid, ignore_colors=None), death_memory(). "
    "RECOMMENDED SEARCH PATTERN: (1) Call find_objects(current_frame, ignore_colors={0}) to identify targets. "
    "(2) Call find_path(...) to compute exact click sequences. "
    "(3) Check death_memory() for post-death state awareness. "
    "(4) Execute the best candidate via action()."
)

# ============================================================
# 2. MONKEY-PATCH POST-DEATH HISTORY CLEANUP (Fixes m0r0 0.00 score)
# ============================================================

_original_update_knowledge = ToolAgent._update_summarized_knowledge_from_step_summary

def _enhanced_update_knowledge(self):
    _original_update_knowledge(self)
    summary = self._last_step_summary
    if summary and summary.get("game_over"):
        # After death, trim chat history to last 4 messages to remove stale GAME_OVER confusion
        if hasattr(self, "_history_messages") and len(self._history_messages) > 4:
            self._history_messages = self._history_messages[-4:]

ToolAgent._update_summarized_knowledge_from_step_summary = _enhanced_update_knowledge

# ============================================================
# 3. INSTALL BASE GRAFT STACK (1.33 Baseline + Banking)
# ============================================================

try:
    install(bm, flags={
        "efficiency": True,
        "retry_guard": True,
        "shortcircuit": True,
        "context_window": 57344,
        "banking": True,
        "schema_helpers": True,
    })
    print("[beat_133_architecture] Level 2 tools + Level 3 search + Banking + 57K context ARMED!")
except Exception as exc:
    print(f"[taaf_grafts] cell-12 graft failed, running stock: {type(exc).__name__}: {exc}")
'''

def generate_exp11_notebook():
    base_nb_path = Path("1.33 scored in arc agi 3 competiotn in kaggle/arc3-duck-v12 (1).ipynb")
    out_nb_path = Path("1.33 scored in arc agi 3 competiotn in kaggle/arc3-duck-v12-optimized.ipynb")

    with open(base_nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    graft_cell_index = 13
    nb["cells"][graft_cell_index]["source"] = [CELL_13_CODE]

    with open(out_nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"Successfully generated {out_nb_path} for Experiment 11 (Beat 1.33)")

if __name__ == "__main__":
    generate_exp11_notebook()
