"""Verify the generated notebook before a nine-hour run depends on it.

Four checks, each of which has a plausible way to fail that a human read-through
would miss:

  1. **Every code cell compiles.** The graft cell is ~200 lines of Python embedded in
     a Python string literal inside the generator, which means every backslash and
     every triple-quote in it had to survive two levels of escaping. A syntax error
     here would surface nine hours into a submission. Cell 7 uses top-level ``await``,
     so compilation needs ``PyCF_ALLOW_TOP_LEVEL_AWAIT``.
  2. **Cell order.** The graft has to land after ``GRAFT_FLAGS`` (source importable,
     ``bm`` unpickled) and before ``bm.run`` (nothing played yet). If it landed after
     the run cell it would install into a finished process and appear to work.
  3. **The addendum survived escaping intact.** Executed in isolation and measured,
     because a mangled ``\\n`` would silently become a literal backslash-n in the
     model's system prompt.
  4. **No outputs.** A stale traceback shipped to Kaggle reads as a real failure.

Run:
    .venv/Scripts/python.exe tools/verify_submission_notebook.py
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NB = REPO / "1.33 scored in arc agi 3 competiotn in kaggle" / "arc3-duck-v13-priors.ipynb"


def main() -> int:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    code = [(i, "".join(c["source"])) for i, c in enumerate(cells) if c.get("cell_type") == "code"]
    problems: list[str] = []

    # 1. compile everything
    for i, src in code:
        try:
            compile(src, f"cell{i}", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        except SyntaxError as exc:
            problems.append(f"cell {i} does not compile: {exc}")
    print(f"[1] compiled {len(code)} code cells")

    # 2. ordering
    def find(needle: str) -> int | None:
        return next((i for i, src in code if needle in src), None)

    flags_at, graft_at, run_at = find("GRAFT_FLAGS = {"), find("INSTALL_FAMILY_PRIORS"), find("await bm.run(")
    print(f"[2] GRAFT_FLAGS@{flags_at}  graft@{graft_at}  bm.run@{run_at}")
    if None in (flags_at, graft_at, run_at):
        problems.append("could not locate all three landmark cells")
    elif not flags_at < graft_at < run_at:
        problems.append(f"graft cell out of order: {flags_at} < {graft_at} < {run_at} is false")

    # 3. the addendum text, executed in isolation
    graft_src = dict(code)[graft_at] if graft_at is not None else ""
    tree = ast.parse(graft_src)
    addendum = None
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "FAMILY_PRIORS_ADDENDUM"
        ):
            addendum = ast.literal_eval(node.value)
    if not isinstance(addendum, str):
        problems.append("FAMILY_PRIORS_ADDENDUM is not a plain string literal")
    else:
        bad = [r"\n" in addendum, "\\\\" in addendum]
        print(
            f"[3] addendum {len(addendum)} chars, {len(addendum.splitlines())} lines, "
            f"~{len(addendum) // 4} tokens, literal-backslash-n={bad[0]}"
        )
        if any(bad):
            problems.append("addendum contains escaped sequences that should have been resolved")
        # Needles must be written exactly as they appear, markdown emphasis included.
        for want in ("*cover* predicate", "segmentation hash", "squared", "costs exactly zero"):
            if want not in addendum:
                problems.append(f"addendum lost the phrase {want!r}")

    # the JSONL newline in the probe writer is the one escape most likely to break
    if graft_at is not None:
        writer = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == "\n"]
        print(f"[3b] real newline constants in graft cell: {len(writer)} (need >=1 for the JSONL writer)")
        if not writer:
            problems.append("the JSONL writer's newline did not survive escaping")

    # 4. outputs
    dirty = [i for i, c in enumerate(cells) if c.get("cell_type") == "code" and c.get("outputs")]
    print(f"[4] code cells carrying outputs: {len(dirty)}")
    if dirty:
        problems.append(f"cells still carry outputs: {dirty}")

    print()
    if problems:
        for p in problems:
            print(f"FAIL  {p}")
        return 1
    print(f"OK  {NB.name} is ready to upload ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
