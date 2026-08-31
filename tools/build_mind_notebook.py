"""Build a Kaggle notebook with the ARC3 human-mind pilot armed.

This is intentionally a separate experimental notebook. It starts from the
known v17 winning-model notebook, embeds the local mind stack, and installs the
pilot immediately before the benchmark run. The pilot is fail-open: when it
cannot justify a batch, the original language-model analyzer is called.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = REPO / "1.33 scored in arc agi 3 competiotn in kaggle"
DEFAULT_SOURCE = NOTEBOOK_DIR / "arc3-duck-v17-winning-model.ipynb"
DEFAULT_OUTPUT = NOTEBOOK_DIR / "arc3-duck-v19-level-solver.ipynb"
# Keep this in dependency order: notebook cells run top-to-bottom and the pilot
# imports Progress at module import time.  Omitting it lets the notebook build
# and parse but fails only when Kaggle imports ``arc3x.pilot``.
MODULES = ("percept", "mind", "progress", "mindgraft", "clicks", "pilot", "autopilot")


def source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text,
    }


ARM = '''# ARC3 human-mind level solver (v19 experimental)
#
# The pilot reads only the runtime state and valid action names already supplied
# to the analyzer. It batches only model-backed actions; otherwise the wrapped
# stock analyzer remains responsible for the turn.
import os

os.environ["ARC3X_PILOT"] = "1"
from arc3x.autopilot import arm

if arm():
    print("arc3x human-mind pilot armed; stock analyzer remains the fallback")
else:
    print("arc3x human-mind pilot was not armed; continuing with stock analyzer")
'''


NOTE = """## 6c. ARC3 human-mind level solver (experimental)

This build embeds the history-only mechanics learner, imagination model, cell
sense, online click-semantics model, and fail-open acting wrapper. The pilot may
batch a short route or a learned click rule; otherwise the original model receives
the turn. This is an experiment, not a claim of a solved competition: only a
Kaggle run can measure the remote gateway result.
"""


def build(source_path: Path, output_path: Path) -> None:
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    cells = list(notebook.get("cells", []))
    if any("arc3x human-mind pilot" in source(cell) for cell in cells):
        raise SystemExit(f"human-mind cells already present in {source_path}")

    run_index = next(
        (i for i, cell in enumerate(cells) if "## 7. Run the benchmark" in source(cell)),
        None,
    )
    if run_index is None:
        raise SystemExit("could not locate the benchmark run cell")

    injected = [
        markdown("## 6c. Embedded human-mind modules"),
        # ``%%writefile`` will not create missing parents.  The base notebook
        # creates /kaggle/working but does not own an arc3x package directory.
        code("from pathlib import Path\nPath('/kaggle/working/arc3x').mkdir(parents=True, exist_ok=True)"),
    ]
    for name in MODULES:
        body = (REPO / "arc3x" / f"{name}.py").read_text(encoding="utf-8")
        injected.append(code(f"%%writefile /kaggle/working/arc3x/{name}.py\n{body}"))
    injected.extend([markdown(NOTE), code(ARM)])
    cells[run_index:run_index] = injected
    notebook["cells"] = cells

    for name in MODULES:
        ast.parse((REPO / "arc3x" / f"{name}.py").read_text(encoding="utf-8"))
    arm_index = next(i for i, cell in enumerate(cells) if source(cell) == ARM)
    if not any("## 7. Run the benchmark" in source(cell) for cell in cells[arm_index + 1 :]):
        raise SystemExit("pilot arm cell is not before the benchmark run")

    output_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"wrote {output_path} ({len(cells)} cells, {output_path.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.src, args.out)


if __name__ == "__main__":
    main()
