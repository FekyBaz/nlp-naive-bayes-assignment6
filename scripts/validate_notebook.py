#!/usr/bin/env python
"""Validate an executed notebook for the Naive Bayes assignment.

The checks are intentionally explicit: every code cell must have run, no cell
may contain an error output, the required figures must be rendered, and the
reported metrics must match the verified 68/17 split.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


REQUIRED_TEXT = [
    "Final dataset shape: (85, 2)",
    "Raw rows: 85",
    "Rows with missing text/label dropped: 0",
    "Rows with empty text/label dropped: 0",
    "Rows after preprocessing: 85",
    "Training rows: 68",
    "Test rows: 17",
    "Vocabulary size: 391",
    "X_train shape: (68, 391)",
    "X_test shape: (17, 391)",
    "Unseen test tokens ignored: 69",
    "Test documents with no in-vocabulary tokens: 0",
    "Train predictions matching: 100.00%",
    "Test predictions matching: 100.00%",
    "The two models made identical predictions on the test set.",
    "Manual NB audit (training set):",
    "Entertainment 0.2206 182 573",
    "Politics 0.2500 200 591",
    "Sports 0.1765 152 543",
    "Technology 0.3529 271 662",
    "Manual Naive Bayes",
    "scikit-learn MultinomialNB",
    "Macro F1",
    "Weighted F1",
    "Entertainment 1.00 0.75 0.86 4",
    "Politics 0.80 1.00 0.89 4",
    "Sports 1.00 0.67 0.80 3",
    "Technology 0.71 0.83 0.77 6",
]


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebook",
        nargs="?",
        default="assignment_1_executed.ipynb",
        help="path to the executed notebook",
    )
    args = parser.parse_args()

    path = Path(args.notebook)
    if not path.is_file():
        fail(f"notebook not found: {path}")

    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"notebook is not valid JSON: {exc}")

    cells = notebook.get("cells", [])
    if not cells:
        fail("notebook has no cells")

    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    if len(code_cells) < 10:
        fail(f"expected at least 10 code cells, found {len(code_cells)}")

    execution_counts = [cell.get("execution_count") for cell in code_cells]
    if any(count is None for count in execution_counts):
        fail("one or more code cells have no execution_count")
    if execution_counts != list(range(1, len(code_cells) + 1)):
        fail(f"execution counts are not sequential: {execution_counts}")

    error_outputs = []
    text_parts = []
    image_count = 0

    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            output_type = output.get("output_type")
            if output_type == "error":
                error_outputs.append(
                    (index, output.get("ename"), output.get("evalue"))
                )
            elif output_type == "stream":
                text_parts.append("".join(output.get("text", [])))
            elif output_type in {"display_data", "execute_result"}:
                data = output.get("data", {})
                image_count += sum(
                    1 for mime in data if mime.startswith("image/")
                )
                for mime in ("text/plain", "text/html"):
                    if mime in data:
                        value = data[mime]
                        text_parts.append(
                            "".join(value) if isinstance(value, list) else str(value)
                        )

    if error_outputs:
        for index, ename, evalue in error_outputs:
            print(f"  cell {index}: {ename}: {evalue}", file=sys.stderr)
        fail(f"{len(error_outputs)} error output(s) found")

    normalized = re.sub(r"\s+", " ", " ".join(text_parts))

    missing = [snippet for snippet in REQUIRED_TEXT if snippet not in normalized]
    if missing:
        for snippet in missing:
            print(f"  missing: {snippet!r}", file=sys.stderr)
        fail(f"{len(missing)} required result(s) missing")

    if image_count < 6:
        fail(f"expected at least 6 rendered figures, found {image_count}")

    print(f"PASS: {path}")
    print(f"  code cells executed: {len(code_cells)}")
    print(f"  error outputs: 0")
    print(f"  rendered figures: {image_count}")
    print(f"  required result checks: {len(REQUIRED_TEXT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
