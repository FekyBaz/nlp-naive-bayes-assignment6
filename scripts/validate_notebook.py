#!/usr/bin/env python
"""Independently validate the executed notebook, report, and README.

The validator does not trust the notebook's printed strings. It recomputes the
preprocessing, split, vocabulary, Bag-of-Words matrices, from-scratch Naive
Bayes model, and scikit-learn baseline from ``synthetic_text_data.csv``, then
checks that the notebook outputs, the committed two-page PDF, and the README
agree with those independently computed values.

Fixed-data expectations
-----------------------
The repository ships a single 85-row synthetic dataset, so the expected row
counts, vocabulary size, and metric values below are treated as assignment
invariants. If the dataset changes, the report generator and this validator
both fail loudly; update the constants only after re-verifying the assignment.

Usage
-----
    python scripts/validate_notebook.py [assignment_1_executed.ipynb]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

try:
    import nbformat
except ImportError:  # pragma: no cover - reported as a validation failure
    nbformat = None

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - reported as a validation failure
    PdfReader = None

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "synthetic_text_data.csv"
SOURCE_PATH = REPO_ROOT / "assignment_1.ipynb"
README_PATH = REPO_ROOT / "README.md"
PDF_PATH = REPO_ROOT / "report" / "report.pdf"
LABELS = ["Entertainment", "Politics", "Sports", "Technology"]
STUDENT_ID = "2305467"

# Fixed-data invariants for the shipped dataset.
EXPECTED = {
    "rows": 85,
    "train_rows": 68,
    "test_rows": 17,
    "vocab": 391,
    "oov": 69,
    "correct": 14,
    "dataset_counts": {
        "Entertainment": 19,
        "Politics": 21,
        "Sports": 15,
        "Technology": 30,
    },
    "train_counts": {
        "Entertainment": 15,
        "Politics": 17,
        "Sports": 12,
        "Technology": 24,
    },
    "test_counts": {
        "Entertainment": 4,
        "Politics": 4,
        "Sports": 3,
        "Technology": 6,
    },
    "per_class": {
        # label: (precision, recall, f1, support)
        "Entertainment": (1.0, 0.75, 6 / 7, 4),
        "Politics": (0.8, 1.0, 8 / 9, 4),
        "Sports": (1.0, 2 / 3, 0.8, 3),
        "Technology": (5 / 7, 5 / 6, 10 / 13, 6),
    },
    "macro_f1": 0.8288,
    "weighted_f1": 0.8235,
    "confusion": [[3, 0, 0, 1], [0, 4, 0, 0], [0, 0, 2, 1], [0, 1, 0, 5]],
}


class Checker:
    def __init__(self) -> None:
        self.passed = 0
        self.failures: list[str] = []

    def check(self, ok: bool, message: str) -> bool:
        if ok:
            self.passed += 1
        else:
            self.failures.append(message)
        return ok

    def fail(self, message: str) -> None:
        self.failures.append(message)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", str(text).lower())


class ManualMultinomialNB:
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ManualMultinomialNB":
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        self.class_log_prior_: dict[Any, float] = {}
        self.feature_log_prob_: dict[Any, np.ndarray] = {}
        self.class_count_: dict[Any, int] = {}
        self.feature_total_: dict[Any, float] = {}
        self.smoothing_denominator_: dict[Any, float] = {}
        for label in self.classes_:
            X_class = X[y == label]
            class_count = X_class.shape[0]
            self.class_count_[label] = class_count
            self.class_log_prior_[label] = float(np.log(class_count / n_samples))
            word_counts = X_class.sum(axis=0).astype(float)
            self.feature_total_[label] = float(word_counts.sum())
            denom = word_counts.sum() + self.alpha * n_features
            self.smoothing_denominator_[label] = float(denom)
            self.feature_log_prob_[label] = np.log((word_counts + self.alpha) / denom)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = []
        for row in X:
            scores = [
                self.class_log_prior_[label] + float(np.dot(row, self.feature_log_prob_[label]))
                for label in self.classes_
            ]
            out.append(self.classes_[int(np.argmax(scores))])
        return np.asarray(out)


def independent_metrics() -> dict[str, Any]:
    raw = pd.read_csv(CSV_PATH)
    if list(raw.columns) != ["text", "label"]:
        raise ValueError(f"CSV columns must be ['text', 'label'], found {list(raw.columns)}")
    df = raw[["text", "label"]].dropna().copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    df = df[df["text"].str.strip().ne("") & df["label"].str.strip().ne("")].reset_index(drop=True)
    df["tokens"] = df["text"].map(tokenize)
    df = df[df["tokens"].map(len) > 0].reset_index(drop=True)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    vocab = sorted({t for tokens in train_df["tokens"] for t in tokens})
    word_to_index = {w: i for i, w in enumerate(vocab)}

    def to_bow(tokens: list[str]) -> np.ndarray:
        counts = Counter(tokens)
        vector = np.zeros(len(word_to_index), dtype=np.int64)
        for word, count in counts.items():
            index = word_to_index.get(word)
            if index is not None:
                vector[index] = count
        return vector

    X_train = np.vstack([to_bow(t) for t in train_df["tokens"]])
    X_test = np.vstack([to_bow(t) for t in test_df["tokens"]])
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    manual = ManualMultinomialNB(alpha=1.0).fit(X_train, y_train)
    manual_train_pred = manual.predict(X_train)
    manual_test_pred = manual.predict(X_test)

    sklearn_model = MultinomialNB(alpha=1.0).fit(X_train, y_train)
    sklearn_train_pred = sklearn_model.predict(X_train)
    sklearn_test_pred = sklearn_model.predict(X_test)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, manual_test_pred, labels=LABELS, zero_division=0
    )
    macro = precision_recall_fscore_support(
        y_test, manual_test_pred, labels=LABELS, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_test, manual_test_pred, labels=LABELS, average="weighted", zero_division=0
    )
    oov = sorted({t for tokens in test_df["tokens"] for t in tokens if t not in word_to_index})
    token_lengths = df["tokens"].map(len)

    return {
        "df": df,
        "train_df": train_df,
        "test_df": test_df,
        "vocab": vocab,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "manual": manual,
        "manual_train_pred": manual_train_pred,
        "manual_test_pred": manual_test_pred,
        "sklearn": sklearn_model,
        "sklearn_train_pred": sklearn_train_pred,
        "sklearn_test_pred": sklearn_test_pred,
        "train_accuracy": accuracy_score(y_train, manual_train_pred),
        "test_accuracy": accuracy_score(y_test, manual_test_pred),
        "sklearn_train_accuracy": accuracy_score(y_train, sklearn_train_pred),
        "sklearn_test_accuracy": accuracy_score(y_test, sklearn_test_pred),
        "train_match": float(np.mean(manual_train_pred == sklearn_train_pred)),
        "test_match": float(np.mean(manual_test_pred == sklearn_test_pred)),
        "n_correct": int((manual_test_pred == y_test).sum()),
        "n_test": int(len(y_test)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "macro": macro,
        "weighted": weighted,
        "oov": oov,
        "empty_test_docs": int((X_test.sum(axis=1) == 0).sum()),
        "token_min": int(token_lengths.min()),
        "token_mean": float(token_lengths.mean()),
        "token_max": int(token_lengths.max()),
        "missing": int(df.isna().sum().sum()),
        "duplicate_texts": int(df["text"].duplicated().sum()),
        "dataset_counts": df["label"].value_counts().to_dict(),
        "train_counts": train_df["label"].value_counts().to_dict(),
        "test_counts": test_df["label"].value_counts().to_dict(),
        "confusion": confusion_matrix(y_test, manual_test_pred, labels=LABELS).tolist(),
    }


def collect_notebook(nb: Any) -> dict[str, Any]:
    streams: list[str] = []
    html: list[str] = []
    plain: list[str] = []
    images = 0
    errors: list[tuple[int, str, str]] = []
    exec_counts: list[Any] = []
    code_count = 0
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        code_count += 1
        exec_counts.append(cell.get("execution_count"))
        for output in cell.get("outputs", []):
            kind = output.get("output_type")
            if kind == "error":
                errors.append((index, output.get("ename", ""), output.get("evalue", "")))
            elif kind == "stream":
                streams.append("".join(output.get("text", [])))
            elif kind in ("display_data", "execute_result"):
                data = output.get("data", {})
                for mime, value in data.items():
                    text = "".join(value) if isinstance(value, list) else str(value)
                    if mime.startswith("image/"):
                        images += 1
                    elif mime == "text/html":
                        html.append(text)
                    elif mime == "text/plain":
                        plain.append(text)
    return {
        "streams": streams,
        "html": html,
        "plain": plain,
        "images": images,
        "errors": errors,
        "exec_counts": exec_counts,
        "code_count": code_count,
    }


def strip_html(fragment: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", fragment)).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebook",
        nargs="?",
        default="assignment_1_executed.ipynb",
        help="path to the executed notebook (default: assignment_1_executed.ipynb)",
    )
    args = parser.parse_args()

    checker = Checker()
    notebook_path = Path(args.notebook)

    if not notebook_path.is_file():
        checker.fail(f"executed notebook not found: {notebook_path}")
    if not SOURCE_PATH.is_file():
        checker.fail(f"source notebook not found: {SOURCE_PATH}")
    if not CSV_PATH.is_file():
        checker.fail(f"dataset not found: {CSV_PATH}")
    if not README_PATH.is_file():
        checker.fail(f"README not found: {README_PATH}")
    if not PDF_PATH.is_file():
        checker.fail(f"report PDF not found: {PDF_PATH}")
    if checker.failures:
        return _report(checker)

    try:
        metrics = independent_metrics()
    except Exception as exc:  # noqa: BLE001 - surfaced as a validation failure
        checker.fail(f"independent metric computation failed: {exc!r}")
        return _report(checker)

    # -------------------------------------------------- structural checks
    if nbformat is None:
        checker.fail("nbformat is required to validate the notebook structure")
        return _report(checker)

    try:
        executed = nbformat.read(notebook_path, as_version=4)
        source = nbformat.read(SOURCE_PATH, as_version=4)
        nbformat.validate(executed)
        nbformat.validate(source)
    except Exception as exc:  # noqa: BLE001
        checker.fail(f"notebook is not valid notebook JSON: {exc!r}")
        return _report(checker)

    checker.check(
        [c.source for c in executed.cells] == [c.source for c in source.cells],
        "executed notebook cell sources differ from assignment_1.ipynb",
    )

    data = collect_notebook(executed)
    checker.check(data["code_count"] == 11, f"expected 11 code cells, found {data['code_count']}")
    checker.check(
        data["exec_counts"] == list(range(1, data["code_count"] + 1)),
        f"execution counts are not sequential: {data['exec_counts']}",
    )
    checker.check(not data["errors"], f"notebook has error outputs: {data['errors'][:3]}")
    checker.check(data["images"] >= 6, f"expected at least 6 figures, found {data['images']}")

    # -------------------------------------------------- independent invariants
    checker.check(metrics["df"].shape[0] == EXPECTED["rows"], "dataset row count changed")
    checker.check(
        metrics["dataset_counts"] == EXPECTED["dataset_counts"],
        f"dataset class counts changed: {metrics['dataset_counts']}",
    )
    checker.check(
        metrics["train_counts"] == EXPECTED["train_counts"], "train class counts changed"
    )
    checker.check(
        metrics["test_counts"] == EXPECTED["test_counts"], "test class counts changed"
    )
    checker.check(metrics["missing"] == 0, "dataset has missing values")
    checker.check(metrics["duplicate_texts"] == 0, "dataset has duplicate texts")
    checker.check(len(metrics["vocab"]) == EXPECTED["vocab"], "vocabulary size changed")
    checker.check(len(metrics["oov"]) == EXPECTED["oov"], "unseen test token count changed")
    checker.check(metrics["empty_test_docs"] == 0, "test documents with no in-vocabulary tokens")
    checker.check(
        (metrics["token_min"], round(metrics["token_mean"], 6), metrics["token_max"])
        == (8, 12.0, 16),
        "token length statistics changed",
    )

    # -------------------------------------------------- notebook output parsing
    text = re.sub(r"\s+", " ", " ".join(data["streams"])).strip()
    html_text = strip_html(" ".join(data["html"]) + " " + " ".join(data["plain"]))

    def expect_int(pattern: str, expected: int, label: str) -> None:
        match = re.search(pattern, text)
        if not match:
            checker.fail(f"notebook output missing {label!r}")
            return
        value = int(match.group(1))
        checker.check(value == expected, f"{label}: expected {expected}, got {value}")

    def expect_float(
        pattern: str, expected: float, label: str, tol: float = 1e-9
    ) -> None:
        match = re.search(pattern, text)
        if not match:
            checker.fail(f"notebook output missing {label!r}")
            return
        value = float(match.group(1))
        checker.check(abs(value - expected) <= tol, f"{label}: expected {expected}, got {value}")

    expect_int(r"Raw rows: (\d+)", EXPECTED["rows"], "Raw rows")
    expect_int(r"Rows with missing text/label dropped: (\d+)", 0, "Rows with missing text/label dropped")
    expect_int(r"Rows with empty text/label dropped: (\d+)", 0, "Rows with empty text/label dropped")
    expect_int(r"Duplicate text rows found: (\d+)", 0, "Duplicate text rows found")
    expect_int(r"Rows after preprocessing: (\d+)", EXPECTED["rows"], "Rows after preprocessing")
    expect_int(r"Training rows: (\d+)", EXPECTED["train_rows"], "Training rows")
    expect_int(r"Test rows: (\d+)", EXPECTED["test_rows"], "Test rows")
    expect_int(r"Vocabulary size: (\d+)", EXPECTED["vocab"], "Vocabulary size")
    expect_int(r"Unseen test tokens ignored: (\d+)", EXPECTED["oov"], "Unseen test tokens ignored")
    expect_int(
        r"Test documents with no in-vocabulary tokens: (\d+)",
        0,
        "Test documents with no in-vocabulary tokens",
    )
    expect_float(r"Train predictions matching: ([\d.]+)%", 100.0, "Train predictions matching")
    expect_float(r"Test predictions matching: ([\d.]+)%", 100.0, "Test predictions matching")
    checker.check(
        "The two models made identical predictions on the test set." in text,
        "notebook does not state that test predictions are identical",
    )

    train_shape = re.search(r"X_train shape: \((\d+), (\d+)\)", text)
    test_shape = re.search(r"X_test shape: \((\d+), (\d+)\)", text)
    checker.check(
        bool(train_shape) and tuple(map(int, train_shape.groups()))
        == (EXPECTED["train_rows"], EXPECTED["vocab"]),
        "X_train shape does not match the independent computation",
    )
    checker.check(
        bool(test_shape) and tuple(map(int, test_shape.groups()))
        == (EXPECTED["test_rows"], EXPECTED["vocab"]),
        "X_test shape does not match the independent computation",
    )

    # Priors / denominators audit table.
    audit_rows = re.findall(
        r"(Entertainment|Politics|Sports|Technology) ([\d.]+) (\d+) (\d+)", text
    )
    checker.check(len(audit_rows) == 4, f"expected 4 audit rows, found {len(audit_rows)}")
    for label, prior, tokens, denom in audit_rows:
        expected_tokens = int(metrics["manual"].feature_total_[label])
        expected_denom = int(metrics["manual"].smoothing_denominator_[label])
        expected_prior = float(np.exp(metrics["manual"].class_log_prior_[label]))
        checker.check(int(tokens) == expected_tokens, f"{label} token total mismatch")
        checker.check(int(denom) == expected_denom, f"{label} denominator mismatch")
        checker.check(
            abs(float(prior) - round(expected_prior, 4)) <= 1e-4,
            f"{label} prior mismatch: expected {round(expected_prior, 4)}, got {prior}",
        )

    # classification_report rows appear twice (manual and scikit-learn).
    report_rows = re.findall(
        r"(Entertainment|Politics|Sports|Technology) ([\d.]+) ([\d.]+) ([\d.]+) (\d+)", text
    )
    checker.check(
        len(report_rows) >= 8, f"expected at least 8 per-class report rows, found {len(report_rows)}"
    )
    for label, precision, recall, f1, support in report_rows:
        exp_p, exp_r, exp_f1, exp_s = EXPECTED["per_class"][label]
        checker.check(abs(float(precision) - round(exp_p, 2)) <= 1e-9, f"{label} precision mismatch")
        checker.check(abs(float(recall) - round(exp_r, 2)) <= 1e-9, f"{label} recall mismatch")
        checker.check(abs(float(f1) - round(exp_f1, 2)) <= 1e-9, f"{label} F1 mismatch")
        checker.check(int(support) == exp_s, f"{label} support mismatch")

    # Accuracy tables emitted as HTML by pandas Styler.
    for model, expected_train, expected_test in [
        ("Manual Naive Bayes", metrics["train_accuracy"], metrics["test_accuracy"]),
        (
            "scikit-learn MultinomialNB",
            metrics["sklearn_train_accuracy"],
            metrics["sklearn_test_accuracy"],
        ),
    ]:
        match = re.search(rf"{re.escape(model)} ([\d.]+) ([\d.]+)", html_text)
        if not match:
            checker.fail(f"accuracy table missing row for {model!r}")
            continue
        checker.check(
            abs(float(match.group(1)) - round(expected_train, 4)) <= 1e-9,
            f"{model} train accuracy mismatch",
        )
        checker.check(
            abs(float(match.group(2)) - round(expected_test, 4)) <= 1e-9,
            f"{model} test accuracy mismatch",
        )

    for model in ("Manual Naive Bayes", "scikit-learn MultinomialNB"):
        match = re.search(
            rf"{re.escape(model)} ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)",
            html_text,
        )
        if not match:
            checker.fail(f"summary table missing row for {model!r}")
            continue
        values = [float(v) for v in match.groups()]
        expected_values = [
            round(metrics["macro"][0], 4),
            round(metrics["macro"][1], 4),
            round(metrics["macro"][2], 4),
            round(metrics["weighted"][0], 4),
            round(metrics["weighted"][1], 4),
            round(metrics["weighted"][2], 4),
        ]
        checker.check(
            all(abs(a - b) <= 1e-9 for a, b in zip(values, expected_values)),
            f"{model} macro/weighted metrics mismatch: {values}",
        )

    # -------------------------------------------------- README delivery checks
    readme = README_PATH.read_text(encoding="utf-8")
    checker.check("assignment_1_executed.ipynb" in readme, "README must mention the executed notebook")
    checker.check("committed" in readme.lower(), "README must say the executed notebook is committed")
    checker.check("report/report.pdf" in readme, "README must link report/report.pdf")
    checker.check(
        "executed-nlp-assignment-notebook" in readme,
        "README must document the Actions artifact name",
    )
    checker.check(
        "7 days" in readme and "expire" in readme.lower(),
        "README must document that the Actions artifact expires after 7 days",
    )
    checker.check("82.35%" in readme, "README must contain the verified test accuracy")
    checker.check("14/17" in readme, "README must contain the verified correct-test count")
    checker.check("391" in readme, "README must contain the verified vocabulary size")

    # -------------------------------------------------- committed PDF checks
    if PdfReader is None:
        checker.fail("pypdf is required to validate report/report.pdf")
    else:
        try:
            reader = PdfReader(str(PDF_PATH))
            pages = len(reader.pages)
            pdf_text = re.sub(r"\s+", " ", " ".join(page.extract_text() or "" for page in reader.pages))
        except Exception as exc:  # noqa: BLE001
            checker.fail(f"could not read report/report.pdf: {exc!r}")
        else:
            checker.check(pages == 2, f"report PDF must have 2 pages, found {pages}")
            checker.check(STUDENT_ID in pdf_text, "student ID missing from the report PDF")
            checker.check(
                f"{metrics['test_accuracy']:.2%}" in pdf_text,
                "report PDF test accuracy does not match the independent computation",
            )
            checker.check(
                f"{metrics['n_correct']}/{metrics['n_test']}" in pdf_text,
                "report PDF correct-test count does not match the independent computation",
            )
            checker.check(
                str(len(metrics["vocab"])) in pdf_text,
                "report PDF vocabulary size does not match the independent computation",
            )
            checker.check("68 training" in pdf_text and "17 test" in pdf_text, "report PDF split text missing")
            checker.check(
                f"{metrics['macro'][2]:.4f}" in pdf_text
                and f"{metrics['weighted'][2]:.4f}" in pdf_text,
                "report PDF macro/weighted F1 does not match the independent computation",
            )
            for label in LABELS:
                idx = LABELS.index(label)
                expected_row = (
                    f"{label} {metrics['precision'][idx]:.4f} {metrics['recall'][idx]:.4f} "
                    f"{metrics['f1'][idx]:.4f} {int(metrics['support'][idx])}"
                )
                checker.check(expected_row in pdf_text, f"report PDF row mismatch: {expected_row}")
            for term in ("Laplace", "Multinomial", "precision", "recall", "F1"):
                checker.check(term.lower() in pdf_text.lower(), f"report PDF missing term: {term}")

    if metrics["manual_train_pred"].shape[0] and not np.array_equal(
        metrics["manual_test_pred"], metrics["sklearn_test_pred"]
    ):
        checker.fail("manual and scikit-learn test predictions differ")
    if not np.array_equal(metrics["manual_train_pred"], metrics["sklearn_train_pred"]):
        checker.fail("manual and scikit-learn train predictions differ")

    return _report(checker)


def _report(checker: Checker) -> int:
    if checker.failures:
        for failure in checker.failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        print(
            f"FAILED: {len(checker.failures)} check(s) failed, {checker.passed} passed",
            file=sys.stderr,
        )
        return 1
    print(f"PASS: {checker.passed} checks passed")
    print("  independent metrics recomputed from synthetic_text_data.csv")
    print("  executed notebook, README, and report/report.pdf all agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
