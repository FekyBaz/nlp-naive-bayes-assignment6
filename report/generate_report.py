#!/usr/bin/env python
"""Build the two-page report (HTML + figures) from the verified results.

Run from the repository root:

    python report/generate_report.py

The script recomputes the preprocessing, split, BoW matrices, and both
classifiers, asserts the expected metrics, writes the figures to
``report/figures/``, and writes a self-contained ``report/report.html``.
Print that HTML to PDF with a headless Chrome/Edge command (see README).
"""

from __future__ import annotations

import base64
import html
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = Path(__file__).resolve().parent
FIG_DIR = REPORT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TEST_SIZE = 0.2
ALPHA = 1.0
LABEL_ORDER = ["Entertainment", "Politics", "Sports", "Technology"]

sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
)


class ManualMultinomialNB:
    """From-scratch Multinomial Naive Bayes with Laplace smoothing."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ManualMultinomialNB":
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        self.class_log_prior_ = {}
        self.feature_log_prob_ = {}

        for class_label in self.classes_:
            X_class = X[y == class_label]
            class_count = X_class.shape[0]
            self.class_log_prior_[class_label] = np.log(class_count / n_samples)

            word_counts = X_class.sum(axis=0).astype(float)
            smoothed_counts = word_counts + self.alpha
            denominator = word_counts.sum() + self.alpha * n_features
            self.feature_log_prob_[class_label] = np.log(
                smoothed_counts / denominator
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        for row in X:
            scores = [
                self.class_log_prior_[c] + np.dot(row, self.feature_log_prob_[c])
                for c in self.classes_
            ]
            predictions.append(self.classes_[int(np.argmax(scores))])
        return np.asarray(predictions)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", str(text).lower())


def main() -> int:
    df = pd.read_csv(ROOT / "synthetic_text_data.csv")
    df = df[["text", "label"]].dropna().copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    df = df[df["text"].str.strip().ne("") & df["label"].str.strip().ne("")]
    df = df.reset_index(drop=True)
    df["tokens"] = df["text"].apply(tokenize)
    df = df[df["tokens"].map(len) > 0].reset_index(drop=True)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    vocabulary = sorted(
        {token for tokens in train_df["tokens"] for token in tokens}
    )
    word_to_index = {word: i for i, word in enumerate(vocabulary)}

    def to_bow(tokens: list[str]) -> np.ndarray:
        vector = np.zeros(len(word_to_index), dtype=np.int64)
        for token in tokens:
            index = word_to_index.get(token)
            if index is not None:
                vector[index] += 1
        return vector

    X_train = np.vstack([to_bow(tokens) for tokens in train_df["tokens"]])
    X_test = np.vstack([to_bow(tokens) for tokens in test_df["tokens"]])
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    manual = ManualMultinomialNB(alpha=ALPHA).fit(X_train, y_train)
    manual_pred = manual.predict(X_test)
    manual_train_pred = manual.predict(X_train)

    sklearn_model = MultinomialNB(alpha=ALPHA).fit(X_train, y_train)
    sklearn_pred = sklearn_model.predict(X_test)

    train_accuracy = accuracy_score(y_train, manual_train_pred)
    test_accuracy = accuracy_score(y_test, manual_pred)
    n_correct = int((manual_pred == y_test).sum())
    n_test = len(y_test)
    prediction_match = float(np.mean(manual_pred == sklearn_pred))

    oov = {
        token
        for tokens in test_df["tokens"]
        for token in tokens
        if token not in word_to_index
    }
    empty_test_docs = int((X_test.sum(axis=1) == 0).sum())

    # These assertions make the report fail loudly if the data or split changes.
    assert len(df) == 85, len(df)
    assert len(train_df) == 68 and len(test_df) == 17
    assert len(vocabulary) == 391
    assert abs(train_accuracy - 1.0) < 1e-12
    assert abs(test_accuracy - 14 / 17) < 1e-12
    assert n_correct == 14
    assert prediction_match == 1.0
    assert len(oov) == 69
    assert empty_test_docs == 0

    labels = LABEL_ORDER
    cm = confusion_matrix(y_test, manual_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, manual_pred, labels=labels, zero_division=0
    )
    macro = precision_recall_fscore_support(
        y_test, manual_pred, labels=labels, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_test, manual_pred, labels=labels, average="weighted", zero_division=0
    )
    report_text = classification_report(
        y_test, manual_pred, labels=labels, zero_division=0
    )

    # ---------------------------------------------------------------- figures
    fig_data_path = FIG_DIR / "fig_data.png"
    fig, axes = plt.subplots(
        1, 2, figsize=(7.4, 2.3), gridspec_kw={"width_ratios": [1, 1.35]}
    )

    class_counts = (
        df["label"].value_counts().reindex(labels).fillna(0).astype(int)
    )
    bars = axes[0].bar(
        class_counts.index,
        class_counts.values,
        color=sns.color_palette("deep")[0],
        label="Full dataset",
    )
    axes[0].bar_label(bars, padding=1, fontsize=7.5)
    axes[0].set_title("Class distribution (full dataset)")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Number of sentences")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(loc="upper right")

    word_counts = Counter(
        token for tokens in train_df["tokens"] for token in tokens
    )
    top_words = pd.DataFrame(
        word_counts.most_common(15), columns=["word", "frequency"]
    )
    axes[1].barh(
        top_words["word"][::-1],
        top_words["frequency"][::-1],
        color=sns.color_palette("deep")[2],
        label="Training set",
    )
    axes[1].set_title("Top 15 training word frequencies")
    axes[1].set_xlabel("Frequency")
    axes[1].set_ylabel("Word")
    axes[1].legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(fig_data_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig_cm_path = FIG_DIR / "fig_confusion.png"
    fig, ax = plt.subplots(figsize=(3.6, 2.75))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Count"},
        ax=ax,
    )
    ax.set_title("Manual NB — test confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(fig_cm_path, dpi=260, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------ HTML
    def data_uri(path: Path) -> str:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def metric(value: float) -> str:
        return f"{value:.4f}"

    class_rows = "".join(
        "<tr>"
        f"<td>{html.escape(label)}</td>"
        f"<td>{metric(precision[i])}</td>"
        f"<td>{metric(recall[i])}</td>"
        f"<td>{metric(f1[i])}</td>"
        f"<td>{int(support[i])}</td>"
        "</tr>"
        for i, label in enumerate(labels)
    )

    error_items = []
    for index in np.where(manual_pred != y_test)[0]:
        error_items.append(
            "<li>"
            f"<b>{html.escape(y_test[index])} → {html.escape(manual_pred[index])}</b>: "
            f"<i>{html.escape(str(test_df.loc[index, 'text']))}</i>"
            "</li>"
        )
    error_html = "".join(error_items)

    avg_tokens = df["tokens"].map(len).mean()
    min_tokens = int(df["tokens"].map(len).min())
    max_tokens = int(df["tokens"].map(len).max())
    class_text = ", ".join(
        f"{label} {count}" for label, count in class_counts.items()
    )

    css = """
@page { size: A4; margin: 11mm 12mm 10mm 12mm; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-size: 8.5pt;
       line-height: 1.32; color: #111827; margin: 0; }
.eyebrow { text-transform: uppercase; letter-spacing: 1.2px; font-size: 7.3pt;
           color: #2563eb; font-weight: 700; }
h1 { font-size: 17pt; margin: 0.8mm 0 1mm; color: #0f172a; letter-spacing: -0.3px; }
.meta { font-size: 7.7pt; color: #4b5563; margin-bottom: 2.5mm; }
h2 { font-size: 10.1pt; margin: 2.7mm 0 1mm; color: #0f172a;
     border-bottom: 1px solid #dbe2ea; padding-bottom: 0.6mm; }
h3 { font-size: 9pt; margin: 1mm 0 0.8mm; color: #0f172a; }
p { margin: 0 0 1.3mm; text-align: justify; }
ul { margin: 0 0 1.2mm 4mm; padding: 0; }
li { margin-bottom: 0.9mm; text-align: justify; }
code { font-family: Consolas, "Courier New", monospace; font-size: 7.8pt;
       background: #f1f5f9; padding: 0 1px; }
.abstract { background: #f8fafc; border-left: 3px solid #2563eb;
            padding: 1.8mm 2.6mm; margin: 0.8mm 0 2.2mm; text-align: justify; }
.eq { font-family: "Cambria Math", Georgia, serif; font-size: 9pt;
      text-align: center; margin: 1.1mm 0; }
table { width: 100%; border-collapse: collapse; margin: 1mm 0 1.5mm;
        font-size: 7.9pt; }
th { background: #e8eef7; color: #0f172a; font-weight: 600; }
th, td { border: 1px solid #cbd5e1; padding: 1mm 1.6mm; text-align: center; }
td:first-child, th:first-child { text-align: left; }
.figure { text-align: center; margin: 1mm 0 0.5mm; }
.figure img { width: 100%; }
.caption { font-size: 7.5pt; color: #4b5563; text-align: left; margin: 0 0 1.8mm; }
.row { display: flex; gap: 4mm; align-items: flex-start; }
.col { flex: 1; }
.page-break { page-break-before: always; }
.note { font-size: 7.7pt; color: #374151; background: #f9fafb;
        border: 1px solid #e5e7eb; padding: 1.5mm 2.2mm; }
.badge { display: inline-block; background: #dcfce7; color: #166534;
         border-radius: 3px; padding: 0.4mm 1.4mm; font-size: 7.5pt; font-weight: 600; }
"""

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NLP Assignment 1 — Naive Bayes Text Classification</title>
<style>{css}</style>
</head>
<body>

<div class="eyebrow">Natural Language Processing — Fall 2026</div>
<h1>Assignment 1: Text Classification with Naive Bayes</h1>
<div class="meta">Student ID: 2305467 &nbsp;|&nbsp; GitHub: FekyBaz &nbsp;|&nbsp;
Repository: github.com/FekyBaz/nlp-naive-bayes-assignment6</div>

<div class="abstract">
This report summarizes a bag-of-words text classifier built for a four-class
sentence dataset. A Multinomial Naive Bayes model is implemented from scratch
with Laplace smoothing and log-space scoring, then compared with
scikit-learn's <code>MultinomialNB</code> on exactly the same 68/17 split.
Both models reach <b>{train_accuracy:.2%} training accuracy</b> and
<b>{test_accuracy:.2%} test accuracy</b> ({n_correct}/{n_test} correct), and make
<b>identical predictions</b>. The small test set and the gap between training
and test accuracy call for cautious interpretation.
</div>

<h2>1. Dataset</h2>
<p>The repository includes <code>synthetic_text_data.csv</code> with 85 English
sentences and four labels: {class_text}. The file has no missing values, no
duplicate texts, and no empty sentences. Sentences average {avg_tokens:.0f}
alphabetic tokens (range {min_tokens}–{max_tokens}). The classes are
imbalanced: Technology is the largest class and Sports the smallest. Figure 1
(on page 2) shows the class distribution and the most frequent training words.
As expected without stopword removal, function words such as “the”, “are”, and
“to” dominate the frequency ranking.</p>

<h2>2. Preprocessing, split, and vocabulary</h2>
<p>Each sentence is lowercased and tokenized with the regular expression
<code>[a-z]+</code>, which removes punctuation, digits, and non-ASCII
characters. Rows that become empty after tokenization are dropped (none did).
The cleaned data is split with <code>test_size=0.2</code>,
<code>random_state=42</code>, and stratification, producing <b>68 training</b>
and <b>17 test</b> sentences. The vocabulary is built from the training tokens
only: <b>391 distinct words</b>. Test-only words are safely ignored, giving
<b>69 unseen test types</b> and <b>0 test documents with no in-vocabulary
tokens</b>. No test sentence is used for the vocabulary, word frequencies, or
word clouds.</p>

<h2>3. Models</h2>
<p><b>Manual Multinomial Naive Bayes.</b> For each class <i>c</i>, the prior and
smoothed word likelihood are estimated from the training counts:</p>
<div class="eq">P(c) = N<sub>c</sub> / N &nbsp;&nbsp;&nbsp;&nbsp;
P(w | c) = (count(w, c) + α) / (Σ<sub>w</sub> count(w, c) + α |V|)</div>
<div class="eq">ŷ = argmax<sub>c</sub> [ log P(c) + Σ<sub>w</sub> x<sub>w</sub>
log P(w | c) ]</div>
<p>with <code>α = 1</code> (Laplace smoothing) and <code>|V| = 391</code>.
The implementation follows the section's four-step formulation (priors,
vocabulary/denominators, likelihoods, class scores) and builds BoW vectors with
<code>Counter</code>. Counts are weighted by the BoW vector and summed in log
space to avoid numerical underflow. Unknown words contribute nothing because
they are absent from the vocabulary. <b>scikit-learn baseline.</b>
<code>MultinomialNB(alpha=1.0)</code>
is trained on the same matrices and labels. Because both models use empirical
priors and the same smoothed multinomial likelihood, identical predictions are
expected.</p>

<h2>4. Results</h2>
<table>
<thead><tr><th>Model</th><th>Train accuracy</th><th>Test accuracy</th>
<th>Correct test predictions</th></tr></thead>
<tbody>
<tr><td>Manual Multinomial NB</td><td>{train_accuracy:.4f}</td>
<td>{test_accuracy:.4f}</td><td>{n_correct}/{n_test}</td></tr>
<tr><td>scikit-learn MultinomialNB</td><td>{train_accuracy:.4f}</td>
<td>{test_accuracy:.4f}</td><td>{n_correct}/{n_test}</td></tr>
</tbody>
</table>

<p>Both models produce the same test predictions (agreement
<b>{prediction_match:.0%}</b>). The per-class test metrics are therefore
identical for the two implementations:</p>

<table>
<thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-score</th>
<th>Support</th></tr></thead>
<tbody>
{class_rows}
<tr><td><b>Macro average</b></td><td>{metric(macro[0])}</td>
<td>{metric(macro[1])}</td><td>{metric(macro[2])}</td><td>{n_test}</td></tr>
<tr><td><b>Weighted average</b></td><td>{metric(weighted[0])}</td>
<td>{metric(weighted[1])}</td><td>{metric(weighted[2])}</td><td>{n_test}</td></tr>
</tbody>
</table>

<p>Politics is the only class with perfect recall (1.00); Technology has the
lowest F1-score (0.77) because it is both the largest class and the source or
target of most confusions. The macro F1 is {macro[2]:.4f} and the weighted F1
is {weighted[2]:.4f}. Perfect training accuracy versus {test_accuracy:.2%} test
accuracy is consistent with overfitting, but with only 17 test sentences a
single error moves accuracy by about 5.9 percentage points, so the estimate is
uncertain.</p>

<div class="page-break"></div>

<h2>5. Visual diagnostics and error analysis</h2>
<div class="figure">
<img src="{data_uri(fig_data_path)}" alt="Class distribution and top word frequencies">
</div>
<p class="caption"><b>Figure 1.</b> Left: class distribution in the full
dataset ({class_text}). Right: the 15 most frequent tokens in the training
split. Frequent function words dominate because the assignment's preprocessing
does not remove stopwords.</p>

<div class="row">
<div class="col">
<div class="figure">
<img src="{data_uri(fig_cm_path)}" alt="Confusion matrix for the manual model">
</div>
<p class="caption"><b>Figure 2.</b> Test-set confusion matrix for the manual
model. The scikit-learn matrix is identical.</p>
</div>
<div class="col">
<h3>What the three errors show</h3>
<ul>
{error_html}
</ul>
<p>All errors are single-label confusions rather than a class collapsing into
another. They reflect the main limitation of a bag-of-words model: individual
words such as “digital”, “training”, “market”, and “impact” carry overlapping
associations across topics, and word order or multi-word context is discarded.
The model nevertheless recovers Politics perfectly and classifies 14 of 17
sentences correctly.</p>
</div>
</div>

<h2>6. Discussion and limitations</h2>
<p>The manual implementation matches scikit-learn exactly, which validates the
count aggregation, prior calculation, Laplace smoothing, and log-space scoring.
The main performance limitation is the data: 85 short synthetic sentences,
noticeable class imbalance, and a 17-sentence test set. The model also uses a
simple BoW representation with no stopword removal, stemming, n-grams, or
hyperparameter tuning. These choices keep the assignment scope controlled but
mean the reported accuracy should not be read as a generalisation estimate.
Cross-validation on a larger, more balanced corpus would give a more reliable
comparison. The test set was used only for final evaluation; all vocabulary and
word-based exploratory analysis came from the training split.</p>

<h2>7. Reproducibility</h2>
<div class="note">
<b>Environment:</b> Python 3.11+ (verified on 3.12 locally and 3.11 in
GitHub Actions), numpy, pandas, matplotlib, seaborn, scikit-learn, wordcloud
(see <code>requirements.txt</code>).<br>
<b>Notebook:</b> <code>assignment_1.ipynb</code> (source) and
<code>assignment_1_executed.ipynb</code> (saved outputs).<br>
<b>Execute:</b> <code>python -m jupyter nbconvert --to notebook --execute
assignment_1.ipynb --output assignment_1_executed.ipynb</code><br>
<b>Validate:</b> <code>python scripts/validate_notebook.py
assignment_1_executed.ipynb</code><br>
<b>Report build:</b> <code>python report/generate_report.py</code>
</div>

</body>
</html>
"""

    html_path = REPORT_DIR / "report.html"
    html_path.write_text(html_doc, encoding="utf-8")

    print(f"Data: {len(df)} rows, {len(train_df)} train / {len(test_df)} test")
    print(f"Vocabulary: {len(vocabulary)}; unseen test types: {len(oov)}")
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy:  {test_accuracy:.4f} ({n_correct}/{n_test})")
    print(f"Prediction agreement: {prediction_match:.2%}")
    print(f"Macro F1: {macro[2]:.4f}; weighted F1: {weighted[2]:.4f}")
    print("\nClassification report:\n")
    print(report_text)
    print(f"Wrote {html_path}")
    print(f"Wrote {fig_data_path}")
    print(f"Wrote {fig_cm_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
