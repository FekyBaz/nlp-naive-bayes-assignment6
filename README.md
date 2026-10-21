# NLP Assignment 1 — Naive Bayes Text Classification

Fall 2026 Natural Language Processing assignment. The project implements a
bag-of-words text classifier from scratch, compares it with scikit-learn, and
evaluates both models with charts, confusion matrices, and per-class metrics.

**Deliverables**

- `assignment_1.ipynb` — clean source notebook (no saved outputs)
- `assignment_1_executed.ipynb` — the same notebook executed from top to bottom with saved outputs
- `report/report.pdf` — the required two-page methodology and results report

## Dataset

`synthetic_text_data.csv` is committed at the repository root. It has two
columns, `text` and `label`, with 85 English sentences in four classes:
Entertainment (19), Politics (21), Sports (15), and Technology (30). The file
has no missing values, duplicate texts, or empty sentences.

The notebook looks for the CSV in `data/synthetic_text_data.csv` first, then in
the repository root, then at `/content/synthetic_text_data.csv`. If none is
found in Google Colab, it prompts for a CSV upload. See `data/README.md` for the
optional `data/` copy.

## Method

- Lowercase each sentence and extract alphabetic tokens with `[a-z]+`
  (punctuation, digits, and non-ASCII characters are removed).
- Use a fixed, stratified 80/20 train/test split with `test_size=0.2` and
  `random_state=42` (68 training and 17 test sentences).
- Build the vocabulary from training tokens only (391 distinct words) and
  ignore unseen test words safely.
- Represent each sentence as a Bag-of-Words count vector using
  `collections.Counter`, following the Section 3 BOW example.
- Implement the Section 3 Naive Bayes steps manually: class priors, per-class
  vocabulary denominators, Laplace-smoothed likelihoods (`alpha=1`), and
  class scores. Log probabilities are used to avoid numerical underflow.
- Train `sklearn.naive_bayes.MultinomialNB(alpha=1.0)` on the same matrices and
  compare predictions.
- Visualize the class distribution, top word frequencies, sentence lengths, one
  word cloud per class (training data only), and test confusion matrices.

No test sentence is used for the vocabulary, word frequencies, word clouds, or
model fitting. The test set is used only for final evaluation.

## Verified results

Both implementations produce identical predictions on the test set:

| Model | Train accuracy | Test accuracy | Correct test predictions |
| --- | ---: | ---: | ---: |
| Manual Multinomial NB | 1.0000 | 0.8235 | 14/17 |
| scikit-learn MultinomialNB | 1.0000 | 0.8235 | 14/17 |

Per-class test metrics (identical for both models):

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| Entertainment | 1.0000 | 0.7500 | 0.8571 | 4 |
| Politics | 0.8000 | 1.0000 | 0.8889 | 4 |
| Sports | 1.0000 | 0.6667 | 0.8000 | 3 |
| Technology | 0.7143 | 0.8333 | 0.7692 | 6 |
| **Macro average** | 0.8786 | 0.8125 | 0.8288 | 17 |
| **Weighted average** | 0.8521 | 0.8235 | 0.8235 | 17 |

The 100% training accuracy versus 82.35% test accuracy is consistent with
overfitting, but the dataset is very small (17 test sentences). A single test
error changes accuracy by about 5.9 percentage points, so these numbers should
not be read as a generalisation estimate.

## Run locally

Python 3.11+ is recommended (verified with Python 3.12.6 on Windows and 3.11 in
GitHub Actions).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m jupyter nbconvert --to notebook --execute assignment_1.ipynb \
    --output assignment_1_executed.ipynb

python scripts/validate_notebook.py assignment_1_executed.ipynb
```

To work interactively instead, run `python -m jupyter notebook
assignment_1.ipynb` and execute all cells from top to bottom.

`requirements.txt` includes the notebook's runtime dependencies plus the
Jupyter/nbconvert packages needed to execute it. The notebook uses relative
paths, so it can run from the repository root without editing.

## Run in Google Colab

1. Open [assignment_1.ipynb](assignment_1.ipynb) on GitHub.
2. In Colab, choose **File → Open notebook → GitHub** and enter
   `https://github.com/FekyBaz/nlp-naive-bayes-assignment6`, then select
   `assignment_1.ipynb`.
3. Colab does not clone the full repository when opening a notebook, so the
   CSV may not be present in the runtime. When the data cell runs, upload
   `synthetic_text_data.csv` if prompted (or place it at `/content/`).
4. Run all cells from top to bottom.

## Source and executed notebooks

- `assignment_1.ipynb` is the clean source file. It is the file GitHub Actions
  executes and the file to edit when changing the analysis.
- `assignment_1_executed.ipynb` is a saved run with execution counts, printed
  results, and rendered figures. It is generated from the source notebook and
  is committed for convenient review and submission.
- GitHub Actions also uploads its own executed copy as the
  `executed-nlp-assignment-notebook` artifact on every workflow run (the
  artifact step uses `if: always()`, so it is retained even if execution fails).

## Two-page report

The required short report is at [`report/report.pdf`](report/report.pdf). It
summarizes the dataset, preprocessing, models, results, error analysis, and
limitations.

- `report/report.html` is the self-contained HTML source (with embedded figures).
- `report/figures/` contains the generated figures.
- Regenerate the HTML and figures with:

  ```bash
  python report/generate_report.py
  ```

- Print the HTML to PDF with headless Chrome or Edge, for example:

  ```bash
  chrome --headless=new --disable-gpu --no-pdf-header-footer \
      --print-to-pdf=report/report.pdf report/report.html
  ```

  The report generator recomputes the models and asserts the verified metrics
  before writing the report.

## Validation performed

- The notebook was executed twice from a clean kernel locally; both runs
  produced the same text metrics and identical figures.
- `scripts/validate_notebook.py` checks that every code cell has an execution
  count, no cell contains an error output, all six required figures are
  rendered, and the key results match the 68/17 split (85 rows, 391 vocabulary
  words, 14/17 correct, 82.35% test accuracy, identical model predictions).
- The GitHub Actions workflow executes and validates the notebook on every push
  to `main` and uploads the executed notebook as an artifact.

Run the validation check with:

```bash
python scripts/validate_notebook.py assignment_1_executed.ipynb
```

## Repository layout

```text
assignment_1.ipynb            clean source notebook
assignment_1_executed.ipynb   executed notebook with saved outputs
synthetic_text_data.csv       the 85-sentence dataset
report/report.pdf             two-page report
report/report.html            self-contained report source
report/generate_report.py     rebuilds the report and figures
scripts/validate_notebook.py  automated output checks
data/README.md                optional data-folder note
requirements.txt              runtime and execution dependencies
.github/workflows/            GitHub Actions notebook execution
```

## Limitations

The dataset has only 85 short synthetic sentences and a 17-sentence test set.
The model uses a simple Bag-of-Words representation without stopword removal,
stemming, n-grams, or hyperparameter tuning, and the evaluation uses a single
held-out split rather than cross-validation. The reported accuracy is therefore
a narrow estimate for this split, not a claim about generalisation to other
text.
