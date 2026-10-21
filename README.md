# NLP Assignment 1 — Naive Bayes Text Classification

This repository contains the notebook and dataset for the Fall 2026 NLP assignment.

## Dataset

The original `synthetic_text_data.csv` is included in the repository root. It contains the `text` and `label` columns.

## Run in Google Colab

1. Open [assignment_1.ipynb](assignment_1.ipynb).
2. In Colab, choose **File → Open notebook → GitHub** and enter: https://github.com/FekyBaz/nlp-naive-bayes-assignment6
3. Select `assignment_1.ipynb` and run all cells from top to bottom. The notebook searches for `synthetic_text_data.csv` in the repository root automatically.

## Run locally

```bash
pip install -r requirements.txt
jupyter notebook assignment_1.ipynb
```

## Assignment checklist

- Text preprocessing and fixed train/test split
- Vocabulary built from training data only
- Bag of Words representation
- Word-frequency, class-distribution, and sentence-length charts
- One word cloud per class using training data
- Manual Naive Bayes with Laplace smoothing
- Comparison with scikit-learn MultinomialNB
- Train/test accuracy, confusion matrices, precision, recall, and F1 per class

## Automated execution

GitHub Actions runs the notebook on pushes to `main` and can also be started manually from the **Actions** tab. A successfully executed notebook is uploaded as an artifact named `executed-nlp-assignment-notebook`. Open the latest run in [Actions](https://github.com/FekyBaz/nlp-naive-bayes-assignment6/actions) to inspect logs and download the notebook with outputs.
