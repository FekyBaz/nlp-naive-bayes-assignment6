# NLP Assignment 1 — Naive Bayes Text Classification

This repository contains the Jupyter notebook for the Fall 2026 NLP assignment.

## Dataset

The notebook expects `synthetic_text_data.csv` with two columns:
- `text`: sentence text
- `label`: category label

Place the CSV in `data/synthetic_text_data.csv`, next to the notebook, or upload it when prompted in Google Colab. The dataset has not yet been added because it was not attached in the conversation.

## Run in Google Colab

1. Open [assignment_1.ipynb](assignment_1.ipynb).
2. In Colab, choose **File → Open notebook → GitHub** and enter: https://github.com/FekyBaz/nlp-naive-bayes-assignment6
3. Run all cells from top to bottom.
4. Upload `synthetic_text_data.csv` if prompted.
5. Save the notebook with outputs after it runs successfully.

## Assignment checklist

- Text preprocessing and fixed train/test split
- Vocabulary built from training data only
- Bag of Words representation
- Word-frequency, class-distribution, and sentence-length charts
- One word cloud per class using training data
- Manual Naive Bayes with Laplace smoothing
- Comparison with scikit-learn MultinomialNB
- Train/test accuracy, confusion matrices, precision, recall, and F1 per class

**Important:** The notebook has not been run against the real dataset yet. Do not report metrics until the actual CSV is loaded and the notebook has run successfully.
