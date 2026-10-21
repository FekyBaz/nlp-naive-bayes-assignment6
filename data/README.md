# Dataset folder

The dataset is committed at the repository root as
`../synthetic_text_data.csv`; that is the copy used by the notebook by default.

This folder is optional. If you place a copy of `synthetic_text_data.csv` here,
the notebook will prefer `data/synthetic_text_data.csv` because it checks this
path first.

Expected columns: `text` and `label`. The committed dataset has 85 sentences in
four classes: Entertainment, Politics, Sports, and Technology.
