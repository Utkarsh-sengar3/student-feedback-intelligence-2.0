# Phase 10 — Transformer Sentiment Add-On

Copy:

`scripts/phase10_transformer_sentiment.py`

into:

`student-feedback-intelligence/scripts/`

Run from the project root:

```powershell
python -u scripts\phase10_transformer_sentiment.py
```

Install dependencies if necessary:

```powershell
pip install torch transformers datasets accelerate scikit-learn pandas joblib
```

The script uses the project's actual Phase 2 files:

- `data/processed/edurabsa/train_phase2.csv`
- `data/processed/edurabsa/test_phase2.csv`

Default transformer:

`distilbert-base-uncased`

Evaluation protocol:

- 3999 labeled train-validation records
- 3199 internal training records
- 800 validation records
- 2499 untouched labeled test records

Model selection is based only on validation Macro F1.

The transformer is then retrained using the best validation checkpoint and
evaluated on the untouched test set.

It compares the final transformer score against the Phase 9.3 classical
champion at approximately 70.49% Macro F1.

Outputs:

`data/processed/edurabsa/phase10/`

and model:

`models/sentiment/phase10_transformer/`
