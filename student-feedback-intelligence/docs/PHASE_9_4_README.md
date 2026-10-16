# Phase 9.4 Add-On

Copy:

`phase9_4_hyperparameter_optimization.py`

to your project's:

`scripts/`

Then run from the project root:

```powershell
python -u scripts\phase9_4_hyperparameter_optimization.py
```

The script uses your actual Phase 2 files:

- `data/processed/edurabsa/train_phase2.csv`
- `data/processed/edurabsa/test_phase2.csv`

It tunes the Phase 9.3 Word+Character TF-IDF + Balanced Logistic Regression
architecture using validation Macro F1 only.

The test set is evaluated only after the best configuration is selected and
retrained on all labeled train-validation rows.

Outputs are written to:

`data/processed/edurabsa/phase9_4/`

and the selected model to:

`models/sentiment/phase9_4_best_sentiment_model.joblib`
