# Phase 12 — Aspect Normalization + Category Intelligence

Place the script in:

`student-feedback-intelligence/scripts/`

Run from project root:

```powershell
..\.venv\Scripts\python.exe -u scripts\phase12_aspect_category_model.py
```

This phase:

- joins canonical annotations with review text
- preserves original aspect spans
- creates conservative aspect canonicalization
- converts category annotations to review-level multilabel targets
- uses a review-grouped validation split to avoid annotation leakage
- compares TF-IDF + One-vs-Rest Logistic Regression and Linear SVM
- selects by validation Macro F1
- saves a production category classifier

It does NOT retrain the DistilBERT sentiment model.

Outputs are written to:

`data/processed/edurabsa/phase12/`

and the production category model to:

`models/aspects/phase12_category_classifier.joblib`.
