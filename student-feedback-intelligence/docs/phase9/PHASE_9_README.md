# Phase 9 — Model Improvement

Run from the `student-feedback-intelligence` project root:

```powershell
python scripts/phase9_model_improvement.py
```

The script compares:
- TF-IDF + Logistic Regression
- TF-IDF + class-balanced Logistic Regression
- TF-IDF + Linear SVM
- TF-IDF + class-balanced Linear SVM
- TF-IDF + Multinomial Naive Bayes

Selection uses Macro F1. Existing Phase 3/4 models are not overwritten.

Expected output:
`data/processed/edurabsa/phase9/`

Final candidate models:
`models/sentiment/phase9_best_sentiment_model.joblib`
`models/category/phase9_best_category_model.joblib`
