# Phase 9.3 Add-On

Copy `scripts/phase9_3_feature_engineering.py` into the `scripts/` folder
of your existing `student-feedback-intelligence` project.

Then, from the project root, run:

```powershell
python scripts\phase9_3_feature_engineering.py
```

The script creates:

```text
data/processed/edurabsa/phase9_3/
├── phase9_3_summary.json
├── validation_model_comparison.csv
├── final_test_confusion_matrix.csv
├── final_test_classification_report.csv
├── final_test_predictions.csv
└── PHASE_9_3_README.md

models/sentiment/
└── phase9_3_best_sentiment_model.joblib
```

Important: model selection uses only an internal validation split of
train_validation. The dev_test set is untouched until final evaluation.
