# Phase 3 — Classical Sentiment Baseline

Built three leakage-aware TF-IDF baselines:
1. Logistic Regression
2. Linear SVM
3. Multinomial Naive Bayes

TF-IDF:
- word n-grams: (1,2)
- sublinear TF
- min_df=2
- max_df=0.95
- max_features=50,000

The official train_validation pool was split 80/20 for validation using stratification.
Cross-split normalized duplicate texts found in the dev/test pool were removed from the training pool only.
Removed from training: 9

The best model by validation macro-F1 was:
**Logistic Regression**

A final evaluation was then run once against the official dev_test pool.

See:
- `baseline_model_comparison.csv`
- `baseline_detailed_metrics.json`
- `final_baseline_test_metrics.json`
- `test_predictions.csv`
- `baseline_sentiment_model.joblib`

Do not compare future transformer results to this baseline unless they use the same evaluation policy.
