# Phase 8.5 — Model Quality Validation

## Evaluation basis
Official EduRABSA `dev_test` evaluation pool. These are held-out results produced during Phases 3 and 4; no training data is used in the reported metrics.

## Sentiment baseline
**Model:** Logistic Regression

- Accuracy: **0.7291**
- Macro Precision: **0.6926**
- Macro Recall: **0.7012**
- Macro F1: **0.6940**
- Weighted F1: **0.7367**

### Class-level result
- Negative F1: **0.7216**
- Neutral F1: **0.5327**
- Positive F1: **0.8278**

**Finding:** Neutral is the weakest class (F1 0.5327), while Positive is strongest (F1 0.8278). This is consistent with the API test where a clearly critical-sounding workload sentence was predicted Neutral; the baseline should therefore not be treated as final production sentiment intelligence.

## Category baseline
**Model:** TF-IDF + Logistic Regression
**Input:** aspect + review text

- Accuracy: **0.4460**
- Macro Precision: **0.3909**
- Macro Recall: **0.3900**
- Macro F1: **0.3732**
- Weighted F1: **0.4463**

**Finding:** Category classification is substantially weaker than sentiment classification. Several lower-support categories have very low F1, so this model should remain a benchmark rather than be presented as production-ready.

## Decision
1. Keep Phase 3 Logistic Regression as the **sentiment baseline benchmark**.
2. Keep Phase 4 TF-IDF + Logistic Regression as the **category baseline benchmark**.
3. Before final deployment/research-paper claims, upgrade sentiment and especially category modeling and re-evaluate on the same held-out split.
4. Do not hard-code sentiment rules to compensate for individual errors.

## Recommended next modeling stage
- Sentiment: compare Linear SVM / calibrated linear model and a transformer fine-tune if compute permits.
- Category: use a hierarchical or transformer-based classifier; preserve rare classes and report macro F1.
- Continue using the official held-out pool for final comparison and avoid test leakage.
