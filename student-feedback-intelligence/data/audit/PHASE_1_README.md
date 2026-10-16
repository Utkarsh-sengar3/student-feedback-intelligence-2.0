# EduRABSA Phase 1 — Canonical Dataset & Leakage Audit

## Source of truth
The canonical dataset is built from the two raw ASQE merged annotation files:
- `Train_validation_Merged_ASQE_N4000.json`
- `DEV_TEST_Merged_ASQE_N2500.json`

The task-specific AEOE/APC/ASTE/OEC representations are not treated as independent datasets.

## Canonical tables
### reviews_canonical
One row per review:
`review_id, original_id, text, domain, overall_sentiment, split, text_sha256`

### annotations_canonical
One row per annotated quadruplet:
`review_id, original_id, annotation_id, aspect, opinion, category, sentiment, split, domain`

## Results
- Total reviews: 6500
- Train/validation pool: 4000
- Dev/test pool: 2500
- Total quadruplet annotations: 27037
- Domains: teacher review=3000, course review=3000, university review=500
- Annotation categories: 24
- Annotation sentiments: Positive=15575, Negative=8700, Neutral=2762

## Leakage audit
- `original_id` overlap across pools: 0
- `review_id` overlap across pools: 0
- Exact text overlap across pools: 6
- Normalized text overlap across pools: 8

### Important finding
There are **8 normalized duplicate-text pairs crossing the split boundary**. Therefore the dataset should **not yet be described as completely leakage-free**.

The affected records are stored in:
`cross_split_duplicate_texts.csv`

These records must be handled explicitly before final model evaluation. A safe approach is to define a leakage-clean evaluation split after reviewing these duplicate groups, while preserving the official EduRABSA split as a reference.

## Next phase
Phase 2 should begin with preprocessing and data-quality validation, followed by a transparent baseline:
1. TF-IDF + Logistic Regression
2. Linear SVM
3. Naive Bayes
4. Evaluation with macro-F1, weighted-F1, precision, recall, confusion matrix
5. Only then move to transformer-based models if justified.
