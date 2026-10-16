# Phase 2 — Preprocessing & Data Quality

Completed directly from the EduRABSA ASQE raw merged annotation files.

## Results
- 6,500 reviews: 4,000 train/validation + 2,500 dev/test
- 27,037 aspect-opinion-category-sentiment annotations
- 0 empty/missing review texts
- 0 missing annotation category/sentiment/opinion
- 8 normalized duplicate-text groups cross the official split boundary (18 rows total)
- 16 rows participate in within-split duplicate groups

## Important policy
The official split is retained. Duplicate texts are explicitly flagged for later leakage-safe evaluation; they are not silently deleted.

## Files
- clean_reviews_phase2.csv
- clean_annotations_phase2.csv
- train_phase2.csv
- test_phase2.csv
- phase2_quality_report.json
- preprocess_edurabsa.py
