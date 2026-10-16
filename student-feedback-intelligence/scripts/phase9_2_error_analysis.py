from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = ROOT / "data/processed/edurabsa/train_phase2.csv"
TEST_FILE = ROOT / "data/processed/edurabsa/test_phase2.csv"

MODEL_FILE = (
    ROOT
    / "models/sentiment/phase9_1_best_sentiment_model.joblib"
)

PHASE9_DIR = ROOT / "data/processed/edurabsa/phase9_1"
OUTPUT_DIR = ROOT / "data/processed/edurabsa/phase9_2"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("PHASE 9.2 — SENTIMENT ERROR & LEAKAGE ANALYSIS")
print("=" * 70)

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

model = joblib.load(MODEL_FILE)

print(f"Train rows : {len(train)}")
print(f"Test rows  : {len(test)}")


# ============================================================
# PREPARE TEST DATA
# ============================================================

test["model_text"] = (
    test["text_clean"]
    .fillna(test["text"])
    .fillna("")
    .astype(str)
    .str.strip()
)

# Only labeled test records can be evaluated.
test_eval = test[
    test["overall_sentiment"].notna()
    & (test["model_text"] != "")
].copy()

X_test = test_eval["model_text"]
y_test = test_eval["overall_sentiment"].astype(str).str.strip()

test_eval["predicted_sentiment"] = model.predict(X_test)


# ============================================================
# 1. OVERALL PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("1. OVERALL PERFORMANCE")
print("=" * 70)

accuracy = accuracy_score(
    y_test,
    test_eval["predicted_sentiment"],
)

precision_macro, recall_macro, f1_macro, _ = (
    precision_recall_fscore_support(
        y_test,
        test_eval["predicted_sentiment"],
        average="macro",
        zero_division=0,
    )
)

precision_weighted, recall_weighted, f1_weighted, _ = (
    precision_recall_fscore_support(
        y_test,
        test_eval["predicted_sentiment"],
        average="weighted",
        zero_division=0,
    )
)

overall = {
    "accuracy": accuracy,
    "precision_macro": precision_macro,
    "recall_macro": recall_macro,
    "f1_macro": f1_macro,
    "f1_weighted": f1_weighted,
}

print(
    f"Accuracy       : {accuracy:.4f}\n"
    f"Macro Precision: {precision_macro:.4f}\n"
    f"Macro Recall   : {recall_macro:.4f}\n"
    f"Macro F1       : {f1_macro:.4f}\n"
    f"Weighted F1    : {f1_weighted:.4f}"
)


# ============================================================
# 2. CROSS-SPLIT DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("2. CROSS-SPLIT DUPLICATE ANALYSIS")
print("=" * 70)

train_hashes = set(
    train["text_hash"].dropna().astype(str)
)

test_hashes = set(
    test["text_hash"].dropna().astype(str)
)

overlap_hashes = train_hashes & test_hashes

duplicate_rows = train[
    train["text_hash"].astype(str).isin(overlap_hashes)
].copy()

duplicate_rows = duplicate_rows[
    [
        c for c in [
            "review_id",
            "original_id",
            "domain",
            "overall_sentiment",
            "text",
            "text_hash",
        ]
        if c in duplicate_rows.columns
    ]
]

duplicate_rows.to_csv(
    OUTPUT_DIR / "cross_split_duplicates.csv",
    index=False,
)

print(f"Unique overlapping text hashes: {len(overlap_hashes)}")
print(f"Train records involved          : {len(duplicate_rows)}")

if len(duplicate_rows):
    print("\nDuplicates:")
    print(duplicate_rows.to_string(index=False))


# ============================================================
# 3. TEST PREDICTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("3. TEST PREDICTION DISTRIBUTION")
print("=" * 70)

prediction_distribution = pd.DataFrame({
    "actual": y_test.value_counts(),
    "predicted": test_eval["predicted_sentiment"].value_counts(),
}).fillna(0)

print(prediction_distribution)

prediction_distribution.to_csv(
    OUTPUT_DIR / "prediction_distribution.csv"
)


# ============================================================
# 4. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    test_eval["predicted_sentiment"],
    labels=["Neg", "Neu", "Pos"],
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(
    OUTPUT_DIR / "classification_report.csv"
)

print("\nCLASSIFICATION REPORT")
print(report_df)


# ============================================================
# 5. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_eval["predicted_sentiment"],
    labels=["Neg", "Neu", "Pos"],
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual_Neg", "Actual_Neu", "Actual_Pos"],
    columns=["Pred_Neg", "Pred_Neu", "Pred_Pos"],
)

cm_df.to_csv(
    OUTPUT_DIR / "confusion_matrix.csv"
)

print("\nCONFUSION MATRIX")
print(cm_df)


# ============================================================
# 6. ALL ERRORS
# ============================================================

test_eval["correct"] = (
    test_eval["overall_sentiment"].astype(str)
    ==
    test_eval["predicted_sentiment"].astype(str)
)

errors = test_eval[
    ~test_eval["correct"]
].copy()

errors["error_type"] = (
    errors["overall_sentiment"].astype(str)
    + "_to_"
    + errors["predicted_sentiment"].astype(str)
)

error_columns = [
    c for c in [
        "review_id",
        "original_id",
        "domain",
        "overall_sentiment",
        "predicted_sentiment",
        "text",
        "text_clean",
        "error_type",
    ]
    if c in errors.columns
]

errors[error_columns].to_csv(
    OUTPUT_DIR / "all_misclassifications.csv",
    index=False,
)

print("\n" + "=" * 70)
print("4. ERROR SUMMARY")
print("=" * 70)

print(f"Total errors: {len(errors)}")

print("\nErrors by type:")
print(
    errors["error_type"]
    .value_counts()
)


# ============================================================
# 7. ERROR ANALYSIS BY DOMAIN
# ============================================================

print("\n" + "=" * 70)
print("5. PERFORMANCE BY DOMAIN")
print("=" * 70)

domain_results = []

for domain, group in test_eval.groupby("domain"):

    if len(group) == 0:
        continue

    acc = accuracy_score(
        group["overall_sentiment"],
        group["predicted_sentiment"],
    )

    _, _, f1, _ = precision_recall_fscore_support(
        group["overall_sentiment"],
        group["predicted_sentiment"],
        average="macro",
        zero_division=0,
    )

    domain_results.append({
        "domain": domain,
        "samples": len(group),
        "accuracy": acc,
        "macro_f1": f1,
    })

domain_df = pd.DataFrame(domain_results)

print(domain_df.to_string(index=False))

domain_df.to_csv(
    OUTPUT_DIR / "domain_performance.csv",
    index=False,
)


# ============================================================
# 8. DOMAIN × SENTIMENT
# ============================================================

domain_sentiment = pd.crosstab(
    test_eval["domain"],
    test_eval["overall_sentiment"],
)

domain_sentiment.to_csv(
    OUTPUT_DIR / "domain_sentiment_distribution.csv"
)

print("\nDOMAIN × SENTIMENT")
print(domain_sentiment)


# ============================================================
# 9. ERROR RATE BY TRUE SENTIMENT
# ============================================================

class_error_results = []

for label in ["Neg", "Neu", "Pos"]:

    group = test_eval[
        test_eval["overall_sentiment"] == label
    ]

    errors_for_class = group[
        group["predicted_sentiment"] != label
    ]

    class_error_results.append({
        "sentiment": label,
        "samples": len(group),
        "errors": len(errors_for_class),
        "error_rate": (
            len(errors_for_class) / len(group)
            if len(group)
            else 0
        ),
    })

class_error_df = pd.DataFrame(
    class_error_results
)

class_error_df.to_csv(
    OUTPUT_DIR / "class_error_rates.csv",
    index=False,
)

print("\nERROR RATE BY TRUE SENTIMENT")
print(class_error_df)


# ============================================================
# 10. NEUTRAL ERROR ANALYSIS
# ============================================================

neutral_errors = errors[
    errors["overall_sentiment"] == "Neu"
].copy()

neutral_errors[
    error_columns
].to_csv(
    OUTPUT_DIR / "neutral_misclassifications.csv",
    index=False,
)

print("\n" + "=" * 70)
print("6. NEUTRAL ERROR ANALYSIS")
print("=" * 70)

print(
    f"Neutral errors: {len(neutral_errors)}"
)

print("\nNeutral error destinations:")
print(
    neutral_errors["predicted_sentiment"]
    .value_counts()
)

print("\nSample Neutral errors:")

for _, row in neutral_errors.head(30).iterrows():

    print(
        f"\nTRUE: {row['overall_sentiment']}"
        f"\nPRED: {row['predicted_sentiment']}"
        f"\nDOMAIN: {row['domain']}"
        f"\nTEXT: {row['text']}"
    )


# ============================================================
# 11. HARD EXAMPLES
# ============================================================

hard_errors = errors.copy()

hard_errors["text_length"] = (
    hard_errors["text"]
    .fillna("")
    .astype(str)
    .str.len()
)

hard_errors = hard_errors.sort_values(
    "text_length"
)

hard_errors[
    error_columns + ["text_length"]
].head(100).to_csv(
    OUTPUT_DIR / "hard_short_errors.csv",
    index=False,
)


# ============================================================
# 12. DUPLICATE-AWARE TEST SCORE
# ============================================================

duplicate_test_mask = test[
    "text_hash"
].astype(str).isin(overlap_hashes)

duplicate_test = test[
    duplicate_test_mask
    & test["overall_sentiment"].notna()
].copy()

duplicate_test["model_text"] = (
    duplicate_test["text_clean"]
    .fillna(duplicate_test["text"])
    .fillna("")
    .astype(str)
    .str.strip()
)

if len(duplicate_test):

    duplicate_pred = model.predict(
        duplicate_test["model_text"]
    )

    duplicate_test["prediction"] = duplicate_pred

    duplicate_accuracy = accuracy_score(
        duplicate_test["overall_sentiment"],
        duplicate_pred,
    )

else:

    duplicate_accuracy = None


# ============================================================
# 13. SUMMARY
# ============================================================

summary = {
    "phase": "9.2",
    "status": "completed",

    "overall": overall,

    "test_rows_evaluated": int(len(test_eval)),

    "cross_split_duplicate_hashes": int(
        len(overlap_hashes)
    ),

    "cross_split_duplicate_train_records": int(
        len(duplicate_rows)
    ),

    "cross_split_duplicate_test_records_with_labels": int(
        len(duplicate_test)
    ),

    "duplicate_test_accuracy": duplicate_accuracy,

    "total_errors": int(len(errors)),

    "errors_by_type": (
        errors["error_type"]
        .value_counts()
        .to_dict()
    ),

    "domain_performance": domain_df.to_dict(
        orient="records"
    ),

    "class_error_rates": class_error_df.to_dict(
        orient="records"
    ),
}

with open(
    OUTPUT_DIR / "phase9_2_summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# README
# ============================================================

readme = f"""# Phase 9.2 — Sentiment Error & Leakage Analysis

## Baseline

Model:
TF-IDF + Balanced Logistic Regression

Test Accuracy:
{accuracy:.4f}

Test Macro F1:
{f1_macro:.4f}

Test Weighted F1:
{f1_weighted:.4f}

## Cross-Split Duplicates

Unique overlapping text hashes:
{len(overlap_hashes)}

Train records involved:
{len(duplicate_rows)}

Test duplicate records with labels:
{len(duplicate_test)}

## Purpose

This phase investigates:

- Cross-split duplicate text
- Classification errors
- Neutral-class weakness
- Domain-specific performance
- Error types
- Train/test distribution

The original dataset is NOT modified.
"""

with open(
    OUTPUT_DIR / "PHASE_9_2_README.md",
    "w",
    encoding="utf-8",
) as f:

    f.write(readme)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("PHASE 9.2 COMPLETE")
print("=" * 70)

print(f"Output directory:")
print(OUTPUT_DIR)

print("\nImportant files:")

for filename in [
    "phase9_2_summary.json",
    "domain_performance.csv",
    "class_error_rates.csv",
    "all_misclassifications.csv",
    "neutral_misclassifications.csv",
    "cross_split_duplicates.csv",
    "confusion_matrix.csv",
]:
    print(f"  {filename}")

print("\nOriginal dataset was not modified.")