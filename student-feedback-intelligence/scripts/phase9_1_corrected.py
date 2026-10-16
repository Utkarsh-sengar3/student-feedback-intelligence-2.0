from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "edurabsa"
    / "train_phase2.csv"
)

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "edurabsa"
    / "test_phase2.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "edurabsa"
    / "phase9_1"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "sentiment"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def calculate_metrics(y_true, y_pred):
    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def build_pipeline(model):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=50000,
    )

    return Pipeline(
        [
            ("tfidf", vectorizer),
            ("classifier", model),
        ]
    )


# ============================================================
# LOAD DATA
# ============================================================

print_header("PHASE 9.1 — CORRECTED SENTIMENT MODEL BENCHMARK")

print(f"Project root : {PROJECT_ROOT}")
print(f"Train file   : {TRAIN_FILE}")
print(f"Test file    : {TEST_FILE}")

if not TRAIN_FILE.exists():
    raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")

if not TEST_FILE.exists():
    raise FileNotFoundError(f"Test file not found: {TEST_FILE}")


train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print()
print(f"Original train/validation rows : {len(train_df)}")
print(f"Original test rows             : {len(test_df)}")


# ============================================================
# SCHEMA VALIDATION
# ============================================================

required_columns = [
    "review_id",
    "original_id",
    "text",
    "domain",
    "overall_sentiment",
    "split",
    "text_clean",
    "text_hash",
]

missing_train = [
    c for c in required_columns
    if c not in train_df.columns
]

missing_test = [
    c for c in required_columns
    if c not in test_df.columns
]

if missing_train:
    raise ValueError(
        f"Missing required training columns: {missing_train}"
    )

if missing_test:
    raise ValueError(
        f"Missing required test columns: {missing_test}"
    )


# ============================================================
# DATA INTEGRITY / LEAKAGE AUDIT
# ============================================================

print_header("DATA INTEGRITY AUDIT")

audit = {}

audit["train_rows_original"] = int(len(train_df))
audit["test_rows_original"] = int(len(test_df))

audit["train_missing_labels"] = int(
    train_df["overall_sentiment"].isna().sum()
)

audit["train_empty_text"] = int(
    train_df["text_clean"]
    .fillna(train_df["text"])
    .fillna("")
    .astype(str)
    .str.strip()
    .eq("")
    .sum()
)

audit["test_empty_text"] = int(
    test_df["text_clean"]
    .fillna(test_df["text"])
    .fillna("")
    .astype(str)
    .str.strip()
    .eq("")
    .sum()
)

audit["train_duplicate_review_ids"] = int(
    train_df["review_id"].duplicated().sum()
)

audit["test_duplicate_review_ids"] = int(
    test_df["review_id"].duplicated().sum()
)

audit["review_id_overlap_train_test"] = int(
    len(
        set(train_df["review_id"].dropna())
        &
        set(test_df["review_id"].dropna())
    )
)

audit["original_id_overlap_train_test"] = int(
    len(
        set(train_df["original_id"].dropna())
        &
        set(test_df["original_id"].dropna())
    )
)

train_hashes = set(
    train_df["text_hash"]
    .dropna()
    .astype(str)
)

test_hashes = set(
    test_df["text_hash"]
    .dropna()
    .astype(str)
)

audit["text_hash_overlap_train_test"] = int(
    len(train_hashes & test_hashes)
)

if "cross_split_duplicate" in train_df.columns:
    audit["train_cross_split_duplicate_flags"] = int(
        train_df["cross_split_duplicate"]
        .fillna(False)
        .astype(bool)
        .sum()
    )

if "cross_split_duplicate" in test_df.columns:
    audit["test_cross_split_duplicate_flags"] = int(
        test_df["cross_split_duplicate"]
        .fillna(False)
        .astype(bool)
        .sum()
    )

for key, value in audit.items():
    print(f"{key}: {value}")


with open(
    OUTPUT_DIR / "leakage_audit.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(audit, f, indent=2)


# ============================================================
# PREPARE SENTIMENT DATA
# ============================================================

print_header("PREPARING SENTIMENT DATA")

df = train_df.copy()

# Remove rows without sentiment labels.
df = df.dropna(
    subset=["overall_sentiment"]
).copy()

# Prefer existing Phase-2 cleaned text.
df["model_text"] = (
    df["text_clean"]
    .fillna(df["text"])
    .fillna("")
    .astype(str)
    .str.strip()
)

# Remove genuinely empty text.
df = df[df["model_text"] != ""].copy()

df["overall_sentiment"] = (
    df["overall_sentiment"]
    .astype(str)
    .str.strip()
)

print(f"Labeled usable rows: {len(df)}")

print()
print("Sentiment distribution:")
print(df["overall_sentiment"].value_counts())


# ============================================================
# VALIDATE LABELS
# ============================================================

allowed_labels = {"Neg", "Neu", "Pos"}

unknown_labels = (
    set(df["overall_sentiment"].unique())
    - allowed_labels
)

if unknown_labels:
    raise ValueError(
        f"Unexpected sentiment labels found: {unknown_labels}"
    )


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X = df["model_text"]
y = df["overall_sentiment"]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=VALIDATION_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)

print_header("TRAIN / VALIDATION SPLIT")

print(f"Training rows   : {len(X_train)}")
print(f"Validation rows : {len(X_val)}")

print()
print("Training distribution:")
print(y_train.value_counts())

print()
print("Validation distribution:")
print(y_val.value_counts())


# ============================================================
# DEFINE MODELS
# ============================================================

models = {
    "TFIDF_LogisticRegression": LogisticRegression(
        max_iter=3000,
        C=1.0,
        class_weight=None,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    ),

    "TFIDF_LogisticRegression_Balanced": LogisticRegression(
        max_iter=3000,
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_STATE,
    ),

    "TFIDF_LinearSVM": LinearSVC(
        C=1.0,
        class_weight=None,
        random_state=RANDOM_STATE,
    ),

    "TFIDF_LinearSVM_Balanced": LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),

    "TFIDF_MultinomialNB": MultinomialNB(
        alpha=1.0
    ),
}


# ============================================================
# MODEL BENCHMARK
# ============================================================

print_header("VALIDATION MODEL COMPARISON")

results = []
validation_predictions = {}

for model_name, classifier in models.items():

    print()
    print(f"Training: {model_name}")

    pipeline = build_pipeline(classifier)

    pipeline.fit(X_train, y_train)

    val_pred = pipeline.predict(X_val)

    metrics = calculate_metrics(
        y_val,
        val_pred,
    )

    row = {
        "task": "sentiment",
        "model": model_name,
        **metrics,
    }

    results.append(row)
    validation_predictions[model_name] = val_pred

    print(
        f"Accuracy={metrics['accuracy']:.4f} | "
        f"Macro F1={metrics['f1_macro']:.4f} | "
        f"Weighted F1={metrics['f1_weighted']:.4f}"
    )


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=[
        "f1_macro",
        "f1_weighted",
        "accuracy",
    ],
    ascending=False,
).reset_index(drop=True)

results_df.to_csv(
    OUTPUT_DIR / "validation_model_comparison.csv",
    index=False,
)

print()
print(results_df.to_string(index=False))


# ============================================================
# SELECT BEST MODEL USING VALIDATION ONLY
# ============================================================

best_model_name = results_df.iloc[0]["model"]

print_header("MODEL SELECTION")

print(
    f"Selected model: {best_model_name}"
)

print(
    "Selection metric: Validation Macro F1"
)

best_validation_pred = validation_predictions[
    best_model_name
]

validation_report = classification_report(
    y_val,
    best_validation_pred,
    labels=["Neg", "Neu", "Pos"],
    output_dict=True,
    zero_division=0,
)

pd.DataFrame(validation_report).transpose().to_csv(
    OUTPUT_DIR / "best_validation_classification_report.csv"
)

validation_cm = confusion_matrix(
    y_val,
    best_validation_pred,
    labels=["Neg", "Neu", "Pos"],
)

pd.DataFrame(
    validation_cm,
    index=["Neg", "Neu", "Pos"],
    columns=["Neg", "Neu", "Pos"],
).to_csv(
    OUTPUT_DIR / "best_validation_confusion_matrix.csv"
)

pd.DataFrame(
    {
        "text": X_val,
        "true_sentiment": y_val,
        "predicted_sentiment": best_validation_pred,
    }
).to_csv(
    OUTPUT_DIR / "validation_predictions.csv",
    index=False,
)


# ============================================================
# RETRAIN WINNER ON ALL AVAILABLE LABELED TRAINING DATA
# ============================================================

print_header("RETRAINING BEST MODEL ON ALL LABELED TRAINING DATA")

best_classifier = models[best_model_name]

final_pipeline = build_pipeline(
    best_classifier
)

final_pipeline.fit(
    X,
    y,
)

model_path = (
    MODEL_DIR
    / "phase9_1_best_sentiment_model.joblib"
)

joblib.dump(
    final_pipeline,
    model_path,
)

print(f"Saved model: {model_path}")


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print_header("FINAL TEST EVALUATION")

test_model_text = (
    test_df["text_clean"]
    .fillna(test_df["text"])
    .fillna("")
    .astype(str)
    .str.strip()
)

test_valid_mask = test_model_text != ""

if not test_valid_mask.all():
    print(
        "WARNING: Some test rows contain empty text. "
        "They will be excluded from final evaluation."
    )

test_eval_df = test_df.loc[test_valid_mask].copy()
X_test = test_model_text.loc[test_valid_mask]

# Remove test records without a gold sentiment label.
# They cannot be used for supervised evaluation.
test_label_mask = test_eval_df["overall_sentiment"].notna()

test_eval_df = test_eval_df.loc[test_label_mask].copy()
X_test = X_test.loc[test_label_mask]

test_labels = (
    test_eval_df["overall_sentiment"]
    .astype(str)
    .str.strip()
)

print(
    f"Test rows with gold labels: {len(test_labels)}"
)

test_pred = final_pipeline.predict(X_test)

final_metrics = calculate_metrics(
    test_labels,
    test_pred,
)

final_metrics_row = {
    "task": "sentiment",
    "model": best_model_name,
    "selection_metric": "validation_macro_f1",
    **final_metrics,
}

pd.DataFrame(
    [final_metrics_row]
).to_csv(
    OUTPUT_DIR / "final_test_metrics.csv",
    index=False,
)


# ============================================================
# FINAL CLASSIFICATION REPORT
# ============================================================

final_report = classification_report(
    test_labels,
    test_pred,
    labels=["Neg", "Neu", "Pos"],
    output_dict=True,
    zero_division=0,
)

pd.DataFrame(final_report).transpose().to_csv(
    OUTPUT_DIR / "final_classification_report.csv"
)


# ============================================================
# FINAL CONFUSION MATRIX
# ============================================================

final_cm = confusion_matrix(
    test_labels,
    test_pred,
    labels=["Neg", "Neu", "Pos"],
)

pd.DataFrame(
    final_cm,
    index=["Neg", "Neu", "Pos"],
    columns=["Neg", "Neu", "Pos"],
).to_csv(
    OUTPUT_DIR / "final_confusion_matrix.csv"
)


# ============================================================
# FINAL TEST PREDICTIONS
# ============================================================

prediction_output = test_eval_df[
    [
        c for c in [
            "review_id",
            "original_id",
            "domain",
            "text",
            "overall_sentiment",
        ]
        if c in test_eval_df.columns
    ]
].copy()

prediction_output["predicted_sentiment"] = test_pred

prediction_output.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "phase": "9.1",
    "status": "completed",
    "selection_protocol": {
        "metric": "validation_macro_f1",
        "validation_size": VALIDATION_SIZE,
        "random_state": RANDOM_STATE,
        "test_used_for_model_selection": False,
    },
    "dataset": {
        "original_train_validation_rows": int(len(train_df)),
        "usable_labeled_training_rows": int(len(df)),
        "original_test_rows": int(len(test_df)),
        "evaluated_test_rows": int(len(test_eval_df)),
    },
    "best_model": best_model_name,
    "validation_results": results_df.to_dict(
        orient="records"
    ),
    "final_test_metrics": final_metrics_row,
    "model_path": str(model_path),
    "leakage_audit": audit,
}


with open(
    OUTPUT_DIR / "phase9_1_summary.json",
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

readme = f"""# Phase 9.1 — Corrected Sentiment Benchmark

## Objective

Establish a clean classical sentiment-analysis benchmark
without selecting the model using the held-out test set.

## Protocol

1. Load Phase-2 train/validation data.
2. Remove records without sentiment labels.
3. Use `text_clean` as model input.
4. Split labeled data into:
   - 80% training
   - 20% validation
5. Compare five TF-IDF classical models.
6. Select the winner using validation Macro F1.
7. Retrain the winner on all labeled training/validation data.
8. Evaluate exactly once on the held-out test set.

## Models

- TF-IDF + Logistic Regression
- TF-IDF + Balanced Logistic Regression
- TF-IDF + Linear SVM
- TF-IDF + Balanced Linear SVM
- TF-IDF + Multinomial Naive Bayes

## Selected Model

{best_model_name}

## Final Test Metrics

Accuracy: {final_metrics["accuracy"]:.6f}

Macro Precision: {final_metrics["precision_macro"]:.6f}

Macro Recall: {final_metrics["recall_macro"]:.6f}

Macro F1: {final_metrics["f1_macro"]:.6f}

Weighted F1: {final_metrics["f1_weighted"]:.6f}

## Important

The final test set was NOT used for model selection.

The previous Phase 9 benchmark should therefore be treated
as exploratory rather than as the final unbiased benchmark.
"""

with open(
    OUTPUT_DIR / "PHASE_9_1_README.md",
    "w",
    encoding="utf-8",
) as f:
    f.write(readme)


# ============================================================
# FINAL OUTPUT
# ============================================================

print_header("PHASE 9.1 COMPLETE")

print(f"Best model       : {best_model_name}")
print(
    f"Final Accuracy   : {final_metrics['accuracy']:.4f}"
)
print(
    f"Final Macro F1   : {final_metrics['f1_macro']:.4f}"
)
print(
    f"Final Weighted F1: {final_metrics['f1_weighted']:.4f}"
)

print()
print("Artifacts:")
print(OUTPUT_DIR)

print()
print("Model:")
print(model_path)

print()
print("IMPORTANT:")
print(
    "Model selection was performed using validation Macro F1."
)
print(
    "The test set was used only for final evaluation."
)