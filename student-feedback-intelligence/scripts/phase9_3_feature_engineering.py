"""
PHASE 9.3 — Feature Engineering
AI-Based Student Feedback Sentiment Intelligence / EduRABSA

Add this file to:
    student-feedback-intelligence/scripts/phase9_3_feature_engineering.py

Run from project root:
    python scripts\phase9_3_feature_engineering.py

Protocol:
- Use only the labeled train_validation data for model selection.
- Create an 80/20 stratified train/validation split.
- Keep dev_test completely untouched during model selection.
- Select the best configuration using validation Macro F1.
- Retrain the selected configuration on all labeled train_validation rows.
- Evaluate exactly once on the labeled dev_test rows.

Compared:
1. Word TF-IDF + balanced Logistic Regression
2. Character TF-IDF + balanced Logistic Regression
3. Word + Character TF-IDF + balanced Logistic Regression
4. Word + Character TF-IDF + balanced Linear SVM
"""

from pathlib import Path
import json
import warnings

import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "edurabsa" / "train_phase2.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "edurabsa" / "test_phase2.csv"

OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "edurabsa" / "phase9_3"
)

MODEL_DIR = PROJECT_ROOT / "models" / "sentiment"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TEXT_COL = "text_clean"
LABEL_COL = "overall_sentiment"
CLASSES = ["Neg", "Neu", "Pos"]
RANDOM_STATE = 42

PHASE_9_1_MACRO_F1 = 0.6954972002473875


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def calculate_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASSES,
        zero_division=0,
    )

    weighted_f1 = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASSES,
        average="weighted",
        zero_division=0,
    )[2]

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "f1_weighted": float(weighted_f1),
    }


# ---------------------------------------------------------------------
# Vectorizers
# ---------------------------------------------------------------------
def create_word_vectorizer():
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        max_features=30000,
    )


def create_char_vectorizer():
    return TfidfVectorizer(
        analyzer="char",
        lowercase=True,
        sublinear_tf=True,
        min_df=2,
        max_df=0.98,
        ngram_range=(3, 5),
        max_features=40000,
    )


def create_logistic_regression():
    return LogisticRegression(
        class_weight="balanced",
        max_iter=2500,
        C=1.0,
        solver="liblinear",
        random_state=RANDOM_STATE,
    )


def create_svm():
    return LinearSVC(
        class_weight="balanced",
        C=1.0,
        random_state=RANDOM_STATE,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():

    print("=" * 78)
    print("PHASE 9.3 — FEATURE ENGINEERING")
    print("=" * 78)

    # ---------------------------------------------------------------
    # Validate files
    # ---------------------------------------------------------------
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"\nTraining file not found:\n{TRAIN_PATH}\n"
            "Make sure you run this script from your project root."
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"\nTest file not found:\n{TEST_PATH}"
        )

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    required_columns = {TEXT_COL, LABEL_COL}

    missing_train = required_columns - set(train.columns)
    missing_test = required_columns - set(test.columns)

    if missing_train:
        raise ValueError(
            f"Training file is missing columns: {sorted(missing_train)}"
        )

    if missing_test:
        raise ValueError(
            f"Test file is missing columns: {sorted(missing_test)}"
        )

    # Only labeled sentiment rows participate in supervised evaluation.
    train = train[train[LABEL_COL].isin(CLASSES)].copy()
    test_eval = test[test[LABEL_COL].isin(CLASSES)].copy()

    train[TEXT_COL] = train[TEXT_COL].fillna("").astype(str)
    test_eval[TEXT_COL] = test_eval[TEXT_COL].fillna("").astype(str)

    # ---------------------------------------------------------------
    # Leakage-safe validation split
    # ---------------------------------------------------------------
    train_split, validation_split = train_test_split(
        train,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=train[LABEL_COL],
    )

    X_train = train_split[TEXT_COL].values
    y_train = train_split[LABEL_COL].values

    X_val = validation_split[TEXT_COL].values
    y_val = validation_split[LABEL_COL].values

    X_test = test_eval[TEXT_COL].values
    y_test = test_eval[LABEL_COL].values

    print(f"\nLabeled train_validation rows : {len(train)}")
    print(f"Internal training rows        : {len(train_split)}")
    print(f"Internal validation rows      : {len(validation_split)}")
    print(f"Labeled test rows             : {len(test_eval)}")

    print("\nValidation label distribution:")
    print(validation_split[LABEL_COL].value_counts().sort_index())

    experiments = []

    # ===============================================================
    # EXPERIMENT 1 — WORD TF-IDF + LOGISTIC REGRESSION
    # ===============================================================
    print("\n[1/4] Word TF-IDF + Balanced Logistic Regression")

    word_vectorizer = create_word_vectorizer()

    Xtr_word = word_vectorizer.fit_transform(X_train)
    Xval_word = word_vectorizer.transform(X_val)

    word_lr = create_logistic_regression()
    word_lr.fit(Xtr_word, y_train)

    word_pred = word_lr.predict(Xval_word)
    word_metrics = calculate_metrics(y_val, word_pred)

    experiments.append(
        {
            "model": "Word_TFIDF_Balanced_LogisticRegression",
            **word_metrics,
            "feature_count": int(Xtr_word.shape[1]),
        }
    )

    print(word_metrics)

    # ===============================================================
    # EXPERIMENT 2 — CHARACTER TF-IDF + LOGISTIC REGRESSION
    # ===============================================================
    print("\n[2/4] Character TF-IDF + Balanced Logistic Regression")

    char_vectorizer = create_char_vectorizer()

    Xtr_char = char_vectorizer.fit_transform(X_train)
    Xval_char = char_vectorizer.transform(X_val)

    char_lr = create_logistic_regression()
    char_lr.fit(Xtr_char, y_train)

    char_pred = char_lr.predict(Xval_char)
    char_metrics = calculate_metrics(y_val, char_pred)

    experiments.append(
        {
            "model": "Char_TFIDF_Balanced_LogisticRegression",
            **char_metrics,
            "feature_count": int(Xtr_char.shape[1]),
        }
    )

    print(char_metrics)

    # ===============================================================
    # EXPERIMENT 3 — WORD + CHARACTER + LOGISTIC REGRESSION
    # ===============================================================
    print("\n[3/4] Word + Character TF-IDF + Balanced Logistic Regression")

    Xtr_word_char = hstack(
        [Xtr_word, Xtr_char]
    ).tocsr()

    Xval_word_char = hstack(
        [Xval_word, Xval_char]
    ).tocsr()

    word_char_lr = create_logistic_regression()
    word_char_lr.fit(Xtr_word_char, y_train)

    word_char_pred = word_char_lr.predict(Xval_word_char)
    word_char_metrics = calculate_metrics(
        y_val,
        word_char_pred,
    )

    experiments.append(
        {
            "model": "WordChar_TFIDF_Balanced_LogisticRegression",
            **word_char_metrics,
            "feature_count": int(Xtr_word_char.shape[1]),
        }
    )

    print(word_char_metrics)

    # ===============================================================
    # EXPERIMENT 4 — WORD + CHARACTER + LINEAR SVM
    # ===============================================================
    print("\n[4/4] Word + Character TF-IDF + Balanced Linear SVM")

    word_char_svm = create_svm()
    word_char_svm.fit(Xtr_word_char, y_train)

    svm_pred = word_char_svm.predict(Xval_word_char)
    svm_metrics = calculate_metrics(
        y_val,
        svm_pred,
    )

    experiments.append(
        {
            "model": "WordChar_TFIDF_Balanced_LinearSVM",
            **svm_metrics,
            "feature_count": int(Xtr_word_char.shape[1]),
        }
    )

    print(svm_metrics)

    # ---------------------------------------------------------------
    # Validation comparison
    # ---------------------------------------------------------------
    results = pd.DataFrame(experiments)

    results = results.sort_values(
        by="f1_macro",
        ascending=False,
    ).reset_index(drop=True)

    results.insert(
        0,
        "validation_rank",
        range(1, len(results) + 1),
    )

    results.to_csv(
        OUTPUT_DIR / "validation_model_comparison.csv",
        index=False,
    )

    best_model_name = results.iloc[0]["model"]

    print("\n" + "=" * 78)
    print("VALIDATION MODEL SELECTION")
    print("=" * 78)

    print(results.to_string(index=False))

    print(f"\nSelected model: {best_model_name}")
    print("Selection metric: VALIDATION MACRO F1")
    print("Test set used for selection: NO")

    # ===============================================================
    # RETRAIN SELECTED MODEL ON ALL LABELED TRAIN-VALIDATION DATA
    # ===============================================================
    print("\nRetraining selected configuration on all labeled train_validation data...")

    X_all = train[TEXT_COL].values
    y_all = train[LABEL_COL].values

    if best_model_name == "Word_TFIDF_Balanced_LogisticRegression":

        final_word_vectorizer = create_word_vectorizer()

        X_all_features = final_word_vectorizer.fit_transform(X_all)
        X_test_features = final_word_vectorizer.transform(X_test)

        final_model = create_logistic_regression()
        final_model.fit(X_all_features, y_all)

        test_pred = final_model.predict(X_test_features)

        artifact = {
            "type": "word_tfidf_logistic_regression",
            "vectorizer": final_word_vectorizer,
            "model": final_model,
        }

    elif best_model_name == "Char_TFIDF_Balanced_LogisticRegression":

        final_char_vectorizer = create_char_vectorizer()

        X_all_features = final_char_vectorizer.fit_transform(X_all)
        X_test_features = final_char_vectorizer.transform(X_test)

        final_model = create_logistic_regression()
        final_model.fit(X_all_features, y_all)

        test_pred = final_model.predict(X_test_features)

        artifact = {
            "type": "char_tfidf_logistic_regression",
            "vectorizer": final_char_vectorizer,
            "model": final_model,
        }

    elif best_model_name == "WordChar_TFIDF_Balanced_LogisticRegression":

        final_word_vectorizer = create_word_vectorizer()
        final_char_vectorizer = create_char_vectorizer()

        X_all_word = final_word_vectorizer.fit_transform(X_all)
        X_test_word = final_word_vectorizer.transform(X_test)

        X_all_char = final_char_vectorizer.fit_transform(X_all)
        X_test_char = final_char_vectorizer.transform(X_test)

        X_all_features = hstack(
            [X_all_word, X_all_char]
        ).tocsr()

        X_test_features = hstack(
            [X_test_word, X_test_char]
        ).tocsr()

        final_model = create_logistic_regression()
        final_model.fit(X_all_features, y_all)

        test_pred = final_model.predict(X_test_features)

        artifact = {
            "type": "word_char_tfidf_logistic_regression",
            "word_vectorizer": final_word_vectorizer,
            "char_vectorizer": final_char_vectorizer,
            "model": final_model,
        }

    elif best_model_name == "WordChar_TFIDF_Balanced_LinearSVM":

        final_word_vectorizer = create_word_vectorizer()
        final_char_vectorizer = create_char_vectorizer()

        X_all_word = final_word_vectorizer.fit_transform(X_all)
        X_test_word = final_word_vectorizer.transform(X_test)

        X_all_char = final_char_vectorizer.fit_transform(X_all)
        X_test_char = final_char_vectorizer.transform(X_test)

        X_all_features = hstack(
            [X_all_word, X_all_char]
        ).tocsr()

        X_test_features = hstack(
            [X_test_word, X_test_char]
        ).tocsr()

        final_model = create_svm()
        final_model.fit(X_all_features, y_all)

        test_pred = final_model.predict(X_test_features)

        artifact = {
            "type": "word_char_tfidf_linear_svm",
            "word_vectorizer": final_word_vectorizer,
            "char_vectorizer": final_char_vectorizer,
            "model": final_model,
        }

    else:
        raise RuntimeError(
            f"Unexpected model name: {best_model_name}"
        )

    # ===============================================================
    # FINAL TEST EVALUATION
    # ===============================================================
    final_metrics = calculate_metrics(
        y_test,
        test_pred,
    )

    report = classification_report(
        y_test,
        test_pred,
        labels=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_test,
        test_pred,
        labels=CLASSES,
    )

    pd.DataFrame(
        cm,
        index=CLASSES,
        columns=CLASSES,
    ).to_csv(
        OUTPUT_DIR / "final_test_confusion_matrix.csv"
    )

    pd.DataFrame(report).T.to_csv(
        OUTPUT_DIR / "final_test_classification_report.csv"
    )

    # Save predictions for Phase 9.4/error analysis.
    prediction_columns = [
        column
        for column in [
            "review_id",
            "original_id",
            "domain",
            "text",
            "text_clean",
            "overall_sentiment",
        ]
        if column in test_eval.columns
    ]

    prediction_output = test_eval[prediction_columns].copy()
    prediction_output["predicted_sentiment"] = test_pred

    prediction_output.to_csv(
        OUTPUT_DIR / "final_test_predictions.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Save model
    # ---------------------------------------------------------------
    model_path = (
        MODEL_DIR / "phase9_3_best_sentiment_model.joblib"
    )

    joblib.dump(
        artifact,
        model_path,
        compress=3,
    )

    # ---------------------------------------------------------------
    # Summary JSON
    # ---------------------------------------------------------------
    summary = {
        "phase": "9.3",
        "status": "completed",
        "selection_protocol": {
            "labeled_train_validation_rows": int(len(train)),
            "internal_train_rows": int(len(train_split)),
            "internal_validation_rows": int(len(validation_split)),
            "labeled_test_rows_evaluated": int(len(test_eval)),
            "random_state": RANDOM_STATE,
            "selection_metric": "validation_macro_f1",
            "test_used_for_model_selection": False,
        },
        "validation_results": results.to_dict(
            orient="records"
        ),
        "selected_model": best_model_name,
        "final_test": final_metrics,
        "phase9_1_baseline": {
            "macro_f1": PHASE_9_1_MACRO_F1
        },
        "improvement_vs_phase9_1": {
            "macro_f1_difference": float(
                final_metrics["f1_macro"]
                - PHASE_9_1_MACRO_F1
            ),
            "accuracy_difference": None,
        },
        "artifacts": {
            "validation_comparison": str(
                OUTPUT_DIR / "validation_model_comparison.csv"
            ),
            "test_confusion_matrix": str(
                OUTPUT_DIR / "final_test_confusion_matrix.csv"
            ),
            "test_classification_report": str(
                OUTPUT_DIR / "final_test_classification_report.csv"
            ),
            "test_predictions": str(
                OUTPUT_DIR / "final_test_predictions.csv"
            ),
            "model": str(model_path),
        },
    }

    summary_path = OUTPUT_DIR / "phase9_3_summary.json"

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    # ---------------------------------------------------------------
    # README
    # ---------------------------------------------------------------
    readme = f"""# Phase 9.3 — Feature Engineering

## Status

Completed when this script finishes successfully.

## Objective

Improve the Phase 9.1 classical sentiment model using richer TF-IDF
representations while preserving a leakage-safe evaluation protocol.

## Compared configurations

1. Word TF-IDF + Balanced Logistic Regression
2. Character TF-IDF + Balanced Logistic Regression
3. Word + Character TF-IDF + Balanced Logistic Regression
4. Word + Character TF-IDF + Balanced Linear SVM

## Model selection

The 3,999 labeled rows from `train_validation` are split into:

- 3,199 internal training rows
- 800 validation rows

The exact counts are generated by the script and recorded in
`phase9_3_summary.json`.

The model is selected using **validation Macro F1**.

The 2,499 labeled `dev_test` rows are NOT used during model selection.

After selection, the winning configuration is retrained on all labeled
train-validation rows and evaluated once on the test set.

## Phase 9.1 reference

Macro F1:

`0.6954972002473875`

## Selected model

`{best_model_name}`

## Final test results

Accuracy:
`{final_metrics["accuracy"]:.6f}`

Macro Precision:
`{final_metrics["precision_macro"]:.6f}`

Macro Recall:
`{final_metrics["recall_macro"]:.6f}`

Macro F1:
`{final_metrics["f1_macro"]:.6f}`

Weighted F1:
`{final_metrics["f1_weighted"]:.6f}`

## Generated artifacts

- `phase9_3_summary.json`
- `validation_model_comparison.csv`
- `final_test_confusion_matrix.csv`
- `final_test_classification_report.csv`
- `final_test_predictions.csv`
- `phase9_3_best_sentiment_model.joblib`
"""

    (OUTPUT_DIR / "PHASE_9_3_README.md").write_text(
        readme,
        encoding="utf-8",
    )

    # ---------------------------------------------------------------
    # Final console output
    # ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 9.3 COMPLETE")
    print("=" * 78)
    print(f"Selected model       : {best_model_name}")
    print(f"Final Accuracy       : {final_metrics['accuracy']:.4f}")
    print(f"Final Macro F1       : {final_metrics['f1_macro']:.4f}")
    print(f"Final Weighted F1    : {final_metrics['f1_weighted']:.4f}")
    print(
        "Change vs Phase 9.1 : "
        f"{final_metrics['f1_macro'] - PHASE_9_1_MACRO_F1:+.4f}"
    )

    print("\nSummary:")
    print(summary_path)

    print("\nModel:")
    print(model_path)


if __name__ == "__main__":
    main()
