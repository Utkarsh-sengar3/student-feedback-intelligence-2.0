"""
PHASE 9.4 — Hyperparameter Optimization
EduRABSA Student Feedback Sentiment Intelligence

Add to:
    student-feedback-intelligence/scripts/phase9_4_hyperparameter_optimization.py

Run from project root:
    python -u scripts\phase9_4_hyperparameter_optimization.py

Protocol:
- train_phase2.csv -> labeled train_validation source
- test_phase2.csv -> untouched final test
- 80/20 stratified validation split from train_validation
- Tune ONLY on validation Macro F1
- Retrain the winning configuration on all labeled train_validation rows
- Evaluate once on the untouched test set

This phase tunes the Phase 9.3 winning architecture:
    Word TF-IDF + Character TF-IDF + Balanced Logistic Regression

No test metric is used for selecting hyperparameters.
"""

from pathlib import Path
import json
import time
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

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "edurabsa" / "train_phase2.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "edurabsa" / "test_phase2.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "edurabsa" / "phase9_4"
MODEL_DIR = PROJECT_ROOT / "models" / "sentiment"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TEXT_COL = "text_clean"
LABEL_COL = "overall_sentiment"
CLASSES = ["Neg", "Neu", "Pos"]
RANDOM_STATE = 42

PHASE_9_3_MACRO_F1 = 0.7049

# ---------------------------------------------------------------------
# Tuning grid
#
# Kept deliberately controlled so the experiment is practical on a
# student laptop. All combinations are evaluated on validation only.
# ---------------------------------------------------------------------
CONFIGS = [
    # Baseline configuration from Phase 9.3
    {
        "name": "baseline_9_3",
        "word_ngram": (1, 2),
        "word_min_df": 2,
        "word_max_features": 30000,
        "char_ngram": (3, 5),
        "char_min_df": 2,
        "char_max_features": 40000,
        "C": 1.0,
    },

    # Logistic regression C
    {
        "name": "C_0_5",
        "word_ngram": (1, 2),
        "word_min_df": 2,
        "word_max_features": 30000,
        "char_ngram": (3, 5),
        "char_min_df": 2,
        "char_max_features": 40000,
        "C": 0.5,
    },
    {
        "name": "C_2",
        "word_ngram": (1, 2),
        "word_min_df": 2,
        "word_max_features": 30000,
        "char_ngram": (3, 5),
        "char_min_df": 2,
        "char_max_features": 40000,
        "C": 2.0,
    },
    {
        "name": "C_4",
        "word_ngram": (1, 2),
        "word_min_df": 2,
        "word_max_features": 30000,
        "char_ngram": (3, 5),
        "char_min_df": 2,
        "char_max_features": 40000,
        "C": 4.0,
    },

    # Word feature range
    {
        "name": "word_1_3",
        "word_ngram": (1, 3),
        "word_min_df": 2,
        "word_max_features": 40000,
        "char_ngram": (3, 5),
        "char_min_df": 2,
        "char_max_features": 40000,
        "C": 1.0,
    },
    {
        "name": "word_min_df_3",
        "word_ngram": (1, 2),
        "word_min_df": 3,
        "word_max_features": 30000,
        "char_ngram": (3, 5),
        "char_min_df": 2,
        "char_max_features": 40000,
        "C": 1.0,
    },
    {
        "name": "word_min_df_5",
        "word_ngram": (1, 2),
        "word_min_df": 5,
        "word_max_features": 30000,
        "char_ngram": (3, 5),
        "char_min_df": 2,
        "char_max_features": 40000,
        "C": 1.0,
    },

    # Character feature range
    {
        "name": "char_2_5",
        "word_ngram": (1, 2),
        "word_min_df": 2,
        "word_max_features": 30000,
        "char_ngram": (2, 5),
        "char_min_df": 2,
        "char_max_features": 40000,
        "C": 1.0,
    },
    {
        "name": "char_3_6",
        "word_ngram": (1, 2),
        "word_min_df": 2,
        "word_max_features": 30000,
        "char_ngram": (3, 6),
        "char_min_df": 2,
        "char_max_features": 40000,
        "C": 1.0,
    },
    {
        "name": "char_min_df_3",
        "word_ngram": (1, 2),
        "word_min_df": 2,
        "word_max_features": 30000,
        "char_ngram": (3, 5),
        "char_min_df": 3,
        "char_max_features": 40000,
        "C": 1.0,
    },
    {
        "name": "char_60000",
        "word_ngram": (1, 2),
        "word_min_df": 2,
        "word_max_features": 30000,
        "char_ngram": (3, 5),
        "char_min_df": 2,
        "char_max_features": 60000,
        "C": 1.0,
    },

    # Combined configurations
    {
        "name": "C_2_char_60000",
        "word_ngram": (1, 2),
        "word_min_df": 2,
        "word_max_features": 30000,
        "char_ngram": (3, 5),
        "char_min_df": 2,
        "char_max_features": 60000,
        "C": 2.0,
    },
    {
        "name": "word_1_3_C_2",
        "word_ngram": (1, 3),
        "word_min_df": 2,
        "word_max_features": 40000,
        "char_ngram": (3, 5),
        "char_min_df": 2,
        "char_max_features": 40000,
        "C": 2.0,
    },
]


def metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
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
        "precision_macro": float(p.mean()),
        "recall_macro": float(r.mean()),
        "f1_macro": float(f.mean()),
        "f1_weighted": float(weighted_f1),
    }


def make_word_vectorizer(cfg):
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        min_df=cfg["word_min_df"],
        max_df=0.95,
        ngram_range=cfg["word_ngram"],
        max_features=cfg["word_max_features"],
    )


def make_char_vectorizer(cfg):
    return TfidfVectorizer(
        analyzer="char",
        lowercase=True,
        sublinear_tf=True,
        min_df=cfg["char_min_df"],
        max_df=0.98,
        ngram_range=cfg["char_ngram"],
        max_features=cfg["char_max_features"],
    )


def main():

    print("=" * 78)
    print("PHASE 9.4 — HYPERPARAMETER OPTIMIZATION")
    print("=" * 78)

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training file not found:\n{TRAIN_PATH}")

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test file not found:\n{TEST_PATH}")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    required = {TEXT_COL, LABEL_COL}

    if not required.issubset(train.columns):
        raise ValueError(
            f"Training file missing columns: "
            f"{sorted(required - set(train.columns))}"
        )

    if not required.issubset(test.columns):
        raise ValueError(
            f"Test file missing columns: "
            f"{sorted(required - set(test.columns))}"
        )

    train = train[train[LABEL_COL].isin(CLASSES)].copy()
    test = test[test[LABEL_COL].isin(CLASSES)].copy()

    train[TEXT_COL] = train[TEXT_COL].fillna("").astype(str)
    test[TEXT_COL] = test[TEXT_COL].fillna("").astype(str)

    # ---------------------------------------------------------------
    # One fixed validation split for every configuration.
    # This makes the comparison fair.
    # ---------------------------------------------------------------
    train_part, val_part = train_test_split(
        train,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=train[LABEL_COL],
    )

    X_train = train_part[TEXT_COL].values
    y_train = train_part[LABEL_COL].values

    X_val = val_part[TEXT_COL].values
    y_val = val_part[LABEL_COL].values

    X_test = test[TEXT_COL].values
    y_test = test[LABEL_COL].values

    print(f"\nLabeled train_validation rows : {len(train)}")
    print(f"Internal training rows        : {len(train_part)}")
    print(f"Internal validation rows      : {len(val_part)}")
    print(f"Labeled final test rows       : {len(test)}")
    print(f"Configurations to evaluate    : {len(CONFIGS)}")

    print("\nIMPORTANT:")
    print("Test set is NOT used during hyperparameter selection.")

    results = []

    # ---------------------------------------------------------------
    # Validation tuning loop
    # ---------------------------------------------------------------
    for i, cfg in enumerate(CONFIGS, start=1):

        print("\n" + "-" * 78)
        print(
            f"[{i}/{len(CONFIGS)}] {cfg['name']}"
        )
        print(
            f"word={cfg['word_ngram']} "
            f"word_min_df={cfg['word_min_df']} "
            f"word_max={cfg['word_max_features']} | "
            f"char={cfg['char_ngram']} "
            f"char_min_df={cfg['char_min_df']} "
            f"char_max={cfg['char_max_features']} | "
            f"C={cfg['C']}"
        )

        start = time.time()

        word_vectorizer = make_word_vectorizer(cfg)
        char_vectorizer = make_char_vectorizer(cfg)

        Xtr_word = word_vectorizer.fit_transform(X_train)
        Xval_word = word_vectorizer.transform(X_val)

        Xtr_char = char_vectorizer.fit_transform(X_train)
        Xval_char = char_vectorizer.transform(X_val)

        Xtr = hstack(
            [Xtr_word, Xtr_char]
        ).tocsr()

        Xval = hstack(
            [Xval_word, Xval_char]
        ).tocsr()

        model = LogisticRegression(
            class_weight="balanced",
            max_iter=2500,
            C=cfg["C"],
            solver="liblinear",
            random_state=RANDOM_STATE,
        )

        model.fit(Xtr, y_train)

        prediction = model.predict(Xval)
        result_metrics = metrics(y_val, prediction)

        elapsed = time.time() - start

        result = {
            "configuration": cfg["name"],
            **cfg,
            **result_metrics,
            "word_features_actual": int(Xtr_word.shape[1]),
            "char_features_actual": int(Xtr_char.shape[1]),
            "combined_features_actual": int(Xtr.shape[1]),
            "runtime_seconds": round(elapsed, 2),
        }

        results.append(result)

        print(
            f"Validation Macro F1: "
            f"{result_metrics['f1_macro']:.6f}"
        )
        print(
            f"Validation Accuracy: "
            f"{result_metrics['accuracy']:.6f}"
        )
        print(
            f"Runtime: {elapsed:.1f}s"
        )

        # Save after every experiment in case a long run is interrupted.
        pd.DataFrame(results).sort_values(
            "f1_macro",
            ascending=False,
        ).to_csv(
            OUTPUT_DIR / "hyperparameter_search_progress.csv",
            index=False,
        )

    # ---------------------------------------------------------------
    # Select best configuration ONLY from validation
    # ---------------------------------------------------------------
    results_df = pd.DataFrame(results).sort_values(
        "f1_macro",
        ascending=False,
    ).reset_index(drop=True)

    results_df.insert(
        0,
        "validation_rank",
        range(1, len(results_df) + 1),
    )

    results_df.to_csv(
        OUTPUT_DIR / "hyperparameter_search_results.csv",
        index=False,
    )

    best = results_df.iloc[0].to_dict()

    print("\n" + "=" * 78)
    print("PHASE 9.4 VALIDATION RESULTS")
    print("=" * 78)

    print(
        results_df[
            [
                "validation_rank",
                "configuration",
                "accuracy",
                "f1_macro",
                "f1_weighted",
                "C",
                "word_ngram",
                "word_min_df",
                "char_ngram",
                "char_min_df",
                "char_max_features",
            ]
        ].to_string(index=False)
    )

    print("\nBEST CONFIGURATION:")
    print(best["configuration"])
    print(
        f"Validation Macro F1: "
        f"{best['f1_macro']:.6f}"
    )

    # ---------------------------------------------------------------
    # Final retraining on ALL labeled train_validation data
    # ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("RETRAINING BEST CONFIGURATION")
    print("=" * 78)

    best_cfg = {
        "word_ngram": tuple(best["word_ngram"]),
        "word_min_df": int(best["word_min_df"]),
        "word_max_features": int(best["word_max_features"]),
        "char_ngram": tuple(best["char_ngram"]),
        "char_min_df": int(best["char_min_df"]),
        "char_max_features": int(best["char_max_features"]),
        "C": float(best["C"]),
    }

    X_all = train[TEXT_COL].values
    y_all = train[LABEL_COL].values

    final_word_vectorizer = make_word_vectorizer(best_cfg)
    final_char_vectorizer = make_char_vectorizer(best_cfg)

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

    final_model = LogisticRegression(
        class_weight="balanced",
        max_iter=2500,
        C=best_cfg["C"],
        solver="liblinear",
        random_state=RANDOM_STATE,
    )

    final_model.fit(
        X_all_features,
        y_all,
    )

    # ---------------------------------------------------------------
    # FINAL TEST — first and only test evaluation
    # ---------------------------------------------------------------
    test_prediction = final_model.predict(
        X_test_features
    )

    final_metrics = metrics(
        y_test,
        test_prediction,
    )

    print("\n" + "=" * 78)
    print("FINAL TEST EVALUATION")
    print("=" * 78)

    print(
        f"Accuracy       : {final_metrics['accuracy']:.6f}"
    )
    print(
        f"Macro Precision: {final_metrics['precision_macro']:.6f}"
    )
    print(
        f"Macro Recall   : {final_metrics['recall_macro']:.6f}"
    )
    print(
        f"Macro F1       : {final_metrics['f1_macro']:.6f}"
    )
    print(
        f"Weighted F1    : {final_metrics['f1_weighted']:.6f}"
    )

    print(
        "\nChange vs Phase 9.3 Macro F1: "
        f"{final_metrics['f1_macro'] - PHASE_9_3_MACRO_F1:+.6f}"
    )

    # ---------------------------------------------------------------
    # Test artifacts
    # ---------------------------------------------------------------
    cm = confusion_matrix(
        y_test,
        test_prediction,
        labels=CLASSES,
    )

    pd.DataFrame(
        cm,
        index=CLASSES,
        columns=CLASSES,
    ).to_csv(
        OUTPUT_DIR / "final_test_confusion_matrix.csv"
    )

    report = classification_report(
        y_test,
        test_prediction,
        labels=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).T.to_csv(
        OUTPUT_DIR / "final_test_classification_report.csv"
    )

    prediction_columns = [
        c
        for c in [
            "review_id",
            "original_id",
            "domain",
            "text",
            "text_clean",
            LABEL_COL,
        ]
        if c in test.columns
    ]

    prediction_output = test[prediction_columns].copy()

    prediction_output[
        "predicted_sentiment"
    ] = test_prediction

    prediction_output.to_csv(
        OUTPUT_DIR / "final_test_predictions.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Save production candidate model
    # ---------------------------------------------------------------
    model_artifact = {
        "phase": "9.4",
        "type": "word_char_tfidf_logistic_regression",
        "config": best_cfg,
        "word_vectorizer": final_word_vectorizer,
        "char_vectorizer": final_char_vectorizer,
        "model": final_model,
    }

    model_path = (
        MODEL_DIR /
        "phase9_4_best_sentiment_model.joblib"
    )

    joblib.dump(
        model_artifact,
        model_path,
        compress=3,
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    summary = {
        "phase": "9.4",
        "status": "completed",
        "protocol": {
            "labeled_train_validation_rows": int(len(train)),
            "internal_train_rows": int(len(train_part)),
            "internal_validation_rows": int(len(val_part)),
            "labeled_test_rows_evaluated": int(len(test)),
            "random_state": RANDOM_STATE,
            "selection_metric": "validation_macro_f1",
            "test_used_for_hyperparameter_selection": False,
        },
        "search": {
            "configurations_evaluated": len(CONFIGS),
            "results": results_df.to_dict(
                orient="records"
            ),
        },
        "selected_configuration": best_cfg,
        "selected_configuration_name": best["configuration"],
        "selected_validation_metrics": {
            "accuracy": best["accuracy"],
            "precision_macro": best["precision_macro"],
            "recall_macro": best["recall_macro"],
            "f1_macro": best["f1_macro"],
            "f1_weighted": best["f1_weighted"],
        },
        "final_test_metrics": final_metrics,
        "comparison": {
            "phase9_3_macro_f1": PHASE_9_3_MACRO_F1,
            "phase9_4_macro_f1": final_metrics["f1_macro"],
            "macro_f1_change": (
                final_metrics["f1_macro"]
                - PHASE_9_3_MACRO_F1
            ),
        },
        "model_path": str(model_path),
    }

    summary_path = (
        OUTPUT_DIR /
        "phase9_4_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    readme = f"""# Phase 9.4 — Hyperparameter Optimization

## Objective

Optimize the Phase 9.3 winning architecture:

**Word TF-IDF + Character TF-IDF + Balanced Logistic Regression**

## Evaluation protocol

- Labeled train-validation rows: {len(train)}
- Internal train: {len(train_part)}
- Internal validation: {len(val_part)}
- Labeled final test: {len(test)}
- Selection metric: Validation Macro F1
- Test used during tuning: **NO**

## Configurations

{len(CONFIGS)} controlled configurations were evaluated.

## Selected configuration

`{best["configuration"]}`

Validation Macro F1:

`{best["f1_macro"]:.6f}`

## Final untouched test

Accuracy:

`{final_metrics["accuracy"]:.6f}`

Macro F1:

`{final_metrics["f1_macro"]:.6f}`

Weighted F1:

`{final_metrics["f1_weighted"]:.6f}`

## Phase 9.3 comparison

Phase 9.3 Macro F1:

`{PHASE_9_3_MACRO_F1:.6f}`

Phase 9.4 Macro F1:

`{final_metrics["f1_macro"]:.6f}`

Change:

`{final_metrics["f1_macro"] - PHASE_9_3_MACRO_F1:+.6f}`

## Artifacts

- `phase9_4_summary.json`
- `hyperparameter_search_results.csv`
- `hyperparameter_search_progress.csv`
- `final_test_confusion_matrix.csv`
- `final_test_classification_report.csv`
- `final_test_predictions.csv`
- `phase9_4_best_sentiment_model.joblib`
"""

    (OUTPUT_DIR / "PHASE_9_4_README.md").write_text(
        readme,
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("PHASE 9.4 COMPLETE")
    print("=" * 78)

    print(
        f"Best configuration    : {best['configuration']}"
    )
    print(
        f"Validation Macro F1   : {best['f1_macro']:.4f}"
    )
    print(
        f"Final Test Macro F1   : {final_metrics['f1_macro']:.4f}"
    )
    print(
        f"Change vs Phase 9.3   : "
        f"{final_metrics['f1_macro'] - PHASE_9_3_MACRO_F1:+.4f}"
    )

    print("\nSummary:")
    print(summary_path)

    print("\nModel:")
    print(model_path)


if __name__ == "__main__":
    main()
