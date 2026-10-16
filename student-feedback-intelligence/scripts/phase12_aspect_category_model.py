"""
PHASE 12 — Aspect Normalization + Category Intelligence

Run from project root:
    ..\.venv\Scripts\python.exe -u scripts\phase12_aspect_category_model.py

Design:
1. Join canonical annotations to canonical reviews.
2. Create a transparent aspect-normalization layer.
3. Aggregate category annotations to review-level multi-label targets.
4. Split by review_id to prevent annotation leakage.
5. Compare TF-IDF + One-vs-Rest Logistic Regression and Linear SVM.
6. Evaluate with multilabel micro/macro F1, subset accuracy, Hamming loss,
   and per-category metrics.
7. Save the best category model.
8. Export normalized annotation data for downstream sentiment × aspect work.

Important:
- This phase does not use the sentiment test set for model selection.
- A review may legitimately contain multiple categories, so category
  intelligence is modeled as multilabel classification.
- Aspect extraction itself is NOT forced into a classifier here. The
  canonical aspect text is preserved and normalized for analytics.
"""

from pathlib import Path
import json
import re
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    classification_report,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "edurabsa"
MODELS = ROOT / "models" / "aspects"
OUT = DATA / "phase12"

ANN = DATA / "annotations_canonical.csv"
REV = DATA / "reviews_canonical.csv"

OUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def normalize_aspect(value):
    """
    Transparent normalization for analytics.

    We preserve the original aspect and add a canonicalized form.
    This is intentionally conservative: we do not invent semantic
    mappings for arbitrary phrases.
    """
    if pd.isna(value):
        return None

    s = str(value).strip()

    if not s or s.lower() == "null":
        return None

    s_clean = re.sub(r"\s+", " ", s).strip()
    low = s_clean.lower()

    # Conservative surface-form normalization.
    aliases = {
        "professor": "professor",
        "prof": "professor",
        "prof.": "professor",
        "professor": "professor",
        "teacher": "teacher",
        "lecturer": "lecturer",
        "course": "course",
        "this course": "course",
        "this class": "class",
        "class": "class",
        "classes": "class",
        "assignments": "assignment",
        "assignment": "assignment",
        "lectures": "lecture",
        "lecture": "lecture",
        "materials": "course materials",
        "material": "course materials",
    }

    if low in aliases:
        return aliases[low]

    # Pronouns are not useful canonical aspects.
    if low in {
        "he", "she", "her", "his", "him", "they", "them",
        "their", "hers", "this", "that"
    }:
        return None

    # Basic plural normalization only where safe.
    if low.endswith("ies") and len(low) > 4:
        low = low[:-3] + "y"
    elif low.endswith("s") and not low.endswith("ss") and len(low) > 4:
        low = low[:-1]

    return low


def build_review_level_categories(annotations):
    """
    Convert annotation-level categories to one multi-label target per review.
    """
    grouped = (
        annotations.groupby("review_id")["category"]
        .apply(lambda x: sorted(set(x.dropna().astype(str))))
        .reset_index(name="categories")
    )
    return grouped


def build_model():
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=40000,
    )

    return Pipeline([
        ("tfidf", vectorizer),
        ("clf", OneVsRestClassifier(
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                C=2.0,
                random_state=RANDOM_STATE,
            ),
            n_jobs=-1,
        )),
    ])


def build_svm():
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=40000,
    )

    return Pipeline([
        ("tfidf", vectorizer),
        ("clf", OneVsRestClassifier(
            LinearSVC(
                C=1.5,
                class_weight="balanced",
            ),
            n_jobs=-1,
        )),
    ])


def evaluate(name, model, X, y_true, labels):
    pred = model.predict(X)

    result = {
        "model": name,
        "subset_accuracy": accuracy_score(y_true, pred),
        "micro_precision": precision_score(
            y_true, pred, average="micro", zero_division=0
        ),
        "micro_recall": recall_score(
            y_true, pred, average="micro", zero_division=0
        ),
        "micro_f1": f1_score(
            y_true, pred, average="micro", zero_division=0
        ),
        "macro_precision": precision_score(
            y_true, pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, pred, average="macro", zero_division=0
        ),
        "hamming_loss": hamming_loss(y_true, pred),
    }

    report = classification_report(
        y_true,
        pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    rows = []
    for label in labels:
        rows.append({
            "category": label,
            "precision": report[label]["precision"],
            "recall": report[label]["recall"],
            "f1": report[label]["f1-score"],
            "support": report[label]["support"],
        })

    return result, pd.DataFrame(rows), pred


def main():

    print("=" * 78)
    print("PHASE 12 — ASPECT NORMALIZATION + CATEGORY INTELLIGENCE")
    print("=" * 78)

    if not ANN.exists():
        raise FileNotFoundError(f"Missing: {ANN}")

    if not REV.exists():
        raise FileNotFoundError(f"Missing: {REV}")

    ann = pd.read_csv(ANN)
    rev = pd.read_csv(REV)

    required_ann = {
        "review_id", "aspect", "opinion", "category", "sentiment"
    }
    required_rev = {
        "review_id", "text", "domain"
    }

    if not required_ann.issubset(ann.columns):
        raise ValueError(
            f"Annotation columns missing: "
            f"{sorted(required_ann - set(ann.columns))}"
        )

    if not required_rev.issubset(rev.columns):
        raise ValueError(
            f"Review columns missing: "
            f"{sorted(required_rev - set(rev.columns))}"
        )

    # ---------------------------------------------------------------
    # Annotation normalization
    # ---------------------------------------------------------------
    ann["aspect_original"] = ann["aspect"]

    ann["aspect_canonical"] = ann["aspect"].apply(
        normalize_aspect
    )

    ann["aspect_is_explicit"] = (
        ann["aspect_canonical"].notna()
    )

    ann.to_csv(
        OUT / "annotations_with_normalized_aspects.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Join review text
    # ---------------------------------------------------------------
    review_cols = [
        "review_id", "text", "domain"
    ]

    joined = ann.merge(
        rev[review_cols],
        on="review_id",
        how="left",
        validate="many_to_one",
    )

    missing_text = joined["text"].isna().sum()

    if missing_text:
        print(
            f"WARNING: {missing_text} annotation rows "
            "could not be matched to a review."
        )

    # ---------------------------------------------------------------
    # Build review-level multilabel category dataset
    # ---------------------------------------------------------------
    category_reviews = build_review_level_categories(
        joined
    )

    category_reviews = category_reviews.merge(
        rev[review_cols],
        on="review_id",
        how="left",
        validate="one_to_one",
    )

    category_reviews["text"] = (
        category_reviews["text"]
        .fillna("")
        .astype(str)
    )

    # We intentionally use review text as the primary category signal.
    # Domain is appended because the taxonomy explicitly distinguishes
    # Course / Staff / University categories.
    category_reviews["model_text"] = (
        category_reviews["domain"].fillna("")
        + " [DOMAIN] "
        + category_reviews["text"]
    )

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(
        category_reviews["categories"]
    )

    labels = list(mlb.classes_)

    print(
        f"\nReview-level category records: "
        f"{len(category_reviews):,}"
    )
    print(
        f"Categories: {len(labels)}"
    )

    # ---------------------------------------------------------------
    # Group-aware train/validation split
    # ---------------------------------------------------------------
    groups = category_reviews["review_id"].values

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    train_idx, val_idx = next(
        splitter.split(
            category_reviews["model_text"],
            Y,
            groups=groups,
        )
    )

    X_train = category_reviews.iloc[train_idx]["model_text"]
    X_val = category_reviews.iloc[val_idx]["model_text"]

    y_train = Y[train_idx]
    y_val = Y[val_idx]

    print(
        f"Training reviews: {len(train_idx):,}"
    )
    print(
        f"Validation reviews: {len(val_idx):,}"
    )

    # ---------------------------------------------------------------
    # Model comparison
    # ---------------------------------------------------------------
    models = {
        "TFIDF_OneVsRest_LogisticRegression": build_model(),
        "TFIDF_OneVsRest_LinearSVM": build_svm(),
    }

    results = []
    per_category = {}
    predictions = {}

    for name, model in models.items():

        print("\n" + "-" * 78)
        print(f"TRAINING: {name}")
        print("-" * 78)

        model.fit(X_train, y_train)

        result, metrics_df, pred = evaluate(
            name,
            model,
            X_val,
            y_val,
            labels,
        )

        results.append(result)
        per_category[name] = metrics_df
        predictions[name] = pred

        print(
            f"Subset accuracy : {result['subset_accuracy']:.4f}"
        )
        print(
            f"Micro F1        : {result['micro_f1']:.4f}"
        )
        print(
            f"Macro F1        : {result['macro_f1']:.4f}"
        )
        print(
            f"Hamming loss    : {result['hamming_loss']:.4f}"
        )

    results_df = (
        pd.DataFrame(results)
        .sort_values("macro_f1", ascending=False)
        .reset_index(drop=True)
    )

    results_df.to_csv(
        OUT / "category_model_comparison.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Select by Macro F1
    # ---------------------------------------------------------------
    best_name = results_df.iloc[0]["model"]
    best_model = models[best_name]

    print("\n" + "=" * 78)
    print("CATEGORY MODEL SELECTION")
    print("=" * 78)
    print(f"Best model: {best_name}")
    print(
        f"Validation Macro F1: "
        f"{results_df.iloc[0]['macro_f1']:.4f}"
    )

    best_metrics = per_category[best_name]
    best_metrics.to_csv(
        OUT / "category_class_metrics.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Retrain selected model on ALL canonical annotated reviews.
    #
    # This model is for production inference, not final scientific
    # test evaluation. The official sentiment test set remains untouched.
    # ---------------------------------------------------------------
    print(
        "\nRetraining selected category model "
        "on all annotated reviews..."
    )

    best_model.fit(
        category_reviews["model_text"],
        Y,
    )

    model_path = (
        MODELS /
        "phase12_category_classifier.joblib"
    )

    joblib.dump(
        {
            "model": best_model,
            "label_binarizer": mlb,
            "categories": labels,
            "feature_description": (
                "review text + domain marker"
            ),
            "split_protocol": (
                "GroupShuffleSplit by review_id "
                "for validation; final production "
                "fit uses all annotated reviews."
            ),
        },
        model_path,
    )

    # ---------------------------------------------------------------
    # Export normalized category dataset
    # ---------------------------------------------------------------
    category_reviews_export = category_reviews[
        [
            "review_id",
            "domain",
            "text",
            "categories",
        ]
    ].copy()

    category_reviews_export["categories"] = (
        category_reviews_export["categories"]
        .apply(lambda x: " | ".join(x))
    )

    category_reviews_export.to_csv(
        OUT / "review_category_multilabel.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    summary = {
        "phase": "12",
        "status": "completed",
        "task": "aspect normalization + category multilabel classification",
        "annotation_rows": int(len(ann)),
        "unique_reviews": int(ann["review_id"].nunique()),
        "unique_raw_aspects": int(
            ann["aspect"].dropna().astype(str).nunique()
        ),
        "unique_canonical_explicit_aspects": int(
            ann["aspect_canonical"].dropna().nunique()
        ),
        "null_or_non_explicit_aspects": int(
            (~ann["aspect_is_explicit"]).sum()
        ),
        "category_count": int(len(labels)),
        "categories": labels,
        "review_level_records": int(len(category_reviews)),
        "train_reviews": int(len(train_idx)),
        "validation_reviews": int(len(val_idx)),
        "best_model": best_name,
        "selection_metric": "validation macro F1",
        "best_validation_macro_f1": float(
            results_df.iloc[0]["macro_f1"]
        ),
        "model_comparison": results_df.to_dict(
            orient="records"
        ),
        "production_model": str(model_path),
        "important_design_decision": (
            "Category prediction is multilabel because a single "
            "student review can contain multiple annotated categories."
        ),
        "aspect_modeling_decision": (
            "Aspect extraction is kept separate from category "
            "classification. Raw aspect spans are preserved and "
            "a conservative normalization layer is added."
        ),
    }

    with open(
        OUT / "phase12_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    readme = f"""# Phase 12 — Aspect Normalization + Category Intelligence

## Status

Completed.

## Main decision

The EduRABSA annotations contain multiple categories for some reviews.
Therefore category intelligence is implemented as **multilabel
classification**, not forced single-label multiclass classification.

## Validation protocol

- Unit of split: review
- Split: GroupShuffleSplit
- Validation size: 20%
- Random seed: {RANDOM_STATE}
- Selection metric: Macro F1
- Test sentiment data: untouched

## Models compared

1. TF-IDF + One-vs-Rest Logistic Regression
2. TF-IDF + One-vs-Rest Linear SVM

## Best model

{best_name}

Validation Macro F1:

{results_df.iloc[0]["macro_f1"]:.4f}

## Aspect normalization

Original aspect text is preserved in `aspect_original`.

A conservative canonical form is stored in `aspect_canonical`.

The normalization layer intentionally does not invent semantic mappings
for arbitrary phrases.

## Outputs

- annotations_with_normalized_aspects.csv
- review_category_multilabel.csv
- category_model_comparison.csv
- category_class_metrics.csv
- phase12_summary.json

Production model:

`models/aspects/phase12_category_classifier.joblib`

## Next phase

Sentiment × aspect/category intelligence.

The next layer should combine:

- DistilBERT overall sentiment
- aspect
- category
- aspect sentiment
- review frequency
- affected-review count
- evidence text

Do not claim temporal emergence unless a reliable time field is established.
"""

    (OUT / "PHASE_12_README.md").write_text(
        readme,
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("PHASE 12 COMPLETE")
    print("=" * 78)
    print(
        f"Best model: {best_name}"
    )
    print(
        f"Validation Macro F1: "
        f"{results_df.iloc[0]['macro_f1']:.4f}"
    )
    print(
        f"Production model: {model_path}"
    )
    print(
        f"Artifacts: {OUT}"
    )


if __name__ == "__main__":
    main()
