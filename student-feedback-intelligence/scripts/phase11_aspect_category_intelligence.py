"""
PHASE 11 — Aspect / Category Intelligence Audit

This phase uses the canonical EduRABSA annotation table already created by
the project. It DOES NOT train a new model.

Run from project root:
    python -u scripts\phase11_aspect_category_intelligence.py

Input:
    data/processed/edurabsa/annotations_canonical.csv

Expected canonical fields:
    review_id, aspect, opinion, category, sentiment

Outputs:
    data/processed/edurabsa/phase11/
        phase11_summary.json
        category_summary.csv
        category_sentiment_matrix.csv
        aspect_summary.csv
        domain_category_summary.csv
        opinion_summary.csv
        PHASE_11_README.md

Why this phase exists:
The EduRABSA data already contains explicit aspect, opinion, category and
aspect-level sentiment annotations. Before training another classifier, we
should establish the real intelligence structure of the annotations.

No test-set predictions are used here and no model is retrained.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "edurabsa"
INPUT = DATA / "annotations_canonical.csv"
OUT = DATA / "phase11"
OUT.mkdir(parents=True, exist_ok=True)

REQUIRED = {
    "review_id",
    "aspect",
    "opinion",
    "category",
    "sentiment",
}

SENTIMENTS = ["Negative", "Neutral", "Positive"]


def main():

    print("=" * 78)
    print("PHASE 11 — ASPECT / CATEGORY INTELLIGENCE AUDIT")
    print("=" * 78)

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Canonical annotation file not found:\n{INPUT}"
        )

    df = pd.read_csv(INPUT)

    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    print(f"\nAnnotation rows: {len(df):,}")

    # ------------------------------------------------------------------
    # Normalize only for analysis.
    # Preserve the source file unchanged.
    # ------------------------------------------------------------------
    for col in ["aspect", "opinion", "category", "sentiment"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Treat explicit string null as missing aspect for reporting only.
    aspect_missing = (
        df["aspect"].eq("")
        | df["aspect"].str.lower().eq("null")
    )

    df["aspect_missing"] = aspect_missing

    # ------------------------------------------------------------------
    # Category summary
    # ------------------------------------------------------------------
    category = (
        df.groupby("category")
        .agg(
            annotations=("review_id", "size"),
            unique_reviews=("review_id", "nunique"),
            negative=("sentiment", lambda x: (x == "Negative").sum()),
            neutral=("sentiment", lambda x: (x == "Neutral").sum()),
            positive=("sentiment", lambda x: (x == "Positive").sum()),
        )
        .reset_index()
    )

    category["negative_rate"] = (
        category["negative"] / category["annotations"]
    )

    category["positive_rate"] = (
        category["positive"] / category["annotations"]
    )

    category["neutral_rate"] = (
        category["neutral"] / category["annotations"]
    )

    category["sentiment_balance"] = (
        category["positive_rate"]
        - category["negative_rate"]
    )

    category = category.sort_values(
        ["annotations", "negative_rate"],
        ascending=[False, False],
    )

    category.to_csv(
        OUT / "category_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Category x sentiment matrix
    # ------------------------------------------------------------------
    category_sentiment = pd.crosstab(
        df["category"],
        df["sentiment"],
    )

    for s in SENTIMENTS:
        if s not in category_sentiment.columns:
            category_sentiment[s] = 0

    category_sentiment = category_sentiment[
        SENTIMENTS
    ]

    category_sentiment["Total"] = (
        category_sentiment.sum(axis=1)
    )

    category_sentiment["Negative_Rate"] = (
        category_sentiment["Negative"]
        / category_sentiment["Total"]
    )

    category_sentiment["Positive_Rate"] = (
        category_sentiment["Positive"]
        / category_sentiment["Total"]
    )

    category_sentiment = category_sentiment.sort_values(
        "Negative_Rate",
        ascending=False,
    )

    category_sentiment.to_csv(
        OUT / "category_sentiment_matrix.csv"
    )

    # ------------------------------------------------------------------
    # Aspect summary
    # ------------------------------------------------------------------
    explicit_aspects = df[
        ~df["aspect_missing"]
    ].copy()

    aspect = (
        explicit_aspects.groupby("aspect")
        .agg(
            annotations=("review_id", "size"),
            unique_reviews=("review_id", "nunique"),
            categories=("category", "nunique"),
            negative=("sentiment", lambda x: (x == "Negative").sum()),
            neutral=("sentiment", lambda x: (x == "Neutral").sum()),
            positive=("sentiment", lambda x: (x == "Positive").sum()),
        )
        .reset_index()
    )

    aspect["negative_rate"] = (
        aspect["negative"] / aspect["annotations"]
    )

    aspect["positive_rate"] = (
        aspect["positive"] / aspect["annotations"]
    )

    aspect["neutral_rate"] = (
        aspect["neutral"] / aspect["annotations"]
    )

    aspect["impact_count"] = (
        aspect["negative"] * 2
        + aspect["positive"]
    )

    aspect = aspect.sort_values(
        ["annotations", "negative_rate"],
        ascending=[False, False],
    )

    aspect.to_csv(
        OUT / "aspect_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Domain/category summary if domain exists.
    # ------------------------------------------------------------------
    domain_category = pd.DataFrame()

    if "domain" in df.columns:

        domain_category = (
            df.groupby(["domain", "category"])
            .agg(
                annotations=("review_id", "size"),
                unique_reviews=("review_id", "nunique"),
                negative=("sentiment", lambda x: (x == "Negative").sum()),
                neutral=("sentiment", lambda x: (x == "Neutral").sum()),
                positive=("sentiment", lambda x: (x == "Positive").sum()),
            )
            .reset_index()
        )

        domain_category["negative_rate"] = (
            domain_category["negative"]
            / domain_category["annotations"]
        )

        domain_category = domain_category.sort_values(
            ["domain", "negative_rate", "annotations"],
            ascending=[True, False, False],
        )

    domain_category.to_csv(
        OUT / "domain_category_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Opinion summary
    # ------------------------------------------------------------------
    opinion = (
        explicit_aspects.groupby("opinion")
        .agg(
            annotations=("review_id", "size"),
            unique_reviews=("review_id", "nunique"),
            negative=("sentiment", lambda x: (x == "Negative").sum()),
            neutral=("sentiment", lambda x: (x == "Neutral").sum()),
            positive=("sentiment", lambda x: (x == "Positive").sum()),
        )
        .reset_index()
        .sort_values("annotations", ascending=False)
    )

    opinion.to_csv(
        OUT / "opinion_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Key findings
    # ------------------------------------------------------------------
    top_categories = category.head(15)[
        [
            "category",
            "annotations",
            "unique_reviews",
            "negative_rate",
            "positive_rate",
        ]
    ]

    highest_negative_categories = (
        category[
            category["annotations"] >= 30
        ]
        .sort_values(
            "negative_rate",
            ascending=False,
        )
        .head(15)
    )

    most_frequent_aspects = aspect.head(20)[
        [
            "aspect",
            "annotations",
            "unique_reviews",
            "negative_rate",
            "positive_rate",
        ]
    ]

    highest_negative_aspects = (
        aspect[
            aspect["annotations"] >= 10
        ]
        .sort_values(
            "negative_rate",
            ascending=False,
        )
        .head(20)
    )

    # ------------------------------------------------------------------
    # Summary JSON
    # ------------------------------------------------------------------
    summary = {
        "phase": "11",
        "status": "completed",
        "purpose": (
            "Establish the empirical aspect/category structure "
            "before further modeling."
        ),
        "annotation_rows": int(len(df)),
        "unique_reviews": int(df["review_id"].nunique()),
        "unique_categories": int(df["category"].nunique()),
        "unique_explicit_aspects": int(
            explicit_aspects["aspect"].nunique()
        ),
        "unique_opinions": int(
            explicit_aspects["opinion"].nunique()
        ),
        "missing_or_null_aspects": int(
            aspect_missing.sum()
        ),
        "sentiment_distribution": {
            str(k): int(v)
            for k, v in df["sentiment"].value_counts().to_dict().items()
        },
        "top_categories": top_categories.to_dict(
            orient="records"
        ),
        "highest_negative_categories_min_30_annotations":
            highest_negative_categories[
                [
                    "category",
                    "annotations",
                    "negative_rate",
                    "positive_rate",
                ]
            ].to_dict(orient="records"),
        "most_frequent_aspects": most_frequent_aspects.to_dict(
            orient="records"
        ),
        "highest_negative_aspects_min_10_annotations":
            highest_negative_aspects[
                [
                    "aspect",
                    "annotations",
                    "negative_rate",
                    "positive_rate",
                ]
            ].to_dict(orient="records"),
        "modeling_recommendation": (
            "Use the annotated categories as supervised labels. "
            "Treat aspect extraction as a separate span-level task. "
            "Do not collapse aspect and category into one target."
        ),
    }

    with open(
        OUT / "phase11_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------------
    # README
    # ------------------------------------------------------------------
    readme = f"""# Phase 11 — Aspect / Category Intelligence Audit

## Status

Completed as a data-intelligence audit. No model training was performed.

## Dataset

Canonical annotation rows: {len(df):,}

Unique reviews: {df["review_id"].nunique():,}

Unique categories: {df["category"].nunique():,}

Unique explicit aspects: {explicit_aspects["aspect"].nunique():,}

Unique opinions: {explicit_aspects["opinion"].nunique():,}

Missing/null aspects: {aspect_missing.sum():,}

## Why this phase matters

EduRABSA contains explicit quadruplet-style information:

- aspect
- opinion
- category
- aspect sentiment

The project should preserve these as separate concepts.

## Modeling decision

### Category classification

This is a conventional supervised classification task because the dataset
contains a finite set of labeled categories.

### Aspect extraction

This is a span/extraction task, not simply a 24-class classification task.
The model should identify the aspect phrase in the review.

### Aspect sentiment

The annotation already provides sentiment attached to each aspect/opinion
annotation. This supports later sentiment × aspect intelligence.

## Important limitation

The dataset does not by itself establish a reliable time axis for all
records. Therefore this phase does not claim temporal trends or "emerging"
concerns.

## Output files

- category_summary.csv
- category_sentiment_matrix.csv
- aspect_summary.csv
- domain_category_summary.csv
- opinion_summary.csv
- phase11_summary.json
"""

    (OUT / "PHASE_11_README.md").write_text(
        readme,
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("PHASE 11 COMPLETE")
    print("=" * 78)

    print(f"\nAnnotation rows: {len(df):,}")
    print(
        f"Unique reviews: {df['review_id'].nunique():,}"
    )
    print(
        f"Unique categories: {df['category'].nunique():,}"
    )
    print(
        f"Unique explicit aspects: "
        f"{explicit_aspects['aspect'].nunique():,}"
    )
    print(
        f"Missing/null aspects: "
        f"{aspect_missing.sum():,}"
    )

    print("\nSentiment distribution:")
    print(
        df["sentiment"]
        .value_counts()
        .to_string()
    )

    print("\nTop categories:")
    print(
        top_categories.to_string(index=False)
    )

    print("\nHighest negative categories "
          "(minimum 30 annotations):")
    print(
        highest_negative_categories[
            [
                "category",
                "annotations",
                "negative_rate",
            ]
        ].to_string(index=False)
    )

    print("\nTop explicit aspects:")
    print(
        most_frequent_aspects.to_string(index=False)
    )

    print("\nOutput:")
    print(OUT)


if __name__ == "__main__":
    main()
