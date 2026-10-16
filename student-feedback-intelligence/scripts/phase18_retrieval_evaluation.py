"""
Phase 18 — Retrieval Evaluation

Evaluates the finalized reusable SemanticRetriever using the same
12-query weakly-supervised benchmark established for Phase 18.

Metrics:
- Precision@K
- Recall@K
- MRR
- NDCG@K

Important:
The benchmark is weakly supervised. Relevance is derived from available
EduRABSA metadata (domain, sentiment, and intent terms).

These metrics are NOT human-judged relevance scores.
"""

from pathlib import Path
import json
import re
import sys

import numpy as np
import pandas as pd


# ================================================================
# PROJECT ROOT
# ================================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ================================================================
# IMPORT REUSABLE RETRIEVER
# ================================================================

from src.retrieval.retriever import SemanticRetriever


# ================================================================
# PATHS
# ================================================================

PHASE17 = (
    ROOT
    / "data"
    / "processed"
    / "edurabsa"
    / "phase17"
)

OUT = (
    ROOT
    / "data"
    / "processed"
    / "edurabsa"
    / "phase18"
)

OUT.mkdir(parents=True, exist_ok=True)


# ================================================================
# BENCHMARK QUERIES
# ================================================================

QUERIES = [
    (
        "Q01",
        "negative feedback about course workload",
        "course review",
        "negative",
        [
            "workload",
            "course",
            "assignment",
            "work",
            "load",
        ],
    ),
    (
        "Q02",
        "complaints about teaching quality",
        None,
        "negative",
        [
            "teaching",
            "teacher",
            "professor",
            "lectures",
            "explain",
            "clarity",
        ],
    ),
    (
        "Q03",
        "problems with exams and assessment",
        None,
        "negative",
        [
            "exam",
            "exams",
            "assessment",
            "quiz",
            "quizzes",
            "test",
            "tests",
            "grading",
        ],
    ),
    (
        "Q04",
        "positive feedback about professors",
        "teacher review",
        "positive",
        [
            "professor",
            "teacher",
            "teaching",
            "helpful",
            "feedback",
        ],
    ),
    (
        "Q05",
        "negative course content and learning activities",
        "course review",
        "negative",
        [
            "course",
            "content",
            "learning",
            "activity",
            "activities",
            "material",
        ],
    ),
    (
        "Q06",
        "students struggling with difficult courses",
        "course review",
        "negative",
        [
            "difficulty",
            "difficult",
            "hard",
            "challenging",
            "course",
        ],
    ),
    (
        "Q07",
        "complaints about course materials",
        "course review",
        "negative",
        [
            "materials",
            "material",
            "readings",
            "notes",
            "textbook",
        ],
    ),
    (
        "Q08",
        "positive feedback about helpful teachers",
        "teacher review",
        "positive",
        [
            "teacher",
            "professor",
            "helpful",
            "support",
            "care",
        ],
    ),
    (
        "Q09",
        "negative feedback about university information and services",
        "university review",
        "negative",
        [
            "university",
            "information",
            "services",
            "service",
            "support",
            "administration",
        ],
    ),
    (
        "Q10",
        "positive feedback about university campus and facilities",
        "university review",
        "positive",
        [
            "campus",
            "facilities",
            "facility",
            "building",
            "library",
        ],
    ),
    (
        "Q11",
        "negative feedback about assignments and marking",
        None,
        "negative",
        [
            "assignment",
            "assignments",
            "marking",
            "marked",
            "grading",
            "grade",
        ],
    ),
    (
        "Q12",
        "mixed or neutral feedback about workload",
        "course review",
        "neutral",
        [
            "workload",
            "work",
            "load",
            "course",
        ],
    ),
]


K_VALUES = [1, 3, 5, 10]


# ================================================================
# HELPERS
# ================================================================

def norm_sent(value):
    """
    Normalize sentiment labels.
    """

    if pd.isna(value):
        return "unknown"

    value = str(value).strip().lower()

    if value in {
        "positive",
        "pos",
        "1",
        "2",
    }:
        return "positive"

    if value in {
        "negative",
        "neg",
        "-1",
    }:
        return "negative"

    if value in {
        "neutral",
        "neu",
        "0",
    }:
        return "neutral"

    return value


def tokenize(text):
    """
    Tokenize text for weakly-supervised benchmark relevance.
    """

    return set(
        re.findall(
            r"[a-z]+",
            str(text).lower(),
        )
    )


def dcg(relevances):
    """
    Discounted Cumulative Gain.
    """

    relevances = np.asarray(
        relevances,
        dtype=float,
    )

    if len(relevances) == 0:
        return 0.0

    discounts = np.log2(
        np.arange(
            2,
            len(relevances) + 2,
        )
    )

    return float(
        np.sum(
            (2 ** relevances - 1)
            / discounts
        )
    )


def ndcg(relevances):
    """
    Normalized Discounted Cumulative Gain.
    """

    relevances = np.asarray(
        relevances,
        dtype=float,
    )

    if len(relevances) == 0:
        return 0.0

    ideal = np.sort(
        relevances
    )[::-1]

    ideal_dcg = dcg(ideal)

    if ideal_dcg == 0:
        return 0.0

    return float(
        dcg(relevances)
        / ideal_dcg
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 78)
    print("PHASE 18 — RETRIEVAL EVALUATION")
    print("=" * 78)

    # ------------------------------------------------------------
    # Phase 17 artifacts
    # ------------------------------------------------------------

    metadata_path = (
        PHASE17
        / "review_metadata.csv"
    )

    embeddings_path = (
        PHASE17
        / "review_embeddings.npy"
    )

    if not metadata_path.exists():

        raise FileNotFoundError(
            f"Missing Phase 17 metadata:\n"
            f"{metadata_path}"
        )

    if not embeddings_path.exists():

        raise FileNotFoundError(
            f"Missing Phase 17 embeddings:\n"
            f"{embeddings_path}"
        )

    # ------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------

    meta = pd.read_csv(
        metadata_path
    )

    if (
        "overall_sentiment"
        not in meta.columns
    ):

        raise ValueError(
            "review_metadata.csv must contain "
            "'overall_sentiment'"
        )

    if "domain" not in meta.columns:

        raise ValueError(
            "review_metadata.csv must contain "
            "'domain'"
        )

    text_col = next(
        (
            column
            for column in [
                "text",
                "review_text",
                "text_clean",
            ]
            if column in meta.columns
        ),
        None,
    )

    if text_col is None:

        raise ValueError(
            "No review text column found.\n"
            f"Available columns: {list(meta.columns)}"
        )

    # ------------------------------------------------------------
    # Normalize metadata
    # ------------------------------------------------------------

    meta = meta.copy()

    meta[
        "sentiment_norm"
    ] = meta[
        "overall_sentiment"
    ].map(
        norm_sent
    )

    meta[
        "domain_norm"
    ] = meta[
        "domain"
    ].fillna(
        ""
    ).astype(
        str
    ).str.strip().str.lower()

    meta[
        "tokens"
    ] = meta[
        text_col
    ].fillna(
        ""
    ).map(
        tokenize
    )

    texts = (
        meta[
            text_col
        ]
        .fillna("")
        .astype(str)
        .tolist()
    )

    # ------------------------------------------------------------
    # Load finalized reusable retriever
    # ------------------------------------------------------------

    print(
        "Loading reusable SemanticRetriever..."
    )

    retriever = SemanticRetriever(
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        model_name="all-MiniLM-L6-v2",
    )

    stats = retriever.stats()

    print(
        f"Reviews: "
        f"{stats['reviews_indexed']:,}"
    )

    print(
        f"Embedding dimension: "
        f"{stats['embedding_dimension']}"
    )

    print(
        f"Embedding model: "
        f"{stats['model']}"
    )

    # ------------------------------------------------------------
    # Save benchmark queries
    # ------------------------------------------------------------

    query_df = pd.DataFrame(
        QUERIES,
        columns=[
            "query_id",
            "query",
            "domain_target",
            "sentiment_target",
            "intent_terms",
        ],
    )

    query_df[
        "intent_terms"
    ] = query_df[
        "intent_terms"
    ].map(
        lambda x: ", ".join(x)
    )

    query_df.to_csv(
        OUT
        / "retrieval_queries.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------

    all_rows = []

    metric_rows = []

    query_mrr_rows = []

    # ------------------------------------------------------------
    # Evaluate queries
    # ------------------------------------------------------------

    for (
        qid,
        query,
        domain_target,
        sentiment_target,
        intent_terms,
    ) in QUERIES:

        print()
        print("-" * 78)
        print(
            f"{qid}: {query}"
        )

        # --------------------------------------------------------
        # IMPORTANT
        #
        # SemanticRetriever.search() returns a DataFrame.
        # --------------------------------------------------------

        retrieved = retriever.search(
            query=query,
            top_k=10,
            domain=domain_target,
            sentiment=sentiment_target,
        )

        if not isinstance(
            retrieved,
            pd.DataFrame,
        ):

            raise TypeError(
                "SemanticRetriever.search() "
                "must return a pandas DataFrame."
            )

        # --------------------------------------------------------
        # Validate retriever output
        # --------------------------------------------------------

        required_columns = {
            "row_index",
            "similarity_score",
            "domain",
            "overall_sentiment",
        }

        missing_columns = (
            required_columns
            - set(retrieved.columns)
        )

        if missing_columns:

            raise ValueError(
                "Retriever result is missing "
                f"columns: {sorted(missing_columns)}\n"
                f"Returned columns: "
                f"{list(retrieved.columns)}"
            )

        # --------------------------------------------------------
        # Weakly supervised relevance labels
        #
        # Relevance =
        # domain match
        # +
        # sentiment match
        # +
        # at least one intent term
        # --------------------------------------------------------

        relevant = np.zeros(
            len(meta),
            dtype=np.int8,
        )

        termset = set(
            intent_terms
        )

        for i, row in meta.iterrows():

            domain_ok = (
                domain_target is None
                or row[
                    "domain_norm"
                ]
                == domain_target
            )

            sentiment_ok = (
                sentiment_target is None
                or row[
                    "sentiment_norm"
                ]
                == sentiment_target
            )

            overlap = len(
                termset.intersection(
                    row["tokens"]
                )
            )

            if (
                domain_ok
                and sentiment_ok
                and overlap > 0
            ):

                relevant[i] = 1

        total_relevant = int(
            relevant.sum()
        )

        # --------------------------------------------------------
        # Convert retrieved DataFrame rows into corpus indices
        # --------------------------------------------------------

        retrieved_indices = []

        for _, result in retrieved.iterrows():

            idx = int(
                result[
                    "row_index"
                ]
            )

            if (
                0 <= idx < len(meta)
            ):

                retrieved_indices.append(
                    idx
                )

        ranked_rel = np.array(
            [
                relevant[idx]
                for idx in retrieved_indices
            ],
            dtype=np.int8,
        )

        # --------------------------------------------------------
        # MRR
        #
        # IMPORTANT:
        # MRR is query-level and calculated once.
        # --------------------------------------------------------

        relevant_positions = np.where(
            ranked_rel == 1
        )[0]

        if len(
            relevant_positions
        ):

            reciprocal_rank = (
                1.0
                / (
                    int(
                        relevant_positions[0]
                    )
                    + 1
                )
            )

        else:

            reciprocal_rank = 0.0

        query_mrr_rows.append(
            {
                "query_id": qid,
                "query": query,
                "reciprocal_rank": reciprocal_rank,
            }
        )

        # --------------------------------------------------------
        # Precision / Recall / NDCG
        # --------------------------------------------------------

        for k in K_VALUES:

            top = ranked_rel[
                :k
            ]

            retrieved_relevant = int(
                top.sum()
            )

            precision = (
                retrieved_relevant
                / k
            )

            recall = (
                retrieved_relevant
                / total_relevant
                if total_relevant > 0
                else 0.0
            )

            metric_rows.append(
                {
                    "query_id": qid,
                    "query": query,
                    "k": k,
                    "precision_at_k": precision,
                    "recall_at_k": recall,
                    "ndcg_at_k": ndcg(
                        top
                    ),
                }
            )

        # --------------------------------------------------------
        # Save top 10 retrieval results
        # --------------------------------------------------------

        for rank, (
            _,
            result,
        ) in enumerate(
            retrieved.head(10).iterrows(),
            start=1,
        ):

            idx = int(
                result[
                    "row_index"
                ]
            )

            if (
                0 <= idx < len(meta)
            ):

                is_relevant = int(
                    relevant[idx]
                )

                if (
                    "review_id"
                    in meta.columns
                ):

                    review_id = (
                        meta.iloc[idx][
                            "review_id"
                        ]
                    )

                else:

                    review_id = idx

                text = texts[
                    idx
                ]

            else:

                is_relevant = 0

                review_id = (
                    result.get(
                        "review_id",
                        idx,
                    )
                )

                text = str(
                    result.get(
                        "text",
                        "",
                    )
                )

            all_rows.append(
                {
                    "query_id": qid,
                    "query": query,
                    "rank": rank,
                    "review_id": review_id,
                    "similarity_score": float(
                        result[
                            "similarity_score"
                        ]
                    ),
                    "domain": result[
                        "domain"
                    ],
                    "overall_sentiment": result[
                        "overall_sentiment"
                    ],
                    "relevant": is_relevant,
                    "text": text,
                }
            )

        # --------------------------------------------------------
        # Console information
        # --------------------------------------------------------

        print(
            f"Relevant corpus reviews: "
            f"{total_relevant:,}"
        )

        print(
            f"Retrieved results: "
            f"{len(retrieved)}"
        )

        print(
            f"Relevant hits in top 10: "
            f"{int(ranked_rel[:10].sum())}"
        )

        print(
            f"MRR: "
            f"{reciprocal_rank:.6f}"
        )

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    results = pd.DataFrame(
        all_rows
    )

    metrics = pd.DataFrame(
        metric_rows
    )

    query_mrr = pd.DataFrame(
        query_mrr_rows
    )

    results.to_csv(
        OUT
        / "retrieval_results.csv",
        index=False,
    )

    metrics.to_csv(
        OUT
        / "query_metrics.csv",
        index=False,
    )

    query_mrr.to_csv(
        OUT
        / "query_mrr.csv",
        index=False,
    )

    # ============================================================
    # OVERALL METRICS
    # ============================================================

    overall_mrr = float(
        query_mrr[
            "reciprocal_rank"
        ].mean()
    )

    summary_rows = []

    for k in K_VALUES:

        m = metrics[
            metrics["k"] == k
        ]

        summary_rows.append(
            {
                "k": k,
                "queries": len(m),
                "mean_precision_at_k": float(
                    m[
                        "precision_at_k"
                    ].mean()
                ),
                "mean_recall_at_k": float(
                    m[
                        "recall_at_k"
                    ].mean()
                ),
                "mean_ndcg_at_k": float(
                    m[
                        "ndcg_at_k"
                    ].mean()
                ),
                "mean_reciprocal_rank": overall_mrr,
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUT
        / "overall_retrieval_metrics.csv",
        index=False,
    )

    # ============================================================
    # FAILURE ANALYSIS
    # ============================================================

    q5 = metrics[
        metrics["k"] == 5
    ].copy()

    failures = []

    for _, row in q5.sort_values(
        [
            "precision_at_k",
            "recall_at_k",
        ]
    ).iterrows():

        qid = row[
            "query_id"
        ]

        query_results = results[
            results[
                "query_id"
            ]
            == qid
        ]

        examples = (
            query_results
            .head(3)[
                "text"
            ]
            .astype(str)
            .tolist()
        )

        failures.append(
            {
                "query_id": qid,
                "query": row[
                    "query"
                ],
                "precision_at_5": row[
                    "precision_at_k"
                ],
                "recall_at_5": row[
                    "recall_at_k"
                ],
                "ndcg_at_5": row[
                    "ndcg_at_k"
                ],
                "relevant_hits_in_top10": int(
                    query_results[
                        "relevant"
                    ].sum()
                ),
                "top10_examples": " || ".join(
                    examples
                ),
            }
        )

    failure_df = pd.DataFrame(
        failures
    )

    failure_df.to_csv(
        OUT
        / "failure_analysis.csv",
        index=False,
    )

    # ============================================================
    # K=5 SUMMARY
    # ============================================================

    k5 = summary[
        summary["k"] == 5
    ].iloc[0]

    # ============================================================
    # JSON SUMMARY
    # ============================================================

    payload = {
        "phase": "18",
        "status": "completed",
        "index_reviews": int(
            stats[
                "reviews_indexed"
            ]
        ),
        "embedding_dimension": int(
            stats[
                "embedding_dimension"
            ]
        ),
        "embedding_model": stats[
            "model"
        ],
        "query_count": len(
            QUERIES
        ),
        "retrieval_engine": (
            "src.retrieval.retriever.SemanticRetriever"
        ),
        "evaluation_type": (
            "weakly_supervised_metadata_and_lexical_relevance"
        ),
        "important_caveat": (
            "Metrics are not human-judged relevance scores. "
            "They evaluate a reproducible proxy benchmark "
            "derived from available EduRABSA review metadata."
        ),
        "mrr": overall_mrr,
        "k5": {
            "precision_at_5": float(
                k5[
                    "mean_precision_at_k"
                ]
            ),
            "recall_at_5": float(
                k5[
                    "mean_recall_at_k"
                ]
            ),
            "ndcg_at_5": float(
                k5[
                    "mean_ndcg_at_k"
                ]
            ),
        },
        "outputs": [
            "retrieval_queries.csv",
            "retrieval_results.csv",
            "query_metrics.csv",
            "query_mrr.csv",
            "overall_retrieval_metrics.csv",
            "failure_analysis.csv",
            "phase18_summary.json",
            "PHASE_18_README.md",
        ],
    }

    (
        OUT
        / "phase18_summary.json"
    ).write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ============================================================
    # README
    # ============================================================

    readme = """# Phase 18 — Retrieval Evaluation

Phase 18 evaluates the finalized reusable
`SemanticRetriever` against the same 12-query benchmark
established for Phase 18.

## Retrieval engine

The evaluation uses:

`src/retrieval/retriever.py`

with the Phase 17 semantic index:

- 6,500 reviews
- 384-dimensional embeddings
- all-MiniLM-L6-v2

## Metrics

- Precision@K
- Recall@K
- MRR
- NDCG@K

## Weakly-supervised relevance methodology

A review is considered relevant when:

1. Domain matches the query domain when specified.
2. Sentiment matches the query sentiment.
3. At least one predefined intent term occurs in the review.

These are proxy relevance labels.

They are NOT human-annotated relevance judgments.

## Purpose

Phase 18 is used for:

- retrieval regression testing
- comparing retrieval implementations
- detecting retrieval degradation
- measuring future retrieval improvements

The metrics should not be presented as human relevance scores.

## MRR

MRR is a query-level metric and is calculated once per query.

The overall MRR is the mean reciprocal rank across all 12 queries.

## Pipeline

Phase 17
Semantic Index
    ↓
SemanticRetriever
    ↓
Phase 18 Retrieval Evaluation
    ↓
Phase 19 RAG Pipeline

"""

    (
        OUT
        / "PHASE_18_README.md"
    ).write_text(
        readme,
        encoding="utf-8",
    )

    # ============================================================
    # FINAL OUTPUT
    # ============================================================

    print()
    print("=" * 78)
    print("PHASE 18 COMPLETE")
    print("=" * 78)

    print(
        f"Queries evaluated: "
        f"{len(QUERIES)}"
    )

    print(
        f"Reviews indexed: "
        f"{stats['reviews_indexed']:,}"
    )

    print()
    print(
        "Overall retrieval metrics:"
    )

    display_summary = summary.copy()

    for column in [
        "mean_precision_at_k",
        "mean_recall_at_k",
        "mean_ndcg_at_k",
        "mean_reciprocal_rank",
    ]:

        display_summary[
            column
        ] = display_summary[
            column
        ].round(6)

    print(
        display_summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Overall MRR: "
        f"{overall_mrr:.6f}"
    )

    print()
    print(
        f"Output: {OUT}"
    )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()