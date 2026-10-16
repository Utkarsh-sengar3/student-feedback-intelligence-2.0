"""
PHASE 19 — RAG / EVIDENCE-GROUNDED INTELLIGENCE

Uses the existing Phase 17 semantic index.
No model retraining and no external LLM/API required.

Pipeline:

Question
    ↓
Query Embedding
    ↓
Semantic Retrieval
    ↓
Evidence Context
    ↓
Deterministic Grounded Answer
"""

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PHASE17 = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "edurabsa"
    / "phase17"
)

OUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "edurabsa"
    / "phase19"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 8


QUERIES = [
    "What are the major complaints about course assessments?",
    "Why are students unhappy about workload?",
    "What do students say about teaching quality?",
    "What are the positive aspects of professors?",
    "What problems are students reporting about exams?",
    "What should administrators investigate first?",
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clean_text(value):
    """Normalize whitespace."""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_sentiment(value):
    """Normalize sentiment labels."""
    sentiment = str(value).strip().lower()

    mapping = {
        "positive": "Positive",
        "pos": "Positive",

        "negative": "Negative",
        "neg": "Negative",

        "neutral": "Neutral",
        "neu": "Neutral",
    }

    return mapping.get(sentiment, str(value).strip())


# ============================================================
# LOAD PHASE 17 INDEX
# ============================================================

def load_index():

    embeddings_path = PHASE17 / "review_embeddings.npy"
    metadata_path = PHASE17 / "review_metadata.csv"

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Missing Phase 17 embeddings:\n{embeddings_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing Phase 17 metadata:\n{metadata_path}"
        )

    embeddings = np.load(
        embeddings_path
    ).astype("float32")

    metadata = pd.read_csv(
        metadata_path
    )

    if len(embeddings) != len(metadata):

        raise ValueError(
            "Embedding and metadata row counts do not match."
        )

    metadata["text"] = (
        metadata["text"]
        .fillna("")
        .map(clean_text)
    )

    if "domain" not in metadata.columns:

        metadata["domain"] = "Unknown"

    if "overall_sentiment" in metadata.columns:

        metadata["overall_sentiment"] = (
            metadata["overall_sentiment"]
            .map(normalize_sentiment)
        )

    else:

        metadata["overall_sentiment"] = "Unknown"

    return embeddings, metadata


# ============================================================
# SEMANTIC RETRIEVAL
# ============================================================

def retrieve(
    query,
    model,
    embeddings,
    metadata,
    top_k=TOP_K
):

    # Encode query
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    # Cosine similarity because vectors are normalized
    scores = embeddings @ query_embedding

    # Highest similarity first
    indices = np.argsort(-scores)[
        :min(top_k, len(scores))
    ]

    results = (
        metadata
        .iloc[indices]
        .copy()
        .reset_index(drop=True)
    )

    results["similarity_score"] = scores[
        indices
    ]

    return results


# ============================================================
# BUILD EVIDENCE CONTEXT
# ============================================================

def build_context(
    query,
    results
):

    lines = [

        "RAG EVIDENCE CONTEXT",

        f"QUESTION: {query}",

        "",

        "Use ONLY the retrieved feedback below as evidence.",

        "Do not invent unsupported facts.",

        "",
    ]

    for _, row in results.iterrows():

        lines.extend([

            f"[Review {row['review_id']}",

            f"Domain: {row['domain']}",

            f"Sentiment: {row['overall_sentiment']}",

            (
                "Similarity: "
                f"{float(row['similarity_score']):.4f}"
            ),

            f"Feedback: {row['text']}",

            "",
        ])

    return "\n".join(lines)


# ============================================================
# DETERMINISTIC GROUNDED ANSWER
# ============================================================

def make_grounded_answer(
    query,
    results
):

    counts = (
        results["overall_sentiment"]
        .value_counts()
    )

    negative = int(
        counts.get("Negative", 0)
    )

    neutral = int(
        counts.get("Neutral", 0)
    )

    positive = int(
        counts.get("Positive", 0)
    )

    if len(results) > 0:

        negative_share = (
            negative / len(results)
        )

    else:

        negative_share = 0.0


    # --------------------------------------------
    # Summary
    # --------------------------------------------

    if negative_share >= 0.60:

        summary = (
            "Most retrieved evidence is negative."
        )

    elif negative_share >= 0.40:

        summary = (
            "A substantial share of retrieved "
            "evidence is negative."
        )

    elif negative_share < 0.20:

        summary = (
            "Only a small share of retrieved "
            "evidence is negative."
        )

    else:

        summary = (
            "The retrieved evidence contains "
            "mixed student feedback."
        )


    # --------------------------------------------
    # Evidence
    # --------------------------------------------

    evidence = []

    for _, row in results.head(5).iterrows():

        evidence.append({

            "review_id":
                str(row["review_id"]),

            "domain":
                str(row["domain"]),

            "sentiment":
                str(row["overall_sentiment"]),

            "similarity_score":
                round(
                    float(
                        row["similarity_score"]
                    ),
                    4
                ),

            "text":
                clean_text(
                    row["text"]
                ),
        })


    # --------------------------------------------
    # Final structured response
    # --------------------------------------------

    return {

        "query": query,

        "answer_type":
            "deterministic_evidence_grounded",

        "summary":
            summary,

        "retrieved_feedback_count":
            int(len(results)),

        "sentiment_distribution": {

            "negative":
                negative,

            "neutral":
                neutral,

            "positive":
                positive,
        },

        "negative_share":
            round(
                negative_share,
                4
            ),

        "supporting_evidence":
            evidence,

        "limitations": [

            "Based only on retrieved top-K feedback.",

            "Semantic retrieval does not establish causality.",

            (
                "Retrieved feedback is not necessarily "
                "representative of all students."
            ),

            (
                "Negative share describes the retrieved "
                "set, not the entire corpus."
            ),
        ],
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 78)

    print(
        "PHASE 19 — RAG / "
        "EVIDENCE-GROUNDED INTELLIGENCE"
    )

    print("=" * 78)


    # Create output directory
    OUT.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------
    # Load Phase 17
    # --------------------------------------------

    embeddings, metadata = load_index()


    print(
        f"Reviews indexed: "
        f"{len(metadata):,}"
    )

    print(
        f"Embedding dimension: "
        f"{embeddings.shape[1]}"
    )

    print(
        f"Loading query encoder: "
        f"{MODEL_NAME}"
    )


    # --------------------------------------------
    # Load encoder
    # --------------------------------------------

    model = SentenceTransformer(
        MODEL_NAME
    )


    retrieval_rows = []

    contexts = {}

    answers = {}


    # --------------------------------------------
    # Process queries
    # --------------------------------------------

    for query_id, query in enumerate(
        QUERIES,
        1
    ):

        print()

        print(
            f"QUERY {query_id}: "
            f"{query}"
        )


        results = retrieve(
            query,
            model,
            embeddings,
            metadata
        )


        print(
            f"Retrieved: "
            f"{len(results)}"
        )


        # Build evidence context
        contexts[str(query_id)] = (
            build_context(
                query,
                results
            )
        )


        # Generate grounded answer
        answers[str(query_id)] = (
            make_grounded_answer(
                query,
                results
            )
        )


        # Save retrieval records
        for rank, (_, row) in enumerate(
            results.iterrows(),
            1
        ):

            retrieval_rows.append({

                "query_id":
                    query_id,

                "query":
                    query,

                "rank":
                    rank,

                "review_id":
                    row["review_id"],

                "similarity_score":
                    float(
                        row["similarity_score"]
                    ),

                "domain":
                    row["domain"],

                "overall_sentiment":
                    row["overall_sentiment"],

                "text":
                    row["text"],
            })


    # ========================================================
    # SAVE RETRIEVAL RESULTS
    # ========================================================

    retrieval_df = pd.DataFrame(
        retrieval_rows
    )

    retrieval_df.to_csv(
        OUT / "rag_retrieval_results.csv",
        index=False
    )


    # ========================================================
    # SAVE EVIDENCE CONTEXTS
    # ========================================================

    (
        OUT / "evidence_contexts.json"
    ).write_text(

        json.dumps(
            contexts,
            indent=2,
            ensure_ascii=False
        ),

        encoding="utf-8"
    )


    # ========================================================
    # SAVE DEMO ANSWERS
    # ========================================================

    (
        OUT / "rag_demo_outputs.json"
    ).write_text(

        json.dumps(
            answers,
            indent=2,
            ensure_ascii=False
        ),

        encoding="utf-8"
    )


    # ========================================================
    # PHASE SUMMARY
    # ========================================================

    summary = {

        "phase":
            "19",

        "status":
            "completed",

        "architecture":
            (
                "semantic retrieval -> "
                "evidence context -> "
                "grounded answer"
            ),

        "llm_required":
            False,

        "embedding_model":
            MODEL_NAME,

        "reviews_indexed":
            int(len(metadata)),

        "embedding_dimension":
            int(embeddings.shape[1]),

        "queries_processed":
            len(QUERIES),

        "top_k":
            TOP_K,

        "grounding_rules": [

            "Use only retrieved records as evidence.",

            "Preserve review_id for traceability.",

            "Do not make causal claims.",

            (
                "Do not treat retrieved negative share "
                "as corpus prevalence."
            ),
        ],
    }


    (
        OUT / "phase19_summary.json"
    ).write_text(

        json.dumps(
            summary,
            indent=2
        ),

        encoding="utf-8"
    )


    # ========================================================
    # README
    # ========================================================

    readme = """# Phase 19 — RAG / Evidence-Grounded Intelligence

## Objective

Build an evidence-grounded question-answering layer over the Phase 17
semantic retrieval index.

## Pipeline

Question
    ↓
Query Embedding
    ↓
Semantic Retrieval
    ↓
Evidence Context
    ↓
Deterministic Grounded Answer

## Why no external LLM yet?

This phase validates retrieval, traceability, and grounding without API
costs, credentials, or hallucination risk.

A future LLM adapter can consume the same evidence context.

## Required Phase 17 artifacts

data/processed/edurabsa/phase17/review_embeddings.npy

data/processed/edurabsa/phase17/review_metadata.csv

## Outputs

rag_retrieval_results.csv
evidence_contexts.json
rag_demo_outputs.json
phase19_summary.json
PHASE_19_README.md

## Grounding rules

Every evidence item keeps:

- review_id
- domain
- sentiment
- similarity score
- original feedback text

The system does not make causal claims.

Retrieved negative share is NOT interpreted as corpus-wide prevalence.
"""

    (
        OUT / "PHASE_19_README.md"
    ).write_text(
        readme,
        encoding="utf-8"
    )


    # ========================================================
    # COMPLETE
    # ========================================================

    print()

    print("=" * 78)

    print(
        "PHASE 19 COMPLETE"
    )

    print("=" * 78)

    print(
        f"Queries processed: "
        f"{len(QUERIES)}"
    )

    print(
        f"Reviews indexed: "
        f"{len(metadata):,}"
    )

    print(
        f"Output: {OUT}"
    )

    print("Generated:")

    for filename in [

        "rag_retrieval_results.csv",

        "evidence_contexts.json",

        "rag_demo_outputs.json",

        "phase19_summary.json",

        "PHASE_19_README.md",

    ]:

        print(
            f" - {filename}"
        )


if __name__ == "__main__":
    main()