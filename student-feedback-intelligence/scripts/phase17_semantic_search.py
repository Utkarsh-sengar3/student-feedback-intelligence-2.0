"""
PHASE 17 — SEMANTIC SEARCH / EVIDENCE RETRIEVAL

Builds a local semantic retrieval index over EduRABSA student feedback.

Design:
- Uses sentence-transformers when available.
- Falls back to TF-IDF cosine retrieval if sentence-transformers is unavailable.
- Indexes canonical reviews, not annotation rows, to avoid duplicate evidence.
- Keeps review_id, domain, sentiment and text metadata.
- Supports free-text semantic queries.
- Optional filters: domain, sentiment, concept/category keyword.
- Saves a reusable local index under data/processed/edurabsa/phase17/.

No LLM/RAG is used in this phase.
"""

from pathlib import Path
import json
import re
import sys
import argparse
import pickle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "edurabsa"
OUT = DATA / "phase17"
OUT.mkdir(parents=True, exist_ok=True)

REVIEWS_PATH = DATA / "reviews_canonical.csv"
EVIDENCE_PATH = DATA / "phase16" / "dashboard_evidence_package.csv"

MODEL_NAME = "all-MiniLM-L6-v2"


def normalize_text(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def normalize_sentiment_label(x):
    """Normalize common EduRABSA sentiment spellings for filtering."""
    x = normalize_text(x).lower()
    mapping = {
        "positive": "positive",
        "pos": "positive",
        "negative": "negative",
        "neg": "negative",
        "neutral": "neutral",
        "neu": "neutral",
    }
    return mapping.get(x, x)


def load_reviews():
    if not REVIEWS_PATH.exists():
        raise FileNotFoundError(f"Missing canonical reviews: {REVIEWS_PATH}")

    df = pd.read_csv(REVIEWS_PATH)

    required = {"review_id", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"reviews_canonical.csv missing columns: {sorted(missing)}")

    df["review_id"] = df["review_id"].astype(str)
    df["text"] = df["text"].apply(normalize_text)

    if "domain" not in df.columns:
        df["domain"] = "unknown"
    if "overall_sentiment" not in df.columns:
        df["overall_sentiment"] = "unknown"

    df["overall_sentiment"] = df["overall_sentiment"].apply(
        normalize_sentiment_label
    )

    df = df[df["text"].str.len() > 0].copy()
    df = df.drop_duplicates("review_id").reset_index(drop=True)
    return df


def try_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except Exception:
        return None


def build_embedding_index(df):
    SentenceTransformer = try_sentence_transformers()

    if SentenceTransformer is None:
        return None, "tfidf"

    print(f"\nLoading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Encoding {len(df):,} reviews...")
    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    np.save(OUT / "review_embeddings.npy", embeddings)

    with open(OUT / "embedding_model.txt", "w", encoding="utf-8") as f:
        f.write(MODEL_NAME)

    return model, "sentence-transformers"


def build_tfidf_index(df):
    from sklearn.feature_extraction.text import TfidfVectorizer

    print("\nSentence-transformers unavailable.")
    print("Building TF-IDF semantic-search fallback...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=50000,
    )

    matrix = vectorizer.fit_transform(df["text"])

    with open(OUT / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open(OUT / "tfidf_matrix.pkl", "wb") as f:
        pickle.dump(matrix, f)

    return vectorizer, matrix


def save_metadata(df):
    cols = [
        c for c in [
            "review_id",
            "original_id",
            "text",
            "domain",
            "overall_sentiment",
        ] if c in df.columns
    ]

    metadata = df[cols].copy()
    metadata.insert(0, "row_index", np.arange(len(metadata)))

    metadata.to_csv(
        OUT / "review_metadata.csv",
        index=False,
    )

    return metadata


def cosine_top_k(query_embedding, embeddings, k=10):
    scores = embeddings @ query_embedding
    k = min(k, len(scores))

    if k == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]


def search_sentence_transformers(
    query,
    model,
    metadata,
    embeddings,
    k=10,
    domain=None,
    sentiment=None,
):
    q = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype("float32")

    scores = embeddings @ q

    mask = np.ones(len(metadata), dtype=bool)

    if domain:
        mask &= (
            metadata["domain"]
            .astype(str)
            .str.lower()
            .eq(domain.lower())
            .to_numpy()
        )

    if sentiment:
        requested_sentiment = normalize_sentiment_label(sentiment)
        mask &= (
            metadata["overall_sentiment"]
            .apply(normalize_sentiment_label)
            .eq(requested_sentiment)
            .to_numpy()
        )

    scores_filtered = np.where(mask, scores, -np.inf)

    k = min(k, int(mask.sum()))
    if k <= 0:
        return pd.DataFrame()

    idx = np.argpartition(-scores_filtered, k - 1)[:k]
    idx = idx[np.argsort(-scores_filtered[idx])]

    result = metadata.iloc[idx].copy()
    result.insert(1, "similarity_score", scores_filtered[idx])
    return result


def search_tfidf(query, vectorizer, matrix, metadata, k=10,
                  domain=None, sentiment=None):
    from sklearn.metrics.pairwise import cosine_similarity

    q = vectorizer.transform([query])
    scores = cosine_similarity(q, matrix).ravel()

    mask = np.ones(len(metadata), dtype=bool)

    if domain:
        mask &= (
            metadata["domain"]
            .astype(str)
            .str.lower()
            .eq(domain.lower())
            .to_numpy()
        )

    if sentiment:
        requested_sentiment = normalize_sentiment_label(sentiment)
        mask &= (
            metadata["overall_sentiment"]
            .apply(normalize_sentiment_label)
            .eq(requested_sentiment)
            .to_numpy()
        )

    scores_filtered = np.where(mask, scores, -np.inf)

    k = min(k, int(mask.sum()))
    if k <= 0:
        return pd.DataFrame()

    idx = np.argpartition(-scores_filtered, k - 1)[:k]
    idx = idx[np.argsort(-scores_filtered[idx])]

    result = metadata.iloc[idx].copy()
    result.insert(1, "similarity_score", scores_filtered[idx])
    return result


def run_demo_searches(engine, model_or_vectorizer, embeddings_or_matrix,
                      metadata):
    queries = [
        ("negative feedback about course workload", None, "Negative"),
        ("complaints about teaching quality", None, "Negative"),
        ("problems with exams and assessment", None, "Negative"),
        ("positive feedback about professors", None, "Positive"),
    ]

    demo_dir = OUT / "demo_searches"
    demo_dir.mkdir(exist_ok=True)

    for i, (query, domain, sentiment) in enumerate(queries, 1):
        if engine == "sentence-transformers":
            result = search_sentence_transformers(
                query, model_or_vectorizer, metadata,
                embeddings_or_matrix, 10, domain, sentiment
            )
        else:
            result = search_tfidf(
                query, model_or_vectorizer, embeddings_or_matrix,
                metadata, 10, domain, sentiment
            )

        result.to_csv(
            demo_dir / f"search_{i}.csv",
            index=False
        )

        print(f"\nDEMO QUERY {i}: {query}")
        print(f"Results: {len(result)}")
        if result.empty:
            print("No results returned.")
        else:
            print(result[
                [c for c in ["review_id", "similarity_score", "domain",
                             "overall_sentiment", "text"] if c in result.columns]
            ].head(5).to_string(index=False))


def interactive_search(engine, model_or_vectorizer, embeddings_or_matrix,
                       metadata):
    print("\nInteractive semantic search")
    print("Type 'exit' to stop.")
    print("Examples:")
    print("  workload complaints")
    print("  problems with assessment")
    print("  positive professor feedback")

    while True:
        try:
            query = input("\nSearch> ").strip()
        except EOFError:
            break

        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        if engine == "sentence-transformers":
            result = search_sentence_transformers(
                query, model_or_vectorizer, metadata,
                embeddings_or_matrix, 10
            )
        else:
            result = search_tfidf(
                query, model_or_vectorizer, embeddings_or_matrix,
                metadata, 10
            )

        if result.empty:
            print("No results.")
        else:
            for _, row in result.iterrows():
                print(
                    f"\n[{row['similarity_score']:.4f}] "
                    f"{row['review_id']} | "
                    f"{row.get('domain', 'unknown')} | "
                    f"{row.get('overall_sentiment', 'unknown')}"
                )
                print(row["text"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open an interactive search prompt after indexing."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run one search query after indexing."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of results."
    )
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--sentiment", type=str, default=None)

    args = parser.parse_args()

    print("=" * 78)
    print("PHASE 17 — SEMANTIC SEARCH / EVIDENCE RETRIEVAL")
    print("=" * 78)

    df = load_reviews()

    print(f"\nCanonical reviews indexed: {len(df):,}")

    metadata = save_metadata(df)

    SentenceTransformer = try_sentence_transformers()

    if SentenceTransformer is not None:
        model, engine = build_embedding_index(df)
        embeddings = np.load(OUT / "review_embeddings.npy")
        vectorizer_or_model = model
        matrix_or_embeddings = embeddings
    else:
        vectorizer, matrix = build_tfidf_index(df)
        engine = "tfidf"
        vectorizer_or_model = vectorizer
        matrix_or_embeddings = matrix

    # Save evidence linkage where available.
    if EVIDENCE_PATH.exists():
        evidence = pd.read_csv(EVIDENCE_PATH)
        evidence_cols = [
            c for c in [
                "concept", "phase15_rank", "priority_score",
                "severity_level", "priority", "review_id",
                "text", "opinion", "category", "domain"
            ] if c in evidence.columns
        ]
        evidence[evidence_cols].to_csv(
            OUT / "retrieval_evidence_metadata.csv",
            index=False
        )

    summary = {
        "phase": "17",
        "status": "completed",
        "engine": engine,
        "embedding_model": MODEL_NAME if engine == "sentence-transformers" else None,
        "reviews_indexed": int(len(df)),
        "unique_review_ids": int(df["review_id"].nunique()),
        "output_directory": str(OUT),
        "design": [
            "One indexed record per canonical review",
            "Metadata retained for domain and sentiment filtering",
            "No LLM/RAG in this phase",
            "Retrieval evaluated independently before RAG",
        ],
    }

    (OUT / "phase17_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (OUT / "PHASE_17_README.md").write_text(
        """# Phase 17 — Semantic Search / Evidence Retrieval

## Objective

Create a reusable local retrieval layer over canonical EduRABSA reviews.

## Preferred engine

`sentence-transformers` with `all-MiniLM-L6-v2`.

If the package/model is unavailable, the script falls back to TF-IDF cosine
retrieval so the phase can still be completed and tested.

## Important methodology

The index uses canonical reviews rather than annotation rows. This prevents
the same review from appearing multiple times because it has multiple aspect
annotations.

## Usage

Build/index:

```powershell
..\\.venv\\Scripts\\python.exe -u scripts/phase17_semantic_search.py
```

Interactive search:

```powershell
..\\.venv\\Scripts\\python.exe -u scripts/phase17_semantic_search.py --interactive
```

One query:

```powershell
..\\.venv\\Scripts\\python.exe -u scripts/phase17_semantic_search.py --query "negative feedback about course workload"
```

## Next

Phase 18 should evaluate retrieval quality using a manually defined query set
and relevance judgments before adding RAG.
""",
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("PHASE 17 COMPLETE")
    print("=" * 78)
    print(f"Reviews indexed: {len(df):,}")
    print(f"Retrieval engine: {engine}")
    print(f"Output: {OUT}")

    # Unfiltered sanity check: verifies the embedding index itself returns
    # results before any metadata filter is applied.
    sanity_query = "course workload"
    if engine == "sentence-transformers":
        sanity = search_sentence_transformers(
            sanity_query, vectorizer_or_model, metadata,
            matrix_or_embeddings, 5
        )
    else:
        sanity = search_tfidf(
            sanity_query, vectorizer_or_model, matrix_or_embeddings,
            metadata, 5
        )

    print(f"\nSANITY QUERY: {sanity_query}")
    print(f"Results: {len(sanity)}")
    if not sanity.empty:
        print(sanity[
            [c for c in ["review_id", "similarity_score", "domain",
                         "overall_sentiment", "text"] if c in sanity.columns]
        ].to_string(index=False))

    run_demo_searches(
        engine,
        vectorizer_or_model,
        matrix_or_embeddings,
        metadata,
    )

    if args.query:
        if engine == "sentence-transformers":
            result = search_sentence_transformers(
                args.query, vectorizer_or_model, metadata,
                matrix_or_embeddings, args.k,
                args.domain, args.sentiment
            )
        else:
            result = search_tfidf(
                args.query, vectorizer_or_model, matrix_or_embeddings,
                metadata, args.k, args.domain, args.sentiment
            )

        print("\nQUERY RESULTS")
        print(result.to_string(index=False))

    if args.interactive:
        interactive_search(
            engine,
            vectorizer_or_model,
            matrix_or_embeddings,
            metadata,
        )

    print("\nGenerated:")
    for p in sorted(OUT.iterdir()):
        print(f" - {p.name}")


if __name__ == "__main__":
    main()
