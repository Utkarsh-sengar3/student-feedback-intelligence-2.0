
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.rag_pipeline import RAGPipeline

PHASE17 = (
    ROOT
    / "data"
    / "processed"
    / "edurabsa"
    / "phase17"
)


def main():

    print("=" * 70)
    print("PHASE 19 — RAG PIPELINE TEST")
    print("=" * 70)

    pipeline = RAGPipeline(
        embeddings_path=(
            PHASE17
            / "review_embeddings.npy"
        ),
        metadata_path=(
            PHASE17
            / "review_metadata.csv"
        ),
        model_name="all-MiniLM-L6-v2",
        top_k=8,
    )

    print("\nRetriever statistics:")
    print(
        pipeline.stats()
    )

    queries = [
        "Why are students unhappy about workload?",
        "What are the major complaints about assessments?",
        "What do students say about teaching quality?",
    ]

    for query in queries:

        print("\n" + "-" * 70)
        print("QUERY:", query)

        response = pipeline.ask(
            query
        )

        print(
            "SUMMARY:",
            response["summary"]
        )

        print(
            "RETRIEVED:",
            response[
                "retrieved_feedback_count"
            ]
        )

        print(
            "SENTIMENT:",
            response[
                "sentiment_distribution"
            ]
        )

        print(
            "EVIDENCE:"
        )

        for evidence in response[
            "supporting_evidence"
        ]:

            print(
                f"  {evidence['review_id']} | "
                f"{evidence['similarity_score']:.4f} | "
                f"{evidence['sentiment']} | "
                f"{evidence['text'][:120]}"
            )

    print("\n" + "=" * 70)
    print("PHASE 19 REUSABLE PIPELINE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()