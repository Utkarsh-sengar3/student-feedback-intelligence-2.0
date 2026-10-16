"""
Phase 19 reusable RAG pipeline.

Architecture:

Question
    ↓
Semantic Retrieval
    ↓
Evidence Context
    ↓
Deterministic Grounded Answer

No external LLM is required.
"""

from pathlib import Path

from src.retrieval.retriever import SemanticRetriever, get_shared_retriever
from src.retrieval.context_builder import (
    build_evidence_context,
    build_evidence_records,
)


class RAGPipeline:

    def __init__(
        self,
        embeddings_path=None,
        metadata_path=None,
        model_name="all-MiniLM-L6-v2",
        top_k=8,
        min_similarity=None,
    ):

        self.top_k = top_k
        self.min_similarity = min_similarity

        self.retriever = get_shared_retriever(
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
            model_name=model_name,
        )

    def retrieve(
        self,
        query,
        domain=None,
        sentiment=None,
        top_k=None,
        min_similarity=None,
    ):

        return self.retriever.search(
            query=query,
            top_k=(
                top_k
                if top_k is not None
                else self.top_k
            ),
            domain=domain,
            sentiment=sentiment,
            min_similarity=(
                min_similarity
                if min_similarity is not None
                else self.min_similarity
            ),
        )

    @staticmethod
    def _build_summary(results):

        if results.empty:

            return (
                "No sufficiently relevant "
                "student feedback was retrieved."
            )

        sentiment_series = (
            results["overall_sentiment"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        sentiment_counts = (
            sentiment_series
            .value_counts()
            .to_dict()
        )

        negative = int(
            sentiment_counts.get(
                "negative",
                0,
            )
        )

        total = len(results)

        negative_share = (
            negative / total
            if total
            else 0.0
        )

        if negative_share >= 0.60:

            return (
                "Most of the retrieved evidence "
                "is negative."
            )

        if negative_share >= 0.40:

            return (
                "A substantial share of the "
                "retrieved evidence is negative."
            )

        if negative_share < 0.20:

            return (
                "Only a small share of the "
                "retrieved evidence is negative."
            )

        return (
            "The retrieved evidence contains "
            "mixed student feedback."
        )
    def ask(
        self,
        query,
        domain=None,
        sentiment=None,
        top_k=None,
        min_similarity=None,
    ):
        """
        Execute the complete evidence-grounded pipeline.
        """

        results = self.retrieve(
            query=query,
            domain=domain,
            sentiment=sentiment,
            top_k=top_k,
            min_similarity=min_similarity,
        )

        context = build_evidence_context(
            query,
            results,
        )

        evidence = build_evidence_records(
            results.head(5)
        )

        sentiment_distribution = {}

        if not results.empty:

            normalized_sentiments = (
                results["overall_sentiment"]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            sentiment_distribution = {
                str(key): int(value)
                for key, value
                in normalized_sentiments
                .value_counts()
                .items()
            }

            negative_count = int(
            sentiment_distribution.get(
                "negative",
                0,
            )
        )

        negative_share = (
            negative_count / len(results)
            if len(results) > 0
            else 0.0
        )

        return {
            "query": query,
            "answer_type": (
                "deterministic_evidence_grounded"
            ),
            "summary": self._build_summary(
                results
            ),
            "retrieved_feedback_count": int(
                len(results)
            ),
            "sentiment_distribution": (
                sentiment_distribution
            ),
            "negative_share": round(
                negative_share,
                4,
            ),
            "supporting_evidence": evidence,
            "evidence_context": context,
            "filters": {
                "domain": domain,
                "sentiment": sentiment,
                "top_k": (
                    top_k
                    if top_k is not None
                    else self.top_k
                ),
                "min_similarity": (
                    min_similarity
                    if min_similarity is not None
                    else self.min_similarity
                ),
            },
            "limitations": [
                (
                    "Based only on retrieved "
                    "top-K feedback."
                ),
                (
                    "Semantic retrieval does not "
                    "establish causality."
                ),
                (
                    "Retrieved feedback is not "
                    "necessarily representative "
                    "of all students."
                ),
                (
                    "Negative share describes the "
                    "retrieved set, not the entire "
                    "corpus."
                ),
            ],
        }

    def stats(self):

        return self.retriever.stats()