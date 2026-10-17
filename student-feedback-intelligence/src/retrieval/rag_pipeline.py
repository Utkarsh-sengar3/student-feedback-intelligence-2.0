"""
Deterministic, evidence-grounded RAG pipeline.

Uses the SemanticRetriever to find relevant student reviews
and synthesises a structured summary without any LLM dependency.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

from src.retrieval.retriever import get_shared_retriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PHASE17 = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "edurabsa"
    / "phase17"
)


class RAGPipeline:
    """
    Evidence-grounded retrieval pipeline for student feedback questions.
    """

    def __init__(self):
        embeddings_path = PHASE17 / "review_embeddings.npy"
        metadata_path = PHASE17 / "review_metadata.csv"

        self.retriever = get_shared_retriever(
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
            model_name="all-MiniLM-L6-v2",
        )

    def ask(
        self,
        query: str,
        domain: Optional[str] = None,
        sentiment: Optional[str] = None,
        top_k: int = 8,
    ) -> dict:
        """
        Answer a question about student feedback using semantic retrieval.

        Returns a dict with:
        - summary: evidence-grounded text summary
        - retrieved_feedback_count: number of reviews retrieved
        - sentiment_distribution: {label: count}
        - negative_share: fraction of retrieved reviews that are negative
        - supporting_evidence: list of evidence dicts
        - evidence_context: concatenated evidence text snippet
        - limitations: list of limitation strings
        - answer_type: 'deterministic_evidence_grounded'
        """
        results_df = self.retriever.search(
            query=query,
            top_k=top_k,
            domain=domain,
            sentiment=sentiment,
        )

        limitations = [
            "This system uses semantic retrieval without generative AI. "
            "The summary is derived from retrieved review patterns, not expert analysis.",
            "Results depend on the quality and coverage of the indexed review dataset.",
        ]

        if results_df is None or results_df.empty:
            return {
                "answer_type": "deterministic_evidence_grounded",
                "summary": (
                    f"No reviews were found matching your query: '{query}'. "
                    "Try different search terms or remove filters."
                ),
                "retrieved_feedback_count": 0,
                "sentiment_distribution": {},
                "negative_share": 0.0,
                "supporting_evidence": [],
                "evidence_context": None,
                "limitations": limitations,
            }

        # Collect evidence
        evidence = []
        metadata = self.retriever.metadata

        for _, row in results_df.iterrows():
            row_idx = int(row.get("row_index", -1))
            similarity = float(row.get("similarity_score", 0.0))

            if 0 <= row_idx < len(metadata):
                orig = metadata.iloc[row_idx]
                text = str(orig.get("text", "") or "")
                domain_val = str(orig.get("domain", "unknown") or "unknown")
                sentiment_val = str(orig.get("overall_sentiment", "unknown") or "unknown")
                review_id = orig.get("review_id", row_idx)
            else:
                text = str(row.get("text", "") or "")
                domain_val = str(row.get("domain", "unknown") or "unknown")
                sentiment_val = str(row.get("overall_sentiment", "unknown") or "unknown")
                review_id = row_idx

            evidence.append(
                {
                    "review_id": review_id,
                    "similarity_score": similarity,
                    "domain": domain_val,
                    "sentiment": sentiment_val,
                    "text": text,
                }
            )

        # Sentiment distribution
        sentiment_counts: Counter = Counter(
            item["sentiment"] for item in evidence
        )
        total = len(evidence)
        negative_count = sum(
            v for k, v in sentiment_counts.items()
            if "negative" in k.lower()
        )
        negative_share = negative_count / total if total > 0 else 0.0

        # Dominant domain
        domain_counts: Counter = Counter(item["domain"] for item in evidence)
        top_domain = domain_counts.most_common(1)[0][0] if domain_counts else "unknown"

        # Evidence context (first 3 snippets)
        evidence_context = " | ".join(
            item["text"][:150]
            for item in evidence[:3]
            if item["text"]
        )

        # Build summary
        sentiment_str = ", ".join(
            f"{count} {label.lower()}"
            for label, count in sorted(sentiment_counts.items(), key=lambda x: -x[1])
        )

        dominant_sentiment = (
            sentiment_counts.most_common(1)[0][0]
            if sentiment_counts
            else "mixed"
        )

        summary_lines = [
            f"Found {total} relevant student reviews for your query.",
            f"Sentiment breakdown: {sentiment_str}.",
        ]

        if negative_share >= 0.5:
            summary_lines.append(
                f"A majority ({negative_count} of {total}) of retrieved reviews express negative sentiment, "
                "indicating a recurring concern worth addressing."
            )
        elif negative_share > 0.3:
            summary_lines.append(
                f"A notable portion ({negative_count} of {total}) of retrieved reviews are negative."
            )
        else:
            summary_lines.append(
                "The retrieved reviews are predominantly positive or neutral."
            )

        if top_domain and top_domain != "unknown":
            summary_lines.append(
                f"Most retrieved feedback comes from the '{top_domain}' domain."
            )

        summary = " ".join(summary_lines)

        return {
            "answer_type": "deterministic_evidence_grounded",
            "summary": summary,
            "retrieved_feedback_count": total,
            "sentiment_distribution": dict(sentiment_counts),
            "negative_share": negative_share,
            "supporting_evidence": evidence,
            "evidence_context": evidence_context if evidence_context else None,
            "limitations": limitations,
        }
