"""
Deterministic, evidence-grounded RAG pipeline.

Uses the SemanticRetriever to find relevant student reviews
and synthesises a structured summary without any LLM dependency.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

from src.retrieval.retriever import get_shared_retriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PHASE17 = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "edurabsa"
    / "phase17"
)

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def parse_query_intent(query: str) -> Tuple[Optional[str], Optional[int], Optional[str], str]:
    """
    Parse a user query to extract requested sentiment, item count, domain filter,
    and a cleaned search query for embedding retrieval.
    """
    q = query.strip()
    q_lower = q.lower()

    # 1. Count extraction (e.g. 'give me 5 positive feedback', 'top 3 complaints')
    detected_count = None
    count_pattern = (
        r"\b(?:give(?:\s+me)?|show(?:\s+me)?|find(?:\s+me)?|get(?:\s+me)?|list|top|first|fetch|display)?\s*"
        r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\s+"
        r"(?:(?:positive|negative|neutral|relevant|good|bad)\s+)?(?:(?:teacher|course|class|university|exam|student|faculty)\s+)?"
        r"(?:feedback|reviews?|comments?|records?|complaints?|opinions?|examples?|items?)\b"
    )
    m_count = re.search(count_pattern, q_lower)
    if not m_count:
        m_count = re.search(
            r"\b(?:top|first|limit)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
            q_lower,
        )

    if m_count:
        val_str = m_count.group(1)
        val = int(val_str) if val_str.isdigit() else NUMBER_WORDS.get(val_str)
        if val and 1 <= val <= 50:
            detected_count = min(max(val, 1), 20)

    # 2. Sentiment extraction
    detected_sentiment = None
    has_pos_word = bool(
        re.search(
            r"\b(positive|positives|good|great|praise|compliment|compliments|strengths?|favorable|like|liked|love|loved|best)\b",
            q_lower,
        )
    )
    has_neg_word = bool(
        re.search(
            r"\b(negative|negatives|bad|worst|complaint|complaints|criticism|criticisms|issues?|problems?|weaknesses?|concerns?|poor|dislike|disliked|hate|hated)\b",
            q_lower,
        )
    )
    has_neu_word = bool(
        re.search(
            r"\b(neutral|neutrals|average|moderate)\b",
            q_lower,
        )
    )

    # Negation checks (e.g. 'not positive', 'not negative')
    pos_negated = bool(
        re.search(
            r"\b(?:not|no|never|hardly|without)\s+(?:any\s+|very\s+)?(?:positive|good|great|praise|compliments?)\b",
            q_lower,
        )
    )
    neg_negated = bool(
        re.search(
            r"\b(?:not|no|never|hardly|without)\s+(?:any\s+|very\s+)?(?:negative|bad|complaint|complaints|issues?|problems?)\b",
            q_lower,
        )
    )

    is_comparison = bool(
        re.search(
            r"\b(positive\s+or\s+negative|both|compare|comparison|versus|vs\.?)\b",
            q_lower,
        )
    )

    if not is_comparison:
        if (has_pos_word and not pos_negated) and not (has_neg_word and not neg_negated):
            detected_sentiment = "Positive"
        elif (has_neg_word and not neg_negated) or pos_negated:
            detected_sentiment = "Negative"
        elif has_neu_word:
            detected_sentiment = "Neutral"

    # 3. Domain extraction
    detected_domain = None
    if re.search(
        r"\b(teacher\s+reviews?|professor\s+reviews?|faculty\s+reviews?|instructor\s+reviews?)\b",
        q_lower,
    ):
        detected_domain = "teacher review"
    elif re.search(
        r"\b(course\s+reviews?|class\s+reviews?|curriculum\s+reviews?)\b",
        q_lower,
    ):
        detected_domain = "course review"
    elif re.search(
        r"\b(university\s+reviews?|campus\s+reviews?|college\s+reviews?)\b",
        q_lower,
    ):
        detected_domain = "university review"

    # 4. Clean search query for semantic vector lookup
    cleaned = re.sub(
        r"^(?:can\s+you\s+)?(?:please\s+)?(?:give(?:\s+me)?|show(?:\s+me)?|find(?:\s+me)?|get(?:\s+me)?|list|tell\s+me\s+about|display|fetch)\s+",
        "",
        q,
        flags=re.IGNORECASE,
    ).strip()
    search_query = re.sub(
        r"^(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    if not search_query or search_query.lower() in (
        "feedback",
        "reviews",
        "comments",
        "positive feedback",
        "negative feedback",
        "positive reviews",
        "negative reviews",
    ):
        search_query = q

    return detected_sentiment, detected_count, detected_domain, search_query


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
        # Automatically detect intent (sentiment, count, domain) from natural language query
        detected_sentiment, detected_count, detected_domain, search_query = parse_query_intent(query)

        final_sentiment = (
            sentiment
            if (sentiment and sentiment.lower() not in ("all", "none", ""))
            else detected_sentiment
        )
        final_domain = (
            domain
            if (domain and domain.lower() not in ("all", "none", ""))
            else detected_domain
        )
        final_top_k = (
            detected_count
            if (detected_count is not None and (top_k == 8 or top_k is None))
            else (detected_count or top_k or 8)
        )

        results_df = self.retriever.search(
            query=search_query,
            top_k=final_top_k,
            domain=final_domain,
            sentiment=final_sentiment,
        )

        # Enforce strict sentiment filtering post-search as an extra guarantee
        if results_df is not None and not results_df.empty and final_sentiment and final_sentiment.lower() not in ("all", "none", ""):
            results_df = results_df[
                results_df["overall_sentiment"].str.lower() == final_sentiment.strip().lower()
            ].head(final_top_k).reset_index(drop=True)

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

        if total > 0 and len(sentiment_counts) == 1:
            pure_sent = list(sentiment_counts.keys())[0].lower()
            if pure_sent == "positive":
                summary_lines.append("All retrieved reviews express positive feedback.")
            elif pure_sent == "negative":
                summary_lines.append(
                    "All retrieved reviews express negative feedback or concerns."
                )
            elif pure_sent == "neutral":
                summary_lines.append("All retrieved reviews express neutral feedback.")
        elif negative_share >= 0.5:
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
