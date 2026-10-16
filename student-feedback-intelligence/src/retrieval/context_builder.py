"""
Evidence context builder for the RAG pipeline.

The context produced here is traceable to the original
student feedback records.
"""


def build_evidence_context(
    query,
    results,
):
    """
    Convert retrieval results into a grounded evidence context.

    No new facts are generated here.
    """

    if results is None:
        raise ValueError(
            "Retrieval results cannot be None."
        )

    lines = [
        "RAG EVIDENCE CONTEXT",
        "",
        f"QUESTION: {query}",
        "",
        (
            "Use ONLY the retrieved student feedback "
            "below as evidence."
        ),
        "Do not invent unsupported facts.",
        "",
    ]

    for _, row in results.iterrows():

        lines.extend(
            [
                (
                    f"[Review "
                    f"{row.get('review_id', 'Unknown')}]"
                ),
                (
                    f"Domain: "
                    f"{row.get('domain', 'Unknown')}"
                ),
                (
                    f"Sentiment: "
                    f"{row.get('overall_sentiment', 'Unknown')}"
                ),
                (
                    "Similarity: "
                    f"{float(row.get('similarity_score', 0.0)):.4f}"
                ),
                (
                    "Feedback: "
                    f"{row.get('text', '')}"
                ),
                "",
            ]
        )

    return "\n".join(lines)


def build_evidence_records(
    results,
):
    """
    Convert retrieval results into structured,
    traceable evidence records.
    """

    evidence = []

    if results is None:
        return evidence

    for _, row in results.iterrows():

        evidence.append(
            {
                "review_id": str(
                    row.get(
                        "review_id",
                        "Unknown",
                    )
                ),
                "domain": str(
                    row.get(
                        "domain",
                        "Unknown",
                    )
                ),
                "sentiment": str(
                    row.get(
                        "overall_sentiment",
                        "Unknown",
                    )
                ),
                "similarity_score": round(
                    float(
                        row.get(
                            "similarity_score",
                            0.0,
                        )
                    ),
                    4,
                ),
                "text": str(
                    row.get(
                        "text",
                        "",
                    )
                ),
            }
        )

    return evidence