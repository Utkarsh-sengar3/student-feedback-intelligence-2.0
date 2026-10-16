from fastapi import APIRouter, HTTPException
from api.schemas.feedback import AssistantRequest, AssistantResponse, EvidenceItem
from src.retrieval.rag_pipeline import RAGPipeline

router = APIRouter(
    prefix="/api/v1/assistant",
    tags=["AI Assistant"],
)

_rag_pipeline = None


def get_rag() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


@router.post("/", response_model=AssistantResponse)
def ask_assistant(payload: AssistantRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        rag = get_rag()
        result = rag.ask(
            query=question,
            domain=payload.domain,
            sentiment=payload.sentiment,
            top_k=payload.top_k,
        )

        evidence_items = [
            EvidenceItem(
                review_id=item.get("review_id"),
                similarity_score=item.get("similarity_score"),
                domain=item.get("domain"),
                sentiment=item.get("sentiment"),
                text=item.get("text", ""),
            )
            for item in result.get("supporting_evidence", [])
        ]

        return AssistantResponse(
            question=question,
            answer_type=result.get(
                "answer_type", "deterministic_evidence_grounded"
            ),
            summary=result.get("summary", ""),
            retrieved_count=result.get("retrieved_feedback_count", 0),
            sentiment_distribution=result.get("sentiment_distribution", {}),
            negative_share=float(result.get("negative_share", 0.0)),
            evidence=evidence_items,
            evidence_context=result.get("evidence_context"),
            limitations=result.get("limitations", []),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Assistant retrieval failure: {str(exc)}",
        )
