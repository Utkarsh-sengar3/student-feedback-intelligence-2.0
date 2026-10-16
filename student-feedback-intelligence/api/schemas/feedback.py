from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        max_length=5000,
        description="Student feedback text",
    )


class EvidenceItem(BaseModel):
    review_id: Any
    similarity_score: Optional[float] = None
    domain: Optional[str] = None
    sentiment: Optional[str] = None
    text: str


class FeedbackResponse(BaseModel):
    text: str

    sentiment: Optional[str] = None
    sentiment_confidence: Optional[float] = None

    # Backward-compatible primary category
    category: Optional[str] = None

    # Multilabel category predictions
    categories: List[str] = Field(default_factory=list)

    # Primary aspect for backward compatibility
    aspect: Optional[str] = None

    # Multiple detected aspects for compound issues
    aspects: List[str] = Field(default_factory=list)

    severity: Optional[str] = None

    # Individual action priority
    priority_level: Optional[str] = None
    priority_score: Optional[float] = None

    # Aggregate dataset concern priority
    concern_priority_level: Optional[str] = None
    concern_priority_score: Optional[float] = None

    recommendation: Optional[str] = None

    evidence: List[EvidenceItem] = Field(default_factory=list)


class SearchResult(BaseModel):
    review_id: Any
    similarity_score: float
    domain: Optional[str] = None
    sentiment: Optional[str] = None
    text: str


class SearchResponse(BaseModel):
    query: str
    count: int
    filters: Optional[Dict[str, Optional[str]]] = None
    results: List[SearchResult]


class AssistantRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Question about student feedback",
    )
    top_k: int = Field(
        default=8,
        ge=1,
        le=20,
        description="Number of evidence reviews to retrieve",
    )
    domain: Optional[str] = Field(
        default=None,
        description="Optional domain filter ('course review', 'teacher review', 'university review')",
    )
    sentiment: Optional[str] = Field(
        default=None,
        description="Optional sentiment filter ('Positive', 'Negative', 'Neutral')",
    )


class AssistantResponse(BaseModel):
    question: str
    answer_type: str = "deterministic_evidence_grounded"
    summary: str
    retrieved_count: int
    sentiment_distribution: Dict[str, int]
    negative_share: float
    evidence: List[EvidenceItem]
    evidence_context: Optional[str] = None
    limitations: List[str] = Field(default_factory=list)


class ReviewRecord(BaseModel):
    row_index: int
    review_id: Any
    original_id: Optional[Any] = None
    text: str
    domain: str
    overall_sentiment: str


class ReviewsListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    total_pages: int
    domain_counts: Dict[str, int]
    sentiment_counts: Dict[str, int]
    reviews: List[ReviewRecord]


class ModelMetricsResponse(BaseModel):
    sentiment_model: Dict[str, Any]
    category_model: Dict[str, Any]
    retrieval_model: Dict[str, Any]
    insights_summary: Dict[str, Any]