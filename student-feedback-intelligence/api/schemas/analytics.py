from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AnalyticsKPIs(BaseModel):
    total_records: int = Field(..., description="Total feedback records matching active filters")
    positive_count: int = Field(..., description="Count of positive feedback records")
    positive_pct: float = Field(..., description="Percentage of positive feedback")
    negative_count: int = Field(..., description="Count of negative feedback records")
    negative_pct: float = Field(..., description="Percentage of negative feedback")
    neutral_count: int = Field(..., description="Count of neutral feedback records")
    neutral_pct: float = Field(..., description="Percentage of neutral feedback")
    distinct_domains: int = Field(..., description="Count of distinct domains represented")
    distinct_categories: int = Field(..., description="Count of distinct categories represented")
    avg_word_count: float = Field(..., description="Average word count per review")
    avg_char_length: float = Field(..., description="Average character length per review")


class SentimentSlice(BaseModel):
    name: str
    count: int
    percentage: float
    color: str


class DomainSlice(BaseModel):
    domain: str
    total: int
    positive: int
    negative: int
    neutral: int
    positive_pct: float
    negative_pct: float


class CategoryVolumeSlice(BaseModel):
    category: str
    total: int
    positive: int
    negative: int
    neutral: int
    positive_pct: float
    negative_pct: float


class TopCategoryMetric(BaseModel):
    category: str
    count: int
    total_in_category: int
    percentage: float


class HistogramBin(BaseModel):
    bin: str
    count: int


class TermFrequency(BaseModel):
    term: str
    count: int


class TextEDA(BaseModel):
    word_count_distribution: List[HistogramBin]
    char_length_distribution: List[HistogramBin]
    top_positive_terms: List[TermFrequency]
    top_negative_terms: List[TermFrequency]
    normalization_notes: str


class TemporalNotice(BaseModel):
    has_temporal_data: bool
    reason: str


class FilterOptions(BaseModel):
    domains: List[str]
    categories: List[str]
    sentiments: List[str]


class AnalyticsOverviewResponse(BaseModel):
    kpis: AnalyticsKPIs
    sentiment_distribution: List[SentimentSlice]
    domain_distribution: List[DomainSlice]
    category_volume: List[CategoryVolumeSlice]
    top_negative_categories: List[TopCategoryMetric]
    top_positive_categories: List[TopCategoryMetric]
    text_eda: TextEDA
    temporal_notice: TemporalNotice
    filter_options: FilterOptions


class AnalyticsRecord(BaseModel):
    review_id: int
    original_id: Optional[str] = None
    domain: str
    overall_sentiment: str
    text: str
    categories: List[str]
    word_count: int
    char_count: int


class AnalyticsRecordsResponse(BaseModel):
    total: int
    page: int
    page_size: int
    total_pages: int
    records: List[AnalyticsRecord]
