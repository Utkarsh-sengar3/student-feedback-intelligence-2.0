export interface FeedbackRequest {
  text: string;
}

export interface EvidenceItem {
  review_id: string | number;
  similarity_score?: number;
  domain?: string;
  sentiment?: string;
  text: string;
}

export interface FeedbackResponse {
  text: string;
  sentiment?: string;
  sentiment_confidence?: number;
  category?: string;
  categories: string[];
  aspect?: string;
  aspects: string[];
  severity?: string;
  priority_level?: string;
  priority_score?: number;
  concern_priority_level?: string;
  concern_priority_score?: number;
  recommendation?: string;
  evidence: EvidenceItem[];
}

export interface SearchResult {
  review_id: string | number;
  similarity_score: number;
  domain?: string;
  sentiment?: string;
  text: string;
}

export interface SearchResponse {
  success: boolean;
  query: string;
  count: number;
  filters?: { domain?: string; sentiment?: string };
  results: SearchResult[];
}

export interface InsightRecord {
  concept?: string;
  priority?: string;
  priority_score?: number;
  severity_level?: string;
  recommended_action?: string;
  recommended_owner?: string;
  [key: string]: unknown;
}

export interface InsightSummary {
  total_insights: number;
  priority_distribution?: Record<string, number>;
  severity_distribution?: Record<string, number>;
  average_priority_score?: number;
}

export interface ReviewRecord {
  row_index: number;
  review_id: string | number;
  original_id?: string | number;
  text: string;
  domain: string;
  overall_sentiment: string;
}

export interface ReviewsListResponse {
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  domain_counts: Record<string, number>;
  sentiment_counts: Record<string, number>;
  reviews: ReviewRecord[];
}

export interface ModelMetrics {
  sentiment_model: {
    name: string;
    base_architecture: string;
    validation_macro_f1: number;
    test_accuracy: number;
    test_macro_f1: number;
    test_macro_precision: number;
    test_macro_recall: number;
    [key: string]: unknown;
  };
  category_model: {
    name: string;
    task: string;
    category_count: number;
    categories: string[];
    micro_f1: number;
    macro_f1: number;
    subset_accuracy: number;
    hamming_loss: number;
    [key: string]: unknown;
  };
  retrieval_model: {
    name: string;
    indexed_reviews: number;
    mrr: number;
    ndcg_at_5: number;
    precision_at_5: number;
    [key: string]: unknown;
  };
  insights_summary: {
    concepts_with_insights: number;
    critical_concerns: number;
    high_concerns: number;
    medium_concerns: number;
    low_or_monitor_concerns?: number;
    [key: string]: unknown;
  };
}

export interface AssistantRequest {
  question: string;
  top_k?: number;
  domain?: string;
  sentiment?: string;
}

export interface AssistantResponse {
  question: string;
  answer_type: string;
  summary: string;
  retrieved_count: number;
  sentiment_distribution: Record<string, number>;
  negative_share: number;
  evidence: EvidenceItem[];
  evidence_context?: string;
  limitations: string[];
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
}

export interface StatusResponse {
  status: string;
  service: string;
  models: {
    sentiment_model_loaded: boolean;
    category_model_loaded: boolean;
    category_count: number;
    phase16_insights_loaded: boolean;
    semantic_retriever_loaded: boolean;
  };
}
