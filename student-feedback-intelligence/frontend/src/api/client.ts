import type {
  FeedbackRequest,
  FeedbackResponse,
  SearchResponse,
  InsightRecord,
  InsightSummary,
  ReviewsListResponse,
  ModelMetrics,
  AssistantRequest,
  AssistantResponse,
  HealthResponse,
  StatusResponse,
} from './types';

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string) || '';

async function request<T>(
  path: string,
  options?: RequestInit,
  timeoutMs = 90000,
  retries = 1
): Promise<T> {
  const url = `${API_BASE}${path}`;

  for (let attempt = 0; attempt <= retries; attempt++) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
      });
      clearTimeout(timeoutId);

      // On Render free tier cold-start, gateways may briefly return 502/503/504 while container boots
      if (!response.ok) {
        if ([502, 503, 504].includes(response.status) && attempt < retries) {
          await new Promise((res) => setTimeout(res, 2000));
          continue;
        }
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error((error as { detail?: string }).detail || `HTTP ${response.status}`);
      }

      return (await response.json()) as T;
    } catch (err) {
      clearTimeout(timeoutId);
      const isAbort = err instanceof Error && err.name === 'AbortError';
      const isNetworkError = err instanceof TypeError;

      if ((isAbort || isNetworkError) && attempt < retries) {
        await new Promise((res) => setTimeout(res, 2000));
        continue;
      }

      if (isAbort) {
        throw new Error('Request timed out. The server may be starting up — please try again in a moment.');
      }
      throw err;
    }
  }

  throw new Error('Request failed after retry.');
}

export const api = {
  health: () => request<HealthResponse>('/health'),

  status: () => request<StatusResponse>('/api/v1/status'),

  analyze: (payload: FeedbackRequest) =>
    request<FeedbackResponse>('/api/v1/analyze', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  search: (query: string, top_k = 10, domain?: string, sentiment?: string) => {
    const params = new URLSearchParams({ query, top_k: String(top_k) });
    if (domain) params.set('domain', domain);
    if (sentiment) params.set('sentiment', sentiment);
    return request<SearchResponse>(`/api/v1/search/?${params}`);
  },

  insights: {
    all: () =>
      request<{ success: boolean; count: number; data: InsightRecord[] }>(
        '/api/v1/insights/'
      ),
    top: (limit = 10) =>
      request<{ success: boolean; data: InsightRecord[] }>(
        `/api/v1/insights/top?limit=${limit}`
      ),
    summary: () =>
      request<{ success: boolean; data: InsightSummary }>(
        '/api/v1/insights/summary'
      ),
  },

  reviews: (
    page = 1,
    page_size = 20,
    domain?: string,
    sentiment?: string,
    search?: string
  ) => {
    const params = new URLSearchParams({
      page: String(page),
      page_size: String(page_size),
    });
    if (domain) params.set('domain', domain);
    if (sentiment) params.set('sentiment', sentiment);
    if (search) params.set('search', search);
    return request<ReviewsListResponse>(`/api/v1/reviews/?${params}`);
  },

  metrics: () => request<ModelMetrics>('/api/v1/model-metrics/'),

  assistant: (payload: AssistantRequest) =>
    request<AssistantResponse>('/api/v1/assistant/', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
};
