import { useEffect, useState, useCallback } from 'react';
import { BookOpen, AlertTriangle, ChevronLeft, ChevronRight } from 'lucide-react';
import { api } from '../api/client';
import type { ReviewsListResponse } from '../api/types';
import { SentimentBadge, Badge } from '../components/ui/Badge';
import { Spinner } from '../components/ui/Spinner';

export default function Reviews() {
  const [data, setData] = useState<ReviewsListResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [domain, setDomain] = useState('');
  const [sentiment, setSentiment] = useState('');
  const [search, setSearch] = useState('');
  const [searchInput, setSearchInput] = useState('');

  const loadReviews = useCallback(() => {
    setLoading(true);
    setError(null);
    api.reviews(page, 20, domain || undefined, sentiment || undefined, search || undefined)
      .then(setData)
      .catch(err => setError(err instanceof Error ? err.message : 'Failed to load reviews.'))
      .finally(() => setLoading(false));
  }, [page, domain, sentiment, search]);

  useEffect(() => { loadReviews(); }, [loadReviews]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setSearch(searchInput);
    setPage(1);
  };

  const handleFilterChange = () => { setPage(1); };

  const domains = data ? Object.keys(data.domain_counts) : [];
  const sentiments = data ? Object.keys(data.sentiment_counts) : [];

  return (
    <div className="page">
      <div className="container">
        <div className="page-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', marginBottom: 'var(--space-2)' }}>
            <BookOpen size={28} style={{ color: 'var(--color-accent)' }} />
            <h1 className="page-title" style={{ margin: 0 }}>Reviews Explorer</h1>
          </div>
          <p className="page-description">
            Browse and filter the 6,500+ indexed student reviews from the EduRABSA dataset.
          </p>
        </div>

        {/* Filters */}
        <div className="card" style={{ marginBottom: 'var(--space-4)' }}>
          <div className="card-body">
            <form onSubmit={handleSearch} style={{ display: 'flex', gap: 'var(--space-3)', flexWrap: 'wrap', alignItems: 'flex-end' }}>
              <div className="form-group" style={{ flex: '1 1 200px' }}>
                <label htmlFor="search-input" className="form-label">Search text</label>
                <input
                  id="search-input"
                  type="text"
                  className="form-input"
                  placeholder="Search review text..."
                  value={searchInput}
                  onChange={e => setSearchInput(e.target.value)}
                />
              </div>
              <div className="form-group" style={{ flex: '0 1 180px' }}>
                <label htmlFor="domain-filter" className="form-label">Domain</label>
                <select
                  id="domain-filter"
                  className="form-select"
                  value={domain}
                  onChange={e => { setDomain(e.target.value); handleFilterChange(); }}
                >
                  <option value="">All domains</option>
                  {domains.map(d => (
                    <option key={d} value={d}>{d} ({data?.domain_counts[d]})</option>
                  ))}
                </select>
              </div>
              <div className="form-group" style={{ flex: '0 1 180px' }}>
                <label htmlFor="sentiment-filter" className="form-label">Sentiment</label>
                <select
                  id="sentiment-filter"
                  className="form-select"
                  value={sentiment}
                  onChange={e => { setSentiment(e.target.value); handleFilterChange(); }}
                >
                  <option value="">All sentiments</option>
                  {sentiments.map(s => (
                    <option key={s} value={s}>{s} ({data?.sentiment_counts[s]})</option>
                  ))}
                </select>
              </div>
              <button type="submit" className="btn btn--primary">
                Search
              </button>
              {(search || domain || sentiment) && (
                <button
                  type="button"
                  className="btn btn--ghost"
                  onClick={() => {
                    setSearch('');
                    setSearchInput('');
                    setDomain('');
                    setSentiment('');
                    setPage(1);
                  }}
                >
                  Clear
                </button>
              )}
            </form>
          </div>
        </div>

        {error && (
          <div className="alert alert--error" role="alert">
            <AlertTriangle size={16} />
            <span>{error}</span>
          </div>
        )}

        {loading ? (
          <div className="loading-overlay"><Spinner size="lg" /><span>Loading reviews...</span></div>
        ) : data && (
          <>
            <div style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-muted)', marginBottom: 'var(--space-4)' }}>
              Showing {((page - 1) * 20) + 1}–{Math.min(page * 20, data.total)} of {data.total.toLocaleString()} reviews
            </div>

            {data.reviews.length === 0 ? (
              <div className="empty-state">
                <div className="empty-title">No reviews found</div>
                <p className="empty-desc">Try adjusting your filters or search terms.</p>
              </div>
            ) : (
              <div className="table-container">
                <table>
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Review Text</th>
                      <th>Domain</th>
                      <th>Sentiment</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.reviews.map(review => (
                      <tr key={review.row_index}>
                        <td style={{ color: 'var(--color-text-muted)', fontFamily: 'var(--font-mono)', fontSize: 'var(--font-size-xs)', whiteSpace: 'nowrap' }}>
                          {review.row_index}
                        </td>
                        <td style={{ maxWidth: 500 }}>
                          <p style={{ margin: 0, fontSize: 'var(--font-size-sm)', lineHeight: 'var(--line-height-relaxed)' }}>
                            {review.text}
                          </p>
                        </td>
                        <td><Badge label={review.domain} variant="muted" /></td>
                        <td><SentimentBadge label={review.overall_sentiment} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* Pagination */}
            {data.total_pages > 1 && (
              <div className="pagination">
                <button
                  className="btn btn--ghost btn--sm"
                  onClick={() => setPage(p => Math.max(1, p - 1))}
                  disabled={page <= 1}
                  aria-label="Previous page"
                >
                  <ChevronLeft size={16} />
                </button>
                <span className="pagination-info">Page {data.page} of {data.total_pages}</span>
                <button
                  className="btn btn--ghost btn--sm"
                  onClick={() => setPage(p => Math.min(data.total_pages, p + 1))}
                  disabled={page >= data.total_pages}
                  aria-label="Next page"
                >
                  <ChevronRight size={16} />
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
