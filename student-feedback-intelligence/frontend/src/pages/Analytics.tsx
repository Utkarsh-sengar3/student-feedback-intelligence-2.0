import { useEffect, useState, useCallback, useMemo } from 'react';
import {
  PieChart as PieIcon,
  Filter,
  RotateCcw,
  Download,
  AlertTriangle,
  Info,
  CheckCircle,
  TrendingDown,
  TrendingUp,
  FileText,
  BarChart2,
  Tag,
  ChevronLeft,
  ChevronRight,
  Search,
} from 'lucide-react';
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import { api } from '../api/client';
import type {
  AnalyticsOverviewResponse,
  AnalyticsRecord,
  AnalyticsFilterParams,
} from '../api/types';
import { SentimentBadge, Badge } from '../components/ui/Badge';
import { SkeletonCard, Spinner } from '../components/ui/Spinner';

const SENTIMENT_COLORS: Record<string, string> = {
  positive: '#10b981',
  negative: '#ef4444',
  neutral: '#f59e0b',
  Positive: '#10b981',
  Negative: '#ef4444',
  Neutral: '#f59e0b',
};

export default function Analytics() {
  // Filter state
  const [selectedDomain, setSelectedDomain] = useState<string>('all');
  const [selectedSentiment, setSelectedSentiment] = useState<string>('all');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [searchInput, setSearchInput] = useState<string>('');

  // Overview data
  const [overview, setOverview] = useState<AnalyticsOverviewResponse | null>(null);
  const [loadingOverview, setLoadingOverview] = useState<boolean>(true);
  const [overviewError, setOverviewError] = useState<string | null>(null);

  // Records / Drilldown state
  const [records, setRecords] = useState<AnalyticsRecord[]>([]);
  const [recordsTotal, setRecordsTotal] = useState<number>(0);
  const [page, setPage] = useState<number>(1);
  const [pageSize] = useState<number>(15);
  const [totalPages, setTotalPages] = useState<number>(1);
  const [loadingRecords, setLoadingRecords] = useState<boolean>(true);
  const [recordsError, setRecordsError] = useState<string | null>(null);

  // Active filter params memo
  const filterParams = useMemo<AnalyticsFilterParams>(() => {
    const params: AnalyticsFilterParams = {};
    if (selectedDomain !== 'all') params.domain = selectedDomain;
    if (selectedSentiment !== 'all') params.sentiment = selectedSentiment;
    if (selectedCategory !== 'all') params.category = selectedCategory;
    if (searchQuery.trim()) params.search = searchQuery.trim();
    return params;
  }, [selectedDomain, selectedSentiment, selectedCategory, searchQuery]);

  // Load Overview Data
  const loadOverview = useCallback(() => {
    setLoadingOverview(true);
    setOverviewError(null);
    api.analytics
      .overview(filterParams)
      .then((data) => setOverview(data))
      .catch((err) =>
        setOverviewError(err instanceof Error ? err.message : 'Failed to load analytics overview.')
      )
      .finally(() => setLoadingOverview(false));
  }, [filterParams]);

  // Load Paginated Records
  const loadRecords = useCallback(() => {
    setLoadingRecords(true);
    setRecordsError(null);
    api.analytics
      .records(page, pageSize, filterParams)
      .then((res) => {
        setRecords(res.records);
        setRecordsTotal(res.total);
        setTotalPages(res.total_pages);
      })
      .catch((err) =>
        setRecordsError(err instanceof Error ? err.message : 'Failed to load drill-down records.')
      )
      .finally(() => setLoadingRecords(false));
  }, [page, pageSize, filterParams]);

  useEffect(() => {
    loadOverview();
  }, [loadOverview]);

  useEffect(() => {
    loadRecords();
  }, [loadRecords]);

  // Handlers
  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setSearchQuery(searchInput);
    setPage(1);
  };

  const handleResetFilters = () => {
    setSelectedDomain('all');
    setSelectedSentiment('all');
    setSelectedCategory('all');
    setSearchQuery('');
    setSearchInput('');
    setPage(1);
  };

  const handleSentimentSliceClick = (entry: { name: string }) => {
    const target = entry.name.toLowerCase();
    if (selectedSentiment === target) {
      setSelectedSentiment('all');
    } else {
      setSelectedSentiment(target);
    }
    setPage(1);
  };

  const handleCategorySelect = (catName: string) => {
    if (selectedCategory === catName) {
      setSelectedCategory('all');
    } else {
      setSelectedCategory(catName);
    }
    setPage(1);
  };

  const activeFiltersCount =
    (selectedDomain !== 'all' ? 1 : 0) +
    (selectedSentiment !== 'all' ? 1 : 0) +
    (selectedCategory !== 'all' ? 1 : 0) +
    (searchQuery.trim() ? 1 : 0);

  const kpis = overview?.kpis;
  const filterOptions = overview?.filter_options;

  // CSV export URL
  const csvExportUrl = api.analytics.exportUrl(filterParams);

  return (
    <div className="page">
      <div className="container">
        {/* Page Header */}
        <div className="page-header" style={{ marginBottom: 'var(--space-5)' }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              flexWrap: 'wrap',
              gap: 'var(--space-3)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  width: '42px',
                  height: '42px',
                  borderRadius: 'var(--radius-md)',
                  background: 'var(--color-accent-dim)',
                  color: 'var(--color-accent)',
                }}
              >
                <PieIcon size={24} />
              </div>
              <div>
                <h1 className="page-title" style={{ margin: 0 }}>
                  EDA &amp; Analytics
                </h1>
                <p className="page-description" style={{ margin: 0, marginTop: '2px' }}>
                  Interactive exploratory data analysis, sentiment crosstabs, and text linguistics inspired by Power BI.
                </p>
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
              <span
                style={{
                  fontSize: 'var(--font-size-xs)',
                  padding: '4px 10px',
                  borderRadius: 'var(--radius-full)',
                  background: 'var(--color-surface)',
                  border: '1px solid var(--color-border)',
                  color: 'var(--color-text-secondary)',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '6px',
                }}
              >
                <CheckCircle size={14} style={{ color: 'var(--color-success)' }} />
                Indexed Benchmark: 6,500 Canonical Reviews
              </span>
              <a
                href={csvExportUrl}
                download="feedback_records_export.csv"
                className="btn btn--secondary btn--sm"
                style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}
                title="Export matching records to CSV"
              >
                <Download size={14} />
                Export CSV
              </a>
            </div>
          </div>
        </div>

        {/* Global Error Banner */}
        {overviewError && (
          <div className="alert alert--error" role="alert" style={{ marginBottom: 'var(--space-4)' }}>
            <AlertTriangle size={16} />
            <span>{overviewError}</span>
          </div>
        )}

        {/* ================================================================= */}
        {/* INTERACTIVE SLICERS / FILTER BAR (Power BI Style)                 */}
        {/* ================================================================= */}
        <div
          className="card"
          style={{
            marginBottom: 'var(--space-6)',
            borderColor: activeFiltersCount > 0 ? 'var(--color-accent)' : undefined,
            boxShadow: 'var(--shadow-sm)',
          }}
        >
          <div
            className="card-header"
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: 'var(--space-3) var(--space-4)',
              borderBottom: '1px solid var(--color-border)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
              <Filter size={16} style={{ color: 'var(--color-accent)' }} />
              <span style={{ fontWeight: 600, fontSize: 'var(--font-size-sm)' }}>
                Interactive Slicers &amp; Cross-Filters
              </span>
              {activeFiltersCount > 0 && (
                <span
                  style={{
                    fontSize: '11px',
                    fontWeight: 700,
                    padding: '2px 8px',
                    borderRadius: 'var(--radius-full)',
                    background: 'var(--color-accent)',
                    color: '#fff',
                  }}
                >
                  {activeFiltersCount} active
                </span>
              )}
            </div>

            {activeFiltersCount > 0 && (
              <button
                type="button"
                className="btn btn--ghost btn--sm"
                onClick={handleResetFilters}
                style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', fontSize: 'var(--font-size-xs)' }}
              >
                <RotateCcw size={12} />
                Reset Slicers
              </button>
            )}
          </div>

          <div className="card-body" style={{ padding: 'var(--space-4)' }}>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))',
                gap: 'var(--space-4)',
                alignItems: 'flex-end',
              }}
            >
              {/* Domain Slicer */}
              <div className="form-group" style={{ margin: 0 }}>
                <label htmlFor="slicer-domain" className="form-label" style={{ fontSize: 'var(--font-size-xs)' }}>
                  Domain Slicer
                </label>
                <select
                  id="slicer-domain"
                  className="form-select"
                  value={selectedDomain}
                  onChange={(e) => {
                    setSelectedDomain(e.target.value);
                    setPage(1);
                  }}
                  style={{ fontSize: 'var(--font-size-sm)' }}
                >
                  <option value="all">All Domains (3)</option>
                  {filterOptions?.domains.map((dom) => (
                    <option key={dom} value={dom}>
                      {dom}
                    </option>
                  ))}
                </select>
              </div>

              {/* Sentiment Slicer */}
              <div className="form-group" style={{ margin: 0 }}>
                <label htmlFor="slicer-sentiment" className="form-label" style={{ fontSize: 'var(--font-size-xs)' }}>
                  Sentiment Slicer
                </label>
                <select
                  id="slicer-sentiment"
                  className="form-select"
                  value={selectedSentiment}
                  onChange={(e) => {
                    setSelectedSentiment(e.target.value);
                    setPage(1);
                  }}
                  style={{ fontSize: 'var(--font-size-sm)' }}
                >
                  <option value="all">All Sentiments</option>
                  <option value="positive">Positive</option>
                  <option value="negative">Negative</option>
                  <option value="neutral">Neutral</option>
                </select>
              </div>

              {/* Aspect Category Slicer */}
              <div className="form-group" style={{ margin: 0 }}>
                <label htmlFor="slicer-category" className="form-label" style={{ fontSize: 'var(--font-size-xs)' }}>
                  Aspect Category Slicer
                </label>
                <select
                  id="slicer-category"
                  className="form-select"
                  value={selectedCategory}
                  onChange={(e) => {
                    setSelectedCategory(e.target.value);
                    setPage(1);
                  }}
                  style={{ fontSize: 'var(--font-size-sm)' }}
                >
                  <option value="all">All Categories ({filterOptions?.categories.length || 24})</option>
                  {filterOptions?.categories.map((cat) => (
                    <option key={cat} value={cat}>
                      {cat}
                    </option>
                  ))}
                </select>
              </div>

              {/* Text Search Slicer */}
              <div className="form-group" style={{ margin: 0 }}>
                <label htmlFor="slicer-search" className="form-label" style={{ fontSize: 'var(--font-size-xs)' }}>
                  Text Search Filter
                </label>
                <form onSubmit={handleSearchSubmit} style={{ display: 'flex', gap: 'var(--space-2)' }}>
                  <input
                    id="slicer-search"
                    type="text"
                    className="form-input"
                    placeholder="Search keywords..."
                    value={searchInput}
                    onChange={(e) => setSearchInput(e.target.value)}
                    style={{ fontSize: 'var(--font-size-sm)' }}
                  />
                  <button type="submit" className="btn btn--primary btn--sm" aria-label="Apply text search">
                    <Search size={14} />
                  </button>
                </form>
              </div>
            </div>
          </div>
        </div>

        {/* Loading State for KPIs & Overview */}
        {loadingOverview && !overview ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)', marginBottom: 'var(--space-6)' }}>
            <SkeletonCard />
            <SkeletonCard />
          </div>
        ) : (
          <>
            {/* =============================================================== */}
            {/* A. EXECUTIVE KPI CARDS RIBBON                                   */}
            {/* =============================================================== */}
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
                gap: 'var(--space-4)',
                marginBottom: 'var(--space-6)',
              }}
            >
              {/* Total Records */}
              <div className="card stat-card" style={{ padding: 'var(--space-4)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span className="stat-label">Feedback Volume</span>
                  <FileText size={16} style={{ color: 'var(--color-accent)' }} />
                </div>
                <div className="stat-value" style={{ margin: 'var(--space-1) 0' }}>
                  {kpis ? kpis.total_records.toLocaleString() : '—'}
                </div>
                <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-muted)' }}>
                  Matching active slicers
                </div>
              </div>

              {/* Positive Sentiment */}
              <div
                className="card stat-card"
                style={{
                  padding: 'var(--space-4)',
                  borderLeft: '4px solid var(--color-success)',
                  cursor: 'pointer',
                }}
                onClick={() => {
                  setSelectedSentiment(selectedSentiment === 'positive' ? 'all' : 'positive');
                  setPage(1);
                }}
                title="Click to toggle positive sentiment filter"
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span className="stat-label">Positive Feedback</span>
                  <TrendingUp size={16} style={{ color: 'var(--color-success)' }} />
                </div>
                <div className="stat-value" style={{ color: 'var(--color-success)', margin: 'var(--space-1) 0' }}>
                  {kpis ? `${kpis.positive_pct}%` : '—'}
                </div>
                <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-muted)' }}>
                  {kpis ? `${kpis.positive_count.toLocaleString()} records` : ''}
                </div>
              </div>

              {/* Negative Sentiment */}
              <div
                className="card stat-card"
                style={{
                  padding: 'var(--space-4)',
                  borderLeft: '4px solid var(--color-danger)',
                  cursor: 'pointer',
                }}
                onClick={() => {
                  setSelectedSentiment(selectedSentiment === 'negative' ? 'all' : 'negative');
                  setPage(1);
                }}
                title="Click to toggle negative sentiment filter"
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span className="stat-label">Negative Feedback</span>
                  <TrendingDown size={16} style={{ color: 'var(--color-danger)' }} />
                </div>
                <div className="stat-value" style={{ color: 'var(--color-danger)', margin: 'var(--space-1) 0' }}>
                  {kpis ? `${kpis.negative_pct}%` : '—'}
                </div>
                <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-muted)' }}>
                  {kpis ? `${kpis.negative_count.toLocaleString()} records` : ''}
                </div>
              </div>

              {/* Neutral Sentiment */}
              <div
                className="card stat-card"
                style={{
                  padding: 'var(--space-4)',
                  borderLeft: '4px solid var(--color-warning)',
                  cursor: 'pointer',
                }}
                onClick={() => {
                  setSelectedSentiment(selectedSentiment === 'neutral' ? 'all' : 'neutral');
                  setPage(1);
                }}
                title="Click to toggle neutral sentiment filter"
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span className="stat-label">Neutral Feedback</span>
                  <BarChart2 size={16} style={{ color: 'var(--color-warning)' }} />
                </div>
                <div className="stat-value" style={{ color: 'var(--color-warning)', margin: 'var(--space-1) 0' }}>
                  {kpis ? `${kpis.neutral_pct}%` : '—'}
                </div>
                <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-muted)' }}>
                  {kpis ? `${kpis.neutral_count.toLocaleString()} records` : ''}
                </div>
              </div>

              {/* Scope & Dimensions */}
              <div className="card stat-card" style={{ padding: 'var(--space-4)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span className="stat-label">Analytical Scope</span>
                  <Tag size={16} style={{ color: 'var(--color-accent)' }} />
                </div>
                <div className="stat-value" style={{ margin: 'var(--space-1) 0' }}>
                  {kpis ? `${kpis.distinct_domains} Dom / ${kpis.distinct_categories} Cat` : '—'}
                </div>
                <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-muted)' }}>
                  Unique domains &amp; categories
                </div>
              </div>

              {/* Avg Length */}
              <div className="card stat-card" style={{ padding: 'var(--space-4)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span className="stat-label">Avg Review Length</span>
                  <FileText size={16} style={{ color: 'var(--color-text-muted)' }} />
                </div>
                <div className="stat-value" style={{ margin: 'var(--space-1) 0' }}>
                  {kpis ? `${kpis.avg_word_count} wds` : '—'}
                </div>
                <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-muted)' }}>
                  {kpis ? `${kpis.avg_char_length} chars average` : ''}
                </div>
              </div>
            </div>

            {/* =============================================================== */}
            {/* B & D. VISUAL ANALYTICS ROW 1: SENTIMENT & DOMAINS             */}
            {/* =============================================================== */}
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))',
                gap: 'var(--space-6)',
                marginBottom: 'var(--space-6)',
              }}
            >
              {/* Sentiment Distribution Donut */}
              <div className="card">
                <div className="card-header">
                  <div>
                    <span className="card-title">Sentiment Share &amp; Ratio</span>
                    <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-muted)' }}>
                      Interactive breakdown by overall sentiment (click slice to filter)
                    </div>
                  </div>
                </div>
                <div className="card-body" style={{ minHeight: '300px', display: 'flex', alignItems: 'center' }}>
                  {overview?.sentiment_distribution && overview.sentiment_distribution.length > 0 ? (
                    <ResponsiveContainer width="100%" height={280}>
                      <PieChart>
                        <Pie
                          data={overview.sentiment_distribution}
                          dataKey="count"
                          nameKey="name"
                          cx="50%"
                          cy="50%"
                          innerRadius={65}
                          outerRadius={95}
                          paddingAngle={3}
                          onClick={handleSentimentSliceClick}
                          style={{ cursor: 'pointer' }}
                        >
                          {overview.sentiment_distribution.map((entry) => (
                            <Cell
                              key={`cell-${entry.name}`}
                              fill={SENTIMENT_COLORS[entry.name.toLowerCase()] || '#6366f1'}
                            />
                          ))}
                        </Pie>
                        <Tooltip
                          contentStyle={{
                            backgroundColor: 'var(--color-surface-solid)',
                            borderColor: 'var(--color-border)',
                            color: 'var(--color-text-primary)',
                            borderRadius: 'var(--radius-md)',
                            boxShadow: 'var(--shadow-md)',
                          }}
                          formatter={(value: any, name: any, item: any) => [
                            `${Number(value).toLocaleString()} reviews (${item.payload.percentage}%)`,
                            name,
                          ]}
                        />
                        <Legend
                          verticalAlign="bottom"
                          height={36}
                          formatter={(value, entry: any) => (
                            <span style={{ color: 'var(--color-text-primary)', fontSize: 'var(--font-size-xs)' }}>
                              {value} ({entry.payload.count.toLocaleString()} · {entry.payload.percentage}%)
                            </span>
                          )}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  ) : (
                    <div style={{ textAlign: 'center', width: '100%', color: 'var(--color-text-muted)' }}>
                      No sentiment data available for current filter.
                    </div>
                  )}
                </div>
              </div>

              {/* Domain Sentiment Cross-Tabulation */}
              <div className="card">
                <div className="card-header">
                  <div>
                    <span className="card-title">Domain Sentiment Distribution</span>
                    <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-muted)' }}>
                      Comparison across Teacher, Course, and University review domains
                    </div>
                  </div>
                </div>
                <div className="card-body" style={{ minHeight: '300px' }}>
                  {overview?.domain_distribution && overview.domain_distribution.length > 0 ? (
                    <ResponsiveContainer width="100%" height={280}>
                      <BarChart
                        data={overview.domain_distribution}
                        margin={{ top: 20, right: 20, left: 0, bottom: 20 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                        <XAxis
                          dataKey="domain"
                          stroke="var(--color-text-muted)"
                          fontSize={12}
                          tickFormatter={(val) =>
                            val.length > 15 ? `${val.substring(0, 15)}…` : val
                          }
                        />
                        <YAxis stroke="var(--color-text-muted)" fontSize={12} />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: 'var(--color-surface-solid)',
                            borderColor: 'var(--color-border)',
                            color: 'var(--color-text-primary)',
                            borderRadius: 'var(--radius-md)',
                            boxShadow: 'var(--shadow-md)',
                          }}
                        />
                        <Legend verticalAlign="top" height={36} />
                        <Bar dataKey="positive" name="Positive" stackId="a" fill="#10b981" />
                        <Bar dataKey="neutral" name="Neutral" stackId="a" fill="#f59e0b" />
                        <Bar dataKey="negative" name="Negative" stackId="a" fill="#ef4444" />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <div style={{ textAlign: 'center', width: '100%', color: 'var(--color-text-muted)' }}>
                      No domain data available for current filter.
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* =============================================================== */}
            {/* C. CATEGORY ANALYSIS                                            */}
            {/* =============================================================== */}
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))',
                gap: 'var(--space-6)',
                marginBottom: 'var(--space-6)',
              }}
            >
              {/* Category Volume Stacked Chart */}
              <div className="card">
                <div className="card-header">
                  <div>
                    <span className="card-title">Top Aspect Categories by Volume</span>
                    <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-muted)' }}>
                      Sentiment breakdown across the most frequent feedback aspect categories
                    </div>
                  </div>
                </div>
                <div className="card-body" style={{ minHeight: '340px' }}>
                  {overview?.category_volume && overview.category_volume.length > 0 ? (
                    <ResponsiveContainer width="100%" height={320}>
                      <BarChart
                        data={overview.category_volume}
                        layout="vertical"
                        margin={{ top: 10, right: 30, left: 100, bottom: 10 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                        <XAxis type="number" stroke="var(--color-text-muted)" fontSize={11} />
                        <YAxis
                          type="category"
                          dataKey="category"
                          stroke="var(--color-text-muted)"
                          fontSize={11}
                          width={95}
                          tickFormatter={(val) =>
                            val.length > 16 ? `${val.substring(0, 15)}…` : val
                          }
                        />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: 'var(--color-surface-solid)',
                            borderColor: 'var(--color-border)',
                            color: 'var(--color-text-primary)',
                            borderRadius: 'var(--radius-md)',
                            boxShadow: 'var(--shadow-md)',
                          }}
                        />
                        <Legend verticalAlign="top" height={36} />
                        <Bar dataKey="positive" name="Positive" stackId="cat" fill="#10b981" />
                        <Bar dataKey="neutral" name="Neutral" stackId="cat" fill="#f59e0b" />
                        <Bar dataKey="negative" name="Negative" stackId="cat" fill="#ef4444" />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <div style={{ textAlign: 'center', width: '100%', color: 'var(--color-text-muted)' }}>
                      No aspect category annotations matching active slicers.
                    </div>
                  )}
                </div>
              </div>

              {/* Strengths vs Critical Areas of Concern */}
              <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
                <div className="card-header">
                  <div>
                    <span className="card-title">Category Sentiment Rankings</span>
                    <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-muted)' }}>
                      Highest dissatisfaction areas vs highest satisfaction strengths
                    </div>
                  </div>
                </div>
                <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
                  {/* Critical Areas */}
                  <div>
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        marginBottom: 'var(--space-2)',
                        fontSize: 'var(--font-size-sm)',
                        fontWeight: 600,
                        color: 'var(--color-danger)',
                      }}
                    >
                      <TrendingDown size={16} />
                      Top Areas of Concern (Highest % Negative)
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      {overview?.top_negative_categories?.map((cat) => (
                        <div
                          key={`neg-${cat.category}`}
                          onClick={() => handleCategorySelect(cat.category)}
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            padding: '6px 10px',
                            borderRadius: 'var(--radius-sm)',
                            background:
                              selectedCategory === cat.category
                                ? 'var(--color-danger-dim)'
                                : 'var(--color-surface)',
                            border: '1px solid var(--color-border)',
                            cursor: 'pointer',
                            fontSize: 'var(--font-size-xs)',
                          }}
                          title="Click to filter by this category"
                        >
                          <span style={{ fontWeight: 500 }}>{cat.category}</span>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                            <span style={{ color: 'var(--color-text-muted)' }}>
                              {cat.count} / {cat.total_in_category}
                            </span>
                            <span
                              style={{
                                fontWeight: 700,
                                color: 'var(--color-danger)',
                                minWidth: '45px',
                                textAlign: 'right',
                              }}
                            >
                              {cat.percentage}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Top Strengths */}
                  <div>
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        marginBottom: 'var(--space-2)',
                        fontSize: 'var(--font-size-sm)',
                        fontWeight: 600,
                        color: 'var(--color-success)',
                      }}
                    >
                      <TrendingUp size={16} />
                      Top Institutional Strengths (Highest % Positive)
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      {overview?.top_positive_categories?.map((cat) => (
                        <div
                          key={`pos-${cat.category}`}
                          onClick={() => handleCategorySelect(cat.category)}
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            padding: '6px 10px',
                            borderRadius: 'var(--radius-sm)',
                            background:
                              selectedCategory === cat.category
                                ? 'var(--color-success-dim)'
                                : 'var(--color-surface)',
                            border: '1px solid var(--color-border)',
                            cursor: 'pointer',
                            fontSize: 'var(--font-size-xs)',
                          }}
                          title="Click to filter by this category"
                        >
                          <span style={{ fontWeight: 500 }}>{cat.category}</span>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                            <span style={{ color: 'var(--color-text-muted)' }}>
                              {cat.count} / {cat.total_in_category}
                            </span>
                            <span
                              style={{
                                fontWeight: 700,
                                color: 'var(--color-success)',
                                minWidth: '45px',
                                textAlign: 'right',
                              }}
                            >
                              {cat.percentage}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* =============================================================== */}
            {/* E. TEXT EDA & LINGUISTICS                                       */}
            {/* =============================================================== */}
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))',
                gap: 'var(--space-6)',
                marginBottom: 'var(--space-6)',
              }}
            >
              {/* Word Count Distribution Histogram */}
              <div className="card">
                <div className="card-header">
                  <div>
                    <span className="card-title">Review Length Distribution</span>
                    <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-muted)' }}>
                      Histogram of feedback word count bins
                    </div>
                  </div>
                </div>
                <div className="card-body" style={{ minHeight: '280px' }}>
                  {overview?.text_eda.word_count_distribution &&
                  overview.text_eda.word_count_distribution.length > 0 ? (
                    <ResponsiveContainer width="100%" height={260}>
                      <BarChart
                        data={overview.text_eda.word_count_distribution}
                        margin={{ top: 15, right: 15, left: 0, bottom: 15 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                        <XAxis dataKey="bin" stroke="var(--color-text-muted)" fontSize={12} />
                        <YAxis stroke="var(--color-text-muted)" fontSize={12} />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: 'var(--color-surface-solid)',
                            borderColor: 'var(--color-border)',
                            color: 'var(--color-text-primary)',
                            borderRadius: 'var(--radius-md)',
                            boxShadow: 'var(--shadow-md)',
                          }}
                          formatter={(val: any) => [`${Number(val).toLocaleString()} reviews`, 'Count']}
                        />
                        <Bar dataKey="count" name="Reviews" fill="#6366f1" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <div style={{ textAlign: 'center', width: '100%', color: 'var(--color-text-muted)' }}>
                      No text length distribution data available.
                    </div>
                  )}
                </div>
              </div>

              {/* Distinctive Sentiment Terms */}
              <div className="card">
                <div className="card-header">
                  <div>
                    <span className="card-title">Top Distinctive Feedback Terms</span>
                    <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-muted)' }}>
                      Frequent vocabulary in positive vs negative feedback (normalized)
                    </div>
                  </div>
                </div>
                <div className="card-body">
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: '1fr 1fr',
                      gap: 'var(--space-4)',
                      marginBottom: 'var(--space-4)',
                    }}
                  >
                    {/* Positive Terms */}
                    <div>
                      <div
                        style={{
                          fontSize: 'var(--font-size-xs)',
                          fontWeight: 700,
                          color: 'var(--color-success)',
                          marginBottom: 'var(--space-2)',
                        }}
                      >
                        Positive Descriptors
                      </div>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                        {overview?.text_eda.top_positive_terms.map((item) => (
                          <span
                            key={`pterm-${item.term}`}
                            style={{
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '4px',
                              padding: '3px 8px',
                              borderRadius: 'var(--radius-sm)',
                              background: 'var(--color-success-dim)',
                              border: '1px solid var(--color-success)',
                              color: 'var(--color-success)',
                              fontSize: '11px',
                              fontWeight: 600,
                            }}
                          >
                            {item.term}
                            <span style={{ opacity: 0.75, fontSize: '10px' }}>({item.count})</span>
                          </span>
                        ))}
                      </div>
                    </div>

                    {/* Negative Terms */}
                    <div>
                      <div
                        style={{
                          fontSize: 'var(--font-size-xs)',
                          fontWeight: 700,
                          color: 'var(--color-danger)',
                          marginBottom: 'var(--space-2)',
                        }}
                      >
                        Negative Descriptors
                      </div>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                        {overview?.text_eda.top_negative_terms.map((item) => (
                          <span
                            key={`nterm-${item.term}`}
                            style={{
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '4px',
                              padding: '3px 8px',
                              borderRadius: 'var(--radius-sm)',
                              background: 'var(--color-danger-dim)',
                              border: '1px solid var(--color-danger)',
                              color: 'var(--color-danger)',
                              fontSize: '11px',
                              fontWeight: 600,
                            }}
                          >
                            {item.term}
                            <span style={{ opacity: 0.75, fontSize: '10px' }}>({item.count})</span>
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div
                    style={{
                      padding: 'var(--space-2) var(--space-3)',
                      borderRadius: 'var(--radius-sm)',
                      background: 'var(--color-surface)',
                      border: '1px solid var(--color-border)',
                      fontSize: '11px',
                      color: 'var(--color-text-muted)',
                      lineHeight: '1.4',
                    }}
                  >
                    <strong>Methodology: </strong>
                    {overview?.text_eda.normalization_notes}
                  </div>
                </div>
              </div>
            </div>

            {/* =============================================================== */}
            {/* F. TEMPORAL ANALYSIS NOTICE CARD                                */}
            {/* =============================================================== */}
            <div
              className="card"
              style={{
                marginBottom: 'var(--space-6)',
                background: 'var(--color-info-dim)',
                borderColor: 'var(--color-info)',
              }}
            >
              <div
                className="card-body"
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: 'var(--space-3)',
                  padding: 'var(--space-4)',
                }}
              >
                <Info size={20} style={{ color: 'var(--color-info)', flexShrink: 0, marginTop: '2px' }} />
                <div>
                  <div style={{ fontWeight: 600, fontSize: 'var(--font-size-sm)', color: 'var(--color-info)' }}>
                    Temporal Trend Analysis Notice
                  </div>
                  <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)', marginTop: '4px' }}>
                    {overview?.temporal_notice.reason}
                  </div>
                </div>
              </div>
            </div>

            {/* =============================================================== */}
            {/* H. DRILL-DOWN RECORD EXPLORER & DATA GRID                       */}
            {/* =============================================================== */}
            <div className="card">
              <div
                className="card-header"
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  flexWrap: 'wrap',
                  gap: 'var(--space-3)',
                }}
              >
                <div>
                  <span className="card-title">Drill-Down Data Grid</span>
                  <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-muted)' }}>
                    Viewing {recordsTotal.toLocaleString()} real feedback records matching active slicers
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                  <a
                    href={csvExportUrl}
                    download="feedback_records_export.csv"
                    className="btn btn--secondary btn--sm"
                    style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}
                  >
                    <Download size={14} />
                    Download CSV ({recordsTotal.toLocaleString()})
                  </a>
                </div>
              </div>

              <div className="card-body" style={{ padding: 0 }}>
                {recordsError && (
                  <div className="alert alert--error" role="alert" style={{ margin: 'var(--space-4)' }}>
                    <AlertTriangle size={16} />
                    <span>{recordsError}</span>
                  </div>
                )}

                {loadingRecords ? (
                  <div style={{ padding: 'var(--space-8)', textAlign: 'center' }}>
                    <Spinner size="md" />
                    <div style={{ marginTop: 'var(--space-2)', fontSize: 'var(--font-size-sm)', color: 'var(--color-text-muted)' }}>
                      Loading records...
                    </div>
                  </div>
                ) : records.length === 0 ? (
                  <div style={{ padding: 'var(--space-8)', textAlign: 'center', color: 'var(--color-text-muted)' }}>
                    No feedback records match the current slicer criteria.
                  </div>
                ) : (
                  <div style={{ overflowX: 'auto' }}>
                    <table
                      style={{
                        width: '100%',
                        borderCollapse: 'collapse',
                        fontSize: 'var(--font-size-sm)',
                        textAlign: 'left',
                      }}
                    >
                      <thead>
                        <tr
                          style={{
                            borderBottom: '1px solid var(--color-border)',
                            background: 'var(--color-surface)',
                            color: 'var(--color-text-muted)',
                            fontSize: 'var(--font-size-xs)',
                            textTransform: 'uppercase',
                            letterSpacing: '0.05em',
                          }}
                        >
                          <th style={{ padding: '10px 16px', width: '80px' }}>ID</th>
                          <th style={{ padding: '10px 16px', width: '130px' }}>Domain</th>
                          <th style={{ padding: '10px 16px', width: '110px' }}>Sentiment</th>
                          <th style={{ padding: '10px 16px', width: '220px' }}>Aspect Categories</th>
                          <th style={{ padding: '10px 16px' }}>Feedback Text</th>
                          <th style={{ padding: '10px 16px', width: '80px', textAlign: 'right' }}>Words</th>
                        </tr>
                      </thead>
                      <tbody>
                        {records.map((rec) => (
                          <tr
                            key={rec.review_id}
                            style={{
                              borderBottom: '1px solid var(--color-border)',
                              transition: 'background var(--transition-fast)',
                            }}
                          >
                            <td style={{ padding: '12px 16px', fontWeight: 600, color: 'var(--color-text-muted)' }}>
                              #{rec.review_id}
                            </td>
                            <td style={{ padding: '12px 16px' }}>
                              <Badge label={rec.domain} />
                            </td>
                            <td style={{ padding: '12px 16px' }}>
                              <SentimentBadge label={rec.overall_sentiment} />
                            </td>
                            <td style={{ padding: '12px 16px' }}>
                              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                                {rec.categories && rec.categories.length > 0 ? (
                                  rec.categories.map((c) => (
                                    <span
                                      key={`rcat-${rec.review_id}-${c}`}
                                      onClick={() => handleCategorySelect(c)}
                                      style={{
                                        fontSize: '11px',
                                        padding: '2px 6px',
                                        borderRadius: 'var(--radius-sm)',
                                        background:
                                          selectedCategory === c
                                            ? 'var(--color-accent)'
                                            : 'var(--color-surface)',
                                        color:
                                          selectedCategory === c
                                            ? '#ffffff'
                                            : 'var(--color-text-secondary)',
                                        border: '1px solid var(--color-border)',
                                        cursor: 'pointer',
                                      }}
                                      title="Filter by this category"
                                    >
                                      {c}
                                    </span>
                                  ))
                                ) : (
                                  <span style={{ fontSize: '11px', color: 'var(--color-text-muted)' }}>
                                    General
                                  </span>
                                )}
                              </div>
                            </td>
                            <td
                              style={{
                                padding: '12px 16px',
                                color: 'var(--color-text-primary)',
                                lineHeight: '1.45',
                                maxWidth: '480px',
                              }}
                            >
                              {rec.text}
                            </td>
                            <td
                              style={{
                                padding: '12px 16px',
                                textAlign: 'right',
                                color: 'var(--color-text-muted)',
                                fontVariantNumeric: 'tabular-nums',
                              }}
                            >
                              {rec.word_count}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}

                {/* Pagination Controls */}
                {totalPages > 1 && (
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: 'var(--space-3) var(--space-4)',
                      borderTop: '1px solid var(--color-border)',
                      fontSize: 'var(--font-size-xs)',
                      color: 'var(--color-text-secondary)',
                    }}
                  >
                    <div>
                      Page <strong>{page}</strong> of <strong>{totalPages}</strong> (
                      {recordsTotal.toLocaleString()} records)
                    </div>
                    <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
                      <button
                        className="btn btn--secondary btn--sm"
                        disabled={page <= 1}
                        onClick={() => setPage((p) => Math.max(1, p - 1))}
                        style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}
                      >
                        <ChevronLeft size={14} />
                        Previous
                      </button>
                      <button
                        className="btn btn--secondary btn--sm"
                        disabled={page >= totalPages}
                        onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                        style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}
                      >
                        Next
                        <ChevronRight size={14} />
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
