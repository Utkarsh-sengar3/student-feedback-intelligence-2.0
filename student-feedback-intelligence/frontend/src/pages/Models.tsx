import { useEffect, useState } from 'react';
import { Activity, AlertTriangle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { api } from '../api/client';
import type { ModelMetrics } from '../api/types';
import { SkeletonCard } from '../components/ui/Spinner';
import { Badge } from '../components/ui/Badge';

function MetricBar({ label, value, max = 1 }: { label: string; value: number; max?: number }) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div className="metric-bar-row">
      <div className="metric-bar-label">{label}</div>
      <div className="metric-bar-track">
        <div className="metric-bar-fill" style={{ width: `${pct}%` }} />
      </div>
      <div className="metric-bar-value">{(value * 100).toFixed(1)}%</div>
    </div>
  );
}

export default function Models() {
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.metrics()
      .then(setMetrics)
      .catch(err => setError(err instanceof Error ? err.message : 'Failed to load metrics.'))
      .finally(() => setLoading(false));
  }, []);

  const sentimentChartData = metrics ? [
    { name: 'Accuracy', value: metrics.sentiment_model.test_accuracy },
    { name: 'Macro F1', value: metrics.sentiment_model.test_macro_f1 },
    { name: 'Precision', value: metrics.sentiment_model.test_macro_precision },
    { name: 'Recall', value: metrics.sentiment_model.test_macro_recall },
  ] : [];

  const retrievalChartData = metrics ? [
    { name: 'MRR', value: metrics.retrieval_model.mrr },
    { name: 'NDCG@5', value: metrics.retrieval_model.ndcg_at_5 },
    { name: 'P@5', value: metrics.retrieval_model.precision_at_5 },
  ] : [];

  return (
    <div className="page">
      <div className="container">
        <div className="page-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', marginBottom: 'var(--space-2)' }}>
            <Activity size={28} style={{ color: 'var(--color-accent)' }} />
            <h1 className="page-title" style={{ margin: 0 }}>Model Performance</h1>
          </div>
          <p className="page-description">
            Real evaluation metrics from the three ML models powering this platform.
          </p>
        </div>

        {error && (
          <div className="alert alert--error" role="alert">
            <AlertTriangle size={16} />
            <span>{error}</span>
          </div>
        )}

        {loading ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
            <SkeletonCard /><SkeletonCard /><SkeletonCard />
          </div>
        ) : metrics && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-6)' }}>

            {/* Sentiment Model */}
            <div className="card">
              <div className="card-header">
                <span className="card-title">Sentiment Model — {metrics.sentiment_model.name}</span>
                <Badge label="Phase 10" variant="accent" />
              </div>
              <div className="card-body">
                <div className="grid-2">
                  <div>
                    <MetricBar label="Test Accuracy" value={metrics.sentiment_model.test_accuracy} />
                    <MetricBar label="Macro F1" value={metrics.sentiment_model.test_macro_f1} />
                    <MetricBar label="Macro Precision" value={metrics.sentiment_model.test_macro_precision} />
                    <MetricBar label="Macro Recall" value={metrics.sentiment_model.test_macro_recall} />
                    <MetricBar label="Val Macro F1" value={metrics.sentiment_model.validation_macro_f1} />
                  </div>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={sentimentChartData} margin={{ top: 0, right: 8, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                      <XAxis dataKey="name" tick={{ fill: 'var(--color-text-muted)', fontSize: 11 }} />
                      <YAxis domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={{ fill: 'var(--color-text-muted)', fontSize: 11 }} />
                      <Tooltip formatter={(v: number) => `${(v * 100).toFixed(1)}%`} contentStyle={{ background: 'var(--color-surface-solid)', border: '1px solid var(--color-border)', borderRadius: 8, color: 'var(--color-text-primary)' }} />
                      <Bar dataKey="value" fill="var(--color-accent)" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-muted)', marginTop: 'var(--space-4)' }}>
                  Architecture: {String(metrics.sentiment_model.base_architecture)}
                </p>
              </div>
            </div>

            {/* Category Model */}
            <div className="card">
              <div className="card-header">
                <span className="card-title">Category Model — {metrics.category_model.name}</span>
                <Badge label="Phase 12" variant="accent" />
              </div>
              <div className="card-body">
                <div className="grid-2" style={{ marginBottom: 'var(--space-6)' }}>
                  <div>
                    <MetricBar label="Micro F1" value={metrics.category_model.micro_f1} />
                    <MetricBar label="Macro F1" value={metrics.category_model.macro_f1} />
                    <MetricBar label="Subset Accuracy" value={metrics.category_model.subset_accuracy} />
                    <div className="metric-bar-row">
                      <div className="metric-bar-label">Hamming Loss</div>
                      <div className="metric-bar-track">
                        <div className="metric-bar-fill" style={{ width: `${metrics.category_model.hamming_loss * 100}%`, background: 'var(--color-success)' }} />
                      </div>
                      <div className="metric-bar-value" style={{ color: 'var(--color-success)' }}>
                        {(metrics.category_model.hamming_loss * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                  <div>
                    <div className="stat-label" style={{ marginBottom: 'var(--space-3)' }}>Task</div>
                    <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-secondary)' }}>
                      {metrics.category_model.task}
                    </p>
                    <div style={{ marginTop: 'var(--space-3)' }}>
                      <div className="stat-label">Categories ({metrics.category_model.category_count})</div>
                    </div>
                  </div>
                </div>
                {metrics.category_model.categories.length > 0 && (
                  <div>
                    <div className="section-label">All {metrics.category_model.category_count} categories</div>
                    <div className="tags">
                      {metrics.category_model.categories.map((cat: string) => (
                        <Badge key={cat} label={cat} variant="muted" />
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Retrieval Model */}
            <div className="card">
              <div className="card-header">
                <span className="card-title">Retrieval Model — {metrics.retrieval_model.name}</span>
                <Badge label="Phase 18" variant="accent" />
              </div>
              <div className="card-body">
                <div className="grid-2">
                  <div>
                    <MetricBar label="MRR" value={metrics.retrieval_model.mrr} />
                    <MetricBar label="NDCG@5" value={metrics.retrieval_model.ndcg_at_5} />
                    <MetricBar label="Precision@5" value={metrics.retrieval_model.precision_at_5} />
                    <div className="metric-bar-row">
                      <div className="metric-bar-label">Reviews Indexed</div>
                      <div style={{ fontWeight: 700, fontSize: 'var(--font-size-lg)', color: 'var(--color-text-primary)' }}>
                        {metrics.retrieval_model.indexed_reviews?.toLocaleString()}
                      </div>
                    </div>
                  </div>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={retrievalChartData} margin={{ top: 0, right: 8, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                      <XAxis dataKey="name" tick={{ fill: 'var(--color-text-muted)', fontSize: 11 }} />
                      <YAxis domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={{ fill: 'var(--color-text-muted)', fontSize: 11 }} />
                      <Tooltip formatter={(v: number) => `${(v * 100).toFixed(1)}%`} contentStyle={{ background: 'var(--color-surface-solid)', border: '1px solid var(--color-border)', borderRadius: 8, color: 'var(--color-text-primary)' }} />
                      <Bar dataKey="value" fill="var(--color-success)" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            {/* Insights Summary */}
            <div className="card">
              <div className="card-header">
                <span className="card-title">Insights Pipeline Summary</span>
                <Badge label="Phase 16" variant="accent" />
              </div>
              <div className="card-body">
                <div className="stats-grid">
                  <div className="stat-card">
                    <div className="stat-label">Concepts Analysed</div>
                    <div className="stat-value">{metrics.insights_summary.concepts_with_insights}</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Critical Concerns</div>
                    <div className="stat-value" style={{ color: 'var(--color-danger)' }}>
                      {metrics.insights_summary.critical_concerns}
                    </div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">High Concerns</div>
                    <div className="stat-value" style={{ color: 'var(--color-warning)' }}>
                      {metrics.insights_summary.high_concerns}
                    </div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Medium Concerns</div>
                    <div className="stat-value">{metrics.insights_summary.medium_concerns}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
