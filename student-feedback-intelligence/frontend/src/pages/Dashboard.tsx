import { useEffect, useState } from 'react';
import { BarChart2, AlertTriangle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { api } from '../api/client';
import type { InsightRecord, InsightSummary, StatusResponse } from '../api/types';
import { SkeletonCard } from '../components/ui/Spinner';
import { PriorityBadge } from '../components/ui/Badge';

export default function Dashboard() {
  const [summary, setSummary] = useState<InsightSummary | null>(null);
  const [insights, setInsights] = useState<InsightRecord[]>([]);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      api.insights.summary(),
      api.insights.top(10),
      api.status(),
    ])
      .then(([summaryRes, topRes, statusRes]) => {
        setSummary(summaryRes.data);
        setInsights(topRes.data);
        setStatus(statusRes);
      })
      .catch(err => setError(err instanceof Error ? err.message : 'Failed to load dashboard.'))
      .finally(() => setLoading(false));
  }, []);

  const priorityChartData = summary?.priority_distribution
    ? Object.entries(summary.priority_distribution).map(([name, value]) => ({ name, value }))
    : [];

  const severityChartData = summary?.severity_distribution
    ? Object.entries(summary.severity_distribution).map(([name, value]) => ({ name, value }))
    : [];

  return (
    <div className="page">
      <div className="container">
        <div className="page-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', marginBottom: 'var(--space-2)' }}>
            <BarChart2 size={28} style={{ color: 'var(--color-accent)' }} />
            <h1 className="page-title" style={{ margin: 0 }}>Insights Dashboard</h1>
          </div>
          <p className="page-description">
            Priority-ranked actionable insights derived from the Phase 16 analysis pipeline.
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
            <SkeletonCard />
            <SkeletonCard />
          </div>
        ) : (
          <>
            {/* API Status */}
            {status && (
              <div className="card" style={{ marginBottom: 'var(--space-6)' }}>
                <div className="card-header">
                  <span className="card-title">System Status</span>
                </div>
                <div className="card-body">
                  <div className="stats-grid">
                    {[
                      { label: 'Sentiment Model', value: status.models.sentiment_model_loaded },
                      { label: 'Category Model', value: status.models.category_model_loaded },
                      { label: 'Insights Loaded', value: status.models.phase16_insights_loaded },
                      { label: 'Retriever Loaded', value: status.models.semantic_retriever_loaded },
                    ].map(item => (
                      <div key={item.label} className="stat-card" style={{ padding: 'var(--space-4)' }}>
                        <div className="stat-label">{item.label}</div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                          <span
                            className={`status-dot status-dot--${item.value ? 'online' : 'offline'}`}
                          />
                          <span style={{ fontSize: 'var(--font-size-sm)', fontWeight: 600 }}>
                            {item.value ? 'Loaded' : 'Unavailable'}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Summary cards */}
            {summary && (
              <div className="stats-grid" style={{ marginBottom: 'var(--space-6)' }}>
                <div className="card stat-card">
                  <div className="stat-label">Total Insights</div>
                  <div className="stat-value">{summary.total_insights}</div>
                </div>
                {summary.priority_distribution && Object.entries(summary.priority_distribution).map(([k, v]) => (
                  <div key={k} className="card stat-card">
                    <div className="stat-label">{k}</div>
                    <div className="stat-value">{v}</div>
                    <div className="stat-sub">priority issues</div>
                  </div>
                ))}
              </div>
            )}

            {/* Charts */}
            {(priorityChartData.length > 0 || severityChartData.length > 0) && (
              <div className="grid-2" style={{ marginBottom: 'var(--space-6)' }}>
                {priorityChartData.length > 0 && (
                  <div className="card">
                    <div className="card-header">
                      <span className="card-title">Priority Distribution</span>
                    </div>
                    <div className="card-body">
                      <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={priorityChartData} margin={{ top: 0, right: 8, left: -20, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                          <XAxis dataKey="name" tick={{ fill: 'var(--color-text-muted)', fontSize: 11 }} />
                          <YAxis tick={{ fill: 'var(--color-text-muted)', fontSize: 11 }} />
                          <Tooltip
                            contentStyle={{
                              background: 'var(--color-surface-solid)',
                              border: '1px solid var(--color-border)',
                              borderRadius: 8,
                              color: 'var(--color-text-primary)',
                            }}
                          />
                          <Bar dataKey="value" fill="var(--color-accent)" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}
                {severityChartData.length > 0 && (
                  <div className="card">
                    <div className="card-header">
                      <span className="card-title">Severity Distribution</span>
                    </div>
                    <div className="card-body">
                      <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={severityChartData} margin={{ top: 0, right: 8, left: -20, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                          <XAxis dataKey="name" tick={{ fill: 'var(--color-text-muted)', fontSize: 11 }} />
                          <YAxis tick={{ fill: 'var(--color-text-muted)', fontSize: 11 }} />
                          <Tooltip
                            contentStyle={{
                              background: 'var(--color-surface-solid)',
                              border: '1px solid var(--color-border)',
                              borderRadius: 8,
                              color: 'var(--color-text-primary)',
                            }}
                          />
                          <Bar dataKey="value" fill="var(--color-warning)" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Top Insights Table */}
            {insights.length > 0 && (
              <div className="card">
                <div className="card-header">
                  <span className="card-title">Top Priority Insights</span>
                  <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-muted)' }}>
                    Ranked by priority score
                  </span>
                </div>
                <div className="table-container">
                  <table>
                    <thead>
                      <tr>
                        <th>Concept</th>
                        <th>Priority</th>
                        <th>Severity</th>
                        <th>Score</th>
                        <th>Recommended Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {insights.map((ins, i) => (
                        <tr key={i}>
                          <td style={{ fontWeight: 500, color: 'var(--color-text-primary)' }}>
                            {String(ins.concept || '—')}
                          </td>
                          <td>
                            {ins.priority ? <PriorityBadge label={String(ins.priority)} /> : '—'}
                          </td>
                          <td>
                            {ins.severity_level ? (
                              <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-muted)' }}>
                                {String(ins.severity_level)}
                              </span>
                            ) : '—'}
                          </td>
                          <td style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--font-size-xs)' }}>
                            {ins.priority_score != null ? Number(ins.priority_score).toFixed(2) : '—'}
                          </td>
                          <td style={{ maxWidth: 320, fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)' }}>
                            {String(ins.recommended_action || '—')}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {insights.length === 0 && !error && (
              <div className="empty-state">
                <div className="empty-title">No insights loaded</div>
                <p className="empty-desc">Phase 16 insight data may be unavailable.</p>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
