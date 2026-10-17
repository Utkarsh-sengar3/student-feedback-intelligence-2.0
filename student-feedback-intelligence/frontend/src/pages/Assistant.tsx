import { useState } from 'react';
import { MessageSquare, Send, AlertTriangle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { api } from '../api/client';
import type { AssistantResponse } from '../api/types';
import { SentimentBadge, Badge } from '../components/ui/Badge';
import { Spinner, SkeletonCard } from '../components/ui/Spinner';
import { CopyButton } from '../components/ui/CopyButton';

const EXAMPLE_QUESTIONS = [
  'What are the most common complaints about teaching quality?',
  'How do students feel about campus Wi-Fi and technology?',
  'What do students say about exam difficulty and grading fairness?',
  'What positive feedback exists about course materials?',
  'What are student concerns about administrative services?',
];

export default function Assistant() {
  const [question, setQuestion] = useState('');
  const [domain, setDomain] = useState('');
  const [sentiment, setSentiment] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AssistantResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.assistant({
        question: question.trim(),
        top_k: 8,
        domain: domain || undefined,
        sentiment: sentiment || undefined,
      });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Assistant request failed.');
    } finally {
      setLoading(false);
    }
  };

  const sentimentChartData = result
    ? Object.entries(result.sentiment_distribution).map(([name, value]) => ({ name, value }))
    : [];

  return (
    <div className="page">
      <div className="container container--narrow">
        <div className="page-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', marginBottom: 'var(--space-2)' }}>
            <MessageSquare size={28} style={{ color: 'var(--color-accent)' }} />
            <h1 className="page-title" style={{ margin: 0 }}>AI Assistant</h1>
          </div>
          <p className="page-description">
            Ask questions about student feedback. Answers are grounded in retrieved evidence
            — no language model, no hallucination.
          </p>
        </div>

        <div className="card">
          <div className="card-body">
            <form onSubmit={handleSubmit}>
              <div className="form-group">
                <label htmlFor="question" className="form-label">Your question</label>
                <textarea
                  id="question"
                  className="form-textarea"
                  placeholder="e.g. What are the most common complaints about teaching quality?"
                  value={question}
                  onChange={e => setQuestion(e.target.value)}
                  rows={3}
                  maxLength={2000}
                  required
                  aria-describedby="question-hint"
                />
                <div className="char-count" id="question-hint">{question.length} / 2000</div>
              </div>

              <div style={{ display: 'flex', gap: 'var(--space-3)', flexWrap: 'wrap' }}>
                <div className="form-group" style={{ flex: '1 1 150px' }}>
                  <label htmlFor="ast-domain" className="form-label">Domain filter</label>
                  <select id="ast-domain" className="form-select" value={domain} onChange={e => setDomain(e.target.value)}>
                    <option value="">All domains</option>
                    <option value="course review">Course Review</option>
                    <option value="teacher review">Teacher Review</option>
                    <option value="university review">University Review</option>
                  </select>
                </div>
                <div className="form-group" style={{ flex: '1 1 150px' }}>
                  <label htmlFor="ast-sentiment" className="form-label">Sentiment filter</label>
                  <select id="ast-sentiment" className="form-select" value={sentiment} onChange={e => setSentiment(e.target.value)}>
                    <option value="">All sentiments</option>
                    <option value="Positive">Positive</option>
                    <option value="Negative">Negative</option>
                    <option value="Neutral">Neutral</option>
                  </select>
                </div>
              </div>

              <div className="form-group">
                <span className="form-label">Example questions</span>
                <div className="examples">
                  {EXAMPLE_QUESTIONS.map((q, i) => (
                    <button
                      key={i}
                      type="button"
                      className="example-btn"
                      onClick={() => setQuestion(q)}
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <button
                  type="submit"
                  className="btn btn--primary"
                  disabled={loading || question.trim().length < 3}
                >
                  {loading ? <Spinner size="sm" /> : <Send size={16} />}
                  {loading ? 'Searching...' : 'Ask'}
                </button>
              </div>
            </form>
          </div>
        </div>

        {loading && (
          <div style={{ marginTop: 'var(--space-6)' }}>
            <SkeletonCard />
          </div>
        )}

        {error && (
          <div className="alert alert--error" style={{ marginTop: 'var(--space-6)' }} role="alert">
            <AlertTriangle size={16} />
            <span>{error}</span>
          </div>
        )}

        {result && !loading && (
          <div style={{ marginTop: 'var(--space-6)', display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
            {/* Summary */}
            <div className="card">
              <div className="card-header">
                <span className="card-title">Summary</span>
                <Badge label={result.answer_type.replace(/_/g, ' ')} variant="accent" />
                <CopyButton text={result.summary} label="Copy" />
              </div>
              <div className="card-body">
                <p style={{ lineHeight: 'var(--line-height-relaxed)', color: 'var(--color-text-secondary)' }}>
                  {result.summary}
                </p>
                <div style={{ marginTop: 'var(--space-3)', display: 'flex', gap: 'var(--space-4)', fontSize: 'var(--font-size-sm)', color: 'var(--color-text-muted)' }}>
                  <span>Retrieved: <strong style={{ color: 'var(--color-text-primary)' }}>{result.retrieved_count}</strong> reviews</span>
                  <span>Negative share: <strong style={{ color: 'var(--color-danger)' }}>{(result.negative_share * 100).toFixed(0)}%</strong></span>
                </div>
              </div>
            </div>

            {/* Sentiment chart */}
            {sentimentChartData.length > 0 && (
              <div className="card">
                <div className="card-header"><span className="card-title">Sentiment Distribution</span></div>
                <div className="card-body">
                  <ResponsiveContainer width="100%" height={160}>
                    <BarChart data={sentimentChartData} margin={{ top: 0, right: 8, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                      <XAxis dataKey="name" tick={{ fill: 'var(--color-text-muted)', fontSize: 12 }} />
                      <YAxis tick={{ fill: 'var(--color-text-muted)', fontSize: 12 }} />
                      <Tooltip contentStyle={{ background: 'var(--color-surface-solid)', border: '1px solid var(--color-border)', borderRadius: 8, color: 'var(--color-text-primary)' }} />
                      <Bar dataKey="value" fill="var(--color-accent)" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* Evidence */}
            {result.evidence.length > 0 && (
              <div className="card">
                <div className="card-header">
                  <span className="card-title">Supporting Evidence</span>
                  <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-muted)' }}>
                    {result.evidence.length} reviews
                  </span>
                </div>
                <div className="card-body">
                  <div className="evidence-list">
                    {result.evidence.map((ev, i) => (
                      <div key={i} className="evidence-item">
                        <div className="evidence-meta">
                          {ev.sentiment && <SentimentBadge label={ev.sentiment} />}
                          {ev.domain && <Badge label={ev.domain} variant="muted" />}
                          {ev.similarity_score != null && (
                            <span className="evidence-score">sim: {ev.similarity_score.toFixed(3)}</span>
                          )}
                          <CopyButton text={ev.text} label="Copy" />
                        </div>
                        <p className="evidence-text">{ev.text}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Limitations */}
            {result.limitations.length > 0 && (
              <div className="alert alert--info">
                <AlertTriangle size={16} />
                <div>
                  <div style={{ fontWeight: 600, marginBottom: 'var(--space-1)' }}>Limitations</div>
                  {result.limitations.map((l, i) => <p key={i} style={{ margin: '2px 0' }}>{l}</p>)}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
