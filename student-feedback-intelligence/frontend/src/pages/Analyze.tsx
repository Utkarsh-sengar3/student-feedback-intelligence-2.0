import { useState, useRef } from 'react';
import { Send, Activity } from 'lucide-react';
import { api } from '../api/client';
import type { FeedbackResponse } from '../api/types';
import { SentimentBadge, SeverityBadge, PriorityBadge, Badge } from '../components/ui/Badge';
import { Spinner, SkeletonCard } from '../components/ui/Spinner';
import { CopyButton } from '../components/ui/CopyButton';

const EXAMPLES = [
  'The professor explains concepts very well and is always available during office hours.',
  'The Wi-Fi on campus keeps disconnecting during lectures. It is very frustrating.',
  'The midterm exam was extremely difficult and graded unfairly.',
  'Campus library facilities are excellent but the cafeteria food quality has declined.',
  'There is too much workload and assignments are piling up with unrealistic deadlines.',
];

export default function Analyze() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<FeedbackResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const MAX_CHARS = 5000;
  const MIN_CHARS = 3;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (text.trim().length < MIN_CHARS) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.analyze({ text: text.trim() });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleExample = (example: string) => {
    setText(example);
    textareaRef.current?.focus();
  };

  return (
    <div className="page">
      <div className="container container--narrow">
        <div className="page-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', marginBottom: 'var(--space-2)' }}>
            <Activity size={28} style={{ color: 'var(--color-accent)' }} />
            <h1 className="page-title" style={{ margin: 0 }}>Feedback Analysis</h1>
          </div>
          <p className="page-description">
            Submit student feedback to receive sentiment classification, aspect detection,
            severity assessment, and evidence-grounded recommendations.
          </p>
        </div>

        <div className="card">
          <div className="card-body">
            <form onSubmit={handleSubmit}>
              <div className="form-group">
                <label htmlFor="feedback-text" className="form-label">
                  Student feedback text
                </label>
                <textarea
                  id="feedback-text"
                  ref={textareaRef}
                  className="form-textarea"
                  style={{ minHeight: '140px' }}
                  placeholder="Enter student feedback (e.g., 'The professor explains concepts clearly but the exams are too difficult.')..."
                  value={text}
                  onChange={e => setText(e.target.value)}
                  maxLength={MAX_CHARS}
                  aria-describedby="char-count-hint"
                  required
                />
                <div className="char-count" id="char-count-hint" aria-live="polite">
                  {text.length} / {MAX_CHARS}
                </div>
              </div>

              <div className="form-group">
                <span className="form-label">Example inputs</span>
                <div className="examples">
                  {EXAMPLES.map((ex, i) => (
                    <button
                      key={i}
                      type="button"
                      className="example-btn"
                      onClick={() => handleExample(ex)}
                      title={ex}
                    >
                      {ex.length > 55 ? ex.slice(0, 52) + '...' : ex}
                    </button>
                  ))}
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 'var(--space-4)' }}>
                <button
                  type="submit"
                  className="btn btn--primary"
                  disabled={loading || text.trim().length < MIN_CHARS}
                >
                  {loading ? <Spinner size="sm" /> : <Send size={16} />}
                  {loading ? 'Analyzing...' : 'Analyze'}
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
            <Activity size={16} />
            <span>{error}</span>
          </div>
        )}

        {result && !loading && (
          <div className="card" style={{ marginTop: 'var(--space-6)' }}>
            <div className="card-header">
              <Activity size={18} style={{ color: 'var(--color-accent)' }} />
              <span className="card-title">Analysis Result</span>
              <CopyButton
                text={JSON.stringify(result, null, 2)}
                label="Copy JSON"
              />
            </div>
            <div className="card-body result-section">
              {/* Sentiment */}
              <div className="result-row">
                <div className="result-label">Sentiment</div>
                <div className="result-value">
                  {result.sentiment ? (
                    <SentimentBadge label={result.sentiment} />
                  ) : (
                    <span style={{ color: 'var(--color-text-muted)' }}>Unavailable</span>
                  )}
                  {result.sentiment_confidence != null && (
                    <div style={{ marginTop: 'var(--space-2)', maxWidth: 220 }}>
                      <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-muted)', marginBottom: 2 }}>
                        Confidence: {(result.sentiment_confidence * 100).toFixed(1)}%
                      </div>
                      <div className="confidence-bar">
                        <div
                          className="confidence-fill"
                          style={{ width: `${result.sentiment_confidence * 100}%` }}
                          role="progressbar"
                          aria-valuenow={Math.round(result.sentiment_confidence * 100)}
                          aria-valuemin={0}
                          aria-valuemax={100}
                        />
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Category */}
              {(result.categories.length > 0 || result.category) && (
                <div className="result-row">
                  <div className="result-label">Categories</div>
                  <div className="result-value">
                    <div className="tags">
                      {(result.categories.length > 0 ? result.categories : result.category ? [result.category] : []).map(cat => (
                        <Badge key={cat} label={cat} variant="accent" />
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Aspects */}
              {result.aspects.length > 0 && (
                <div className="result-row">
                  <div className="result-label">Aspects</div>
                  <div className="result-value">
                    <div className="tags">
                      {result.aspects.map(a => (
                        <Badge key={a} label={a} variant="muted" />
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Severity */}
              {result.severity && (
                <div className="result-row">
                  <div className="result-label">Severity</div>
                  <div className="result-value">
                    <SeverityBadge label={result.severity} />
                  </div>
                </div>
              )}

              {/* Priority */}
              {result.priority_level && (
                <div className="result-row">
                  <div className="result-label">Priority</div>
                  <div className="result-value">
                    <PriorityBadge label={result.priority_level} />
                    {result.priority_score != null && (
                      <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-muted)', marginLeft: 8 }}>
                        score: {result.priority_score.toFixed(2)}
                      </span>
                    )}
                  </div>
                </div>
              )}

              {/* Recommendation */}
              {result.recommendation && (
                <div className="result-row" style={{ alignItems: 'flex-start' }}>
                  <div className="result-label">Recommendation</div>
                  <div className="result-value">
                    <div className="recommendation-box">
                      {result.recommendation}
                      <div className="recommendation-actions">
                        <CopyButton text={result.recommendation} label="Copy" />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Evidence */}
              {result.evidence.length > 0 && (
                <div>
                  <div className="section-label" style={{ marginTop: 'var(--space-2)' }}>Supporting Evidence</div>
                  <div className="evidence-list">
                    {result.evidence.slice(0, 5).map((ev, i) => (
                      <div key={i} className="evidence-item">
                        <div className="evidence-meta">
                          {ev.sentiment && <SentimentBadge label={ev.sentiment} />}
                          {ev.domain && <Badge label={ev.domain} variant="muted" />}
                          {ev.similarity_score != null && (
                            <span className="evidence-score">
                              sim: {ev.similarity_score.toFixed(3)}
                            </span>
                          )}
                          <CopyButton text={ev.text} label="Copy" />
                        </div>
                        <p className="evidence-text">{ev.text}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
