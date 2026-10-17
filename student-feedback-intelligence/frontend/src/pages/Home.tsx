import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Activity, BarChart2, BookOpen, Search, MessageSquare, Zap,
  ArrowRight, CheckCircle
} from 'lucide-react';
import { api } from '../api/client';
import type { HealthResponse } from '../api/types';

const FEATURES = [
  {
    icon: <Activity size={22} />,
    title: 'Sentiment Analysis',
    desc: 'Fine-tuned DistilBERT classifier trained on 6,500+ student reviews achieves 81.6% accuracy across Positive, Negative, and Neutral classes.',
    path: '/analyze',
  },
  {
    icon: <BarChart2 size={22} />,
    title: 'Category Classification',
    desc: 'Multilabel TF-IDF + LinearSVC classifier identifies 24 aspect categories (Course, Staff, University) per review with 71.6% micro-F1.',
    path: '/analyze',
  },
  {
    icon: <Search size={22} />,
    title: 'Semantic Search',
    desc: 'all-MiniLM-L6-v2 sentence embeddings enable semantic similarity search over 6,500 indexed reviews with MRR of 95.8%.',
    path: '/reviews',
  },
  {
    icon: <BookOpen size={22} />,
    title: 'Evidence Retrieval',
    desc: 'Every analysis is grounded in real student reviews. Retrieve the most relevant evidence to support any finding.',
    path: '/dashboard',
  },
  {
    icon: <MessageSquare size={22} />,
    title: 'RAG Assistant',
    desc: 'Ask natural-language questions about student feedback. Get deterministic, evidence-grounded answers without hallucination.',
    path: '/assistant',
  },
  {
    icon: <Zap size={22} />,
    title: 'Actionable Insights',
    desc: '85 concepts analysed across 16 pipeline phases. Priority-ranked recommendations backed by dataset evidence.',
    path: '/dashboard',
  },
];

const STATS = [
  { label: 'Reviews Indexed', value: '6,500+', sub: 'EduRABSA dataset' },
  { label: 'Aspect Categories', value: '24', sub: 'Course, Staff, University' },
  { label: 'Sentiment Accuracy', value: '81.6%', sub: 'DistilBERT fine-tuned' },
  { label: 'Retrieval MRR', value: '95.8%', sub: 'all-MiniLM-L6-v2' },
];

export default function Home() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthStatus, setHealthStatus] = useState<'loading' | 'online' | 'offline'>('loading');

  useEffect(() => {
    api.health()
      .then(r => { setHealth(r); setHealthStatus('online'); })
      .catch(() => setHealthStatus('offline'));
  }, []);

  return (
    <div className="page">
      {/* HERO */}
      <section className="hero">
        <div className="container">
          <div className="hero-eyebrow">
            <CheckCircle size={12} />
            Research-Grade AI Platform
          </div>
          <h1 className="hero-title">
            Student Feedback{' '}
            <span className="accent">Intelligence</span>
          </h1>
          <p className="hero-description">
            AI-powered sentiment analysis, aspect classification, and semantic evidence retrieval
            for educational institutions — built on real student feedback data.
          </p>
          <div className="hero-actions">
            <Link to="/analyze" className="btn btn--primary btn--lg">
              Analyze Feedback <ArrowRight size={16} />
            </Link>
            <Link to="/dashboard" className="btn btn--ghost btn--lg">
              View Dashboard
            </Link>
          </div>
          <div className="hero-status">
            <span
              className={`status-dot status-dot--${healthStatus}`}
              aria-hidden="true"
            />
            {healthStatus === 'loading' && 'Connecting to API...'}
            {healthStatus === 'online' && `API online — ${health?.service || 'student-feedback-intelligence-api'} v${health?.version || '2.1.0'}`}
            {healthStatus === 'offline' && 'API offline — some features may be unavailable'}
          </div>
        </div>
      </section>

      {/* STATS */}
      <section style={{ padding: 'var(--space-8) 0' }}>
        <div className="container">
          <div className="stats-grid">
            {STATS.map(s => (
              <div key={s.label} className="card stat-card card--hoverable">
                <div className="stat-label">{s.label}</div>
                <div className="stat-value">{s.value}</div>
                <div className="stat-sub">{s.sub}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* FEATURES */}
      <section style={{ padding: 'var(--space-12) 0' }}>
        <div className="container">
          <div className="section-label">Capabilities</div>
          <h2 className="section-title">What the platform does</h2>
          <div className="features-grid">
            {FEATURES.map(f => (
              <Link key={f.title} to={f.path} style={{ textDecoration: 'none' }}>
                <div className="card feature-card card--hoverable">
                  <div className="feature-icon" aria-hidden="true">{f.icon}</div>
                  <div className="feature-title">{f.title}</div>
                  <p className="feature-desc">{f.desc}</p>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* PIPELINE */}
      <section style={{ padding: 'var(--space-8) 0 var(--space-12)' }}>
        <div className="container container--narrow">
          <div className="card card-body">
            <div className="section-label">Architecture</div>
            <h2 className="section-title" style={{ fontSize: 'var(--font-size-2xl)' }}>18-Phase Research Pipeline</h2>
            <p style={{ color: 'var(--color-text-secondary)', lineHeight: 'var(--line-height-relaxed)' }}>
              Built on the EduRABSA student review dataset, this platform implements an
              end-to-end 18-phase ML pipeline: from raw data audit and quality filtering
              through DistilBERT fine-tuning, multilabel category classification,
              concern prioritisation, semantic embedding, and evidence-grounded
              retrieval. All model metrics are real evaluation results from the test set.
            </p>
            <div style={{ display: 'flex', gap: 'var(--space-3)', flexWrap: 'wrap', marginTop: 'var(--space-4)' }}>
              <Link to="/models" className="btn btn--secondary">View Model Metrics</Link>
              <Link to="/assistant" className="btn btn--ghost">Try the Assistant</Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
