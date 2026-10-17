import { Link } from 'react-router-dom';
import { Home } from 'lucide-react';

export default function NotFound() {
  return (
    <div className="page">
      <div className="container">
        <div className="empty-state">
          <div style={{ fontSize: 64, fontWeight: 800, color: 'var(--color-accent)', lineHeight: 1 }}>404</div>
          <div className="empty-title" style={{ marginTop: 'var(--space-4)' }}>Page not found</div>
          <p className="empty-desc">The page you&apos;re looking for doesn&apos;t exist or has been moved.</p>
          <Link to="/" className="btn btn--primary" style={{ marginTop: 'var(--space-6)' }}>
            <Home size={16} /> Go home
          </Link>
        </div>
      </div>
    </div>
  );
}
