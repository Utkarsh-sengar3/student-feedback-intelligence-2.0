import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, X } from 'lucide-react';

const NAV_ITEMS = [
  { label: 'Home', path: '/', description: 'Platform overview and quick start' },
  { label: 'Analyze Feedback', path: '/analyze', description: 'Analyze student feedback with AI' },
  { label: 'Dashboard', path: '/dashboard', description: 'Insights and priority summary' },
  { label: 'Reviews Explorer', path: '/reviews', description: 'Browse 6,500+ indexed reviews' },
  { label: 'Model Performance', path: '/models', description: 'Sentiment, category, and retrieval metrics' },
  { label: 'AI Assistant', path: '/assistant', description: 'Ask questions about student feedback' },
];

interface SiteSearchProps {
  isOpen: boolean;
  onClose: () => void;
}

export function SiteSearch({ isOpen, onClose }: SiteSearchProps) {
  const [query, setQuery] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  const filtered = query.trim()
    ? NAV_ITEMS.filter(
        item =>
          item.label.toLowerCase().includes(query.toLowerCase()) ||
          item.description.toLowerCase().includes(query.toLowerCase())
      )
    : NAV_ITEMS;

  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 50);
      setQuery('');
    }
  }, [isOpen]);

  const handleKey = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    },
    [onClose]
  );

  useEffect(() => {
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [handleKey]);

  const go = (path: string) => {
    navigate(path);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="search-overlay" onClick={onClose} role="dialog" aria-modal="true" aria-label="Site search">
      <div className="search-modal" onClick={e => e.stopPropagation()}>
        <div className="search-modal-input-row">
          <Search size={18} style={{ color: 'var(--color-text-muted)', flexShrink: 0 }} />
          <input
            ref={inputRef}
            className="search-modal-input"
            placeholder="Search pages and features..."
            value={query}
            onChange={e => setQuery(e.target.value)}
            aria-label="Search"
          />
          <button className="btn btn--ghost btn--icon btn--sm" onClick={onClose} aria-label="Close search">
            <X size={16} />
          </button>
        </div>

        <div className="search-results-list" role="listbox">
          {filtered.length === 0 ? (
            <div className="search-empty">No results for &ldquo;{query}&rdquo;</div>
          ) : (
            filtered.map(item => (
              <div
                key={item.path}
                className="search-result-item"
                role="option"
                tabIndex={0}
                aria-selected={false}
                onClick={() => go(item.path)}
                onKeyDown={e => e.key === 'Enter' && go(item.path)}
              >
                <div>
                  <div style={{ fontWeight: 600, fontSize: 'var(--font-size-sm)' }}>{item.label}</div>
                  <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-muted)' }}>
                    {item.description}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>

        <div className="search-modal-footer">
          <span><kbd>↑↓</kbd> navigate</span>
          <span><kbd>Enter</kbd> open</span>
          <span><kbd>Esc</kbd> close</span>
        </div>
      </div>
    </div>
  );
}
