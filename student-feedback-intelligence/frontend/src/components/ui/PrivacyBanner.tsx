import { useState } from 'react';

export function PrivacyBanner() {
  const [dismissed, setDismissed] = useState(() => {
    try { return localStorage.getItem('sfi-privacy') === 'dismissed'; }
    catch { return false; }
  });

  if (dismissed) return null;

  const handleDismiss = () => {
    try { localStorage.setItem('sfi-privacy', 'dismissed'); } catch { /* ignore */ }
    setDismissed(true);
  };

  return (
    <div className="privacy-banner" role="region" aria-label="Privacy notice">
      <p className="privacy-banner-text">
        This application uses <strong>localStorage</strong> only to remember your theme
        preference. No tracking cookies or analytics scripts are used.
      </p>
      <div className="privacy-banner-actions">
        <button className="btn btn--primary btn--sm" onClick={handleDismiss}>
          Got it
        </button>
      </div>
    </div>
  );
}
