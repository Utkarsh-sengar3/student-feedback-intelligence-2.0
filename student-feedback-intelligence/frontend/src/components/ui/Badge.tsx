interface BadgeProps {
  label: string;
  variant?: string;
  className?: string;
}

function getSentimentVariant(label: string): string {
  const l = label.toLowerCase();
  if (l === 'positive') return 'positive';
  if (l === 'negative') return 'negative';
  if (l === 'neutral') return 'neutral';
  return 'muted';
}

function getSeverityVariant(label: string): string {
  const l = label.toLowerCase();
  if (l === 'critical') return 'critical';
  if (l === 'high') return 'high';
  if (l === 'medium') return 'medium';
  if (l === 'low') return 'low';
  return 'muted';
}

function getPriorityVariant(label: string): string {
  const l = label.toLowerCase();
  if (l === 'critical') return 'critical';
  if (l === 'high') return 'high';
  if (l === 'medium') return 'medium';
  if (l === 'low') return 'low';
  if (l === 'informational') return 'info';
  return 'muted';
}

export function SentimentBadge({ label }: { label: string }) {
  return <span className={`badge badge--${getSentimentVariant(label)}`}>{label}</span>;
}

export function SeverityBadge({ label }: { label: string }) {
  return <span className={`badge badge--${getSeverityVariant(label)}`}>{label}</span>;
}

export function PriorityBadge({ label }: { label: string }) {
  return <span className={`badge badge--${getPriorityVariant(label)}`}>{label}</span>;
}

export function Badge({ label, variant = 'accent', className = '' }: BadgeProps) {
  return <span className={`badge badge--${variant} ${className}`}>{label}</span>;
}
