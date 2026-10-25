import { useEffect } from 'react';
import { NavLink } from 'react-router-dom';
import { Home, BarChart2, Search, BookOpen, Activity, MessageSquare, PieChart } from 'lucide-react';

const ICONS: Record<string, React.ReactNode> = {
  '/': <Home size={18} />,
  '/analyze': <Activity size={18} />,
  '/dashboard': <BarChart2 size={18} />,
  '/analytics': <PieChart size={18} />,
  '/reviews': <BookOpen size={18} />,
  '/models': <Search size={18} />,
  '/assistant': <MessageSquare size={18} />,
};

interface MobileNavProps {
  id: string;
  open: boolean;
  links: { to: string; label: string; exact?: boolean }[];
  onClose: () => void;
}

export function MobileNav({ id, open, links, onClose }: MobileNavProps) {
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [open, onClose]);

  return (
    <nav
      id={id}
      className={`mobile-nav${open ? ' open' : ''}`}
      aria-label="Mobile navigation"
      aria-hidden={!open}
    >
      {links.map(link => (
        <NavLink
          key={link.to}
          to={link.to}
          end={link.exact}
          className={({ isActive }) => `mobile-nav-link${isActive ? ' active' : ''}`}
          onClick={onClose}
        >
          {ICONS[link.to]}
          {link.label}
        </NavLink>
      ))}
    </nav>
  );
}
