import { useState, useEffect } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { Moon, Sun, Search } from 'lucide-react';
import { MobileNav } from './MobileNav';
import { SiteSearch } from '../ui/SiteSearch';

const NAV_LINKS = [
  { to: '/', label: 'Home', exact: true },
  { to: '/analyze', label: 'Analyze' },
  { to: '/dashboard', label: 'Dashboard' },
  { to: '/reviews', label: 'Reviews' },
  { to: '/models', label: 'Models' },
  { to: '/assistant', label: 'Assistant' },
];

interface HeaderProps {
  theme: 'dark' | 'light';
  onToggleTheme: () => void;
}

export function Header({ theme, onToggleTheme }: HeaderProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    setMenuOpen(false);
  }, [location.pathname]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setSearchOpen(true);
      }
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, []);

  useEffect(() => {
    document.body.style.overflow = menuOpen ? 'hidden' : '';
    return () => { document.body.style.overflow = ''; };
  }, [menuOpen]);

  return (
    <>
      <header className="header" role="banner">
        <div className="header-inner">
          <NavLink to="/" className="header-brand" aria-label="Student Feedback Intelligence home">
            <div className="header-brand-icon" aria-hidden="true">SFI</div>
            <div>
              <div className="header-brand-name">FeedbackIntel</div>
              <div className="header-brand-abbr">AI-Based Student Feedback Intelligence</div>
            </div>
          </NavLink>

          <nav className="header-nav" role="navigation" aria-label="Main navigation">
            {NAV_LINKS.map(link => (
              <NavLink
                key={link.to}
                to={link.to}
                end={link.exact}
                className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
              >
                {link.label}
              </NavLink>
            ))}
          </nav>

          <div className="header-actions">
            <button
              className="btn btn--ghost btn--icon site-search-trigger"
              onClick={() => setSearchOpen(true)}
              aria-label="Search (Ctrl+K)"
              title="Search (Ctrl+K)"
            >
              <Search size={16} />
            </button>
            <button
              className="theme-toggle"
              onClick={onToggleTheme}
              aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
              title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            >
              {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
            </button>
            <button
              className={`hamburger${menuOpen ? ' open' : ''}`}
              onClick={() => setMenuOpen(o => !o)}
              aria-label={menuOpen ? 'Close menu' : 'Open menu'}
              aria-expanded={menuOpen}
              aria-controls="mobile-nav"
            >
              <span /><span /><span />
            </button>
          </div>
        </div>
      </header>

      <MobileNav
        id="mobile-nav"
        open={menuOpen}
        links={NAV_LINKS}
        onClose={() => setMenuOpen(false)}
      />

      <SiteSearch isOpen={searchOpen} onClose={() => setSearchOpen(false)} />
    </>
  );
}
