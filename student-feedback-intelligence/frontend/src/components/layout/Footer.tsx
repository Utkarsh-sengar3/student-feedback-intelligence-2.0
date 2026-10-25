import { NavLink } from 'react-router-dom';

export function Footer() {
  return (
    <footer className="footer" role="contentinfo">
      <div className="container">
        <div className="footer-inner">
          <div>
            <div className="footer-brand">Student Feedback Intelligence</div>
            <div className="footer-tech">
              FastAPI · React · Recharts · sentence-transformers · DistilBERT
            </div>
          </div>
          <nav className="footer-links" aria-label="Footer navigation">
            <NavLink to="/" className="footer-link">Home</NavLink>
            <NavLink to="/analyze" className="footer-link">Analyze</NavLink>
            <NavLink to="/dashboard" className="footer-link">Dashboard</NavLink>
            <NavLink to="/analytics" className="footer-link">EDA & Analytics</NavLink>
            <NavLink to="/reviews" className="footer-link">Reviews</NavLink>
            <NavLink to="/models" className="footer-link">Models</NavLink>
            <NavLink to="/assistant" className="footer-link">Assistant</NavLink>
          </nav>
          <div className="footer-tech" style={{ textAlign: 'right' }}>
            AI-Based Student Feedback Intelligence v2.1.0
          </div>
        </div>
      </div>
    </footer>
  );
}
