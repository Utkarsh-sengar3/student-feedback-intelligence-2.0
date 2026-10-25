import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Header } from './components/layout/Header';
import { Footer } from './components/layout/Footer';
import { ScrollProgress } from './components/ui/ScrollProgress';
import { ScrollToTop } from './components/ui/ScrollToTop';
import { PrivacyBanner } from './components/ui/PrivacyBanner';
import Home from './pages/Home';
import Analyze from './pages/Analyze';
import Dashboard from './pages/Dashboard';
import Analytics from './pages/Analytics';
import Reviews from './pages/Reviews';
import Models from './pages/Models';
import Assistant from './pages/Assistant';
import NotFound from './pages/NotFound';
import { useTheme } from './hooks/useTheme';

export default function App() {
  const { theme, toggle } = useTheme();

  return (
    <BrowserRouter>
      <ScrollProgress />
      <Header theme={theme} onToggleTheme={toggle} />
      <div className="main-content">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/analyze" element={<Analyze />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/reviews" element={<Reviews />} />
          <Route path="/models" element={<Models />} />
          <Route path="/assistant" element={<Assistant />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </div>
      <Footer />
      <ScrollToTop />
      <PrivacyBanner />
    </BrowserRouter>
  );
}
