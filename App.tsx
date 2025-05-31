import React from 'react';
import { HashRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import { FrontPage } from './screens/FrontPage'; // Import new FrontPage
import { WelcomeScreen } from './screens/WelcomeScreen'; // Keep for potential future use, but not primary
import { DashboardScreen } from './screens/DashboardScreen';
import { JournalScreen } from './screens/JournalScreen';
import { ObserverScreen } from './screens/ObserverScreen';
import { InsightsScreen } from './screens/InsightsScreen';
import { CommunityScreen } from './screens/CommunityScreen';
import { HippoIcon, BookOpenIcon, EyeIcon, LightBulbIcon, UsersIcon } from './components/common/Icons'; 
import { APP_NAME, TAGLINE, ROUTES } from './constants';

const NAV_ITEMS = [
  // { path: ROUTES.WELCOME, label: 'Home', icon: HomeIcon }, // Home is now the FrontPage, not in main nav
  { path: ROUTES.DASHBOARD, label: 'Dashboard', icon: HippoIcon },
  { path: ROUTES.JOURNAL, label: 'Journal', icon: BookOpenIcon },
  { path: ROUTES.OBSERVER, label: 'Observer', icon: EyeIcon },
  { path: ROUTES.INSIGHTS, label: 'Insights', icon: LightBulbIcon },
  { path: ROUTES.COMMUNITY, label: 'Join Our Pod', icon: UsersIcon },
];

const App: React.FC = () => {
  return (
    <HashRouter>
      <AppContent />
    </HashRouter>
  );
};

const AppContent: React.FC = () => {
  const location = useLocation();
  // Check if the current path is the FrontPage
  const isFrontPage = location.pathname === ROUTES.WELCOME;

  if (isFrontPage) {
    return (
      <Routes>
        <Route path={ROUTES.WELCOME} element={<FrontPage />} />
      </Routes>
    );
  }

  // Render MainLayout for all other pages
  return <MainLayout />;
};

const MainLayout: React.FC = () => {
  const location = useLocation();

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-jungle-green via-earthy-brown to-watery-blue text-gray-100">
      <nav className="bg-jungle-green/80 backdrop-blur-md shadow-lg p-4 sticky top-0 z-50">
        <div className="container mx-auto flex justify-between items-center">
          <Link to={ROUTES.DASHBOARD} className="flex items-center space-x-2 text-2xl font-bold text-accent-teal hover:text-sky-blue transition-colors">
            <HippoIcon className="w-10 h-10" />
            <span>{APP_NAME}</span>
          </Link>
          <ul className="flex space-x-2 sm:space-x-4">
            {NAV_ITEMS.map((item) => (
              <li key={item.path}>
                <Link
                  to={item.path}
                  className={`px-2 py-2 sm:px-3 text-sm sm:text-base font-medium rounded-md flex items-center space-x-1 transition-all
                    ${location.pathname === item.path ? 'bg-accent-teal text-jungle-green' : 'text-gray-200 hover:bg-earthy-brown hover:text-white'}`}
                >
                  <item.icon className="w-4 h-4 sm:w-5 sm:h-5" />
                  <span className="hidden sm:inline">{item.label}</span>
                </Link>
              </li>
            ))}
          </ul>
        </div>
      </nav>

      <main className="flex-grow container mx-auto p-4 sm:p-6 lg:p-8">
        <Routes>
          {/* FrontPage is handled by AppContent, redirect to dashboard if needed or show dashboard */}
          <Route path={ROUTES.DASHBOARD} element={<DashboardScreen />} />
          <Route path={ROUTES.JOURNAL} element={<JournalScreen />} />
          <Route path={ROUTES.OBSERVER} element={<ObserverScreen />} />
          <Route path={ROUTES.INSIGHTS} element={<InsightsScreen />} />
          <Route path={ROUTES.COMMUNITY} element={<CommunityScreen />} />
          {/* Default to dashboard if no other match inside MainLayout */}
          <Route path="*" element={<DashboardScreen />} /> 
        </Routes>
      </main>

      <footer className="bg-jungle-green/60 text-center p-4 text-sm border-t-2 border-accent-teal/50">
        <p className="text-gray-200">&copy; {new Date().getFullYear()} {TAGLINE}</p>
        <p className="text-xs mt-1 text-gray-400">Stories woven from AI insights & human creativity.</p>
      </footer>
    </div>
  );
};

export default App;