
import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { APP_NAME, TAGLINE, PICSUM_SEEDS, ROUTES } from '../constants';
import { HippoIcon } from '../components/common/Icons';

// This screen is now a secondary splash, potentially used after the FrontPage
// if a timed introduction is desired before the dashboard.
// Currently, it's not directly routed from App.tsx's initial load.
export const WelcomeScreen: React.FC = () => {
  const navigate = useNavigate();

  useEffect(() => {
    const timer = setTimeout(() => {
      navigate(ROUTES.DASHBOARD); // Navigate to Dashboard
    }, 4000); // Navigate after 4 seconds
    return () => clearTimeout(timer);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [navigate]);

  // USER INSTRUCTION: Create 'welcome-screen-bg.png' in your './public/images/' folder.
  return (
    <div
      className="min-h-screen flex flex-col items-center justify-center p-8 text-center bg-cover bg-center"
      style={{ backgroundImage: `url(./public/images/welcome-screen-bg.png)` }} // Updated path
    >
      <div className="absolute inset-0 bg-jungle-green opacity-70"></div>
      <div className="relative z-10 animate-fadeInSplash">
        <HippoIcon className="w-24 h-24 md:w-32 md:h-32 text-accent-teal mx-auto mb-6" />
        <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold text-white mb-4">
          Welcome to <span className="text-accent-teal">{APP_NAME}</span>: Their Stories Begin Here.
        </h1>
        <p className="text-xl sm:text-2xl text-sky-blue mb-2">
          Discover Moodeng & Piko's Adventures, Told by AI & Human Storytellers.
        </p>
        <p className="text-lg text-gray-300 mb-8">{TAGLINE}</p>
        <div className="w-20 h-1 bg-accent-teal mx-auto rounded-full animate-pulse"></div>
        <p className="mt-8 text-gray-400 text-sm">Unveiling their stories...</p>
      </div>
      <style>{`
        @keyframes fadeInSplash {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fadeInSplash { animation: fadeInSplash 1.5s ease-out forwards; }
      `}</style>
    </div>
  );
};