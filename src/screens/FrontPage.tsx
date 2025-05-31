
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { APP_NAME, ROUTES, TAGLINE } from '../constants'; // Removed PICSUM_SEEDS as it's not used here anymore
import { Button } from '../components/common/Button';
import { HippoIcon } from '../components/common/Icons';

export const FrontPage: React.FC = () => {
  const navigate = useNavigate();

  const handleExplore = () => {
    navigate(ROUTES.DASHBOARD);
  };

  // INSTRUCTION FOR USER:
  // To use your own background image for this page:
  // 1. Create a folder named 'public' at the root of your project if it doesn't exist.
  // 2. Inside 'public', create a folder named 'images'.
  // 3. Place your desired background image in 'public/images/' and name it 'frontpage-bg.png'.
  //    (e.g., the path should be ./public/images/frontpage-bg.png)
  // The image will then be displayed below. Recommended size: 1920x1080 pixels.

  return (
    <div
      className="min-h-screen flex flex-col items-center justify-center p-8 text-center bg-cover bg-center relative"
      style={{ backgroundImage: `url(./public/images/frontpage-bg.png)` }} // Updated to use explicitly relative local image path
    >
      <div className="absolute inset-0 bg-jungle-green opacity-75"></div>
      <div className="relative z-10 animate-fadeInBasic">
        <HippoIcon className="w-28 h-28 md:w-36 md:h-36 text-accent-teal mx-auto mb-6" />
        <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold text-white mb-4">
          {APP_NAME}: Moodeng and Piko's World
        </h1>
        <p className="text-xl sm:text-2xl text-sky-blue mb-10 max-w-2xl mx-auto">
          {TAGLINE} Step into their habitat, discover daily adventures, and connect with nature through the lens of AI.
        </p>
        <Button
          variant="primary"
          size="lg"
          onClick={handleExplore}
          className="shadow-xl transform hover:scale-105"
        >
          Explore Their World
        </Button>
      </div>
      <style>{`
        @keyframes fadeInBasic {
          from { opacity: 0; transform: translateY(15px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fadeInBasic { animation: fadeInBasic 1s ease-out forwards; }
      `}</style>
    </div>
  );
};