
import React from 'react';
import { Link } from 'react-router-dom';
import { Card } from './common/Card';
import { ROUTES, PICSUM_SEEDS, PLACEHOLDER_IMAGE_DIMENSIONS } from '../constants';
import { BookOpenIcon, EyeIcon, LightBulbIcon, UsersIcon } from './common/Icons';

interface NavItem {
  path: string;
  label: string;
  description: string;
  icon: React.FC<React.SVGProps<SVGSVGElement>>;
  imageSrc: string; // Changed from imageSeed to imageSrc
}

// USER INSTRUCTION: Create these images in your './public/images/' folder:
// nav-journal.png, nav-observer.png, nav-insights.png, nav-community.png
const NAV_ITEMS: NavItem[] = [
  {
    path: ROUTES.JOURNAL,
    label: "Moodeng & Piko's Journal",
    description: "Tales from the Waterhole: Daily adventures and stories.",
    icon: BookOpenIcon,
    imageSrc: "./public/images/nav-journal.png" // Updated path
  },
  {
    path: ROUTES.OBSERVER,
    label: "Live Habitat Observer",
    description: "The Lookout Point: Real-time hippo behaviors and activity.",
    icon: EyeIcon,
    imageSrc: "./public/images/nav-observer.png" // Updated path
  },
  {
    path: ROUTES.INSIGHTS,
    label: "Hippo Insights & Analytics",
    description: "The Researcher's Den: Learn about hippos and conservation.",
    icon: LightBulbIcon,
    imageSrc: "./public/images/nav-insights.png" // Updated path
  },
  {
    path: ROUTES.COMMUNITY,
    label: "Join Our Pod",
    description: "The Community Spring: Connect, support, and learn more.",
    icon: UsersIcon,
    imageSrc: "./public/images/nav-community.png" // Updated path
  },
];

export const NavigationMenu: React.FC = () => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-6 lg:gap-8">
      {NAV_ITEMS.map((item) => (
        <Link to={item.path} key={item.path} className="block group">
          <Card className="h-full flex flex-col overflow-hidden group-hover:shadow-2xl group-hover:border-accent-teal border-2 border-transparent transition-all duration-300" hoverEffect>
            <img
              src={item.imageSrc} // Use the new imageSrc
              alt={item.label}
              className="w-full h-48 object-cover"
            />
            <div className="p-6 flex-grow">
              <div className="flex items-center mb-3">
                <item.icon className="w-8 h-8 text-accent-teal mr-3" />
                <h3 className="text-xl font-semibold text-sky-blue group-hover:text-accent-teal transition-colors">{item.label}</h3>
              </div>
              <p className="text-gray-300 text-sm">{item.description}</p>
            </div>
             <div className="p-4 bg-jungle-green/30 text-right">
                <span className="text-accent-teal group-hover:underline">Explore &rarr;</span>
            </div>
          </Card>
        </Link>
      ))}
    </div>
  );
};