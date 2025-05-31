
import React, { useState, useEffect } from 'react';
import { Section } from '../components/common/Section';
import { NavigationMenu } from '../components/NavigationMenu';
import { Card } from '../components/common/Card';
import { PICSUM_SEEDS } from '../constants';
import { PawPrintIcon }  from '../components/common/Icons';

const storySnippets = [
  "Story Teaser: Moodeng was just observed teaching Piko a new swimming trick by the reeds!",
  "From the AI Journal: Piko discovered a patch of delicious new plants today. What adventures will they lead to?",
  "AI Insight: A moment of quiet bonding between Moodeng and Piko was captured, inspiring tomorrow's tale.",
  "Latest Story Hint: The AI noticed Piko curiously watching the forest birds. A new friendship perhaps?",
  "Whispers from the Waterhole: Moodeng seems extra watchful today. What could be on her mind?"
];

export const DashboardScreen: React.FC = () => {
  const [dynamicSnippet, setDynamicSnippet] = useState<string>('');

  useEffect(() => {
    setDynamicSnippet(storySnippets[Math.floor(Math.random() * storySnippets.length)]);
  }, []);

  // USER INSTRUCTION: Create 'dashboard-bg.png' in your './public/images/' folder.
  return (
    <div
      className="min-h-full bg-no-repeat bg-cover bg-center"
      style={{ backgroundImage: `linear-gradient(rgba(16, 79, 85, 0.8), rgba(89, 69, 69, 0.8)), url(./public/images/dashboard-bg.png)` }} // Updated path
    >
      <Section
        title="HippoSphere Stories: Dive Into Their Adventures"
        subtitle="Explore daily tales, observe real moments, and learn about our pygmy hippos through AI-enhanced narratives."
        className="text-center"
        titleClassName="text-4xl font-bold text-accent-teal mb-4"
        subtitleClassName="text-xl text-sky-blue mb-10"
      >
        {dynamicSnippet && (
          <Card className="mb-10 max-w-2xl mx-auto bg-jungle-green/70 border border-accent-teal">
            <div className="flex items-center justify-center">
              <PawPrintIcon className="w-6 h-6 text-sunrise-orange mr-3 animate-pulse" />
              <p className="text-lg text-gray-200">{dynamicSnippet}</p>
            </div>
          </Card>
        )}
        <NavigationMenu />
      </Section>
    </div>
  );
};