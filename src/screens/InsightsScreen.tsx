
import React, { useState, useEffect } from 'react';
import { Section } from '../components/common/Section';
import { Card } from '../components/common/Card';
import { Button } from '../components/common/Button'; // Added missing import
import { HippoProfile, BehavioralTrendDataPoint, PygmyHippoFact } from '../types';
import { getMockHippoProfiles, getMockBehavioralTrends, getMockPygmyHippoFacts } from '../services/geminiService';
import { PICSUM_SEEDS, PLACEHOLDER_IMAGE_DIMENSIONS } from '../constants';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend } from 'recharts';
import { LeafIcon, ExternalLinkIcon, DataWaveIcon } from '../components/common/Icons';

const ProfileCard: React.FC<{ hippo: HippoProfile }> = ({ hippo }) => (
  <Card className="bg-earthy-brown/40 flex flex-col md:flex-row items-center md:items-start gap-6">
    <img 
      src={hippo.imageUrl} 
      alt={hippo.name} 
      className="w-40 h-40 md:w-48 md:h-48 object-cover rounded-full md:rounded-lg shadow-lg"
    />
    <div className="text-center md:text-left">
      <h3 className="text-2xl font-semibold text-accent-teal mb-1">{hippo.name}</h3>
      <p className="text-sm text-sky-blue mb-1">{hippo.species}</p>
      <p className="text-sm text-gray-300 mb-1">Estimated Age: {hippo.estimatedAge}</p>
      <p className="text-gray-300 text-sm leading-relaxed">{hippo.originStory}</p>
    </div>
  </Card>
);

export const InsightsScreen: React.FC = () => {
  const [hippos, setHippos] = useState<HippoProfile[]>([]);
  const [behavioralTrends, setBehavioralTrends] = useState<BehavioralTrendDataPoint[]>([]);
  const [pygmyHippoFacts, setPygmyHippoFacts] = useState<PygmyHippoFact[]>([]);

  useEffect(() => {
    setHippos(getMockHippoProfiles());
    setBehavioralTrends(getMockBehavioralTrends());
    setPygmyHippoFacts(getMockPygmyHippoFacts());
  }, []);

  return (
    <div>
      <Section
        title="The Researcher's Den"
        subtitle="Dive deep into the world of Moodeng, Piko, and pygmy hippos. Explore data, facts, and conservation efforts."
      >
        {/* Meet Moodeng & Piko */}
        <Section title="Meet Moodeng & Piko" titleClassName="text-2xl font-semibold text-accent-teal mb-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            {hippos.map(hippo => <ProfileCard key={hippo.id} hippo={hippo} />)}
          </div>
        </Section>

        {/* Behavioral Trends */}
        <Section title="Behavioral Trends Over Time" titleClassName="text-2xl font-semibold text-accent-teal mb-4">
          <Card className="bg-jungle-green/40 p-4 md:p-6">
            <h4 className="text-xl font-semibold text-sky-blue mb-4">Active Hours (Simulated Weekly Data)</h4>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={behavioralTrends}>
                <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.3} />
                <XAxis dataKey="name" stroke="#94A3B8" />
                <YAxis stroke="#94A3B8" label={{ value: 'Avg. Active Hours/Day', angle: -90, position: 'insideLeft', fill: '#94A3B8' }} />
                <RechartsTooltip 
                  contentStyle={{ backgroundColor: 'rgba(40, 42, 54, 0.8)', border: '1px solid #44475A', borderRadius: '0.5rem' }}
                  labelStyle={{ color: '#F8F8F2' }}
                  itemStyle={{ color: '#BD93F9' }}
                />
                <Legend />
                <Line type="monotone" dataKey="MoodengActiveHours" name="Moodeng" stroke="#2DD4BF" strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                <Line type="monotone" dataKey="PikoActiveHours" name="Piko" stroke="#F59E0B" strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 6 }} />
              </LineChart>
            </ResponsiveContainer>
            <p className="text-xs text-gray-400 mt-2 text-center">Heatmaps of enclosure usage would also be displayed here (conceptual).</p>
          </Card>
        </Section>

        {/* Pygmy Hippo Facts */}
        <Section title="Pygmy Hippo Facts" titleClassName="text-2xl font-semibold text-accent-teal mb-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {pygmyHippoFacts.map(fact => (
              <Card key={fact.id} className="bg-earthy-brown/40 flex flex-col">
                {fact.imageUrl && <img src={fact.imageUrl} alt={fact.title} className="w-full h-40 object-cover rounded-t-lg mb-4" />}
                <div className="p-1 flex-grow">
                    <div className="flex items-center mb-2">
                        {fact.icon ? <fact.icon className="w-6 h-6 text-accent-teal mr-2" /> : <LeafIcon className="w-6 h-6 text-accent-teal mr-2" />}
                        <h4 className="text-lg font-semibold text-sky-blue">{fact.title}</h4>
                    </div>
                    <p className="text-gray-300 text-sm">{fact.description}</p>
                </div>
              </Card>
            ))}
          </div>
          <Card className="mt-6 bg-jungle-green/40">
            <h4 className="text-lg font-semibold text-sky-blue mb-2">Conservation Status: <span className="text-sunrise-orange">Endangered</span> (IUCN Red List)</h4>
            <p className="text-gray-300 text-sm">Pygmy hippos face threats from habitat loss (deforestation, agriculture) and poaching. Conservation efforts focus on habitat protection, anti-poaching patrols, and captive breeding programs.</p>
            <Button 
              variant="outline" 
              size="sm" 
              className="mt-4" 
              onClick={() => window.open('https://www.iucnredlist.org/species/10032/18567171', '_blank')}
              rightIcon={<ExternalLinkIcon className="w-4 h-4" />}
            >
              Learn More on IUCN
            </Button>
          </Card>
        </Section>

        {/* AI & Conservation Tech */}
        <Section title="AI & Conservation Technology" titleClassName="text-2xl font-semibold text-accent-teal mb-4">
          <Card className="bg-jungle-green/40">
            <div className="flex items-start gap-4">
                <DataWaveIcon className="w-12 h-12 text-accent-teal flex-shrink-0 mt-1"/>
                <div>
                    <h4 className="text-xl font-semibold text-sky-blue mb-2">How AI Helps Us Understand & Conserve</h4>
                    <p className="text-gray-300 mb-3">
                        Artificial Intelligence, like the systems used in HippoSphere AI, plays a crucial role in modern animal conservation. By analyzing video footage, AI can:
                    </p>
                    <ul className="list-disc list-inside text-gray-300 space-y-1 mb-3">
                        <li>Automatically detect and classify animal behaviors (e.g., feeding, resting, social interactions).</li>
                        <li>Monitor activity levels and habitat usage patterns over long periods.</li>
                        <li>Identify individual animals based on unique markings (not currently implemented for hippos here, but a common AI use).</li>
                        <li>Flag unusual behaviors that might indicate stress or illness, enabling early intervention.</li>
                        <li>Provide quantitative data to support research and conservation management decisions.</li>
                    </ul>
                    <p className="text-gray-300">
                        Technologies like pose estimation (e.g., DeepLabCut), object detection (e.g., YOLO), and advanced language models (like Gemini API for generating narratives) help transform raw data into meaningful insights and engaging educational content.
                    </p>
                    <Button 
                        variant="secondary" 
                        size="sm" 
                        className="mt-4"
                        onClick={() => alert("Detailed methodology document would be linked here.")}
                    >
                        Project Methodology (Conceptual)
                    </Button>
                </div>
            </div>
          </Card>
        </Section>
      </Section>
    </div>
  );
};
