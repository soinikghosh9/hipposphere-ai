
import React, { useState, useEffect } from 'react';
import { Section } from '../components/common/Section';
import { Card } from '../components/common/Card';
import { ActivityLogEntry, ActivityDataPoint, BehaviorCategory, EmotionDataPoint } from '../types';
import { getMockActivityLog, getMockActivityData, getMockEmotionData } from '../services/geminiService';
import { EyeIcon, AlertTriangleIcon, PawPrintIcon, LeafIcon, WaterDropletIcon } from '../components/common/Icons';
import { PICSUM_SEEDS, BEHAVIOR_COLORS, PLACEHOLDER_IMAGE_DIMENSIONS } from '../constants';
import { ResponsiveContainer, PieChart, Pie, Cell, Legend, Tooltip as RechartsTooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts';

const behaviorIcons: Record<BehaviorCategory, React.FC<React.SVGProps<SVGSVGElement>>> = {
  [BehaviorCategory.RESTING]: PawPrintIcon,
  [BehaviorCategory.FEEDING]: LeafIcon,
  [BehaviorCategory.MOVING]: EyeIcon, // generic
  [BehaviorCategory.SOCIAL_INTERACTION]: PawPrintIcon, // generic
  [BehaviorCategory.PLAYING]: WaterDropletIcon,
  [BehaviorCategory.FORAGING]: LeafIcon,
};

export const ObserverScreen: React.FC = () => {
  const [activityLog, setActivityLog] = useState<ActivityLogEntry[]>([]);
  const [moodengActivity, setMoodengActivity] = useState<ActivityDataPoint[]>([]);
  const [pikoActivity, setPikoActivity] = useState<ActivityDataPoint[]>([]);
  const [moodengEmotions, setMoodengEmotions] = useState<EmotionDataPoint[]>([]);
  const [pikoEmotions, setPikoEmotions] = useState<EmotionDataPoint[]>([]);
  const [alerts, setAlerts] = useState<string[]>([]); // For zookeeper view

  useEffect(() => {
    setActivityLog(getMockActivityLog(10));
    setMoodengActivity(getMockActivityData('Moodeng'));
    setPikoActivity(getMockActivityData('Piko'));
    setMoodengEmotions(getMockEmotionData('Moodeng'));
    setPikoEmotions(getMockEmotionData('Piko'));

    const mockAlertTimer = setTimeout(() => {
        setAlerts(prevAlerts => [...prevAlerts, "Alert: Moodeng - Repetitive pacing detected for 30 mins (Simulated)."]);
    }, 5000);

    return () => clearTimeout(mockAlertTimer);
  }, []);

  const renderActivityIcon = (behavior: string) => {
    const category = Object.values(BehaviorCategory).find(cat => behavior.toLowerCase().includes(cat.toLowerCase()));
    const IconComponent = category ? behaviorIcons[category] : EyeIcon;
    return <IconComponent className="w-5 h-5 text-accent-teal mr-2" />;
  };

  const EmotionChart: React.FC<{ data: EmotionDataPoint[]; hippoName: string }> = ({ data, hippoName }) => (
    <Card className="bg-jungle-green/40 h-[350px] flex flex-col">
      <h4 className="text-lg font-semibold text-sky-blue mb-3 text-center">{hippoName}'s Inferred Emotional State</h4>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.2} />
          <XAxis type="number" domain={[0, 100]} stroke="#94A3B8" />
          <YAxis type="category" dataKey="name" stroke="#94A3B8" width={80} />
          <RechartsTooltip
            contentStyle={{ backgroundColor: 'rgba(40, 42, 54, 0.9)', border: '1px solid #44475A', borderRadius: '0.5rem' }}
            labelStyle={{ color: '#F8F8F2' }}
            formatter={(value: number) => [`${value}%`, 'Intensity']}
          />
          <Bar dataKey="value"  barSize={20} label={{ position: 'right', fill: '#e0e0e0', fontSize: 12, formatter: (value: number) => `${value}%`}}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </Card>
  );


  return (
    <div>
      <Section
        title="The Lookout Point"
        subtitle="Real-time (simulated) behavioral data from Moodeng and Piko's habitat."
      >
        {alerts.length > 0 && (
          <Card className="mb-6 bg-sunrise-orange/20 border-sunrise-orange border-2">
            <h3 className="text-xl font-semibold text-sunrise-orange mb-2 flex items-center">
              <AlertTriangleIcon className="w-6 h-6 mr-2"/> Important Alerts
            </h3>
            <ul className="list-disc list-inside text-orange-200">
              {alerts.map((alert, index) => <li key={index}>{alert}</li>)}
            </ul>
            <p className="text-xs text-orange-300 mt-2">Note: These alerts are for demonstration and require zookeeper verification.</p>
          </Card>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <Card className="lg:col-span-1 bg-jungle-green/40">
            <h3 className="text-xl font-semibold text-sky-blue mb-3">Live Video Feed Snippet</h3>
            {/* USER INSTRUCTION: Create 'observer-feed-placeholder.png' in your './public/images/' folder. */}
            <img
              src={`./public/images/observer-feed-placeholder.png`} // Updated path
              alt="Live habitat feed placeholder"
              className="w-full rounded-md shadow-lg aspect-video object-cover"
            />
            <p className="text-xs text-gray-400 mt-2">Showing most recent activity (simulated live feed).</p>
          </Card>

          <Card className="lg:col-span-2 bg-jungle-green/40 h-[450px] overflow-y-auto">
            <h3 className="text-xl font-semibold text-sky-blue mb-3">Activity Log</h3>
            <ul className="space-y-3">
              {activityLog.map(log => (
                <li key={log.id} className="flex items-center p-2 bg-earthy-brown/30 rounded-md text-sm">
                  {renderActivityIcon(log.behavior)}
                  <span className="text-gray-400 mr-2">{log.timestamp}</span>
                  <span className="font-semibold text-gray-200 mr-1">{log.hippoName}:</span>
                  <span className="text-gray-300">{log.behavior}</span>
                </li>
              ))}
            </ul>
          </Card>
        </div>

        <Section title="Current Activity Levels (Last Hour - Simulated)" titleClassName="text-2xl font-semibold text-accent-teal mb-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="bg-jungle-green/40">
              <h4 className="text-lg font-semibold text-sky-blue mb-3">Moodeng's Activity</h4>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie data={moodengActivity} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={100} labelLine={false} label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}>
                    {moodengActivity.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                  </Pie>
                  <RechartsTooltip formatter={(value: number, name: string) => [`${value}%`, name]} />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </Card>
            <Card className="bg-jungle-green/40">
              <h4 className="text-lg font-semibold text-sky-blue mb-3">Piko's Activity</h4>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie data={pikoActivity} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={100} labelLine={false} label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}>
                    {pikoActivity.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                  </Pie>
                  <RechartsTooltip formatter={(value: number, name: string) => [`${value}%`, name]}/>
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </Section>

        <Section title="Inferred Emotional State Indicators" titleClassName="text-2xl font-semibold text-accent-teal mb-4">
             <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <EmotionChart data={moodengEmotions} hippoName="Moodeng" />
                <EmotionChart data={pikoEmotions} hippoName="Piko" />
            </div>
            <p className="text-xs text-gray-400 text-center italic">Disclaimer: These are AI inferences based on behavioral patterns and not direct emotional readings. They are for illustrative purposes. Always consult with animal behavior experts for definitive assessments.</p>
        </Section>
      </Section>
    </div>
  );
};