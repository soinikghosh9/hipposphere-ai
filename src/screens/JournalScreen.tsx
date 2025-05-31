
import React, { useState, useEffect, useCallback } from 'react';
import { Section } from '../components/common/Section';
import { Card } from '../components/common/Card';
import { Button } from '../components/common/Button';
import { JournalEntry, HippoProfile } from '../types';
import { DEFAULT_ARTIST_NAME, PICSUM_SEEDS, PLACEHOLDER_IMAGE_DIMENSIONS } from '../constants';
import { getMockJournalEntries, getMockHippoProfiles, askGeminiMock } from '../services/geminiService';
import { ChevronDownIcon, ChevronUpIcon, LoadingSpinner } from '../components/common/Icons';

const JournalPost: React.FC<{ entry: JournalEntry }> = ({ entry }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <Card className="mb-8 bg-earthy-brown/50 overflow-hidden">
      <img
        src={entry.imageUrl} // This will now come from getMockJournalEntries with local paths
        alt={entry.title}
        className="w-full h-64 object-cover"
      />
      <div className="p-6">
        <h3 className="text-2xl font-semibold text-sky-blue mb-2">{entry.title}</h3>
        <p className="text-sm text-gray-400 mb-1">Date: {entry.date} | Perspective: {entry.perspective}</p>
        <p className="text-sm text-gray-400 mb-4">Artist: {entry.artistName}</p>

        <div className="bg-jungle-green/30 p-4 rounded-md mb-4">
          <p className="text-gray-300 italic">
            <span className="font-semibold text-accent-teal">AI Snippet & Storyteller's Touch:</span> "{entry.aiSnippet}"
          </p>
          <p className="text-xs text-gray-400 mt-2">Key AI-detected behaviors: {entry.behaviorsObserved.join(', ')}</p>
        </div>

        {isExpanded && (
          <div className="mt-4 prose prose-invert max-w-none text-gray-200">
            <p>{entry.fullStory}</p>
          </div>
        )}
        <Button
          variant="outline"
          size="sm"
          onClick={() => setIsExpanded(!isExpanded)}
          className="mt-4"
          rightIcon={isExpanded ? <ChevronUpIcon className="w-4 h-4" /> : <ChevronDownIcon className="w-4 h-4" />}
        >
          {isExpanded ? "Show Less" : "Read Today's Full Story"}
        </Button>
      </div>
    </Card>
  );
};

// USER INSTRUCTION: For MilestoneItem images, ensure you have corresponding images like
// 'milestone-piko-first-swim.png', 'milestone-moodeng-teaching.png' in your './public/images/' folder.
const MilestoneItem: React.FC<{ title: string; description: string; date: string; hippoName: string; imageUrl: string }> = ({ title, description, date, hippoName, imageUrl }) => (
  <Card className="mb-4 bg-jungle-green/40">
    <div className="flex items-start space-x-4">
      <img src={imageUrl} alt={title} className="w-20 h-20 object-cover rounded-md"/>
      <div>
        <h4 className="text-lg font-semibold text-sky-blue">{title} ({hippoName})</h4>
        <p className="text-sm text-gray-400 mb-1">{date}</p>
        <p className="text-gray-300 text-sm">{description}</p>
      </div>
    </div>
  </Card>
);

export const JournalScreen: React.FC = () => {
  const [journalEntries, setJournalEntries] = useState<JournalEntry[]>([]);
  const [hippos, setHippos] = useState<HippoProfile[]>([]);
  const [userQuestion, setUserQuestion] = useState('');
  const [aiAnswer, setAiAnswer] = useState<string | null>(null);
  const [isLoadingAnswer, setIsLoadingAnswer] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setJournalEntries(getMockJournalEntries(3));
    setHippos(getMockHippoProfiles());
  }, []);

  const handleAskQuestion = useCallback(async () => {
    if (!userQuestion.trim()) return;
    setIsLoadingAnswer(true);
    setAiAnswer(null);
    setError(null);
    try {
      const responseText = await askGeminiMock(userQuestion);
      setAiAnswer(responseText);
    } catch (e) {
      setError("Sorry, I couldn't fetch an answer right now. Please try again later.");
      console.error("Error asking Gemini:", e);
    } finally {
      setIsLoadingAnswer(false);
    }
  }, [userQuestion]);

  const moodeng = hippos.find(h => h.name === 'Moodeng');
  const piko = hippos.find(h => h.name === 'Piko');

  return (
    <div>
      <Section
        title="Tales from the Waterhole"
        subtitle="Engage with stories from Moodeng & Piko's world, a blend of AI insights and human creativity."
      >
        <h3 className="text-2xl font-semibold text-accent-teal mb-4 mt-8">Today's Adventures</h3>
        {journalEntries.length > 0 ? (
          journalEntries.map(entry => <JournalPost key={entry.id} entry={entry} />)
        ) : (
          <p className="text-gray-400">No journal entries yet. Check back soon!</p>
        )}
      </Section>

      <Section title="Growth Milestones & Memorable Moments">
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="text-xl font-semibold text-sky-blue mb-3">Piko's Milestones</h4>
            {piko && (
              <>
                <MilestoneItem title="First Swim" description="Piko took their first solo swim today! AI detected confident paddling." date="July 15, 2024" hippoName="Piko" imageUrl={`./public/images/milestone-piko-first-swim.png`} />
                <MilestoneItem title="New Foraging Behavior" description="AI identified Piko attempting to forage for riverbed plants." date="July 22, 2024" hippoName="Piko" imageUrl={`./public/images/milestone-piko-foraging.png`} />
              </>
            )}
          </div>
          <div>
            <h4 className="text-xl font-semibold text-sky-blue mb-3">Moodeng's Moments</h4>
            {moodeng && (
              <>
                <MilestoneItem title="Teaching Piko" description="Moodeng was observed patiently guiding Piko near the deeper water." date="July 10, 2024" hippoName="Moodeng" imageUrl={`./public/images/milestone-moodeng-teaching.png`} />
                <MilestoneItem title="Relaxed Sunbathing" description="AI noted an extended period of calm sunbathing, indicating contentment." date="July 20, 2024" hippoName="Moodeng" imageUrl={`./public/images/milestone-moodeng-sunbathing.png`} />
              </>
            )}
          </div>
        </div>
      </Section>

      <Section title="Ask Moodeng & Piko (via AI)">
        <Card className="bg-jungle-green/40">
          <p className="text-gray-300 mb-4">Curious about hippos? Type your question below. Our AI, trained on hippo facts and observed data, will try to answer from their perspective!</p>
          <div className="flex flex-col sm:flex-row gap-4 mb-4">
            <input
              type="text"
              value={userQuestion}
              onChange={(e) => setUserQuestion(e.target.value)}
              placeholder="e.g., What's Piko's favorite food?"
              className="flex-grow p-3 rounded-md bg-earthy-brown/50 text-gray-100 placeholder-gray-400 border border-gray-600 focus:ring-accent-teal focus:border-accent-teal"
            />
            <Button onClick={handleAskQuestion} disabled={isLoadingAnswer || !userQuestion.trim()}>
              {isLoadingAnswer && <LoadingSpinner className="w-5 h-5 mr-2"/>}
              Ask AI
            </Button>
          </div>
          {isLoadingAnswer && <p className="text-sky-blue">Thinking...</p>}
          {error && <p className="text-red-400">{error}</p>}
          {aiAnswer && (
            <div className="mt-4 p-4 bg-earthy-brown/30 rounded-md">
              <p className="text-accent-teal font-semibold mb-1">AI Hippo says:</p>
              <p className="text-gray-200 whitespace-pre-wrap">{aiAnswer}</p>
            </div>
          )}
          <p className="text-xs text-gray-500 mt-4">Disclaimer: Answers are generated by AI based on general hippo knowledge and observed behaviors in our HippoSphere. They are for entertainment and education.</p>
        </Card>
      </Section>
    </div>
  );
};