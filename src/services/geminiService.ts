
// IMPORTANT: This is a MOCK service.
// In a real application, you would use @google/genai library
// and ensure API_KEY is securely managed (e.g., via environment variables on a backend).
// The frontend should ideally not directly call GoogleGenAI with an API key.
// These functions simulate what a backend service interacting with Gemini might provide.

import { GoogleGenAI, GenerateContentResponse, Chat, GroundingChunk } from "@google/genai"; // For types and conceptual structure
import {
  HippoProfile,
  JournalEntry,
  ActivityLogEntry,
  BehaviorCategory,
  ActivityDataPoint,
  BehavioralTrendDataPoint,
  PygmyHippoFact,
  TeamMember,
  GeminiStoryRequest,
  GeminiQuestionRequest,
  EmotionDataPoint,
  EmotionCategory
} from '../types';
import { DEFAULT_ARTIST_NAME, PICSUM_SEEDS, PLACEHOLDER_IMAGE_DIMENSIONS, GEMINI_TEXT_MODEL, BEHAVIOR_COLORS, EMOTION_COLORS } from '../constants';
import { LeafIcon, PawPrintIcon, EyeIcon, WaterDropletIcon, UsersIcon, LightBulbIcon } from '../components/common/Icons';


// MOCK API_KEY - DO NOT USE IN PRODUCTION. This is just for conceptual representation.
const MOCK_API_KEY = "YOUR_GEMINI_API_KEY_WOULD_BE_HERE_FROM_PROCESS_ENV";
// let ai: GoogleGenAI | null = null;
// if (typeof process !== 'undefined' && process.env && process.env.API_KEY) {
//   ai = new GoogleGenAI({apiKey: process.env.API_KEY});
// } else {
//   console.warn("Gemini API key not found. Using mock data. For real calls, set API_KEY in your environment.");
// }

// --- Mock Data Generators ---
// USER INSTRUCTION: Replace 'placeholder-image.png' and other specific image names below
// with your actual image filenames located in the './public/images/' directory.

export const getMockHippoProfiles = (): HippoProfile[] => [
  {
    id: 'moodeng',
    name: 'Moodeng',
    species: 'Pygmy Hippo (Choeropsis liberiensis)',
    estimatedAge: 'Approx. 8 years old',
    originStory: 'Born at this conservation center, Moodeng is a calm and experienced mother.',
    imageUrl: `./public/images/moodeng-profile.png`, // Updated path
  },
  {
    id: 'piko',
    name: 'Piko',
    species: 'Pygmy Hippo (Choeropsis liberiensis)',
    estimatedAge: 'Approx. 6 months old',
    originStory: 'Piko, meaning "little one", is Moodeng\'s curious and playful calf, always eager to explore.',
    imageUrl: `./public/images/piko-profile.png`, // Updated path
  },
];

export const getMockJournalEntries = (count: number): JournalEntry[] => {
  const entries: JournalEntry[] = [];
  const today = new Date();
  for (let i = 0; i < count; i++) {
    const date = new Date(today);
    date.setDate(today.getDate() - i);
    const behaviors = [BehaviorCategory.FEEDING, BehaviorCategory.PLAYING, BehaviorCategory.RESTING].sort(() => 0.5 - Math.random()).slice(0,2);
    entries.push({
      id: `journal-${i}`,
      date: date.toLocaleDateString(),
      title: `A Day of ${behaviors[0]} and ${behaviors[1]}`,
      aiSnippet: `The AI detected Piko playfully splashing Moodeng at ${Math.floor(Math.random()*12)+1}:${Math.floor(Math.random()*59).toString().padStart(2,'0')} PM! Storyteller ${DEFAULT_ARTIST_NAME} imagines Piko was saying, 'Wake up, Mama, let's explore!'`,
      fullStory: `The sun dappled through the leaves, painting shifting patterns on the water's surface. Piko, feeling a surge of morning energy, nudged Moodeng gently. The AI noted this interaction as 'nuzzling' at 9:03 AM. Moodeng responded with a soft grunt, her eyes still half-closed. Later, around 2:15 PM, Piko was observed ${behaviors[0].toLowerCase()} near the water's edge, while Moodeng engaged in some ${behaviors[1].toLowerCase()}. This full story is a creative interpretation by our human storyteller, inspired by AI-detected behaviors like "${behaviors.join('", "')}".`,
      // USER INSTRUCTION: You might want to create multiple images like 'journal-entry-1.png', 'journal-entry-2.png'
      imageUrl: `./public/images/journal-adventure-default-${i % 2 === 0 ? '1' : '2'}.png`, // Updated path
      artistName: DEFAULT_ARTIST_NAME,
      perspective: i % 2 === 0 ? 'Piko' : 'Observer',
      behaviorsObserved: behaviors,
    });
  }
  return entries;
};

export const getMockActivityLog = (count: number): ActivityLogEntry[] => {
  const log: ActivityLogEntry[] = [];
  const now = new Date();
  const behaviors = [
    { name: BehaviorCategory.RESTING, icon: PawPrintIcon},
    { name: BehaviorCategory.FEEDING, icon: LeafIcon },
    { name: BehaviorCategory.MOVING, icon: EyeIcon },
    { name: BehaviorCategory.SOCIAL_INTERACTION, icon: UsersIcon },
    { name: BehaviorCategory.PLAYING, icon: WaterDropletIcon },
    { name: BehaviorCategory.FORAGING, icon: LeafIcon }
  ];
  for (let i = 0; i < count; i++) {
    const time = new Date(now.getTime() - i * 5 * 60 * 1000); // Every 5 minutes
    const randomBehavior = behaviors[Math.floor(Math.random() * behaviors.length)];
    log.push({
      id: `log-${i}`,
      timestamp: time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      hippoName: i % 2 === 0 ? 'Moodeng' : 'Piko',
      behavior: randomBehavior.name,
      icon: randomBehavior.icon
    });
  }
  return log;
};

export const getMockActivityData = (hippoName: 'Moodeng' | 'Piko'): ActivityDataPoint[] => {
  const allCategories = Object.values(BehaviorCategory);
  const data: ActivityDataPoint[] = [];
  let remainingPercentage = 100;

  for (let i = 0; i < allCategories.length; i++) {
    const category = allCategories[i];
    let value = 0;
    if (i === allCategories.length - 1) {
      value = remainingPercentage;
    } else {
      value = Math.floor(Math.random() * (remainingPercentage / (allCategories.length - i))) + (hippoName === 'Piko' && category === BehaviorCategory.PLAYING ? 10 : 0);
      value = Math.min(value, remainingPercentage);
    }
    remainingPercentage -= value;
    data.push({ name: category, value: value, fill: BEHAVIOR_COLORS[category] || '#8884d8' });
  }
  const currentSum = data.reduce((sum, item) => sum + item.value, 0);
  if (currentSum !== 100 && currentSum > 0) {
    const lastItem = data[data.length -1];
    lastItem.value += (100-currentSum);
    if(lastItem.value < 0) lastItem.value = 0;
  }

  return data;
};


export const getMockBehavioralTrends = (): BehavioralTrendDataPoint[] => {
  const trends: BehavioralTrendDataPoint[] = [];
  for (let i = 1; i <= 8; i++) {
    trends.push({
      name: `Week ${i}`,
      MoodengActiveHours: Math.floor(Math.random() * 4) + 6,
      PikoActiveHours: Math.floor(Math.random() * 5) + 8,
    });
  }
  return trends;
};

export const getMockPygmyHippoFacts = (): PygmyHippoFact[] => [
  {
    id: 'fact-1',
    title: 'Habitat',
    description: 'Pygmy hippos are native to the forests and swamps of West Africa, primarily Liberia, Sierra Leone, Guinea, and Ivory Coast.',
    imageUrl: `./public/images/fact-habitat.png`, // Updated path
    icon: LeafIcon
  },
  {
    id: 'fact-2',
    title: 'Diet',
    description: 'They are herbivores, feeding on ferns, broad-leaved plants, and fruits they find in the forests.',
    imageUrl: `./public/images/fact-diet.png`, // Updated path
    icon: LeafIcon
  },
  {
    id: 'fact-3',
    title: 'Social Structure',
    description: 'Unlike common hippos, pygmy hippos are mostly solitary or live in pairs. They are also more nocturnal.',
    imageUrl: `./public/images/fact-social.png`, // Updated path
    icon: UsersIcon
  },
  {
    id: 'fact-4',
    title: 'Unique Adaptations',
    description: 'Their eyes are positioned more on the sides of their head, better suited for navigating forests than open water.',
    imageUrl: `./public/images/fact-adaptations.png`, // Updated path
    icon: EyeIcon
  },
  {
    id: 'fact-5',
    title: 'Size',
    description: 'Pygmy hippos are much smaller than their common hippo cousins, typically about half as tall and weighing only about 1/4th as much.',
    imageUrl: `./public/images/fact-size.png`, // Updated path
    icon: LightBulbIcon
  },
  {
    id: 'fact-6',
    title: 'Water Dependency',
    description: 'Like common hippos, they are semi-aquatic and rely on water to keep their skin moisturized and their body temperature cool.',
    imageUrl: `./public/images/fact-water.png`, // Updated path
    icon: WaterDropletIcon
  },
];

export const getMockTeamMembers = (): TeamMember[] => [
  {
    id: 'team-1',
    name: 'Dr. Elara Vance',
    role: 'Lead Ethologist & Storyteller',
    bio: 'Elara combines her passion for animal behavior with captivating storytelling to bring Moodeng and Piko\'s world to life.',
    imageUrl: `./public/images/team-elara-vance.png`, // Updated path
  },
  {
    id: 'team-2',
    name: 'Ben Carter',
    role: 'AI & Systems Engineer',
    bio: 'Ben designs and maintains the AI systems that power HippoSphere, turning data into insights.',
    imageUrl: `./public/images/team-ben-carter.png`, // Updated path
  },
  {
    id: 'team-3',
    name: 'Aisha Khan',
    role: 'Conservation Coordinator',
    bio: 'Aisha liaises with conservation partners and educates the public on pygmy hippo protection.',
    imageUrl: `./public/images/team-aisha-khan.png`, // Updated path
  },
];


export const getMockEmotionData = (hippoName: 'Moodeng' | 'Piko'): EmotionDataPoint[] => {
  const emotions: EmotionCategory[] = [
    EmotionCategory.CONTENT,
    EmotionCategory.PLAYFUL,
    EmotionCategory.CURIOUS,
    EmotionCategory.ALERT,
    EmotionCategory.CALM,
    EmotionCategory.ENERGETIC,
    EmotionCategory.SLEEPY,
  ];

  // Assign base values and then randomize slightly
  let baseValues: Record<EmotionCategory, number>;

  if (hippoName === 'Moodeng') {
    baseValues = {
      [EmotionCategory.CONTENT]: 60,
      [EmotionCategory.CALM]: 70,
      [EmotionCategory.ALERT]: 30,
      [EmotionCategory.SLEEPY]: 20,
      [EmotionCategory.CURIOUS]: 25,
      [EmotionCategory.PLAYFUL]: 15,
      [EmotionCategory.ENERGETIC]: 20,
    };
  } else { // Piko
    baseValues = {
      [EmotionCategory.PLAYFUL]: 75,
      [EmotionCategory.ENERGETIC]: 70,
      [EmotionCategory.CURIOUS]: 60,
      [EmotionCategory.CONTENT]: 40,
      [EmotionCategory.ALERT]: 20,
      [EmotionCategory.CALM]: 30,
      [EmotionCategory.SLEEPY]: 40,
    };
  }
  
  return emotions.map(emotion => ({
    name: emotion,
    // Ensure value is between 0 and 100 after randomization
    value: Math.max(0, Math.min(100, baseValues[emotion] + Math.floor(Math.random() * 30) - 15)),
    fill: EMOTION_COLORS[emotion] || '#8884d8'
  })).sort((a,b) => b.value - a.value).slice(0, 5); // Show top 5 dominant emotions
};


// --- Conceptual Gemini API Functions (MOCK IMPLEMENTATIONS) ---

export const generateStoryWithGeminiMock = async (request: GeminiStoryRequest): Promise<string> => {
  console.log("Simulating Gemini story generation with request:", request);
  await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate network delay
  return `Piko wriggled with excitement! Mama Moodeng was ${request.moodengBehaviors[0] || 'resting peacefully'}, but the world was full of adventure. Piko, having just been seen ${request.pikoBehaviors[0] || 'splashing'}, thought, "What's that rustling in the bushes?" With a brave little heart, Piko decided to investigate, always knowing Mama was close by. Their bond was as strong as the oldest trees in their jungle home.`;
};

export const askGeminiMock = async (question: string): Promise<string> => {
  console.log("Simulating asking Gemini:", question);
  await new Promise(resolve => setTimeout(resolve, 1500)); // Simulate network delay
  if (question.toLowerCase().includes("favorite food")) {
    return "Ooh, I love yummy river plants and tasty fallen fruits! Piko especially enjoys the sweet berries we find near the big tree!";
  } else if (question.toLowerCase().includes("piko") && question.toLowerCase().includes("play")) {
    return "Piko loves to play! Splashing in the water is super fun, and sometimes Piko tries to chase the little fish. It's important to explore and have a good time!";
  }
  return `That's a great question! As a hippo, I can tell you that "${question}" is something we think about... while munching on delicious leaves! We spend a lot of time near the water, keeping cool and exploring our lush home.`;
};

export const getStructuredDataFromGeminiMock = async <T,>(prompt: string): Promise<T | null> => {
    console.log("Simulating getting structured data from Gemini with prompt:", prompt);
    await new Promise(resolve => setTimeout(resolve, 1000));
    if (prompt.includes("enrichment ideas")) {
        const mockJsonResponse = {
            ideas: [
                { name: "Floating fruit puzzle", description: "Fruits hidden in a floating device." },
                { name: "Scent trails", description: "New interesting smells introduced in the habitat." }
            ]
        };
        return mockJsonResponse as unknown as T;
    }
    return null;
};