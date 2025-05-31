// src/types.ts

/**
 * Enum of possible hippo behavior categories.
 * Match these values to the keys used in BEHAVIOR_COLORS in constants.ts.
 */
export enum BehaviorCategory {
  RESTING = "Resting",
  FEEDING = "Feeding",
  MOVING = "Moving",
  SOCIAL_INTERACTION = "Social Interaction",
  PLAYING = "Playing",
  FORAGING = "Foraging",
}

/**
 * A single data point representing a hippo's behavior for charting purposes.
 * - name: the behavior category (one of BehaviorCategory)
 * - value: a numeric measurement (e.g. percentage or duration)
 * - fill: a color string to use when rendering a chart
 */
export interface ActivityDataPoint {
  name: BehaviorCategory;
  value: number;
  fill: string;
}

/**
 * A single log entry describing what a hippo did at a given timestamp.
 */
export interface ActivityLogEntry {
  id: string;                 // Unique ID, e.g. "log-001"
  timestamp: string;          // ISO timestamp, e.g. "2025-05-31T12:34:56Z"
  hippoName: "Moodeng" | "Piko"; // Which hippo was observed
  behavior: BehaviorCategory; // One of the BehaviorCategory enum
  icon?: React.FC<React.SVGProps<SVGSVGElement>>; // Optional icon component
  details?: string;           // Optional descriptive details (e.g. "Piko started grazing")
}

/**
 * A single data point representing a hippo's emotional state for charting.
 * - name: one of the EmotionCategory enum values
 * - value: numeric intensity (0–100 or 0–1 scale)
 * - fill: color for chart segments
 */
export interface EmotionDataPoint {
  name: EmotionCategory;
  value: number;
  fill: string;
}

/**
 * Profile information for a single hippo, used in Insights or Profile screens.
 */
export interface HippoProfile {
  id: string;                 // e.g. "moodeng-01"
  name: string;               // e.g. "Moodeng"
  species: string;            // e.g. "Pygmy Hippo"
  estimatedAge: string;       // e.g. "4 years"
  originStory: string;        // A short narrative
  imageUrl: string;           // URL or imported asset
  thumbnailUrl?: string;      // Optional smaller version
}

/**
 * A journal entry created by a user or AI, including metadata for display.
 */
export interface JournalEntry {
  id: string;                 // e.g. "entry-20250531-001"
  date: string;               // e.g. "2025-05-31"
  title: string;              // e.g. "Piko’s First Dive"
  aiSnippet: string;          // Short AI summary
  fullStory: string;          // Full narrative text
  imageUrl: string;           // URL or imported asset
  artistName: string;         // e.g. "The HippoSphere Team"
  perspective: "Piko" | "Moodeng" | "Observer"; // Whose POV
  behaviorsObserved: BehaviorCategory[]; // List of behaviors recorded
}

/**
 * A single data point showing combined behavior trends over time,
 * for a multi-series chart (e.g. two hippos' active hours per week).
 */
export interface BehavioralTrendDataPoint {
  name: string;               // e.g. "Week 1", "Week 2"
  MoodengActiveHours: number; // e.g. total hours Moodeng was active
  PikoActiveHours: number;    // e.g. total hours Piko was active
}

/**
 * A fact card about pygmy hippos, used in InsightsScreen.
 */
export interface PygmyHippoFact {
  id: string;                 // Unique ID, e.g. "fact-001"
  title: string;              // e.g. "Pygmy Hippo Lifespan"
  description: string;        // e.g. "Pygmy hippos live 30–40 years..."
  imageUrl?: string;          // Optional URL or asset
  icon?: React.FC<React.SVGProps<SVGSVGElement>>; // Optional icon component
}

/**
 * A single team member profile (for community/team screen).
 */
export interface TeamMember {
  id: string;                 // e.g. "team-01"
  name: string;               // e.g. "Dr. Jane Doe"
  role: string;               // e.g. "Wildlife Conservation Coordinator"
  bio: string;                // Short biography
  imageUrl: string;           // URL or imported asset
}

/** 
 * Request payload when generating a joint “story” from Gemini:
 * Arrays of behavior categories observed for each hippo.
 */
export interface GeminiStoryRequest {
  moodengBehaviors: BehaviorCategory[];
  pikoBehaviors: BehaviorCategory[];
}

/** 
 * Request payload when asking a question to the Gemini chat endpoint.
 */
export interface GeminiQuestionRequest {
  question: string;
}

/**
 * A chunk of grounding information (e.g., a web URI + title) used for LLM grounding.
 */
export interface GroundingChunkWeb {
  uri: string;                // e.g. "https://en.wikipedia.org/wiki/Pygmy_hippo"
  title: string;              // e.g. "Pygmy hippo"
}
export interface GroundingChunk {
  web: GroundingChunkWeb;
}

/**
 * Because EmotionCategory is already defined in constants.ts and re-exported there,
 * we can import it here if needed, but we’ll declare it again to avoid circular imports.
 */
export enum EmotionCategory {
  CONTENT = "Content",
  PLAYFUL = "Playful",
  CURIOUS = "Curious",
  ALERT = "Alert",
  CALM = "Calm",
  ENERGETIC = "Energetic",
  SLEEPY = "Sleepy",
}
