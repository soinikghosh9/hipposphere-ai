// src/constants.ts
import { BehaviorCategory } from "./types";

export const APP_NAME = "HippoSphere AI";
export const TAGLINE = "HippoSphere AI: Where Data Becomes a Story.";

/**
 * ROUTES for the HashRouter-based navigation.
 * 
 * - WELCOME is the “front page” (Root URL after the `#`).
 * - All other routes live under /dashboard, /journal, etc.
 */
export const ROUTES = {
  WELCOME: "/",      // HashRouter will map “#/” to the FrontPage
  DASHBOARD: "/dashboard",
  JOURNAL: "/journal",
  OBSERVER: "/observer",
  INSIGHTS: "/insights",
  COMMUNITY: "/community",
};

/**
 * If you need to read an API key from process.env (e.g. VITE_‐prefixed or otherwise),
 * you can refer to it via this constant. In Vite, you’d do:
 *    import.meta.env.VITE_API_KEY
 * or any custom mode variable you set up.
 */
export const API_KEY_ENV_VAR = "API_KEY";

/**
 * Model names for any Gemini‐style text or image APIs you might call.
 * (These are just placeholders; replace as needed.)
 */
export const GEMINI_TEXT_MODEL = "gemini-2.5-flash-preview-04-17";
export const GEMINI_IMAGE_MODEL = "imagen-3.0-generate-002";

/**
 * Default “artist” credit if you’re auto‐generating images or pulling from a service.
 */
export const DEFAULT_ARTIST_NAME = "The HippoSphere Team";

/**
 * Define a color mapping based on your BehaviorCategory enum.
 * Make sure BehaviorCategory is actually exported from “src/types.ts”.
 */
export const BEHAVIOR_COLORS: Record<BehaviorCategory, string> = {
  [BehaviorCategory.RESTING]: "#60A5FA",           // sky-blue
  [BehaviorCategory.FEEDING]: "#34D399",           // emerald-400
  [BehaviorCategory.MOVING]: "#F59E0B",            // sunrise-orange
  [BehaviorCategory.SOCIAL_INTERACTION]: "#A78BFA",// violet-400
  [BehaviorCategory.PLAYING]: "#EC4899",           // pink-500
  [BehaviorCategory.FORAGING]: "#84CC16",          // lime-500
};

/**
 * If you ever need to import the enum itself elsewhere,
 * re‐export it from this file as well.
 */
export type { BehaviorCategory };

/**
 * Define categories of emotions, plus a color map for each category.
 * These are standalone—no external “types.ts” needed for EmotionCategory.
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

export const EMOTION_COLORS: Record<EmotionCategory, string> = {
  [EmotionCategory.CONTENT]: "#4ADE80",     // green-400
  [EmotionCategory.PLAYFUL]: "#FACC15",     // yellow-400
  [EmotionCategory.CURIOUS]: "#C084FC",     // purple-400
  [EmotionCategory.ALERT]: "#FB923C",       // orange-400
  [EmotionCategory.CALM]: "#38BDF8",        // cyan-400
  [EmotionCategory.ENERGETIC]: "#2DD4BF",   // accent-teal
  [EmotionCategory.SLEEPY]: "#94A3B8",      // slate-400
};

/**
 * Preset dimensions for various UI image placeholders.
 * You can override or extend this as needed.
 */
export const PLACEHOLDER_IMAGE_DIMENSIONS = {
  banner: { width: 1200, height: 400 },
  card: { width: 400, height: 300 },
  avatar: { width: 100, height: 100 },
  gallery: { width: 600, height: 400 },
};

/**
 * PICSUM_SEEDS: A set of seed strings to pass to
 * a picture‐generation service (e.g., Lorem Picsum, your own API).
 * Each key corresponds to a unique image use‐case in the UI.
 */
export const PICSUM_SEEDS = {
  // Specific hippo profile pictures:
  moodengProfile: "moodeng-pygmy-hippo-portrait",
  pikoProfile: "piko-baby-pygmy-hippo-cute",

  // Screen‐specific backgrounds or hero images:
  welcomeBg: "moodeng-piko-jungle-stream",    // for FrontPage / WelcomeScreen
  dashboardBg: "pygmy-hippo-lush-lagoon",     // for Dashboard background
  observerFeed: "live-pygmy-hippo-habitat-view", // for Observer live‐feed placeholder

  // Content‐specific assets:
  journalAdventure: "moodeng-piko-daily-discovery",
  milestonePiko: "piko-hippo-growth-milestone",
  milestoneMoodeng: "moodeng-hippo-motherly-moment",
  pygmyHippoFactImage: "pygmy-hippo-species-info",

  // Team & community themed seeds:
  teamMemberArtist: "nature-storyteller-artist",
  teamMemberTech: "ai-conservation-tech-expert",
  teamMemberConservation: "wildlife-conservation-coordinator",
  communitySupport: "hippo-conservation-donation",

  // General “theme” seeds (navigation cards, thumbnails, etc.):
  lushJungle: "deep-lush-jungle-foliage",
  sereneWaterhole: "peaceful-hippo-waterhole",
  natureCommunity: "connect-with-nature-community",
  aiNatureTech: "ai-technology-in-nature",

  // Fallback / generic seeds:
  habitat: "general-hippo-habitat-scene",
  conservation: "general-wildlife-conservation-theme",
};
