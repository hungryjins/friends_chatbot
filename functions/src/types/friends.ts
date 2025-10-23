import { Timestamp } from "firebase-admin/firestore";

/**
 * Friends Episode data structure
 */
export interface FriendsEpisode {
  id: string; // S01E01
  title: string;
  season: number;
  episode_number: number;
  scenes: string[]; // scene IDs
  plot_summary: string;
  created_at: Timestamp;
  updated_at: Timestamp;
}

/**
 * Friends Scene data structure
 */
export interface SceneLine {
  line_number: number;
  type: "dialogue" | "action" | "narration";
  speaker?: string;
  text: string;
}

export interface FriendsScene {
  id: string; // S01E01_001
  episode_id: string;
  season: number;
  episode_number: number;
  scene_number: number;
  location: string;
  scene_description: string;
  characters: string[];
  raw_text: string;
  lines: SceneLine[];
  firestore_path: string;
  created_at: Timestamp;
  updated_at: Timestamp;
}

/**
 * Friends Character data structure
 */
export interface FriendsCharacter {
  id: string;
  name: string;
  full_name: string;
  description: string;
  personality_traits: string[];
  catchphrases: string[];
  relationships: Record<string, string>;
  background: string;
  created_at: Timestamp;
  updated_at: Timestamp;
}

/**
 * Cultural Context data structure
 */
export interface CulturalContext {
  id: string;
  topic: string;
  explanation: string;
  examples: string[];
  related_episodes: string[];
  created_at: Timestamp;
  updated_at: Timestamp;
}

/**
 * Episode Plot data structure
 */
export interface EpisodePlot {
  id: string;
  episode_id: string;
  plot_summary: string;
  main_themes: string[];
  key_moments: string[];
  character_development: Record<string, string>;
  created_at: Timestamp;
  updated_at: Timestamp;
}

/**
 * Intent Analysis for RAG system
 */
export interface IntentAnalysis {
  intent:
    | "episode_recommendation"
    | "character_info"
    | "plot_summary"
    | "scene_script"
    | "cultural_context"
    | "practice_session"
    | "general_chat";
  topic: string;
  details: string;
  confidence: number;
}

/**
 * Filter conditions for Pinecone search
 */
export interface FilterConditions {
  chunk_type?: "scene" | "plot";
  season?: number;
  episode_id?: string;
  character?: string;
  location?: string;
}

/**
 * Search result from Pinecone
 */
export interface SearchResult {
  text: string;
  metadata: {
    episode_id: string;
    scene_id?: string;
    character?: string;
    chunk_type: string;
    season: number;
    episode_number: number;
  };
  score: number;
}

/**
 * Chat message structure
 */
export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  timestamp: Timestamp;
}

/**
 * Chat context for maintaining conversation state
 */
export interface ChatContext {
  session_id: string;
  messages: ChatMessage[];
  current_intent?: IntentAnalysis;
  practice_session?: PracticeSession;
}

/**
 * Practice session data
 */
export interface PracticeSession {
  session_id: string;
  character: string;
  scenario: string;
  target_expressions: string[];
  current_scene?: FriendsScene;
  score: number;
  created_at: Timestamp;
  updated_at: Timestamp;
}

/**
 * API Response types
 */
export interface ApiResponse<T = any> {
  success: boolean;
  message: string;
  data?: T;
  error?: string;
}

/**
 * Text similarity calculation result interface
 */
export interface SimilarityResult {
  similarity: number;           // Overall similarity score (0-1)
  isCorrect: boolean;          // Whether the answer is correct
  feedback: string;            // User feedback message
  detailedAnalysis: {
    wordSimilarity: number;
    characterSimilarity: number;
    embeddingSimilarity?: number;
    exactMatch: boolean;
    veryCloseMatch: boolean;
  };
}

/**
 * Enhanced practice session data structure
 */
export interface EnhancedPracticeSession {
  session_id: string;
  user_id: string;
  episode_id: string;
  character: string;
  scene_number: number;
  scene_id: string;
  current_line_index: number;
  total_lines: number;
  correct_answers: number;
  total_attempts: number;
  status: 'active' | 'completed' | 'paused';
  started_at: Timestamp;
  completed_at?: Timestamp;
  created_at: Timestamp;
  updated_at: Timestamp;
}

/**
 * Structured cultural explanation result
 */
export interface StructuredCulturalExplanation {
  expression: string;
  meaning: string;
  origin: string;
  usage_context: string;
  conversation_examples: string[];
  similar_expressions: string[];
  friends_context: string;
  follow_up_suggestions: string[];
  related_episodes: string[];
  practice_tips: string[];
}

/**
 * Friends RAG Chatbot Response
 */
export interface FriendsResponse {
  response: string;
  intent: IntentAnalysis;
  search_results?: SearchResult[];
  episode_recommendation?: FriendsEpisode[];
  character_info?: FriendsCharacter;
  episode_plot?: EpisodePlot;
  scene_script?: FriendsScene;
  cultural_context?: CulturalContext;
  practice_session?: PracticeSession;
  similarity_result?: SimilarityResult;
  enhanced_practice_session?: EnhancedPracticeSession;
  structured_cultural_explanation?: StructuredCulturalExplanation;
}
