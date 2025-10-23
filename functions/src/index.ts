import * as functions from "firebase-functions";
import * as admin from "firebase-admin";
import express from "express";
import cors from "cors";
import { FriendsRAGService } from "./services/friends-rag-service";

// Initialize Firebase Admin
admin.initializeApp();

// Create Express app
const app = express();

// Middleware
app.use(
  cors({
    origin: [
      "https://conversation-practice-f2199.web.app",
      "https://conversation-practice-f2199.firebaseapp.com",
      "https://dailyconvo.com",
      "https://www.dailyconvo.com",
      "http://localhost:3000",
      "http://localhost:5173",
      "http://localhost:5174",
      "http://localhost:8080",
    ],
    credentials: true,
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
  })
);
app.use(express.json());

// Initialize Friends RAG Service
const friendsRAGService = new FriendsRAGService();

/**
 * Health check endpoint
 */
app.get("/", (_req, res) => {
  res.json({
    status: "Friends Chatbot Firebase Functions API is running",
    version: "2.0.0",
    description:
      "Enhanced Friends English Learning Chatbot with Firestore and Pinecone integration",
    data_sources: {
      firestore: {
        total_collections: 5,
        collections: [
          "friends_scenes",
          "friends_plots",
          "friends_characters",
          "friends_cultural_contexts",
          "practice_sessions",
        ],
      },
      pinecone: {
        index: "convo",
        vector_dimensions: 1536,
        model: "text-embedding-3-small",
      },
    },
    features: [
      "episode_recommendation",
      "character_info",
      "plot_summary",
      "scene_script",
      "cultural_context",
      "practice_session",
      "general_chat",
      "scene_by_id",
      "episode_scenes",
      "episodes_list",
      "database_stats",
    ],
    endpoints: {
      chat: "POST /chat",
      episodes_recommend: "POST /episodes/recommend",
      character_info: "GET /characters/:characterName",
      episode_plot: "GET /episodes/:episodeId/plot",
      scene_search: "POST /scenes/search",
      scene_by_id: "GET /scenes/:sceneId",
      episode_scenes: "GET /episodes/:episodeId/scenes",
      episodes_list: "GET /episodes",
      cultural_context: "POST /cultural-context",
      practice_start: "POST /practice/start",
      search: "POST /search",
      stats: "GET /stats",
      migrate_data: "POST /migrate-data",
    },
  });
});

/**
 * Main chat endpoint
 * POST /api/friends/chat
 */
app.post("/chat", async (req, res) => {
  try {
    const { message, chatHistory = [] } = req.body;

    if (!message || typeof message !== "string") {
      res.status(400).json({
        success: false,
        error: "Message is required and must be a string",
      });
      return;
    }

    console.log(`Processing chat message: "${message}"`);

    // Process chat with Friends RAG Service
    const response = await friendsRAGService.chat(message, chatHistory);

    res.json({
      success: true,
      message: "Chat processed successfully",
      data: response,
    });
  } catch (error) {
    console.error("Error in chat endpoint:", error);
    res.status(500).json({
      success: false,
      error: "Internal server error",
      message: "Failed to process chat message",
    });
  }
});

/**
 * Episode recommendation endpoint
 * POST /api/friends/episodes/recommend
 */
app.post("/episodes/recommend", async (req, res) => {
  try {
    const { topic } = req.body;

    if (!topic || typeof topic !== "string") {
      res.status(400).json({
        success: false,
        error: "Topic is required and must be a string",
      });
      return;
    }

    const response = await friendsRAGService.chat(
      `Recommend episodes about ${topic}`,
      []
    );

    res.json({
      success: true,
      message: "Episode recommendations generated",
      data: {
        topic,
        recommendations: response.response,
        intent: response.intent,
      },
    });
  } catch (error) {
    console.error("Error in episode recommendation:", error);
    res.status(500).json({
      success: false,
      error: "Failed to generate episode recommendations",
    });
  }
});

/**
 * Character information endpoint
 * GET /api/friends/characters/:characterName
 */
app.get("/characters/:characterName", async (req, res) => {
  try {
    const { characterName } = req.params;

    const response = await friendsRAGService.chat(
      `Tell me about ${characterName}`,
      []
    );

    res.json({
      success: true,
      message: "Character information retrieved",
      data: {
        character: characterName,
        info: response.response,
        character_data: response.character_info,
        intent: response.intent,
      },
    });
  } catch (error) {
    console.error("Error getting character info:", error);
    res.status(500).json({
      success: false,
      error: "Failed to retrieve character information",
    });
  }
});

/**
 * Episode plot endpoint
 * GET /api/friends/episodes/:episodeId/plot
 */
app.get("/episodes/:episodeId/plot", async (req, res) => {
  try {
    const { episodeId } = req.params;

    const response = await friendsRAGService.chat(
      `What's the plot of ${episodeId}?`,
      []
    );

    res.json({
      success: true,
      message: "Episode plot retrieved",
      data: {
        episode_id: episodeId,
        plot: response.response,
        plot_data: response.episode_plot,
        intent: response.intent,
      },
    });
  } catch (error) {
    console.error("Error getting episode plot:", error);
    res.status(500).json({
      success: false,
      error: "Failed to retrieve episode plot",
    });
  }
});

/**
 * Scene script endpoint
 * POST /api/friends/scenes/search
 */
app.post("/scenes/search", async (req, res) => {
  try {
    const { query } = req.body;

    if (!query || typeof query !== "string") {
      res.status(400).json({
        success: false,
        error: "Query is required and must be a string",
      });
      return;
    }

    const response = await friendsRAGService.chat(
      `Show me a scene about ${query}`,
      []
    );

    res.json({
      success: true,
      message: "Scene script retrieved",
      data: {
        query,
        script: response.response,
        scene_data: response.scene_script,
        intent: response.intent,
      },
    });
  } catch (error) {
    console.error("Error searching scenes:", error);
    res.status(500).json({
      success: false,
      error: "Failed to search scenes",
    });
  }
});

/**
 * Cultural context endpoint
 * POST /api/friends/cultural-context
 */
app.post("/cultural-context", async (req, res) => {
  try {
    const { topic } = req.body;

    if (!topic || typeof topic !== "string") {
      res.status(400).json({
        success: false,
        error: "Topic is required and must be a string",
      });
      return;
    }

    const response = await friendsRAGService.chat(
      `Explain the cultural context of ${topic} from Friends`,
      []
    );

    res.json({
      success: true,
      message: "Cultural context explained",
      data: {
        topic,
        explanation: response.response,
        cultural_data: response.cultural_context,
        intent: response.intent,
      },
    });
  } catch (error) {
    console.error("Error explaining cultural context:", error);
    res.status(500).json({
      success: false,
      error: "Failed to explain cultural context",
    });
  }
});

/**
 * Enhanced cultural context endpoint - New advanced functionality
 * POST /api/friends/cultural-context/enhanced
 */
app.post("/cultural-context/enhanced", async (req, res) => {
  try {
    const { topic, details, originalMessage, episodeContext } = req.body;

    if (!topic || typeof topic !== "string") {
      res.status(400).json({
        success: false,
        error: "Topic is required and must be a string",
      });
      return;
    }

    console.log(`Getting enhanced cultural context for: ${topic}`);

    const explanation = await friendsRAGService.getEnhancedCulturalContext(
      topic,
      details,
      originalMessage,
      episodeContext
    );

    res.json({
      success: true,
      message: "Enhanced cultural context explained successfully",
      data: {
        structured_explanation: explanation,
        practice_suggestions: explanation.practice_tips,
        follow_up_topics: explanation.follow_up_suggestions,
        related_episodes: explanation.related_episodes,
      },
    });
  } catch (error) {
    console.error("Error explaining enhanced cultural context:", error);
    res.status(500).json({
      success: false,
      error: "Failed to explain enhanced cultural context",
      message: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

/**
 * Practice session endpoint
 * POST /api/friends/practice/start
 */
app.post("/practice/start", async (req, res) => {
  try {
    const { character, topic } = req.body;

    if (!character || !topic) {
      res.status(400).json({
        success: false,
        error: "Character and topic are required",
      });
      return;
    }

    const response = await friendsRAGService.chat(
      `Start a practice session with ${character} about ${topic}`,
      []
    );

    res.json({
      success: true,
      message: "Practice session started",
      data: {
        character,
        topic,
        session: response.response,
        practice_data: response.practice_session,
        intent: response.intent,
      },
    });
  } catch (error) {
    console.error("Error starting practice session:", error);
    res.status(500).json({
      success: false,
      error: "Failed to start practice session",
    });
  }
});

/**
 * Practice response evaluation endpoint - New text similarity functionality
 * POST /api/friends/practice/evaluate-response
 */
app.post("/practice/evaluate-response", async (req, res) => {
  try {
    const { userResponse, expectedResponse, sessionId, context } = req.body;

    if (!userResponse || typeof userResponse !== "string") {
      res.status(400).json({
        success: false,
        error: "userResponse is required and must be a string",
      });
      return;
    }

    if (!expectedResponse || typeof expectedResponse !== "string") {
      res.status(400).json({
        success: false,
        error: "expectedResponse is required and must be a string",
      });
      return;
    }

    console.log(`Evaluating practice response - Session: ${sessionId || 'none'}`);
    console.log(`User: "${userResponse}"`);
    console.log(`Expected: "${expectedResponse}"`);

    // Use the new text similarity service
    const result = await friendsRAGService.evaluatePracticeResponse(
      userResponse,
      expectedResponse,
      context
    );

    res.json({
      success: true,
      message: "Response evaluated successfully",
      data: {
        similarity: result.similarity,
        isCorrect: result.isCorrect,
        feedback: result.feedback,
        detailedAnalysis: result.detailedAnalysis,
        sessionId,
        context,
        next_action: result.isCorrect ? "continue" : "retry"
      },
    });
  } catch (error) {
    console.error("Error evaluating practice response:", error);
    res.status(500).json({
      success: false,
      error: "Failed to evaluate practice response",
      message: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

/**
 * Start interactive practice session - New enhanced functionality
 * POST /api/friends/practice/start-interactive
 */
app.post("/practice/start-interactive", async (req, res) => {
  try {
    const { userId, episodeId, character, sceneNumber } = req.body;

    if (!userId || !episodeId || !character) {
      res.status(400).json({
        success: false,
        error: "userId, episodeId, and character are required",
      });
      return;
    }

    console.log(`Starting interactive practice session for ${character} in ${episodeId}`);

    const sessionData = await friendsRAGService.startInteractivePracticeSession(
      userId,
      episodeId,
      character,
      sceneNumber
    );

    res.json({
      success: true,
      message: "Interactive practice session started successfully",
      data: sessionData,
    });
  } catch (error) {
    console.error("Error starting interactive practice session:", error);
    res.status(500).json({
      success: false,
      error: "Failed to start interactive practice session",
      message: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

/**
 * Continue practice session - New enhanced functionality
 * POST /api/friends/practice/continue
 */
app.post("/practice/continue", async (req, res) => {
  try {
    const { sessionId, userResponse } = req.body;

    if (!sessionId || !userResponse) {
      res.status(400).json({
        success: false,
        error: "sessionId and userResponse are required",
      });
      return;
    }

    console.log(`Continuing practice session ${sessionId} with response: "${userResponse}"`);

    const result = await friendsRAGService.continuePracticeSession(
      sessionId,
      userResponse
    );

    res.json({
      success: true,
      message: "Practice session continued successfully",
      data: result,
    });
  } catch (error) {
    console.error("Error continuing practice session:", error);
    res.status(500).json({
      success: false,
      error: "Failed to continue practice session",
      message: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

/**
 * Get practice session status - New enhanced functionality
 * GET /api/friends/practice/session/:sessionId/status
 */
app.get("/practice/session/:sessionId/status", async (req, res) => {
  try {
    const { sessionId } = req.params;

    const sessionStatus = await friendsRAGService.getPracticeSessionStatus(sessionId);

    res.json({
      success: true,
      message: "Practice session status retrieved successfully",
      data: sessionStatus,
    });
  } catch (error) {
    console.error("Error getting practice session status:", error);
    res.status(500).json({
      success: false,
      error: "Failed to get practice session status",
      message: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

/**
 * Complete practice session - New enhanced functionality
 * POST /api/friends/practice/session/:sessionId/complete
 */
app.post("/practice/session/:sessionId/complete", async (req, res) => {
  try {
    const { sessionId } = req.params;

    const completionResult = await friendsRAGService.completePracticeSession(sessionId);

    res.json({
      success: true,
      message: "Practice session completed successfully",
      data: completionResult,
    });
  } catch (error) {
    console.error("Error completing practice session:", error);
    res.status(500).json({
      success: false,
      error: "Failed to complete practice session",
      message: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

/**
 * Search endpoint for general queries
 * POST /api/friends/search
 */
app.post("/search", async (req, res) => {
  try {
    const { query } = req.body;

    if (!query || typeof query !== "string") {
      res.status(400).json({
        success: false,
        error: "Query is required and must be a string",
      });
      return;
    }

    const response = await friendsRAGService.chat(query, []);

    res.json({
      success: true,
      message: "Search completed",
      data: {
        query,
        response: response.response,
        search_results: response.search_results,
        intent: response.intent,
      },
    });
  } catch (error) {
    console.error("Error in search:", error);
    res.status(500).json({
      success: false,
      error: "Failed to process search query",
    });
  }
});

/**
 * Get specific scene by ID
 * GET /api/friends/scenes/:sceneId
 */
app.get("/scenes/:sceneId", async (req, res) => {
  try {
    const { sceneId } = req.params;

    // Get scene directly from Firestore
    const sceneDoc = await admin
      .firestore()
      .collection("friends_scenes")
      .doc(sceneId)
      .get();

    if (!sceneDoc.exists) {
      res.status(404).json({
        success: false,
        error: "Scene not found",
      });
      return;
    }

    const sceneData = sceneDoc.data();

    res.json({
      success: true,
      message: "Scene retrieved successfully",
      data: {
        scene_id: sceneId,
        scene: sceneData,
      },
    });
  } catch (error) {
    console.error("Error getting scene:", error);
    res.status(500).json({
      success: false,
      error: "Failed to retrieve scene",
    });
  }
});

/**
 * Get scenes by episode
 * GET /api/friends/episodes/:episodeId/scenes
 */
app.get("/episodes/:episodeId/scenes", async (req, res) => {
  try {
    const { episodeId } = req.params;
    const { limit = 10, offset = 0 } = req.query;

    // Query scenes for specific episode
    const scenesQuery = admin
      .firestore()
      .collection("friends_scenes")
      .where("episode_id", "==", episodeId)
      .orderBy("scene_number")
      .limit(parseInt(limit as string))
      .offset(parseInt(offset as string));

    const scenesSnapshot = await scenesQuery.get();

    const scenes = scenesSnapshot.docs.map((doc) => ({
      id: doc.id,
      ...doc.data(),
    }));

    res.json({
      success: true,
      message: "Episode scenes retrieved successfully",
      data: {
        episode_id: episodeId,
        scenes,
        total_scenes: scenes.length,
      },
    });
  } catch (error) {
    console.error("Error getting episode scenes:", error);
    res.status(500).json({
      success: false,
      error: "Failed to retrieve episode scenes",
    });
  }
});

/**
 * Get all available episodes
 * GET /api/friends/episodes
 */
app.get("/episodes", async (req, res) => {
  try {
    const { limit = 50 } = req.query;

    // Get unique episodes from plots collection
    const plotsQuery = admin
      .firestore()
      .collection("friends_plots")
      .limit(parseInt(limit as string));

    const plotsSnapshot = await plotsQuery.get();

    const episodes = plotsSnapshot.docs.map((doc) => {
      const data = doc.data();
      return {
        episode_id: doc.id,
        title: data.title || doc.id,
        season: data.season,
        episode_number: data.episode_number,
        plot_summary: data.plot_summary,
        main_themes: data.main_themes,
      };
    });

    res.json({
      success: true,
      message: "Episodes retrieved successfully",
      data: {
        episodes,
        total_episodes: episodes.length,
      },
    });
  } catch (error) {
    console.error("Error getting episodes:", error);
    res.status(500).json({
      success: false,
      error: "Failed to retrieve episodes",
    });
  }
});

/**
 * Data migration endpoint (admin only)
 * POST /api/migrate-data
 */
app.post("/migrate-data", async (_req, res) => {
  try {
    // This would be called from Firebase Functions console or admin interface
    const { FriendsDataMigrator } = await import(
      "./scripts/migrate-to-firestore"
    );
    const migrator = new FriendsDataMigrator();

    await migrator.migrateAllDataReadyFiles();

    res.json({
      success: true,
      message: "Friends data migrated successfully to Firestore",
    });
  } catch (error) {
    console.error("Migration error:", error);
    res.status(500).json({
      success: false,
      error: "Migration failed",
      details: error instanceof Error ? error.message : String(error),
    });
  }
});

/**
 * Database stats endpoint
 * GET /api/stats
 */
app.get("/stats", async (_req, res) => {
  try {
    const db = admin.firestore();

    // Get collection sizes
    const [scenesSnapshot, plotsSnapshot] = await Promise.all([
      db.collection("friends_scenes").count().get(),
      db.collection("friends_plots").count().get(),
    ]);

    const stats = {
      total_scenes: scenesSnapshot.data().count,
      total_plots: plotsSnapshot.data().count,
      collections: [
        "friends_scenes",
        "friends_plots",
        "friends_characters",
        "friends_cultural_contexts",
        "practice_sessions",
      ],
      last_updated: new Date().toISOString(),
    };

    res.json({
      success: true,
      message: "Database statistics retrieved",
      data: stats,
    });
  } catch (error) {
    console.error("Error getting stats:", error);
    res.status(500).json({
      success: false,
      error: "Failed to retrieve database statistics",
    });
  }
});

// Export the API
export const api = functions.https.onRequest(app);
