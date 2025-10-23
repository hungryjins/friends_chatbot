import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";
import * as functions from "firebase-functions";
import * as admin from "firebase-admin";
import {
  IntentAnalysis,
  SearchResult,
  FilterConditions,
  FriendsResponse,
  FriendsCharacter,
  EpisodePlot,
  FriendsScene,
  CulturalContext,
  PracticeSession,
} from "../types/friends";
import { TextSimilarityService, SimilarityResult } from "./text-similarity-service";
import { 
  PracticeSessionManager, 
  PracticeSessionInitResponse,
  PracticeContinueResponse,
  PracticeCompletionResult,
  EnhancedPracticeSession
} from "./practice-session-manager";
import { 
  AdvancedCulturalContextService,
  StructuredCulturalExplanation
} from "./advanced-cultural-context-service";

/**
 * Enhanced Friends RAG Service
 * Converts Python FriendsRAGChatbot to TypeScript with Firebase integration
 */
export class FriendsRAGService {
  private openai: OpenAI;
  private pinecone: Pinecone;
  private index: any;
  private db: admin.firestore.Firestore;
  private textSimilarityService: TextSimilarityService;
  private practiceSessionManager: PracticeSessionManager;
  private culturalContextService: AdvancedCulturalContextService;

  constructor() {
    // Initialize OpenAI
    const openaiKey = functions.config().openai?.api_key;
    if (!openaiKey) {
      throw new Error("OpenAI API key not configured. Run: firebase functions:config:set openai.api_key=\"your-key\"");
    }
    this.openai = new OpenAI({
      apiKey: openaiKey,
    });

    // Initialize Pinecone
    const pineconeKey = functions.config().pinecone?.api_key;
    if (!pineconeKey) {
      throw new Error("Pinecone API key not configured. Run: firebase functions:config:set pinecone.api_key=\"your-key\"");
    }
    this.pinecone = new Pinecone({
      apiKey: pineconeKey,
    });
    this.index = this.pinecone.index("convo");

    // Initialize Firestore
    this.db = admin.firestore();
    
    // Initialize Text Similarity Service
    this.textSimilarityService = new TextSimilarityService(this.openai);
    
    // Initialize Practice Session Manager
    this.practiceSessionManager = new PracticeSessionManager(this.db, this.textSimilarityService);
    
    // Initialize Advanced Cultural Context Service
    this.culturalContextService = new AdvancedCulturalContextService(this.openai);
  }

  /**
   * Main chat method - equivalent to Python's chat method
   */
  async chat(
    userMessage: string,
    chatHistory: string[] = []
  ): Promise<FriendsResponse> {
    try {
      // Step 1: Condense user intent
      const intent = await this.condenseUserIntent(userMessage, chatHistory);

      // Step 2: Route to appropriate function
      const functionName = this.routeToFunction(intent);

      // Step 3: Execute the function and get results
      let response: string;
      let searchResults: SearchResult[] = [];
      let additionalData: any = {};

      switch (functionName) {
        case "recommend_episodes":
          response = await this.recommendEpisodes(intent);
          break;
        case "get_character_info":
          const characterResult = await this.getCharacterInfo(intent);
          response = characterResult.response;
          additionalData.character_info = characterResult.character_info;
          break;
        case "get_episode_plot":
          const plotResult = await this.getEpisodePlot(intent);
          response = plotResult.response;
          additionalData.episode_plot = plotResult.episode_plot;
          break;
        case "get_scene_script":
          const scriptResult = await this.getSceneScript(intent);
          response = scriptResult.response;
          additionalData.scene_script = scriptResult.scene_script;
          break;
        case "explain_cultural_context":
          const cultureResult = await this.explainCulturalContext(intent);
          response = cultureResult.response;
          additionalData.cultural_context = cultureResult.cultural_context;
          break;
        case "start_practice_session":
          const practiceResult = await this.startPracticeSession(intent);
          response = practiceResult.response;
          additionalData.practice_session = practiceResult.practice_session;
          break;
        default:
          // General chat - query Pinecone and generate response
          searchResults = await this.queryPinecone(intent.topic);
          response = await this.generateGeneralResponse(searchResults, intent);
      }

      return {
        response,
        intent,
        search_results: searchResults,
        ...additionalData,
      };
    } catch (error) {
      console.error("Error in chat:", error);
      return {
        response: "I'm sorry, I encountered an error. Please try again.",
        intent: {
          intent: "general_chat",
          topic: userMessage,
          details: "",
          confidence: 0,
        },
      };
    }
  }

  /**
   * Step 1: Condense user intent using OpenAI
   */
  private async condenseUserIntent(
    userMessage: string,
    chatHistory: string[]
  ): Promise<IntentAnalysis> {
    try {
      const historyContext = chatHistory.slice(-5).join(" ");
      const prompt = `
Analyze the user's intent for a Friends TV show English learning chatbot. Be flexible and interpret natural language requests.

User message: "${userMessage}"
Recent conversation: "${historyContext}"

Intent categories:
1. episode_recommendation - Any request for episode suggestions, recommendations, or finding episodes about topics
   Examples: "recommend episodes", "show me episodes about", "find episodes", "which episodes have"

2. character_info - Questions about Friends characters or their personalities, traits, relationships
   Examples: "tell me about Monica", "what's Ross like", "Monica's personality", "character analysis"

3. plot_summary - Requests for episode plots, summaries, or what happens in specific episodes
   Examples: "plot of S01E01", "what happens in", "episode summary", "storyline", "what's about"

4. scene_script - Requests for dialogue, scenes, conversations, script content, or episode scripts
   Examples: "show me a scene", "find dialogue about", "script where they", "conversation between", "episode script", "full script", "season X episode Y script"

5. cultural_context - Questions about American culture, expressions, meanings, or explanations
   Examples: "what does X mean", "explain the culture", "American dating", "cultural reference"

6. practice_session - Requests to practice, learn, or roleplay with characters
   Examples: "practice with", "start session", "roleplay as", "learn conversation"

7. general_chat - General greetings, questions about the bot, or unclear requests

Extract the main topic/keyword and be liberal in interpretation. For ambiguous requests, choose the most likely intent.

Respond with JSON:
{
  "intent": "intent_type",
  "topic": "extracted_main_topic",
  "details": "specific_context_or_parameters",
  "confidence": 0.8
}`;

      const response = await this.openai.chat.completions.create({
        model: "gpt-4o",
        messages: [{ role: "user", content: prompt }],
        response_format: { type: "json_object" },
        temperature: 0.3,
      });

      const content = response.choices[0].message.content || "{}";
      const result = JSON.parse(content);
      return result as IntentAnalysis;
    } catch (error) {
      console.error("Error condensing user intent:", error);
      return {
        intent: "general_chat",
        topic: userMessage,
        details: "",
        confidence: 0,
      };
    }
  }

  /**
   * Step 2: Route to appropriate function based on intent
   */
  private routeToFunction(intent: IntentAnalysis): string {
    const intentMap: Record<string, string> = {
      episode_recommendation: "recommend_episodes",
      character_info: "get_character_info",
      plot_summary: "get_episode_plot",
      scene_script: "get_scene_script",
      cultural_context: "explain_cultural_context",
      practice_session: "start_practice_session",
      general_chat: "general_chat",
    };

    return intentMap[intent.intent] || "general_chat";
  }

  /**
   * Step 3: Query Pinecone for relevant content
   */
  private async queryPinecone(
    query: string,
    filterConditions?: FilterConditions,
    topK: number = 5
  ): Promise<SearchResult[]> {
    try {
      // Generate embedding
      const embeddingResponse = await this.openai.embeddings.create({
        input: [query],
        model: "text-embedding-3-small",
      });

      const embedding = embeddingResponse.data[0].embedding;

      // Query Pinecone
      const queryResponse = await this.index.query({
        vector: embedding,
        topK,
        includeMetadata: true,
        filter: filterConditions,
      });

      const results: SearchResult[] = [];
      for (const match of queryResponse.matches || []) {
        if (match.metadata) {
          results.push({
            text: match.metadata.text as string,
            metadata: match.metadata as any,
            score: match.score || 0,
          });
        }
      }

      return results;
    } catch (error) {
      console.error("Error querying Pinecone:", error);
      return [];
    }
  }

  /**
   * Function 1: Recommend episodes based on topic
   */
  private async recommendEpisodes(intent: IntentAnalysis): Promise<string> {
    try {
      // Query Pinecone for relevant episodes
      const searchResults = await this.queryPinecone(
        intent.topic,
        { chunk_type: "plot" },
        3
      );

      if (searchResults.length === 0) {
        return "I couldn't find specific episodes for that topic. Try asking about relationships, jobs, family, or everyday situations that Friends characters deal with.";
      }

      let response = `🎬 **Episodes about "${intent.topic}":**\n\n`;

      for (let i = 0; i < searchResults.length; i++) {
        const result = searchResults[i];
        const episodeId = result.metadata.episode_id;
        const title = (result.metadata as any).episode_title || episodeId;
        const plotText = (result.metadata as any).plot_text || result.text || "Plot not available";
        const score = Math.round((result.score || 0.8) * 100);
        
        response += `**${i + 1}. ${title}** (${episodeId})\n`;
        response += `📊 Relevance: ${score}%\n`;
        response += `📝 Plot: ${plotText.substring(0, 200)}${plotText.length > 200 ? '...' : ''}\n`;
        response += `💡 Why it matches: Contains themes about ${intent.topic}\n\n`;
      }

      response += "These episodes feature great examples of natural English conversation around this topic! 🗣️";

      return response;
    } catch (error) {
      console.error("Error recommending episodes:", error);
      return "I had trouble finding relevant episodes. Please try asking about a specific topic or theme.";
    }
  }

  /**
   * Function 2: Get character information
   */
  private async getCharacterInfo(intent: IntentAnalysis): Promise<{
    response: string;
    character_info?: FriendsCharacter;
  }> {
    try {
      const characterName = intent.topic.toLowerCase();

      // Get character from Firestore
      const characterDoc = await this.db
        .collection("friends_characters")
        .doc(characterName)
        .get();

      if (!characterDoc.exists) {
        // Provide basic character information as fallback
        const basicCharacterInfo = this.getBasicCharacterInfo(characterName);
        if (basicCharacterInfo) {
          return {
            response: basicCharacterInfo.response,
            character_info: basicCharacterInfo.character,
          };
        }
        
        return {
          response: `I don't have detailed information about ${intent.topic}. The main Friends characters are Monica, Rachel, Ross, Chandler, Joey, and Phoebe.`,
        };
      }

      const character = characterDoc.data() as FriendsCharacter;

      const response = `${character.name} (${character.full_name}):\n\n${
        character.description
      }\n\nPersonality: ${character.personality_traits.join(
        ", "
      )}\n\nKey traits: ${
        character.background
      }\n\nCatchphrases: ${character.catchphrases.join(", ")}`;

      return {
        response,
        character_info: character,
      };
    } catch (error) {
      console.error("Error getting character info:", error);
      return {
        response:
          "I had trouble finding information about that character. Please try asking about Monica, Rachel, Ross, Chandler, Joey, or Phoebe.",
      };
    }
  }

  /**
   * Basic character information fallback
   */
  private getBasicCharacterInfo(characterName: string): { response: string; character: any } | null {
    const characters: Record<string, any> = {
      monica: {
        name: "Monica Geller",
        description: "Monica is a chef who's obsessively clean and competitive. She's Ross's sister and the heart of the group.",
        personality: ["perfectionist", "competitive", "caring", "organized"],
        catchphrases: ["I know!", "That's not how you do it!"],
        background: "Professional chef, lives in a rent-controlled apartment in Manhattan"
      },
      rachel: {
        name: "Rachel Green", 
        description: "Rachel starts as a spoiled rich girl but grows into an independent woman working in fashion.",
        personality: ["fashionable", "sometimes spoiled", "determined", "caring"],
        catchphrases: ["No!", "Oh my God!"],
        background: "Fashion industry, from wealthy family, Ross's on-and-off girlfriend"
      },
      ross: {
        name: "Ross Geller",
        description: "Ross is a paleontologist, Monica's older brother, and Rachel's on-and-off boyfriend.",
        personality: ["intellectual", "nerdy", "romantic", "sometimes jealous"],
        catchphrases: ["We were on a break!", "Unagi"],
        background: "Paleontologist, divorced from Carol, father to Ben and Emma"
      },
      chandler: {
        name: "Chandler Bing",
        description: "Chandler uses humor to deflect emotions and works in statistical analysis and data reconfiguration.",
        personality: ["sarcastic", "funny", "commitment-phobic", "loyal"],
        catchphrases: ["Could this BE any more...?", "I'm not great at the advice thing"],
        background: "Works in data processing, later in advertising, marries Monica"
      },
      joey: {
        name: "Joey Tribbiani",
        description: "Joey is a struggling actor who loves food and women, known for his childlike innocence.",
        personality: ["loyal", "simple", "food-loving", "womanizing"],
        catchphrases: ["How you doin'?", "Joey doesn't share food!"],
        background: "Struggling actor, lives with Chandler, very close to his family"
      },
      phoebe: {
        name: "Phoebe Buffay",
        description: "Phoebe is a quirky massage therapist and musician with an unconventional worldview.",
        personality: ["quirky", "honest", "spiritual", "independent"],
        catchphrases: ["Smelly Cat", "Oh, that's nice"],
        background: "Massage therapist, street musician, had a difficult childhood"
      }
    };

    const char = characters[characterName];
    if (!char) return null;

    const response = `**${char.name}**\n\n${char.description}\n\n**Personality:** ${char.personality.join(", ")}\n\n**Famous quotes:** "${char.catchphrases.join('", "')}"\n\n**Background:** ${char.background}`;

    return {
      response,
      character: {
        id: characterName,
        name: char.name,
        full_name: char.name,
        description: char.description,
        personality_traits: char.personality,
        catchphrases: char.catchphrases,
        background: char.background,
        relationships: {},
        created_at: new Date(),
        updated_at: new Date(),
      }
    };
  }

  /**
   * Function 3: Get episode plot
   */
  private async getEpisodePlot(intent: IntentAnalysis): Promise<{
    response: string;
    episode_plot?: EpisodePlot;
  }> {
    try {
      // Extract episode ID from intent
      const episodeId = this.extractEpisodeId(intent.topic);

      if (!episodeId) {
        return {
          response:
            "Please specify an episode (e.g., 'S01E01' or 'The One Where Monica Gets a New Roommate')",
        };
      }

      // Get plot from Firestore (try both episodeId and episodeId_plot formats)
      let plotDoc = await this.db
        .collection("friends_plots")
        .doc(episodeId)
        .get();
      
      // If not found, try with _plot suffix
      if (!plotDoc.exists) {
        plotDoc = await this.db
          .collection("friends_plots")
          .doc(`${episodeId}_plot`)
          .get();
      }

      if (!plotDoc.exists) {
        return {
          response: `I couldn't find the plot for ${episodeId}. Please check the episode ID and try again.`,
        };
      }

      const plotData = plotDoc.data();
      
      // Handle different data formats - check for plot_text in metadata or text field
      let plotText = "";
      let episodeTitle = "";
      
      if (plotData?.metadata?.plot_text) {
        plotText = plotData.metadata.plot_text;
        episodeTitle = plotData.metadata.episode_title || episodeId;
      } else if (plotData?.text) {
        plotText = plotData.text;
        episodeTitle = plotData.episode_title || episodeId;
      } else if (plotData?.plot_summary) {
        plotText = plotData.plot_summary;
        episodeTitle = plotData.title || episodeId;
      }

      if (!plotText) {
        return {
          response: `I found the episode ${episodeId} but couldn't extract the plot text. The data might be in an unexpected format.`,
        };
      }

      const response = `**${episodeTitle} Plot Summary:**\n\n${plotText}`;

      // Create a compatible plot object
      const plot = {
        id: episodeId,
        episode_id: episodeId,
        plot_summary: plotText,
        title: episodeTitle,
        main_themes: plotData?.main_themes || [],
        key_moments: plotData?.key_moments || [],
        character_development: plotData?.character_development || {},
        created_at: plotData?.created_at,
        updated_at: plotData?.updated_at,
      };

      return {
        response,
        episode_plot: plot,
      };
    } catch (error) {
      console.error("Error getting episode plot:", error);
      return {
        response:
          "I had trouble finding the episode plot. Please try again with a specific episode ID.",
      };
    }
  }

  /**
   * Function 4: Get scene script
   */
  private async getSceneScript(intent: IntentAnalysis): Promise<{
    response: string;
    scene_script?: FriendsScene;
  }> {
    try {
      // First, check if this is a request for a specific episode script
      const requestedEpisodeId = this.extractEpisodeId(intent.topic);
      console.log(`Scene script request - Original query: "${intent.topic}", Extracted episode ID: ${requestedEpisodeId}`);
      
      if (requestedEpisodeId) {
        // This is a request for a specific episode's script
        console.log(`🎯 Detected specific episode script request: ${requestedEpisodeId}`);
        
        // Search specifically for this episode
        const episodeResults = await this.queryPinecone(
          requestedEpisodeId,
          { 
            chunk_type: "scene",
            episode_id: requestedEpisodeId 
          },
          5
        );
        
        console.log(`First search results: ${episodeResults.length} scenes found`);
        
        if (episodeResults.length === 0) {
          console.log(`No results with scene filter, trying without chunk_type filter...`);
          // Try without chunk_type filter
          const alternativeResults = await this.queryPinecone(
            requestedEpisodeId,
            { episode_id: requestedEpisodeId },
            5
          );
          console.log(`Alternative search results: ${alternativeResults.length} items found`);
          
          if (alternativeResults.length === 0) {
            console.log(`No alternative results, trying broad search without filters...`);
            // Try just searching for the episode ID without filters
            const broadResults = await this.queryPinecone(requestedEpisodeId, {}, 5);
            console.log(`Broad search results: ${broadResults.length} items found`);
            
            if (broadResults.length > 0) {
              // Format multiple scenes from the episode
              const episodeScenes = broadResults
                .filter(result => result.metadata.episode_id === requestedEpisodeId)
                .slice(0, 3);
              
              if (episodeScenes.length > 0) {
                console.log(`Processing ${episodeScenes.length} episode scenes. First result structure:`, JSON.stringify(episodeScenes[0], null, 2));
                
                // Try to fetch scene content from Firestore if Pinecone text is empty
                const sceneContents = await Promise.all(
                  episodeScenes.map(async (result, index) => {
                    const sceneNumber = (result.metadata as any).scene_number || index + 1;
                    let text = result.text || (result as any).content || (result as any).script || (result as any).dialogue || "";
                    
                    // If no text in Pinecone, try to fetch from Firestore
                    if (!text && (result.metadata as any).firestore_path) {
                      try {
                        console.log(`Fetching scene content from Firestore: ${(result.metadata as any).firestore_path}`);
                        const sceneDoc = await this.db.doc((result.metadata as any).firestore_path).get();
                        if (sceneDoc.exists) {
                          const sceneData = sceneDoc.data();
                          text = sceneData?.raw_text || sceneData?.text || sceneData?.script || sceneData?.dialogue || "";
                          console.log(`Fetched scene data from Firestore. Text length: ${text.length}`);
                        }
                      } catch (error) {
                        console.error(`Failed to fetch scene from Firestore:`, error);
                      }
                    }
                    
                    // If still no text, try direct scene ID lookup
                    if (!text && (result.metadata as any).scene_id) {
                      try {
                        console.log(`Trying direct scene lookup: ${(result.metadata as any).scene_id}`);
                        const sceneDoc = await this.db.collection("friends_scenes").doc((result.metadata as any).scene_id).get();
                        if (sceneDoc.exists) {
                          const sceneData = sceneDoc.data();
                          text = sceneData?.raw_text || sceneData?.text || sceneData?.script || sceneData?.dialogue || "";
                          console.log(`Direct scene lookup - Text length: ${text.length}`);
                        }
                      } catch (error) {
                        console.error(`Failed direct scene lookup:`, error);
                      }
                    }
                    
                    if (!text) {
                      console.warn(`Scene ${index + 1} (${sceneNumber}) has no text content after all attempts. Available properties:`, Object.keys(result));
                      return `**Scene ${sceneNumber}:** [Script content not available]`;
                    }
                    return `**Scene ${sceneNumber}:**\n${text.substring(0, 600)}${text.length > 600 ? '...' : ''}`;
                  })
                );
                
                const combinedScript = sceneContents.join('\n\n---\n\n');
                
                const response = `🎬 **${requestedEpisodeId} Script (Partial):**\n\n${combinedScript}\n\n*Note: This is a partial script. The full episode contains more scenes.*`;
                
                return {
                  response,
                  scene_script: {
                    id: `${requestedEpisodeId}_combined`,
                    episode_id: requestedEpisodeId,
                    season: broadResults[0].metadata.season,
                    episode_number: broadResults[0].metadata.episode_number,
                    scene_number: 1,
                    characters: [...new Set(broadResults.map(r => (r.metadata as any).character).filter(Boolean))],
                    raw_text: combinedScript,
                    lines: [],
                    location: "Multiple locations",
                    scene_description: `Combined scenes from ${requestedEpisodeId}`,
                    firestore_path: "",
                    created_at: admin.firestore.Timestamp.now(),
                    updated_at: admin.firestore.Timestamp.now(),
                  },
                };
              }
            }
            
            return {
              response: `I couldn't find script data for ${requestedEpisodeId}. The episode might not be in our database yet, or it might be stored under a different format. Try asking for specific scenes or characters from this episode.`,
            };
          }
          
          // Use the alternative results
          const result = alternativeResults[0];
          console.log(`Alternative result structure:`, JSON.stringify(result, null, 2));
          let sceneText = result.text || (result as any).content || (result as any).script || (result as any).dialogue || "";
          
          // If no text in Pinecone, try to fetch from Firestore
          if (!sceneText && (result.metadata as any).firestore_path) {
            try {
              console.log(`Fetching alternative scene content from Firestore: ${(result.metadata as any).firestore_path}`);
              const sceneDoc = await this.db.doc((result.metadata as any).firestore_path).get();
              if (sceneDoc.exists) {
                const sceneData = sceneDoc.data();
                sceneText = sceneData?.raw_text || sceneData?.text || sceneData?.script || sceneData?.dialogue || "";
                console.log(`Fetched alternative scene from Firestore. Text length: ${sceneText.length}`);
              }
            } catch (error) {
              console.error(`Failed to fetch alternative scene from Firestore:`, error);
            }
          }
          
          // If still no text, try direct scene ID lookup
          if (!sceneText && (result.metadata as any).scene_id) {
            try {
              console.log(`Trying alternative direct scene lookup: ${(result.metadata as any).scene_id}`);
              const sceneDoc = await this.db.collection("friends_scenes").doc((result.metadata as any).scene_id).get();
              if (sceneDoc.exists) {
                const sceneData = sceneDoc.data();
                sceneText = sceneData?.raw_text || sceneData?.text || sceneData?.script || sceneData?.dialogue || "";
                console.log(`Alternative direct scene lookup - Text length: ${sceneText.length}`);
              }
            } catch (error) {
              console.error(`Failed alternative direct scene lookup:`, error);
            }
          }
          
          if (!sceneText) {
            console.warn(`Alternative result has no text content after all attempts. Available properties:`, Object.keys(result));
            return {
              response: `Found ${requestedEpisodeId} but the script content is not available in the expected format.`,
            };
          }
          const score = Math.round((result.score || 0.8) * 100);

          const response = `🎬 **${requestedEpisodeId} Script:**\n\n**Relevance:** ${score}%\n\n**Script:**\n${sceneText.substring(0, 1000)}${sceneText.length > 1000 ? '...\n\n[Episode continues...]' : ''}`;

          return {
            response,
            scene_script: {
              id: (result.metadata as any).scene_id || `${requestedEpisodeId}_scene`,
              episode_id: requestedEpisodeId,
              season: result.metadata.season,
              episode_number: result.metadata.episode_number,
              scene_number: (result.metadata as any).scene_number || 1,
              characters: [(result.metadata as any).character || "Friends"],
              raw_text: sceneText,
              lines: [],
              location: (result.metadata as any).location || "Unknown",
              scene_description: (result.metadata as any).scene_description || "",
              firestore_path: "",
              created_at: admin.firestore.Timestamp.now(),
              updated_at: admin.firestore.Timestamp.now(),
            },
          };
        }
        
        // Found scenes for the specific episode
        if (episodeResults.length === 1) {
          const result = episodeResults[0];
          console.log(`Single episode result structure:`, JSON.stringify(result, null, 2));
          let sceneText = result.text || (result as any).content || (result as any).script || (result as any).dialogue || "";
          
          // If no text in Pinecone, try to fetch from Firestore
          if (!sceneText && (result.metadata as any).firestore_path) {
            try {
              console.log(`Fetching single scene content from Firestore: ${(result.metadata as any).firestore_path}`);
              const sceneDoc = await this.db.doc((result.metadata as any).firestore_path).get();
              if (sceneDoc.exists) {
                const sceneData = sceneDoc.data();
                sceneText = sceneData?.raw_text || sceneData?.text || sceneData?.script || sceneData?.dialogue || "";
                console.log(`Fetched single scene from Firestore. Text length: ${sceneText.length}`);
              }
            } catch (error) {
              console.error(`Failed to fetch single scene from Firestore:`, error);
            }
          }
          
          // If still no text, try direct scene ID lookup
          if (!sceneText && (result.metadata as any).scene_id) {
            try {
              console.log(`Trying single direct scene lookup: ${(result.metadata as any).scene_id}`);
              const sceneDoc = await this.db.collection("friends_scenes").doc((result.metadata as any).scene_id).get();
              if (sceneDoc.exists) {
                const sceneData = sceneDoc.data();
                sceneText = sceneData?.raw_text || sceneData?.text || sceneData?.script || sceneData?.dialogue || "";
                console.log(`Single direct scene lookup - Text length: ${sceneText.length}`);
              }
            } catch (error) {
              console.error(`Failed single direct scene lookup:`, error);
            }
          }
          
          if (!sceneText) {
            console.warn(`Single result has no text content after all attempts. Available properties:`, Object.keys(result));
            return {
              response: `Found ${requestedEpisodeId} but the script content is not available in the expected format.`,
            };
          }
          const score = Math.round((result.score || 0.9) * 100);

          const response = `🎬 **${requestedEpisodeId} Script:**\n\n**Relevance:** ${score}%\n\n**Script:**\n${sceneText.substring(0, 1000)}${sceneText.length > 1000 ? '...\n\n[Episode continues...]' : ''}`;

          return {
            response,
            scene_script: {
              id: (result.metadata as any).scene_id || `${requestedEpisodeId}_scene`,
              episode_id: requestedEpisodeId,
              season: result.metadata.season,
              episode_number: result.metadata.episode_number,
              scene_number: (result.metadata as any).scene_number || 1,
              characters: [(result.metadata as any).character || "Friends"],
              raw_text: sceneText,
              lines: [],
              location: (result.metadata as any).location || "Unknown",
              scene_description: (result.metadata as any).scene_description || "",
              firestore_path: "",
              created_at: admin.firestore.Timestamp.now(),
              updated_at: admin.firestore.Timestamp.now(),
            },
          };
        } else {
          // Multiple scenes found - combine them
          console.log(`Processing ${episodeResults.length} episode results. First result structure:`, JSON.stringify(episodeResults[0], null, 2));
          
          // Use Promise.all to handle async Firestore fetches properly
          const sceneContents = await Promise.all(
            episodeResults.slice(0, 3).map(async (result, index) => {
              const sceneNumber = (result.metadata as any).scene_number || index + 1;
              let text = result.text || (result as any).content || (result as any).script || (result as any).dialogue || "";
              console.log(`Episode result ${index + 1} - text length: ${text ? text.length : 'null/undefined'}, type: ${typeof text}`);
              
              // If no text in Pinecone, try to fetch from Firestore
              if (!text && (result.metadata as any).firestore_path) {
                try {
                  console.log(`Fetching episode scene ${index + 1} content from Firestore: ${(result.metadata as any).firestore_path}`);
                  const sceneDoc = await this.db.doc((result.metadata as any).firestore_path).get();
                  if (sceneDoc.exists) {
                    const sceneData = sceneDoc.data();
                    text = sceneData?.raw_text || sceneData?.text || sceneData?.script || sceneData?.dialogue || "";
                    console.log(`Fetched episode scene ${index + 1} from Firestore. Text length: ${text.length}`);
                  } else {
                    console.log(`Firestore document not found: ${(result.metadata as any).firestore_path}`);
                  }
                } catch (error) {
                  console.error(`Failed to fetch episode scene ${index + 1} from Firestore:`, error);
                }
              }
              
              // If still no text, try direct scene ID lookup
              if (!text && (result.metadata as any).scene_id) {
                try {
                  console.log(`Trying episode direct scene lookup ${index + 1}: ${(result.metadata as any).scene_id}`);
                  const sceneDoc = await this.db.collection("friends_scenes").doc((result.metadata as any).scene_id).get();
                  if (sceneDoc.exists) {
                    const sceneData = sceneDoc.data();
                    text = sceneData?.raw_text || sceneData?.text || sceneData?.script || sceneData?.dialogue || "";
                    console.log(`Episode direct scene lookup ${index + 1} - Text length: ${text.length}`);
                  } else {
                    console.log(`Scene document not found: ${(result.metadata as any).scene_id}`);
                  }
                } catch (error) {
                  console.error(`Failed episode direct scene lookup ${index + 1}:`, error);
                }
              }
              
              if (!text) {
                console.warn(`Episode result ${index + 1} (${sceneNumber}) has no text content after all attempts. Available properties:`, Object.keys(result));
                return `**Scene ${sceneNumber}:** [Script content not available]`;
              }
              return `**Scene ${sceneNumber}:**\n${text.substring(0, 600)}${text.length > 600 ? '...' : ''}`;
            })
          );
          
          const combinedScript = sceneContents.join('\n\n---\n\n');
          
          const response = `🎬 **${requestedEpisodeId} Script (Multiple Scenes):**\n\n${combinedScript}\n\n*Note: This episode contains ${episodeResults.length} scenes. Showing first 3.*`;
          
          return {
            response,
            scene_script: {
              id: `${requestedEpisodeId}_combined`,
              episode_id: requestedEpisodeId,
              season: episodeResults[0].metadata.season,
              episode_number: episodeResults[0].metadata.episode_number,
              scene_number: 1,
              characters: [...new Set(episodeResults.map(r => (r.metadata as any).character).filter(Boolean))],
              raw_text: combinedScript,
              lines: [],
              location: "Multiple locations",
              scene_description: `Combined scenes from ${requestedEpisodeId}`,
              firestore_path: "",
              created_at: admin.firestore.Timestamp.now(),
              updated_at: admin.firestore.Timestamp.now(),
            },
          };
        }
      }
      
      // General scene search (not a specific episode request)
      const searchResults = await this.queryPinecone(
        intent.topic,
        { chunk_type: "scene" },
        1
      );

      if (searchResults.length === 0) {
        // Try alternative search without filters
        const alternativeResults = await this.queryPinecone(intent.topic, {}, 3);
        
        if (alternativeResults.length === 0) {
          return {
            response: `I couldn't find scenes about "${intent.topic}". Try asking about:\n• Coffee shop scenes\n• Apartment conversations\n• Work situations\n• Relationships\n• Specific characters like Monica, Rachel, Ross, Chandler, Joey, or Phoebe\n• Specific episodes like "S04E03 script" or "episode 3 season 4 script"`,
          };
        }
        
        // Use direct text from search results
        const result = alternativeResults[0];
        console.log(`General search result structure:`, JSON.stringify(result, null, 2));
        const sceneText = result.text || (result as any).content || (result as any).script || (result as any).dialogue || "";
        if (!sceneText) {
          console.warn(`General search result has no text content. Available properties:`, Object.keys(result));
          return {
            response: `Found a scene but the content is not available in the expected format.`,
          };
        }
        const episodeId = result.metadata.episode_id;
        const score = Math.round((result.score || 0.7) * 100);

        const response = `🎬 **Found a relevant scene:**\n\n**Episode:** ${episodeId}\n**Relevance:** ${score}%\n\n**Script:**\n${sceneText.substring(0, 800)}${sceneText.length > 800 ? '...\n\n[Scene continues...]' : ''}`;

        return {
          response,
          scene_script: {
            id: (result.metadata as any).scene_id || `${episodeId}_scene`,
            episode_id: episodeId,
            season: result.metadata.season,
            episode_number: result.metadata.episode_number,
            scene_number: (result.metadata as any).scene_number || 1,
            characters: [(result.metadata as any).character || "Friends"],
            raw_text: sceneText,
            lines: [],
            location: (result.metadata as any).location || "Unknown",
            scene_description: (result.metadata as any).scene_description || "",
            firestore_path: "",
            created_at: admin.firestore.Timestamp.now(),
            updated_at: admin.firestore.Timestamp.now(),
          },
        };
      }

      // Use direct text from Pinecone results instead of Firestore lookup
      const result = searchResults[0];
      console.log(`Direct search result structure:`, JSON.stringify(result, null, 2));
      const sceneText = result.text || (result as any).content || (result as any).script || (result as any).dialogue || "";
      if (!sceneText) {
        console.warn(`Direct search result has no text content. Available properties:`, Object.keys(result));
        return {
          response: `Found a scene but the content is not available in the expected format.`,
        };
      }
      const episodeId = result.metadata.episode_id;
      const score = Math.round((result.score || 0.8) * 100);

      const response = `🎬 **Scene from ${episodeId}:**\n\n**Relevance:** ${score}%\n\n**Script:**\n${sceneText.substring(0, 800)}${sceneText.length > 800 ? '...\n\n[Scene continues...]' : ''}`;

      return {
        response,
        scene_script: {
          id: (result.metadata as any).scene_id || `${episodeId}_scene`,
          episode_id: episodeId,
          season: result.metadata.season,
          episode_number: result.metadata.episode_number,
          scene_number: (result.metadata as any).scene_number || 1,
          characters: [(result.metadata as any).character || "Friends"],
          raw_text: sceneText,
          lines: [],
          location: (result.metadata as any).location || "Unknown",
          scene_description: (result.metadata as any).scene_description || "",
          firestore_path: "",
          created_at: admin.firestore.Timestamp.now(),
          updated_at: admin.firestore.Timestamp.now(),
        },
      };
    } catch (error) {
      console.error("Error getting scene script:", error);
      return {
        response: "I had trouble finding the scene script. Please try again.",
      };
    }
  }

  /**
   * Function 5: Explain cultural context
   */
  private async explainCulturalContext(intent: IntentAnalysis): Promise<{
    response: string;
    cultural_context?: CulturalContext;
  }> {
    try {
      // Get cultural context from Firestore
      const contextDoc = await this.db
        .collection("friends_cultural_contexts")
        .doc(intent.topic.toLowerCase().replace(/\s+/g, "_"))
        .get();

      if (!contextDoc.exists) {
        // Fallback to general explanation
        return {
          response: `I can help explain American cultural references from Friends! The show often features 1990s American culture, dating customs, workplace dynamics, and New York City life. What specific cultural aspect would you like to learn about?`,
        };
      }

      const context = contextDoc.data() as CulturalContext;

      const response = `**Cultural Context: ${context.topic}**\n\n${
        context.explanation
      }\n\n**Examples from Friends:**\n${context.examples
        .map((example, i) => `${i + 1}. ${example}`)
        .join("\n")}\n\n**Related Episodes:** ${context.related_episodes.join(
        ", "
      )}`;

      return {
        response,
        cultural_context: context,
      };
    } catch (error) {
      console.error("Error explaining cultural context:", error);
      return {
        response:
          "I had trouble finding cultural context information. Please try asking about a specific American cultural reference from Friends.",
      };
    }
  }

  /**
   * Function 6: Start practice session
   */
  private async startPracticeSession(intent: IntentAnalysis): Promise<{
    response: string;
    practice_session?: PracticeSession;
  }> {
    try {
      // Determine character for practice
      const character = this.extractCharacter(intent.topic) || "Chandler";

      // Create practice session
      const sessionId = `session_${Date.now()}`;
      const practiceSession: PracticeSession = {
        session_id: sessionId,
        character,
        scenario: `Practice conversation with ${character} about ${intent.topic}`,
        target_expressions: [],
        score: 0,
        created_at: admin.firestore.Timestamp.now(),
        updated_at: admin.firestore.Timestamp.now(),
      };

      // Save to Firestore
      await this.db
        .collection("practice_sessions")
        .doc(sessionId)
        .set(practiceSession);

      const response = `Great! I've started a practice session with ${character}. You can now have a conversation with this Friends character about "${intent.topic}". Try to use natural English expressions and I'll help you improve!`;

      return {
        response,
        practice_session: practiceSession,
      };
    } catch (error) {
      console.error("Error starting practice session:", error);
      return {
        response:
          "I had trouble starting the practice session. Please try again.",
      };
    }
  }

  /**
   * Generate general response when no specific function is needed
   */
  private async generateGeneralResponse(
    searchResults: SearchResult[],
    intent: IntentAnalysis
  ): Promise<string> {
    try {
      if (searchResults.length === 0) {
        return "I'm here to help you practice English with Friends! Ask me about episodes, characters, cultural references, or start a conversation practice session.";
      }

      const bestMatch = searchResults[0];
      const context = `Based on Friends content: ${bestMatch.text}`;

      const prompt = `You are a helpful English learning assistant using Friends TV show content. 

User asked: "${intent.topic}"
Friends context: "${context}"

Provide a helpful, educational response that:
1. References the Friends content naturally
2. Helps with English learning
3. Encourages further practice
4. Is conversational and friendly

Keep it concise and engaging.`;

      const response = await this.openai.chat.completions.create({
        model: "gpt-4o",
        messages: [{ role: "user", content: prompt }],
        temperature: 0.7,
      });

      return (
        response.choices[0].message.content ||
        "I found some relevant Friends content but couldn't generate a response. Please try again."
      );
    } catch (error) {
      console.error("Error generating general response:", error);
      return "I'm here to help you practice English with Friends! What would you like to explore?";
    }
  }

  /**
   * Helper: Extract episode ID from text
   */
  private extractEpisodeId(text: string): string | null {
    // Standard S##E## format
    const episodeMatch = text.match(/S\d{2}E\d{2}/i);
    if (episodeMatch) {
      return episodeMatch[0].toUpperCase();
    }

    // Try to match "season X episode Y" or "episode Y season X" patterns
    const seasonEpisodeMatch = text.match(/(?:season\s*(\d+)[^\d]*episode\s*(\d+)|episode\s*(\d+)[^\d]*season\s*(\d+))/i);
    if (seasonEpisodeMatch) {
      let season, episode;
      if (seasonEpisodeMatch[1] && seasonEpisodeMatch[2]) {
        season = seasonEpisodeMatch[1];
        episode = seasonEpisodeMatch[2];
      } else if (seasonEpisodeMatch[3] && seasonEpisodeMatch[4]) {
        episode = seasonEpisodeMatch[3];
        season = seasonEpisodeMatch[4];
      }
      
      if (season && episode) {
        const seasonPadded = season.padStart(2, '0');
        const episodePadded = episode.padStart(2, '0');
        return `S${seasonPadded}E${episodePadded}`;
      }
    }

    // Try to match "season X" only (get first episode)
    const seasonMatch = text.match(/season\s*(\d+)/i);
    if (seasonMatch) {
      const season = seasonMatch[1].padStart(2, '0');
      return `S${season}E01`;
    }

    // Try to match common episode title patterns
    const titleMatch = text.match(
      /(?:The One Where|The One With|The One About)/i
    );
    if (titleMatch) {
      // This would need more sophisticated matching in a real implementation
      return null;
    }

    return null;
  }

  /**
   * Helper: Extract character name from text
   */
  private extractCharacter(text: string): string | null {
    const characters = [
      "monica",
      "rachel",
      "ross",
      "chandler",
      "joey",
      "phoebe",
    ];
    const lowerText = text.toLowerCase();

    for (const character of characters) {
      if (lowerText.includes(character)) {
        return character.charAt(0).toUpperCase() + character.slice(1);
      }
    }

    return null;
  }

  /**
   * Evaluate practice response using text similarity - New enhanced functionality
   */
  async evaluatePracticeResponse(
    userResponse: string,
    expectedResponse: string,
    context?: {
      episode_id: string;
      scene_number: number;
      character: string;
    }
  ): Promise<SimilarityResult> {
    try {
      console.log(`Evaluating practice response for ${context?.character || 'unknown'}`);
      console.log(`User: "${userResponse}"`);
      console.log(`Expected: "${expectedResponse}"`);
      
      const result = await this.textSimilarityService.calculateSimilarity(
        userResponse,
        expectedResponse
      );
      
      console.log(`Similarity result: ${result.similarity}, Correct: ${result.isCorrect}`);
      
      return result;
    } catch (error) {
      console.error("Error evaluating practice response:", error);
      return {
        similarity: 0,
        isCorrect: false,
        feedback: "❌ Error evaluating response. Please try again.",
        detailedAnalysis: {
          wordSimilarity: 0,
          characterSimilarity: 0,
          exactMatch: false,
          veryCloseMatch: false
        }
      };
    }
  }

  /**
   * Start interactive practice session - New enhanced functionality
   */
  async startInteractivePracticeSession(
    userId: string,
    episodeId: string,
    character: string,
    sceneNumber?: number
  ): Promise<PracticeSessionInitResponse> {
    return this.practiceSessionManager.startInteractiveSession(
      userId,
      episodeId,
      character,
      sceneNumber
    );
  }

  /**
   * Continue practice session - New enhanced functionality
   */
  async continuePracticeSession(
    sessionId: string,
    userResponse: string
  ): Promise<PracticeContinueResponse> {
    return this.practiceSessionManager.continuePractice(sessionId, userResponse);
  }

  /**
   * Get practice session status - New enhanced functionality
   */
  async getPracticeSessionStatus(sessionId: string): Promise<EnhancedPracticeSession> {
    return this.practiceSessionManager.getSessionStatus(sessionId);
  }

  /**
   * Complete practice session - New enhanced functionality
   */
  async completePracticeSession(sessionId: string): Promise<PracticeCompletionResult> {
    return this.practiceSessionManager.completeSession(sessionId);
  }

  /**
   * Get enhanced cultural context explanation - New advanced functionality
   */
  async getEnhancedCulturalContext(
    topic: string,
    details?: string,
    originalMessage?: string,
    episodeContext?: string
  ): Promise<StructuredCulturalExplanation> {
    try {
      console.log(`Getting enhanced cultural context for: ${topic}`);
      
      const explanation = await this.culturalContextService.generateStructuredExplanation(
        topic,
        details || "",
        originalMessage || `Explain the cultural context of ${topic}`,
        episodeContext
      );
      
      console.log(`Generated cultural explanation for: ${explanation.expression}`);
      
      return explanation;
      
    } catch (error) {
      console.error("Error getting enhanced cultural context:", error);
      throw error;
    }
  }

}
