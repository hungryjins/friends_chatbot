import * as admin from "firebase-admin";
import { TextSimilarityService, SimilarityResult } from "./text-similarity-service";

/**
 * Dialogue line structure for practice sessions
 */
export interface DialogueLine {
  line_number: number;
  type: 'dialogue' | 'action' | 'narration';
  speaker?: string;
  text: string;
  is_user_line: boolean;  // Whether this is a line the user should practice
}

/**
 * Practice progress tracking for each user response
 */
export interface PracticeProgress {
  line_number: number;
  expected_text: string;
  user_response: string;
  similarity_score: number;
  is_correct: boolean;
  feedback: string;
  attempt_number: number;
  timestamp: admin.firestore.Timestamp;
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
  
  // Progress state
  current_line_index: number;
  total_lines: number;
  dialogue_lines: DialogueLine[];
  
  // Score tracking
  correct_answers: number;
  total_attempts: number;
  session_progress: PracticeProgress[];
  
  // Session state
  status: 'active' | 'completed' | 'paused';
  started_at: admin.firestore.Timestamp;
  completed_at?: admin.firestore.Timestamp;
  created_at: admin.firestore.Timestamp;
  updated_at: admin.firestore.Timestamp;
}

/**
 * Response for practice session initialization
 */
export interface PracticeSessionInitResponse {
  session_id: string;
  scene_context: {
    location: string;
    characters: string[];
    description: string;
  };
  total_lines: number;
  first_context: string;
  user_turn: {
    expected: string;
    speaker: string;
    line_number: number;
  };
}

/**
 * Response for practice continuation
 */
export interface PracticeContinueResponse {
  evaluation: SimilarityResult;
  next_context?: string;
  next_user_turn?: {
    expected: string;
    speaker: string;
    line_number: number;
  };
  session_status: {
    current_line: number;
    total_lines: number;
    correct_answers: number;
    accuracy: number;
  };
  is_session_complete: boolean;
}

/**
 * Practice completion result
 */
export interface PracticeCompletionResult {
  session_id: string;
  final_score: number;
  accuracy: number;
  total_lines_practiced: number;
  correct_answers: number;
  session_duration_minutes: number;
  completed_at: admin.firestore.Timestamp;
}

/**
 * Practice Session Manager
 * Ports the interactive practice session functionality from local Python implementation
 */
export class PracticeSessionManager {
  private db: admin.firestore.Firestore;
  private textSimilarityService: TextSimilarityService;

  constructor(
    db: admin.firestore.Firestore,
    textSimilarityService: TextSimilarityService
  ) {
    this.db = db;
    this.textSimilarityService = textSimilarityService;
  }

  /**
   * Start interactive practice session - ports run_practice_session() from Python
   */
  async startInteractiveSession(
    userId: string,
    episodeId: string, 
    character: string,
    sceneNumber?: number
  ): Promise<PracticeSessionInitResponse> {
    try {
      // Find the scene for practice
      const sceneData = await this.findPracticeScene(episodeId, character, sceneNumber);
      if (!sceneData) {
        throw new Error(`No suitable scene found for ${character} in ${episodeId}`);
      }

      // Parse scene dialogue
      const dialogueLines = this.parseSceneDialogue(sceneData.text || sceneData.raw_text);
      const userLines = this.filterUserLines(dialogueLines, character);

      if (userLines.length === 0) {
        throw new Error(`No dialogue lines found for ${character} in this scene`);
      }

      // Create practice session
      const sessionId = `session_${Date.now()}_${userId}`;
      const session: EnhancedPracticeSession = {
        session_id: sessionId,
        user_id: userId,
        episode_id: episodeId,
        character: character,
        scene_number: sceneData.metadata?.scene_number || sceneData.scene_number,
        scene_id: sceneData.id,
        current_line_index: 0,
        total_lines: userLines.length,
        dialogue_lines: dialogueLines,
        correct_answers: 0,
        total_attempts: 0,
        session_progress: [],
        status: 'active',
        started_at: admin.firestore.Timestamp.now(),
        created_at: admin.firestore.Timestamp.now(),
        updated_at: admin.firestore.Timestamp.now(),
      };

      // Save session to Firestore
      await this.db.collection("practice_sessions").doc(sessionId).set({
        ...session,
        created_at: admin.firestore.Timestamp.now(),
        updated_at: admin.firestore.Timestamp.now()
      });

      // Prepare initial response
      const firstUserLine = userLines[0];
      const contextLines = this.getContextBefore(dialogueLines, firstUserLine.line_number);

      return {
        session_id: sessionId,
        scene_context: {
          location: sceneData.location || "Unknown location",
          characters: sceneData.characters || [character],
          description: sceneData.scene_description || "Practice scene"
        },
        total_lines: userLines.length,
        first_context: contextLines,
        user_turn: {
          expected: firstUserLine.text,
          speaker: character,
          line_number: firstUserLine.line_number
        }
      };

    } catch (error) {
      console.error("Error starting interactive session:", error);
      throw error;
    }
  }

  /**
   * Continue practice session - ports sequential dialogue progression from Python
   */
  async continuePractice(
    sessionId: string,
    userResponse: string
  ): Promise<PracticeContinueResponse> {
    try {
      // Get current session
      const sessionDoc = await this.db.collection("practice_sessions").doc(sessionId).get();
      if (!sessionDoc.exists) {
        throw new Error("Practice session not found");
      }

      const session = sessionDoc.data() as EnhancedPracticeSession;
      if (session.status !== 'active') {
        throw new Error("Practice session is not active");
      }

      // Get current user line to practice
      const userLines = this.filterUserLines(session.dialogue_lines, session.character);
      const currentUserLine = userLines[session.current_line_index];

      if (!currentUserLine) {
        throw new Error("No more lines to practice");
      }

      // Evaluate user response
      const evaluation = await this.textSimilarityService.calculateSimilarity(
        userResponse,
        currentUserLine.text
      );

      // Update session progress
      const progress: PracticeProgress = {
        line_number: currentUserLine.line_number,
        expected_text: currentUserLine.text,
        user_response: userResponse,
        similarity_score: evaluation.similarity,
        is_correct: evaluation.isCorrect,
        feedback: evaluation.feedback,
        attempt_number: session.total_attempts + 1,
        timestamp: admin.firestore.Timestamp.now()
      };

      session.session_progress.push(progress);
      session.total_attempts += 1;
      if (evaluation.isCorrect) {
        session.correct_answers += 1;
      }
      
      // Always move to next line after any attempt (correct or incorrect)
      session.current_line_index += 1;
      session.updated_at = admin.firestore.Timestamp.now();

      // Check if session is complete
      const isComplete = session.current_line_index >= userLines.length;
      if (isComplete) {
        session.status = 'completed';
        session.completed_at = admin.firestore.Timestamp.now();
      }

      // Prepare response for next line
      let nextContext: string | undefined;
      let nextUserTurn: { expected: string; speaker: string; line_number: number } | undefined;

      if (!isComplete) {
        const nextUserLine = userLines[session.current_line_index];
        if (nextUserLine) {
          nextContext = this.getContextBefore(session.dialogue_lines, nextUserLine.line_number);
          nextUserTurn = {
            expected: nextUserLine.text,
            speaker: session.character,
            line_number: nextUserLine.line_number
          };
        }
      }

      // Update session in Firestore
      await this.db.collection("practice_sessions").doc(sessionId).update({
        ...session,
        updated_at: admin.firestore.Timestamp.now()
      });

      const accuracy = session.total_attempts > 0 ? 
        (session.correct_answers / session.total_attempts) * 100 : 0;

      return {
        evaluation,
        next_context: nextContext,
        next_user_turn: nextUserTurn,
        session_status: {
          current_line: session.current_line_index + 1,
          total_lines: userLines.length,
          correct_answers: session.correct_answers,
          accuracy: Math.round(accuracy * 10) / 10
        },
        is_session_complete: isComplete
      };

    } catch (error) {
      console.error("Error continuing practice:", error);
      throw error;
    }
  }

  /**
   * Get session status
   */
  async getSessionStatus(sessionId: string): Promise<EnhancedPracticeSession> {
    const sessionDoc = await this.db.collection("practice_sessions").doc(sessionId).get();
    if (!sessionDoc.exists) {
      throw new Error("Practice session not found");
    }
    return sessionDoc.data() as EnhancedPracticeSession;
  }

  /**
   * Complete practice session
   */
  async completeSession(sessionId: string): Promise<PracticeCompletionResult> {
    const session = await this.getSessionStatus(sessionId);
    
    if (session.status === 'completed') {
      // Calculate final metrics
      const durationMs = session.completed_at!.toMillis() - session.started_at.toMillis();
      const durationMinutes = Math.round(durationMs / (1000 * 60) * 10) / 10;
      const accuracy = session.total_attempts > 0 ? 
        (session.correct_answers / session.total_attempts) * 100 : 0;

      return {
        session_id: sessionId,
        final_score: Math.round(accuracy),
        accuracy: Math.round(accuracy * 10) / 10,
        total_lines_practiced: session.total_attempts,
        correct_answers: session.correct_answers,
        session_duration_minutes: durationMinutes,
        completed_at: session.completed_at!
      };
    } else {
      throw new Error("Session is not yet completed");
    }
  }

  /**
   * Parse scene dialogue - ports parse_scene_dialogue() from Python
   */
  private parseSceneDialogue(sceneText: string): DialogueLine[] {
    const lines: DialogueLine[] = [];
    const textLines = sceneText.split('\n');
    let lineNumber = 1;

    for (const line of textLines) {
      const trimmedLine = line.trim();
      if (!trimmedLine) continue;

      // Detect different line types
      if (trimmedLine.startsWith('[') && trimmedLine.endsWith(']')) {
        // Stage direction
        lines.push({
          line_number: lineNumber++,
          type: 'action',
          text: trimmedLine,
          is_user_line: false
        });
      } else if (trimmedLine.includes(':')) {
        // Dialogue line
        const colonIndex = trimmedLine.indexOf(':');
        const speaker = trimmedLine.substring(0, colonIndex).trim();
        const text = trimmedLine.substring(colonIndex + 1).trim();
        
        lines.push({
          line_number: lineNumber++,
          type: 'dialogue',
          speaker: speaker,
          text: text,
          is_user_line: false // Will be set by filterUserLines
        });
      } else {
        // Narration or other
        lines.push({
          line_number: lineNumber++,
          type: 'narration',
          text: trimmedLine,
          is_user_line: false
        });
      }
    }

    return lines;
  }

  /**
   * Filter lines for specific character practice
   */
  private filterUserLines(lines: DialogueLine[], character: string): DialogueLine[] {
    const characterLower = character.toLowerCase();
    return lines
      .filter(line => 
        line.type === 'dialogue' && 
        line.speaker?.toLowerCase() === characterLower
      )
      .map(line => ({ ...line, is_user_line: true }));
  }

  /**
   * Get context lines before a specific line number
   */
  private getContextBefore(lines: DialogueLine[], targetLineNumber: number, contextLines: number = 2): string {
    const contextStart = Math.max(0, 
      lines.findIndex(line => line.line_number === targetLineNumber) - contextLines
    );
    const contextEnd = lines.findIndex(line => line.line_number === targetLineNumber);
    
    return lines
      .slice(contextStart, contextEnd)
      .filter(line => line.type === 'dialogue' || line.type === 'action')
      .map(line => {
        if (line.type === 'dialogue' && line.speaker) {
          return `${line.speaker}: ${line.text}`;
        }
        return line.text;
      })
      .join('\n');
  }

  /**
   * Find suitable scene for practice
   */
  private async findPracticeScene(
    episodeId: string, 
    character: string, 
    sceneNumber?: number
  ): Promise<any> {
    try {
      // If specific scene number is provided
      if (sceneNumber) {
        const sceneId = `${episodeId}_${sceneNumber.toString().padStart(3, '0')}`;
        const sceneDoc = await this.db.collection("friends_scenes").doc(sceneId).get();
        if (sceneDoc.exists) {
          return { id: sceneId, ...sceneDoc.data() };
        }
      }

      // Otherwise, find any scene with the character
      const scenesQuery = await this.db
        .collection("friends_scenes")
        .where("metadata.episode_id", "==", episodeId)
        .where("metadata.characters", "array-contains", character)
        .limit(1)
        .get();

      if (!scenesQuery.empty) {
        const sceneDoc = scenesQuery.docs[0];
        return { id: sceneDoc.id, ...sceneDoc.data() };
      }

      return null;
    } catch (error) {
      console.error("Error finding practice scene:", error);
      return null;
    }
  }
}