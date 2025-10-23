import { PracticeSessionManager } from '../services/practice-session-manager';
import { TextSimilarityService } from '../services/text-similarity-service';
import * as admin from 'firebase-admin';

// Mock Firestore and other dependencies
jest.mock('firebase-admin');
jest.mock('../services/text-similarity-service');

describe('PracticeSessionManager', () => {
  let manager: PracticeSessionManager;
  let mockDb: jest.Mocked<admin.firestore.Firestore>;
  let mockTextSimilarity: jest.Mocked<TextSimilarityService>;

  beforeEach(() => {
    mockDb = {
      collection: jest.fn().mockReturnThis(),
      doc: jest.fn().mockReturnThis(),
      get: jest.fn(),
      set: jest.fn(),
      update: jest.fn(),
      where: jest.fn().mockReturnThis(),
      limit: jest.fn().mockReturnThis(),
    } as any;

    mockTextSimilarity = {
      calculateSimilarity: jest.fn(),
    } as any;

    manager = new PracticeSessionManager(mockDb, mockTextSimilarity);
  });

  describe('startInteractiveSession', () => {
    it('should start a new practice session successfully', async () => {
      // Mock scene data
      const mockSceneData = {
        id: 'S09E19_013',
        episode_id: 'S09E19',
        scene_number: 13,
        location: 'The playground',
        characters: ['Ross', 'Rachel'],
        scene_description: 'Ross and Rachel practice with Emma',
        raw_text: 'Rachel: Ok... I got a spider.\nRoss: (nervously) Ok...\nRachel: Ok...'
      };

      // Mock Firestore responses
      mockDb.get = jest.fn().mockResolvedValue({
        exists: true,
        data: () => mockSceneData
      });

      mockDb.set = jest.fn().mockResolvedValue(undefined);

      const result = await manager.startInteractiveSession(
        'user123',
        'S09E19',
        'Ross',
        13
      );

      expect(result.session_id).toBeDefined();
      expect(result.scene_context.location).toBe('The playground');
      expect(result.total_lines).toBeGreaterThan(0);
      expect(result.user_turn.speaker).toBe('Ross');
    });

    it('should throw error when no suitable scene is found', async () => {
      mockDb.get = jest.fn().mockResolvedValue({
        exists: false
      });

      await expect(
        manager.startInteractiveSession('user123', 'S99E99', 'InvalidCharacter')
      ).rejects.toThrow('No suitable scene found');
    });
  });

  describe('continuePractice', () => {
    it('should evaluate user response and update session', async () => {
      // Mock session data
      const mockSession = {
        session_id: 'session_123',
        user_id: 'user123',
        episode_id: 'S09E19',
        character: 'Ross',
        scene_number: 13,
        scene_id: 'S09E19_013',
        current_line_index: 0,
        total_lines: 2,
        dialogue_lines: [
          {
            line_number: 1,
            type: 'dialogue',
            speaker: 'Ross',
            text: '(nervously) Ok...',
            is_user_line: true
          },
          {
            line_number: 2,
            type: 'dialogue',
            speaker: 'Ross',
            text: 'This feels perfectly normal.',
            is_user_line: true
          }
        ],
        correct_answers: 0,
        total_attempts: 0,
        session_progress: [],
        status: 'active',
        started_at: admin.firestore.Timestamp.now(),
        created_at: admin.firestore.Timestamp.now(),
        updated_at: admin.firestore.Timestamp.now(),
      };

      // Mock Firestore responses
      mockDb.get = jest.fn().mockResolvedValue({
        exists: true,
        data: () => mockSession
      });

      mockDb.update = jest.fn().mockResolvedValue(undefined);

      // Mock similarity service response
      mockTextSimilarity.calculateSimilarity.mockResolvedValue({
        similarity: 0.9,
        isCorrect: true,
        feedback: 'Excellent! Perfect match!',
        detailedAnalysis: {
          wordSimilarity: 0.9,
          characterSimilarity: 0.9,
          exactMatch: false,
          veryCloseMatch: true
        }
      });

      const result = await manager.continuePractice('session_123', 'Ok');

      expect(result.evaluation.isCorrect).toBe(true);
      expect(result.session_status.correct_answers).toBe(1);
      expect(result.next_user_turn).toBeDefined();
      expect(result.is_session_complete).toBe(false);
    });

    it('should complete session when all lines are practiced', async () => {
      const mockSession = {
        session_id: 'session_123',
        current_line_index: 1, // Last line
        total_lines: 2,
        dialogue_lines: [
          {
            line_number: 1,
            type: 'dialogue',
            speaker: 'Ross',
            text: 'Ok...',
            is_user_line: true
          },
          {
            line_number: 2,
            type: 'dialogue',
            speaker: 'Ross',
            text: 'This feels perfectly normal.',
            is_user_line: true
          }
        ],
        character: 'Ross',
        correct_answers: 1,
        total_attempts: 1,
        session_progress: [],
        status: 'active',
        started_at: admin.firestore.Timestamp.now(),
        created_at: admin.firestore.Timestamp.now(),
        updated_at: admin.firestore.Timestamp.now(),
      };

      mockDb.get = jest.fn().mockResolvedValue({
        exists: true,
        data: () => mockSession
      });

      mockDb.update = jest.fn().mockResolvedValue(undefined);

      mockTextSimilarity.calculateSimilarity.mockResolvedValue({
        similarity: 0.9,
        isCorrect: true,
        feedback: 'Perfect!',
        detailedAnalysis: {
          wordSimilarity: 0.9,
          characterSimilarity: 0.9,
          exactMatch: true,
          veryCloseMatch: true
        }
      });

      const result = await manager.continuePractice('session_123', 'This feels perfectly normal');

      expect(result.is_session_complete).toBe(true);
      expect(result.next_user_turn).toBeUndefined();
    });
  });

  describe('completeSession', () => {
    it('should calculate final metrics correctly', async () => {
      const startTime = admin.firestore.Timestamp.fromDate(new Date('2023-01-01T10:00:00Z'));
      const endTime = admin.firestore.Timestamp.fromDate(new Date('2023-01-01T10:05:00Z'));

      const mockCompletedSession = {
        session_id: 'session_123',
        status: 'completed',
        correct_answers: 4,
        total_attempts: 5,
        started_at: startTime,
        completed_at: endTime,
      };

      mockDb.get = jest.fn().mockResolvedValue({
        exists: true,
        data: () => mockCompletedSession
      });

      const result = await manager.completeSession('session_123');

      expect(result.final_score).toBe(80); // 4/5 * 100
      expect(result.accuracy).toBe(80);
      expect(result.correct_answers).toBe(4);
      expect(result.session_duration_minutes).toBe(5);
    });
  });

  describe('parseSceneDialogue', () => {
    it('should parse scene text correctly', () => {
      const sceneText = `[Scene: The playground]
Rachel: Ok... I got a spider.
Ross: (nervously) Ok...
[Ross takes the spider]
Rachel: Ok...`;

      // Access private method through any cast
      const lines = (manager as any).parseSceneDialogue(sceneText);

      expect(lines).toHaveLength(4);
      expect(lines[0].type).toBe('action');
      expect(lines[1].type).toBe('dialogue');
      expect(lines[1].speaker).toBe('Rachel');
      expect(lines[2].speaker).toBe('Ross');
    });
  });
});