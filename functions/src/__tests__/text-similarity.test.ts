import { TextSimilarityService } from '../services/text-similarity-service';
import OpenAI from 'openai';

// Mock OpenAI
jest.mock('openai');
const MockedOpenAI = OpenAI as jest.MockedClass<typeof OpenAI>;

describe('TextSimilarityService', () => {
  let service: TextSimilarityService;
  let mockOpenAI: jest.Mocked<OpenAI>;

  beforeEach(() => {
    mockOpenAI = new MockedOpenAI() as jest.Mocked<OpenAI>;
    service = new TextSimilarityService(mockOpenAI);
  });

  describe('calculateSimilarity', () => {
    it('should return perfect match for identical texts', async () => {
      const result = await service.calculateSimilarity('Hello world', 'Hello world');
      
      expect(result.similarity).toBe(1.0);
      expect(result.isCorrect).toBe(true);
      expect(result.feedback).toContain('Perfect match');
      expect(result.detailedAnalysis.exactMatch).toBe(true);
    });

    it('should handle dialogue extraction correctly', async () => {
      const userInput = 'Ok';
      const expectedText = '(nervously) Ok...';
      
      const result = await service.calculateSimilarity(userInput, expectedText);
      
      expect(result.similarity).toBeGreaterThan(0.8);
      expect(result.isCorrect).toBe(true);
    });

    it('should calculate word similarity correctly', async () => {
      const userInput = 'This feels perfectly normal';
      const expectedText = 'This feels perfectly normal. Ok, get on the swing!';
      
      const result = await service.calculateSimilarity(userInput, expectedText);
      
      expect(result.similarity).toBeGreaterThan(0.6);
      expect(result.detailedAnalysis.wordSimilarity).toBeGreaterThan(0.5);
    });

    it('should handle very close matches', async () => {
      const userInput = 'Helo world';  // One character typo
      const expectedText = 'Hello world';
      
      const result = await service.calculateSimilarity(userInput, expectedText);
      
      expect(result.similarity).toBe(0.95);
      expect(result.detailedAnalysis.veryCloseMatch).toBe(true);
    });

    it('should provide appropriate feedback for different similarity levels', async () => {
      // High similarity
      const highResult = await service.calculateSimilarity('Great job', 'Great job!');
      expect(highResult.feedback).toContain('Excellent');
      
      // Medium similarity
      const medResult = await service.calculateSimilarity('Good work', 'Great job');
      expect(medResult.feedback).toContain('Good') || expect(medResult.feedback).toContain('idea');
      
      // Low similarity
      const lowResult = await service.calculateSimilarity('Hello', 'Goodbye everyone');
      expect(lowResult.feedback).toContain('Try again');
    });

    it('should handle empty or invalid inputs gracefully', async () => {
      const result = await service.calculateSimilarity('', '');
      
      expect(result.similarity).toBeDefined();
      expect(result.isCorrect).toBeDefined();
      expect(result.feedback).toBeDefined();
    });
  });

  describe('Character similarity calculation', () => {
    it('should calculate character-based similarity for short phrases', async () => {
      const result = await service.calculateSimilarity('Hi', 'Hi!');
      
      expect(result.detailedAnalysis.characterSimilarity).toBeGreaterThan(0.8);
    });
  });

  describe('Error handling', () => {
    it('should handle errors gracefully', async () => {
      // Mock an error
      const errorService = new TextSimilarityService(null as any);
      
      const result = await errorService.calculateSimilarity('test', 'test');
      
      expect(result.similarity).toBe(0);
      expect(result.isCorrect).toBe(false);
      expect(result.feedback).toContain('Error');
    });
  });
});