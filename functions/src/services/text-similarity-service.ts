import OpenAI from "openai";

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
 * Text Similarity Calculation Service
 * Ports the sophisticated similarity calculation logic from local friends_chatbot.py
 */
export class TextSimilarityService {
  private openai: OpenAI;

  constructor(openai: OpenAI) {
    this.openai = openai;
  }

  /**
   * Main similarity calculation method - ports calculate_text_similarity() from Python
   */
  async calculateSimilarity(userInput: string, expectedText: string): Promise<SimilarityResult> {
    try {
      // Step 1: Extract actual dialogue from expected text (remove action descriptions)
      const actualDialogue = this.extractDialogueOnly(expectedText);
      
      // Step 2: Clean both texts for comparison
      const userClean = this.cleanTextForComparison(userInput);
      const expectedClean = this.cleanTextForComparison(actualDialogue);
      
      // Step 3: Multiple comparison methods
      
      // Exact match (highest score)
      if (userClean === expectedClean) {
        return {
          similarity: 1.0,
          isCorrect: true,
          feedback: "✅ Excellent! Perfect match!",
          detailedAnalysis: {
            wordSimilarity: 1.0,
            characterSimilarity: 1.0,
            embeddingSimilarity: 1.0,
            exactMatch: true,
            veryCloseMatch: true
          }
        };
      }
      
      // Very close match (minor differences)
      if (this.isVeryCloseMatch(userClean, expectedClean)) {
        return {
          similarity: 0.95,
          isCorrect: true,
          feedback: "✅ Excellent! Very close match!",
          detailedAnalysis: {
            wordSimilarity: 0.95,
            characterSimilarity: 0.95,
            embeddingSimilarity: 0.95,
            exactMatch: false,
            veryCloseMatch: true
          }
        };
      }
      
      // Word-based similarity (fast, no API calls)
      const wordSimilarity = this.calculateWordSimilarity(userClean, expectedClean);
      
      // Character-based similarity for short phrases
      let characterSimilarity = 0;
      if (expectedClean.length <= 20) {  // Short phrases
        characterSimilarity = this.calculateCharacterSimilarity(userClean, expectedClean);
      }
      
      // Use the higher of word or character similarity for short phrases
      let finalSimilarity = wordSimilarity;
      if (expectedClean.length <= 20) {
        finalSimilarity = Math.max(wordSimilarity, characterSimilarity);
      }
      
      // For longer text or if word similarity is low, use embedding
      let embeddingSimilarity: number | undefined;
      if (wordSimilarity < 0.4 && expectedClean.length > 20) {
        try {
          embeddingSimilarity = await this.calculateEmbeddingSimilarity(userClean, expectedClean);
          finalSimilarity = Math.max(finalSimilarity, embeddingSimilarity);
        } catch (error) {
          console.error("Error calculating embedding similarity:", error);
          // Fall back to word similarity
        }
      }
      
      // Determine if correct based on similarity threshold
      const isCorrect = finalSimilarity >= 0.6; // Threshold from local implementation
      
      // Generate feedback
      let feedback: string;
      if (finalSimilarity >= 0.8) {
        feedback = "✅ Excellent! Perfect match!";
      } else if (finalSimilarity >= 0.6) {
        feedback = "👍 Good! Close enough.";
      } else if (finalSimilarity >= 0.4) {
        feedback = "🤔 Not quite right, but you got the idea.";
      } else {
        feedback = `❌ Try again! The correct line was: '${actualDialogue}'`;
      }
      
      return {
        similarity: finalSimilarity,
        isCorrect,
        feedback,
        detailedAnalysis: {
          wordSimilarity,
          characterSimilarity,
          embeddingSimilarity,
          exactMatch: false,
          veryCloseMatch: false
        }
      };
      
    } catch (error) {
      console.error("Error in calculateSimilarity:", error);
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
   * Extract dialogue from text, removing action descriptions - ports extract_dialogue_only() from Python
   */
  private extractDialogueOnly(text: string): string {
    // Remove parenthetical actions like "(mortified)", "(laughing)", etc.
    let cleaned = text.replace(/\([^)]*\)/g, '');
    
    // Remove stage directions in brackets
    cleaned = cleaned.replace(/\[[^\]]*\]/g, '');
    
    // Clean up extra whitespace
    cleaned = cleaned.split(/\s+/).join(' ');
    
    return cleaned.trim();
  }

  /**
   * Clean text for comparison - ports clean_text_for_comparison() from Python
   */
  private cleanTextForComparison(text: string): string {
    // Convert to lowercase
    let cleaned = text.toLowerCase().trim();
    
    // Remove common punctuation that doesn't affect meaning
    cleaned = cleaned.replace(/[.!?,"'-]/g, '').replace(/-/g, ' ');
    
    // Normalize whitespace
    cleaned = cleaned.split(/\s+/).join(' ');
    
    return cleaned;
  }

  /**
   * Check for very close matches - ports is_very_close_match() from Python
   */
  private isVeryCloseMatch(text1: string, text2: string): boolean {
    // Same length, most characters match
    if (text1.length === text2.length) {
      let differences = 0;
      for (let i = 0; i < text1.length; i++) {
        if (text1[i] !== text2[i]) {
          differences++;
        }
      }
      if (differences <= 1) {  // At most 1 character different
        return true;
      }
    }
    
    // One character added or removed
    if (Math.abs(text1.length - text2.length) === 1) {
      const shorter = text1.length < text2.length ? text1 : text2;
      const longer = text1.length < text2.length ? text2 : text1;
      
      for (let i = 0; i < longer.length; i++) {
        const withoutChar = longer.slice(0, i) + longer.slice(i + 1);
        if (withoutChar === shorter) {
          return true;
        }
      }
    }
    
    return false;
  }

  /**
   * Calculate word-based similarity - ports calculate_word_similarity() from Python
   */
  private calculateWordSimilarity(text1: string, text2: string): number {
    const words1 = new Set(text1.split(/\s+/).filter(w => w.length > 0));
    const words2 = new Set(text2.split(/\s+/).filter(w => w.length > 0));
    
    if (words1.size === 0 && words2.size === 0) {
      return 1.0;
    }
    if (words1.size === 0 || words2.size === 0) {
      return 0.0;
    }
    
    // Calculate intersection and union (Jaccard similarity)
    const intersection = new Set([...words1].filter(w => words2.has(w)));
    const union = new Set([...words1, ...words2]);
    
    let jaccard = intersection.size / union.size;
    
    // Bonus for same word count
    if (words1.size === words2.size) {
      jaccard += 0.1;
    }
    
    return Math.min(1.0, jaccard);
  }

  /**
   * Calculate character-based similarity - ports calculate_character_similarity() from Python
   * Uses Levenshtein distance ratio implementation
   */
  private calculateCharacterSimilarity(text1: string, text2: string): number {
    if (!text1 || !text2) {
      return 0.0;
    }
    
    // Make sure text1 is the longer string
    if (text1.length < text2.length) {
      [text1, text2] = [text2, text1];
    }
    
    if (text2.length === 0) {
      return 0.0;
    }
    
    // Calculate Levenshtein distance
    let previousRow = Array.from({length: text2.length + 1}, (_, i) => i);
    
    for (let i = 0; i < text1.length; i++) {
      const currentRow = [i + 1];
      
      for (let j = 0; j < text2.length; j++) {
        const insertions = previousRow[j + 1] + 1;
        const deletions = currentRow[j] + 1;
        const substitutions = previousRow[j] + (text1[i] !== text2[j] ? 1 : 0);
        currentRow.push(Math.min(insertions, deletions, substitutions));
      }
      
      previousRow = currentRow;
    }
    
    const maxLen = Math.max(text1.length, text2.length);
    return (maxLen - previousRow[previousRow.length - 1]) / maxLen;
  }

  /**
   * Calculate embedding-based similarity - ports calculate_embedding_similarity() from Python
   */
  private async calculateEmbeddingSimilarity(text1: string, text2: string): Promise<number> {
    try {
      // Generate embeddings for both texts
      const embeddingResponse = await this.openai.embeddings.create({
        input: [text1, text2],
        model: "text-embedding-3-small",
      });

      const [emb1, emb2] = embeddingResponse.data.map(d => d.embedding);
      
      if (!emb1 || !emb2) {
        return 0.0;
      }
      
      // Calculate cosine similarity
      const dotProduct = emb1.reduce((sum, a, i) => sum + a * emb2[i], 0);
      const norm1 = Math.sqrt(emb1.reduce((sum, a) => sum + a * a, 0));
      const norm2 = Math.sqrt(emb2.reduce((sum, a) => sum + a * a, 0));
      
      if (norm1 === 0 || norm2 === 0) {
        return 0.0;
      }
      
      const similarity = dotProduct / (norm1 * norm2);
      return Math.max(0.0, similarity);
      
    } catch (error) {
      console.error("Error with embedding similarity:", error);
      return 0.0;
    }
  }
}