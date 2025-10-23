import OpenAI from "openai";

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
 * Advanced Cultural Context Explanation Service
 * Ports the sophisticated cultural explanation system from local Python implementation
 */
export class AdvancedCulturalContextService {
  private openai: OpenAI;

  // Advanced cultural explanation prompt template - ported from Python
  private readonly ADVANCED_CULTURAL_PROMPT = `You are a specialized American cultural linguist and English teacher for Korean students learning through Friends TV show.

Your task: Analyze expressions, idioms, or cultural references and provide comprehensive explanations.

Response Format (always follow this EXACT structure):
**American Cultural Reference: '[EXPRESSION]'** 🇺🇸

**Meaning**: [One clear sentence explaining what it means]
**Origin**: [Where it comes from - keep it interesting but brief]
**When to use**: [Context and situations - practical advice]

**Examples in conversation**:
• "[Natural example 1]"
• "[Natural example 2]" 
• "[Natural example 3]"

**Similar expressions**: [2-3 alternatives they might hear]

💡 **Friends Context**: [How this might appear in Friends episodes or 90s American culture]

**What's next?**
• 'Find episodes about [topic]' - See it in actual scenes
• 'Practice as [character]' - Use this expression in conversation
• 'Show me S01E01 scene 2' - See real dialogue examples

Rules:
- Keep explanations clear for non-native speakers
- Use conversational, engaging tone
- Include practical usage tips
- Focus on expressions common in American TV/movies
- If it's not a clear idiom/expression, explain the cultural concept instead
- Always end with the Friends context connection
- Provide actionable next steps for practice`;

  constructor(openai: OpenAI) {
    this.openai = openai;
  }

  /**
   * Generate structured cultural explanation - ports get_direct_explanation() from Python
   */
  async generateStructuredExplanation(
    topic: string,
    details: string,
    originalMessage: string,
    episodeContext?: string
  ): Promise<StructuredCulturalExplanation> {
    try {
      console.log(`Generating advanced cultural explanation for: ${topic}`);
      
      // Get GPT-based cultural explanation
      const gptExplanation = await this.getGPTCulturalExplanation(
        topic,
        details,
        originalMessage,
        episodeContext
      );

      // Parse the structured response and extract components
      const structured = this.parseStructuredResponse(gptExplanation, topic);
      
      // Add contextual enhancements
      const enhanced = await this.enhanceWithContextualData(structured, topic, episodeContext);
      
      return enhanced;
      
    } catch (error) {
      console.error("Error generating structured cultural explanation:", error);
      
      // Fallback to basic explanation
      return {
        expression: topic,
        meaning: "This is an American cultural reference or expression.",
        origin: "American culture and language",
        usage_context: "Used in everyday American English conversation",
        conversation_examples: [
          `"I heard about ${topic} on Friends."`,
          `"Can you explain ${topic} to me?"`,
          `"That's a typical ${topic} situation."`
        ],
        similar_expressions: ["related terms", "similar phrases"],
        friends_context: `This appears in Friends episodes and represents 1990s American culture.`,
        follow_up_suggestions: [
          "Watch more Friends episodes to see this in context",
          "Practice using this expression in conversation",
          "Look for similar cultural references"
        ],
        related_episodes: [],
        practice_tips: [
          "Pay attention to when characters use this expression",
          "Try using it in your own conversations",
          "Notice the cultural context when it appears"
        ]
      };
    }
  }

  /**
   * Get GPT-based cultural explanation - ports get_gpt_cultural_explanation() from Python
   */
  private async getGPTCulturalExplanation(
    topic: string,
    details: string,
    originalMessage: string,
    episodeContext?: string
  ): Promise<string> {
    try {
      const contextualPrompt = this.buildContextualPrompt(
        topic,
        details,
        originalMessage,
        episodeContext
      );

      const response = await this.openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          {
            role: "system",
            content: this.ADVANCED_CULTURAL_PROMPT
          },
          {
            role: "user",
            content: contextualPrompt
          }
        ],
        temperature: 0.7,
        max_tokens: 1000
      });

      return response.choices[0].message.content || "";
      
    } catch (error) {
      console.error("Error with GPT cultural explanation:", error);
      throw error;
    }
  }

  /**
   * Build contextual prompt based on user input and episode context
   */
  private buildContextualPrompt(
    topic: string,
    details: string,
    originalMessage: string,
    episodeContext?: string
  ): string {
    let prompt = `User asked: "${originalMessage}"\n\n`;
    prompt += `Topic to explain: "${topic}"\n`;
    
    if (details) {
      prompt += `Additional context: "${details}"\n`;
    }
    
    if (episodeContext) {
      prompt += `Episode context: This appeared in Friends episode ${episodeContext}\n`;
    }
    
    prompt += `\nPlease provide a comprehensive cultural explanation following the exact format specified.`;
    
    return prompt;
  }

  /**
   * Parse structured response from GPT into components
   */
  private parseStructuredResponse(
    gptResponse: string,
    fallbackTopic: string
  ): StructuredCulturalExplanation {
    try {
      // Extract structured components using regex patterns
      const extractSection = (pattern: RegExp, defaultValue: string | string[]) => {
        const match = gptResponse.match(pattern);
        if (match && match[1]) {
          return Array.isArray(defaultValue) 
            ? match[1].split('•').map(s => s.trim()).filter(s => s.length > 0)
            : match[1].trim();
        }
        return defaultValue;
      };

      const expression = extractSection(
        /\*\*American Cultural Reference: ['"]([^'"]+)['"]?\*\*/i,
        fallbackTopic
      ) as string;

      const meaning = extractSection(
        /\*\*Meaning\*\*:?\s*([^*]+)/i,
        "This is an American cultural reference."
      ) as string;

      const origin = extractSection(
        /\*\*Origin\*\*:?\s*([^*]+)/i,
        "American culture and language"
      ) as string;

      const usage_context = extractSection(
        /\*\*When to use\*\*:?\s*([^*]+)/i,
        "Used in everyday American conversation"
      ) as string;

      const examples = extractSection(
        /\*\*Examples in conversation\*\*:?\s*([^*]+)/i,
        []
      ) as string[];

      const similar = extractSection(
        /\*\*Similar expressions\*\*:?\s*([^*]+)/i,
        []
      ) as string[];

      const friendsContext = extractSection(
        /💡\s*\*\*Friends Context\*\*:?\s*([^*]+)/i,
        "This appears in Friends episodes representing 1990s American culture."
      ) as string;

      // Extract follow-up suggestions from "What's next?" section
      const followUpMatch = gptResponse.match(/\*\*What's next\?\*\*\s*(.*?)(?:\n\n|$)/s);
      const followUpSuggestions: string[] = [];
      if (followUpMatch) {
        const suggestions = followUpMatch[1].split('•').map(s => s.trim()).filter(s => s.length > 0);
        followUpSuggestions.push(...suggestions);
      }

      return {
        expression,
        meaning,
        origin,
        usage_context,
        conversation_examples: examples.length > 0 ? examples : [
          `"I learned about '${expression}' from Friends."`,
          `"That's a great example of '${expression}'."`,
          `"Can you use '${expression}' in a sentence?"`
        ],
        similar_expressions: similar.length > 0 ? similar : ["related terms", "similar phrases"],
        friends_context: friendsContext,
        follow_up_suggestions: followUpSuggestions.length > 0 ? followUpSuggestions : [
          "Watch more Friends episodes to see this in context",
          "Practice using this expression in conversation"
        ],
        related_episodes: [],
        practice_tips: [
          "Listen for this expression in Friends episodes",
          "Practice using it in your own conversations",
          "Pay attention to the cultural context"
        ]
      };

    } catch (error) {
      console.error("Error parsing structured response:", error);
      
      // Return basic fallback structure
      return {
        expression: fallbackTopic,
        meaning: "This is an American cultural reference or expression.",
        origin: "American culture and language",
        usage_context: "Used in everyday American English conversation",
        conversation_examples: [
          `"I learned about '${fallbackTopic}' from Friends."`,
          `"That's a great example of '${fallbackTopic}'."`,
          `"Can you use '${fallbackTopic}' in a sentence?"`
        ],
        similar_expressions: ["related terms", "similar phrases"],
        friends_context: "This appears in Friends episodes representing 1990s American culture.",
        follow_up_suggestions: [
          "Watch more Friends episodes to see this in context",
          "Practice using this expression in conversation"
        ],
        related_episodes: [],
        practice_tips: [
          "Listen for this expression in Friends episodes",
          "Practice using it in your own conversations",
          "Pay attention to the cultural context"
        ]
      };
    }
  }

  /**
   * Enhance structured explanation with additional contextual data
   */
  private async enhanceWithContextualData(
    explanation: StructuredCulturalExplanation,
    topic: string,
    episodeContext?: string
  ): Promise<StructuredCulturalExplanation> {
    try {
      // Add episode context if provided
      if (episodeContext) {
        explanation.related_episodes.push(episodeContext);
        explanation.friends_context += ` This was featured in episode ${episodeContext}.`;
      }

      // Enhance practice tips based on the expression type
      const enhancedTips = this.generateEnhancedPracticeTips(explanation.expression, topic);
      explanation.practice_tips = [...explanation.practice_tips, ...enhancedTips];

      // Add contextual follow-up suggestions
      const contextualSuggestions = this.generateContextualSuggestions(topic, episodeContext);
      explanation.follow_up_suggestions = [
        ...explanation.follow_up_suggestions,
        ...contextualSuggestions
      ];

      return explanation;
      
    } catch (error) {
      console.error("Error enhancing with contextual data:", error);
      return explanation;
    }
  }

  /**
   * Generate enhanced practice tips based on expression analysis
   */
  private generateEnhancedPracticeTips(expression: string, topic: string): string[] {
    const tips: string[] = [];

    // Check for common expression types and add specific tips
    if (expression.toLowerCase().includes('idiom') || expression.includes('saying')) {
      tips.push("Practice using this idiom in different contexts");
      tips.push("Learn the literal vs. figurative meaning");
    }

    if (expression.toLowerCase().includes('slang')) {
      tips.push("Be aware this is informal language");
      tips.push("Practice with friends in casual settings");
    }

    if (topic.toLowerCase().includes('work') || topic.toLowerCase().includes('job')) {
      tips.push("Practice in professional conversation contexts");
      tips.push("Notice how Friends characters use this at work");
    }

    if (topic.toLowerCase().includes('relationship') || topic.toLowerCase().includes('dating')) {
      tips.push("Observe romantic dialogue patterns in Friends");
      tips.push("Practice expressing feelings and relationships");
    }

    return tips;
  }

  /**
   * Generate contextual follow-up suggestions
   */
  private generateContextualSuggestions(topic: string, episodeContext?: string): string[] {
    const suggestions: string[] = [];

    if (episodeContext) {
      suggestions.push(`Watch ${episodeContext} again to see this in context`);
      suggestions.push(`Find other scenes in ${episodeContext} with similar expressions`);
    }

    // Add topic-specific suggestions
    if (topic.toLowerCase().includes('food')) {
      suggestions.push("Explore other food-related expressions from Friends");
      suggestions.push("Practice ordering food using this expression");
    }

    if (topic.toLowerCase().includes('emotion') || topic.toLowerCase().includes('feeling')) {
      suggestions.push("Practice expressing emotions like Friends characters");
      suggestions.push("Learn more emotional expressions from the show");
    }

    suggestions.push("Ask for more cultural explanations as you watch");
    suggestions.push("Practice with other Friends fans");

    return suggestions;
  }
}