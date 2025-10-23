#!/usr/bin/env python3
"""
Friends English Conversation Practice Chatbot - MVP
RAG System: Condense → Route → Query → Respond
"""

import os
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# External dependencies
import openai
from pinecone import Pinecone
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore

# Load environment variables
load_dotenv()

@dataclass
class ChatContext:
    user_id: str
    conversation_history: List[Dict[str, str]]
    current_topic: Optional[str] = None
    practice_session: Optional[Dict] = None

class FriendsRAGChatbot:
    def __init__(self):
        """Initialize the Friends RAG Chatbot"""
        # Initialize clients
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pinecone_client.Index("convo")
        
        # Initialize Firebase
        if not firebase_admin._apps:
            cred = credentials.Certificate("conversation-practice-f2199-firebase-adminsdk-fbsvc-1e1af80c9c.json")
            firebase_admin.initialize_app(cred)
        self.db = firestore.client()
        
        # Character information
        self.characters = {
            "Monica": {
                "personality": "Perfectionist chef, obsessed with cleanliness",
                "traits": "Competitive, caring, neurotic about organization",
                "speech_patterns": "Fast-talking, expressive, uses food metaphors",
                "practice_focus": "Giving advice, organizing events, cooking vocabulary"
            },
            "Rachel": {
                "personality": "Fashion-focused, wealthy background turned independent",
                "traits": "Shopping enthusiast, romantic, growing confident",
                "speech_patterns": "Valley girl accent initially, sophisticated later",
                "practice_focus": "Fashion vocabulary, workplace conversations, relationship talks"
            },
            "Ross": {
                "personality": "Paleontologist, intellectual, awkward in relationships",
                "traits": "Nerdy, passionate about dinosaurs, jealous tendencies",
                "speech_patterns": "Academic vocabulary, explains things in detail",
                "practice_focus": "Academic discussions, explaining concepts, awkward situations"
            },
            "Chandler": {
                "personality": "Sarcastic office worker, commitment issues",
                "traits": "Witty, defensive through humor, loyal friend",
                "speech_patterns": "Heavy use of sarcasm, rhetorical questions, catchphrases",
                "practice_focus": "Sarcasm, office humor, witty comebacks"
            },
            "Joey": {
                "personality": "Struggling actor, simple but loyal",
                "traits": "Food-loving, ladies' man, childlike innocence",
                "speech_patterns": "Simple vocabulary, catchphrase 'How you doin'?'",
                "practice_focus": "Casual conversations, expressing confusion, food-related talks"
            },
            "Phoebe": {
                "personality": "Eccentric musician with unconventional past",
                "traits": "Free-spirited, honest to a fault, believes in alternative medicine",
                "speech_patterns": "Unique worldview, blunt honesty, spiritual references",
                "practice_focus": "Creative expressions, giving unconventional advice, music vocabulary"
            }
        }
        
        # Common Friends expressions
        self.friends_expressions = {
            "How you doin'?": {
                "character": "Joey",
                "meaning": "Flirtatious greeting, Joey's pickup line",
                "usage": "Casual, humorous way to greet someone you find attractive",
                "context": "Informal, mostly used by Joey as his signature line"
            },
            "Could I BE any more": {
                "character": "Chandler", 
                "meaning": "Sarcastic emphasis pattern",
                "usage": "To sarcastically emphasize something obvious or extreme",
                "context": "Chandler's signature sarcastic speech pattern"
            },
            "We were on a break!": {
                "character": "Ross",
                "meaning": "Excuse for dating someone else during relationship pause",
                "usage": "Defensive justification, became a running gag",
                "context": "Ross's defense for his actions during break with Rachel"
            }
        }
        
        print("🎭 Friends English Chatbot initialized!")
        print("Available features:")
        print("1. Episode recommendations by topic")
        print("2. Character information")
        print("3. Episode plot summaries")
        print("4. Scene script viewing")
        print("5. Cultural context explanations")
        print("6. Conversation practice")

    def get_embedding(self, text: str) -> List[float]:
        """Get OpenAI embedding for text"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return []

    def condense_user_intent(self, user_message: str, chat_history: List[str]) -> Dict[str, Any]:
        """Step 1: Condense user intent from message and history"""
        
        history_context = "\n".join(chat_history[-5:]) if chat_history else ""
        
        system_prompt = """
        Analyze the user's message and determine their intent for a Friends English learning chatbot.
        
        Possible intents:
        1. episode_recommendation - Want episode suggestions for specific topics/situations
        2. character_info - Ask about Friends characters 
        3. plot_summary - Want episode plot/summary
        4. scene_script - Want to see specific scene dialogue
        5. cultural_context - Need explanation of cultural references/expressions
        6. practice_session - Want to practice conversation/dialogue
        7. general_chat - General conversation about Friends
        
        Return in this format:
        Intent: [intent_type]
        Topic: [main topic/subject]
        Details: [any specific details like episode, character, etc.]
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Chat History:\n{history_context}\n\nUser Message: {user_message}"}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            
            # Parse the response
            intent_match = re.search(r"Intent:\s*(\w+)", content)
            topic_match = re.search(r"Topic:\s*(.+?)(?=\n|Details:|$)", content)
            details_match = re.search(r"Details:\s*(.+?)$", content, re.MULTILINE)
            
            return {
                "intent": intent_match.group(1) if intent_match else "general_chat",
                "topic": topic_match.group(1).strip() if topic_match else "",
                "details": details_match.group(1).strip() if details_match else "",
                "original_message": user_message
            }
            
        except Exception as e:
            print(f"Error condensing intent: {e}")
            return {
                "intent": "general_chat",
                "topic": user_message,
                "details": "",
                "original_message": user_message
            }

    def route_to_function(self, condensed_intent: Dict[str, Any]) -> str:
        """Step 2: Route to appropriate function based on intent"""
        intent = condensed_intent["intent"]
        
        route_mapping = {
            "episode_recommendation": "recommend_episodes",
            "character_info": "get_character_info", 
            "plot_summary": "get_episode_plot",
            "scene_script": "get_scene_script",
            "cultural_context": "explain_cultural_context",
            "practice_session": "start_practice_session",
            "general_chat": "general_friends_chat"
        }
        
        return route_mapping.get(intent, "general_friends_chat")

    def query_pinecone(self, query: str, filter_conditions: Dict = None, top_k: int = 5) -> List[Dict]:
        """Step 3: Query Pinecone for relevant content"""
        try:
            # Get embedding for query
            query_embedding = self.get_embedding(query)
            if not query_embedding:
                return []
            
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter_conditions or {},
                include_metadata=True
            )
            
            return [
                {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                }
                for match in results.matches
            ]
            
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return []

    # FEATURE 1: Episode Recommendation
    def recommend_episodes(self, condensed_intent: Dict[str, Any]) -> str:
        """Recommend episodes based on user's topic/situation"""
        topic = condensed_intent["topic"]
        
        print(f"🔍 Searching for episodes about: {topic}")
        
        # Query plot embeddings
        results = self.query_pinecone(
            query=f"episodes about {topic} situations conversations",
            filter_conditions={"chunk_type": "plot"},
            top_k=5
        )
        
        if not results:
            return f"I couldn't find episodes specifically about '{topic}'. Try asking about dating, work, friendship, or family situations!"
        
        response = f"Great choice! Here are Friends episodes perfect for practicing '{topic}':\n\n"
        
        for i, result in enumerate(results[:3], 1):
            metadata = result["metadata"]
            episode_id = metadata.get("episode_id", "")
            title = metadata.get("episode_title", "")
            plot = metadata.get("plot_text", "")
            score = result["score"]
            
            response += f"{i}. **{episode_id}: {title}**\n"
            response += f"   📖 Plot: {plot[:200]}...\n"
            response += f"   🎯 Match: {score:.2f}\n"
            response += f"   💡 Why it's perfect: Contains relevant vocabulary and situations for {topic}\n\n"
        
        response += "Would you like to:\n"
        response += "- See the script for any of these episodes?\n"
        response += "- Learn about the characters in these episodes?\n"
        response += "- Start practicing dialogue from one of them?"
        
        return response

    # FEATURE 2: Character Information
    def get_character_info(self, condensed_intent: Dict[str, Any]) -> str:
        """Provide character information and analysis"""
        topic = condensed_intent["topic"].lower()
        details = condensed_intent["details"].lower()
        
        # Find which character they're asking about
        character_name = None
        for char in self.characters.keys():
            if char.lower() in topic or char.lower() in details:
                character_name = char
                break
        
        if not character_name:
            # Show all characters
            response = "Here are the 6 main Friends characters you can practice with:\n\n"
            
            for name, info in self.characters.items():
                response += f"**{name}** 🎭\n"
                response += f"Personality: {info['personality']}\n"
                response += f"Speech Style: {info['speech_patterns']}\n"
                response += f"Best for practicing: {info['practice_focus']}\n\n"
            
            response += "Which character would you like to learn more about or practice as?"
            return response
        
        # Show specific character info
        char_info = self.characters[character_name]
        
        response = f"**{character_name}** - Perfect for English practice! 🎭\n\n"
        response += f"**Personality**: {char_info['personality']}\n"
        response += f"**Character Traits**: {char_info['traits']}\n"
        response += f"**Speech Patterns**: {char_info['speech_patterns']}\n"
        response += f"**Great for practicing**: {char_info['practice_focus']}\n\n"
        
        # Get some example scenes with this character
        print(f"🔍 Finding scenes with {character_name}...")
        
        scenes = self.query_pinecone(
            query=f"{character_name} funny scenes dialogue",
            filter_conditions={
                "chunk_type": "scene",
                "characters": {"$in": [character_name]}
            },
            top_k=3
        )
        
        if scenes:
            response += f"**Popular {character_name} scenes to practice:**\n"
            for scene in scenes:
                metadata = scene["metadata"]
                episode_id = metadata.get("episode_id", "")
                location = metadata.get("location", "")
                preview = metadata.get("text", "")[:100]
                
                response += f"- {episode_id} at {location}: '{preview}...'\n"
        
        response += f"\nWould you like to practice as {character_name} or see their dialogue from a specific episode?"
        
        return response

    # FEATURE 3: Episode Plot Summary  
    def get_episode_plot(self, condensed_intent: Dict[str, Any]) -> str:
        """Get episode plot summary"""
        topic = condensed_intent["topic"]
        details = condensed_intent["details"]
        
        # Extract episode ID if mentioned
        episode_pattern = r'S(\d{2})E(\d{2})|Season\s+(\d+)\s+Episode\s+(\d+)'
        episode_match = re.search(episode_pattern, f"{topic} {details}", re.IGNORECASE)
        
        if episode_match:
            if episode_match.group(1):
                season = int(episode_match.group(1))
                episode = int(episode_match.group(2))
                episode_id = f"S{season:02d}E{episode:02d}"
            else:
                season = int(episode_match.group(3))
                episode = int(episode_match.group(4))
                episode_id = f"S{season:02d}E{episode:02d}"
            
            print(f"🔍 Looking for plot of {episode_id}")
            
            results = self.query_pinecone(
                query=f"episode {episode_id} plot summary",
                filter_conditions={
                    "chunk_type": "plot",
                    "episode_id": episode_id
                },
                top_k=1
            )
        else:
            # Search by topic/title
            print(f"🔍 Searching for episode about: {topic}")
            results = self.query_pinecone(
                query=f"{topic} episode plot",
                filter_conditions={"chunk_type": "plot"},
                top_k=3
            )
        
        if not results:
            return "I couldn't find that episode. Try asking like 'Tell me about S01E01' or 'What happens in the pilot episode?'"
        
        if len(results) == 1:
            # Single episode
            metadata = results[0]["metadata"]
            episode_id = metadata.get("episode_id", "")
            title = metadata.get("episode_title", "")
            plot = metadata.get("plot_text", "")
            season = metadata.get("season", "")
            episode_num = metadata.get("episode_number", "")
            
            response = f"**{episode_id}: {title}** 📺\n"
            response += f"Season {season}, Episode {episode_num}\n\n"
            response += f"**Plot Summary:**\n{plot}\n\n"
            response += "Would you like to:\n"
            response += f"- See scenes from {episode_id}?\n"
            response += f"- Practice dialogue from this episode?\n"
            response += f"- Learn about cultural references in this episode?"
            
        else:
            # Multiple episodes
            response = f"Found {len(results)} episodes matching '{topic}':\n\n"
            
            for i, result in enumerate(results, 1):
                metadata = result["metadata"] 
                episode_id = metadata.get("episode_id", "")
                title = metadata.get("episode_title", "")
                plot = metadata.get("plot_text", "")
                
                response += f"{i}. **{episode_id}**: {title}\n"
                response += f"   📖 {plot[:150]}...\n\n"
            
            response += "Which episode would you like to learn more about?"
        
        return response

    # FEATURE 4: Scene Script Viewing
    def get_scene_script(self, condensed_intent: Dict[str, Any]) -> str:
        """Get scene script for viewing/practice"""
        topic = condensed_intent["topic"]
        details = condensed_intent["details"]
        
        # Extract episode and scene info
        episode_pattern = r'S(\d{2})E(\d{2})|Season\s+(\d+)\s+Episode\s+(\d+)'
        scene_pattern = r'scene\s+(\d+)'
        
        episode_match = re.search(episode_pattern, f"{topic} {details}", re.IGNORECASE)
        scene_match = re.search(scene_pattern, f"{topic} {details}", re.IGNORECASE)
        
        filter_conditions = {"chunk_type": "scene"}
        
        if episode_match:
            if episode_match.group(1):
                season = int(episode_match.group(1))
                episode = int(episode_match.group(2))
                episode_id = f"S{season:02d}E{episode:02d}"
            else:
                season = int(episode_match.group(3))
                episode = int(episode_match.group(4))
                episode_id = f"S{season:02d}E{episode:02d}"
            
            filter_conditions["episode_id"] = episode_id
            
            if scene_match:
                scene_number = int(scene_match.group(1))
                filter_conditions["scene_number"] = scene_number
                
                print(f"🔍 Looking for {episode_id} scene {scene_number}")
                
                # Get full script from local JSON file
                return self.get_script_from_firestore(episode_id, scene_number)
            else:
                print(f"🔍 Looking for scenes from {episode_id}")
        
        # Check for character mentions
        for char in self.characters.keys():
            if char.lower() in topic.lower() or char.lower() in details.lower():
                if "characters" not in filter_conditions:
                    filter_conditions["characters"] = {"$in": []}
                filter_conditions["characters"]["$in"].append(char)
        
        # Use Pinecone only for finding/recommending scenes, not getting full text
        results = self.query_pinecone(
            query=f"scene script dialogue {topic}",
            filter_conditions=filter_conditions,
            top_k=5
        )
        
        if not results:
            return "I couldn't find that scene. Try asking like 'Show me S01E01 scene 2' or 'Show me a Monica scene'"
        
        if len(results) == 1:
            # Single scene - get full script from local file
            metadata = results[0]["metadata"]
            scene_id = metadata.get("scene_id", "")
            episode_id = metadata.get("episode_id", "")
            scene_number = metadata.get("scene_number", 1)
            
            return self.get_script_from_firestore(episode_id, scene_number, scene_id)
            
        else:
            # Multiple scenes - show options
            response = f"Found {len(results)} scenes:\n\n"
            
            for i, result in enumerate(results[:5], 1):
                metadata = result["metadata"]
                scene_id = metadata.get("scene_id", "")
                location = metadata.get("location", "")
                characters = metadata.get("characters", [])
                description = metadata.get("scene_description", "")
                
                response += f"{i}. **{scene_id}** at {location}\n"
                response += f"   👥 {', '.join(characters[:3])}\n"
                if description:
                    response += f"   📝 {description}\n"
                response += "\n"
            
            response += "Which scene would you like to see the full script for?"
            response += "\nJust say the scene number (e.g., 'Show me scene 2')"
        
        return response
    
    def get_script_from_firestore(self, episode_id: str, scene_number: int, scene_id: str = None) -> str:
        """Get full script from Firestore"""
        try:
            print(f"📖 Loading script from Firestore for {episode_id}")
            
            # Query Firestore for the scene
            scenes_ref = self.db.collection('friends_scenes')
            
            # Build query based on available parameters
            if scene_id:
                # Direct lookup by scene_id
                doc = scenes_ref.document(scene_id).get()
                if doc.exists:
                    scene_data = doc.to_dict()
                else:
                    return f"❌ Scene not found: {scene_id}"
            else:
                # Query by episode_id and scene_number from metadata
                # Note: episode_id and scene_number are stored in metadata object
                docs = scenes_ref.get()
                matching_docs = []
                
                for doc in docs:
                    doc_data = doc.to_dict()
                    metadata = doc_data.get('metadata', {})
                    if (metadata.get('episode_id') == episode_id and 
                        (not scene_number or metadata.get('scene_number') == scene_number)):
                        matching_docs.append(doc)
                
                if not matching_docs:
                    return f"❌ Scene not found for {episode_id}, scene {scene_number}"
                
                scene_data = matching_docs[0].to_dict()
                scene_id = matching_docs[0].id
            
            # Extract scene information from metadata
            metadata = scene_data.get("metadata", {})
            location = metadata.get("location", "Unknown location") 
            characters = metadata.get("characters", [])
            description = metadata.get("scene_description", "")
            raw_text = scene_data.get("text", "")  # Raw text is stored in 'text' field
            
            # If no raw_text, try to construct from lines
            if not raw_text and "lines" in scene_data:
                lines = scene_data["lines"]
                dialogue_lines = []
                for line in lines:
                    if line.get("type") == "dialogue":
                        speaker = line.get("speaker", "")
                        text = line.get("text", "")
                        dialogue_lines.append(f"{speaker}: {text}")
                raw_text = "\n".join(dialogue_lines)
            
            response = f"**{scene_id}** 🎬\n"
            response += f"📍 Location: {location}\n"
            response += f"👥 Characters: {', '.join(characters[:5])}\n"
            if description:
                response += f"📝 Scene: {description}\n"
            response += "\n" + "="*60 + "\n"
            response += f"**FULL SCRIPT:**\n\n{raw_text}"
            
            # Show word/line count for reference
            word_count = len(raw_text.split())
            line_count = len(raw_text.split('\n'))
            response += f"\n\n📊 Script Stats: {word_count} words, {line_count} lines"
            response += "\n" + "="*60 + "\n"
            response += "Would you like to:\n"
            response += f"- Practice this scene as one of the characters?\n"
            response += f"- Explain any cultural references in this scene?\n"
            response += f"- See another scene from this episode?"
            
            return response
            
        except Exception as e:
            print(f"Error loading script from file: {e}")
            return f"❌ Error loading script: {e}\nTry asking for a different scene."

    # FEATURE 5: Cultural Context Explanations
    def explain_cultural_context(self, condensed_intent: Dict[str, Any]) -> str:
        """Explain cultural context and expressions"""
        topic = condensed_intent["topic"]
        details = condensed_intent["details"]
        original_message = condensed_intent.get("original_message", "")
        
        # Check if it's a known Friends expression first
        for expression, info in self.friends_expressions.items():
            if expression.lower() in topic.lower() or expression.lower() in details.lower():
                response = f"**'{expression}'** - {info['character']}'s signature! 🎭\n\n"
                response += f"**Meaning**: {info['meaning']}\n"
                response += f"**When to use**: {info['usage']}\n"
                response += f"**Context**: {info['context']}\n\n"
                response += f"**Example in Friends**: This is {info['character']}'s catchphrase that appears throughout the series.\n\n"
                response += "Want to practice using this expression or learn about other Friends phrases?"
                return response
        
        # Check for specific season request (e.g., "find it from s01")
        season_filter = None
        if "s01" in original_message.lower() or "season 1" in original_message.lower():
            season_filter = 1
        elif "s02" in original_message.lower() or "season 2" in original_message.lower():
            season_filter = 2
        # Add more seasons as needed...
        
        # Search for cultural references in scenes
        print(f"🔍 Searching for cultural context about: {topic}")
        
        filter_conditions = {"chunk_type": "scene"}
        if season_filter:
            filter_conditions["season"] = season_filter
            print(f"🎯 Filtering by Season {season_filter}")
        
        results = self.query_pinecone(
            query=f"{topic} {details} cultural reference idiom expression",
            filter_conditions=filter_conditions,
            top_k=5
        )
        
        # If no results, try broader search without filters
        if not results and season_filter:
            print("🔍 No results with season filter, trying broader search...")
            results = self.query_pinecone(
                query=f"{topic} {details} cultural reference idiom",
                filter_conditions={"chunk_type": "scene"},
                top_k=3
            )
        
        # Always try advanced prompt engineering first for better explanations
        explanation = self.get_direct_explanation(topic, details, original_message)
        if explanation:
            return explanation
        
        if not results:
            # Use fallback GPT explanation if direct explanation fails
            try:
                gpt_explanation = self.get_gpt_cultural_explanation(topic, details, original_message)
                return gpt_explanation
            except Exception as e:
                print(f"Error getting GPT explanation: {e}")
                return self.get_fallback_explanation(topic, details)
        
        # Show cultural context from scenes
        season_note = f" (Season {season_filter})" if season_filter else ""
        response = f"**Cultural Context: '{topic}'{season_note}** 🇺🇸\n\n"
        
        if len(results) == 1:
            # Single example - show more detail
            result = results[0]
            metadata = result["metadata"]
            episode_id = metadata.get("episode_id", "")
            location = metadata.get("location", "")
            scene_id = metadata.get("scene_id", "")
            
            # Get full scene from local file for better context
            full_scene = self.get_script_from_firestore(episode_id, metadata.get("scene_number", 1), scene_id)
            
            response += f"**Found in {episode_id}** at {location}:\n\n"
            response += "Here's the context from the actual scene:\n"
            response += f"(Use 'Show me {scene_id}' to see the full script)\n\n"
            
        else:
            # Multiple examples
            response += "Here's how this appears in Friends:\n\n"
            for i, result in enumerate(results, 1):
                metadata = result["metadata"]
                episode_id = metadata.get("episode_id", "")
                location = metadata.get("location", "")
                
                response += f"**Example {i}** - {episode_id} at {location}\n"
                # Show just the episode info, not the text snippet
                response += f"   💡 Found reference in this scene\n\n"
        
        # Add explanation based on the topic
        response += self.add_cultural_explanation(topic, details, original_message)
        
        return response
    
    def get_direct_explanation(self, topic: str, details: str, original_message: str) -> str:
        """Generate explanations using advanced prompt engineering"""
        
        try:
            # Advanced prompt engineering for cultural explanations
            system_prompt = """You are a specialized American cultural linguist and English teacher for Korean students learning through Friends TV show.

Your task: Analyze expressions, idioms, or cultural references and provide comprehensive explanations.

Response Format (always follow this EXACT structure):
**American Expression: '[EXPRESSION]'** 🇺🇸

**Meaning**: [One clear sentence explaining what it means]
**Origin**: [Where it comes from - keep it interesting but brief]
**When to use**: [Context and situations - practical advice]

**Examples in conversation**:
• "[Natural example 1]"
• "[Natural example 2]" 
• "[Natural example 3]"

**Similar expressions**: [2-3 alternatives they might hear]

💡 **Friends Context**: [How this might appear in Friends episodes or 90s American culture]

Rules:
- Keep explanations clear for non-native speakers
- Use conversational, engaging tone
- Include practical usage tips
- Focus on expressions common in American TV/movies
- If it's not a clear idiom/expression, explain the cultural concept instead
- Always end with the Friends context connection"""

            user_prompt = f"""Analyze this query from a Korean English learner: "{original_message}"

Key elements to explain:
- Topic: {topic}
- Details: {details}
- Context: Learning English through Friends TV show

Please provide a comprehensive cultural/linguistic explanation following the exact format specified."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=600,
                temperature=0.3  # Lower temperature for more consistent formatting
            )
            
            explanation = response.choices[0].message.content
            
            # Add follow-up suggestions
            explanation += "\n\n**What's next?**\n"
            explanation += "• 'Find episodes about [topic]' - See it in actual scenes\n"
            explanation += "• 'Practice as [character]' - Use this expression in conversation\n"
            explanation += "• 'Show me S01E01 scene 2' - See real dialogue examples"
            
            return explanation
            
        except Exception as e:
            print(f"Error generating cultural explanation: {e}")
            return None
    
    def get_gpt_cultural_explanation(self, topic: str, details: str, original_message: str) -> str:
        """Get explanation from GPT when no scenes found"""
        
        system_prompt = """You are an expert on American culture and English expressions. 
        Explain cultural references, idioms, and expressions that English learners might not understand.
        Focus on:
        1. The meaning and origin
        2. When and how it's used
        3. Examples in context
        4. Similar expressions
        Be helpful, educational, and engaging."""
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Explain this American expression or cultural reference: '{original_message}'. Topic: {topic}, Details: {details}"}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        explanation = response.choices[0].message.content
        
        formatted_response = f"**Cultural Context Explanation** 🇺🇸\n\n{explanation}\n\n"
        formatted_response += "💡 **Want to see more?**\n"
        formatted_response += "• Ask me to find specific episodes with this expression\n"
        formatted_response += "• Try 'Show me S01E01 scene 2' to see actual dialogue\n"
        formatted_response += "• Say 'Practice conversation' to use these expressions!"
        
        return formatted_response
    
    def get_fallback_explanation(self, topic: str, details: str) -> str:
        """Fallback explanation when everything else fails"""
        response = f"**Cultural Context Help** 🇺🇸\n\n"
        response += f"I'd be happy to explain '{topic}'! \n\n"
        response += "**Try asking like this:**\n"
        response += "• 'What does nail the interview mean?'\n"
        response += "• 'Explain break a leg'\n" 
        response += "• 'What is American dating culture?'\n"
        response += "• 'Show me S01E01 cultural references'\n\n"
        response += "**Or explore specific episodes:**\n"
        response += "• 'Show me S01E01 scene 2' - See actual dialogue\n"
        response += "• 'Find episodes about dating' - Get recommendations\n"
        response += "• 'Practice as Rachel' - Interactive practice"
        
        return response
    
    def add_cultural_explanation(self, topic: str, details: str, original_message: str) -> str:
        """Add relevant cultural explanation based on topic"""
        
        # Job interview culture
        if any(word in original_message.lower() for word in ['interview', 'job', 'work', 'career']):
            return """
💼 **American Job Interview Culture:**
• "Nailing the interview" = performing excellently
• Americans value confidence and self-promotion in interviews
• It's normal to ask questions about the company
• Follow-up thank you emails are expected
• Dress codes vary by industry (business formal vs. casual)

Want to practice job interview conversations from Friends episodes?"""
        
        # Dating culture  
        elif any(word in original_message.lower() for word in ['date', 'dating', 'relationship', 'boyfriend', 'girlfriend']):
            return """
💕 **American Dating Culture in the 90s (Friends era):**
• Dating multiple people before being "exclusive" was normal
• "Going Dutch" = splitting the bill
• Meeting through friends was very common (no dating apps!)
• Coffee dates were popular casual first dates
• "The three-day rule" = wait 3 days before calling after a date

Want to see dating scenes from Friends episodes?"""
        
        # General friendship culture
        elif any(word in original_message.lower() for word in ['friend', 'friendship', 'hang out']):
            return """
👥 **American Friendship Culture:**
• Close friends often share very personal details
• "Hanging out" = spending casual time together
• Friends often give dating advice (sometimes unwanted!)
• Group dynamics are important (like the Friends group!)
• Sarcasm and teasing show closeness

Want to practice friendship conversations from the show?"""
        
        else:
            return """
Would you like me to:
• Find more examples from specific episodes?
• Explain other cultural references?
• Show you scenes with this topic?
• Start a practice conversation using these concepts?"""

    # FEATURE 6: Practice Session
    def start_practice_session(self, condensed_intent: Dict[str, Any]) -> str:
        """Start conversation practice session"""
        response = "🎭 **Let's start your Friends conversation practice!** 🎭\n\n"
        response += "To begin, I need to know:\n\n"
        response += "1. **Which episode?** (e.g., 'S01E01' or 'the pilot episode')\n"
        response += "2. **Which character do you want to practice as?**\n"
        response += "   - Monica (chef, organized, fast-talking)\n"
        response += "   - Rachel (fashion-focused, growing confident)\n"
        response += "   - Ross (academic, explains things in detail)\n"
        response += "   - Chandler (sarcastic, witty comebacks)\n"
        response += "   - Joey (simple, food-loving, casual)\n"
        response += "   - Phoebe (eccentric, honest, spiritual)\n\n"
        response += "3. **Specific scene?** (optional - I can recommend good practice scenes)\n\n"
        response += "Example: 'I want to practice as Monica in S01E01 scene 2'\n\n"
        response += "Once you tell me these details, I'll:\n"
        response += "- Show you the scene context\n"
        response += "- Play other characters' lines\n"
        response += "- Wait for you to type your character's responses\n"
        response += "- Give you feedback on accuracy and suggestions"
        
        return response
    
    def parse_practice_request(self, user_message: str, context: ChatContext = None) -> Dict[str, str]:
        """Parse practice session request"""
        # Extract episode from current message
        episode_pattern = r'S(\d{2})E(\d{2})|Season\s+(\d+)\s+Episode\s+(\d+)'
        episode_match = re.search(episode_pattern, user_message, re.IGNORECASE)
        
        episode_id = ""
        scene_number = 0
        if episode_match:
            if episode_match.group(1):
                season = int(episode_match.group(1))
                episode = int(episode_match.group(2))
                episode_id = f"S{season:02d}E{episode:02d}"
            else:
                season = int(episode_match.group(3))
                episode = int(episode_match.group(4))
                episode_id = f"S{season:02d}E{episode:02d}"
        
        # If no episode found in current message, check conversation history
        if not episode_id and context and context.conversation_history:
            for msg in reversed(context.conversation_history[-5:]):  # Check last 5 messages
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    episode_match = re.search(episode_pattern, content, re.IGNORECASE)
                    if episode_match:
                        if episode_match.group(1):
                            season = int(episode_match.group(1))
                            episode = int(episode_match.group(2))
                            episode_id = f"S{season:02d}E{episode:02d}"
                        else:
                            season = int(episode_match.group(3))
                            episode = int(episode_match.group(4))
                            episode_id = f"S{season:02d}E{episode:02d}"
                        
                        # Also check for scene number in the same message (e.g., S09E19_002)
                        scene_id_pattern = r'S\d{2}E\d{2}_(\d{3})'
                        scene_id_match = re.search(scene_id_pattern, content)
                        if scene_id_match:
                            scene_number = int(scene_id_match.group(1))
                        break
        
        # Extract character
        character = ""
        for char in self.characters.keys():
            if char.lower() in user_message.lower():
                character = char
                break
        
        # Extract scene from current message (only if not already found from history)
        if scene_number == 0:
            # First try to find scene_id pattern (e.g., S09E19_013)
            scene_id_pattern = r'S\d{2}E\d{2}_(\d{3})'
            scene_id_match = re.search(scene_id_pattern, user_message)
            if scene_id_match:
                scene_number = int(scene_id_match.group(1))
            else:
                # Then try scene pattern (e.g., "scene 2")
                scene_pattern = r'scene\s+(\d+)'
                scene_match = re.search(scene_pattern, user_message, re.IGNORECASE)
                scene_number = int(scene_match.group(1)) if scene_match else 0
        
        return {
            "episode_id": episode_id,
            "character": character,
            "scene_number": scene_number
        }
    
    def run_practice_session(self, episode_id: str, character: str, scene_number: int = 0, context: ChatContext = None):
        """Run interactive practice session"""
        print(f"\n🎭 Starting practice session as {character} in {episode_id}")
        
        # Get scene from local JSON file instead of Pinecone
        if scene_number:
            scene_data = self.get_scene_data_from_firestore(episode_id, scene_number)
        else:
            # Find a good scene with this character
            scene_data = self.find_character_scene_from_firestore(episode_id, character)
        
        if not scene_data:
            print(f"❌ No scenes found for {character} in {episode_id}")
            return
        
        # Extract data from metadata
        metadata = scene_data.get("metadata", {})
        scene_text = scene_data.get("text", "")  # Raw text is in 'text' field
        scene_id = metadata.get("scene_id", "")
        location = metadata.get("location", "Unknown")
        characters = metadata.get("characters", [])
        
        print(f"\n🎬 Scene: {scene_id}")
        print(f"📍 Location: {location}")
        print(f"👥 Characters: {', '.join(characters)}")
        print("\n" + "="*60)
        
        # Parse dialogue lines
        lines = self.parse_scene_dialogue(scene_text)
        
        # Practice session
        user_lines = [line for line in lines if line.get('speaker') == character]
        total_lines = len(user_lines)
        correct_answers = 0
        
        print(f"You'll be practicing as {character}. When it's your turn, type your line!")
        print(f"Total lines to practice: {total_lines}")
        print("\nPress Enter to start...")
        input()
        
        current_line = 0
        for line in lines:
            speaker = line.get('speaker', '')
            text = line.get('text', '')
            line_type = line.get('type', 'dialogue')
            
            if line_type == 'action':
                print(f"📝 {text}")
                continue
            elif line_type == 'narration':
                print(f"🎬 {text}")
                continue
            
            if speaker == character:
                # User's turn
                current_line += 1
                print(f"\n[YOUR TURN - {character}]")
                print("Expected:", text)
                
                user_input = input(f"{character}: ").strip()
                
                if user_input.lower() == 'skip':
                    print("⏭️ Skipped!")
                    continue
                elif user_input.lower() == 'quit':
                    break
                
                # Calculate similarity
                similarity = self.calculate_text_similarity(user_input, text)
                
                if similarity > 0.8:
                    print("✅ Excellent! Perfect match!")
                    correct_answers += 1
                elif similarity > 0.6:
                    print("👍 Good! Close enough.")
                    correct_answers += 1
                elif similarity > 0.4:
                    print("🤔 Not quite right, but you got the idea.")
                else:
                    print("❌ Try again! The correct line was:")
                    print(f"   '{text}'")
                
                print(f"Similarity: {similarity:.1%}")
                
            else:
                # Other character's line
                print(f"{speaker}: {text}")
                input("  (Press Enter to continue...)")
        
        # Show results
        print("\n" + "="*60)
        print("🏁 PRACTICE SESSION COMPLETE!")
        print("="*60)
        
        if total_lines > 0:
            accuracy = correct_answers / total_lines
            print(f"Character practiced: {character}")
            print(f"Correct answers: {correct_answers}/{total_lines}")
            print(f"Accuracy rate: {accuracy:.1%}")
            
            if accuracy >= 0.9:
                print("🌟 Outstanding! You've mastered this character!")
            elif accuracy >= 0.7:
                print("🎉 Great job! You're getting the hang of it!")
            elif accuracy >= 0.5:
                print("👍 Good effort! Keep practicing to improve.")
            else:
                print("💪 Keep practicing! You'll get better with time.")
        
        print("\nWould you like to:")
        print("1. Practice another scene?")
        print("2. Try a different character?") 
        print("3. Get episode recommendations?")
        
        next_action = input("\nWhat would you like to do next? ").strip()
        
        if '1' in next_action or 'scene' in next_action.lower():
            print("Great! Ask me for another scene to practice.")
        elif '2' in next_action or 'character' in next_action.lower():
            print("Perfect! Tell me which character you'd like to practice as.")
        elif '3' in next_action or 'recommend' in next_action.lower():
            print("Awesome! Tell me what topic or situation you want to practice.")
    
    def parse_scene_dialogue(self, scene_text: str) -> List[Dict[str, str]]:
        """Parse scene text into individual dialogue lines"""
        lines = []
        
        for line in scene_text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Action/narration in parentheses
            if line.startswith('(') and line.endswith(')'):
                lines.append({
                    'type': 'action',
                    'text': line,
                    'speaker': ''
                })
            # Scene headers
            elif line.startswith('[') and line.endswith(']'):
                lines.append({
                    'type': 'narration', 
                    'text': line,
                    'speaker': ''
                })
            # Dialogue: "Speaker: text"
            elif ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    speaker = parts[0].strip()
                    text = parts[1].strip()
                    lines.append({
                        'type': 'dialogue',
                        'speaker': speaker,
                        'text': text
                    })
            else:
                # Other narration
                lines.append({
                    'type': 'narration',
                    'text': line,
                    'speaker': ''
                })
        
        return lines
    
    def calculate_text_similarity(self, user_input: str, expected_text: str) -> float:
        """Calculate similarity between user input and expected dialogue"""
        
        # Step 1: Extract actual dialogue from expected text (remove action descriptions)
        actual_dialogue = self.extract_dialogue_only(expected_text)
        
        # Step 2: Clean both texts
        user_clean = self.clean_text_for_comparison(user_input)
        expected_clean = self.clean_text_for_comparison(actual_dialogue)
        
        # Step 3: Multiple comparison methods
        
        # Exact match (highest score)
        if user_clean == expected_clean:
            return 1.0
        
        # Very close match (minor differences)
        if self.is_very_close_match(user_clean, expected_clean):
            return 0.95
        
        # Word-based similarity (fast, no API calls)
        word_similarity = self.calculate_word_similarity(user_clean, expected_clean)
        
        # If word similarity is high enough, don't bother with expensive embedding
        if word_similarity >= 0.8:
            return word_similarity
        
        # Character-based similarity for short phrases
        if len(expected_clean) <= 20:  # Short phrases
            char_similarity = self.calculate_character_similarity(user_clean, expected_clean)
            return max(word_similarity, char_similarity)
        
        # For longer text, use embedding only if really needed
        if word_similarity < 0.4:
            try:
                return self.calculate_embedding_similarity(user_clean, expected_clean)
            except:
                return word_similarity
        
        return word_similarity
    
    def extract_dialogue_only(self, text: str) -> str:
        """Extract just the dialogue part, removing action descriptions"""
        # Remove parenthetical actions like "(mortified)", "(laughing)", etc.
        import re
        
        # Remove actions in parentheses
        text = re.sub(r'\([^)]*\)', '', text)
        
        # Remove stage directions in brackets  
        text = re.sub(r'\[[^\]]*\]', '', text)
        
        # Clean up extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def clean_text_for_comparison(self, text: str) -> str:
        """Clean text for fair comparison"""
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove common punctuation that doesn't affect meaning
        text = text.replace('.', '').replace('!', '').replace('?', '').replace(',', '')
        text = text.replace('"', '').replace("'", '').replace('-', ' ')
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def is_very_close_match(self, text1: str, text2: str) -> bool:
        """Check if texts are very close (minor typos, etc.)"""
        # Same length, most characters match
        if len(text1) == len(text2):
            differences = sum(1 for a, b in zip(text1, text2) if a != b)
            if differences <= 1:  # At most 1 character different
                return True
        
        # One character added or removed
        if abs(len(text1) - len(text2)) == 1:
            shorter, longer = (text1, text2) if len(text1) < len(text2) else (text2, text1)
            for i in range(len(longer)):
                if longer[:i] + longer[i+1:] == shorter:
                    return True
        
        return False
    
    def calculate_word_similarity(self, text1: str, text2: str) -> float:
        """Fast word-based similarity calculation"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard = len(intersection) / len(union)
        
        # Bonus for same word count
        if len(words1) == len(words2):
            jaccard += 0.1
        
        return min(1.0, jaccard)
    
    def calculate_character_similarity(self, text1: str, text2: str) -> float:
        """Character-level similarity for short phrases"""
        if not text1 or not text2:
            return 0.0
        
        # Levenshtein distance ratio
        def levenshtein_ratio(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_ratio(s2, s1)
            
            if len(s2) == 0:
                return 0.0
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            max_len = max(len(s1), len(s2))
            return (max_len - previous_row[-1]) / max_len
        
        return levenshtein_ratio(text1, text2)
    
    def calculate_embedding_similarity(self, text1: str, text2: str) -> float:
        """Expensive but accurate embedding-based similarity"""
        try:
            emb1 = self.get_embedding(text1)
            emb2 = self.get_embedding(text2)
            
            if not emb1 or not emb2:
                return 0.0
            
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            return max(0.0, float(similarity))
            
        except Exception as e:
            print(f"Error with embedding similarity: {e}")
            return 0.0
    
    def get_scene_data_from_firestore(self, episode_id: str, scene_number: int) -> Dict:
        """Get scene data from Firestore"""
        try:
            # Query Firestore for the scene
            scenes_ref = self.db.collection('friends_scenes')
            docs = scenes_ref.get()
            
            for doc in docs:
                doc_data = doc.to_dict()
                metadata = doc_data.get('metadata', {})
                if (metadata.get('episode_id') == episode_id and 
                    metadata.get('scene_number') == scene_number):
                    return doc_data
            
            return None
            
        except Exception as e:
            print(f"Error reading scene from Firestore: {e}")
            return None
    
    def find_character_scene_from_firestore(self, episode_id: str, character: str) -> Dict:
        """Find a good scene with specific character from Firestore"""
        try:
            # Query Firestore for scenes with the character
            scenes_ref = self.db.collection('friends_scenes')
            docs = scenes_ref.get()
            
            # Find scenes with this character in metadata
            character_scenes = []
            for doc in docs:
                doc_data = doc.to_dict()
                metadata = doc_data.get('metadata', {})
                if (metadata.get('episode_id') == episode_id and 
                    character in metadata.get('characters', [])):
                    character_scenes.append(doc_data)
            
            if not character_scenes:
                print(f"❌ No scenes found with {character} in {episode_id}")
                return None
            
            # Return the first scene with good dialogue length
            for scene in character_scenes:
                raw_text = scene.get("raw_text", "")
                if not raw_text and "lines" in scene:
                    # Construct from lines if raw_text not available
                    lines = scene["lines"]
                    dialogue_lines = []
                    for line in lines:
                        if line.get("type") == "dialogue":
                            speaker = line.get("speaker", "")
                            text = line.get("text", "")
                            dialogue_lines.append(f"{speaker}: {text}")
                    raw_text = "\n".join(dialogue_lines)
                
                if len(raw_text) > 200:  # Good length for practice
                    return scene
            
            # Fallback to first scene
            return character_scenes[0] if character_scenes else None
            
        except Exception as e:
            print(f"Error finding character scene from Firestore: {e}")
            return None

    def general_friends_chat(self, condensed_intent: Dict[str, Any]) -> str:
        """General conversation about Friends"""
        topic = condensed_intent["topic"]
        
        system_prompt = """
        You are a friendly Friends TV show expert and English learning assistant. 
        Help users learn English through Friends episodes. Be encouraging, informative, and always suggest specific ways to practice English with Friends content.
        Keep responses conversational and helpful.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User is asking about: {topic}"}
                ],
                max_tokens=300
            )
            
            ai_response = response.choices[0].message.content
            
            # Add suggestions for specific features
            ai_response += "\n\n💡 **What would you like to do next?**\n"
            ai_response += "- 'Recommend episodes about [topic]' - Get episode suggestions\n"
            ai_response += "- 'Tell me about [character]' - Learn about characters\n"
            ai_response += "- 'Show me [episode] script' - See episode scenes\n"
            ai_response += "- 'Explain [phrase/reference]' - Cultural explanations\n"
            ai_response += "- 'Start practice session' - Practice conversations"
            
            return ai_response
            
        except Exception as e:
            return f"I'd love to chat about Friends and help you learn English! What specific aspect of Friends interests you? Episodes, characters, or maybe you want to start practicing conversations?"

    def respond(self, query_results: List[Dict], condensed_intent: Dict[str, Any], function_name: str) -> str:
        """Step 4: Generate final response"""
        
        # Route to the appropriate function
        if function_name == "recommend_episodes":
            return self.recommend_episodes(condensed_intent)
        elif function_name == "get_character_info":
            return self.get_character_info(condensed_intent)
        elif function_name == "get_episode_plot":
            return self.get_episode_plot(condensed_intent)
        elif function_name == "get_scene_script":
            return self.get_scene_script(condensed_intent)
        elif function_name == "explain_cultural_context":
            return self.explain_cultural_context(condensed_intent)
        elif function_name == "start_practice_session":
            return self.start_practice_session(condensed_intent)
        else:
            return self.general_friends_chat(condensed_intent)

    def chat(self, user_message: str, context: ChatContext) -> str:
        """Main chat function implementing RAG pipeline"""
        
        print(f"\n👤 User: {user_message}")
        print("🤖 Processing...")
        
        # Add to conversation history
        context.conversation_history.append({"role": "user", "content": user_message})
        
        # Check if this is a practice session request with specific details
        practice_keywords = ['practice', 'start practice', 'practice as', 'practice session']
        
        # Also check for episode + character pattern (e.g., "S01E01 Ross")
        practice_details = self.parse_practice_request(user_message, context)
        is_practice_request = (
            any(keyword in user_message.lower() for keyword in practice_keywords) or
            (practice_details['episode_id'] and practice_details['character'])
        )
        
        if is_practice_request and practice_details['episode_id'] and practice_details['character']:
            print(f"🎭 Detected practice request: {practice_details}")
            
            # Start interactive practice session
            self.run_practice_session(
                episode_id=practice_details['episode_id'],
                character=practice_details['character'], 
                scene_number=practice_details['scene_number'],
                context=context
            )
            return "Practice session completed! What would you like to do next?"
        
        # Step 1: Condense user intent
        condensed_intent = self.condense_user_intent(
            user_message, 
            [msg["content"] for msg in context.conversation_history[-10:]]
        )
        
        print(f"🎯 Intent: {condensed_intent['intent']}")
        print(f"📝 Topic: {condensed_intent['topic']}")
        
        # Step 2: Route to function
        function_name = self.route_to_function(condensed_intent)
        print(f"🔀 Route: {function_name}")
        
        # Step 3: Query (handled within individual functions)
        # Step 4: Respond
        response = self.respond([], condensed_intent, function_name)
        
        # Add to conversation history
        context.conversation_history.append({"role": "assistant", "content": response})
        
        return response

def main():
    """Main function to run the chatbot"""
    print("="*60)
    print("🎭 FRIENDS ENGLISH CONVERSATION PRACTICE CHATBOT 🎭")
    print("="*60)
    print("RAG System: Condense → Route → Query → Respond")
    print("\nAvailable commands:")
    print("- Type your message normally")
    print("- 'help' - Show features")
    print("- 'quit' - Exit chatbot")
    print("="*60)
    
    try:
        # Initialize chatbot
        chatbot = FriendsRAGChatbot()
        
        # Create user context
        context = ChatContext(
            user_id="demo_user",
            conversation_history=[]
        )
        
        print("\n🤖 Bot: Hi! I'm your Friends English practice buddy! 👋")
        print("I can help you:")
        print("- Find episodes perfect for your learning goals")
        print("- Learn about the characters and their speech patterns")  
        print("- Explore episode plots and scenes")
        print("- Explain cultural references and American expressions")
        print("- Practice conversations from actual Friends scenes")
        print("\nWhat would you like to start with?")
        
        while True:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n🤖 Bot: Thanks for practicing English with Friends! Keep watching and learning! 👋")
                break
            
            if user_input.lower() == 'help':
                print("\n🤖 Bot: Here's what I can help you with:")
                print("1. 📺 Episode Recommendations: 'Find episodes about dating' or 'Show me funny episodes'")
                print("2. 👥 Character Info: 'Tell me about Monica' or 'Who is Chandler?'")
                print("3. 📖 Episode Plots: 'What happens in S01E01?' or 'Tell me about the pilot'")
                print("4. 🎬 Scene Scripts: 'Show me S01E01 scene 2' or 'Monica scene with Rachel'")
                print("5. 🇺🇸 Cultural Context: 'What does How you doin mean?' or 'Explain American dating'")
                print("6. 🎭 Practice Sessions: 'Start practice session' or 'I want to practice as Joey'")
                continue
            
            if not user_input:
                continue
            
            try:
                response = chatbot.chat(user_input, context)
                print(f"\n🤖 Bot: {response}")
                
            except Exception as e:
                print(f"\n🤖 Bot: Sorry, I had a technical issue: {e}")
                print("Please try asking again or type 'help' for guidance.")
    
    except KeyboardInterrupt:
        print("\n\n👋 Thanks for using Friends English Chatbot!")
    except Exception as e:
        print(f"\nError initializing chatbot: {e}")
        print("Make sure you have:")
        print("1. OPENAI_API_KEY in your .env file")
        print("2. PINECONE_API_KEY in your .env file")
        print("3. Required packages installed: pip install openai pinecone python-dotenv scikit-learn")

if __name__ == "__main__":
    main()