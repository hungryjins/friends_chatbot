#!/usr/bin/env python3
"""
Test script for Friends English Chatbot
"""

import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from friends_chatbot import FriendsRAGChatbot, ChatContext

def test_chatbot_features():
    """Test all 6 core features of the chatbot"""
    
    print("🧪 Testing Friends English Chatbot...")
    print("="*50)
    
    # Load environment
    load_dotenv()
    
    # Check API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment")
        return False
    if not os.getenv("PINECONE_API_KEY"):
        print("❌ PINECONE_API_KEY not found in environment")
        return False
    
    try:
        # Initialize chatbot
        print("🔄 Initializing chatbot...")
        chatbot = FriendsRAGChatbot()
        
        # Create test context
        context = ChatContext(
            user_id="test_user",
            conversation_history=[]
        )
        
        print("✅ Chatbot initialized successfully!")
        print("\n" + "="*50)
        
        # Test cases for each feature
        test_cases = [
            {
                "feature": "Episode Recommendation",
                "query": "I want to learn about dating conversations",
                "expected_intent": "episode_recommendation"
            },
            {
                "feature": "Character Information", 
                "query": "Tell me about Monica",
                "expected_intent": "character_info"
            },
            {
                "feature": "Episode Plot",
                "query": "What happens in S01E01?",
                "expected_intent": "plot_summary"
            },
            {
                "feature": "Scene Script",
                "query": "Show me S01E01 scene 2",
                "expected_intent": "scene_script"
            },
            {
                "feature": "Cultural Context",
                "query": "What does 'How you doin' mean?",
                "expected_intent": "cultural_context"
            },
            {
                "feature": "Practice Session",
                "query": "I want to start practicing conversations",
                "expected_intent": "practice_session"
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 Test {i}: {test_case['feature']}")
            print(f"Query: '{test_case['query']}'")
            print("-" * 30)
            
            try:
                # Test intent recognition
                condensed_intent = chatbot.condense_user_intent(
                    test_case['query'], 
                    []
                )
                
                detected_intent = condensed_intent['intent']
                print(f"🎯 Detected Intent: {detected_intent}")
                print(f"📝 Topic: {condensed_intent['topic']}")
                
                # Test routing
                function_name = chatbot.route_to_function(condensed_intent)
                print(f"🔀 Routed to: {function_name}")
                
                # Test full response
                response = chatbot.chat(test_case['query'], context)
                
                print(f"✅ Response Length: {len(response)} characters")
                print(f"📤 Response Preview: {response[:100]}...")
                
                # Check if expected intent matches
                intent_match = detected_intent == test_case['expected_intent']
                print(f"🎯 Intent Match: {'✅' if intent_match else '❌'}")
                
                results.append({
                    "feature": test_case['feature'],
                    "intent_match": intent_match,
                    "response_generated": len(response) > 0,
                    "error": None
                })
                
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append({
                    "feature": test_case['feature'],
                    "intent_match": False,
                    "response_generated": False,
                    "error": str(e)
                })
        
        # Summary
        print("\n" + "="*50)
        print("🏁 TEST RESULTS SUMMARY")
        print("="*50)
        
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r['intent_match'] and r['response_generated'])
        
        for result in results:
            status = "✅" if result['intent_match'] and result['response_generated'] else "❌"
            error_msg = f" ({result['error']})" if result['error'] else ""
            print(f"{status} {result['feature']}{error_msg}")
        
        print(f"\n📊 Success Rate: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)")
        
        if successful_tests == total_tests:
            print("🎉 All tests passed! Chatbot is ready for use.")
        else:
            print("⚠️ Some tests failed. Check the errors above.")
        
        return successful_tests == total_tests
        
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        return False

def interactive_test():
    """Run interactive test session"""
    
    print("\n" + "="*50)
    print("🎮 INTERACTIVE TEST MODE")
    print("="*50)
    print("Test the chatbot interactively!")
    print("Commands:")
    print("- Type any message to test")
    print("- 'quit' to exit")
    print("- 'test [feature]' to test specific feature")
    print("="*50)
    
    try:
        chatbot = FriendsRAGChatbot()
        context = ChatContext(user_id="interactive_test", conversation_history=[])
        
        sample_queries = {
            "recommendation": "Find me episodes about friendship",
            "character": "Who is Chandler?", 
            "plot": "Tell me about the pilot episode",
            "script": "Show me a funny Monica scene",
            "context": "Explain 'We were on a break'",
            "practice": "I want to practice as Joey"
        }
        
        while True:
            user_input = input("\n🧪 Test Query: ").strip()
            
            if user_input.lower() == 'quit':
                break
                
            if user_input.lower().startswith('test '):
                feature = user_input[5:].strip().lower()
                if feature in sample_queries:
                    user_input = sample_queries[feature]
                    print(f"🔄 Testing with: {user_input}")
                else:
                    print(f"Available features: {', '.join(sample_queries.keys())}")
                    continue
            
            if not user_input:
                continue
            
            try:
                response = chatbot.chat(user_input, context)
                print(f"\n🤖 Response:\n{response}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
    
    except Exception as e:
        print(f"❌ Failed to start interactive test: {e}")

def main():
    """Main test function"""
    
    print("🎭 Friends English Chatbot - Test Suite")
    print("Choose test mode:")
    print("1. Automated feature tests")
    print("2. Interactive testing")
    print("3. Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice in ['1', '3']:
        success = test_chatbot_features()
        if not success:
            print("\n⚠️ Some automated tests failed!")
    
    if choice in ['2', '3']:
        interactive_test()
    
    print("\n👋 Testing complete!")

if __name__ == "__main__":
    main()