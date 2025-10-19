#!/usr/bin/env python3
"""
Simple ReservoirPy Chat Demo

A minimal example showing the ReservoirPy LLM inference engine functionality.
This demo works without requiring actual API keys by showing the architecture.
"""

import asyncio
import os
import sys
# Add the parent directory to the path to import reservoirpy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reservoirpy.cognitive import create_chat_engine


async def demo_basic_functionality():
    """Demonstrate basic chat functionality without real API calls."""
    print("🧠 ReservoirPy LLM Inference Engine Demo")
    print("=" * 50)
    
    try:
        # Create chat engine (will work without real API key for basic functions)
        print("📝 Creating chat engine...")
        engine = create_chat_engine(
            llm_provider="cohere",
            llm_config={"api_key": "demo-key-for-testing"}  # Mock key
        )
        print("✅ Chat engine created successfully")
        
        # Test session management
        print("\n💬 Testing session management...")
        session_id = await engine.create_session(title="Demo Session")
        print(f"✅ Session created: {session_id[:8]}...")
        
        # List sessions
        sessions = await engine.list_sessions()
        print(f"✅ Sessions available: {len(sessions)}")
        
        # Show session details
        session = await engine.get_session(session_id)
        if session:
            print(f"✅ Session retrieved: {session.title}")
        
        # Test knowledge addition (this works without API calls)
        print("\n📚 Testing knowledge expansion...")
        node_id = await engine.add_knowledge_from_message(
            content="ReservoirPy is a Python library for reservoir computing with Echo State Networks",
            source="demo"
        )
        if node_id:
            print(f"✅ Knowledge added successfully: {node_id}")
        
        # Show conversation history (empty initially)
        history = await engine.get_conversation_history(session_id)
        print(f"✅ Conversation history: {len(history)} messages")
        
        # Export session data
        export_data = await engine.export_session(session_id)
        if export_data:
            print(f"✅ Session export successful: {len(export_data)} fields")
        
        print("\n🎉 Basic functionality demonstration completed!")
        print("\nNext steps to enable full functionality:")
        print("1. Set COHERE_API_KEY environment variable for Cohere")
        print("2. Or set OPENAI_API_KEY environment variable for OpenAI")
        print("3. Run the web interface with: python examples/reservoirpy_chat_demo.py --mode web")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


def show_architecture():
    """Show the architecture of the ReservoirPy LLM inference engine."""
    print("\n🏗️ ReservoirPy LLM Inference Engine Architecture")
    print("=" * 60)
    
    architecture = """
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │  Web Interface  │    │  Chat Engine    │    │ LLM Interface   │
    │  (FastAPI)      │◄──►│  (Orchestrator) │◄──►│ (Cohere/OpenAI) │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
                                     │
                                     ▼
                          ┌─────────────────┐    ┌─────────────────┐
                          │  GraphRAG       │◄──►│  AtomSpace      │
                          │  (Knowledge)    │    │  (ReservoirPy)  │
                          └─────────────────┘    └─────────────────┘
                                     │
                                     ▼
                          ┌─────────────────┐
                          │  Cognitive      │
                          │  Orchestrator   │
                          │  (Agents)       │
                          └─────────────────┘
    
    Components:
    
    1. Web Interface (FastAPI)
       - REST API endpoints (/api/chat, /api/sessions, etc.)
       - HTML chat interface (similar to chat.reservoirpy.inria.fr)
       - Streaming support for real-time responses
    
    2. Chat Engine (ReservoirPyChatEngine)
       - Session and conversation management
       - Integration between all components
       - Message routing and context handling
    
    3. LLM Interface (Cohere/OpenAI)
       - Text generation and embeddings
       - Multiple provider support
       - Async streaming capabilities
    
    4. GraphRAG Engine
       - Knowledge graph enhanced retrieval
       - Context-aware response generation
       - Integration with AtomSpace knowledge
    
    5. AtomSpace (ReservoirPy Cognitive)
       - ConceptNode, PredicateNode storage
       - Reservoir computing dynamics
       - Cognitive agent networks
    
    6. Cognitive Orchestrator
       - AttentionAgent, PatternMatchAgent
       - Distributed processing
       - Multi-phase cognitive cycles
    """
    
    print(architecture)


async def main():
    """Main demo function."""
    await demo_basic_functionality()
    show_architecture()


if __name__ == "__main__":
    asyncio.run(main())