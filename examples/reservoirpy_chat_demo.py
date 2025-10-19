#!/usr/bin/env python3
"""
ReservoirPy Chat Engine Demonstration

This example shows how to use the new LLM inference engine that combines
ReservoirPy's cognitive architecture with GraphRAG and Cohere/OpenAI for
intelligent conversational AI.

The demo creates a chat system that:
- Uses ReservoirPy's AtomSpace for knowledge representation
- Integrates GraphRAG for context retrieval
- Provides LLM-based natural language generation
- Supports both programmatic and web interfaces

Usage:
    python examples/reservoirpy_chat_demo.py --mode [cli|web|api]
    
    # For web interface, set API key:
    export COHERE_API_KEY=your_cohere_api_key
    # or
    export OPENAI_API_KEY=your_openai_api_key
"""

import asyncio
import argparse
import os
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import ReservoirPy cognitive components
try:
    from reservoirpy.cognitive import (
        create_chat_engine,
        create_web_interface,
        ReservoirPyChatEngine
    )
    print("✅ ReservoirPy cognitive components imported successfully")
except ImportError as e:
    print(f"❌ Failed to import ReservoirPy cognitive components: {e}")
    print("Make sure you have installed all dependencies: pip install cohere llama-index fastapi uvicorn")
    exit(1)


class ReservoirPyChatDemo:
    """Demonstration of ReservoirPy chat capabilities."""
    
    def __init__(self, llm_provider: str = "cohere", api_key: str = None):
        """
        Initialize the demo.
        
        Parameters
        ----------
        llm_provider : str
            LLM provider ("cohere" or "openai")
        api_key : str, optional
            API key for the LLM provider
        """
        self.llm_provider = llm_provider
        self.api_key = api_key
        self.chat_engine = None
        
    async def initialize(self):
        """Initialize the chat engine."""
        print(f"🧠 Initializing ReservoirPy Chat Engine with {self.llm_provider}...")
        
        # Setup LLM configuration
        llm_config = {}
        if self.api_key:
            llm_config["api_key"] = self.api_key
        elif self.llm_provider == "cohere":
            llm_config["api_key"] = os.getenv("COHERE_API_KEY")
        elif self.llm_provider == "openai":
            llm_config["api_key"] = os.getenv("OPENAI_API_KEY")
        
        if not llm_config.get("api_key"):
            print(f"⚠️  No API key found for {self.llm_provider}. Set {self.llm_provider.upper()}_API_KEY environment variable.")
            print("Demo will run with mock responses.")
        
        # Create chat engine
        self.chat_engine = create_chat_engine(
            llm_provider=self.llm_provider,
            llm_config=llm_config,
            cognitive_config={
                "network_size": 3  # Smaller network for demo
            }
        )
        
        # Initialize the engine
        try:
            await self.chat_engine.initialize()
            print("✅ Chat engine initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize chat engine: {e}")
            print("Running in demo mode without LLM...")
            return False
    
    async def demo_basic_chat(self):
        """Demonstrate basic chat functionality."""
        print("\n" + "="*60)
        print("🗨️  BASIC CHAT DEMONSTRATION")
        print("="*60)
        
        if not await self.initialize():
            return
        
        # Sample questions about ReservoirPy
        questions = [
            "What is ReservoirPy?",
            "How do Echo State Networks work?",
            "What are the main features of ReservoirPy?",
            "Can you explain reservoir computing?",
            "How do I create a simple ESN with ReservoirPy?"
        ]
        
        session_id = await self.chat_engine.create_session(title="Demo Chat Session")
        print(f"📝 Created chat session: {session_id[:8]}...")
        
        for i, question in enumerate(questions, 1):
            print(f"\n👤 Question {i}: {question}")
            
            try:
                response = await self.chat_engine.chat(
                    message=question,
                    session_id=session_id,
                    use_cognitive_processing=True
                )
                
                print(f"🤖 Assistant: {response['message'][:200]}...")
                
                # Show cognitive context info
                context = response['cognitive_context']
                if context.get('context_used'):
                    print(f"🧠 Cognitive Context: Retrieved {context['retrieved_nodes']} concepts, {context['retrieved_edges']} relations")
                else:
                    print("💭 Direct LLM response (no cognitive context)")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
            
            # Short delay between questions
            await asyncio.sleep(1)
        
        print("\n✅ Basic chat demonstration completed!")
    
    async def demo_knowledge_expansion(self):
        """Demonstrate adding knowledge to the system."""
        print("\n" + "="*60)
        print("📚 KNOWLEDGE EXPANSION DEMONSTRATION")
        print("="*60)
        
        if not await self.initialize():
            return
        
        # Add some custom knowledge
        knowledge_items = [
            "ReservoirPy version 2.0 introduced improved JAX support for GPU acceleration.",
            "The spectral radius parameter in ReservoirPy controls the memory capacity of the reservoir.",
            "LeakyReLU activation function often works well for reservoir nodes in ReservoirPy.",
            "ReservoirPy supports both offline and online learning algorithms.",
        ]
        
        print("📝 Adding custom knowledge to the system...")
        for i, knowledge in enumerate(knowledge_items, 1):
            node_id = await self.chat_engine.add_knowledge_from_message(
                content=knowledge,
                source="demo_expansion"
            )
            print(f"   {i}. Added: {knowledge[:60]}... (ID: {node_id})")
        
        # Test retrieval of added knowledge
        session_id = await self.chat_engine.create_session(title="Knowledge Test Session")
        
        test_questions = [
            "Tell me about JAX support in ReservoirPy",
            "What does spectral radius control?",
            "What activation functions work well with ReservoirPy?"
        ]
        
        print("\n🔍 Testing retrieval of added knowledge...")
        for question in test_questions:
            print(f"\n👤 Question: {question}")
            
            try:
                response = await self.chat_engine.chat(
                    message=question,
                    session_id=session_id,
                    use_cognitive_processing=True
                )
                
                print(f"🤖 Assistant: {response['message'][:150]}...")
                
                # Show if custom knowledge was used
                context = response['cognitive_context']
                if context.get('context_used'):
                    print(f"✅ Used cognitive context with {context['retrieved_nodes']} relevant concepts")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n✅ Knowledge expansion demonstration completed!")
    
    async def demo_streaming_chat(self):
        """Demonstrate streaming responses."""
        print("\n" + "="*60)
        print("🌊 STREAMING CHAT DEMONSTRATION")
        print("="*60)
        
        if not await self.initialize():
            return
        
        session_id = await self.chat_engine.create_session(title="Streaming Demo")
        question = "Explain how to build and train an Echo State Network step by step"
        
        print(f"👤 Question: {question}")
        print("🤖 Assistant (streaming): ", end="", flush=True)
        
        try:
            response_text = ""
            async for chunk in await self.chat_engine.chat(
                message=question,
                session_id=session_id,
                stream=True,
                use_cognitive_processing=True
            ):
                if chunk.get("type") == "token":
                    content = chunk.get("content", "")
                    print(content, end="", flush=True)
                    response_text += content
                elif chunk.get("type") == "complete":
                    context = chunk.get("cognitive_context", {})
                    print(f"\n\n🧠 Context: {context}")
            
            print("\n✅ Streaming demonstration completed!")
            
        except Exception as e:
            print(f"\n❌ Streaming error: {e}")
    
    def run_web_interface(self, host: str = "localhost", port: int = 8000):
        """Run the web interface."""
        print("\n" + "="*60)
        print("🌐 WEB INTERFACE DEMONSTRATION")
        print("="*60)
        
        try:
            # Setup LLM config
            llm_config = {}
            if self.api_key:
                llm_config["api_key"] = self.api_key
            elif self.llm_provider == "cohere":
                llm_config["api_key"] = os.getenv("COHERE_API_KEY")
            elif self.llm_provider == "openai":
                llm_config["api_key"] = os.getenv("OPENAI_API_KEY")
            
            # Create web interface
            web_interface = create_web_interface(
                llm_provider=self.llm_provider,
                llm_config=llm_config,
                host=host,
                port=port
            )
            
            print(f"🚀 Starting ReservoirPy Chat Interface at http://{host}:{port}")
            print("   - Main interface: http://{host}:{port}")
            print(f"   - API docs: http://{host}:{port}/docs")
            print("   - Press Ctrl+C to stop")
            
            # Run the web server
            web_interface.run()
            
        except ImportError as e:
            print(f"❌ Web interface not available: {e}")
            print("Install FastAPI and uvicorn: pip install fastapi uvicorn")
        except Exception as e:
            print(f"❌ Failed to start web interface: {e}")
    
    async def run_cli_demo(self):
        """Run interactive CLI demonstration."""
        print("\n" + "="*60)
        print("💬 INTERACTIVE CLI DEMONSTRATION")
        print("="*60)
        
        if not await self.initialize():
            return
        
        session_id = await self.chat_engine.create_session(title="CLI Demo Session")
        
        print("🤖 Welcome! Ask me anything about ReservoirPy or reservoir computing.")
        print("   Type 'quit', 'exit', or 'bye' to end the session.")
        print("   Type 'help' for available commands.")
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    print("""
Available commands:
- Ask any question about ReservoirPy or reservoir computing
- 'sessions' - List all sessions
- 'history' - Show conversation history
- 'add <knowledge>' - Add knowledge to the system
- 'quit', 'exit', 'bye' - End session
                    """)
                    continue
                
                if user_input.lower() == 'sessions':
                    sessions = await self.chat_engine.list_sessions()
                    print(f"📋 Active sessions: {len(sessions)}")
                    for session in sessions[:5]:  # Show last 5
                        print(f"   - {session['id'][:8]}: {session.get('title', 'Untitled')}")
                    continue
                
                if user_input.lower() == 'history':
                    history = await self.chat_engine.get_conversation_history(session_id, limit=10)
                    print(f"📜 Recent conversation history ({len(history)} messages):")
                    for msg in history[-5:]:  # Show last 5
                        role_emoji = "👤" if msg['role'] == 'user' else "🤖"
                        print(f"   {role_emoji} {msg['content'][:60]}...")
                    continue
                
                if user_input.lower().startswith('add '):
                    knowledge = user_input[4:].strip()
                    if knowledge:
                        node_id = await self.chat_engine.add_knowledge_from_message(
                            content=knowledge,
                            source="cli_input"
                        )
                        print(f"✅ Added knowledge (ID: {node_id})")
                    else:
                        print("❌ Please provide knowledge content after 'add'")
                    continue
                
                # Regular chat message
                print("🤖 Assistant: ", end="", flush=True)
                
                response = await self.chat_engine.chat(
                    message=user_input,
                    session_id=session_id,
                    use_cognitive_processing=True
                )
                
                print(response['message'])
                
                # Show context info
                context = response['cognitive_context']
                if context.get('context_used'):
                    print(f"   🧠 (Used {context['retrieved_nodes']} concepts, {context['retrieved_edges']} relations)")
                
            except KeyboardInterrupt:
                print("\n👋 Session interrupted. Goodbye!")
                break
            except EOFError:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


async def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="ReservoirPy Chat Engine Demo")
    parser.add_argument(
        "--mode", 
        choices=["cli", "web", "api", "demo"], 
        default="demo",
        help="Demo mode to run"
    )
    parser.add_argument(
        "--llm-provider", 
        choices=["cohere", "openai"], 
        default="cohere",
        help="LLM provider to use"
    )
    parser.add_argument("--api-key", help="API key for LLM provider")
    parser.add_argument("--host", default="localhost", help="Host for web interface")
    parser.add_argument("--port", type=int, default=8000, help="Port for web interface")
    
    args = parser.parse_args()
    
    # Create demo instance
    demo = ReservoirPyChatDemo(
        llm_provider=args.llm_provider,
        api_key=args.api_key
    )
    
    print("🧠 ReservoirPy Chat Engine Demonstration")
    print("="*50)
    print(f"Provider: {args.llm_provider}")
    print(f"Mode: {args.mode}")
    
    if args.mode == "demo":
        # Run all demonstrations
        await demo.demo_basic_chat()
        await demo.demo_knowledge_expansion()
        await demo.demo_streaming_chat()
        
    elif args.mode == "cli":
        # Interactive CLI
        await demo.run_cli_demo()
        
    elif args.mode == "web":
        # Web interface
        demo.run_web_interface(host=args.host, port=args.port)
        
    elif args.mode == "api":
        # Just show API info
        print(f"""
API Documentation:
- Main interface: http://{args.host}:{args.port}
- API docs: http://{args.host}:{args.port}/docs
- Health check: http://{args.host}:{args.port}/health

Available endpoints:
- POST /api/chat - Send chat messages
- POST /api/chat/stream - Stream chat responses  
- POST /api/sessions - Create chat sessions
- GET /api/sessions - List all sessions
- GET /api/sessions/{{id}} - Get specific session
- POST /api/knowledge - Add knowledge to system
        """)


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())