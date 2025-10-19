# Licence: MIT License
# Copyright: ReservoirCog Contributors (2024)

"""
Tests for the ReservoirPy chat engine.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from reservoirpy.cognitive.chat_engine import (
    ChatMessage,
    ChatSession,
    ReservoirPyChatEngine,
    create_chat_engine
)


class MockLLMInterface:
    """Mock LLM interface for testing."""
    
    def __init__(self):
        self.generate_calls = []
        self.stream_calls = []
        self.embed_calls = []
    
    async def generate(self, prompt: str, system_message: str = None, **kwargs) -> str:
        self.generate_calls.append({
            "prompt": prompt,
            "system_message": system_message,
            **kwargs
        })
        return f"Mock response to: {prompt[:30]}..."
    
    async def stream_generate(self, prompt: str, system_message: str = None, **kwargs):
        self.stream_calls.append({
            "prompt": prompt,
            "system_message": system_message,
            **kwargs
        })
        for token in ["Mock", " streaming", " response"]:
            yield token
    
    async def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        self.embed_calls.append(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]


class MockGraphRAGEngine:
    """Mock GraphRAG engine for testing."""
    
    def __init__(self):
        self.add_knowledge_calls = []
        self.generate_calls = []
        self.retrieve_calls = []
    
    async def add_knowledge(self, content: str, node_type: str = "document", metadata=None):
        self.add_knowledge_calls.append({
            "content": content,
            "node_type": node_type,
            "metadata": metadata
        })
        return f"node_{len(self.add_knowledge_calls)}"
    
    async def generate_with_context(self, query: str, **kwargs):
        self.generate_calls.append({"query": query, **kwargs})
        return {
            "response": f"GraphRAG response to: {query[:30]}...",
            "context_used": True,
            "retrieved_nodes": 3,
            "retrieved_edges": 2
        }
    
    async def retrieve_relevant_context(self, query: str, **kwargs):
        self.retrieve_calls.append({"query": query, **kwargs})
        return {
            "nodes": [{"id": "node1", "content": "Mock node 1"}],
            "edges": [{"source": "node1", "target": "node2", "relation": "related"}],
            "context_text": "Mock context text"
        }


class MockCognitiveOrchestrator:
    """Mock cognitive orchestrator for testing."""
    
    def __init__(self):
        self.atomspace = Mock()
        self.add_concept_calls = []
        self.add_predicate_calls = []
        self.create_link_calls = []
    
    def add_concept(self, name: str, attention=0.5, truth_value=1.0):
        self.add_concept_calls.append({
            "name": name,
            "attention": attention,
            "truth_value": truth_value
        })
        concept = Mock()
        concept.name = name
        return concept
    
    def add_predicate(self, name: str, arity=2, confidence=0.9):
        self.add_predicate_calls.append({
            "name": name,
            "arity": arity,
            "confidence": confidence
        })
        predicate = Mock()
        predicate.name = name
        predicate.arity = arity
        return predicate
    
    def create_link(self, source, target, relation):
        self.create_link_calls.append({
            "source": source,
            "target": target,
            "relation": relation
        })


class TestChatMessage:
    """Test ChatMessage data class."""
    
    def test_creation(self):
        """Test creating a chat message."""
        now = datetime.now()
        message = ChatMessage(
            id="msg_1",
            role="user",
            content="Hello world",
            timestamp=now,
            metadata={"source": "test"}
        )
        
        assert message.id == "msg_1"
        assert message.role == "user"
        assert message.content == "Hello world"
        assert message.timestamp == now
        assert message.metadata == {"source": "test"}


class TestChatSession:
    """Test ChatSession data class."""
    
    def test_creation(self):
        """Test creating a chat session."""
        now = datetime.now()
        message = ChatMessage("msg_1", "user", "Hello", now)
        
        session = ChatSession(
            id="session_1",
            messages=[message],
            created_at=now,
            updated_at=now,
            title="Test Session",
            metadata={"test": True}
        )
        
        assert session.id == "session_1"
        assert len(session.messages) == 1
        assert session.messages[0] == message
        assert session.title == "Test Session"
        assert session.metadata == {"test": True}


class TestReservoirPyChatEngine:
    """Test ReservoirPy chat engine."""
    
    def test_initialization(self):
        """Test chat engine initialization."""
        engine = ReservoirPyChatEngine(
            llm_provider="cohere",
            llm_config={"api_key": "test"},
            temperature=0.5
        )
        
        assert engine.llm_provider == "cohere"
        assert engine.llm_config == {"api_key": "test"}
        assert engine.temperature == 0.5
        assert engine.max_response_tokens == 1000
        assert "ReservoirPy Assistant" in engine.system_message
    
    @pytest.mark.asyncio
    async def test_create_session(self):
        """Test creating a chat session."""
        engine = ReservoirPyChatEngine()
        
        session_id = await engine.create_session(
            title="Test Session",
            metadata={"test": True}
        )
        
        assert session_id in engine.sessions
        session = engine.sessions[session_id]
        assert session.title == "Test Session"
        assert session.metadata == {"test": True}
        assert len(session.messages) == 0
    
    @pytest.mark.asyncio
    async def test_get_session(self):
        """Test getting a session by ID."""
        engine = ReservoirPyChatEngine()
        session_id = await engine.create_session()
        
        session = await engine.get_session(session_id)
        assert session is not None
        assert session.id == session_id
        
        # Test non-existent session
        non_existent = await engine.get_session("non_existent")
        assert non_existent is None
    
    @pytest.mark.asyncio
    async def test_list_sessions(self):
        """Test listing all sessions."""
        engine = ReservoirPyChatEngine()
        
        # Create multiple sessions
        session1 = await engine.create_session(title="Session 1")
        session2 = await engine.create_session(title="Session 2")
        
        sessions = await engine.list_sessions()
        assert len(sessions) == 2
        
        # Check session data
        session_ids = [s["id"] for s in sessions]
        assert session1 in session_ids
        assert session2 in session_ids
    
    @patch('reservoirpy.cognitive.chat_engine.create_llm_interface')
    @patch('reservoirpy.cognitive.chat_engine.CognitiveOrchestrator')
    @patch('reservoirpy.cognitive.chat_engine.create_graphrag_engine')
    @pytest.mark.asyncio
    async def test_initialize_components(self, mock_graphrag, mock_orchestrator, mock_llm):
        """Test initializing chat engine components."""
        # Setup mocks
        mock_llm_interface = MockLLMInterface()
        mock_llm.return_value = mock_llm_interface
        
        mock_cognitive = MockCognitiveOrchestrator()
        mock_orchestrator.return_value = mock_cognitive
        
        mock_graphrag_engine = MockGraphRAGEngine()
        mock_graphrag.return_value = mock_graphrag_engine
        
        # Initialize engine
        engine = ReservoirPyChatEngine(llm_provider="cohere")
        await engine.initialize()
        
        # Check components were created
        assert engine.llm_interface == mock_llm_interface
        assert engine.cognitive_orchestrator == mock_cognitive
        assert engine.graphrag_engine == mock_graphrag_engine
        
        # Check ReservoirPy knowledge was added
        assert len(mock_cognitive.add_concept_calls) >= 3  # reservoir_computing, etc.
        assert len(mock_graphrag_engine.add_knowledge_calls) >= 3  # documentation
    
    @patch('reservoirpy.cognitive.chat_engine.create_llm_interface')
    @patch('reservoirpy.cognitive.chat_engine.CognitiveOrchestrator') 
    @patch('reservoirpy.cognitive.chat_engine.create_graphrag_engine')
    @pytest.mark.asyncio
    async def test_chat_with_cognitive_processing(self, mock_graphrag, mock_orchestrator, mock_llm):
        """Test chat with cognitive processing enabled."""
        # Setup mocks
        mock_llm_interface = MockLLMInterface()
        mock_llm.return_value = mock_llm_interface
        
        mock_cognitive = MockCognitiveOrchestrator()
        mock_orchestrator.return_value = mock_cognitive
        
        mock_graphrag_engine = MockGraphRAGEngine()
        mock_graphrag.return_value = mock_graphrag_engine
        
        # Test chat
        engine = ReservoirPyChatEngine()
        await engine.initialize()
        
        response = await engine.chat(
            message="What is ReservoirPy?",
            use_cognitive_processing=True
        )
        
        # Check response structure
        assert "session_id" in response
        assert "message" in response
        assert "cognitive_context" in response
        assert "message_id" in response
        assert "timestamp" in response
        
        # Check GraphRAG was used
        assert len(mock_graphrag_engine.generate_calls) == 1
        assert mock_graphrag_engine.generate_calls[0]["query"] == "What is ReservoirPy?"
        
        # Check session was created and message added
        session_id = response["session_id"]
        session = await engine.get_session(session_id)
        assert len(session.messages) == 2  # user + assistant
        assert session.messages[0].role == "user"
        assert session.messages[1].role == "assistant"
    
    @patch('reservoirpy.cognitive.chat_engine.create_llm_interface')
    @patch('reservoirpy.cognitive.chat_engine.CognitiveOrchestrator')
    @pytest.mark.asyncio
    async def test_chat_without_cognitive_processing(self, mock_orchestrator, mock_llm):
        """Test chat without cognitive processing."""
        # Setup mocks
        mock_llm_interface = MockLLMInterface()
        mock_llm.return_value = mock_llm_interface
        
        mock_cognitive = MockCognitiveOrchestrator()
        mock_orchestrator.return_value = mock_cognitive
        
        # Test chat (GraphRAG engine will be None)
        engine = ReservoirPyChatEngine()
        engine.llm_interface = mock_llm_interface
        engine.cognitive_orchestrator = mock_cognitive
        engine.atomspace = mock_cognitive.atomspace
        engine.graphrag_engine = None  # Disable GraphRAG
        
        response = await engine.chat(
            message="What is ReservoirPy?",
            use_cognitive_processing=False
        )
        
        # Check direct LLM was used
        assert len(mock_llm_interface.generate_calls) == 1
        assert mock_llm_interface.generate_calls[0]["prompt"] == "What is ReservoirPy?"
        
        # Check cognitive context shows no context used
        assert response["cognitive_context"]["context_used"] is False
    
    @patch('reservoirpy.cognitive.chat_engine.create_llm_interface')
    @patch('reservoirpy.cognitive.chat_engine.CognitiveOrchestrator')
    @patch('reservoirpy.cognitive.chat_engine.create_graphrag_engine')
    @pytest.mark.asyncio
    async def test_streaming_chat(self, mock_graphrag, mock_orchestrator, mock_llm):
        """Test streaming chat responses."""
        # Setup mocks
        mock_llm_interface = MockLLMInterface()
        mock_llm.return_value = mock_llm_interface
        
        mock_cognitive = MockCognitiveOrchestrator()
        mock_orchestrator.return_value = mock_cognitive
        
        mock_graphrag_engine = MockGraphRAGEngine()
        mock_graphrag.return_value = mock_graphrag_engine
        
        # Test streaming chat
        engine = ReservoirPyChatEngine()
        await engine.initialize()
        
        session_id = await engine.create_session()
        
        chunks = []
        async for chunk in await engine.chat(
            message="What is ReservoirPy?",
            session_id=session_id,
            stream=True,
            use_cognitive_processing=True
        ):
            chunks.append(chunk)
        
        # Check streaming chunks
        assert len(chunks) >= 4  # start, tokens, complete
        assert chunks[0]["type"] == "start"
        assert any(chunk["type"] == "token" for chunk in chunks)
        assert chunks[-1]["type"] == "complete"
        
        # Check context retrieval was called
        assert len(mock_graphrag_engine.retrieve_calls) == 1
        assert len(mock_llm_interface.stream_calls) == 1
    
    @patch('reservoirpy.cognitive.chat_engine.create_llm_interface')
    @patch('reservoirpy.cognitive.chat_engine.CognitiveOrchestrator')
    @patch('reservoirpy.cognitive.chat_engine.create_graphrag_engine')
    @pytest.mark.asyncio
    async def test_add_knowledge(self, mock_graphrag, mock_orchestrator, mock_llm):
        """Test adding knowledge to the system."""
        # Setup mocks
        mock_llm_interface = MockLLMInterface()
        mock_llm.return_value = mock_llm_interface
        
        mock_cognitive = MockCognitiveOrchestrator()
        mock_orchestrator.return_value = mock_cognitive
        
        mock_graphrag_engine = MockGraphRAGEngine()
        mock_graphrag.return_value = mock_graphrag_engine
        
        # Test adding knowledge
        engine = ReservoirPyChatEngine()
        await engine.initialize()
        
        node_id = await engine.add_knowledge_from_message(
            content="ReservoirPy supports JAX backend for GPU acceleration",
            source="test_source"
        )
        
        # Check knowledge was added to GraphRAG
        assert len(mock_graphrag_engine.add_knowledge_calls) > 4  # Initial docs + new knowledge
        last_call = mock_graphrag_engine.add_knowledge_calls[-1]
        assert "JAX backend" in last_call["content"]
        assert last_call["metadata"]["source"] == "test_source"
        assert node_id == f"node_{len(mock_graphrag_engine.add_knowledge_calls)}"
    
    @pytest.mark.asyncio
    async def test_conversation_history(self):
        """Test getting conversation history."""
        engine = ReservoirPyChatEngine()
        session_id = await engine.create_session()
        
        # Add some messages manually
        session = await engine.get_session(session_id)
        now = datetime.now()
        
        for i in range(5):
            msg = ChatMessage(
                id=f"msg_{i}",
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}",
                timestamp=now
            )
            session.messages.append(msg)
        
        # Get history
        history = await engine.get_conversation_history(session_id, limit=3)
        
        assert len(history) == 3
        assert history[0]["content"] == "Message 2"  # Last 3 messages
        assert history[1]["content"] == "Message 3"
        assert history[2]["content"] == "Message 4"
    
    @pytest.mark.asyncio
    async def test_export_session(self):
        """Test exporting session data."""
        engine = ReservoirPyChatEngine()
        session_id = await engine.create_session(title="Export Test")
        
        # Add a message
        session = await engine.get_session(session_id)
        now = datetime.now()
        msg = ChatMessage("msg_1", "user", "Hello", now)
        session.messages.append(msg)
        
        # Export session
        exported = await engine.export_session(session_id)
        
        assert exported["id"] == session_id
        assert exported["title"] == "Export Test"
        assert len(exported["messages"]) == 1
        assert exported["messages"][0]["content"] == "Hello"


class TestCreateChatEngine:
    """Test convenience function for creating chat engines."""
    
    def test_create_default_engine(self):
        """Test creating default chat engine."""
        engine = create_chat_engine()
        
        assert isinstance(engine, ReservoirPyChatEngine)
        assert engine.llm_provider == "cohere"
    
    def test_create_openai_engine(self):
        """Test creating OpenAI chat engine."""
        engine = create_chat_engine(
            llm_provider="openai",
            llm_config={"api_key": "test"},
            temperature=0.5
        )
        
        assert isinstance(engine, ReservoirPyChatEngine)
        assert engine.llm_provider == "openai"
        assert engine.llm_config == {"api_key": "test"}
        assert engine.temperature == 0.5