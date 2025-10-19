# Licence: MIT License
# Copyright: ReservoirCog Contributors (2024)

"""
Chat Inference Engine for ReservoirPy Cognitive Systems

This module implements a conversational AI interface that combines ReservoirPy's
cognitive architecture with LLM capabilities via GraphRAG, providing intelligent
chat functionality similar to https://chat.reservoirpy.inria.fr/
"""

import logging
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator, Union
from dataclasses import dataclass, asdict
import asyncio

from .atoms import AtomSpace, ConceptNode, PredicateNode
from .networks import CognitiveOrchestrator
from .llm_interface import BaseLLMInterface, create_llm_interface
from .graphrag import GraphRAGEngine, create_graphrag_engine

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Represents a single message in a conversation."""
    id: str
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ChatSession:
    """Represents a chat conversation session."""
    id: str
    messages: List[ChatMessage]
    created_at: datetime
    updated_at: datetime
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ReservoirPyChatEngine:
    """
    Main chat engine that integrates ReservoirPy cognitive architecture
    with LLM-based conversational AI using GraphRAG for knowledge retrieval.
    """
    
    def __init__(
        self,
        llm_provider: str = "cohere",
        llm_config: Dict[str, Any] = None,
        cognitive_config: Dict[str, Any] = None,
        system_message: str = None,
        max_context_length: int = 4000,
        max_response_tokens: int = 1000,
        temperature: float = 0.7
    ):
        """
        Initialize the ReservoirPy Chat Engine.
        
        Parameters
        ----------
        llm_provider : str, default="cohere"
            LLM provider ("cohere", "openai")
        llm_config : Dict[str, Any], optional
            LLM-specific configuration
        cognitive_config : Dict[str, Any], optional
            Cognitive orchestrator configuration
        system_message : str, optional
            System message for the chat assistant
        max_context_length : int, default=4000
            Maximum context length for conversations
        max_response_tokens : int, default=1000
            Maximum tokens for responses
        temperature : float, default=0.7
            Generation temperature
        """
        self.llm_provider = llm_provider
        self.llm_config = llm_config or {}
        self.cognitive_config = cognitive_config or {}
        self.max_context_length = max_context_length
        self.max_response_tokens = max_response_tokens
        self.temperature = temperature
        
        # Default system message
        self.system_message = system_message or self._get_default_system_message()
        
        # Initialize components
        self.llm_interface = None
        self.cognitive_orchestrator = None
        self.atomspace = None
        self.graphrag_engine = None
        
        # Session management
        self.sessions: Dict[str, ChatSession] = {}
        
        # Initialize async components
        self._initialization_task = None
    
    async def initialize(self):
        """Initialize async components."""
        if self._initialization_task is None:
            self._initialization_task = asyncio.create_task(self._async_initialize())
        await self._initialization_task
    
    async def _async_initialize(self):
        """Async initialization of components."""
        logger.info("Initializing ReservoirPy Chat Engine...")
        
        try:
            # Initialize LLM interface
            self.llm_interface = create_llm_interface(
                self.llm_provider, 
                **self.llm_config
            )
            
            # Initialize cognitive orchestrator
            orchestrator_config = {
                "network_size": 5,
                **self.cognitive_config
            }
            self.cognitive_orchestrator = CognitiveOrchestrator(**orchestrator_config)
            self.atomspace = self.cognitive_orchestrator.atomspace
            
            # Initialize with some basic ReservoirPy knowledge
            await self._initialize_reservoirpy_knowledge()
            
            # Initialize GraphRAG engine
            self.graphrag_engine = create_graphrag_engine(
                self.atomspace,
                self.llm_interface
            )
            
            logger.info("ReservoirPy Chat Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize chat engine: {e}")
            raise
    
    async def _initialize_reservoirpy_knowledge(self):
        """Initialize the AtomSpace with basic ReservoirPy knowledge."""
        # Add ReservoirPy concepts
        reservoir_computing = self.cognitive_orchestrator.add_concept(
            "reservoir_computing", attention=0.9, truth_value=1.0
        )
        echo_state_networks = self.cognitive_orchestrator.add_concept(
            "echo_state_networks", attention=0.8, truth_value=1.0
        )
        reservoirpy = self.cognitive_orchestrator.add_concept(
            "reservoirpy", attention=0.9, truth_value=1.0
        )
        machine_learning = self.cognitive_orchestrator.add_concept(
            "machine_learning", attention=0.7, truth_value=1.0
        )
        
        # Add relationships
        implements = self.cognitive_orchestrator.add_predicate(
            "implements", arity=2, confidence=0.9
        )
        is_part_of = self.cognitive_orchestrator.add_predicate(
            "is_part_of", arity=2, confidence=0.9
        )
        
        # Create links
        self.cognitive_orchestrator.create_link(
            reservoirpy, reservoir_computing, "implements"
        )
        self.cognitive_orchestrator.create_link(
            echo_state_networks, reservoir_computing, "is_part_of"
        )
        self.cognitive_orchestrator.create_link(
            reservoir_computing, machine_learning, "is_part_of"
        )
        
        # Add some documentation knowledge
        await self._add_reservoirpy_documentation()
    
    async def _add_reservoirpy_documentation(self):
        """Add ReservoirPy documentation to the knowledge base."""
        docs = [
            {
                "content": "ReservoirPy is a Python library for reservoir computing and echo state networks. It provides easy creation of complex architectures with multiple reservoirs.",
                "type": "documentation"
            },
            {
                "content": "Echo State Networks (ESN) are a type of recurrent neural network where the reservoir weights are fixed and only the readout is trained.",
                "type": "concept_definition"
            },
            {
                "content": "Reservoir computing uses the dynamics of a randomly connected recurrent neural network (the reservoir) to process temporal information.",
                "type": "concept_definition"
            },
            {
                "content": "ReservoirPy supports hyperparameter optimization, parallel implementation, and interfaces with scikit-learn.",
                "type": "feature_description"
            }
        ]
        
        for doc in docs:
            await self.graphrag_engine.add_knowledge(
                content=doc["content"],
                node_type=doc["type"],
                metadata={"source": "reservoirpy_docs"}
            )
    
    def _get_default_system_message(self) -> str:
        """Get the default system message for the chat assistant."""
        return """You are ReservoirPy Assistant, an AI helper specialized in reservoir computing, echo state networks, and the ReservoirPy library. 

You have access to a knowledge base about ReservoirPy and reservoir computing concepts through a cognitive architecture that combines symbolic reasoning with neural reservoir networks.

Your capabilities:
- Answer questions about reservoir computing theory and applications
- Help with ReservoirPy library usage, code examples, and troubleshooting
- Explain echo state networks, reservoir dynamics, and related concepts
- Provide guidance on hyperparameter optimization and model design
- Assist with scientific research in reservoir computing

Always provide accurate, helpful information and cite relevant concepts from your knowledge base when appropriate. If you're unsure about something, say so clearly."""
    
    async def create_session(
        self, 
        title: str = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Create a new chat session.
        
        Parameters
        ----------
        title : str, optional
            Title for the session
        metadata : Dict[str, Any], optional
            Additional session metadata
            
        Returns
        -------
        str
            Session ID
        """
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        session = ChatSession(
            id=session_id,
            messages=[],
            created_at=now,
            updated_at=now,
            title=title,
            metadata=metadata or {}
        )
        
        self.sessions[session_id] = session
        logger.info(f"Created new chat session: {session_id}")
        
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a chat session by ID."""
        return self.sessions.get(session_id)
    
    async def list_sessions(self) -> List[Dict[str, Any]]:
        """List all chat sessions."""
        return [
            {
                "id": session.id,
                "title": session.title,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "message_count": len(session.messages)
            }
            for session in self.sessions.values()
        ]
    
    async def chat(
        self, 
        message: str,
        session_id: str = None,
        stream: bool = False,
        use_cognitive_processing: bool = True
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, str], None]]:
        """
        Process a chat message and generate a response.
        
        Parameters
        ----------
        message : str
            User message
        session_id : str, optional
            Session ID (creates new session if None)
        stream : bool, default=False
            Whether to stream the response
        use_cognitive_processing : bool, default=True
            Whether to use cognitive processing
            
        Returns
        -------
        Union[Dict[str, Any], AsyncGenerator[Dict[str, str], None]]
            Response data or async generator for streaming
        """
        await self.initialize()
        
        # Create session if needed
        if session_id is None:
            session_id = await self.create_session()
        
        session = await self.get_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")
        
        # Add user message
        user_msg = ChatMessage(
            id=str(uuid.uuid4()),
            role="user",
            content=message,
            timestamp=datetime.now()
        )
        session.messages.append(user_msg)
        
        try:
            if stream:
                return self._stream_response(session, message, use_cognitive_processing)
            else:
                return await self._generate_response(session, message, use_cognitive_processing)
                
        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            error_response = {
                "session_id": session_id,
                "message": "I apologize, but I encountered an error processing your request. Please try again.",
                "error": str(e)
            }
            
            # Add error message to session
            error_msg = ChatMessage(
                id=str(uuid.uuid4()),
                role="assistant",
                content=error_response["message"],
                timestamp=datetime.now(),
                metadata={"error": True}
            )
            session.messages.append(error_msg)
            session.updated_at = datetime.now()
            
            return error_response
    
    async def _generate_response(
        self, 
        session: ChatSession, 
        message: str,
        use_cognitive_processing: bool
    ) -> Dict[str, Any]:
        """Generate a complete response."""
        # Use GraphRAG to get context-enhanced response
        if self.graphrag_engine and use_cognitive_processing:
            response_data = await self.graphrag_engine.generate_with_context(
                query=message,
                system_message=self.system_message,
                max_tokens=self.max_response_tokens,
                temperature=self.temperature
            )
            
            response_text = response_data["response"]
            cognitive_context = {
                "context_used": response_data["context_used"],
                "retrieved_nodes": response_data["retrieved_nodes"],
                "retrieved_edges": response_data["retrieved_edges"]
            }
        else:
            # Fallback to direct LLM generation
            response_text = await self.llm_interface.generate(
                prompt=message,
                system_message=self.system_message,
                max_tokens=self.max_response_tokens,
                temperature=self.temperature
            )
            cognitive_context = {"context_used": False}
        
        # Add assistant message to session
        assistant_msg = ChatMessage(
            id=str(uuid.uuid4()),
            role="assistant",
            content=response_text,
            timestamp=datetime.now(),
            metadata={"cognitive_context": cognitive_context}
        )
        session.messages.append(assistant_msg)
        session.updated_at = datetime.now()
        
        return {
            "session_id": session.id,
            "message": response_text,
            "cognitive_context": cognitive_context,
            "message_id": assistant_msg.id,
            "timestamp": assistant_msg.timestamp.isoformat()
        }
    
    async def _stream_response(
        self, 
        session: ChatSession, 
        message: str,
        use_cognitive_processing: bool
    ) -> AsyncGenerator[Dict[str, str], None]:
        """Stream response generation."""
        # For streaming, we'll use direct LLM interface for now
        # GraphRAG streaming would require more complex implementation
        
        response_parts = []
        message_id = str(uuid.uuid4())
        
        yield {
            "type": "start",
            "session_id": session.id,
            "message_id": message_id
        }
        
        try:
            # Get context if using cognitive processing
            context_text = ""
            cognitive_context = {"context_used": False}
            
            if self.graphrag_engine and use_cognitive_processing:
                context_data = await self.graphrag_engine.retrieve_relevant_context(message)
                context_text = context_data["context_text"]
                cognitive_context = {
                    "context_used": bool(context_text),
                    "retrieved_nodes": len(context_data["nodes"]),
                    "retrieved_edges": len(context_data["edges"])
                }
            
            # Build enhanced prompt
            if context_text:
                enhanced_message = f"""Context Information:
{context_text}

Based on the above context, please answer the following question:
{message}"""
            else:
                enhanced_message = message
            
            # Stream response
            async for token in self.llm_interface.stream_generate(
                prompt=enhanced_message,
                system_message=self.system_message,
                max_tokens=self.max_response_tokens,
                temperature=self.temperature
            ):
                response_parts.append(token)
                yield {
                    "type": "token",
                    "content": token,
                    "session_id": session.id,
                    "message_id": message_id
                }
            
            # Complete response
            full_response = "".join(response_parts)
            
            # Add to session
            assistant_msg = ChatMessage(
                id=message_id,
                role="assistant",
                content=full_response,
                timestamp=datetime.now(),
                metadata={"cognitive_context": cognitive_context}
            )
            session.messages.append(assistant_msg)
            session.updated_at = datetime.now()
            
            yield {
                "type": "complete",
                "session_id": session.id,
                "message_id": message_id,
                "cognitive_context": cognitive_context,
                "timestamp": assistant_msg.timestamp.isoformat()
            }
            
        except Exception as e:
            yield {
                "type": "error",
                "error": str(e),
                "session_id": session.id,
                "message_id": message_id
            }
    
    async def get_conversation_history(
        self, 
        session_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        session = await self.get_session(session_id)
        if session is None:
            return []
        
        messages = session.messages[-limit:]  # Get last N messages
        
        return [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata
            }
            for msg in messages
        ]
    
    async def add_knowledge_from_message(
        self, 
        content: str,
        source: str = "user_input"
    ) -> str:
        """Add knowledge from user message to the knowledge base."""
        await self.initialize()
        
        if self.graphrag_engine:
            node_id = await self.graphrag_engine.add_knowledge(
                content=content,
                node_type="user_knowledge",
                metadata={"source": source, "added_at": datetime.now().isoformat()}
            )
            logger.info(f"Added user knowledge: {node_id}")
            return node_id
        else:
            logger.warning("GraphRAG engine not available for adding knowledge")
            return None
    
    async def export_session(self, session_id: str) -> Dict[str, Any]:
        """Export a session to JSON-serializable format."""
        session = await self.get_session(session_id)
        if session is None:
            return None
        
        return {
            "id": session.id,
            "title": session.title,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": msg.metadata
                }
                for msg in session.messages
            ],
            "metadata": session.metadata
        }


# Convenience function for creating chat engines
def create_chat_engine(
    llm_provider: str = "cohere",
    **kwargs
) -> ReservoirPyChatEngine:
    """
    Create a ReservoirPy chat engine with the specified configuration.
    
    Parameters
    ----------
    llm_provider : str, default="cohere"
        LLM provider ("cohere", "openai")
    **kwargs
        Additional configuration options
        
    Returns
    -------
    ReservoirPyChatEngine
        Configured chat engine
        
    Examples
    --------
    >>> # Create Cohere-based chat engine
    >>> engine = create_chat_engine("cohere", llm_config={"api_key": "your-key"})
    >>> 
    >>> # Create OpenAI-based chat engine
    >>> engine = create_chat_engine("openai", llm_config={"api_key": "your-key"})
    """
    return ReservoirPyChatEngine(llm_provider=llm_provider, **kwargs)