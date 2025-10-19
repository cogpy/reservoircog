# Licence: MIT License
# Copyright: ReservoirCog Contributors (2024)

"""
Web Interface for ReservoirPy Chat Engine

This module provides a FastAPI-based web interface that replicates the
functionality of https://chat.reservoirpy.inria.fr/ using the ReservoirPy
cognitive architecture with LLM and GraphRAG integration.
"""

import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

try:
    from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from .chat_engine import ReservoirPyChatEngine, create_chat_engine

logger = logging.getLogger(__name__)


# Pydantic models for API
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID")
    stream: bool = Field(False, description="Stream response")
    use_cognitive_processing: bool = Field(True, description="Use cognitive processing")


class ChatResponse(BaseModel):
    session_id: str
    message: str
    message_id: str
    timestamp: str
    cognitive_context: Dict[str, Any]


class SessionRequest(BaseModel):
    title: Optional[str] = Field(None, description="Session title")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Session metadata")


class KnowledgeRequest(BaseModel):
    content: str = Field(..., description="Knowledge content")
    source: str = Field("api", description="Knowledge source")


class ReservoirPyWebInterface:
    """Web interface for the ReservoirPy Chat Engine."""
    
    def __init__(
        self,
        chat_engine: ReservoirPyChatEngine = None,
        llm_provider: str = "cohere",
        llm_config: Dict[str, Any] = None,
        host: str = "0.0.0.0",
        port: int = 8000,
        title: str = "ReservoirPy Chat Interface",
        description: str = None,
        version: str = "1.0.0"
    ):
        """
        Initialize the web interface.
        
        Parameters
        ----------
        chat_engine : ReservoirPyChatEngine, optional
            Pre-configured chat engine
        llm_provider : str, default="cohere"
            LLM provider if creating new chat engine
        llm_config : Dict[str, Any], optional
            LLM configuration if creating new chat engine
        host : str, default="0.0.0.0"
            Host to bind the server
        port : int, default=8000
            Port to bind the server
        title : str, default="ReservoirPy Chat Interface"
            API title
        description : str, optional
            API description
        version : str, default="1.0.0"
            API version
        """
        if not HAS_FASTAPI:
            raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")
        
        self.host = host
        self.port = port
        
        # Initialize chat engine
        if chat_engine is None:
            self.chat_engine = create_chat_engine(
                llm_provider=llm_provider,
                llm_config=llm_config or {}
            )
        else:
            self.chat_engine = chat_engine
        
        # Create FastAPI app
        self.app = FastAPI(
            title=title,
            description=description or self._get_default_description(),
            version=version,
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware - configure origins appropriately for production
        cors_origins = kwargs.get('cors_origins', ["http://localhost:3000", "http://localhost:8000"])
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,  # Restricted origins for security
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["Content-Type", "Authorization"],
        )
        
        # Setup routes
        self._setup_routes()
    
    def _get_default_description(self) -> str:
        """Get the default API description."""
        return """
        ReservoirPy Chat Interface - An intelligent conversational AI system 
        built on ReservoirPy's cognitive architecture with GraphRAG and LLM integration.
        
        Features:
        - Reservoir computing-based cognitive processing
        - Knowledge graph enhanced retrieval (GraphRAG)
        - Multiple LLM backends (Cohere, OpenAI)
        - Conversation management and context handling
        - Streaming responses
        - Knowledge base expansion
        """
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def home():
            """Serve the main chat interface."""
            return self._get_chat_html()
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "service": "ReservoirPy Chat Engine"
            }
        
        @self.app.post("/api/chat", response_model=ChatResponse)
        async def chat_endpoint(request: ChatRequest):
            """Main chat endpoint."""
            try:
                response = await self.chat_engine.chat(
                    message=request.message,
                    session_id=request.session_id,
                    stream=False,  # Non-streaming for this endpoint
                    use_cognitive_processing=request.use_cognitive_processing
                )
                
                return ChatResponse(
                    session_id=response["session_id"],
                    message=response["message"],
                    message_id=response["message_id"],
                    timestamp=response["timestamp"],
                    cognitive_context=response["cognitive_context"]
                )
                
            except Exception as e:
                logger.error(f"Chat endpoint error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/chat/stream")
        async def chat_stream_endpoint(request: ChatRequest):
            """Streaming chat endpoint."""
            try:
                if not request.stream:
                    request.stream = True
                
                async def generate():
                    async for chunk in await self.chat_engine.chat(
                        message=request.message,
                        session_id=request.session_id,
                        stream=True,
                        use_cognitive_processing=request.use_cognitive_processing
                    ):
                        yield f"data: {json.dumps(chunk)}\n\n"
                
                return StreamingResponse(
                    generate(),
                    media_type="text/plain",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    }
                )
                
            except Exception as e:
                logger.error(f"Stream endpoint error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/sessions")
        async def create_session(request: SessionRequest):
            """Create a new chat session."""
            try:
                session_id = await self.chat_engine.create_session(
                    title=request.title,
                    metadata=request.metadata
                )
                return {"session_id": session_id}
                
            except Exception as e:
                logger.error(f"Create session error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/sessions")
        async def list_sessions():
            """List all chat sessions."""
            try:
                sessions = await self.chat_engine.list_sessions()
                return {"sessions": sessions}
                
            except Exception as e:
                logger.error(f"List sessions error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/sessions/{session_id}")
        async def get_session(session_id: str):
            """Get a specific session."""
            try:
                session = await self.chat_engine.get_session(session_id)
                if session is None:
                    raise HTTPException(status_code=404, detail="Session not found")
                
                return await self.chat_engine.export_session(session_id)
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Get session error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/sessions/{session_id}/history")
        async def get_conversation_history(
            session_id: str,
            limit: int = Query(50, description="Maximum number of messages")
        ):
            """Get conversation history for a session."""
            try:
                history = await self.chat_engine.get_conversation_history(
                    session_id, limit
                )
                return {"messages": history}
                
            except Exception as e:
                logger.error(f"Get history error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/knowledge")
        async def add_knowledge(request: KnowledgeRequest):
            """Add knowledge to the system."""
            try:
                node_id = await self.chat_engine.add_knowledge_from_message(
                    content=request.content,
                    source=request.source
                )
                return {"node_id": node_id, "status": "added"}
                
            except Exception as e:
                logger.error(f"Add knowledge error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/info")
        async def system_info():
            """Get system information."""
            return {
                "name": "ReservoirPy Chat Engine",
                "version": "1.0.0",
                "llm_provider": self.chat_engine.llm_provider,
                "features": {
                    "cognitive_processing": True,
                    "graphrag": True,
                    "streaming": True,
                    "knowledge_expansion": True
                },
                "session_count": len(self.chat_engine.sessions)
            }
    
    def _get_chat_html(self) -> str:
        """Generate the main chat interface HTML."""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ReservoirPy Chat</title>
            <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: #f5f5f5;
                    height: 100vh;
                    display: flex;
                    flex-direction: column;
                }
                
                .header {
                    background: #2c3e50;
                    color: white;
                    padding: 1rem 2rem;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                
                .header h1 {
                    font-size: 1.5rem;
                    font-weight: 600;
                }
                
                .header p {
                    margin-top: 0.5rem;
                    opacity: 0.8;
                    font-size: 0.9rem;
                }
                
                .chat-container {
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    max-width: 800px;
                    margin: 0 auto;
                    width: 100%;
                    background: white;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }
                
                .messages {
                    flex: 1;
                    padding: 1rem;
                    overflow-y: auto;
                    display: flex;
                    flex-direction: column;
                    gap: 1rem;
                }
                
                .message {
                    max-width: 70%;
                    padding: 0.75rem 1rem;
                    border-radius: 1rem;
                    line-height: 1.4;
                }
                
                .message.user {
                    background: #007bff;
                    color: white;
                    align-self: flex-end;
                    border-bottom-right-radius: 0.25rem;
                }
                
                .message.assistant {
                    background: #e9ecef;
                    color: #333;
                    align-self: flex-start;
                    border-bottom-left-radius: 0.25rem;
                }
                
                .message.assistant.cognitive {
                    border-left: 3px solid #28a745;
                }
                
                .input-area {
                    padding: 1rem;
                    border-top: 1px solid #dee2e6;
                    display: flex;
                    gap: 0.5rem;
                }
                
                .message-input {
                    flex: 1;
                    padding: 0.75rem;
                    border: 1px solid #ced4da;
                    border-radius: 2rem;
                    outline: none;
                    font-size: 0.9rem;
                }
                
                .message-input:focus {
                    border-color: #007bff;
                    box-shadow: 0 0 0 0.2rem rgba(0,123,255,.25);
                }
                
                .send-btn {
                    padding: 0.75rem 1.5rem;
                    background: #007bff;
                    color: white;
                    border: none;
                    border-radius: 2rem;
                    cursor: pointer;
                    font-weight: 500;
                    transition: background-color 0.2s;
                }
                
                .send-btn:hover {
                    background: #0056b3;
                }
                
                .send-btn:disabled {
                    background: #6c757d;
                    cursor: not-allowed;
                }
                
                .typing {
                    display: none;
                    align-self: flex-start;
                    background: #e9ecef;
                    padding: 0.75rem 1rem;
                    border-radius: 1rem;
                    border-bottom-left-radius: 0.25rem;
                }
                
                .typing-dots {
                    display: flex;
                    gap: 0.25rem;
                }
                
                .typing-dots span {
                    width: 6px;
                    height: 6px;
                    background: #6c757d;
                    border-radius: 50%;
                    animation: typing 1.4s infinite ease-in-out;
                }
                
                .typing-dots span:nth-child(2) {
                    animation-delay: 0.2s;
                }
                
                .typing-dots span:nth-child(3) {
                    animation-delay: 0.4s;
                }
                
                @keyframes typing {
                    0%, 60%, 100% {
                        transform: translateY(0);
                        opacity: 0.5;
                    }
                    30% {
                        transform: translateY(-10px);
                        opacity: 1;
                    }
                }
                
                .info-badge {
                    font-size: 0.75rem;
                    color: #6c757d;
                    margin-top: 0.25rem;
                }
                
                .cognitive-badge {
                    display: inline-block;
                    background: #28a745;
                    color: white;
                    padding: 0.125rem 0.5rem;
                    border-radius: 0.75rem;
                    font-size: 0.7rem;
                    margin-right: 0.5rem;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧠 ReservoirPy Chat</h1>
                <p>Powered by Reservoir Computing & GraphRAG</p>
            </div>
            
            <div class="chat-container">
                <div class="messages" id="messages">
                    <div class="message assistant">
                        <div class="cognitive-badge">COGNITIVE</div>
                        Hello! I'm your ReservoirPy assistant. I can help you with reservoir computing, 
                        echo state networks, and the ReservoirPy library. My responses are enhanced 
                        with cognitive processing and knowledge retrieval. What would you like to know?
                    </div>
                </div>
                
                <div class="typing" id="typing">
                    <div class="typing-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
                
                <div class="input-area">
                    <input 
                        type="text" 
                        class="message-input" 
                        id="messageInput"
                        placeholder="Ask me about ReservoirPy, reservoir computing, or echo state networks..."
                        maxlength="1000"
                    >
                    <button class="send-btn" id="sendBtn">Send</button>
                </div>
            </div>
            
            <script>
                let currentSessionId = null;
                
                const messagesContainer = document.getElementById('messages');
                const messageInput = document.getElementById('messageInput');
                const sendBtn = document.getElementById('sendBtn');
                const typing = document.getElementById('typing');
                
                // Initialize session
                async function initializeSession() {
                    try {
                        const response = await fetch('/api/sessions', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({title: 'Chat Session'})
                        });
                        const data = await response.json();
                        currentSessionId = data.session_id;
                    } catch (error) {
                        console.error('Failed to initialize session:', error);
                    }
                }
                
                // Send message
                async function sendMessage() {
                    const message = messageInput.value.trim();
                    if (!message) return;
                    
                    // Add user message
                    addMessage(message, 'user');
                    messageInput.value = '';
                    sendBtn.disabled = true;
                    typing.style.display = 'flex';
                    
                    try {
                        const response = await fetch('/api/chat', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({
                                message: message,
                                session_id: currentSessionId,
                                use_cognitive_processing: true
                            })
                        });
                        
                        const data = await response.json();
                        
                        // Add assistant message
                        addMessage(data.message, 'assistant', data.cognitive_context);
                        
                    } catch (error) {
                        console.error('Failed to send message:', error);
                        addMessage('Sorry, I encountered an error. Please try again.', 'assistant');
                    } finally {
                        typing.style.display = 'none';
                        sendBtn.disabled = false;
                        messageInput.focus();
                    }
                }
                
                // Add message to chat
                function addMessage(text, role, cognitiveContext = null) {
                    const messageDiv = document.createElement('div');
                    messageDiv.className = `message ${role}`;
                    
                    if (role === 'assistant' && cognitiveContext?.context_used) {
                        messageDiv.classList.add('cognitive');
                    }
                    
                    let content = text;
                    if (role === 'assistant' && cognitiveContext?.context_used) {
                        content = `<div class="cognitive-badge">COGNITIVE</div>${content}`;
                    }
                    
                    if (role === 'assistant' && cognitiveContext) {
                        const nodes = cognitiveContext.retrieved_nodes || 0;
                        const edges = cognitiveContext.retrieved_edges || 0;
                        if (nodes > 0) {
                            content += `<div class="info-badge">Retrieved ${nodes} concepts, ${edges} relations</div>`;
                        }
                    }
                    
                    messageDiv.innerHTML = content;
                    messagesContainer.appendChild(messageDiv);
                    messagesContainer.scrollTop = messagesContainer.scrollHeight;
                }
                
                // Event listeners
                sendBtn.addEventListener('click', sendMessage);
                messageInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        sendMessage();
                    }
                });
                
                // Initialize
                initializeSession();
                messageInput.focus();
            </script>
        </body>
        </html>
        """
    
    async def startup(self):
        """Initialize the chat engine on startup."""
        await self.chat_engine.initialize()
    
    def run(self, **kwargs):
        """Run the web server."""
        # Setup startup event
        @self.app.on_event("startup")
        async def startup_event():
            await self.startup()
        
        # Run with uvicorn
        uvicorn_kwargs = {
            "host": self.host,
            "port": self.port,
            "log_level": "info",
            **kwargs
        }
        
        uvicorn.run(self.app, **uvicorn_kwargs)


# Convenience function for creating web interface
def create_web_interface(
    llm_provider: str = "cohere",
    llm_config: Dict[str, Any] = None,
    **kwargs
) -> ReservoirPyWebInterface:
    """
    Create a ReservoirPy web interface with the specified configuration.
    
    Parameters
    ----------
    llm_provider : str, default="cohere"
        LLM provider ("cohere", "openai")
    llm_config : Dict[str, Any], optional
        LLM configuration
    **kwargs
        Additional web interface options
        
    Returns
    -------
    ReservoirPyWebInterface
        Configured web interface
    """
    return ReservoirPyWebInterface(
        llm_provider=llm_provider,
        llm_config=llm_config,
        **kwargs
    )


def main():
    """Main entry point for running the web interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ReservoirPy Chat Web Interface")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--llm-provider", default="cohere", help="LLM provider")
    parser.add_argument("--cohere-api-key", help="Cohere API key")
    parser.add_argument("--openai-api-key", help="OpenAI API key")
    
    args = parser.parse_args()
    
    # Setup LLM config
    llm_config = {}
    if args.llm_provider == "cohere" and args.cohere_api_key:
        llm_config["api_key"] = args.cohere_api_key
    elif args.llm_provider == "openai" and args.openai_api_key:
        llm_config["api_key"] = args.openai_api_key
    
    # Create and run interface
    interface = create_web_interface(
        llm_provider=args.llm_provider,
        llm_config=llm_config,
        host=args.host,
        port=args.port
    )
    
    print(f"Starting ReservoirPy Chat Interface at http://{args.host}:{args.port}")
    interface.run()


if __name__ == "__main__":
    main()