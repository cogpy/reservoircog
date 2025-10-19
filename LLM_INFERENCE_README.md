# ReservoirPy LLM Inference Engine

This implementation adds a complete LLM inference engine to ReservoirPy that replicates the functionality of `https://chat.reservoirpy.inria.fr/` using GraphRAG and Cohere/OpenAI integration.

## 🚀 Features

### Core Components

1. **LLM Interface** (`llm_interface.py`)
   - Multi-provider support (Cohere, OpenAI)
   - Async text generation and embeddings
   - Streaming response capabilities
   - Extensible architecture for new providers

2. **GraphRAG Engine** (`graphrag.py`)
   - Knowledge graph enhanced retrieval
   - Integration with ReservoirPy AtomSpace
   - Vector similarity search
   - Context-aware response generation

3. **Chat Engine** (`chat_engine.py`)
   - Conversation management and session handling
   - Integration between cognitive architecture and LLM
   - Streaming and non-streaming chat modes
   - Knowledge base expansion

4. **Web Interface** (`web_interface.py`)
   - FastAPI-based REST API
   - HTML chat interface similar to chat.reservoirpy.inria.fr
   - Real-time streaming responses
   - Session management UI

## 📦 Installation

### Dependencies

```bash
# Core ReservoirPy dependencies
pip install numpy scipy joblib scikit-learn

# LLM inference dependencies
pip install cohere llama-index fastapi uvicorn

# Optional: For development and testing
pip install pytest pytest-asyncio
```

### Environment Setup

Set your API keys:

```bash
# For Cohere (recommended)
export COHERE_API_KEY="your-cohere-api-key"

# Or for OpenAI
export OPENAI_API_KEY="your-openai-api-key"
```

## 🚀 Quick Start

### 1. Basic Chat Engine

```python
import asyncio
from reservoirpy.cognitive import create_chat_engine

async def basic_example():
    # Create chat engine
    engine = create_chat_engine(
        llm_provider="cohere",  # or "openai"
        llm_config={"api_key": "your-key"}
    )
    
    # Initialize the engine
    await engine.initialize()
    
    # Create a chat session
    session_id = await engine.create_session(title="My Chat")
    
    # Send a message
    response = await engine.chat(
        message="What is ReservoirPy?",
        session_id=session_id,
        use_cognitive_processing=True
    )
    
    print(f"Assistant: {response['message']}")
    print(f"Used cognitive context: {response['cognitive_context']['context_used']}")

# Run the example
asyncio.run(basic_example())
```

### 2. Web Interface

```python
from reservoirpy.cognitive import create_web_interface

# Create and run web interface
interface = create_web_interface(
    llm_provider="cohere",
    llm_config={"api_key": "your-cohere-key"},
    host="localhost",
    port=8000
)

# This starts the web server
interface.run()
```

Then visit `http://localhost:8000` for the chat interface.

### 3. Command Line Demo

```bash
# Run the comprehensive demo
python examples/reservoirpy_chat_demo.py --mode demo

# Interactive CLI chat
python examples/reservoirpy_chat_demo.py --mode cli

# Web interface
python examples/reservoirpy_chat_demo.py --mode web --port 8000
```

## 🏗️ Architecture

```
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
```

## 💡 Key Features

### Cognitive Processing
- **AtomSpace Integration**: Leverages ReservoirPy's cognitive architecture
- **Attention Allocation**: Dynamic resource management through reservoir networks
- **Pattern Matching**: Complex structural pattern recognition
- **Distributed Agents**: Multi-agent cognitive processing

### GraphRAG Capabilities
- **Context Retrieval**: Intelligent context extraction from knowledge graphs
- **Similarity Search**: Vector-based content retrieval
- **Knowledge Expansion**: Dynamic addition of new knowledge
- **Relation Tracking**: Graph-based relationship understanding

### LLM Integration
- **Multi-Provider**: Support for Cohere, OpenAI, and extensible to others
- **Streaming**: Real-time response generation
- **Embeddings**: Semantic vector representations
- **Error Handling**: Robust error recovery and fallbacks

## 🔧 Configuration

### LLM Providers

#### Cohere Configuration
```python
llm_config = {
    "api_key": "your-cohere-key",
    "model_name": "command-r-plus",  # Default
    "embed_model": "embed-english-v3.0"  # Default
}
```

#### OpenAI Configuration
```python
llm_config = {
    "api_key": "your-openai-key", 
    "model_name": "gpt-3.5-turbo",  # Default
    "embed_model": "text-embedding-3-small"  # Default
}
```

### Cognitive Configuration
```python
cognitive_config = {
    "network_size": 5,  # Number of cognitive agents
    "attention_threshold": 0.7,
    "max_retrieved_nodes": 10
}
```

## 🌐 API Reference

### REST API Endpoints

- `POST /api/chat` - Send chat messages
- `POST /api/chat/stream` - Stream chat responses
- `POST /api/sessions` - Create chat sessions
- `GET /api/sessions` - List all sessions
- `GET /api/sessions/{id}` - Get specific session
- `GET /api/sessions/{id}/history` - Get conversation history
- `POST /api/knowledge` - Add knowledge to system
- `GET /api/info` - System information
- `GET /health` - Health check

### Main Classes

#### `ReservoirPyChatEngine`
Main orchestrator that integrates all components.

#### `BaseLLMInterface`
Abstract base for LLM providers.

#### `GraphRAGEngine`
Handles knowledge graph operations and retrieval.

#### `ReservoirPyWebInterface`
FastAPI web application with chat UI.

## 🧪 Testing

```bash
# Run all cognitive tests
python -m pytest reservoirpy/cognitive/tests/ -v

# Run LLM interface tests
python -m pytest reservoirpy/cognitive/tests/test_llm_interface.py -v

# Run chat engine tests  
python -m pytest reservoirpy/cognitive/tests/test_chat_engine.py -v
```

## 📋 Examples

### Streaming Chat
```python
async def streaming_example():
    engine = create_chat_engine("cohere")
    await engine.initialize()
    
    session_id = await engine.create_session()
    
    print("Assistant: ", end="", flush=True)
    async for chunk in await engine.chat(
        message="Explain reservoir computing",
        session_id=session_id,
        stream=True
    ):
        if chunk.get("type") == "token":
            print(chunk["content"], end="", flush=True)
    print()  # New line
```

### Knowledge Addition
```python
async def knowledge_example():
    engine = create_chat_engine("cohere")
    await engine.initialize()
    
    # Add custom knowledge
    node_id = await engine.add_knowledge_from_message(
        content="ReservoirPy 2.0 includes advanced JAX support for GPU acceleration",
        source="documentation"
    )
    
    # Query using the new knowledge
    response = await engine.chat(
        message="Tell me about JAX support in ReservoirPy",
        use_cognitive_processing=True
    )
    
    if response['cognitive_context']['context_used']:
        print("✅ Used custom knowledge in response")
```

## 🔗 Integration with ReservoirPy

This LLM inference engine seamlessly integrates with existing ReservoirPy components:

- **ConceptNode/PredicateNode**: Converted to GraphRAG knowledge nodes
- **AtomSpace**: Used as the knowledge graph backend
- **CognitiveOrchestrator**: Provides cognitive processing capabilities
- **Reservoir Networks**: Power the underlying cognitive dynamics

## 🤝 Contributing

1. **Add new LLM providers**: Extend `BaseLLMInterface`
2. **Enhance GraphRAG**: Improve retrieval algorithms
3. **UI improvements**: Enhance the web interface
4. **Performance**: Optimize cognitive processing

## 📄 License

This implementation follows the same MIT License as ReservoirPy.

## 🙏 Acknowledgments

- Built on ReservoirPy's cognitive architecture
- Integrates Cohere and OpenAI LLM capabilities
- Uses LlamaIndex for advanced RAG functionality
- FastAPI for modern web API development

---

*This implementation provides a complete ChatGPT-like interface for ReservoirPy, combining the power of reservoir computing with modern LLM capabilities.*