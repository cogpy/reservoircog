"""
==========================================
ReservoirCog Cognitive Orchestrator (:mod:`reservoirpy.cognitive`)
==========================================

OpenCog-inspired cognitive architecture using ReservoirPy's reservoir computing
framework with LLM inference capabilities via GraphRAG and Cohere integration.
This module provides AtomSpace-like functionality through reservoir computing 
nodes and distributed echo state agent networks, enhanced with modern LLM capabilities.

AtomSpace Components
===================

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   ConceptNode - Concept representation using reservoir dynamics
   PredicateNode - Predicate evaluation through reservoir networks
   NumberNode - Numerical atom with reservoir-based processing
   AtomSpace - Cognitive knowledge graph using distributed reservoirs

Cognitive Agents
===============

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   CognitiveAgent - Echo state network agent for distributed reasoning
   AttentionAgent - Attention allocation mechanism
   PatternMatchAgent - Pattern matching through reservoir dynamics

Distributed Networks
===================

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   DistributedEchoNetwork - Network of interconnected echo state agents
   CognitiveOrchestrator - Main orchestrator for cognitive processes

LLM Inference Engine
===================

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   BaseLLMInterface - Base class for LLM interfaces
   CohereLLMInterface - Cohere LLM interface for generation and embeddings
   OpenAILLMInterface - OpenAI LLM interface for GPT models
   GraphRAGEngine - Graph-based Retrieval Augmented Generation
   ReservoirPyChatEngine - Main chat interface combining cognitive + LLM
   ReservoirPyWebInterface - Web interface for chat functionality

Factory Functions
================

.. autosummary::
   :toctree: generated/
   :template: autosummary/function.rst

   create_llm_interface - Create LLM interface for specified provider
   create_graphrag_engine - Create GraphRAG engine with AtomSpace
   create_chat_engine - Create complete chat engine
   create_web_interface - Create web interface for chat
"""

from .atoms import AtomSpace, ConceptNode, NumberNode, PredicateNode
from .agents import AttentionAgent, CognitiveAgent, PatternMatchAgent
from .networks import CognitiveOrchestrator, DistributedEchoNetwork
from .llm_interface import BaseLLMInterface, CohereLLMInterface, OpenAILLMInterface, create_llm_interface
from .graphrag import GraphRAGEngine, create_graphrag_engine
from .chat_engine import ReservoirPyChatEngine, create_chat_engine

# Optional web interface (only if FastAPI is available)
try:
    from .web_interface import ReservoirPyWebInterface, create_web_interface
    WEB_INTERFACE_AVAILABLE = True
except ImportError:
    WEB_INTERFACE_AVAILABLE = False

__all__ = [
    # Core cognitive components
    "ConceptNode",
    "PredicateNode", 
    "NumberNode",
    "AtomSpace",
    "CognitiveAgent",
    "AttentionAgent",
    "PatternMatchAgent",
    "DistributedEchoNetwork",
    "CognitiveOrchestrator",
    # LLM inference components
    "BaseLLMInterface",
    "CohereLLMInterface", 
    "OpenAILLMInterface",
    "create_llm_interface",
    "GraphRAGEngine",
    "create_graphrag_engine",
    "ReservoirPyChatEngine",
    "create_chat_engine",
]

# Add web interface to __all__ if available
if WEB_INTERFACE_AVAILABLE:
    __all__.extend([
        "ReservoirPyWebInterface",
        "create_web_interface"
    ])