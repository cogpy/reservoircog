"""
==========================================
ReservoirCog Cognitive Orchestrator (:mod:`reservoirpy.cognitive`)
==========================================

OpenCog-inspired cognitive architecture using ReservoirPy's reservoir computing
framework. This module provides AtomSpace-like functionality through reservoir
computing nodes and distributed echo state agent networks.

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
"""

from .atoms import AtomSpace, ConceptNode, NumberNode, PredicateNode
from .agents import AttentionAgent, CognitiveAgent, PatternMatchAgent
from .networks import CognitiveOrchestrator, DistributedEchoNetwork

__all__ = [
    "ConceptNode",
    "PredicateNode", 
    "NumberNode",
    "AtomSpace",
    "CognitiveAgent",
    "AttentionAgent",
    "PatternMatchAgent",
    "DistributedEchoNetwork",
    "CognitiveOrchestrator",
]