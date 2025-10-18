# Licence: MIT License
# Copyright: ReservoirCog Contributors (2024)

"""Tests for cognitive agents."""

import pytest
import numpy as np

from reservoirpy.cognitive.agents import CognitiveAgent, AttentionAgent, PatternMatchAgent
from reservoirpy.cognitive.atoms import AtomSpace, ConceptNode


class TestCognitiveAgent:
    """Tests for CognitiveAgent."""
    
    def test_agent_initialization(self):
        """Test cognitive agent initialization."""
        agent = CognitiveAgent("test_agent", specialty="reasoning")
        assert agent.agent_name == "test_agent"
        assert agent.specialty == "reasoning"
        assert not agent.initialized
        assert len(agent.connected_agents) == 0
        
    def test_agent_initialize_with_input(self):
        """Test agent initialization with input data."""
        agent = CognitiveAgent("test_agent")
        input_data = np.random.randn(10)
        
        agent.initialize(input_data)
        
        assert agent.initialized
        assert agent.input_dim == 10
        assert agent.output_dim == 10
        assert "out" in agent.state
        assert "reservoir" in agent.state
        
    def test_agent_step_processing(self):
        """Test agent step processing."""
        agent = CognitiveAgent("test_agent", specialty="memory")
        input_data = np.random.randn(10)
        
        agent.initialize(input_data)
        output = agent.step(input_data)
        
        assert output.shape == (agent.output_dim,)
        assert np.all(np.isfinite(output))
        assert "activation_level" in agent.state
        assert "attention_focus" in agent.state
        
    def test_agent_connection(self):
        """Test connecting agents."""
        agent1 = CognitiveAgent("agent1")
        agent2 = CognitiveAgent("agent2")
        
        agent1.connect_agent(agent2)
        
        assert agent2 in agent1.connected_agents
        assert agent1 in agent2.connected_agents
        
    def test_agent_messaging(self):
        """Test agent messaging."""
        agent1 = CognitiveAgent("agent1")
        agent2 = CognitiveAgent("agent2")
        
        agent1.connect_agent(agent2)
        
        message = np.array([1.0, 2.0, 3.0])
        agent1.send_message(message, agent2)
        
        assert len(agent2.message_buffer) == 1
        assert np.array_equal(agent2.message_buffer[0]['message'], message)
        assert agent2.message_buffer[0]['sender'] == agent1
        
    def test_agent_broadcast(self):
        """Test agent message broadcasting."""
        agent1 = CognitiveAgent("agent1")
        agent2 = CognitiveAgent("agent2")
        agent3 = CognitiveAgent("agent3")
        
        agent1.connect_agent(agent2)
        agent1.connect_agent(agent3)
        
        message = np.array([1.0, 2.0])
        agent1.send_message(message)  # Broadcast
        
        assert len(agent2.message_buffer) == 1
        assert len(agent3.message_buffer) == 1
        
    def test_set_atomspace(self):
        """Test connecting agent to AtomSpace."""
        agent = CognitiveAgent("test_agent")
        atomspace = AtomSpace("TestSpace")
        
        agent.set_atomspace(atomspace)
        
        assert agent.atomspace_ref == atomspace


class TestAttentionAgent:
    """Tests for AttentionAgent."""
    
    def test_attention_agent_creation(self):
        """Test attention agent creation."""
        agent = AttentionAgent()
        assert agent.specialty == "attention"
        assert agent.agent_name == "AttentionAgent"
        assert agent.focus_threshold == 0.7
        
    def test_attention_allocation(self):
        """Test attention allocation across AtomSpace."""
        agent = AttentionAgent()
        atomspace = AtomSpace()
        
        # Create test atoms
        concept1 = ConceptNode("concept1", attention=0.5)
        concept2 = ConceptNode("concept2", attention=0.3)
        
        atomspace.add_atom(concept1)
        atomspace.add_atom(concept2)
        
        # Initialize atoms
        input_data = np.random.randn(10)
        concept1.initialize(input_data)
        concept2.initialize(input_data)
        
        agent.allocate_attention(atomspace)
        
        # Check that attention values were updated
        assert concept1.attention >= 0.0
        assert concept2.attention >= 0.0
        assert len(agent.attention_map) == 2
        
    def test_get_focus_atoms(self):
        """Test getting atoms in focus."""
        agent = AttentionAgent()
        atomspace = AtomSpace()
        
        # Create atoms with different attention levels
        high_attention = ConceptNode("high", attention=0.8)
        low_attention = ConceptNode("low", attention=0.5)
        
        atomspace.add_atom(high_attention)
        atomspace.add_atom(low_attention)
        
        focus_atoms = agent.get_focus_atoms(atomspace)
        
        assert high_attention in focus_atoms
        assert low_attention not in focus_atoms  # Below threshold


class TestPatternMatchAgent:
    """Tests for PatternMatchAgent."""
    
    def test_pattern_agent_creation(self):
        """Test pattern matching agent creation."""
        agent = PatternMatchAgent()
        assert agent.specialty == "reasoning"
        assert agent.agent_name == "PatternMatchAgent"
        assert agent.match_threshold == 0.8
        
    def test_add_pattern_template(self):
        """Test adding pattern templates."""
        agent = PatternMatchAgent()
        
        template = {'type': 'ConceptNode', 'property': 'similarity'}
        agent.add_pattern_template("similarity_pattern", template)
        
        assert "similarity_pattern" in agent.pattern_templates
        assert agent.pattern_templates["similarity_pattern"] == template
        
    def test_match_pattern(self):
        """Test pattern matching against AtomSpace."""
        agent = PatternMatchAgent()
        atomspace = AtomSpace()
        
        # Setup atoms and agent
        concept = ConceptNode("test_concept")
        atomspace.add_atom(concept)
        
        input_data = np.random.randn(10)
        agent.initialize(input_data)
        concept.initialize(input_data)
        
        # Add pattern template
        template = {'type': 'ConceptNode'}
        agent.add_pattern_template("concept_pattern", template)
        
        # Test pattern matching
        matches = agent.match_pattern(atomspace, "concept_pattern")
        
        # Should return list (may be empty depending on threshold)
        assert isinstance(matches, list)
        
    def test_find_similar_atoms(self):
        """Test finding similar atoms."""
        agent = PatternMatchAgent()
        atomspace = AtomSpace()
        
        # Create test atoms
        concept1 = ConceptNode("concept1")
        concept2 = ConceptNode("concept2")
        concept3 = ConceptNode("concept3")
        
        atomspace.add_atom(concept1)
        atomspace.add_atom(concept2)
        atomspace.add_atom(concept3)
        
        # Initialize all
        input_data = np.random.randn(10)
        agent.initialize(input_data)
        concept1.initialize(input_data)
        concept2.initialize(input_data)
        concept3.initialize(input_data)
        
        # Find similar atoms
        similar = agent.find_similar_atoms(atomspace, concept1, top_k=2)
        
        assert len(similar) <= 2
        assert concept1 not in similar  # Query atom should not be in results
        
    def test_agent_specialty_attention(self):
        """Test different agent specialties produce different attention patterns."""
        # Create agents with different specialties
        memory_agent = CognitiveAgent("memory", specialty="memory")
        reasoning_agent = CognitiveAgent("reasoning", specialty="reasoning")
        attention_agent = CognitiveAgent("attention", specialty="attention")
        
        input_data = np.random.randn(10)
        
        # Initialize all agents
        memory_agent.initialize(input_data)
        reasoning_agent.initialize(input_data)
        attention_agent.initialize(input_data)
        
        # Process same input
        memory_output = memory_agent.step(input_data)
        reasoning_output = reasoning_agent.step(input_data)
        attention_output = attention_agent.step(input_data)
        
        # All should produce valid outputs
        assert np.all(np.isfinite(memory_output))
        assert np.all(np.isfinite(reasoning_output))
        assert np.all(np.isfinite(attention_output))
        
        # Attention patterns should be different due to different specialties
        memory_attention = memory_agent.state["attention_focus"]
        reasoning_attention = reasoning_agent.state["attention_focus"]
        attention_focus = attention_agent.state["attention_focus"]
        
        # They should not all be identical (very unlikely with reservoir dynamics)
        assert not (np.allclose(memory_attention, reasoning_attention) and 
                   np.allclose(reasoning_attention, attention_focus))