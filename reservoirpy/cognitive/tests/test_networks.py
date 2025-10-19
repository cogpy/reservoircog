# Licence: MIT License
# Copyright: ReservoirCog Contributors (2024)

"""Tests for distributed networks and cognitive orchestrator."""

import pytest
import numpy as np

from reservoirpy.cognitive.networks import DistributedEchoNetwork, CognitiveOrchestrator
from reservoirpy.cognitive.atoms import ConceptNode, PredicateNode, NumberNode


class TestDistributedEchoNetwork:
    """Tests for DistributedEchoNetwork."""
    
    def test_network_creation(self):
        """Test distributed network creation."""
        network = DistributedEchoNetwork(num_agents=5)
        
        assert network.name == "DistributedEchoNetwork"
        assert len(network.agents) >= 2  # Should have at least attention and pattern agents
        assert network.attention_agent is not None
        assert network.pattern_agent is not None
        
    def test_agent_connections(self):
        """Test agent connections in network."""
        network = DistributedEchoNetwork(num_agents=4)
        
        # Check that agents are connected
        total_connections = sum(len(connections) for connections in network.agent_connections.values())
        assert total_connections > 0
        
        # Check communication matrix exists
        assert network.communication_matrix is not None
        assert network.communication_matrix.shape[0] == len(network.agents)
        
    def test_distributed_processing(self):
        """Test distributed processing across agents."""
        network = DistributedEchoNetwork(num_agents=3)
        input_data = np.random.randn(10)
        
        results = network.process_distributed(input_data, steps=5)
        
        assert len(results) == len(network.agents)
        for agent_name, states in results.items():
            assert states.shape[0] == 5  # 5 processing steps
            assert states.shape[1] == 10  # Output dimension matches input
            assert np.all(np.isfinite(states))
            
    def test_network_synchronization(self):
        """Test network synchronization."""
        network = DistributedEchoNetwork(num_agents=3)
        input_data = np.random.randn(10)
        
        # Process with synchronization
        results = network.process_distributed(input_data, steps=10)
        
        # Check that global activity was updated
        assert hasattr(network, 'global_activity')
        assert network.global_activity >= 0.0
        
    def test_get_network_state(self):
        """Test getting network state."""
        network = DistributedEchoNetwork(num_agents=3)
        
        state = network.get_network_state()
        
        assert 'global_activity' in state
        assert 'num_agents' in state
        assert 'agent_activations' in state
        assert 'connections' in state
        assert state['num_agents'] == len(network.agents)


class TestCognitiveOrchestrator:
    """Tests for CognitiveOrchestrator."""
    
    def test_orchestrator_creation(self):
        """Test cognitive orchestrator creation."""
        orchestrator = CognitiveOrchestrator()
        
        assert orchestrator.atomspace is not None
        assert orchestrator.agent_network is not None
        assert orchestrator.attention_cycles == 0
        
    def test_add_concept(self):
        """Test adding concepts to orchestrator."""
        orchestrator = CognitiveOrchestrator()
        
        concept = orchestrator.add_concept("test_concept", truth_value=0.8)
        
        assert isinstance(concept, ConceptNode)
        assert concept.concept_name == "test_concept"
        assert concept.truth_value == 0.8
        assert len(orchestrator.atomspace.atoms) == 1
        
    def test_add_predicate(self):
        """Test adding predicates to orchestrator."""
        orchestrator = CognitiveOrchestrator()
        
        predicate = orchestrator.add_predicate("likes", arity=2, confidence=0.9)
        
        assert isinstance(predicate, PredicateNode)
        assert predicate.predicate_name == "likes"
        assert predicate.arity == 2
        assert predicate.confidence == 0.9
        
    def test_add_number(self):
        """Test adding numbers to orchestrator."""
        orchestrator = CognitiveOrchestrator()
        
        number = orchestrator.add_number(42.0, attention=0.7)
        
        assert isinstance(number, NumberNode)
        assert number.value == 42.0
        assert number.attention == 0.7
        
    def test_create_link(self):
        """Test creating links between atoms."""
        orchestrator = CognitiveOrchestrator()
        
        concept1 = orchestrator.add_concept("concept1")
        concept2 = orchestrator.add_concept("concept2")
        
        orchestrator.create_link(concept1, concept2, "similarity")
        
        assert len(concept1.outgoing_edges) == 1
        assert concept1.outgoing_edges[0] == (concept2, "similarity")
        
    def test_cognitive_cycle_basic(self):
        """Test basic cognitive cycle execution."""
        orchestrator = CognitiveOrchestrator(network_size=3)
        
        # Add some atoms to work with
        orchestrator.add_concept("test_concept")
        orchestrator.add_predicate("test_predicate")
        
        # Run cognitive cycle
        input_stimulus = np.random.randn(10)
        results = orchestrator.cognitive_cycle(input_stimulus, cycles=2)
        
        assert 'cycles' in results
        assert 'total_cycles' in results
        assert 'average_cycle_time' in results
        assert results['total_cycles'] == 2
        
        # Check cycle structure
        for cycle_result in results['cycles']:
            assert 'perception' in cycle_result
            assert 'attention' in cycle_result
            assert 'patterns' in cycle_result
            assert 'reasoning' in cycle_result
            assert 'actions' in cycle_result
            
    def test_cognitive_cycle_no_input(self):
        """Test cognitive cycle without input stimulus."""
        orchestrator = CognitiveOrchestrator(network_size=2)
        
        results = orchestrator.cognitive_cycle(cycles=1)
        
        assert results['total_cycles'] == 1
        # Should still run other phases even without input
        cycle = results['cycles'][0]
        assert 'attention' in cycle
        assert 'patterns' in cycle
        
    def test_perception_phase(self):
        """Test perception phase processing."""
        orchestrator = CognitiveOrchestrator(network_size=3)
        
        input_stimulus = np.random.randn(10)
        perception_results = orchestrator._perception_phase(input_stimulus)
        
        assert 'agent_responses' in perception_results
        assert 'concepts_created' in perception_results
        assert 'average_salience' in perception_results
        assert isinstance(perception_results['concepts_created'], int)
        
    def test_attention_phase(self):
        """Test attention phase processing."""
        orchestrator = CognitiveOrchestrator()
        
        # Add some atoms
        orchestrator.add_concept("concept1", attention=0.8)
        orchestrator.add_concept("concept2", attention=0.3)
        
        attention_results = orchestrator._attention_phase()
        
        assert 'focus_atoms' in attention_results
        assert 'total_atoms' in attention_results
        assert 'attention_cycles' in attention_results
        assert orchestrator.attention_cycles > 0
        
    def test_pattern_matching_phase(self):
        """Test pattern matching phase."""
        orchestrator = CognitiveOrchestrator()
        
        # Add atoms with high attention
        concept = orchestrator.add_concept("high_attention", attention=0.9)
        input_data = np.random.randn(10)
        concept.initialize(input_data)
        
        pattern_results = orchestrator._pattern_matching_phase()
        
        assert 'patterns_found' in pattern_results
        assert 'pattern_details' in pattern_results
        assert isinstance(pattern_results['patterns_found'], int)
        
    def test_reasoning_phase(self):
        """Test reasoning phase processing."""
        orchestrator = CognitiveOrchestrator()
        
        # Add connected atoms
        concept1 = orchestrator.add_concept("concept1", attention=0.8)
        concept2 = orchestrator.add_concept("concept2", attention=0.5)
        orchestrator.create_link(concept1, concept2)
        
        reasoning_results = orchestrator._reasoning_phase()
        
        assert 'atoms_processed' in reasoning_results
        assert 'total_activations' in reasoning_results
        assert 'reasoning_details' in reasoning_results
        
    def test_action_selection_phase(self):
        """Test action selection phase."""
        orchestrator = CognitiveOrchestrator(network_size=2)
        
        # Process something to create network activity
        input_data = np.random.randn(10)
        orchestrator.agent_network.process_distributed(input_data, steps=1)
        
        action_results = orchestrator._action_selection_phase()
        
        assert 'selected_actions' in action_results
        assert 'global_activity' in action_results
        assert 'action_confidence' in action_results
        assert isinstance(action_results['selected_actions'], list)
        assert len(action_results['selected_actions']) > 0
        
    def test_get_cognitive_state(self):
        """Test getting comprehensive cognitive state."""
        orchestrator = CognitiveOrchestrator()
        
        # Add some atoms
        orchestrator.add_concept("concept1")
        orchestrator.add_predicate("predicate1")
        
        state = orchestrator.get_cognitive_state()
        
        assert 'atomspace' in state
        assert 'agent_network' in state
        assert 'processing_stats' in state
        
        atomspace_info = state['atomspace']
        assert 'total_atoms' in atomspace_info
        assert 'atom_types' in atomspace_info
        assert atomspace_info['total_atoms'] == 2
        
    def test_orchestrator_integration(self):
        """Test full integration of orchestrator components."""
        orchestrator = CognitiveOrchestrator(network_size=4)
        
        # Build a small knowledge base
        alice = orchestrator.add_concept("Alice", attention=0.7)
        bob = orchestrator.add_concept("Bob", attention=0.6)
        likes = orchestrator.add_predicate("likes", arity=2)
        
        orchestrator.create_link(alice, bob, "knows")
        orchestrator.create_link(alice, likes, "subject")
        
        # Initialize atoms
        input_data = np.random.randn(10)
        alice.initialize(input_data)
        bob.initialize(input_data)
        likes.initialize(input_data)
        
        # Run multiple cognitive cycles
        results = orchestrator.cognitive_cycle(input_data, cycles=3)
        
        # Verify processing occurred
        assert results['total_cycles'] == 3
        assert results['average_cycle_time'] > 0
        
        # Check that attention and reasoning affected the atoms
        final_state = orchestrator.get_cognitive_state()
        assert final_state['atomspace']['total_atoms'] >= 3
        assert final_state['processing_stats']['attention_cycles'] >= 3
        
    def test_orchestrator_with_numbers(self):
        """Test orchestrator with numerical atoms."""
        orchestrator = CognitiveOrchestrator()
        
        # Add numerical concepts
        pi = orchestrator.add_number(3.14159)
        age = orchestrator.add_number(25.0)
        
        # Link numbers conceptually
        orchestrator.create_link(pi, age, "different_domains")
        
        # Process through cognitive cycle
        input_data = np.array([3.14159, 25.0, 1.0, 0.0])
        results = orchestrator.cognitive_cycle(input_data, cycles=1)
        
        assert results['total_cycles'] == 1
        cycle = results['cycles'][0]
        
        # Should have processed the numerical input
        assert cycle['atomspace_size'] >= 2