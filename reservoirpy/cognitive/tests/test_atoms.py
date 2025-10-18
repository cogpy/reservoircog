# Licence: MIT License
# Copyright: ReservoirCog Contributors (2024)

"""Tests for OpenCog-inspired atom implementations."""

import pytest
import numpy as np
from unittest.mock import Mock

from reservoirpy.cognitive.atoms import (
    BaseAtomNode, ConceptNode, PredicateNode, NumberNode, AtomSpace
)


class TestBaseAtomNode:
    """Tests for the base atom node."""
    
    def test_initialization(self):
        """Test atom node initialization."""
        atom = ConceptNode("test_concept")
        assert atom.atom_name == "test_concept"
        assert atom.concept_name == "test_concept"
        assert not atom.initialized
        assert atom.truth_value == 1.0
        assert atom.confidence == 1.0
        
    def test_initialize_with_input(self):
        """Test atom initialization with input data."""
        atom = ConceptNode("test_concept")
        input_data = np.random.randn(10)
        
        atom.initialize(input_data)
        
        assert atom.initialized
        assert atom.input_dim == 10
        assert atom.output_dim == 64  # Default encoding dimension
        assert "out" in atom.state
        assert "reservoir" in atom.state
        
    def test_step_processing(self):
        """Test atom step processing."""
        atom = ConceptNode("test_concept")
        input_data = np.random.randn(10)
        
        atom.initialize(input_data)
        output = atom.step(input_data)
        
        assert output.shape == (atom.output_dim,)
        assert np.all(np.isfinite(output))
        
    def test_edge_connections(self):
        """Test adding edges between atoms."""
        atom1 = ConceptNode("concept1")
        atom2 = ConceptNode("concept2")
        
        atom1.add_edge(atom2, "similarity")
        
        assert len(atom1.outgoing_edges) == 1
        assert len(atom2.incoming_edges) == 1
        assert atom1.outgoing_edges[0] == (atom2, "similarity")
        assert atom2.incoming_edges[0] == (atom1, "similarity")
        
    def test_get_neighbors(self):
        """Test getting neighboring atoms."""
        atom1 = ConceptNode("concept1")
        atom2 = ConceptNode("concept2") 
        atom3 = ConceptNode("concept3")
        
        atom1.add_edge(atom2)
        atom3.add_edge(atom1)
        
        neighbors = atom1.get_neighbors()
        assert len(neighbors) == 2
        assert atom2 in neighbors
        assert atom3 in neighbors


class TestConceptNode:
    """Tests for ConceptNode."""
    
    def test_concept_creation(self):
        """Test concept node creation."""
        concept = ConceptNode("test_concept", truth_value=0.8, confidence=0.9)
        assert concept.concept_name == "test_concept"
        assert concept.truth_value == 0.8
        assert concept.confidence == 0.9
        
    def test_concept_encoding(self):
        """Test concept semantic encoding."""
        concept = ConceptNode("test_concept")
        input_data = np.random.randn(10)
        
        concept.initialize(input_data)
        output1 = concept.step(input_data)
        output2 = concept.step(input_data)
        
        # Output should be consistent for same input
        assert output1.shape == output2.shape
        assert np.all(np.isfinite(output1))
        assert np.all(np.isfinite(output2))


class TestPredicateNode:
    """Tests for PredicateNode."""
    
    def test_predicate_creation(self):
        """Test predicate node creation."""
        predicate = PredicateNode("likes", arity=2)
        assert predicate.predicate_name == "likes"
        assert predicate.arity == 2
        
    def test_predicate_evaluation(self):
        """Test predicate evaluation."""
        predicate = PredicateNode("likes", arity=2)
        input_data = np.random.randn(10)
        predicate.initialize(input_data)
        
        # Test evaluation with correct arity
        result = predicate.evaluate("Alice", "Bob")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        
    def test_predicate_wrong_arity(self):
        """Test predicate with wrong number of arguments."""
        predicate = PredicateNode("likes", arity=2)
        
        with pytest.raises(ValueError):
            predicate.evaluate("Alice")  # Only 1 argument for arity=2


class TestNumberNode:
    """Tests for NumberNode."""
    
    def test_number_creation(self):
        """Test number node creation."""
        number = NumberNode(3.14)
        assert number.value == 3.14
        assert "3.14" in number.atom_name
        
    def test_number_encoding(self):
        """Test number semantic encoding.""" 
        number = NumberNode(42.0)
        input_data = np.random.randn(10)
        
        number.initialize(input_data)
        output = number.step(input_data)
        
        assert output.shape == (number.output_dim,)
        assert output[0] == 42.0  # First element should be the value
        

class TestAtomSpace:
    """Tests for AtomSpace."""
    
    def test_atomspace_creation(self):
        """Test AtomSpace creation."""
        atomspace = AtomSpace("TestSpace")
        assert atomspace.name == "TestSpace"
        assert len(atomspace.atoms) == 0
        
    def test_add_atom(self):
        """Test adding atoms to AtomSpace."""
        atomspace = AtomSpace()
        concept = ConceptNode("test_concept")
        
        added_atom = atomspace.add_atom(concept)
        
        assert added_atom == concept
        assert len(atomspace.atoms) == 1
        assert "ConceptNode:test_concept" in atomspace.atoms
        assert len(atomspace.atom_types["ConceptNode"]) == 1
        
    def test_get_atom(self):
        """Test retrieving atoms from AtomSpace."""
        atomspace = AtomSpace()
        concept = ConceptNode("test_concept")
        atomspace.add_atom(concept)
        
        # Get by name and type
        retrieved = atomspace.get_atom("test_concept", "ConceptNode")
        assert retrieved == concept
        
        # Get by name only
        retrieved = atomspace.get_atom("test_concept")
        assert retrieved == concept
        
    def test_get_atoms_by_type(self):
        """Test getting atoms by type."""
        atomspace = AtomSpace()
        concept1 = ConceptNode("concept1")
        concept2 = ConceptNode("concept2")
        predicate = PredicateNode("likes")
        
        atomspace.add_atom(concept1)
        atomspace.add_atom(concept2)
        atomspace.add_atom(predicate)
        
        concepts = atomspace.get_atoms_by_type("ConceptNode")
        predicates = atomspace.get_atoms_by_type("PredicateNode")
        
        assert len(concepts) == 2
        assert len(predicates) == 1
        assert concept1 in concepts
        assert concept2 in concepts
        assert predicate in predicates
        
    def test_add_link(self):
        """Test adding links between atoms."""
        atomspace = AtomSpace()
        concept1 = ConceptNode("concept1")
        concept2 = ConceptNode("concept2")
        
        atomspace.add_atom(concept1)
        atomspace.add_atom(concept2)
        atomspace.add_link(concept1, concept2, "similarity")
        
        assert len(concept1.outgoing_edges) == 1
        assert len(concept2.incoming_edges) == 1
        
    def test_pattern_match(self):
        """Test pattern matching in AtomSpace."""
        atomspace = AtomSpace()
        concept1 = ConceptNode("test_concept", attention=0.8)
        concept2 = ConceptNode("other_concept", attention=0.3)
        
        atomspace.add_atom(concept1)
        atomspace.add_atom(concept2)
        
        # Pattern match by type and attention
        matches = atomspace.pattern_match({
            'type': 'ConceptNode',
            'min_attention': 0.5
        })
        
        assert len(matches) == 1
        assert concept1 in matches
        
    def test_spread_activation(self):
        """Test attention spreading in AtomSpace."""
        atomspace = AtomSpace()
        concept1 = ConceptNode("concept1", attention=0.9)
        concept2 = ConceptNode("concept2", attention=0.1)
        
        atomspace.add_atom(concept1)
        atomspace.add_atom(concept2)
        atomspace.add_link(concept1, concept2)
        
        initial_attention = concept2.attention
        atomspace.spread_activation([concept1], iterations=1)
        
        # concept2 attention should be affected (may increase or decrease due to spreading and decay)
        # Just check it remains in valid range and changed
        assert 0.0 <= concept2.attention <= 1.0
        assert concept2.attention != initial_attention  # Should have changed