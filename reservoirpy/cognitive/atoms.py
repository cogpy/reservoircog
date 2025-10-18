# Licence: MIT License
# Copyright: ReservoirCog Contributors (2024)

"""
OpenCog-inspired atom types implemented as ReservoirPy nodes.

This module provides AtomSpace functionality through reservoir computing,
mapping OpenCog atom concepts to dynamic neural reservoirs.
"""

from typing import Dict, List, Optional, Union, Any
import numpy as np
from collections import defaultdict

from ..node import Node
from ..nodes import Reservoir, Input, Output, Ridge
from ..model import Model
from ..type import NodeInput, State, Timestep, Timeseries


class BaseAtomNode(Node):
    """Base class for OpenCog-inspired atom nodes using reservoir dynamics.
    
    Each atom maintains internal reservoir dynamics that encode its semantic
    content and relationships with other atoms in the AtomSpace.
    """
    
    def __init__(
        self,
        name: str,
        units: int = 100,
        sr: float = 0.9,
        lr: float = 0.3,
        input_scaling: float = 1.0,
        bias: float = 0.0,
        **kwargs
    ):
        """Initialize base atom node.
        
        Parameters
        ----------
        name : str
            Name/identifier for this atom
        units : int, default=100  
            Number of reservoir units
        sr : float, default=0.9
            Spectral radius of reservoir
        lr : float, default=0.3
            Leak rate of reservoir neurons
        input_scaling : float, default=1.0
            Input connection scaling
        bias_scaling : float, default=0.0
            Bias scaling for reservoir
        """
        self.atom_name = name
        self.units = units
        self.sr = sr
        self.lr = lr
        self.input_scaling = input_scaling
        self.bias = bias
        
        # Internal reservoir for semantic dynamics
        self._reservoir = None
        self._readout = None
        self._encoding_dim = kwargs.get('encoding_dim', 64)
        
        # Atom relationships and truth values
        self.truth_value = kwargs.get('truth_value', 1.0)
        self.confidence = kwargs.get('confidence', 1.0)
        self.attention = kwargs.get('attention', 0.5)
        
        # Connected atoms (edges in AtomSpace graph)
        self.incoming_edges = []
        self.outgoing_edges = []
        
        super().__init__()
        
    def initialize(self, x: Union[NodeInput, Timestep], y: Optional[Union[NodeInput, Timestep]] = None):
        """Initialize the atom's reservoir dynamics."""
        if isinstance(x, np.ndarray):
            if len(x.shape) == 1:
                input_dim = len(x)
            else:
                input_dim = x.shape[-1]
        else:
            input_dim = self._encoding_dim
            
        self.input_dim = input_dim
        self.output_dim = self._encoding_dim
        
        # Initialize internal reservoir for semantic processing
        self._reservoir = Reservoir(
            units=self.units,
            sr=self.sr,
            lr=self.lr,
            input_scaling=self.input_scaling,
            bias=self.bias
        )
        
        # Readout for atom encoding/representation
        self._readout = Ridge(ridge=1e-6)
        
        # Initialize reservoir with input
        self._reservoir.initialize(x)
        
        # Create initial state
        reservoir_state = np.zeros(self.units)
        self.state = {
            "out": np.zeros(self.output_dim),
            "reservoir": reservoir_state,
            "activation": 0.0,
            "truth_value": self.truth_value,
            "confidence": self.confidence,
            "attention": self.attention
        }
        
        self.initialized = True
        
    def _step(self, state: State, x: Timestep) -> State:
        """Process one timestep through the atom's reservoir."""
        # Process through internal reservoir
        reservoir_out = self._reservoir.step(x)
        
        # Update activation based on reservoir dynamics
        activation = np.mean(np.abs(reservoir_out))
        
        # Create atom encoding/representation
        encoding = self._encode_semantics(reservoir_out, x)
        
        return {
            "out": encoding,
            "reservoir": reservoir_out,
            "activation": activation,
            "truth_value": state["truth_value"],
            "confidence": state["confidence"],
            "attention": state["attention"]
        }
        
    def _pad_or_truncate(self, data: np.ndarray, target_size: int) -> np.ndarray:
        """Helper method to pad or truncate array to target size."""
        if len(data) >= target_size:
            return data[:target_size]
        else:
            padded = np.zeros(target_size)
            padded[:len(data)] = data
            return padded
    
    def _encode_semantics(self, reservoir_out: np.ndarray, input_data: Timestep) -> np.ndarray:
        """Encode semantic content from reservoir dynamics."""
        # Default encoding: combine reservoir state with input
        if len(input_data) == 0:
            # Auto-regressive mode
            encoding = self._pad_or_truncate(reservoir_out, self.output_dim)
        else:
            # Combine input with reservoir dynamics
            combined = np.concatenate([input_data, reservoir_out])
            encoding = self._pad_or_truncate(combined, self.output_dim)
            
        return encoding
        
    def add_edge(self, other_atom: 'BaseAtomNode', edge_type: str = "link"):
        """Add edge to another atom in the AtomSpace graph."""
        self.outgoing_edges.append((other_atom, edge_type))
        other_atom.incoming_edges.append((self, edge_type))
        
    def get_neighbors(self) -> List['BaseAtomNode']:
        """Get all neighboring atoms."""
        neighbors = []
        for atom, _ in self.incoming_edges + self.outgoing_edges:
            neighbors.append(atom)
        return neighbors


class ConceptNode(BaseAtomNode):
    """OpenCog ConceptNode implemented with reservoir dynamics.
    
    Represents general concepts within the knowledge graph using
    reservoir computing for semantic processing.
    """
    
    def __init__(self, concept_name: str, **kwargs):
        """Initialize concept node.
        
        Parameters
        ---------- 
        concept_name : str
            Name of the concept
        """
        super().__init__(name=concept_name, **kwargs)
        self.concept_name = concept_name
        
    def _encode_semantics(self, reservoir_out: np.ndarray, input_data: Timestep) -> np.ndarray:
        """Encode concept semantics with enhanced abstraction."""
        # Concepts use reservoir dynamics for abstract representation
        activation_pattern = np.tanh(reservoir_out)
        
        # Apply concept-specific transformations
        concept_encoding = np.zeros(self.output_dim)
        
        if len(activation_pattern) >= self.output_dim:
            concept_encoding = activation_pattern[:self.output_dim]
        else:
            concept_encoding[:len(activation_pattern)] = activation_pattern
            
        # Add conceptual bias based on truth value and confidence
        concept_encoding *= self.truth_value * self.confidence
        
        return concept_encoding


class PredicateNode(BaseAtomNode):
    """OpenCog PredicateNode implemented with reservoir dynamics.
    
    Defines predicates or properties that evaluate conditions using
    reservoir-based pattern matching.
    """
    
    def __init__(self, predicate_name: str, arity: int = 2, **kwargs):
        """Initialize predicate node.
        
        Parameters
        ----------
        predicate_name : str
            Name of the predicate
        arity : int, default=2
            Number of arguments this predicate takes
        """
        super().__init__(name=predicate_name, **kwargs)
        self.predicate_name = predicate_name
        self.arity = arity
        
    def _encode_semantics(self, reservoir_out: np.ndarray, input_data: Timestep) -> np.ndarray:
        """Encode predicate evaluation through reservoir dynamics."""
        # Predicates use reservoir for pattern matching and evaluation
        evaluation_pattern = 1 / (1 + np.exp(-reservoir_out))  # sigmoid function
        
        # Create predicate encoding
        predicate_encoding = np.zeros(self.output_dim)
        
        if len(evaluation_pattern) >= self.output_dim:
            predicate_encoding = evaluation_pattern[:self.output_dim]
        else:
            predicate_encoding[:len(evaluation_pattern)] = evaluation_pattern
            
        # Modulate by confidence for uncertain reasoning
        predicate_encoding *= self.confidence
        
        return predicate_encoding
        
    def evaluate(self, *args) -> float:
        """Evaluate predicate with given arguments."""
        if len(args) != self.arity:
            raise ValueError(f"Predicate {self.predicate_name} expects {self.arity} arguments, got {len(args)}")
            
        # Encode arguments as input and get reservoir response - use deterministic encoding
        arg_encoding = np.concatenate([np.array([abs(hash(str(arg) + str(i))) % 1000 / 1000.0]) for i, arg in enumerate(args)])
        
        # Ensure proper input dimension
        if not self.initialized:
            # Initialize with padded argument encoding
            init_input = np.zeros(max(10, len(arg_encoding)))
            init_input[:len(arg_encoding)] = arg_encoding
            self.initialize(init_input)
            
        # Pad or truncate to match input dimension
        if len(arg_encoding) != self.input_dim:
            padded_input = np.zeros(self.input_dim)
            copy_len = min(len(arg_encoding), self.input_dim)
            padded_input[:copy_len] = arg_encoding[:copy_len]
            arg_encoding = padded_input
        
        # Process through reservoir
        result = self.step(arg_encoding)
        # Return evaluation as scalar
        return float(np.mean(result))


class NumberNode(BaseAtomNode):
    """OpenCog NumberNode implemented with reservoir dynamics.
    
    Stores and processes numerical values using reservoir computing
    for numerical reasoning and operations.
    """
    
    def __init__(self, value: float, **kwargs):
        """Initialize number node.
        
        Parameters
        ----------
        value : float
            Numerical value to store
        """
        super().__init__(name=f"Number_{value}", **kwargs)
        self.value = value
        
    def _encode_semantics(self, reservoir_out: np.ndarray, input_data: Timestep) -> np.ndarray:
        """Encode numerical value with reservoir dynamics."""
        # Numbers use reservoir for numerical processing and operations
        numerical_pattern = reservoir_out * self.value
        
        # Create number encoding
        number_encoding = np.zeros(self.output_dim) 
        
        if len(numerical_pattern) >= self.output_dim:
            number_encoding = numerical_pattern[:self.output_dim]
        else:
            number_encoding[:len(numerical_pattern)] = numerical_pattern
            
        # Add direct numerical representation
        if self.output_dim > 0:
            number_encoding[0] = self.value
            
        return number_encoding


class AtomSpace:
    """OpenCog AtomSpace implemented using reservoir computing.
    
    Represents the cognitive knowledge graph using distributed
    reservoir computing architecture. This is a specialized container
    for atoms rather than a Model subclass.
    """
    
    def __init__(self, name: str = "AtomSpace", **kwargs):
        """Initialize AtomSpace.
        
        Parameters
        ----------
        name : str, default="AtomSpace"
            Name of the AtomSpace
        """
        self.name = name
        
        # Atom storage and indexing
        self.atoms: Dict[str, BaseAtomNode] = {}
        self.atom_types: Dict[str, List[BaseAtomNode]] = defaultdict(list)
        
        # Pattern matching and attention
        self.attention_values: Dict[str, float] = {}
        self.pattern_cache = {}
        
    def add_atom(self, atom: BaseAtomNode) -> BaseAtomNode:
        """Add atom to the AtomSpace."""
        atom_key = f"{type(atom).__name__}:{atom.atom_name}"
        
        if atom_key not in self.atoms:
            self.atoms[atom_key] = atom
            self.atom_types[type(atom).__name__].append(atom)
            self.attention_values[atom_key] = atom.attention
            
        return self.atoms[atom_key]
        
    def get_atom(self, atom_name: str, atom_type: str = None) -> Optional[BaseAtomNode]:
        """Retrieve atom by name and optionally by type."""
        if atom_type:
            atom_key = f"{atom_type}:{atom_name}"
            return self.atoms.get(atom_key)
        else:
            # Search across all types
            for key, atom in self.atoms.items():
                if atom.atom_name == atom_name:
                    return atom
        return None
        
    def get_atoms_by_type(self, atom_type: str) -> List[BaseAtomNode]:
        """Get all atoms of a specific type."""
        return self.atom_types.get(atom_type, [])
        
    def add_link(self, atom1: BaseAtomNode, atom2: BaseAtomNode, link_type: str = "link"):
        """Add link between two atoms."""
        atom1.add_edge(atom2, link_type)
        
    def pattern_match(self, pattern: Dict[str, Any]) -> List[BaseAtomNode]:
        """Pattern matching in the AtomSpace using reservoir dynamics."""
        # Simple pattern matching implementation
        results = []
        
        atom_type = pattern.get('type')
        name_pattern = pattern.get('name')
        min_attention = pattern.get('min_attention', 0.0)
        
        candidates = self.get_atoms_by_type(atom_type) if atom_type else list(self.atoms.values())
        
        for atom in candidates:
            if name_pattern and name_pattern not in atom.atom_name:
                continue
            if atom.attention < min_attention:
                continue
                
            results.append(atom)
            
        return results
        
    def spread_activation(self, source_atoms: List[BaseAtomNode], iterations: int = 3):
        """Spread activation through the AtomSpace network."""
        for _ in range(iterations):
            new_attention = {}
            
            for atom_key, atom in self.atoms.items():
                current_attention = self.attention_values[atom_key]
                neighbors = atom.get_neighbors()
                
                # Spread attention to neighbors
                spread_amount = current_attention * 0.1
                for neighbor in neighbors:
                    neighbor_key = f"{type(neighbor).__name__}:{neighbor.atom_name}"
                    if neighbor_key not in new_attention:
                        new_attention[neighbor_key] = self.attention_values.get(neighbor_key, 0.0)
                    new_attention[neighbor_key] += spread_amount
                    
                # Decay current attention
                new_attention[atom_key] = current_attention * 0.95
                
            # Update attention values
            self.attention_values.update(new_attention)
            for atom_key, attention in new_attention.items():
                if atom_key in self.atoms:
                    self.atoms[atom_key].attention = attention