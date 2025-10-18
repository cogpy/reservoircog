# Licence: MIT License
# Copyright: ReservoirCog Contributors (2024)

"""
Cognitive agents for distributed echo state reasoning networks.

This module provides OpenCog-inspired cognitive agents implemented as
specialized reservoir computing nodes for distributed cognitive processing.
"""

from typing import Dict, List, Optional, Union, Any, Callable
import numpy as np

from ..node import Node
from ..nodes import Reservoir, Ridge, Input, Output
from ..model import Model
from ..type import NodeInput, State, Timestep, Timeseries
from .atoms import BaseAtomNode, AtomSpace


class CognitiveAgent(Node):
    """Echo state network agent for distributed cognitive reasoning.
    
    Each cognitive agent specializes in specific cognitive functions
    using reservoir computing dynamics and can communicate with other
    agents in a distributed network.
    """
    
    def __init__(
        self,
        agent_name: str,
        specialty: str = "general",
        units: int = 200,
        sr: float = 1.1,
        lr: float = 0.2,
        input_scaling: float = 1.0,
        **kwargs
    ):
        """Initialize cognitive agent.
        
        Parameters
        ----------
        agent_name : str
            Unique name for this agent
        specialty : str, default="general" 
            Cognitive specialty (e.g., "attention", "memory", "reasoning")
        units : int, default=200
            Number of reservoir units
        sr : float, default=1.1
            Spectral radius for edge-of-chaos dynamics
        lr : float, default=0.2
            Leak rate for temporal processing
        input_scaling : float, default=1.0
            Input scaling factor
        """
        self.agent_name = agent_name
        self.specialty = specialty
        self.units = units
        self.sr = sr
        self.lr = lr
        self.input_scaling = input_scaling
        
        # Agent communication and memory
        self.message_buffer = []
        self.memory_trace = []
        self.connected_agents = []
        self.atomspace_ref = None
        
        # Internal reservoir for cognitive processing
        self._reservoir = None
        self._readout = None
        self._memory_size = kwargs.get('memory_size', 100)
        
        super().__init__()
        
    def initialize(self, x: Union[NodeInput, Timestep], y: Optional[Union[NodeInput, Timestep]] = None):
        """Initialize the cognitive agent's neural architecture."""
        if isinstance(x, np.ndarray):
            if len(x.shape) == 1:
                input_dim = len(x)
            else:
                input_dim = x.shape[-1]
        else:
            input_dim = 64  # Default encoding dimension
            
        self.input_dim = input_dim
        self.output_dim = input_dim  # Maintain dimensionality for communication
        
        # Initialize cognitive reservoir with edge-of-chaos dynamics
        self._reservoir = Reservoir(
            units=self.units,
            sr=self.sr,
            lr=self.lr,
            input_scaling=self.input_scaling,
            activation='tanh'
        )
        
        # Readout for decision making and communication
        self._readout = Ridge(ridge=1e-5)
        
        self._reservoir.initialize(x)
        
        # Initialize agent state
        self.state = {
            "out": np.zeros(self.output_dim),
            "reservoir": np.zeros(self.units),
            "activation_level": 0.0,
            "attention_focus": np.zeros(self.output_dim),
            "memory_state": np.zeros(self._memory_size),
            "communication_intent": 0.0
        }
        
        self.initialized = True
        
    def _step(self, state: State, x: Timestep) -> State:
        """Process one cognitive step."""
        # Process input through cognitive reservoir
        reservoir_out = self._reservoir.step(x)
        
        # Update activation level based on reservoir dynamics
        activation = np.std(reservoir_out)  # Use variability as activation measure
        
        # Update memory trace with exponential decay
        memory_update = np.zeros(self._memory_size)
        if len(reservoir_out) >= self._memory_size:
            memory_update = reservoir_out[:self._memory_size]
        else:
            memory_update[:len(reservoir_out)] = reservoir_out
            
        new_memory = 0.9 * state["memory_state"] + 0.1 * memory_update
        
        # Compute attention focus based on specialty
        attention_focus = self._compute_attention(reservoir_out, x)
        
        # Generate output for communication with other agents
        output = self._generate_output(reservoir_out, attention_focus, x)
        
        # Determine communication intent
        comm_intent = np.mean(np.abs(reservoir_out)) > 0.5
        
        return {
            "out": output,
            "reservoir": reservoir_out,
            "activation_level": activation,
            "attention_focus": attention_focus,
            "memory_state": new_memory,
            "communication_intent": float(comm_intent)
        }
        
    def _compute_attention(self, reservoir_out: np.ndarray, input_data: Timestep) -> np.ndarray:
        """Compute attention focus based on agent specialty."""
        attention = np.zeros(self.output_dim)
        
        if self.specialty == "attention":
            # Attention specialists focus on salient features
            attention = np.abs(reservoir_out[:self.output_dim] 
                             if len(reservoir_out) >= self.output_dim 
                             else np.pad(reservoir_out, (0, self.output_dim - len(reservoir_out))))
        elif self.specialty == "memory":
            # Memory specialists maintain stable patterns
            attention = np.tanh(reservoir_out[:self.output_dim]
                              if len(reservoir_out) >= self.output_dim
                              else np.pad(reservoir_out, (0, self.output_dim - len(reservoir_out))))
        elif self.specialty == "reasoning":
            # Reasoning specialists use complex dynamics
            attention = np.sin(reservoir_out[:self.output_dim]
                             if len(reservoir_out) >= self.output_dim
                             else np.pad(reservoir_out, (0, self.output_dim - len(reservoir_out))))
        else:
            # General agents use reservoir output directly
            attention = reservoir_out[:self.output_dim] if len(reservoir_out) >= self.output_dim else np.pad(reservoir_out, (0, self.output_dim - len(reservoir_out)))
            
        return attention
        
    def _generate_output(self, reservoir_out: np.ndarray, attention: np.ndarray, input_data: Timestep) -> np.ndarray:
        """Generate output for inter-agent communication."""
        # Combine reservoir dynamics with attention and input
        output = 0.5 * attention + 0.3 * (reservoir_out[:self.output_dim] 
                                          if len(reservoir_out) >= self.output_dim 
                                          else np.pad(reservoir_out, (0, self.output_dim - len(reservoir_out))))
        
        if len(input_data) > 0:
            input_contrib = input_data[:self.output_dim] if len(input_data) >= self.output_dim else np.pad(input_data, (0, self.output_dim - len(input_data)))
            output += 0.2 * input_contrib
            
        return output
        
    def connect_agent(self, other_agent: 'CognitiveAgent'):
        """Connect to another cognitive agent for communication."""
        if other_agent not in self.connected_agents:
            self.connected_agents.append(other_agent)
            other_agent.connected_agents.append(self)
            
    def send_message(self, message: np.ndarray, target_agent: 'CognitiveAgent' = None):
        """Send message to connected agents."""
        if target_agent:
            target_agent.receive_message(message, self)
        else:
            # Broadcast to all connected agents
            for agent in self.connected_agents:
                agent.receive_message(message, self)
                
    def receive_message(self, message: np.ndarray, sender: 'CognitiveAgent'):
        """Receive message from another agent."""
        self.message_buffer.append({
            'message': message,
            'sender': sender,
            'timestamp': len(self.memory_trace)
        })
        
    def set_atomspace(self, atomspace: AtomSpace):
        """Connect agent to an AtomSpace for knowledge access."""
        self.atomspace_ref = atomspace


class AttentionAgent(CognitiveAgent):
    """Specialized agent for attention allocation and focus management.
    
    Implements OpenCog-style attention allocation using reservoir dynamics
    to manage cognitive resources across the AtomSpace.
    """
    
    def __init__(self, **kwargs):
        """Initialize attention allocation agent."""
        kwargs['specialty'] = 'attention'
        kwargs['sr'] = kwargs.get('sr', 0.95)  # More stable dynamics for attention
        super().__init__(agent_name="AttentionAgent", **kwargs)
        
        self.attention_map = {}
        self.focus_threshold = 0.7
        
    def _compute_attention(self, reservoir_out: np.ndarray, input_data: Timestep) -> np.ndarray:
        """Compute attention allocation across cognitive space."""
        # Use reservoir dynamics to determine attention allocation
        attention_strengths = 1 / (1 + np.exp(-reservoir_out))  # sigmoid function
        
        # Apply winner-take-all competition
        max_indices = np.argsort(attention_strengths)[-int(0.1 * len(attention_strengths)):]
        competitive_attention = np.zeros_like(attention_strengths)
        competitive_attention[max_indices] = attention_strengths[max_indices]
        
        # Normalize and return focus vector
        if np.sum(competitive_attention) > 0:
            competitive_attention /= np.sum(competitive_attention)
            
        return competitive_attention[:self.output_dim] if len(competitive_attention) >= self.output_dim else np.pad(competitive_attention, (0, self.output_dim - len(competitive_attention)))
        
    def allocate_attention(self, atomspace: AtomSpace, focus_atoms: List[BaseAtomNode] = None):
        """Allocate attention across atoms in the AtomSpace."""
        if not focus_atoms:
            focus_atoms = list(atomspace.atoms.values())
            
        # Process each atom through attention mechanism
        for atom in focus_atoms:
            default_dim = self.input_dim if self.input_dim is not None else 64
            atom_encoding = atom.state.get("out", np.zeros(default_dim))
            
            # Initialize agent if not done yet
            if not self.initialized:
                self.initialize(atom_encoding)
                
            # Ensure dimensional compatibility
            if len(atom_encoding) != self.input_dim:
                # Truncate or pad as needed
                if len(atom_encoding) > self.input_dim:
                    atom_encoding = atom_encoding[:self.input_dim]
                else:
                    padded = np.zeros(self.input_dim)
                    padded[:len(atom_encoding)] = atom_encoding
                    atom_encoding = padded
            attention_response = self.step(atom_encoding)
            
            # Update atom attention based on response
            attention_strength = np.mean(attention_response)
            atom.attention = max(0.0, min(1.0, attention_strength))
            
            # Update attention map
            atom_key = f"{type(atom).__name__}:{atom.atom_name}"
            self.attention_map[atom_key] = attention_strength
            
    def get_focus_atoms(self, atomspace: AtomSpace) -> List[BaseAtomNode]:
        """Get atoms that are currently in focus."""
        focus_atoms = []
        for atom_key, atom in atomspace.atoms.items():
            if atom.attention > self.focus_threshold:
                focus_atoms.append(atom)
        return focus_atoms


class PatternMatchAgent(CognitiveAgent):
    """Specialized agent for pattern matching and recognition.
    
    Uses reservoir computing for dynamic pattern matching across
    the AtomSpace, similar to OpenCog's pattern matcher.
    """
    
    def __init__(self, **kwargs):
        """Initialize pattern matching agent."""
        kwargs['specialty'] = 'reasoning'
        kwargs['sr'] = kwargs.get('sr', 1.2)  # Higher SR for complex pattern dynamics
        super().__init__(agent_name="PatternMatchAgent", **kwargs)
        
        self.pattern_templates = {}
        self.match_threshold = 0.8
        
    def _compute_attention(self, reservoir_out: np.ndarray, input_data: Timestep) -> np.ndarray:
        """Compute pattern matching attention."""
        # Use complex reservoir dynamics for pattern recognition
        pattern_response = np.tanh(reservoir_out) * np.cos(reservoir_out * 0.5)
        
        # Apply lateral inhibition for pattern selection
        inhibited = pattern_response - 0.1 * np.mean(pattern_response)
        
        return np.maximum(0, inhibited)[:self.output_dim] if len(inhibited) >= self.output_dim else np.pad(np.maximum(0, inhibited), (0, max(0, self.output_dim - len(inhibited))))
        
    def add_pattern_template(self, name: str, template: Dict[str, Any]):
        """Add a pattern template for matching."""
        self.pattern_templates[name] = template
        
    def match_pattern(self, atomspace: AtomSpace, pattern_name: str) -> List[BaseAtomNode]:
        """Match pattern against AtomSpace."""
        if pattern_name not in self.pattern_templates:
            return []
            
        template = self.pattern_templates[pattern_name]
        matches = []
        
        # Use reservoir dynamics to evaluate pattern matches
        for atom in atomspace.atoms.values():
            default_dim = self.input_dim if self.input_dim is not None else 64
            atom_encoding = atom.state.get("out", np.zeros(default_dim))
            
            # Initialize agent if not done yet
            if not self.initialized:
                self.initialize(atom_encoding)
                
            # Ensure dimensional compatibility
            if len(atom_encoding) != self.input_dim:
                if len(atom_encoding) > self.input_dim:
                    atom_encoding = atom_encoding[:self.input_dim]
                else:
                    padded = np.zeros(self.input_dim)
                    padded[:len(atom_encoding)] = atom_encoding
                    atom_encoding = padded
            match_response = self.step(atom_encoding)
            
            # Compute match strength
            match_strength = np.mean(match_response)
            
            if match_strength > self.match_threshold:
                matches.append(atom)
                
        return matches
        
    def find_similar_atoms(self, atomspace: AtomSpace, query_atom: BaseAtomNode, top_k: int = 5) -> List[BaseAtomNode]:
        """Find atoms similar to the query atom."""
        default_dim = self.input_dim if self.input_dim is not None else 64
        query_encoding = query_atom.state.get("out", np.zeros(default_dim))
        
        # Initialize agent if not done yet
        if not self.initialized:
            self.initialize(query_encoding)
        similarities = []
        
        for atom in atomspace.atoms.values():
            if atom == query_atom:
                continue
                
            atom_encoding = atom.state.get("out", np.zeros(self.input_dim))
            
            # Ensure dimensional compatibility for both encodings
            if len(query_encoding) > self.input_dim:
                query_enc = query_encoding[:self.input_dim]
            else:
                query_enc = np.zeros(self.input_dim)
                query_enc[:len(query_encoding)] = query_encoding
                
            if len(atom_encoding) > self.input_dim:
                atom_enc = atom_encoding[:self.input_dim]
            else:
                atom_enc = np.zeros(self.input_dim)  
                atom_enc[:len(atom_encoding)] = atom_encoding
            
            # Use reservoir to compute similarity - simple approach with first half of input
            half_dim = self.input_dim // 2
            combined_input = np.zeros(self.input_dim)
            combined_input[:half_dim] = query_enc[:half_dim]
            combined_input[half_dim:half_dim*2] = atom_enc[:half_dim]
            similarity_response = self.step(combined_input)
            similarity_score = np.mean(similarity_response)
            
            similarities.append((atom, similarity_score))
            
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [atom for atom, _ in similarities[:top_k]]