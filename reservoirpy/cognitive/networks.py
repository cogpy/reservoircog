# Licence: MIT License
# Copyright: ReservoirCog Contributors (2024)

"""
Distributed echo state networks for cognitive orchestration.

This module provides the main orchestrator and distributed network
architectures for OpenCog-inspired cognitive processing using
reservoir computing.
"""

from typing import Dict, List, Optional, Union, Any, Callable
import numpy as np
from collections import defaultdict
import time

from ..model import Model
from ..type import NodeInput, Timestep, Timeseries
from .atoms import AtomSpace, BaseAtomNode, ConceptNode, PredicateNode, NumberNode
from .agents import CognitiveAgent, AttentionAgent, PatternMatchAgent


class DistributedEchoNetwork:
    """Network of interconnected echo state agents for distributed cognitive processing.
    
    Implements a distributed cognitive architecture where multiple specialized
    agents collaborate using reservoir computing dynamics.
    """
    
    def __init__(
        self,
        name: str = "DistributedEchoNetwork",
        num_agents: int = 5,
        agent_types: List[str] = None,
        **kwargs
    ):
        """Initialize distributed echo state network.
        
        Parameters
        ----------
        name : str, default="DistributedEchoNetwork"
            Name of the network
        num_agents : int, default=5
            Number of cognitive agents
        agent_types : List[str], optional
            Types of agents to create. If None, uses default mix.
        """
        self.name = name
        
        if agent_types is None:
            agent_types = ['general', 'attention', 'memory', 'reasoning', 'communication']
            
        # Create cognitive agents
        self.agents: Dict[str, CognitiveAgent] = {}
        self.agent_connections: Dict[str, List[str]] = defaultdict(list)
        
        # Special agents
        self.attention_agent: Optional[AttentionAgent] = None
        self.pattern_agent: Optional[PatternMatchAgent] = None
        
        # Network dynamics
        self.communication_matrix = None
        self.synchronization_rate = 0.1
        self.global_activity = 0.0
        
        # Initialize agents
        self._create_agents(num_agents, agent_types)
        self._setup_connections()
        
    def _create_agents(self, num_agents: int, agent_types: List[str]):
        """Create and initialize cognitive agents."""
        # Ensure we have specialized agents
        if 'attention' in agent_types:
            self.attention_agent = AttentionAgent()
            self.agents['attention_agent'] = self.attention_agent
            
        if 'reasoning' in agent_types:
            self.pattern_agent = PatternMatchAgent()
            self.agents['pattern_agent'] = self.pattern_agent
            
        # Create remaining agents
        agent_count = 0
        for i in range(num_agents):
            agent_type = agent_types[i % len(agent_types)]
            
            if agent_type == 'attention' and self.attention_agent:
                continue
            if agent_type == 'reasoning' and self.pattern_agent:
                continue
                
            agent_name = f"{agent_type}_agent_{agent_count}"
            agent = CognitiveAgent(
                agent_name=agent_name,
                specialty=agent_type,
                units=150 + np.random.randint(-50, 50),  # Varied reservoir sizes
                sr=0.9 + np.random.normal(0, 0.1),  # Varied spectral radius
                lr=0.3 + np.random.normal(0, 0.05)  # Varied leak rates
            )
            
            self.agents[agent_name] = agent
            agent_count += 1
            
    def _setup_connections(self):
        """Setup connections between agents."""
        agent_names = list(self.agents.keys())
        
        # Create small-world network topology
        for i, agent_name in enumerate(agent_names):
            agent = self.agents[agent_name]
            
            # Connect to nearest neighbors
            for j in range(max(1, len(agent_names) // 3)):
                neighbor_idx = (i + j + 1) % len(agent_names)
                neighbor_name = agent_names[neighbor_idx]
                neighbor = self.agents[neighbor_name]
                
                agent.connect_agent(neighbor)
                self.agent_connections[agent_name].append(neighbor_name)
                
            # Add random long-range connections
            if np.random.random() < 0.3:  # 30% chance of long-range connection
                random_idx = np.random.randint(0, len(agent_names))
                if random_idx != i:
                    random_agent = self.agents[agent_names[random_idx]]
                    agent.connect_agent(random_agent)
                    self.agent_connections[agent_name].append(agent_names[random_idx])
                    
        # Create communication matrix
        n_agents = len(self.agents)
        self.communication_matrix = np.random.random((n_agents, n_agents)) * 0.1
        np.fill_diagonal(self.communication_matrix, 0.0)
        
    def process_distributed(self, input_data: np.ndarray, steps: int = 10) -> Dict[str, np.ndarray]:
        """Process data through distributed agent network."""
        results = {}
        agent_states = {}
        
        # Initialize all agents
        for agent_name, agent in self.agents.items():
            if not agent.initialized:
                agent.initialize(input_data)
            agent_states[agent_name] = []
            
        # Distributed processing steps
        for step in range(steps):
            step_outputs = {}
            
            # Process each agent
            for agent_name, agent in self.agents.items():
                # Collect messages from connected agents
                agent_input = input_data.copy()
                
                # Add influence from connected agents
                for connected_name in self.agent_connections[agent_name]:
                    if connected_name in step_outputs:
                        agent_input += 0.1 * step_outputs[connected_name]
                        
                # Process through agent
                output = agent.step(agent_input)
                step_outputs[agent_name] = output
                agent_states[agent_name].append(output.copy())
                
            # Synchronization step
            if step % int(1.0 / self.synchronization_rate) == 0:
                self._synchronize_agents(step_outputs)
                
        # Collect final results
        for agent_name, states in agent_states.items():
            results[agent_name] = np.array(states)
            
        return results
        
    def _synchronize_agents(self, outputs: Dict[str, np.ndarray]):
        """Synchronize agents using global coupling."""
        # Compute global activity
        all_outputs = np.concatenate(list(outputs.values()))
        self.global_activity = np.mean(np.abs(all_outputs))
        
        # Send synchronization signals
        sync_signal = np.array([self.global_activity])
        
        for agent_name, agent in self.agents.items():
            # Modulate agent state based on global activity
            if hasattr(agent, 'state') and agent.state:
                current_activation = agent.state.get('activation_level', 0.0)
                # Weak coupling to global dynamics
                adjusted_activation = 0.95 * current_activation + 0.05 * self.global_activity
                agent.state['activation_level'] = adjusted_activation
                
    def get_network_state(self) -> Dict[str, Any]:
        """Get current state of the distributed network."""
        state = {
            'global_activity': self.global_activity,
            'num_agents': len(self.agents),
            'agent_activations': {},
            'connections': dict(self.agent_connections)
        }
        
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'state') and agent.state:
                state['agent_activations'][agent_name] = agent.state.get('activation_level', 0.0)
                
        return state


class CognitiveOrchestrator:
    """Main orchestrator for OpenCog-inspired cognitive processing.
    
    Coordinates AtomSpace operations, agent networks, and cognitive
    processes using reservoir computing architectures.
    """
    
    def __init__(
        self,
        atomspace_name: str = "MainAtomSpace",
        network_size: int = 8,
        **kwargs
    ):
        """Initialize cognitive orchestrator.
        
        Parameters
        ----------
        atomspace_name : str, default="MainAtomSpace"
            Name of the main AtomSpace
        network_size : int, default=8
            Size of the distributed agent network
        """
        # Core components
        self.atomspace = AtomSpace(name=atomspace_name)
        self.agent_network = DistributedEchoNetwork(
            name="CognitiveNetwork",
            num_agents=network_size,
            **kwargs
        )
        
        # Connect agents to AtomSpace
        for agent in self.agent_network.agents.values():
            agent.set_atomspace(self.atomspace)
            
        # Cognitive processes
        self.attention_cycles = 0
        self.reasoning_depth = 3
        self.pattern_cache = {}
        
        # Performance monitoring
        self.processing_history = []
        self.attention_history = []
        
    def add_concept(self, concept_name: str, **kwargs) -> ConceptNode:
        """Add a concept to the AtomSpace."""
        concept = ConceptNode(concept_name, **kwargs)
        return self.atomspace.add_atom(concept)
        
    def add_predicate(self, predicate_name: str, arity: int = 2, **kwargs) -> PredicateNode:
        """Add a predicate to the AtomSpace."""
        predicate = PredicateNode(predicate_name, arity=arity, **kwargs)
        return self.atomspace.add_atom(predicate)
        
    def add_number(self, value: float, **kwargs) -> NumberNode:
        """Add a number to the AtomSpace."""
        number = NumberNode(value, **kwargs)
        return self.atomspace.add_atom(number)
        
    def create_link(self, atom1: BaseAtomNode, atom2: BaseAtomNode, link_type: str = "similarity"):
        """Create a link between two atoms."""
        self.atomspace.add_link(atom1, atom2, link_type)
        
    def cognitive_cycle(self, input_stimulus: Optional[np.ndarray] = None, cycles: int = 1) -> Dict[str, Any]:
        """Execute cognitive processing cycles.
        
        Parameters
        ----------
        input_stimulus : np.ndarray, optional
            External input stimulus
        cycles : int, default=1
            Number of cognitive cycles to execute
            
        Returns
        -------
        Dict[str, Any]
            Results of cognitive processing
        """
        cycle_results = []
        
        for cycle in range(cycles):
            cycle_start = time.time()
            
            # 1. Perception phase - process input stimulus
            if input_stimulus is not None:
                perception_results = self._perception_phase(input_stimulus)
            else:
                perception_results = {}
                
            # 2. Attention allocation phase
            attention_results = self._attention_phase()
            
            # 3. Pattern matching and recognition phase
            pattern_results = self._pattern_matching_phase()
            
            # 4. Reasoning phase
            reasoning_results = self._reasoning_phase()
            
            # 5. Action selection phase
            action_results = self._action_selection_phase()
            
            cycle_time = time.time() - cycle_start
            
            cycle_result = {
                'cycle': cycle,
                'perception': perception_results,
                'attention': attention_results,
                'patterns': pattern_results,
                'reasoning': reasoning_results,
                'actions': action_results,
                'cycle_time': cycle_time,
                'atomspace_size': len(self.atomspace.atoms),
                'network_state': self.agent_network.get_network_state()
            }
            
            cycle_results.append(cycle_result)
            self.processing_history.append(cycle_result)
            
        return {
            'cycles': cycle_results,
            'total_cycles': len(cycle_results),
            'average_cycle_time': np.mean([r['cycle_time'] for r in cycle_results])
        }
        
    def _perception_phase(self, input_stimulus: np.ndarray) -> Dict[str, Any]:
        """Process perception through distributed agents."""
        # Distribute input to agent network
        agent_responses = self.agent_network.process_distributed(input_stimulus, steps=3)
        
        # Create or update concept nodes based on perception
        perception_concepts = []
        
        for agent_name, response in agent_responses.items():
            # Extract salient features from agent responses
            final_response = response[-1] if len(response) > 0 else np.zeros(64)
            salience = np.mean(np.abs(final_response))
            
            if salience > 0.5:  # Threshold for concept creation
                concept_name = f"percept_{agent_name}_{len(perception_concepts)}"
                concept = self.add_concept(concept_name, truth_value=salience)
                
                # Initialize concept with agent response
                if not concept.initialized:
                    concept.initialize(final_response)
                else:
                    concept.step(final_response)
                    
                perception_concepts.append(concept)
                
        return {
            'agent_responses': {k: v.shape for k, v in agent_responses.items()},
            'concepts_created': len(perception_concepts),
            'average_salience': np.mean([c.truth_value for c in perception_concepts]) if perception_concepts else 0.0
        }
        
    def _attention_phase(self) -> Dict[str, Any]:
        """Allocate attention across the AtomSpace."""
        if self.agent_network.attention_agent:
            # Use specialized attention agent
            self.agent_network.attention_agent.allocate_attention(self.atomspace)
            
            # Get atoms in focus
            focus_atoms = self.agent_network.attention_agent.get_focus_atoms(self.atomspace)
            
        else:
            # Fallback attention mechanism
            self.atomspace.spread_activation(list(self.atomspace.atoms.values())[:5])
            focus_atoms = [atom for atom in self.atomspace.atoms.values() if atom.attention > 0.5]
            
        self.attention_cycles += 1
        
        attention_result = {
            'focus_atoms': len(focus_atoms),
            'total_atoms': len(self.atomspace.atoms),
            'attention_cycles': self.attention_cycles,
            'average_attention': np.mean([atom.attention for atom in self.atomspace.atoms.values()])
        }
        
        self.attention_history.append(attention_result)
        return attention_result
        
    def _pattern_matching_phase(self) -> Dict[str, Any]:
        """Perform pattern matching across the AtomSpace."""
        patterns_found = []
        
        if self.agent_network.pattern_agent:
            # Use specialized pattern matching agent
            for atom in self.atomspace.atoms.values():
                if atom.attention > 0.7:  # Only check high-attention atoms
                    similar_atoms = self.agent_network.pattern_agent.find_similar_atoms(
                        self.atomspace, atom, top_k=3
                    )
                    if similar_atoms:
                        patterns_found.append({
                            'query_atom': atom.atom_name,
                            'similar_atoms': [a.atom_name for a in similar_atoms],
                            'pattern_type': 'similarity'
                        })
                        
        return {
            'patterns_found': len(patterns_found),
            'pattern_details': patterns_found[:5]  # Limit output size
        }
        
    def _reasoning_phase(self) -> Dict[str, Any]:
        """Perform reasoning over focused atoms."""
        reasoning_results = []
        
        # Get high-attention atoms for reasoning
        focus_atoms = [atom for atom in self.atomspace.atoms.values() if atom.attention > 0.6]
        
        # Simple inference: activate connected atoms
        for atom in focus_atoms[:self.reasoning_depth]:
            neighbors = atom.get_neighbors()
            for neighbor in neighbors:
                # Boost attention of connected atoms
                neighbor.attention = min(1.0, neighbor.attention + 0.1)
                
            reasoning_results.append({
                'atom': atom.atom_name,
                'neighbors_activated': len(neighbors),
                'truth_value': atom.truth_value
            })
            
        return {
            'atoms_processed': len(reasoning_results),
            'total_activations': sum(r['neighbors_activated'] for r in reasoning_results),
            'reasoning_details': reasoning_results[:3]
        }
        
    def _action_selection_phase(self) -> Dict[str, Any]:
        """Select actions based on cognitive state."""
        # Simple action selection based on network activity
        network_state = self.agent_network.get_network_state()
        global_activity = network_state['global_activity']
        
        selected_actions = []
        
        if global_activity > 0.7:
            selected_actions.append('explore')
        elif global_activity > 0.4:
            selected_actions.append('consolidate')
        else:
            selected_actions.append('rest')
            
        return {
            'selected_actions': selected_actions,
            'global_activity': global_activity,
            'action_confidence': min(1.0, global_activity)
        }
        
    def get_cognitive_state(self) -> Dict[str, Any]:
        """Get comprehensive cognitive state."""
        return {
            'atomspace': {
                'total_atoms': len(self.atomspace.atoms),
                'atom_types': {t: len(atoms) for t, atoms in self.atomspace.atom_types.items()},
                'average_attention': np.mean([a.attention for a in self.atomspace.atoms.values()]) if self.atomspace.atoms else 0.0
            },
            'agent_network': self.agent_network.get_network_state(),
            'processing_stats': {
                'total_cycles': len(self.processing_history),
                'attention_cycles': self.attention_cycles
            }
        }
        
    def save_state(self, filepath: str):
        """Save cognitive state (placeholder for future implementation)."""
        # This would serialize the AtomSpace and agent states
        pass
        
    def load_state(self, filepath: str):
        """Load cognitive state (placeholder for future implementation)."""
        # This would deserialize the AtomSpace and agent states  
        pass