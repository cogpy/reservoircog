# ReservoirCog: OpenCog-Inspired Cognitive Orchestrator

ReservoirCog implements an OpenCog-inspired cognitive architecture using ReservoirPy's reservoir computing framework. This provides a novel approach to artificial general intelligence (AGI) by combining symbolic cognitive processing with dynamic neural reservoir networks.

## Overview

ReservoirCog maps core OpenCog concepts to reservoir computing architectures:

- **AtomSpace** → Distributed knowledge representation using reservoir dynamics
- **Atoms (ConceptNode, PredicateNode, NumberNode)** → Specialized reservoir nodes with semantic processing
- **Cognitive Agents** → Echo state networks specialized for different cognitive functions
- **Distributed Processing** → Network of interconnected cognitive agents
- **Attention Allocation** → Dynamic resource management through reservoir-based attention mechanisms

## Architecture

### Core Components

#### 1. OpenCog-Inspired Atoms (`reservoirpy.cognitive.atoms`)

**ConceptNode**: Represents concepts using reservoir dynamics for semantic processing
```python
concept = ConceptNode("Alice", attention=0.8, truth_value=0.9)
concept.initialize(input_encoding)
semantic_output = concept.step(stimulus)
```

**PredicateNode**: Evaluates relationships and properties through reservoir networks
```python
likes = PredicateNode("likes", arity=2, confidence=0.9)
evaluation_score = likes.evaluate("Alice", "music")  # Returns 0.0-1.0
```

**NumberNode**: Processes numerical values with reservoir-based numerical reasoning
```python
number = NumberNode(42.0, attention=0.5)
numerical_output = number.step(mathematical_stimulus)
```

**AtomSpace**: Container and manager for all atoms in the knowledge base
```python
atomspace = AtomSpace("KnowledgeBase")
atomspace.add_atom(concept)
atomspace.add_link(atom1, atom2, "similarity")
matches = atomspace.pattern_match({'type': 'ConceptNode', 'min_attention': 0.7})
```

#### 2. Cognitive Agents (`reservoirpy.cognitive.agents`)

**CognitiveAgent**: Base echo state network agent for distributed reasoning
- Specialties: general, memory, reasoning, attention, communication
- Inter-agent communication and message passing
- Attention-based processing with reservoir dynamics

**AttentionAgent**: Specialized agent for attention allocation across the AtomSpace
- Winner-take-all competition for attention resources  
- Attention spreading through knowledge graph connections
- Dynamic focus management

**PatternMatchAgent**: Reservoir-based pattern matching and similarity detection
- Template-based pattern matching
- Similarity computation through reservoir dynamics
- Complex pattern recognition across atom networks

#### 3. Distributed Networks (`reservoirpy.cognitive.networks`)

**DistributedEchoNetwork**: Network of interconnected cognitive agents
- Small-world network topology for efficient communication
- Synchronization mechanisms for coherent processing
- Distributed computation across multiple agents

**CognitiveOrchestrator**: Main orchestrator coordinating all cognitive processes
- Multi-phase cognitive cycles (perception → attention → pattern matching → reasoning → action)
- Integration of AtomSpace with agent networks
- High-level API for cognitive operations

## Key Features

### 1. Reservoir Computing Foundation
- Uses ReservoirPy's proven reservoir computing architecture
- Dynamic neural processing with temporal memory
- Edge-of-chaos dynamics for complex computation

### 2. OpenCog-Inspired Semantics
- Familiar AtomSpace concepts and operations
- Truth values, confidence, and attention mechanisms  
- Symbolic reasoning with sub-symbolic implementation

### 3. Distributed Cognitive Processing
- Multiple specialized agents working in parallel
- Inter-agent communication and coordination
- Scalable architecture for complex reasoning

### 4. Attention Allocation
- Dynamic attention spreading through knowledge graphs
- Resource allocation based on relevance and importance
- Focus management for efficient processing

### 5. Pattern Matching
- Template-based pattern recognition
- Similarity detection through reservoir dynamics
- Complex structural pattern matching

## Usage Examples

### Basic Usage

```python
from reservoirpy.cognitive import CognitiveOrchestrator
import numpy as np

# Create cognitive orchestrator
orchestrator = CognitiveOrchestrator(network_size=5)

# Add knowledge to AtomSpace
alice = orchestrator.add_concept("Alice", attention=0.8)
bob = orchestrator.add_concept("Bob", attention=0.7) 
likes = orchestrator.add_predicate("likes", arity=2)

# Create relationships
orchestrator.create_link(alice, bob, "friends")
orchestrator.create_link(alice, likes, "uses_predicate")

# Initialize atoms
input_data = np.random.randn(10)
alice.initialize(input_data)
bob.initialize(input_data)
likes.initialize(input_data)

# Run cognitive processing
stimulus = np.array([0.8, 0.7, 0.9, 0.6, 0.8, 0.5, 0.7, 0.9, 0.4, 0.8])
results = orchestrator.cognitive_cycle(stimulus, cycles=3)

print(f"Completed {results['total_cycles']} cognitive cycles")
print(f"Average cycle time: {results['average_cycle_time']:.4f}s")
```

### Advanced Pattern Matching

```python
# Get pattern matching agent
pattern_agent = orchestrator.agent_network.pattern_agent

# Add pattern templates
pattern_agent.add_pattern_template("social_concepts", {
    'type': 'ConceptNode',
    'domain': 'social',
    'attention_threshold': 0.6
})

# Find patterns
matches = pattern_agent.match_pattern(orchestrator.atomspace, "social_concepts")
similar_to_alice = pattern_agent.find_similar_atoms(orchestrator.atomspace, alice, top_k=5)
```

### Predicate Evaluation

```python
# Create and use predicates
friendship = orchestrator.add_predicate("friendship", arity=2, confidence=0.9)
friendship.initialize(np.random.randn(10))

# Evaluate relationships
score1 = friendship.evaluate("Alice", "Bob")    # Returns similarity score
score2 = friendship.evaluate("Bob", "Charlie")  # Returns similarity score

print(f"Alice-Bob friendship: {score1:.3f}")
print(f"Bob-Charlie friendship: {score2:.3f}")
```

### Agent Communication

```python
# Access agent network
agents = list(orchestrator.agent_network.agents.values())
agent1, agent2 = agents[0], agents[1]

# Send messages between agents
message = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
agent1.send_message(message, agent2)

# Check received messages
print(f"Agent2 has {len(agent2.message_buffer)} messages")
if agent2.message_buffer:
    last_msg = agent2.message_buffer[-1]
    print(f"Last message from: {last_msg['sender'].agent_name}")
```

## Cognitive Processing Phases

The CognitiveOrchestrator implements a multi-phase cognitive cycle:

1. **Perception Phase**: Process external stimuli through distributed agent network
2. **Attention Phase**: Allocate attention resources across atoms
3. **Pattern Matching Phase**: Identify patterns and similarities
4. **Reasoning Phase**: Perform inference and knowledge manipulation  
5. **Action Selection Phase**: Choose actions based on cognitive state

Each phase leverages reservoir computing for dynamic, adaptive processing.

## Installation and Setup

The cognitive module is part of the reservoirpy package. After installing the base package:

```python
# Import cognitive components
from reservoirpy.cognitive import (
    CognitiveOrchestrator,
    ConceptNode, PredicateNode, NumberNode,
    AtomSpace,
    CognitiveAgent, AttentionAgent, PatternMatchAgent,
    DistributedEchoNetwork
)

# Create and use cognitive systems
orchestrator = CognitiveOrchestrator()
```

## Testing

Run the test suite to verify functionality:

```bash
python -m pytest reservoirpy/cognitive/tests/ -v
```

Run the demonstration example:

```bash
python examples/cognitive_orchestrator_demo.py
```

## Research Applications

ReservoirCog enables research in:

- **Artificial General Intelligence (AGI)**: Combining symbolic and connectionist approaches
- **Cognitive Architectures**: Biologically-inspired cognitive processing
- **Knowledge Representation**: Dynamic semantic networks with temporal processing
- **Distributed AI**: Multi-agent cognitive systems
- **Attention Mechanisms**: Resource allocation in cognitive systems
- **Pattern Recognition**: Complex structural pattern matching
- **Temporal Reasoning**: Processing sequences and temporal relationships

## Comparison with OpenCog

| OpenCog Feature | ReservoirCog Implementation |
|----------------|---------------------------|
| AtomSpace | Distributed reservoir-based knowledge graph |
| ConceptNode | Reservoir with semantic encoding |
| PredicateNode | Reservoir-based predicate evaluation |
| NumberNode | Numerical processing reservoirs |
| Attention Allocation | AttentionAgent with reservoir dynamics |
| Pattern Matching | PatternMatchAgent with template matching |
| PLN (Probabilistic Logic Networks) | Distributed reasoning through agent networks |
| MOSES (Machine Learning) | Reservoir adaptation and learning |

## Performance Characteristics

- **Scalability**: Distributed processing across multiple agents
- **Efficiency**: Sparse reservoir computations with O(N) complexity per timestep
- **Adaptability**: Dynamic reconfiguration of attention and processing
- **Robustness**: Graceful degradation with agent failures
- **Temporal Processing**: Native support for sequential and temporal data

## Future Extensions

Planned enhancements include:

- **Probabilistic Logic Networks**: Full PLN implementation with reservoirs
- **Learning Mechanisms**: Online adaptation of reservoir parameters
- **Hierarchical Processing**: Multi-level cognitive architectures  
- **External Interfaces**: Integration with sensors, actuators, and databases
- **Visualization Tools**: Interactive cognitive state visualization
- **Benchmarking**: Standardized AGI benchmarks and evaluations

## Contributing

ReservoirCog follows ReservoirPy's contribution guidelines. Key areas for contribution:

- New atom types and specialized cognitive functions
- Enhanced agent specializations and communication protocols
- Advanced pattern matching algorithms
- Performance optimizations and scalability improvements
- Application domains and use cases
- Documentation and examples

## License

ReservoirCog is released under the MIT License, same as ReservoirPy.

## Citation

When using ReservoirCog in research, please cite both ReservoirPy and this cognitive extension:

```bibtex
@misc{reservoircog2024,
  title={ReservoirCog: OpenCog-Inspired Cognitive Orchestrator using Reservoir Computing},
  author={ReservoirCog Contributors},
  year={2024},
  url={https://github.com/cogpy/reservoircog}
}
```

## References

- [OpenCog Documentation](https://wiki.opencog.org/)
- [ReservoirPy Documentation](https://reservoirpy.readthedocs.io/)
- [Reservoir Computing Theory](https://www.nature.com/articles/s41467-019-12717-0)
- [Echo State Networks](https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf)