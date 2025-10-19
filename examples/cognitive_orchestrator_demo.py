#!/usr/bin/env python3
"""
ReservoirCog Cognitive Orchestrator Demonstration

This example shows how to use the OpenCog-inspired cognitive orchestrator
built on top of ReservoirPy's reservoir computing architecture.

The demo creates a knowledge base about people, their relationships, and
preferences, then demonstrates cognitive processing including:
- Knowledge representation using ConceptNodes and PredicateNodes
- Distributed reasoning through echo state agent networks  
- Attention allocation and spreading
- Pattern matching and similarity detection
- Multi-cycle cognitive processing
"""

import numpy as np
from reservoirpy.cognitive import CognitiveOrchestrator

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def create_social_knowledge_base(orchestrator):
    """Create a social knowledge base with people, relationships, and preferences."""
    print("Building social knowledge base...")
    
    # Create person concepts
    alice = orchestrator.add_concept("Alice", attention=0.9, truth_value=1.0)
    bob = orchestrator.add_concept("Bob", attention=0.8, truth_value=1.0)
    charlie = orchestrator.add_concept("Charlie", attention=0.7, truth_value=1.0)
    diana = orchestrator.add_concept("Diana", attention=0.6, truth_value=1.0)
    
    # Create activity concepts
    music = orchestrator.add_concept("music", attention=0.8, truth_value=0.9)
    sports = orchestrator.add_concept("sports", attention=0.7, truth_value=0.9) 
    reading = orchestrator.add_concept("reading", attention=0.6, truth_value=0.9)
    cooking = orchestrator.add_concept("cooking", attention=0.5, truth_value=0.9)
    
    # Create relationship predicates
    likes = orchestrator.add_predicate("likes", arity=2, confidence=0.95)
    friends = orchestrator.add_predicate("friends", arity=2, confidence=0.9)
    similar = orchestrator.add_predicate("similar", arity=2, confidence=0.8)
    
    # Create numerical concepts
    age_25 = orchestrator.add_number(25.0, attention=0.4)
    age_30 = orchestrator.add_number(30.0, attention=0.4)
    rating_high = orchestrator.add_number(9.0, attention=0.3)
    
    # Initialize all atoms with varied but related patterns
    base_pattern = np.array([0.8, 0.6, 0.9, 0.7, 0.5, 0.8, 0.6, 0.7, 0.9, 0.5])
    
    # People get similar patterns (social similarity)
    alice.initialize(base_pattern + np.random.normal(0, 0.1, 10))
    bob.initialize(base_pattern + np.random.normal(0, 0.1, 10))
    charlie.initialize(base_pattern + np.random.normal(0, 0.15, 10))
    diana.initialize(base_pattern + np.random.normal(0, 0.12, 10))
    
    # Activities get distinct patterns
    music.initialize(np.array([0.9, 0.8, 0.7, 0.9, 0.8, 0.6, 0.9, 0.7, 0.8, 0.9]))
    sports.initialize(np.array([0.7, 0.9, 0.8, 0.6, 0.9, 0.8, 0.7, 0.9, 0.6, 0.8]))
    reading.initialize(np.array([0.6, 0.7, 0.9, 0.8, 0.5, 0.9, 0.8, 0.6, 0.9, 0.7]))
    cooking.initialize(np.array([0.8, 0.6, 0.5, 0.9, 0.7, 0.8, 0.5, 0.8, 0.7, 0.6]))
    
    # Initialize predicates and numbers
    likes.initialize(np.random.randn(10) * 0.3)
    friends.initialize(np.random.randn(10) * 0.3)
    similar.initialize(np.random.randn(10) * 0.3)
    age_25.initialize(np.random.randn(10) * 0.2)
    age_30.initialize(np.random.randn(10) * 0.2)
    rating_high.initialize(np.random.randn(10) * 0.2)
    
    # Create social network relationships
    print("  Creating social relationships...")
    orchestrator.create_link(alice, bob, "friend")
    orchestrator.create_link(bob, charlie, "friend")  
    orchestrator.create_link(charlie, diana, "friend")
    orchestrator.create_link(alice, diana, "acquaintance")
    
    # Create preference relationships
    print("  Creating preference relationships...")
    orchestrator.create_link(alice, music, "enjoys")
    orchestrator.create_link(alice, reading, "enjoys")
    orchestrator.create_link(bob, sports, "enjoys")
    orchestrator.create_link(bob, music, "enjoys")
    orchestrator.create_link(charlie, cooking, "enjoys")
    orchestrator.create_link(charlie, reading, "enjoys")
    orchestrator.create_link(diana, music, "enjoys")
    orchestrator.create_link(diana, sports, "enjoys")
    
    # Connect predicates to concepts
    orchestrator.create_link(likes, alice, "evaluates")
    orchestrator.create_link(likes, music, "evaluates")
    orchestrator.create_link(friends, alice, "relates")
    orchestrator.create_link(friends, bob, "relates")
    
    return {
        'people': [alice, bob, charlie, diana],
        'activities': [music, sports, reading, cooking],
        'predicates': [likes, friends, similar],
        'numbers': [age_25, age_30, rating_high]
    }


def demonstrate_cognitive_processing(orchestrator, knowledge):
    """Demonstrate multi-phase cognitive processing."""
    print("\n" + "="*60)
    print("COGNITIVE PROCESSING DEMONSTRATION") 
    print("="*60)
    
    # Social interaction stimulus (focus on Alice and music)
    social_stimulus = np.array([0.9, 0.8, 0.9, 0.7, 0.6, 0.8, 0.9, 0.7, 0.8, 0.9])
    
    print(f"\nProcessing social stimulus: {social_stimulus[:5]}...")
    
    # Run multiple cognitive cycles
    results = orchestrator.cognitive_cycle(social_stimulus, cycles=5)
    
    print(f"\nCompleted {results['total_cycles']} cognitive cycles")
    print(f"Average cycle time: {results['average_cycle_time']:.4f} seconds")
    
    # Analyze results
    print("\nCognitive Processing Analysis:")
    print("-" * 40)
    
    total_perceptions = sum(cycle['perception']['concepts_created'] for cycle in results['cycles'])
    total_patterns = sum(cycle['patterns']['patterns_found'] for cycle in results['cycles'])
    total_reasoning = sum(cycle['reasoning']['atoms_processed'] for cycle in results['cycles'])
    
    print(f"  Total concepts created: {total_perceptions}")
    print(f"  Total patterns found: {total_patterns}")
    print(f"  Total reasoning steps: {total_reasoning}")
    
    # Show attention dynamics
    print(f"\nAttention Dynamics:")
    print("-" * 20)
    for person in knowledge['people']:
        print(f"  {person.concept_name}: attention = {person.attention:.3f}")
        
    for activity in knowledge['activities']:
        print(f"  {activity.concept_name}: attention = {activity.attention:.3f}")
    
    return results


def demonstrate_pattern_matching(orchestrator, knowledge):
    """Demonstrate pattern matching and similarity detection."""
    print("\n" + "="*60)
    print("PATTERN MATCHING DEMONSTRATION")
    print("="*60)
    
    pattern_agent = orchestrator.agent_network.pattern_agent
    if pattern_agent is None:
        print("No pattern matching agent available")
        return
        
    # Add pattern templates for social similarity
    pattern_agent.add_pattern_template("music_lovers", {
        'type': 'ConceptNode',
        'interest': 'music',
        'threshold': 0.7
    })
    
    pattern_agent.add_pattern_template("social_people", {
        'type': 'ConceptNode', 
        'category': 'person',
        'attention_level': 'high'
    })
    
    # Find similar atoms to Alice
    alice = knowledge['people'][0]  # Alice
    print(f"\nFinding atoms similar to {alice.concept_name}...")
    
    similar_atoms = pattern_agent.find_similar_atoms(
        orchestrator.atomspace, alice, top_k=3
    )
    
    print(f"Found {len(similar_atoms)} similar atoms:")
    for i, atom in enumerate(similar_atoms):
        print(f"  {i+1}. {atom.atom_name} (attention: {atom.attention:.3f})")
    
    # Pattern matching
    print(f"\nMatching patterns in AtomSpace...")
    
    for pattern_name in pattern_agent.pattern_templates:
        matches = pattern_agent.match_pattern(orchestrator.atomspace, pattern_name)
        print(f"  Pattern '{pattern_name}': {len(matches)} matches")


def demonstrate_predicate_evaluation(orchestrator, knowledge):
    """Demonstrate predicate evaluation with various arguments."""
    print("\n" + "="*60)
    print("PREDICATE EVALUATION DEMONSTRATION")
    print("="*60)
    
    likes_predicate = knowledge['predicates'][0]  # likes
    friends_predicate = knowledge['predicates'][1]  # friends
    
    # Test various predicate evaluations
    evaluations = [
        ("Alice", "music"),
        ("Bob", "sports"), 
        ("Charlie", "cooking"),
        ("Diana", "reading"),
        ("Alice", "Bob"),
        ("Bob", "Charlie")
    ]
    
    print(f"\nEvaluating '{likes_predicate.predicate_name}' predicate:")
    print("-" * 30)
    for arg1, arg2 in evaluations[:4]:
        score = likes_predicate.evaluate(arg1, arg2)
        print(f"  likes({arg1}, {arg2}) = {score:.3f}")
        
    print(f"\nEvaluating '{friends_predicate.predicate_name}' predicate:")
    print("-" * 30)
    for arg1, arg2 in evaluations[4:]:
        score = friends_predicate.evaluate(arg1, arg2)
        print(f"  friends({arg1}, {arg2}) = {score:.3f}")


def demonstrate_distributed_agents(orchestrator):
    """Demonstrate distributed agent network behavior."""
    print("\n" + "="*60)
    print("DISTRIBUTED AGENT NETWORK DEMONSTRATION")
    print("="*60)
    
    network = orchestrator.agent_network
    
    print(f"Network has {len(network.agents)} cognitive agents:")
    for agent_name, agent in network.agents.items():
        print(f"  {agent_name}: specialty = {agent.specialty}")
    
    # Test agent communication
    agents_list = list(network.agents.values())
    if len(agents_list) >= 2:
        sender = agents_list[0]
        receiver = agents_list[1]
        
        print(f"\nTesting communication between agents...")
        print(f"  {sender.agent_name} -> {receiver.agent_name}")
        
        # Send test message
        test_message = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        sender.send_message(test_message, receiver)
        
        print(f"  Messages in {receiver.agent_name} buffer: {len(receiver.message_buffer)}")
        if receiver.message_buffer:
            received = receiver.message_buffer[-1]
            print(f"  Last message from: {received['sender'].agent_name}")
            print(f"  Message content: {received['message'][:3]}...")
    
    # Demonstrate distributed processing
    print(f"\nRunning distributed processing...")
    
    test_input = np.random.randn(10)
    distributed_results = network.process_distributed(test_input, steps=3)
    
    print(f"  Processing completed across {len(distributed_results)} agents")
    print(f"  Each agent processed {distributed_results[list(distributed_results.keys())[0]].shape[0]} steps")
    
    # Show network state
    network_state = network.get_network_state()
    print(f"\nNetwork State:")
    print(f"  Global activity: {network_state['global_activity']:.3f}")
    print(f"  Agent connections: {sum(len(conns) for conns in network_state['connections'].values())}")


def visualize_cognitive_state(orchestrator, results):
    """Create visualizations of the cognitive processing results."""
    print("\n" + "="*60)
    print("COGNITIVE STATE VISUALIZATION")
    print("="*60)
    
    if not HAS_MATPLOTLIB:
        print("  matplotlib not available - skipping visualization")
        return
    
    try:
        # Extract data for visualization
        cycles = list(range(len(results['cycles'])))
        
        perception_data = [cycle['perception']['concepts_created'] for cycle in results['cycles']]
        attention_data = [cycle['attention']['focus_atoms'] for cycle in results['cycles']]
        pattern_data = [cycle['patterns']['patterns_found'] for cycle in results['cycles']]
        reasoning_data = [cycle['reasoning']['atoms_processed'] for cycle in results['cycles']]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('ReservoirCog Cognitive Processing Results', fontsize=14)
        
        # Perception over time
        ax1.plot(cycles, perception_data, 'b-o', linewidth=2, markersize=6)
        ax1.set_title('Concepts Created per Cycle')
        ax1.set_xlabel('Cognitive Cycle')
        ax1.set_ylabel('New Concepts')
        ax1.grid(True, alpha=0.3)
        
        # Attention dynamics
        ax2.plot(cycles, attention_data, 'r-s', linewidth=2, markersize=6)
        ax2.set_title('Attention Focus per Cycle')
        ax2.set_xlabel('Cognitive Cycle')
        ax2.set_ylabel('Focused Atoms')
        ax2.grid(True, alpha=0.3)
        
        # Pattern matching
        ax3.plot(cycles, pattern_data, 'g-^', linewidth=2, markersize=6)
        ax3.set_title('Patterns Found per Cycle')
        ax3.set_xlabel('Cognitive Cycle')
        ax3.set_ylabel('Pattern Matches')
        ax3.grid(True, alpha=0.3)
        
        # Reasoning activity
        ax4.plot(cycles, reasoning_data, 'm-d', linewidth=2, markersize=6)
        ax4.set_title('Reasoning Steps per Cycle')
        ax4.set_xlabel('Cognitive Cycle')
        ax4.set_ylabel('Atoms Processed')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cognitive_processing_results.png', dpi=150, bbox_inches='tight')
        print("  Cognitive processing visualization saved as 'cognitive_processing_results.png'")
        
        # Create attention distribution chart
        fig2, ax = plt.subplots(figsize=(10, 6))
        
        atom_names = []
        attention_values = []
        
        for atom_key, atom in orchestrator.atomspace.atoms.items():
            atom_names.append(atom.atom_name)
            attention_values.append(atom.attention)
        
        colors = ['red' if att > 0.7 else 'orange' if att > 0.4 else 'blue' for att in attention_values]
        
        bars = ax.bar(range(len(atom_names)), attention_values, color=colors, alpha=0.7)
        ax.set_title('Attention Distribution Across Atoms')
        ax.set_xlabel('Atoms')
        ax.set_ylabel('Attention Level')
        ax.set_xticks(range(len(atom_names)))
        ax.set_xticklabels(atom_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add attention threshold lines
        ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='High Attention')
        ax.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Medium Attention')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('attention_distribution.png', dpi=150, bbox_inches='tight')
        print("  Attention distribution visualization saved as 'attention_distribution.png'")
        
    except ImportError:
        print("  matplotlib not available - skipping visualization")
    except Exception as e:
        print(f"  Visualization error: {e}")


def main():
    """Main demonstration function."""
    print("="*80)
    print("RESERVOIRCOG: OpenCog-Inspired Cognitive Orchestrator Demo")
    print("Built on ReservoirPy reservoir computing architecture")
    print("="*80)
    
    # Create cognitive orchestrator with distributed agent network
    print("\nInitializing cognitive orchestrator...")
    orchestrator = CognitiveOrchestrator(
        atomspace_name="SocialKnowledgeSpace",
        network_size=8  # Larger network for more complex processing
    )
    
    print(f"  Created AtomSpace: {orchestrator.atomspace.name}")
    print(f"  Initialized agent network with {len(orchestrator.agent_network.agents)} agents")
    
    # Build knowledge base
    knowledge = create_social_knowledge_base(orchestrator)
    print(f"  Knowledge base created with {len(orchestrator.atomspace.atoms)} atoms")
    
    # Demonstrate core cognitive processing
    cognitive_results = demonstrate_cognitive_processing(orchestrator, knowledge)
    
    # Show pattern matching capabilities
    demonstrate_pattern_matching(orchestrator, knowledge)
    
    # Demonstrate predicate evaluation
    demonstrate_predicate_evaluation(orchestrator, knowledge)
    
    # Show distributed agent behavior  
    demonstrate_distributed_agents(orchestrator)
    
    # Create visualizations
    visualize_cognitive_state(orchestrator, cognitive_results)
    
    # Final system summary
    print("\n" + "="*80)
    print("SYSTEM SUMMARY")
    print("="*80)
    
    final_state = orchestrator.get_cognitive_state()
    
    print(f"AtomSpace Statistics:")
    print(f"  Total atoms: {final_state['atomspace']['total_atoms']}")
    print(f"  Atom types: {dict(final_state['atomspace']['atom_types'])}")
    print(f"  Average attention: {final_state['atomspace']['average_attention']:.3f}")
    
    print(f"\nAgent Network Statistics:")
    print(f"  Number of agents: {final_state['agent_network']['num_agents']}")
    print(f"  Global activity: {final_state['agent_network']['global_activity']:.3f}")
    
    print(f"\nProcessing Statistics:")
    print(f"  Total cognitive cycles: {final_state['processing_stats']['total_cycles']}")
    print(f"  Attention cycles: {final_state['processing_stats']['attention_cycles']}")
    
    print(f"\nDemo completed successfully!")
    print(f"This demonstrates OpenCog-style cognitive architecture")
    print(f"implemented using ReservoirPy's reservoir computing framework.")


if __name__ == "__main__":
    main()