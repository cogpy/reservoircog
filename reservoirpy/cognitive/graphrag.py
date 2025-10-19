# Licence: MIT License
# Copyright: ReservoirCog Contributors (2024)

"""
GraphRAG Integration for ReservoirPy Cognitive Systems

This module integrates Graph-based Retrieval Augmented Generation (GraphRAG) 
with ReservoirPy's AtomSpace, enabling knowledge graph enhanced reasoning
for LLM-based cognitive processing.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from collections import defaultdict, Counter
import asyncio
from dataclasses import dataclass

try:
    from llama_index.core import Document, VectorStoreIndex, StorageContext
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.vector_stores import SimpleVectorStore
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine
    HAS_LLAMAINDEX = True
except ImportError:
    HAS_LLAMAINDEX = False

from .atoms import AtomSpace, BaseAtomNode, ConceptNode, PredicateNode
from .llm_interface import BaseLLMInterface

logger = logging.getLogger(__name__)


@dataclass
class GraphRAGNode:
    """A node in the GraphRAG knowledge graph."""
    id: str
    content: str
    node_type: str
    embeddings: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    connections: Optional[List[str]] = None


@dataclass
class GraphRAGEdge:
    """An edge in the GraphRAG knowledge graph."""
    source: str
    target: str
    relation: str
    weight: float = 1.0
    metadata: Optional[Dict[str, Any]] = None


class GraphRAGEngine:
    """
    Graph-based Retrieval Augmented Generation engine for ReservoirPy.
    
    This class integrates AtomSpace knowledge graphs with LLM-based reasoning,
    providing enhanced context retrieval and knowledge-grounded generation.
    """
    
    def __init__(
        self,
        atomspace: AtomSpace,
        llm_interface: BaseLLMInterface,
        embedding_dim: int = 1536,
        max_retrieved_nodes: int = 10,
        similarity_threshold: float = 0.7,
        use_llamaindex: bool = True
    ):
        """
        Initialize GraphRAG engine.
        
        Parameters
        ----------
        atomspace : AtomSpace
            ReservoirPy AtomSpace containing knowledge
        llm_interface : BaseLLMInterface
            LLM interface for generation and embeddings
        embedding_dim : int, default=1536
            Dimensionality of embeddings
        max_retrieved_nodes : int, default=10
            Maximum number of nodes to retrieve
        similarity_threshold : float, default=0.7
            Minimum similarity for retrieval
        use_llamaindex : bool, default=True
            Whether to use LlamaIndex for advanced retrieval
        """
        self.atomspace = atomspace
        self.llm_interface = llm_interface
        self.embedding_dim = embedding_dim
        self.max_retrieved_nodes = max_retrieved_nodes
        self.similarity_threshold = similarity_threshold
        self.use_llamaindex = use_llamaindex and HAS_LLAMAINDEX
        
        # Knowledge graph storage
        self.nodes: Dict[str, GraphRAGNode] = {}
        self.edges: List[GraphRAGEdge] = []
        self.node_embeddings: Dict[str, np.ndarray] = {}
        
        # LlamaIndex components (if available)
        self.vector_index = None
        self.retriever = None
        self.query_engine = None
        
        # Initialize from AtomSpace - will be called explicitly
        self._initialized = False
    
    async def initialize(self):
        """Initialize GraphRAG from existing AtomSpace knowledge."""
        if self._initialized:
            return
            
        logger.info("Initializing GraphRAG from AtomSpace...")
        
        # Convert atoms to GraphRAG nodes
        for atom_id, atom in self.atomspace.atoms.items():
            await self._add_atom_as_node(atom_id, atom)
        
        # Convert links to GraphRAG edges
        for link_data in self.atomspace.links:
            self._add_link_as_edge(link_data)
        
        # Build vector index if using LlamaIndex
        if self.use_llamaindex:
            await self._build_vector_index()
        
        self._initialized = True
        logger.info(f"GraphRAG initialized with {len(self.nodes)} nodes and {len(self.edges)} edges")
    
    async def _add_atom_as_node(self, atom_id: str, atom: BaseAtomNode):
        """Convert an AtomSpace atom to a GraphRAG node."""
        # Generate content representation
        if isinstance(atom, ConceptNode):
            content = f"Concept: {atom.name}"
            if hasattr(atom, 'truth_value'):
                content += f" (Truth Value: {atom.truth_value:.2f})"
        elif isinstance(atom, PredicateNode):
            content = f"Predicate: {atom.name} (Arity: {atom.arity})"
            if hasattr(atom, 'confidence'):
                content += f" (Confidence: {atom.confidence:.2f})"
        else:
            content = f"Atom: {atom.name}"
        
        # Add metadata
        metadata = {
            "atom_type": type(atom).__name__,
            "attention": getattr(atom, 'attention', 0.0),
            "created_from_atomspace": True
        }
        
        # Create GraphRAG node
        node = GraphRAGNode(
            id=atom_id,
            content=content,
            node_type=type(atom).__name__,
            metadata=metadata
        )
        
        # Generate embeddings
        try:
            embeddings = await self.llm_interface.embed([content])
            node.embeddings = embeddings[0]
            self.node_embeddings[atom_id] = np.array(embeddings[0])
        except Exception as e:
            logger.warning(f"Failed to generate embeddings for atom {atom_id}: {e}")
        
        self.nodes[atom_id] = node
    
    def _add_link_as_edge(self, link_data: Dict[str, Any]):
        """Convert an AtomSpace link to a GraphRAG edge."""
        source = link_data.get("source")
        target = link_data.get("target") 
        relation = link_data.get("relation", "connected")
        strength = link_data.get("strength", 1.0)
        
        if source and target:
            edge = GraphRAGEdge(
                source=source,
                target=target,
                relation=relation,
                weight=strength,
                metadata={"created_from_atomspace": True}
            )
            self.edges.append(edge)
    
    async def _build_vector_index(self):
        """Build LlamaIndex vector store and retriever."""
        if not HAS_LLAMAINDEX:
            logger.warning("LlamaIndex not available, skipping vector index building")
            return
            
        try:
            # Create documents from nodes
            documents = []
            for node_id, node in self.nodes.items():
                doc = Document(
                    text=node.content,
                    metadata=node.metadata or {}
                )
                doc.id_ = node_id
                documents.append(doc)
            
            # Build vector index
            self.vector_index = VectorStoreIndex.from_documents(documents)
            
            # Create retriever
            self.retriever = VectorIndexRetriever(
                index=self.vector_index,
                similarity_top_k=self.max_retrieved_nodes
            )
            
            # Create query engine
            self.query_engine = RetrieverQueryEngine(retriever=self.retriever)
            
            logger.info("LlamaIndex vector index built successfully")
            
        except Exception as e:
            logger.error(f"Failed to build vector index: {e}")
            self.use_llamaindex = False
    
    async def retrieve_relevant_context(
        self, 
        query: str, 
        max_nodes: int = None,
        include_relations: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context for a query using GraphRAG.
        
        Parameters
        ----------
        query : str
            Query string to retrieve context for
        max_nodes : int, optional
            Maximum nodes to retrieve (overrides default)
        include_relations : bool, default=True
            Whether to include relation information
            
        Returns
        -------
        Dict[str, Any]
            Retrieved context including nodes, edges, and structured information
        """
        max_nodes = max_nodes or self.max_retrieved_nodes
        
        # Get query embeddings
        try:
            query_embeddings = await self.llm_interface.embed([query])
            query_vector = np.array(query_embeddings[0])
        except Exception as e:
            logger.error(f"Failed to generate query embeddings: {e}")
            return {"nodes": [], "edges": [], "context_text": ""}
        
        # Retrieve nodes using similarity search
        relevant_nodes = await self._similarity_retrieve(query_vector, max_nodes)
        
        # Get connected edges if requested
        relevant_edges = []
        if include_relations:
            relevant_edges = self._get_relevant_edges(
                [node["id"] for node in relevant_nodes]
            )
        
        # Build context text
        context_text = self._build_context_text(relevant_nodes, relevant_edges)
        
        return {
            "nodes": relevant_nodes,
            "edges": relevant_edges,
            "context_text": context_text,
            "query": query
        }
    
    async def _similarity_retrieve(
        self, 
        query_vector: np.ndarray, 
        max_nodes: int
    ) -> List[Dict[str, Any]]:
        """Retrieve nodes using similarity search."""
        if self.use_llamaindex and self.retriever:
            return await self._llamaindex_retrieve(query_vector, max_nodes)
        else:
            return await self._manual_similarity_retrieve(query_vector, max_nodes)
    
    async def _llamaindex_retrieve(
        self, 
        query_vector: np.ndarray, 
        max_nodes: int
    ) -> List[Dict[str, Any]]:
        """Retrieve using LlamaIndex."""
        try:
            # Use the query engine for retrieval
            # Note: This is a simplified approach; in practice you might want
            # to use the vector directly with the retriever
            retrieved_nodes = []
            
            # Get similarities manually since we have embeddings
            similarities = []
            for node_id, node_embedding in self.node_embeddings.items():
                similarity = np.dot(query_vector, node_embedding) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(node_embedding)
                )
                if similarity >= self.similarity_threshold:
                    similarities.append((node_id, similarity))
            
            # Sort by similarity and take top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_similarities = similarities[:max_nodes]
            
            for node_id, similarity in top_similarities:
                node = self.nodes[node_id]
                retrieved_nodes.append({
                    "id": node_id,
                    "content": node.content,
                    "type": node.node_type,
                    "similarity": similarity,
                    "metadata": node.metadata
                })
            
            return retrieved_nodes
            
        except Exception as e:
            logger.error(f"LlamaIndex retrieval failed: {e}")
            return await self._manual_similarity_retrieve(query_vector, max_nodes)
    
    async def _manual_similarity_retrieve(
        self, 
        query_vector: np.ndarray, 
        max_nodes: int
    ) -> List[Dict[str, Any]]:
        """Manual similarity-based retrieval."""
        similarities = []
        
        for node_id, node_embedding in self.node_embeddings.items():
            # Cosine similarity
            similarity = np.dot(query_vector, node_embedding) / (
                np.linalg.norm(query_vector) * np.linalg.norm(node_embedding)
            )
            
            if similarity >= self.similarity_threshold:
                similarities.append((node_id, similarity))
        
        # Sort by similarity and take top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similarities = similarities[:max_nodes]
        
        retrieved_nodes = []
        for node_id, similarity in top_similarities:
            node = self.nodes[node_id]
            retrieved_nodes.append({
                "id": node_id,
                "content": node.content,
                "type": node.node_type,
                "similarity": similarity,
                "metadata": node.metadata
            })
        
        return retrieved_nodes
    
    def _get_relevant_edges(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """Get edges connecting the retrieved nodes."""
        node_set = set(node_ids)
        relevant_edges = []
        
        for edge in self.edges:
            if edge.source in node_set or edge.target in node_set:
                relevant_edges.append({
                    "source": edge.source,
                    "target": edge.target,
                    "relation": edge.relation,
                    "weight": edge.weight,
                    "metadata": edge.metadata
                })
        
        return relevant_edges
    
    def _build_context_text(
        self, 
        nodes: List[Dict[str, Any]], 
        edges: List[Dict[str, Any]]
    ) -> str:
        """Build structured context text from retrieved nodes and edges."""
        context_parts = []
        
        if nodes:
            context_parts.append("Relevant Knowledge:")
            for i, node in enumerate(nodes, 1):
                context_parts.append(
                    f"{i}. {node['content']} "
                    f"(Similarity: {node['similarity']:.3f})"
                )
        
        if edges:
            context_parts.append("\nRelevant Relations:")
            relations_by_type = defaultdict(list)
            
            for edge in edges:
                source_content = self.nodes.get(edge["source"], {}).get("content", edge["source"])
                target_content = self.nodes.get(edge["target"], {}).get("content", edge["target"])
                relations_by_type[edge["relation"]].append(
                    f"{source_content} -> {target_content}"
                )
            
            for relation, connections in relations_by_type.items():
                context_parts.append(f"\n{relation.title()}:")
                for connection in connections[:5]:  # Limit to avoid too long context
                    context_parts.append(f"  - {connection}")
        
        return "\n".join(context_parts)
    
    async def add_knowledge(
        self, 
        content: str, 
        node_type: str = "document",
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Add new knowledge to the GraphRAG system.
        
        Parameters
        ----------
        content : str
            Content to add
        node_type : str, default="document"
            Type of the knowledge node
        metadata : Dict[str, Any], optional
            Additional metadata
            
        Returns
        -------
        str
            ID of the created node
        """
        node_id = f"{node_type}_{len(self.nodes)}"
        
        # Generate embeddings
        try:
            embeddings = await self.llm_interface.embed([content])
            embedding_vector = embeddings[0]
            self.node_embeddings[node_id] = np.array(embedding_vector)
        except Exception as e:
            logger.error(f"Failed to generate embeddings for new content: {e}")
            embedding_vector = None
        
        # Create node
        node = GraphRAGNode(
            id=node_id,
            content=content,
            node_type=node_type,
            embeddings=embedding_vector,
            metadata=metadata or {}
        )
        
        self.nodes[node_id] = node
        
        # Rebuild index if using LlamaIndex
        if self.use_llamaindex:
            await self._build_vector_index()
        
        logger.info(f"Added new knowledge node: {node_id}")
        return node_id
    
    async def generate_with_context(
        self, 
        query: str,
        system_message: str = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        include_context_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Generate response using GraphRAG context retrieval.
        
        Parameters
        ----------
        query : str
            User query
        system_message : str, optional
            System message for the LLM
        max_tokens : int, default=1000
            Maximum tokens to generate
        temperature : float, default=0.7
            Generation temperature
        include_context_sources : bool, default=True
            Whether to include context source information
            
        Returns
        -------
        Dict[str, Any]
            Generated response with context information
        """
        # Retrieve relevant context
        context_data = await self.retrieve_relevant_context(query)
        
        # Build enhanced prompt with context
        context_text = context_data["context_text"]
        
        if context_text:
            enhanced_query = f"""Context Information:
{context_text}

Based on the above context, please answer the following question:
{query}"""
        else:
            enhanced_query = query
        
        # Generate response
        try:
            response = await self.llm_interface.generate(
                prompt=enhanced_query,
                system_message=system_message,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            result = {
                "response": response,
                "query": query,
                "context_used": bool(context_text),
                "retrieved_nodes": len(context_data["nodes"]),
                "retrieved_edges": len(context_data["edges"])
            }
            
            if include_context_sources:
                result["context_nodes"] = context_data["nodes"]
                result["context_edges"] = context_data["edges"]
                result["context_text"] = context_text
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise


# Convenience function for creating GraphRAG engines
def create_graphrag_engine(
    atomspace: AtomSpace,
    llm_interface: BaseLLMInterface,
    **kwargs
) -> GraphRAGEngine:
    """
    Create a GraphRAG engine with the given AtomSpace and LLM interface.
    
    Parameters
    ----------
    atomspace : AtomSpace
        ReservoirPy AtomSpace containing knowledge
    llm_interface : BaseLLMInterface
        LLM interface for generation and embeddings
    **kwargs
        Additional configuration options
        
    Returns
    -------
    GraphRAGEngine
        Configured GraphRAG engine
    """
    return GraphRAGEngine(atomspace, llm_interface, **kwargs)