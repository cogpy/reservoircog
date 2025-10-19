# Licence: MIT License
# Copyright: ReservoirCog Contributors (2024)

"""
LLM Interface for ReservoirPy Cognitive Systems

This module provides interfaces to Large Language Models (LLMs) for natural
language processing within the ReservoirPy cognitive architecture. It supports
multiple backends including Cohere, OpenAI, and local models.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from abc import ABC, abstractmethod
import asyncio

try:
    import cohere
    HAS_COHERE = True
except ImportError:
    HAS_COHERE = False

try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

logger = logging.getLogger(__name__)


class BaseLLMInterface(ABC):
    """Base class for LLM interfaces."""
    
    def __init__(self, model_name: str = None, **kwargs):
        self.model_name = model_name
        self.config = kwargs
        
    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        system_message: str = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text completion from prompt."""
        pass
    
    @abstractmethod
    async def stream_generate(
        self, 
        prompt: str, 
        system_message: str = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream text completion from prompt."""
        pass
    
    @abstractmethod
    async def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Generate embeddings for text(s)."""
        pass


class CohereLLMInterface(BaseLLMInterface):
    """Cohere LLM interface for natural language generation and embeddings."""
    
    def __init__(
        self, 
        api_key: str = None,
        model_name: str = "command-r-plus",
        embed_model: str = "embed-english-v3.0",
        **kwargs
    ):
        if not HAS_COHERE:
            raise ImportError("cohere package not installed. Run: pip install cohere")
            
        super().__init__(model_name, **kwargs)
        
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("Cohere API key required. Set COHERE_API_KEY environment variable.")
            
        self.client = cohere.AsyncClient(api_key=self.api_key)
        self.embed_model = embed_model
        
    async def generate(
        self, 
        prompt: str, 
        system_message: str = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text using Cohere's command model."""
        try:
            # Combine system message and prompt if provided
            if system_message:
                full_prompt = f"System: {system_message}\n\nUser: {prompt}"
            else:
                full_prompt = prompt
                
            response = await self.client.generate(
                model=self.model_name,
                prompt=full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            return response.generations[0].text.strip()
            
        except Exception as e:
            logger.error(f"Error generating with Cohere: {e}")
            raise
    
    async def stream_generate(
        self, 
        prompt: str, 
        system_message: str = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream text generation using Cohere."""
        try:
            # Combine system message and prompt if provided
            if system_message:
                full_prompt = f"System: {system_message}\n\nUser: {prompt}"
            else:
                full_prompt = prompt
                
            response = await self.client.generate(
                model=self.model_name,
                prompt=full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                **kwargs
            )
            
            async for token in response:
                if hasattr(token, 'text') and token.text:
                    yield token.text
                    
        except Exception as e:
            logger.error(f"Error streaming with Cohere: {e}")
            raise
    
    async def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Generate embeddings using Cohere's embed model."""
        try:
            if isinstance(texts, str):
                texts = [texts]
                
            response = await self.client.embed(
                texts=texts,
                model=self.embed_model,
                input_type="search_document"
            )
            
            return response.embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings with Cohere: {e}")
            raise


class OpenAILLMInterface(BaseLLMInterface):
    """OpenAI LLM interface for GPT models."""
    
    def __init__(
        self, 
        api_key: str = None,
        model_name: str = "gpt-3.5-turbo",
        embed_model: str = "text-embedding-3-small",
        **kwargs
    ):
        if not HAS_OPENAI:
            raise ImportError("openai package not installed. Run: pip install openai")
            
        super().__init__(model_name, **kwargs)
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
            
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.embed_model = embed_model
        
    async def generate(
        self, 
        prompt: str, 
        system_message: str = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text using OpenAI's chat models."""
        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
                
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating with OpenAI: {e}")
            raise
    
    async def stream_generate(
        self, 
        prompt: str, 
        system_message: str = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream text generation using OpenAI."""
        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
                
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                **kwargs
            )
            
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error streaming with OpenAI: {e}")
            raise
    
    async def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Generate embeddings using OpenAI's embedding model."""
        try:
            if isinstance(texts, str):
                texts = [texts]
                
            response = await self.client.embeddings.create(
                input=texts,
                model=self.embed_model
            )
            
            return [item.embedding for item in response.data]
            
        except Exception as e:
            logger.error(f"Error generating embeddings with OpenAI: {e}")
            raise


class LLMRegistry:
    """Registry for managing multiple LLM interfaces."""
    
    _interfaces = {
        "cohere": CohereLLMInterface,
        "openai": OpenAILLMInterface,
    }
    
    @classmethod
    def create_interface(cls, provider: str, **kwargs) -> BaseLLMInterface:
        """Create an LLM interface for the specified provider."""
        if provider not in cls._interfaces:
            raise ValueError(f"Unknown LLM provider: {provider}. Available: {list(cls._interfaces.keys())}")
            
        interface_class = cls._interfaces[provider]
        return interface_class(**kwargs)
    
    @classmethod
    def register_interface(cls, name: str, interface_class: type):
        """Register a custom LLM interface."""
        cls._interfaces[name] = interface_class
    
    @classmethod
    def available_providers(cls) -> List[str]:
        """Get list of available LLM providers."""
        return list(cls._interfaces.keys())


# Convenience function for creating LLM interfaces
def create_llm_interface(provider: str = "cohere", **kwargs) -> BaseLLMInterface:
    """
    Create an LLM interface for the specified provider.
    
    Parameters
    ----------
    provider : str, default="cohere"
        LLM provider ("cohere", "openai")
    **kwargs
        Provider-specific configuration
        
    Returns
    -------
    BaseLLMInterface
        Configured LLM interface
        
    Examples
    --------
    >>> # Create Cohere interface
    >>> cohere_llm = create_llm_interface("cohere", api_key="your-key")
    >>> 
    >>> # Create OpenAI interface
    >>> openai_llm = create_llm_interface("openai", model_name="gpt-4")
    """
    return LLMRegistry.create_interface(provider, **kwargs)