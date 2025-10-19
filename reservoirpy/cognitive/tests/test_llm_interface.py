# Licence: MIT License
# Copyright: ReservoirCog Contributors (2024)

"""
Tests for LLM interface components.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch

from reservoirpy.cognitive.llm_interface import (
    BaseLLMInterface,
    CohereLLMInterface,
    OpenAILLMInterface,
    LLMRegistry,
    create_llm_interface
)


class MockLLMInterface(BaseLLMInterface):
    """Mock LLM interface for testing."""
    
    def __init__(self, **kwargs):
        super().__init__("mock-model", **kwargs)
        self.generate_calls = []
        self.stream_calls = []
        self.embed_calls = []
    
    async def generate(self, prompt: str, system_message: str = None, **kwargs) -> str:
        self.generate_calls.append({
            "prompt": prompt,
            "system_message": system_message,
            **kwargs
        })
        return f"Mock response to: {prompt[:50]}"
    
    async def stream_generate(self, prompt: str, system_message: str = None, **kwargs):
        self.stream_calls.append({
            "prompt": prompt,
            "system_message": system_message,
            **kwargs
        })
        for token in ["Mock", " streaming", " response"]:
            yield token
    
    async def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        self.embed_calls.append(texts)
        # Return mock embeddings
        return [[0.1, 0.2, 0.3] for _ in texts]


class TestBaseLLMInterface:
    """Test base LLM interface functionality."""
    
    def test_initialization(self):
        """Test base LLM interface initialization."""
        interface = MockLLMInterface(model_name="test-model", param1="value1")
        
        assert interface.model_name == "test-model"
        assert interface.config["param1"] == "value1"
    
    @pytest.mark.asyncio
    async def test_generate(self):
        """Test text generation."""
        interface = MockLLMInterface()
        
        response = await interface.generate("Test prompt", system_message="System msg")
        
        assert "Mock response to: Test prompt" in response
        assert len(interface.generate_calls) == 1
        assert interface.generate_calls[0]["prompt"] == "Test prompt"
        assert interface.generate_calls[0]["system_message"] == "System msg"
    
    @pytest.mark.asyncio
    async def test_stream_generate(self):
        """Test streaming generation."""
        interface = MockLLMInterface()
        
        tokens = []
        async for token in interface.stream_generate("Test prompt"):
            tokens.append(token)
        
        assert tokens == ["Mock", " streaming", " response"]
        assert len(interface.stream_calls) == 1
    
    @pytest.mark.asyncio
    async def test_embed_single_text(self):
        """Test embedding single text."""
        interface = MockLLMInterface()
        
        embeddings = await interface.embed("Test text")
        
        assert len(embeddings) == 1
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert interface.embed_calls[0] == ["Test text"]
    
    @pytest.mark.asyncio
    async def test_embed_multiple_texts(self):
        """Test embedding multiple texts."""
        interface = MockLLMInterface()
        
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = await interface.embed(texts)
        
        assert len(embeddings) == 3
        assert all(emb == [0.1, 0.2, 0.3] for emb in embeddings)
        assert interface.embed_calls[0] == texts


class TestCohereLLMInterface:
    """Test Cohere LLM interface."""
    
    @patch('reservoirpy.cognitive.llm_interface.HAS_COHERE', True)
    @patch('reservoirpy.cognitive.llm_interface.cohere')
    def test_initialization(self, mock_cohere):
        """Test Cohere interface initialization."""
        mock_client = Mock()
        mock_cohere.AsyncClient.return_value = mock_client
        
        interface = CohereLLMInterface(api_key="test-key")
        
        assert interface.api_key == "test-key"
        assert interface.model_name == "command-r-plus"
        mock_cohere.AsyncClient.assert_called_once_with(api_key="test-key")
    
    def test_initialization_no_cohere(self):
        """Test initialization fails without cohere package."""
        with patch('reservoirpy.cognitive.llm_interface.HAS_COHERE', False):
            with pytest.raises(ImportError, match="cohere package not installed"):
                CohereLLMInterface(api_key="test-key")
    
    def test_initialization_no_api_key(self):
        """Test initialization fails without API key."""
        with patch('reservoirpy.cognitive.llm_interface.HAS_COHERE', True):
            with patch('os.getenv', return_value=None):
                with pytest.raises(ValueError, match="Cohere API key required"):
                    CohereLLMInterface()
    
    @patch('reservoirpy.cognitive.llm_interface.HAS_COHERE', True)
    @patch('reservoirpy.cognitive.llm_interface.cohere')
    @pytest.mark.asyncio
    async def test_generate(self, mock_cohere):
        """Test Cohere text generation."""
        # Setup mock client
        mock_client = AsyncMock()
        mock_cohere.AsyncClient.return_value = mock_client
        
        # Mock response
        mock_generation = Mock()
        mock_generation.text = "Generated response"
        mock_response = Mock()
        mock_response.generations = [mock_generation]
        mock_client.generate.return_value = mock_response
        
        # Test generation
        interface = CohereLLMInterface(api_key="test-key")
        response = await interface.generate("Test prompt", system_message="System")
        
        assert response == "Generated response"
        mock_client.generate.assert_called_once()
        call_args = mock_client.generate.call_args
        assert call_args[1]["model"] == "command-r-plus"
        assert "System: System\n\nUser: Test prompt" in call_args[1]["prompt"]
    
    @patch('reservoirpy.cognitive.llm_interface.HAS_COHERE', True)
    @patch('reservoirpy.cognitive.llm_interface.cohere')
    @pytest.mark.asyncio
    async def test_embed(self, mock_cohere):
        """Test Cohere embeddings."""
        # Setup mock client
        mock_client = AsyncMock()
        mock_cohere.AsyncClient.return_value = mock_client
        
        # Mock response
        mock_response = Mock()
        mock_response.embeddings = [[0.1, 0.2], [0.3, 0.4]]
        mock_client.embed.return_value = mock_response
        
        # Test embedding
        interface = CohereLLMInterface(api_key="test-key")
        embeddings = await interface.embed(["Text 1", "Text 2"])
        
        assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
        mock_client.embed.assert_called_once_with(
            texts=["Text 1", "Text 2"],
            model="embed-english-v3.0",
            input_type="search_document"
        )


class TestOpenAILLMInterface:
    """Test OpenAI LLM interface."""
    
    @patch('reservoirpy.cognitive.llm_interface.HAS_OPENAI', True)
    @patch('reservoirpy.cognitive.llm_interface.AsyncOpenAI')
    def test_initialization(self, mock_openai):
        """Test OpenAI interface initialization."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        interface = OpenAILLMInterface(api_key="test-key")
        
        assert interface.api_key == "test-key"
        assert interface.model_name == "gpt-3.5-turbo"
        mock_openai.assert_called_once_with(api_key="test-key")
    
    def test_initialization_no_openai(self):
        """Test initialization fails without openai package."""
        with patch('reservoirpy.cognitive.llm_interface.HAS_OPENAI', False):
            with pytest.raises(ImportError, match="openai package not installed"):
                OpenAILLMInterface(api_key="test-key")
    
    @patch('reservoirpy.cognitive.llm_interface.HAS_OPENAI', True)
    @patch('reservoirpy.cognitive.llm_interface.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_generate(self, mock_openai):
        """Test OpenAI text generation."""
        # Setup mock client
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        
        # Mock response
        mock_choice = Mock()
        mock_choice.message.content = "Generated response"
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        
        # Test generation
        interface = OpenAILLMInterface(api_key="test-key")
        response = await interface.generate("Test prompt", system_message="System")
        
        assert response == "Generated response"
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["model"] == "gpt-3.5-turbo"
        messages = call_args[1]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Test prompt"


class TestLLMRegistry:
    """Test LLM registry functionality."""
    
    def test_create_cohere_interface(self):
        """Test creating Cohere interface through registry."""
        with patch('reservoirpy.cognitive.llm_interface.CohereLLMInterface') as mock_cohere:
            mock_instance = Mock()
            mock_cohere.return_value = mock_instance
            
            interface = LLMRegistry.create_interface("cohere", api_key="test")
            
            assert interface == mock_instance
            mock_cohere.assert_called_once_with(api_key="test")
    
    def test_create_openai_interface(self):
        """Test creating OpenAI interface through registry."""
        with patch('reservoirpy.cognitive.llm_interface.OpenAILLMInterface') as mock_openai:
            mock_instance = Mock()
            mock_openai.return_value = mock_instance
            
            interface = LLMRegistry.create_interface("openai", api_key="test")
            
            assert interface == mock_instance
            mock_openai.assert_called_once_with(api_key="test")
    
    def test_create_unknown_provider(self):
        """Test creating interface with unknown provider."""
        with pytest.raises(ValueError, match="Unknown LLM provider: unknown"):
            LLMRegistry.create_interface("unknown")
    
    def test_register_custom_interface(self):
        """Test registering custom interface."""
        class CustomInterface(BaseLLMInterface):
            pass
        
        LLMRegistry.register_interface("custom", CustomInterface)
        
        assert "custom" in LLMRegistry.available_providers()
        
        interface = LLMRegistry.create_interface("custom", model_name="custom-model")
        assert isinstance(interface, CustomInterface)
    
    def test_available_providers(self):
        """Test listing available providers."""
        providers = LLMRegistry.available_providers()
        
        assert "cohere" in providers
        assert "openai" in providers
        assert isinstance(providers, list)


class TestCreateLLMInterface:
    """Test convenience function for creating LLM interfaces."""
    
    def test_create_default_interface(self):
        """Test creating default (Cohere) interface."""
        with patch('reservoirpy.cognitive.llm_interface.LLMRegistry.create_interface') as mock_create:
            mock_interface = Mock()
            mock_create.return_value = mock_interface
            
            interface = create_llm_interface()
            
            assert interface == mock_interface
            mock_create.assert_called_once_with("cohere")
    
    def test_create_openai_interface(self):
        """Test creating OpenAI interface."""
        with patch('reservoirpy.cognitive.llm_interface.LLMRegistry.create_interface') as mock_create:
            mock_interface = Mock()
            mock_create.return_value = mock_interface
            
            interface = create_llm_interface("openai", api_key="test")
            
            assert interface == mock_interface
            mock_create.assert_called_once_with("openai", api_key="test")