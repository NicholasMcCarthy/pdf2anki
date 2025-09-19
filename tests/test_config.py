"""Tests for configuration management."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from pdf2anki.config import Config


def test_default_config():
    """Test that default configuration loads successfully."""
    config = Config()
    
    assert config.project.name == "pdf2anki"
    assert config.llm.provider.value == "openai"
    assert config.llm.model == "gpt-4-1106-preview"
    assert config.llm.temperature == 0.0
    assert config.ids.strategy.value == "content_hash"
    assert config.anki.deck_name == "PDF2Anki"


def test_config_from_yaml():
    """Test loading configuration from YAML file."""
    config_data = {
        "project": {
            "name": "Test Project",
            "author": "Test Author"
        },
        "llm": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.5
        },
        "anki": {
            "deck_name": "Test Deck"
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        yaml_path = f.name
    
    try:
        config = Config.from_yaml(yaml_path)
        
        assert config.project.name == "Test Project"
        assert config.project.author == "Test Author"
        assert config.llm.model == "gpt-3.5-turbo"
        assert config.llm.temperature == 0.5
        assert config.anki.deck_name == "Test Deck"
        
    finally:
        os.unlink(yaml_path)


def test_config_to_yaml():
    """Test saving configuration to YAML file."""
    config = Config()
    config.project.name = "Test Export"
    config.llm.temperature = 0.7
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml_path = f.name
    
    try:
        config.to_yaml(yaml_path)
        
        # Load it back and verify
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        assert data['project']['name'] == "Test Export"
        assert data['llm']['temperature'] == 0.7
        
    finally:
        os.unlink(yaml_path)


def test_env_var_substitution():
    """Test environment variable substitution in API key."""
    # Set test environment variable
    test_key = "test-api-key-12345"
    os.environ["TEST_API_KEY"] = test_key
    
    try:
        config_data = {
            "llm": {
                "api_key": "${TEST_API_KEY}"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            yaml_path = f.name
        
        try:
            config = Config.from_yaml(yaml_path)
            assert config.llm.api_key == test_key
            
        finally:
            os.unlink(yaml_path)
            
    finally:
        del os.environ["TEST_API_KEY"]


def test_workspace_creation():
    """Test workspace directory creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config()
        config.output.workspace = Path(tmpdir) / "test_workspace"
        config.output.media_path = config.output.workspace / "media"
        
        # Verify directories don't exist yet
        assert not config.output.workspace.exists()
        assert not config.output.media_path.exists()
        
        # Create workspace
        config.create_workspace()
        
        # Verify directories were created
        assert config.output.workspace.exists()
        assert config.output.media_path.exists()


def test_strategy_config_validation():
    """Test strategy configuration validation."""
    config = Config()
    
    # Test default strategy configs
    assert config.strategies.key_points.enabled is True
    assert config.strategies.cloze_definitions.enabled is True
    assert config.strategies.figure_based.enabled is True
    
    # Test strategy params
    assert isinstance(config.strategies.key_points.params, dict)
    assert config.strategies.key_points.template_version == "1.0"


def test_invalid_enum_values():
    """Test handling of invalid enum values."""
    with pytest.raises(ValueError):
        config_data = {
            "llm": {
                "provider": "invalid_provider"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            yaml_path = f.name
        
        try:
            Config.from_yaml(yaml_path)
        finally:
            os.unlink(yaml_path)