"""
Unit tests for PipelineConfigManager integration and runtime updates.
"""
import os
import pytest
from src.config.pipeline_config_manager import PipelineConfigManager


def test_load_default_config():
    manager = PipelineConfigManager()
    config = manager.get_config()
    assert isinstance(config, dict)
    assert "num_channels" in config
    assert "compression_type" in config


def test_update_config_runtime(tmp_path):
    config_name = "test_config.json"
    config_path = tmp_path / config_name
    manager = PipelineConfigManager(config_name=config_name)
    updates = {"num_channels": 128, "quality_level": 0.8}
    manager.update_config(updates)
    config = manager.get_config()
    assert config["num_channels"] == 128
    assert config["quality_level"] == 0.8
    # Check file written
    assert os.path.exists(manager.config_path)
    with open(manager.config_path, "r", encoding="utf-8") as f:
        data = f.read()
        assert "128" in data
        assert "0.8" in data


def test_update_config_invalid():
    manager = PipelineConfigManager()
    with pytest.raises(Exception):
        manager.update_config(None)  # Should raise TypeError
