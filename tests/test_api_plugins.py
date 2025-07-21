import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from fastapi.testclient import TestClient

from src.bci_compression.api import app


def test_plugins_endpoint():
    client = TestClient(app)
    response = client.get("/plugins")
    assert response.status_code == 200
    plugins = response.json()
    assert isinstance(plugins, list)
    assert all('name' in p and 'doc' in p for p in plugins)
    # Optionally check for known plugin names
    known = {p['name'] for p in plugins}
    assert 'dummy_lz' in known
    assert 'dictionary' in known

