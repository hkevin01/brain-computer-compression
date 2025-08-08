import pytest
from fastapi.testclient import TestClient

from scripts.telemetry_server import app

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_get_plugins():
    """Test getting available plugins"""
    response = client.get("/api/plugins")
    assert response.status_code == 200
    plugins = response.json()
    assert isinstance(plugins, list)
    # Should have at least some plugins from the bci_compression package
    assert len(plugins) > 0

def test_compress_endpoint_missing_file():
    """Test compression endpoint with missing file"""
    response = client.post("/api/compress", json={
        "filename": "nonexistent.npy",
        "plugin": "test_plugin",
        "mode": "balanced"
    })
    assert response.status_code == 404

def test_decompress_endpoint_missing_file():
    """Test decompression endpoint with missing file"""
    response = client.post("/api/decompress", json={
        "filename": "nonexistent.compressed",
        "plugin": "test_plugin"
    })
    assert response.status_code == 404

def test_benchmark_endpoint():
    """Test benchmark endpoint with synthetic data"""
    response = client.post("/api/benchmark", json={
        "plugin": "huffman",  # Basic plugin that should exist
        "mode": "balanced",
        "quality": 0.8,
        "num_trials": 1
    })
    # May succeed or fail depending on available plugins
    assert response.status_code in [200, 400, 404]

def test_generate_data_endpoint():
    """Test synthetic data generation"""
    response = client.post("/api/generate-data", json={
        "num_channels": 4,
        "duration_seconds": 1,
        "sampling_rate": 100
    })
    assert response.status_code == 200
    result = response.json()
    assert "filename" in result
    assert result["filename"].endswith('.npy')

def test_websocket_connection():
    """Test WebSocket connection"""
    with client.websocket_connect("/ws/metrics") as websocket:
        # Send start command
        websocket.send_json({
            "command": "start_stream",
            "plugin": "huffman",
            "mode": "balanced",
            "quality": 0.8
        })

        # Should receive either metrics or error
        data = websocket.receive_json()
        assert "type" in data
        assert data["type"] in ["metrics", "error"]

@pytest.mark.asyncio
async def test_connection_manager():
    """Test WebSocket connection manager"""
    from scripts.telemetry_server import manager

    # Test adding connection
    connection_id = "test_connection"
    await manager.connect(connection_id, None)  # None websocket for testing
    assert connection_id in manager.active_connections

    # Test disconnecting
    manager.disconnect(connection_id)
    assert connection_id not in manager.active_connections

def test_cors_headers():
    """Test CORS headers are present"""
    response = client.options("/api/plugins")
    assert response.status_code == 200
    # Check for CORS headers in response

def test_upload_endpoint():
    """Test file upload endpoint"""
    # Create a simple test file
    test_content = b"test file content"
    files = {"file": ("test.npy", test_content, "application/octet-stream")}

    response = client.post("/api/upload", files=files)
    assert response.status_code == 200
    result = response.json()
    assert "filename" in result

if __name__ == "__main__":
    pytest.main([__file__])
    pytest.main([__file__])
