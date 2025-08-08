#!/usr/bin/env python3
"""
FastAPI Telemetry Server for BCI Compression Toolkit

Provides REST + WebSocket endpoints for real-time compression monitoring,
file compression/decompression, and benchmarking via web dashboard.
"""

import json
import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

# FastAPI imports - these may not be available in all environments
try:
    from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Import core compression functionality
from bci_compression.benchmarking.metrics import BenchmarkMetrics
from bci_compression.core import create_compressor
from bci_compression.plugins import get_plugin, list_plugins

try:
    # Check if mobile module is available
    import bci_compression.mobile.mobile_compressor
    MOBILE_AVAILABLE = True
except ImportError:
    MOBILE_AVAILABLE = False

if not FASTAPI_AVAILABLE:
    print("FastAPI not available. Install with: pip install fastapi uvicorn")
    exit(1)

# Create logs directory if it doesn't exist
LOGS_DIR = Path(__file__).parent.parent / "logs"
GUI_ARTIFACTS_DIR = LOGS_DIR / "gui_artifacts"
GUI_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "gui_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="BCI Compression Telemetry Server", version="1.0.0")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.active_sessions: Dict[str, Dict] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                disconnected.append(connection)

        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)


manager = ConnectionManager()


# Pydantic models
class PluginInfo(BaseModel):
    name: str
    modes: List[str]
    capabilities: Dict[str, Any]
    is_lossless: bool
    is_lossy: bool
    supports_streaming: bool
    supports_gpu: bool


class CompressRequest(BaseModel):
    plugin: str
    mode: Optional[str] = "balanced"
    quality_level: Optional[float] = 0.8
    options: Optional[Dict[str, Any]] = {}


class BenchmarkRequest(BaseModel):
    dataset: str = "synthetic"  # synthetic or file
    plugin: str
    duration_s: int = 10
    channels: int = 32
    sample_rate: int = 30000
    metrics: List[str] = ["compression_ratio", "latency_ms", "snr_db", "psnr_db"]


class MetricsTelemetry(BaseModel):
    timestamp: float
    session_id: str
    compression_ratio: float
    latency_ms: float
    snr_db: Optional[float] = None
    psnr_db: Optional[float] = None
    spectral_coherence_error: Optional[float] = None
    spike_f1: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    gpu_available: bool = False


# GPU detection
def detect_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


GPU_AVAILABLE = detect_gpu()


# Helper functions
def generate_synthetic_data(channels: int, samples: int, sample_rate: int = 30000) -> np.ndarray:
    """Generate synthetic neural data for testing."""
    # Create multi-frequency neural-like signal
    t = np.linspace(0, samples / sample_rate, samples)
    data = np.zeros((channels, samples))

    for ch in range(channels):
        # Alpha band (8-12 Hz) + noise
        alpha = 0.5 * np.sin(2 * np.pi * (8 + ch * 0.1) * t)
        # Beta band (13-30 Hz)
        beta = 0.3 * np.sin(2 * np.pi * (20 + ch * 0.2) * t)
        # Gamma band (30-100 Hz)
        gamma = 0.2 * np.sin(2 * np.pi * (60 + ch * 0.5) * t)
        # White noise
        noise = 0.1 * np.random.randn(samples)

        data[ch] = alpha + beta + gamma + noise

    return data.astype(np.float32)


def save_artifact(data: bytes, metadata: dict, session_id: str) -> Dict[str, str]:
    """Save compressed artifact and metadata to disk."""
    session_dir = GUI_ARTIFACTS_DIR / session_id
    session_dir.mkdir(exist_ok=True)

    # Save compressed data
    compressed_path = session_dir / "compressed.npz"
    metadata_path = session_dir / "metadata.json"

    # Use npz format for consistency
    np.savez_compressed(compressed_path, payload=np.frombuffer(data, dtype=np.uint8), **metadata)

    # Save metadata separately for easy inspection
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    return {
        "compressed_path": str(compressed_path),
        "metadata_path": str(metadata_path)
    }


def load_artifact(artifact_path: str) -> Tuple[bytes, dict]:
    """Load compressed artifact from disk."""
    npz_file = np.load(artifact_path)
    payload = npz_file['payload'].tobytes()

    # Extract metadata from npz file (all non-payload keys)
    metadata = {key: npz_file[key].item() if npz_file[key].shape == () else npz_file[key]
                for key in npz_file.files if key != 'payload'}

    return payload, metadata


# REST API endpoints
@app.get("/api/plugins", response_model=List[PluginInfo])
async def get_plugins():
    """List available compression plugins with capabilities."""
    try:
        plugin_names = list_plugins(discover=True)
        plugins = []

        for name in plugin_names:
            try:
                plugin_cls = get_plugin(name, discover=True)
                capabilities = getattr(plugin_cls, 'capabilities', lambda: {})()

                # Determine plugin characteristics
                modes = capabilities.get('modes', ['balanced'])
                if isinstance(modes, str):
                    modes = [modes]

                # Basic capability detection
                is_lossless = 'lossless' in capabilities.get('compression_types', ['lossless'])
                is_lossy = 'lossy' in capabilities.get('compression_types', ['lossy'])
                supports_streaming = 'streaming' in capabilities.get('features', [])
                supports_gpu = 'gpu' in capabilities.get('features', [])

                # Fallback for plugins without explicit capabilities
                if not (is_lossless or is_lossy):
                    if 'lz' in name.lower() or 'arithmetic' in name.lower():
                        is_lossless = True
                    else:
                        is_lossy = True

                plugins.append(PluginInfo(
                    name=name,
                    modes=modes,
                    capabilities=capabilities,
                    is_lossless=is_lossless,
                    is_lossy=is_lossy,
                    supports_streaming=supports_streaming,
                    supports_gpu=supports_gpu
                ))
            except Exception as e:
                logger.warning(f"Failed to get info for plugin {name}: {e}")
                # Add basic plugin info as fallback
                plugins.append(PluginInfo(
                    name=name,
                    modes=["balanced"],
                    capabilities={},
                    is_lossless=True,
                    is_lossy=False,
                    supports_streaming=False,
                    supports_gpu=False
                ))

        return plugins
    except Exception as e:
        logger.error(f"Error listing plugins: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/compress")
async def compress_file(file: UploadFile = File(...), request: str = "{}"):
    """Compress uploaded file and return metrics."""
    try:
        # Parse request JSON
        req_data = json.loads(request)
        compress_req = CompressRequest(**req_data)

        # Generate session ID
        session_id = str(uuid.uuid4())

        # Read uploaded file
        content = await file.read()

        # Determine file format and load data
        if file.filename.endswith('.npy'):
            with tempfile.NamedTemporaryFile() as tmp:
                tmp.write(content)
                tmp.flush()
                data = np.load(tmp.name)
        elif file.filename.endswith('.h5'):
            with tempfile.NamedTemporaryFile() as tmp:
                tmp.write(content)
                tmp.flush()
                with h5py.File(tmp.name, 'r') as h5f:
                    # Try common dataset names
                    if 'data' in h5f:
                        data = h5f['data'][:]
                    elif 'neural_data' in h5f:
                        data = h5f['neural_data'][:]
                    else:
                        # Take first dataset
                        data = h5f[list(h5f.keys())[0]][:]
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use .npy or .h5")

        # Create compressor
        compressor = create_compressor(
            compress_req.plugin,
            config=None  # Use default config for now
        )

        # Perform compression
        start_time = time.time()
        if hasattr(compressor, 'compress'):
            if isinstance(data, np.ndarray) and len(data.shape) == 2:
                compressed, compression_meta = compressor.compress(data)
            else:
                # Reshape data if needed
                if len(data.shape) == 1:
                    data = data.reshape(1, -1)
                compressed, compression_meta = compressor.compress(data)
        else:
            raise HTTPException(status_code=400, detail=f"Plugin {compress_req.plugin} does not support compression")

        compression_time = time.time() - start_time

        # Calculate metrics
        original_size = data.nbytes
        if isinstance(compressed, (bytes, bytearray)):
            compressed_size = len(compressed)
            compressed_bytes = bytes(compressed)
        else:
            # Handle other formats (e.g., numpy arrays)
            compressed_bytes = np.array(compressed).tobytes()
            compressed_size = len(compressed_bytes)

        # Basic metrics
        compression_ratio = BenchmarkMetrics.compression_ratio(original_size, compressed_size)
        latency_ms = BenchmarkMetrics.processing_latency(start_time, start_time + compression_time)

        # Quality metrics (if lossless, perfect reconstruction)
        snr_db = None
        psnr_db = None

        if hasattr(compressor, 'decompress'):
            try:
                if isinstance(compressed, (bytes, bytearray)):
                    reconstructed = compressor.decompress(compressed)
                else:
                    reconstructed = compressor.decompress(compressed_bytes)

                if isinstance(reconstructed, np.ndarray) and reconstructed.shape == data.shape:
                    snr_db = BenchmarkMetrics.snr(data, reconstructed)
                    psnr_db = BenchmarkMetrics.psnr(data, reconstructed, max_value=np.max(np.abs(data)))
            except Exception as e:
                logger.warning(f"Failed to compute quality metrics: {e}")

        # Prepare metadata
        metadata = {
            "session_id": session_id,
            "plugin": compress_req.plugin,
            "mode": compress_req.mode,
            "original_shape": data.shape,
            "original_dtype": str(data.dtype),
            "original_size_bytes": original_size,
            "compressed_size_bytes": compressed_size,
            "compression_ratio": compression_ratio,
            "compression_time_ms": latency_ms,
            "snr_db": snr_db,
            "psnr_db": psnr_db,
            "compression_metadata": compression_meta if isinstance(compression_meta, dict) else {}
        }

        # Save artifacts
        paths = save_artifact(compressed_bytes, metadata, session_id)

        # Return response
        return {
            "session_id": session_id,
            "metrics": {
                "overall_compression_ratio": compression_ratio,
                "latency_ms": latency_ms,
                "snr_db": snr_db,
                "psnr_db": psnr_db,
                "original_size_bytes": original_size,
                "compressed_size_bytes": compressed_size
            },
            "artifact_path": paths["compressed_path"],
            "metadata_path": paths["metadata_path"]
        }

    except Exception as e:
        logger.error(f"Compression error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "gpu_available": GPU_AVAILABLE,
        "mobile_available": MOBILE_AVAILABLE,
        "plugins_count": len(list_plugins()),
        "active_sessions": len(manager.active_sessions)
    }


if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    except ImportError:
        print("Uvicorn not available. Install with: pip install uvicorn")
        print("Then run: uvicorn scripts.telemetry_server:app --reload --host 0.0.0.0 --port 8000")
