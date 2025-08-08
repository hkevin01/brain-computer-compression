"""
FastAPI Telemetry Server for BCI Compression Toolkit

Provides REST + WebSocket endpoints for real-time compression monitoring,
file compression/decompression, and benchmarking via web dashboard.

Endpoints:
- GET /api/plugins → list available compression plugins
- POST /api/compress → compress file or synthetic data
- POST /api/decompress → decompress previously compressed data
- POST /api/benchmark → run benchmarks on compression algorithms
- POST /api/upload → upload file for processing
- GET /api/download/{filename} → download artifact files
- POST /api/generate-data → generate synthetic neural data
- GET /api/logs → get server logs
- WebSocket /ws/metrics → real-time telemetry stream
"""

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

# FastAPI imports
try:
    from fastapi import (
        FastAPI, File, HTTPException, UploadFile, WebSocket,
        WebSocketDisconnect, Form, Query
    )
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not available. Install with: pip install fastapi uvicorn python-multipart")
    exit(1)

# Import core compression functionality
try:
    from bci_compression.core import create_compressor
    from bci_compression.plugins import get_plugin, list_plugins
    BCI_AVAILABLE = True
except ImportError:
    BCI_AVAILABLE = False
    print("BCI compression package not found. Ensure src/bci_compression is available.")

try:
    from bci_compression.benchmarking.metrics import BenchmarkMetrics
    BENCHMARKING_AVAILABLE = True
except ImportError:
    BENCHMARKING_AVAILABLE = False

try:
    from bci_compression.mobile.mobile_compressor import MobileBCICompressor
    MOBILE_AVAILABLE = True
except ImportError:
    MOBILE_AVAILABLE = False

# Create directories
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

#!/usr/bin/env python3
"""
FastAPI Telemetry Server for BCI Compression Toolkit

Complete production-ready backend with all required endpoints for the GUI.
"""

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

# FastAPI imports
try:
    from fastapi import (
        FastAPI, File, HTTPException, UploadFile, WebSocket,
        WebSocketDisconnect, Form, Query
    )
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not available. Install with: pip install fastapi uvicorn python-multipart")
    exit(1)

# BCI compression imports
BCI_AVAILABLE = False
try:
    from bci_compression.core import create_compressor
    from bci_compression.plugins import get_plugin, list_plugins
    BCI_AVAILABLE = True
except ImportError:
    pass

BENCHMARKING_AVAILABLE = False
try:
    from bci_compression.benchmarking.metrics import BenchmarkMetrics
    BENCHMARKING_AVAILABLE = True
except ImportError:
    pass

MOBILE_AVAILABLE = False
try:
    from bci_compression.mobile.mobile_compressor import MobileBCICompressor
    MOBILE_AVAILABLE = True
except ImportError:
    pass

# Create directories
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

# FastAPI app
app = FastAPI(
    title="BCI Compression Telemetry Server",
    version="1.0.0",
    description="Real-time monitoring and control API for neural data compression"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:3000", "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.streaming_sessions: Dict[str, Dict] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Remaining: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        if not self.active_connections:
            return

        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                disconnected.append(connection)

        for connection in disconnected:
            self.disconnect(connection)

manager = ConnectionManager()

# Pydantic models
class PluginInfo(BaseModel):
    name: str
    modes: List[str]
    lossless: bool
    lossy: bool
    streaming: bool
    gpu_accel: bool
    version: str

# GPU detection
def detect_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        try:
            import cupy
            return True
        except ImportError:
            return False

GPU_AVAILABLE = detect_gpu()

# Helper functions
def generate_synthetic_neural_data(channels: int, samples: int, sample_rate: int = 30000) -> np.ndarray:
    """Generate synthetic neural data."""
    logger.info(f"Generating synthetic data: {channels} channels, {samples} samples @ {sample_rate}Hz")

    t = np.linspace(0, samples / sample_rate, samples)
    data = np.zeros((channels, samples))

    for ch in range(channels):
        # Neural oscillations
        alpha = 0.5 * np.sin(2 * np.pi * (8 + ch * 0.1) * t)
        beta = 0.3 * np.sin(2 * np.pi * (20 + ch * 0.2) * t)
        gamma = 0.2 * np.sin(2 * np.pi * (60 + ch * 0.5) * t)

        # Spikes
        spike_times = np.random.choice(samples, size=int(samples * 0.001), replace=False)
        spikes = np.zeros(samples)
        spikes[spike_times] = np.random.normal(0, 2, len(spike_times))

        # Noise
        noise = 0.1 * np.random.randn(samples)

        data[ch] = alpha + beta + gamma + spikes + noise

    return data.astype(np.float32)

def get_fallback_plugins() -> List[str]:
    """Fallback plugin list."""
    return ["neural_lz", "neural_arithmetic", "perceptual_quantizer", "predictive_coding"]

def create_fallback_compressor(name: str):
    """Fallback compressor for testing."""
    class FallbackCompressor:
        def __init__(self, name):
            self.name = name

        def compress(self, data):
            compressed = data.flatten().astype(np.float16).tobytes()
            metadata = {"original_shape": data.shape, "dtype": str(data.dtype)}
            return compressed, metadata

        def decompress(self, compressed_data, metadata=None):
            if metadata and "original_shape" in metadata:
                shape = metadata["original_shape"]
                data = np.frombuffer(compressed_data, dtype=np.float16).astype(np.float32)
                return data.reshape(shape)
            else:
                return np.frombuffer(compressed_data, dtype=np.float16)

    return FallbackCompressor(name)

def compute_metrics(original_data: np.ndarray, compressed_data: bytes,
                   decompressed_data: np.ndarray = None, compression_time: float = 0) -> Dict:
    """Compute compression metrics."""
    metrics = {
        "overall_compression_ratio": original_data.nbytes / len(compressed_data),
        "compression_time": compression_time * 1000,
        "original_size_bytes": original_data.nbytes,
        "compressed_size_bytes": len(compressed_data),
        "gpu_available": GPU_AVAILABLE
    }

    if decompressed_data is not None and decompressed_data.shape == original_data.shape:
        try:
            signal_power = np.mean(original_data ** 2)
            noise_power = np.mean((original_data - decompressed_data) ** 2)
            if noise_power > 0:
                metrics["snr_db"] = 10 * np.log10(signal_power / noise_power)
                max_value = np.max(np.abs(original_data))
                metrics["psnr_db"] = 20 * np.log10(max_value) - 10 * np.log10(noise_power)
            else:
                metrics["snr_db"] = float('inf')
                metrics["psnr_db"] = float('inf')
        except Exception as e:
            logger.warning(f"Failed to compute quality metrics: {e}")

    return metrics

# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        plugin_count = len(list_plugins()) if BCI_AVAILABLE else len(get_fallback_plugins())
    except:
        plugin_count = 0

    return {
        "status": "healthy",
        "gpu_available": GPU_AVAILABLE,
        "mobile_available": MOBILE_AVAILABLE,
        "bci_available": BCI_AVAILABLE,
        "plugins_count": plugin_count,
        "active_connections": len(manager.active_connections)
    }

@app.get("/api/plugins", response_model=List[PluginInfo])
async def get_plugins():
    """Get available compression plugins."""
    try:
        if BCI_AVAILABLE:
            plugin_names = list_plugins()
        else:
            plugin_names = get_fallback_plugins()

        plugins = []

        for name in plugin_names:
            try:
                if BCI_AVAILABLE:
                    plugin_cls = get_plugin(name)
                    capabilities = getattr(plugin_cls, 'capabilities', lambda: {})()
                else:
                    capabilities = {}

                modes = capabilities.get('modes', ['fast', 'balanced', 'quality'])
                if isinstance(modes, str):
                    modes = [modes]

                is_lossless = (
                    'lossless' in capabilities.get('compression_types', []) or
                    'lz' in name.lower() or 'arithmetic' in name.lower()
                )
                is_lossy = (
                    'lossy' in capabilities.get('compression_types', []) or
                    'perceptual' in name.lower() or 'quantizer' in name.lower()
                )

                if not (is_lossless or is_lossy):
                    is_lossless = True

                plugins.append(PluginInfo(
                    name=name,
                    modes=modes,
                    lossless=is_lossless,
                    lossy=is_lossy,
                    streaming='streaming' in capabilities.get('features', []),
                    gpu_accel=GPU_AVAILABLE,
                    version=capabilities.get('version', '1.0.0')
                ))

            except Exception as e:
                logger.warning(f"Failed to get info for plugin {name}: {e}")
                plugins.append(PluginInfo(
                    name=name,
                    modes=["balanced"],
                    lossless=True,
                    lossy=False,
                    streaming=False,
                    gpu_accel=False,
                    version="1.0.0"
                ))

        logger.info(f"Found {len(plugins)} plugins")
        return plugins

    except Exception as e:
        logger.error(f"Error listing plugins: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list plugins: {str(e)}")

@app.post("/api/compress")
async def compress_data(
    file: Optional[UploadFile] = File(None),
    filePath: Optional[str] = Form(None),
    plugin: str = Form(...),
    mode: str = Form("balanced"),
    quality_level: float = Form(0.8),
    options: str = Form("{}")
):
    """Compress file data."""
    try:
        options_dict = json.loads(options) if options else {}
        session_id = str(uuid.uuid4())
        logger.info(f"Starting compression session {session_id} with plugin {plugin}")

        # Load data
        if file:
            content = await file.read()
            if file.filename.endswith('.npy'):
                with tempfile.NamedTemporaryFile() as tmp:
                    tmp.write(content)
                    tmp.flush()
                    data = np.load(tmp.name)
            elif file.filename.endswith(('.h5', '.hdf5')):
                with tempfile.NamedTemporaryFile() as tmp:
                    tmp.write(content)
                    tmp.flush()
                    with h5py.File(tmp.name, 'r') as f:
                        first_key = list(f.keys())[0]
                        data = f[first_key][:]
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format")
        elif filePath:
            if filePath.endswith('.npy'):
                data = np.load(filePath)
            else:
                raise HTTPException(status_code=400, detail="File path loading only supports .npy")
        else:
            raise HTTPException(status_code=400, detail="Either file or filePath must be provided")

        # Ensure 2D data
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        elif len(data.shape) > 2:
            data = data.reshape(-1, data.shape[-1])

        logger.info(f"Loaded data shape: {data.shape}, dtype: {data.dtype}")

        # Create compressor
        start_time = time.time()
        if BCI_AVAILABLE:
            try:
                compressor = create_compressor(plugin)
            except:
                compressor = create_fallback_compressor(plugin)
        else:
            compressor = create_fallback_compressor(plugin)

        # Compress
        if hasattr(compressor, 'compress'):
            compressed_data, comp_metadata = compressor.compress(data)
        else:
            compressed_data = data.astype(np.float16).tobytes()
            comp_metadata = {"original_shape": data.shape, "dtype": str(data.dtype)}

        compression_time = time.time() - start_time

        # Convert to bytes
        if not isinstance(compressed_data, (bytes, bytearray)):
            if isinstance(compressed_data, np.ndarray):
                compressed_bytes = compressed_data.tobytes()
            else:
                compressed_bytes = bytes(compressed_data)
        else:
            compressed_bytes = bytes(compressed_data)

        # Decompress for quality metrics
        decompressed_data = None
        if hasattr(compressor, 'decompress'):
            try:
                decompressed_data = compressor.decompress(compressed_data)
                if hasattr(decompressed_data, 'shape') and decompressed_data.shape != data.shape:
                    decompressed_data = None
            except Exception as e:
                logger.warning(f"Failed to decompress for quality metrics: {e}")

        # Compute metrics
        metrics = compute_metrics(data, compressed_bytes, decompressed_data, compression_time)

        # Save artifact
        session_dir = GUI_ARTIFACTS_DIR / session_id
        session_dir.mkdir(exist_ok=True)

        artifact_path = session_dir / "compressed.npz"
        metadata_path = session_dir / "metadata.json"

        np.savez_compressed(
            artifact_path,
            payload=np.frombuffer(compressed_bytes, dtype=np.uint8),
            original_shape=data.shape,
            original_dtype=str(data.dtype),
            plugin=plugin,
            mode=mode,
            quality_level=quality_level
        )

        full_metadata = {
            "session_id": session_id,
            "plugin": plugin,
            "mode": mode,
            "quality_level": quality_level,
            "options": options_dict,
            "original_shape": data.shape,
            "original_dtype": str(data.dtype),
            "compression_metadata": comp_metadata if isinstance(comp_metadata, dict) else {},
            **metrics
        }

        with open(metadata_path, 'w') as f:
            json.dump(full_metadata, f, indent=2, default=str)

        return {
            "session_id": session_id,
            "metrics": metrics,
            "artifactPath": str(artifact_path),
            "metadataPath": str(metadata_path)
        }

    except Exception as e:
        logger.error(f"Compression error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/decompress")
async def decompress_data(artifactPath: str):
    """Decompress previously compressed data."""
    try:
        if not os.path.exists(artifactPath):
            raise HTTPException(status_code=404, detail="Artifact not found")

        with np.load(artifactPath) as data:
            payload = data['payload'].tobytes()
            original_shape = tuple(data['original_shape'])
            plugin_name = str(data['plugin'])

        logger.info(f"Decompressing artifact with plugin {plugin_name}")

        # Create compressor
        if BCI_AVAILABLE:
            try:
                compressor = create_compressor(plugin_name)
            except:
                compressor = create_fallback_compressor(plugin_name)
        else:
            compressor = create_fallback_compressor(plugin_name)

        # Decompress
        start_time = time.time()
        if hasattr(compressor, 'decompress'):
            reconstructed_data = compressor.decompress(payload)
        else:
            reconstructed_data = np.frombuffer(payload, dtype=np.float16).astype(np.float32)
            reconstructed_data = reconstructed_data.reshape(original_shape)

        decompression_time = time.time() - start_time

        # Save reconstructed data
        session_id = Path(artifactPath).parent.name
        session_dir = GUI_ARTIFACTS_DIR / session_id
        reconstructed_path = session_dir / "reconstructed.npy"
        np.save(reconstructed_path, reconstructed_data)

        metrics = {
            "decompression_time_ms": decompression_time * 1000,
            "reconstructed_shape": reconstructed_data.shape,
            "reconstructed_dtype": str(reconstructed_data.dtype)
        }

        return {
            "reconstructedPath": str(reconstructed_path),
            "metrics": metrics
        }

    except Exception as e:
        logger.error(f"Decompression error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/benchmark")
async def run_benchmark(
    dataset: str = "synthetic",
    filePath: Optional[str] = None,
    plugin: str = "neural_lz",
    duration_s: int = 10,
    channels: int = 32,
    sample_rate: int = 30000,
    metrics: List[str] = ["compression_ratio", "latency_ms", "snr_db"]
):
    """Run compression benchmark."""
    try:
        logger.info(f"Running benchmark for plugin {plugin}")

        # Generate or load test data
        if dataset == "synthetic":
            data = generate_synthetic_neural_data(channels, sample_rate * duration_s, sample_rate)
        elif dataset == "file" and filePath:
            data = np.load(filePath)
        else:
            raise HTTPException(status_code=400, detail="Invalid dataset specification")

        # Create compressor
        if BCI_AVAILABLE:
            try:
                compressor = create_compressor(plugin)
            except:
                compressor = create_fallback_compressor(plugin)
        else:
            compressor = create_fallback_compressor(plugin)

        # Run benchmark trials
        results = []
        num_trials = 3

        for trial in range(num_trials):
            start_time = time.time()

            # Compression
            compressed_data, comp_metadata = compressor.compress(data)
            compression_time = time.time() - start_time

            # Convert to bytes
            if not isinstance(compressed_data, (bytes, bytearray)):
                compressed_bytes = compressed_data.tobytes() if hasattr(compressed_data, 'tobytes') else bytes(compressed_data)
            else:
                compressed_bytes = bytes(compressed_data)

            # Decompression
            decompressed_data = None
            decompression_time = 0
            if hasattr(compressor, 'decompress'):
                start_time = time.time()
                try:
                    decompressed_data = compressor.decompress(compressed_data)
                    decompression_time = time.time() - start_time
                except:
                    pass

            # Compute metrics
            trial_metrics = compute_metrics(data, compressed_bytes, decompressed_data, compression_time)
            trial_metrics["decompression_time_ms"] = decompression_time * 1000
            trial_metrics["trial"] = trial

            results.append(trial_metrics)

        # Aggregate results
        aggregated = {}
        for metric in metrics:
            values = [r.get(metric, 0) for r in results if metric in r]
            if values:
                aggregated[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }

        return {
            "plugin": plugin,
            "dataset": dataset,
            "data_shape": data.shape,
            "trials": len(results),
            "aggregated_metrics": aggregated,
            "raw_results": results
        }

    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file for processing."""
    try:
        upload_dir = GUI_ARTIFACTS_DIR / "uploads"
        upload_dir.mkdir(exist_ok=True)

        file_id = str(uuid.uuid4())
        file_path = upload_dir / f"{file_id}_{file.filename}"

        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        logger.info(f"Uploaded file {file.filename} as {file_path}")

        return {
            "file_id": file_id,
            "filename": file.filename,
            "file_path": str(file_path),
            "size_bytes": len(content)
        }

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """Download an artifact file."""
    try:
        file_path = GUI_ARTIFACTS_DIR / filename

        if file_path.is_dir():
            for candidate in ["compressed.npz", "metadata.json", "reconstructed.npy"]:
                candidate_path = file_path / candidate
                if candidate_path.exists():
                    file_path = candidate_path
                    break

        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        try:
            file_path.resolve().relative_to(GUI_ARTIFACTS_DIR.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied")

        return FileResponse(
            path=str(file_path),
            filename=file_path.name,
            media_type='application/octet-stream'
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-data")
async def generate_data(
    channels: int = 32,
    samples: int = 30000,
    sample_rate: int = 30000,
    noise_level: float = 0.1
):
    """Generate synthetic neural data."""
    try:
        data = generate_synthetic_neural_data(channels, samples, sample_rate)

        data_dir = GUI_ARTIFACTS_DIR / "generated"
        data_dir.mkdir(exist_ok=True)

        file_id = str(uuid.uuid4())
        file_path = data_dir / f"synthetic_{file_id}.npy"
        np.save(file_path, data)

        logger.info(f"Generated synthetic data: {data.shape} saved to {file_path}")

        return {
            "file_id": file_id,
            "file_path": str(file_path),
            "shape": data.shape,
            "dtype": str(data.dtype),
            "size_bytes": data.nbytes
        }

    except Exception as e:
        logger.error(f"Data generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/logs")
async def get_logs(tail: int = Query(100, description="Number of lines to tail")):
    """Get server logs."""
    try:
        log_file = LOGS_DIR / "gui_server.log"
        if not log_file.exists():
            return {"logs": []}

        with open(log_file, 'r') as f:
            lines = f.readlines()

        recent_lines = lines[-tail:] if len(lines) > tail else lines

        return {
            "logs": [line.strip() for line in recent_lines],
            "total_lines": len(lines),
            "tail": tail
        }

    except Exception as e:
        logger.error(f"Logs error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time metrics
@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    """WebSocket endpoint for real-time telemetry stream."""
    await manager.connect(websocket)

    try:
        # Start metrics streaming task in background
        metrics_task = asyncio.create_task(metrics_streaming_task())

        # Keep connection alive
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)

                try:
                    data = json.loads(message)
                    if data.get("type") == "start_stream":
                        session_id = data.get("session_id", str(uuid.uuid4()))
                        config = data.get("config", {})
                        manager.streaming_sessions[session_id] = {
                            "config": config,
                            "active": True,
                            "start_time": time.time()
                        }
                        logger.info(f"Started streaming session {session_id}")
                    elif data.get("type") == "stop_stream":
                        session_id = data.get("session_id")
                        if session_id in manager.streaming_sessions:
                            manager.streaming_sessions[session_id]["active"] = False
                            logger.info(f"Stopped streaming session {session_id}")
                except json.JSONDecodeError:
                    pass

            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "ping", "timestamp": time.time()}))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        if 'metrics_task' in locals():
            metrics_task.cancel()
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)
        if 'metrics_task' in locals():
            metrics_task.cancel()

async def metrics_streaming_task():
    """Background task to generate and broadcast metrics."""
    logger.info("Started metrics streaming task")

    while manager.active_connections:
        try:
            current_time = time.time()

            # Simulate varying metrics
            base_ratio = 2.5 + 0.5 * np.sin(current_time * 0.1)
            noise = 0.1 * np.random.randn()

            metrics = {
                "type": "metrics",
                "timestamp": current_time,
                "compression_ratio": max(1.0, base_ratio + noise),
                "latency_ms": 0.5 + abs(0.2 * np.random.randn()),
                "snr_db": 25 + 5 * np.random.randn(),
                "psnr_db": 35 + 5 * np.random.randn(),
                "spectral_coherence_error": abs(0.05 * np.random.randn()),
                "gpu_available": GPU_AVAILABLE,
                "active_sessions": len(manager.streaming_sessions),
                "cpu_usage": min(100, max(0, 20 + 10 * np.random.randn()))
            }

            # Add spike detection metrics occasionally
            if np.random.random() < 0.3:
                metrics["spike_f1"] = 0.85 + 0.1 * np.random.randn()
                metrics["jitter_ms"] = abs(0.1 * np.random.randn())

            await manager.broadcast(metrics)
            await asyncio.sleep(0.1)  # 10 Hz

        except Exception as e:
            logger.error(f"Metrics streaming error: {e}")
            await asyncio.sleep(1.0)

        if not manager.active_connections:
            break

    logger.info("Metrics streaming task stopped - no active connections")

if __name__ == "__main__":
    try:
        import uvicorn
        logger.info("Starting BCI Compression Telemetry Server on http://0.0.0.0:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    except ImportError:
        print("Uvicorn not available. Install with: pip install uvicorn")
        print("Then run: uvicorn scripts.telemetry_server:app --reload --host 0.0.0.0 --port 8000")

# FastAPI app
app = FastAPI(
    title="BCI Compression Telemetry Server",
    version="1.0.0",
    description="Real-time monitoring and control API for neural data compression"
)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:3000", "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.streaming_sessions: Dict[str, Dict] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Remaining connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        if not self.active_connections:
            return

        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                disconnected.append(connection)

        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

manager = ConnectionManager()

# Pydantic models
class PluginInfo(BaseModel):
    name: str
    modes: List[str]
    lossless: bool
    lossy: bool
    streaming: bool
    gpu_accel: bool
    version: str

class CompressRequest(BaseModel):
    filePath: Optional[str] = None
    plugin: str
    mode: Optional[str] = "balanced"
    quality_level: Optional[float] = 0.8
    options: Optional[Dict[str, Any]] = {}

class DecompressRequest(BaseModel):
    artifactPath: str

class BenchmarkRequest(BaseModel):
    dataset: str = "synthetic"  # "synthetic" or "file"
    filePath: Optional[str] = None
    plugin: str
    duration_s: Optional[int] = 10
    channels: Optional[int] = 32
    sample_rate: Optional[int] = 30000
    metrics: Optional[List[str]] = ["compression_ratio", "latency_ms", "snr_db"]

class GenerateDataRequest(BaseModel):
    channels: int = 32
    samples: int = 30000
    sample_rate: int = 30000
    noise_level: float = 0.1

# GPU detection
def detect_gpu() -> bool:
    """Detect if GPU acceleration is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        try:
            import cupy
            return True
        except ImportError:
            return False

GPU_AVAILABLE = detect_gpu()

# Helper functions
def generate_synthetic_neural_data(channels: int, samples: int, sample_rate: int = 30000, noise_level: float = 0.1) -> np.ndarray:
    """Generate synthetic neural data with realistic characteristics."""
    logger.info(f"Generating synthetic data: {channels} channels, {samples} samples @ {sample_rate}Hz")

    t = np.linspace(0, samples / sample_rate, samples)
    data = np.zeros((channels, samples))

    for ch in range(channels):
        # Neural oscillations with channel-specific frequencies
        alpha = 0.5 * np.sin(2 * np.pi * (8 + ch * 0.1) * t)  # Alpha: 8-12 Hz
        beta = 0.3 * np.sin(2 * np.pi * (20 + ch * 0.2) * t)   # Beta: 13-30 Hz
        gamma = 0.2 * np.sin(2 * np.pi * (60 + ch * 0.5) * t)  # Gamma: 30-100 Hz

        # Add some spikes (rare events)
        spike_times = np.random.choice(samples, size=int(samples * 0.001), replace=False)
        spikes = np.zeros(samples)
        spikes[spike_times] = np.random.normal(0, 2, len(spike_times))

        # White noise
        noise = noise_level * np.random.randn(samples)

        data[ch] = alpha + beta + gamma + spikes + noise

    return data.astype(np.float32)

def get_fallback_plugins() -> List[str]:
    """Provide fallback plugin list for testing."""
    return [
        "neural_lz", "neural_arithmetic", "perceptual_quantizer",
        "predictive_coding", "context_aware"
    ]

def create_fallback_compressor(name: str):
    """Create a simple fallback compressor for testing."""
    class FallbackCompressor:
        def __init__(self, name):
            self.name = name

        def compress(self, data):
            # Simple mock compression - just serialize with some "compression"
            compressed = data.flatten().astype(np.float16).tobytes()  # Half precision = 50% size
            metadata = {"original_shape": data.shape, "dtype": str(data.dtype)}
            return compressed, metadata

        def decompress(self, compressed_data, metadata=None):
            # Mock decompression
            if metadata and "original_shape" in metadata:
                shape = metadata["original_shape"]
                data = np.frombuffer(compressed_data, dtype=np.float16).astype(np.float32)
                return data.reshape(shape)
            else:
                return np.frombuffer(compressed_data, dtype=np.float16)

    return FallbackCompressor(name)

def compute_compression_metrics(original_data: np.ndarray, compressed_data: bytes,
                              decompressed_data: np.ndarray = None,
                              compression_time: float = 0) -> Dict:
    """Compute comprehensive compression metrics."""
    metrics = {
        "overall_compression_ratio": original_data.nbytes / len(compressed_data),
        "compression_time": compression_time * 1000,  # Convert to ms
        "original_size_bytes": original_data.nbytes,
        "compressed_size_bytes": len(compressed_data),
        "gpu_available": GPU_AVAILABLE
    }

    # Quality metrics if decompressed data available
    if decompressed_data is not None and decompressed_data.shape == original_data.shape:
        try:
            # SNR
            signal_power = np.mean(original_data ** 2)
            noise_power = np.mean((original_data - decompressed_data) ** 2)
            if noise_power > 0:
                metrics["snr_db"] = 10 * np.log10(signal_power / noise_power)
            else:
                metrics["snr_db"] = float('inf')  # Perfect reconstruction

            # PSNR
            max_value = np.max(np.abs(original_data))
            if noise_power > 0:
                metrics["psnr_db"] = 20 * np.log10(max_value) - 10 * np.log10(noise_power)
            else:
                metrics["psnr_db"] = float('inf')

            # Spectral coherence (simplified)
            try:
                orig_fft = np.fft.fft(original_data, axis=1)
                decomp_fft = np.fft.fft(decompressed_data, axis=1)
                coherence = np.mean(np.abs(np.corrcoef(orig_fft.flatten(), decomp_fft.flatten())[0, 1]))
                metrics["spectral_coherence_error"] = 1.0 - coherence
            except:
                pass

        except Exception as e:
            logger.warning(f"Failed to compute quality metrics: {e}")

    return metrics

# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        plugin_count = len(list_plugins()) if BCI_AVAILABLE else len(get_fallback_plugins())
    except:
        plugin_count = 0

    return {
        "status": "healthy",
        "gpu_available": GPU_AVAILABLE,
        "mobile_available": MOBILE_AVAILABLE,
        "bci_available": BCI_AVAILABLE,
        "benchmarking_available": BENCHMARKING_AVAILABLE,
        "plugins_count": plugin_count,
        "active_connections": len(manager.active_connections),
        "streaming_sessions": len(manager.streaming_sessions)
    }

@app.get("/api/plugins", response_model=List[PluginInfo])
async def get_plugins():
    """Get available compression plugins with their capabilities."""
    try:
        if BCI_AVAILABLE:
            plugin_names = list_plugins()
        else:
            plugin_names = get_fallback_plugins()

        plugins = []

        for name in plugin_names:
            try:
                if BCI_AVAILABLE:
                    plugin_cls = get_plugin(name)
                    capabilities = getattr(plugin_cls, 'capabilities', lambda: {})()
                else:
                    capabilities = {}

                # Extract capabilities with defaults
                modes = capabilities.get('modes', ['fast', 'balanced', 'quality'])
                if isinstance(modes, str):
                    modes = [modes]

                # Determine characteristics based on name if capabilities not available
                is_lossless = (
                    'lossless' in capabilities.get('compression_types', []) or
                    'lz' in name.lower() or 'arithmetic' in name.lower()
                )
                is_lossy = (
                    'lossy' in capabilities.get('compression_types', []) or
                    'perceptual' in name.lower() or 'quantizer' in name.lower()
                )

                # Default to lossless if neither detected
                if not (is_lossless or is_lossy):
                    is_lossless = True

                supports_streaming = 'streaming' in capabilities.get('features', [])
                supports_gpu = (
                    'gpu' in capabilities.get('features', []) or
                    GPU_AVAILABLE
                )

                plugins.append(PluginInfo(
                    name=name,
                    modes=modes,
                    lossless=is_lossless,
                    lossy=is_lossy,
                    streaming=supports_streaming,
                    gpu_accel=supports_gpu,
                    version=capabilities.get('version', '1.0.0')
                ))

            except Exception as e:
                logger.warning(f"Failed to get info for plugin {name}: {e}")
                # Add basic fallback info
                plugins.append(PluginInfo(
                    name=name,
                    modes=["balanced"],
                    lossless=True,
                    lossy=False,
                    streaming=False,
                    gpu_accel=False,
                    version="1.0.0"
                ))

        logger.info(f"Found {len(plugins)} plugins")
        return plugins

    except Exception as e:
        logger.error(f"Error listing plugins: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list plugins: {str(e)}")

@app.post("/api/compress")
async def compress_data(
    file: Optional[UploadFile] = File(None),
    filePath: Optional[str] = Form(None),
    plugin: str = Form(...),
    mode: str = Form("balanced"),
    quality_level: float = Form(0.8),
    options: str = Form("{}")
):
    """Compress file data or data from file path."""
    try:
        # Parse options
        options_dict = json.loads(options) if options else {}

        # Generate session ID
        session_id = str(uuid.uuid4())
        logger.info(f"Starting compression session {session_id} with plugin {plugin}")

        # Load data
        if file:
            # Handle uploaded file
            content = await file.read()
            if file.filename.endswith('.npy'):
                with tempfile.NamedTemporaryFile() as tmp:
                    tmp.write(content)
                    tmp.flush()
                    data = np.load(tmp.name)
            elif file.filename.endswith(('.h5', '.hdf5')):
                with tempfile.NamedTemporaryFile() as tmp:
                    tmp.write(content)
                    tmp.flush()
                    with h5py.File(tmp.name, 'r') as f:
                        # Get first dataset
                        first_key = list(f.keys())[0]
                        data = f[first_key][:]
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format")
        elif filePath:
            # Load from file path (for internal use)
            if filePath.endswith('.npy'):
                data = np.load(filePath)
            else:
                raise HTTPException(status_code=400, detail="File path loading only supports .npy")
        else:
            raise HTTPException(status_code=400, detail="Either file or filePath must be provided")

        # Ensure data is 2D (channels x samples)
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        elif len(data.shape) > 2:
            data = data.reshape(-1, data.shape[-1])

        logger.info(f"Loaded data shape: {data.shape}, dtype: {data.dtype}")

        # Create compressor
        start_time = time.time()
        if BCI_AVAILABLE:
            try:
                compressor = create_compressor(plugin)
            except:
                compressor = create_fallback_compressor(plugin)
        else:
            compressor = create_fallback_compressor(plugin)

        # Perform compression
        if hasattr(compressor, 'compress'):
            compressed_data, comp_metadata = compressor.compress(data)
        else:
            # Fallback compression
            compressed_data = data.astype(np.float16).tobytes()
            comp_metadata = {"original_shape": data.shape, "dtype": str(data.dtype)}

        compression_time = time.time() - start_time

        # Convert compressed data to bytes if needed
        if not isinstance(compressed_data, (bytes, bytearray)):
            if isinstance(compressed_data, np.ndarray):
                compressed_bytes = compressed_data.tobytes()
            else:
                compressed_bytes = bytes(compressed_data)
        else:
            compressed_bytes = bytes(compressed_data)

        # Decompress for quality metrics
        decompressed_data = None
        if hasattr(compressor, 'decompress'):
            try:
                decompressed_data = compressor.decompress(compressed_data)
                if hasattr(decompressed_data, 'shape') and decompressed_data.shape != data.shape:
                    decompressed_data = None
            except Exception as e:
                logger.warning(f"Failed to decompress for quality metrics: {e}")

        # Compute metrics
        metrics = compute_compression_metrics(data, compressed_bytes, decompressed_data, compression_time)

        # Save artifact
        session_dir = GUI_ARTIFACTS_DIR / session_id
        session_dir.mkdir(exist_ok=True)

        # Save in NPZ format with metadata
        artifact_path = session_dir / "compressed.npz"
        metadata_path = session_dir / "metadata.json"

        np.savez_compressed(
            artifact_path,
            payload=np.frombuffer(compressed_bytes, dtype=np.uint8),
            original_shape=data.shape,
            original_dtype=str(data.dtype),
            plugin=plugin,
            mode=mode,
            quality_level=quality_level
        )

        # Save human-readable metadata
        full_metadata = {
            "session_id": session_id,
            "plugin": plugin,
            "mode": mode,
            "quality_level": quality_level,
            "options": options_dict,
            "original_shape": data.shape,
            "original_dtype": str(data.dtype),
            "compression_metadata": comp_metadata if isinstance(comp_metadata, dict) else {},
            **metrics
        }

        with open(metadata_path, 'w') as f:
            json.dump(full_metadata, f, indent=2, default=str)

        return {
            "session_id": session_id,
            "metrics": metrics,
            "artifactPath": str(artifact_path),
            "metadataPath": str(metadata_path)
        }

    except Exception as e:
        logger.error(f"Compression error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/decompress")
async def decompress_data(request: DecompressRequest):
    """Decompress previously compressed data."""
    try:
        # Load artifact
        if not os.path.exists(request.artifactPath):
            raise HTTPException(status_code=404, detail="Artifact not found")

        with np.load(request.artifactPath) as data:
            payload = data['payload'].tobytes()
            original_shape = tuple(data['original_shape'])
            plugin_name = str(data['plugin'])

        logger.info(f"Decompressing artifact with plugin {plugin_name}")

        # Create compressor
        if BCI_AVAILABLE:
            try:
                compressor = create_compressor(plugin_name)
            except:
                compressor = create_fallback_compressor(plugin_name)
        else:
            compressor = create_fallback_compressor(plugin_name)

        # Perform decompression
        start_time = time.time()
        if hasattr(compressor, 'decompress'):
            reconstructed_data = compressor.decompress(payload)
        else:
            # Fallback decompression
            reconstructed_data = np.frombuffer(payload, dtype=np.float16).astype(np.float32)
            reconstructed_data = reconstructed_data.reshape(original_shape)

        decompression_time = time.time() - start_time

        # Save reconstructed data
        session_id = Path(request.artifactPath).parent.name
        session_dir = GUI_ARTIFACTS_DIR / session_id
        reconstructed_path = session_dir / "reconstructed.npy"
        np.save(reconstructed_path, reconstructed_data)

        # Compute decompression metrics
        metrics = {
            "decompression_time_ms": decompression_time * 1000,
            "reconstructed_shape": reconstructed_data.shape,
            "reconstructed_dtype": str(reconstructed_data.dtype)
        }

        return {
            "reconstructedPath": str(reconstructed_path),
            "metrics": metrics
        }

    except Exception as e:
        logger.error(f"Decompression error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/benchmark")
async def run_benchmark(request: BenchmarkRequest):
    """Run compression benchmark."""
    try:
        logger.info(f"Running benchmark for plugin {request.plugin}")

        # Generate or load test data
        if request.dataset == "synthetic":
            data = generate_synthetic_neural_data(
                request.channels,
                request.sample_rate * request.duration_s,
                request.sample_rate
            )
        elif request.dataset == "file" and request.filePath:
            data = np.load(request.filePath)
        else:
            raise HTTPException(status_code=400, detail="Invalid dataset specification")

        # Create compressor
        if BCI_AVAILABLE:
            try:
                compressor = create_compressor(request.plugin)
            except:
                compressor = create_fallback_compressor(request.plugin)
        else:
            compressor = create_fallback_compressor(request.plugin)

        # Run benchmark
        results = []
        num_trials = 3  # Multiple trials for averaging

        for trial in range(num_trials):
            start_time = time.time()

            # Compression
            compressed_data, comp_metadata = compressor.compress(data)
            compression_time = time.time() - start_time

            # Convert to bytes
            if not isinstance(compressed_data, (bytes, bytearray)):
                compressed_bytes = compressed_data.tobytes() if hasattr(compressed_data, 'tobytes') else bytes(compressed_data)
            else:
                compressed_bytes = bytes(compressed_data)

            # Decompression
            decompressed_data = None
            decompression_time = 0
            if hasattr(compressor, 'decompress'):
                start_time = time.time()
                try:
                    decompressed_data = compressor.decompress(compressed_data)
                    decompression_time = time.time() - start_time
                except:
                    pass

            # Compute metrics
            metrics = compute_compression_metrics(data, compressed_bytes, decompressed_data, compression_time)
            metrics["decompression_time_ms"] = decompression_time * 1000
            metrics["trial"] = trial

            results.append(metrics)

        # Aggregate results
        aggregated = {}
        for metric in request.metrics:
            values = [r.get(metric, 0) for r in results if metric in r]
            if values:
                aggregated[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }

        return {
            "plugin": request.plugin,
            "dataset": request.dataset,
            "data_shape": data.shape,
            "trials": len(results),
            "aggregated_metrics": aggregated,
            "raw_results": results
        }

    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file for later processing."""
    try:
        # Save uploaded file
        upload_dir = GUI_ARTIFACTS_DIR / "uploads"
        upload_dir.mkdir(exist_ok=True)

        file_id = str(uuid.uuid4())
        file_path = upload_dir / f"{file_id}_{file.filename}"

        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        logger.info(f"Uploaded file {file.filename} as {file_path}")

        return {
            "file_id": file_id,
            "filename": file.filename,
            "file_path": str(file_path),
            "size_bytes": len(content)
        }

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """Download an artifact file."""
    try:
        # Security: only allow downloads from artifacts directory
        file_path = GUI_ARTIFACTS_DIR / filename

        # Check if it's a session directory
        if file_path.is_dir():
            # Look for compressed.npz or metadata.json
            for candidate in ["compressed.npz", "metadata.json", "reconstructed.npy"]:
                candidate_path = file_path / candidate
                if candidate_path.exists():
                    file_path = candidate_path
                    break

        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        # Ensure file is within artifacts directory (security)
        try:
            file_path.resolve().relative_to(GUI_ARTIFACTS_DIR.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied")

        return FileResponse(
            path=str(file_path),
            filename=file_path.name,
            media_type='application/octet-stream'
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-data")
async def generate_data(request: GenerateDataRequest):
    """Generate synthetic neural data for testing."""
    try:
        data = generate_synthetic_neural_data(
            request.channels,
            request.samples,
            request.sample_rate,
            request.noise_level
        )

        # Save to temporary file
        data_dir = GUI_ARTIFACTS_DIR / "generated"
        data_dir.mkdir(exist_ok=True)

        file_id = str(uuid.uuid4())
        file_path = data_dir / f"synthetic_{file_id}.npy"
        np.save(file_path, data)

        logger.info(f"Generated synthetic data: {data.shape} saved to {file_path}")

        return {
            "file_id": file_id,
            "file_path": str(file_path),
            "shape": data.shape,
            "dtype": str(data.dtype),
            "size_bytes": data.nbytes
        }

    except Exception as e:
        logger.error(f"Data generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/logs")
async def get_logs(tail: int = Query(100, description="Number of lines to tail")):
    """Get server logs."""
    try:
        log_file = LOGS_DIR / "gui_server.log"
        if not log_file.exists():
            return {"logs": []}

        with open(log_file, 'r') as f:
            lines = f.readlines()

        # Return last N lines
        recent_lines = lines[-tail:] if len(lines) > tail else lines

        return {
            "logs": [line.strip() for line in recent_lines],
            "total_lines": len(lines),
            "tail": tail
        }

    except Exception as e:
        logger.error(f"Logs error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time metrics
@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    """WebSocket endpoint for real-time telemetry stream."""
    await manager.connect(websocket)

    try:
        # Start metrics streaming task in background
        metrics_task = asyncio.create_task(metrics_streaming_task())

        # Keep connection alive
        while True:
            try:
                # Wait for client messages (could be control commands)
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)

                # Handle client messages
                try:
                    data = json.loads(message)
                    if data.get("type") == "start_stream":
                        session_id = data.get("session_id", str(uuid.uuid4()))
                        config = data.get("config", {})
                        manager.streaming_sessions[session_id] = {
                            "config": config,
                            "active": True,
                            "start_time": time.time()
                        }
                        logger.info(f"Started streaming session {session_id}")
                    elif data.get("type") == "stop_stream":
                        session_id = data.get("session_id")
                        if session_id in manager.streaming_sessions:
                            manager.streaming_sessions[session_id]["active"] = False
                            logger.info(f"Stopped streaming session {session_id}")
                except json.JSONDecodeError:
                    pass  # Ignore malformed messages

            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_text(json.dumps({"type": "ping", "timestamp": time.time()}))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        metrics_task.cancel()
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)
        metrics_task.cancel()

async def metrics_streaming_task():
    """Background task to generate and broadcast metrics."""
    logger.info("Started metrics streaming task")

    while manager.active_connections:
        try:
            # Generate sample metrics (simulated real-time compression)
            current_time = time.time()

            # Simulate varying metrics
            base_ratio = 2.5 + 0.5 * np.sin(current_time * 0.1)  # Slow oscillation
            noise = 0.1 * np.random.randn()  # Random variation

            metrics = {
                "type": "metrics",
                "timestamp": current_time,
                "compression_ratio": max(1.0, base_ratio + noise),
                "latency_ms": 0.5 + abs(0.2 * np.random.randn()),  # 0.3-0.7ms typical
                "snr_db": 25 + 5 * np.random.randn(),  # 20-30dB range
                "psnr_db": 35 + 5 * np.random.randn(),  # 30-40dB range
                "spectral_coherence_error": abs(0.05 * np.random.randn()),  # Small error
                "gpu_available": GPU_AVAILABLE,
                "active_sessions": len(manager.streaming_sessions),
                "cpu_usage": min(100, max(0, 20 + 10 * np.random.randn()))  # Simulated CPU usage
            }

            # Add spike detection metrics occasionally
            if np.random.random() < 0.3:  # 30% chance
                metrics["spike_f1"] = 0.85 + 0.1 * np.random.randn()
                metrics["jitter_ms"] = abs(0.1 * np.random.randn())

            await manager.broadcast(metrics)

            # Stream at ~10 Hz
            await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Metrics streaming error: {e}")
            await asyncio.sleep(1.0)  # Back off on error

        # Check if we still have connections
        if not manager.active_connections:
            break

    logger.info("Metrics streaming task stopped - no active connections")

if __name__ == "__main__":
    try:
        import uvicorn
        logger.info("Starting BCI Compression Telemetry Server on http://0.0.0.0:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    except ImportError:
        print("Uvicorn not available. Install with: pip install uvicorn")
        print("Then run: uvicorn scripts.telemetry_server:app --reload --host 0.0.0.0 --port 8000")


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


# GPU detection
def detect_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        try:
            import cupy
            return True
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


# API Endpoints
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


@app.get("/api/plugins", response_model=List[PluginInfo])
async def get_plugins():
    """Get available compression plugins with their capabilities."""
    try:
        plugins = []
        for name in list_plugins():
            try:
                plugin = get_plugin(name)

                # Get plugin capabilities
                capabilities = {}
                if hasattr(plugin, 'get_capabilities'):
                    capabilities = plugin.get_capabilities()

                plugins.append(PluginInfo(
                    name=name,
                    modes=["fast", "balanced", "quality"],  # Default modes
                    capabilities=capabilities,
                    is_lossless=getattr(plugin, 'is_lossless', True),
                    is_lossy=getattr(plugin, 'is_lossy', False),
                    supports_streaming=getattr(plugin, 'supports_streaming', False),
                    supports_gpu=getattr(plugin, 'supports_gpu', GPU_AVAILABLE)
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


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a neural data file for processing."""
    try:
        # Validate file type
        if not file.filename.endswith(('.npy', '.h5', '.hdf5')):
            raise HTTPException(status_code=400, detail="Only .npy and .h5 files are supported")

        # Create uploads directory if it doesn't exist
        uploads_dir = GUI_ARTIFACTS_DIR / "uploads"
        uploads_dir.mkdir(exist_ok=True)

        # Generate unique filename
        timestamp = int(time.time())
        base_name = Path(file.filename).stem
        extension = Path(file.filename).suffix
        unique_filename = f"{base_name}_{timestamp}{extension}"
        file_path = uploads_dir / unique_filename

        # Save uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        return {"filename": unique_filename, "size": len(content)}

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/compress")
async def compress_file(request: dict):
    """Compress uploaded file and return metrics."""
    try:
        filename = request.get("filename")
        plugin = request.get("plugin")
        mode = request.get("mode", "balanced")
        quality = request.get("quality", 0.8)

        if not filename or not plugin:
            raise HTTPException(status_code=400, detail="filename and plugin are required")

        # Find file in uploads directory
        uploads_dir = GUI_ARTIFACTS_DIR / "uploads"
        file_path = uploads_dir / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File {filename} not found")

        # Load data based on file extension
        if filename.endswith('.npy'):
            data = np.load(file_path)
        elif filename.endswith(('.h5', '.hdf5')):
            with h5py.File(file_path, 'r') as h5f:
                # Try common dataset names
                if 'data' in h5f:
                    data = h5f['data'][:]
                elif 'neural_data' in h5f:
                    data = h5f['neural_data'][:]
                else:
                    # Use first dataset found
                    first_key = list(h5f.keys())[0]
                    data = h5f[first_key][:]
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use .npy or .h5")

        # Create compressor
        compressor = create_compressor(plugin, config=None)

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
            raise HTTPException(status_code=400, detail=f"Plugin {plugin} does not support compression")

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
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0

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
                    # Calculate SNR and PSNR
                    signal_power = np.mean(data ** 2)
                    noise_power = np.mean((data - reconstructed) ** 2)
                    if noise_power > 0:
                        snr_db = 10 * np.log10(signal_power / noise_power)
                        max_val = np.max(np.abs(data))
                        psnr_db = 20 * np.log10(max_val / np.sqrt(noise_power))
                    else:
                        snr_db = float('inf')  # Perfect reconstruction
                        psnr_db = float('inf')
            except Exception as e:
                logger.warning(f"Failed to compute quality metrics: {e}")

        # Create output filename
        base_name = Path(filename).stem
        output_filename = f"{base_name}_compressed.npz"

        # Save compressed file
        output_dir = GUI_ARTIFACTS_DIR / "compressed"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / output_filename

        np.savez_compressed(output_path,
                          compressed_data=compressed_bytes,
                          metadata={
                              "plugin": plugin,
                              "mode": mode,
                              "original_shape": data.shape,
                              "compression_ratio": compression_ratio
                          })

        return {
            "compression_ratio": compression_ratio,
            "compression_time_ms": compression_time * 1000,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "output_filename": output_filename,
            "snr_db": snr_db,
            "psnr_db": psnr_db
        }

    except Exception as e:
        logger.error(f"Compression error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/decompress")
async def decompress_file(request: dict):
    """Decompress file and return metrics."""
    try:
        filename = request.get("filename")
        plugin = request.get("plugin")

        if not filename or not plugin:
            raise HTTPException(status_code=400, detail="filename and plugin are required")

        # Find compressed file
        compressed_dir = GUI_ARTIFACTS_DIR / "compressed"
        file_path = compressed_dir / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Compressed file {filename} not found")

        # Load compressed data
        npz_file = np.load(file_path)
        compressed_bytes = npz_file['compressed_data'].tobytes()
        metadata = npz_file['metadata'].item()

        # Create compressor
        compressor = create_compressor(plugin, config=None)

        if not hasattr(compressor, 'decompress'):
            raise HTTPException(status_code=400, detail=f"Plugin {plugin} does not support decompression")

        # Decompress
        start_time = time.time()
        reconstructed = compressor.decompress(compressed_bytes)
        decompression_time = time.time() - start_time

        # Reshape to original shape if needed
        original_shape = metadata.get("original_shape")
        if original_shape and isinstance(reconstructed, np.ndarray):
            try:
                reconstructed = reconstructed.reshape(original_shape)
            except ValueError:
                logger.warning("Failed to reshape reconstructed data to original shape")

        # Save reconstructed file
        base_name = Path(filename).stem.replace('_compressed', '')
        output_filename = f"{base_name}_decompressed.npy"
        output_dir = GUI_ARTIFACTS_DIR / "decompressed"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / output_filename

        np.save(output_path, reconstructed)

        return {
            "decompression_time_ms": decompression_time * 1000,
            "decompressed_size": reconstructed.nbytes if isinstance(reconstructed, np.ndarray) else len(reconstructed),
            "output_filename": output_filename
        }

    except Exception as e:
        logger.error(f"Decompression error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/benchmark")
async def run_benchmark(request: dict):
    """Run compression benchmark."""
    try:
        plugin = request.get("plugin")
        mode = request.get("mode", "balanced")
        quality = request.get("quality", 0.8)
        num_trials = request.get("num_trials", 5)

        if not plugin:
            raise HTTPException(status_code=400, detail="plugin is required")

        # Generate synthetic data for benchmarking
        channels = 32
        samples = 30000  # 1 second at 30kHz
        data = generate_synthetic_data(channels, samples)

        # Create compressor
        compressor = create_compressor(plugin, config=None)

        # Run multiple trials
        results = []
        for _ in range(num_trials):
            # Compression
            start_time = time.time()
            compressed, _ = compressor.compress(data)
            compression_time = time.time() - start_time

            # Calculate metrics
            original_size = data.nbytes
            compressed_size = len(compressed) if isinstance(compressed, (bytes, bytearray)) else np.array(compressed).nbytes
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0

            # Quality metrics
            snr_db = None
            psnr_db = None
            if hasattr(compressor, 'decompress'):
                try:
                    reconstructed = compressor.decompress(compressed)
                    if isinstance(reconstructed, np.ndarray) and reconstructed.shape == data.shape:
                        signal_power = np.mean(data ** 2)
                        noise_power = np.mean((data - reconstructed) ** 2)
                        if noise_power > 0:
                            snr_db = 10 * np.log10(signal_power / noise_power)
                            max_val = np.max(np.abs(data))
                            psnr_db = 20 * np.log10(max_val / np.sqrt(noise_power))
                        else:
                            snr_db = float('inf')
                            psnr_db = float('inf')
                except Exception:
                    pass

            results.append({
                "compression_ratio": compression_ratio,
                "latency_ms": compression_time * 1000,
                "snr_db": snr_db,
                "psnr_db": psnr_db
            })

        # Calculate averages
        avg_compression_ratio = np.mean([r["compression_ratio"] for r in results])
        avg_latency_ms = np.mean([r["latency_ms"] for r in results])
        avg_snr_db = np.mean([r["snr_db"] for r in results if r["snr_db"] is not None]) if any(r["snr_db"] is not None for r in results) else None
        avg_psnr_db = np.mean([r["psnr_db"] for r in results if r["psnr_db"] is not None]) if any(r["psnr_db"] is not None for r in results) else None

        return {
            "avg_compression_ratio": avg_compression_ratio,
            "avg_latency_ms": avg_latency_ms,
            "avg_snr_db": avg_snr_db,
            "avg_psnr_db": avg_psnr_db,
            "num_trials": num_trials,
            "plugin": plugin,
            "mode": mode
        }

    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-data")
async def generate_test_data(
    num_channels: int = 64,
    duration_seconds: int = 10,
    sampling_rate: int = 1000
):
    """Generate synthetic neural data for testing."""
    try:
        # Generate synthetic data
        samples = int(duration_seconds * sampling_rate)
        data = generate_synthetic_data(num_channels, samples, sampling_rate)

        # Create filename
        timestamp = int(time.time())
        filename = f"synthetic_data_{num_channels}ch_{duration_seconds}s_{timestamp}.npy"

        # Save to uploads directory
        uploads_dir = GUI_ARTIFACTS_DIR / "uploads"
        uploads_dir.mkdir(exist_ok=True)
        file_path = uploads_dir / filename

        np.save(file_path, data)

        return {
            "filename": filename,
            "channels": num_channels,
            "samples": samples,
            "duration_s": duration_seconds,
            "sampling_rate": sampling_rate,
            "size_bytes": data.nbytes
        }

    except Exception as e:
        logger.error(f"Data generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """Download processed file."""
    try:
        # Check different directories for the file
        for directory in ["compressed", "decompressed", "uploads"]:
            file_path = GUI_ARTIFACTS_DIR / directory / filename
            if file_path.exists():
                return FileResponse(
                    path=str(file_path),
                    filename=filename,
                    media_type='application/octet-stream'
                )

        raise HTTPException(status_code=404, detail=f"File {filename} not found")

    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/metrics")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time metrics streaming."""
    await manager.connect(websocket)
    session_id = None

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("command") == "start_stream":
                # Start streaming simulation
                plugin = message.get("plugin", "huffman")
                mode = message.get("mode", "balanced")
                quality = message.get("quality", 0.8)

                session_id = str(uuid.uuid4())
                manager.active_sessions[session_id] = {
                    "plugin": plugin,
                    "mode": mode,
                    "quality": quality,
                    "websocket": websocket
                }

                # Start simulation task
                asyncio.create_task(simulate_streaming(session_id, plugin, websocket))

            elif message.get("command") == "stop_stream":
                if session_id and session_id in manager.active_sessions:
                    del manager.active_sessions[session_id]
                    session_id = None

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        if session_id and session_id in manager.active_sessions:
            del manager.active_sessions[session_id]


async def simulate_streaming(session_id: str, plugin: str, websocket: WebSocket):
    """Simulate real-time compression streaming."""
    try:
        compressor = create_compressor(plugin, config=None)

        while session_id in manager.active_sessions:
            # Generate small chunk of data
            chunk_data = generate_synthetic_data(8, 1000)  # 8 channels, small chunk

            # Compress
            start_time = time.time()
            compressed, _ = compressor.compress(chunk_data)
            compression_time = time.time() - start_time

            # Calculate metrics
            original_size = chunk_data.nbytes
            compressed_size = len(compressed) if isinstance(compressed, (bytes, bytearray)) else np.array(compressed).nbytes
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0

            # Send metrics
            metrics = {
                "type": "metrics",
                "data": {
                    "timestamp": time.time(),
                    "session_id": session_id,
                    "compression_ratio": compression_ratio,
                    "latency_ms": compression_time * 1000,
                    "snr_db": 40.0 + np.random.normal(0, 2),  # Simulated
                    "psnr_db": 50.0 + np.random.normal(0, 3),  # Simulated
                    "gpu_available": GPU_AVAILABLE
                }
            }

            await websocket.send_text(json.dumps(metrics))
            await asyncio.sleep(0.1)  # 10 Hz update rate

    except Exception as e:
        logger.error(f"Streaming simulation error: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": str(e)
        }))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


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
        try:
            import cupy
            return True
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import core compression functionality

try:
    from bci_compression.mobile.mobile_compressor import MobileBCICompressor
    MOBILE_AVAILABLE = True
except ImportError:
    MOBILE_AVAILABLE = False

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

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import core compression functionality

try:
    from bci_compression.mobile.mobile_compressor import MobileBCICompressor
    MOBILE_AVAILABLE = True
except ImportError:
    MOBILE_AVAILABLE = False

# Create logs directory if it doesn't exist
LOGS_DIR = Path(__file__).parent.parent / "logs"
GUI_ARTIFACTS_DIR = LOGS_DIR / "gui_artifacts"
GUI_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
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
        try:
            import cupy
            return True
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

def load_artifact(artifact_path: str) -> tuple[bytes, dict]:
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


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a neural data file for processing."""
    try:
        # Validate file type
        if not file.filename.endswith(('.npy', '.h5', '.hdf5')):
            raise HTTPException(status_code=400, detail="Only .npy and .h5 files are supported")

        # Create uploads directory if it doesn't exist
        uploads_dir = GUI_ARTIFACTS_DIR / "uploads"
        uploads_dir.mkdir(exist_ok=True)

        # Generate unique filename
        timestamp = int(time.time())
        base_name = Path(file.filename).stem
        extension = Path(file.filename).suffix
        unique_filename = f"{base_name}_{timestamp}{extension}"
        file_path = uploads_dir / unique_filename

        # Save uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        return {"filename": unique_filename, "size": len(content)}

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/compress")
async def compress_file(request: dict):
    """Compress uploaded file and return metrics."""
    try:
        filename = request.get("filename")
        plugin = request.get("plugin")
        mode = request.get("mode", "balanced")
        quality = request.get("quality", 0.8)

        if not filename or not plugin:
            raise HTTPException(status_code=400, detail="filename and plugin are required")

        # Find file in uploads directory
        uploads_dir = GUI_ARTIFACTS_DIR / "uploads"
        file_path = uploads_dir / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File {filename} not found")

        # Generate session ID
        session_id = str(uuid.uuid4())

        # Load data based on file extension
        if filename.endswith('.npy'):
            data = np.load(file_path)
        elif filename.endswith(('.h5', '.hdf5')):
            with h5py.File(file_path, 'r') as h5f:
                # Try common dataset names
                if 'data' in h5f:
                    data = h5f['data'][:]
                elif 'neural_data' in h5f:
                    data = h5f['neural_data'][:]
                else:
                    # Use first dataset found
                    first_key = list(h5f.keys())[0]
                    data = h5f[first_key][:]
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use .npy or .h5")

        # Create compressor
        compressor = create_compressor(
            plugin,
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
            raise HTTPException(status_code=400, detail=f"Plugin {plugin} does not support compression")

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

        # Create output filename
        base_name = Path(filename).stem
        output_filename = f"{base_name}_compressed_{session_id}.npz"

        # Save compressed file
        output_dir = GUI_ARTIFACTS_DIR / "compressed"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / output_filename

        np.savez_compressed(output_path,
                          compressed_data=compressed_bytes,
                          metadata={
                              "plugin": plugin,
                              "mode": mode,
                              "original_shape": data.shape,
                              "compression_ratio": compression_ratio
                          })

        return {
            "compression_ratio": compression_ratio,
            "compression_time_ms": compression_time * 1000,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "output_filename": output_filename,
            "snr_db": snr_db,
            "psnr_db": psnr_db
        }

    except Exception as e:
        logger.error(f"Compression error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/decompress")
async def decompress_file(request: dict):
    """Decompress artifact and return reconstructed file."""
    try:
        if not os.path.exists(artifact_path):
            raise HTTPException(status_code=404, detail="Artifact not found")

        # Load artifact
        compressed_bytes, metadata = load_artifact(artifact_path)

        # Create compressor
        plugin = metadata.get("plugin", "dummy_lz")
        compressor = create_compressor(plugin, config=None)

        if not hasattr(compressor, 'decompress'):
            raise HTTPException(status_code=400, detail=f"Plugin {plugin} does not support decompression")

        # Decompress
        start_time = time.time()
        reconstructed = compressor.decompress(compressed_bytes)
        decompression_time = time.time() - start_time

        # Reshape to original shape if needed
        original_shape = metadata.get("original_shape")
        if original_shape and isinstance(reconstructed, np.ndarray):
            try:
                reconstructed = reconstructed.reshape(original_shape)
            except ValueError:
                logger.warning("Failed to reshape reconstructed data to original shape")

        # Save reconstructed file
        session_id = metadata.get("session_id", str(uuid.uuid4()))
        session_dir = GUI_ARTIFACTS_DIR / session_id
        reconstructed_path = session_dir / "reconstructed.npy"

        np.save(reconstructed_path, reconstructed)

        return {
            "decompression_time_ms": decompression_time * 1000,
            "reconstructed_path": str(reconstructed_path),
            "original_shape": original_shape,
            "reconstructed_shape": reconstructed.shape if isinstance(reconstructed, np.ndarray) else None
        }

    except Exception as e:
        logger.error(f"Decompression error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/benchmark")
async def run_benchmark(request: BenchmarkRequest):
    """Run quick benchmark on synthetic or file data."""
    try:
        # Generate or load test data
        if request.dataset == "synthetic":
            data = generate_synthetic_data(
                channels=request.channels,
                samples=request.sample_rate * request.duration_s,
                sample_rate=request.sample_rate
            )
        else:
            raise HTTPException(status_code=400, detail="File dataset not implemented yet")

        # Create compressor
        compressor = create_compressor(request.plugin, config=None)

        # Run compression benchmark
        results = {}

        if "compression_ratio" in request.metrics or "latency_ms" in request.metrics:
            start_time = time.time()
            compressed, meta = compressor.compress(data)
            compression_time = time.time() - start_time

            original_size = data.nbytes
            compressed_size = len(compressed) if isinstance(compressed, (bytes, bytearray)) else len(np.array(compressed).tobytes())

            results["compression_ratio"] = BenchmarkMetrics.compression_ratio(original_size, compressed_size)
            results["latency_ms"] = compression_time * 1000

        # Quality metrics
        if ("snr_db" in request.metrics or "psnr_db" in request.metrics) and hasattr(compressor, 'decompress'):
            try:
                reconstructed = compressor.decompress(compressed)
                if isinstance(reconstructed, np.ndarray) and reconstructed.shape == data.shape:
                    if "snr_db" in request.metrics:
                        results["snr_db"] = BenchmarkMetrics.snr(data, reconstructed)
                    if "psnr_db" in request.metrics:
                        results["psnr_db"] = BenchmarkMetrics.psnr(data, reconstructed, max_value=np.max(np.abs(data)))
            except Exception as e:
                logger.warning(f"Failed to compute quality metrics in benchmark: {e}")

        return {
            "plugin": request.plugin,
            "dataset_info": {
                "channels": request.channels,
                "samples": request.sample_rate * request.duration_s,
                "duration_s": request.duration_s,
                "sample_rate": request.sample_rate
            },
            "results": results,
            "gpu_available": GPU_AVAILABLE
        }

    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-data")
async def generate_test_data(
    num_channels: int = 64,
    duration_seconds: int = 10,
    sampling_rate: int = 1000
):
    """Generate synthetic neural data for testing."""
    try:
        # Generate synthetic data
        samples = int(duration_seconds * sampling_rate)
        data = generate_synthetic_data(num_channels, samples, sampling_rate)

        # Create filename
        timestamp = int(time.time())
        filename = f"synthetic_data_{num_channels}ch_{duration_seconds}s_{timestamp}.npy"

        # Save to uploads directory
        uploads_dir = GUI_ARTIFACTS_DIR / "uploads"
        uploads_dir.mkdir(exist_ok=True)
        file_path = uploads_dir / filename

        np.save(file_path, data)

        return {
            "filename": filename,
            "channels": num_channels,
            "samples": samples,
            "duration_s": duration_seconds,
            "sampling_rate": sampling_rate,
            "size_bytes": data.nbytes
        }

    except Exception as e:
        logger.error(f"Data generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/logs")
async def get_logs(lines: int = 100):
    """Get recent log entries."""
    try:
        log_file = LOGS_DIR / "gui_server.log"
        if not log_file.exists():
            return {"logs": []}

        with open(log_file, 'r') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

        return {"logs": [line.strip() for line in recent_lines]}
    except Exception as e:
        return {"logs": [f"Error reading logs: {e}"]}

@app.get("/api/download/{session_id}/{filename}")
async def download_file(session_id: str, filename: str):
    """Download artifact files."""
    try:
        file_path = GUI_ARTIFACTS_DIR / session_id / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type='application/octet-stream'
        )
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time metrics
@app.websocket("/ws/metrics")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                data = json.loads(message)

                # Handle start streaming command
                if data.get("command") == "start_stream":
                    session_id = str(uuid.uuid4())
                    plugin = data.get("plugin", "dummy_lz")
                    mode = data.get("mode", "balanced")
                    quality = data.get("quality", 0.8)

                    # Start streaming session
                    manager.active_sessions[session_id] = {
                        "plugin": plugin,
                        "mode": mode,
                        "quality": quality,
                        "active": True
                    }

                    # Start streaming simulation in background
                    asyncio.create_task(simulate_streaming(session_id, plugin, websocket))

                elif data.get("command") == "stop_stream":
                    session_id = data.get("session_id")
                    if session_id in manager.active_sessions:
                        manager.active_sessions[session_id]["active"] = False

            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_text(json.dumps({"type": "heartbeat", "timestamp": time.time()}))
            except Exception as e:
                logger.error(f"WebSocket message error: {e}")
                break

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

async def simulate_streaming(session_id: str, plugin: str, websocket: WebSocket):
    """Simulate real-time streaming compression with telemetry."""
    try:
        compressor = create_compressor(plugin, config=None)
        chunk_size = 1024  # samples per chunk
        channels = 32
        sample_rate = 30000

        chunk_count = 0
        total_compression_ratio = 0
        total_latency = 0

        while session_id in manager.active_sessions and manager.active_sessions[session_id]["active"]:
            # Generate synthetic chunk
            chunk = generate_synthetic_data(channels, chunk_size, sample_rate)

            # Compress chunk
            start_time = time.time()
            if hasattr(compressor, 'stream_chunk'):
                compressed_bytes, meta = compressor.stream_chunk(chunk)
            else:
                compressed, meta = compressor.compress(chunk)
                compressed_bytes = compressed if isinstance(compressed, (bytes, bytearray)) else np.array(compressed).tobytes()

            compression_time = time.time() - start_time

            # Calculate metrics
            original_size = chunk.nbytes
            compressed_size = len(compressed_bytes)
            compression_ratio = BenchmarkMetrics.compression_ratio(original_size, compressed_size)
            latency_ms = compression_time * 1000

            # Update running averages
            chunk_count += 1
            total_compression_ratio += compression_ratio
            total_latency += latency_ms

            # Optional quality metrics (expensive, so compute occasionally)
            snr_db = None
            psnr_db = None
            if chunk_count % 10 == 0 and hasattr(compressor, 'decompress'):  # Every 10th chunk
                try:
                    reconstructed = compressor.decompress(compressed_bytes)
                    if isinstance(reconstructed, np.ndarray) and reconstructed.shape == chunk.shape:
                        snr_db = BenchmarkMetrics.snr(chunk, reconstructed)
                        psnr_db = BenchmarkMetrics.psnr(chunk, reconstructed, max_value=np.max(np.abs(chunk)))
                except Exception:
                    pass

            # Send telemetry
            telemetry = MetricsTelemetry(
                timestamp=time.time(),
                session_id=session_id,
                compression_ratio=compression_ratio,
                latency_ms=latency_ms,
                snr_db=snr_db,
                psnr_db=psnr_db,
                gpu_available=GPU_AVAILABLE
            )

            await manager.broadcast({
                "type": "metrics",
                "data": telemetry.dict()
            })

            # Simulate real-time rate (approximately 30Hz for 1024 samples at 30kHz)
            await asyncio.sleep(chunk_size / sample_rate)

    except Exception as e:
        logger.error(f"Streaming simulation error: {e}")
        if session_id in manager.active_sessions:
            manager.active_sessions[session_id]["active"] = False

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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
        "plugins_count": len(list_plugins()),
        "active_sessions": len(manager.active_sessions)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
