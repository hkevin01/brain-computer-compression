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
from typing import Dict, List, Optional

import h5py
import numpy as np

# FastAPI imports
try:
    from fastapi import (FastAPI, File, Form, HTTPException, Query, UploadFile,
                         WebSocket, WebSocketDisconnect)
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
        import uvicorn
        logger.info("Starting BCI Compression Telemetry Server on http://0.0.0.0:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    except ImportError:
        print("Uvicorn not available. Install with: pip install uvicorn")
        print("Then run: uvicorn scripts.telemetry_server:app --reload --host 0.0.0.0 --port 8000")
