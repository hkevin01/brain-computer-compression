import asyncio
import json
import logging
import math
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import (FastAPI, File, Form, HTTPException, Query, UploadFile,
                     WebSocket, WebSocketDisconnect)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Optional dependencies
try:  # Compression framework (may not be installed yet)
    from bci_compression.core import create_compressor
    from bci_compression.plugins import get_plugin, list_plugins
    BCI_AVAILABLE = True
except Exception:  # pragma: no cover
    BCI_AVAILABLE = False

try:
    import h5py
    H5_AVAILABLE = True
except Exception:  # pragma: no cover
    H5_AVAILABLE = False

try:
    import torch  # noqa: F401
    GPU_AVAILABLE = torch.cuda.is_available()
except Exception:  # pragma: no cover
    try:
        import cupy  # noqa: F401
        GPU_AVAILABLE = True
    except Exception:
        GPU_AVAILABLE = False

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT_DIR / "logs"
GUI_ARTIFACTS_DIR = LOGS_DIR / "gui_artifacts"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
GUI_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Logging
LOG_FILE = LOGS_DIR / "gui_server.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("telemetry_server")

app = FastAPI(title="BCI Compression Telemetry Server", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
class PluginDescriptor(BaseModel):
    name: str
    modes: List[str]
    lossless: bool
    lossy: bool
    streaming: bool
    gpu_accel: bool
    version: str = "unknown"


class CompressResponse(BaseModel):
    session_id: str
    artifactPath: str
    metadataPath: str
    metrics: Dict[str, Any]


class DecompressResponse(BaseModel):
    reconstructedPath: str
    metrics: Dict[str, Any]


class BenchmarkRequest(BaseModel):
    plugins: List[str]
    dataset: str = "synthetic"  # synthetic | file
    filePath: Optional[str] = None
    channels: int = 32
    sample_rate: int = 30000
    duration_s: int = 2
    metrics: List[str] = ["compression_ratio", "latency_ms", "snr_db", "psnr_db"]
    trials: int = 3


class BenchmarkResult(BaseModel):
    plugin: str
    aggregated_metrics: Dict[str, Dict[str, float]]


class BenchmarkResponse(BaseModel):
    dataset: str
    shape: Tuple[int, int]
    results: List[BenchmarkResult]


class GenerateDataRequest(BaseModel):
    channels: int = 32
    samples: int = 30000
    sample_rate: int = 30000
    noise_level: float = 0.05


class GenerateDataResponse(BaseModel):
    file_path: str
    shape: Tuple[int, int]
    dtype: str
    size_bytes: int


class DecompressRequest(BaseModel):
    artifactPath: str


class CompressOptions(BaseModel):
    plugin: str
    mode: str = "balanced"
    quality_level: float = 0.8
    options: Dict[str, Any] = {}
    session_id: Optional[str] = None
    filePath: Optional[str] = None  # existing file on disk

# --------------------------------------------------------------------------------------
# Utility / Synthetic data
# --------------------------------------------------------------------------------------


def generate_synthetic_neural_data(channels: int, samples: int, sample_rate: int, noise_level: float = 0.05) -> np.ndarray:
    t = np.linspace(0, samples / sample_rate, samples, endpoint=False)
    data = []
    for ch in range(channels):
        f1 = 8 + ch * 0.05
        f2 = 40 + ch * 0.1
        base = 50 * np.sin(2 * np.pi * f1 * t) + 20 * np.sin(2 * np.pi * f2 * t)
        noise = noise_level * 50 * np.random.randn(samples)
        data.append(base + noise)
    return np.asarray(data, dtype=np.float32)


def snr_db(original: np.ndarray, reconstructed: np.ndarray) -> float:
    noise = original - reconstructed
    p_signal = np.mean(original ** 2)
    p_noise = np.mean(noise ** 2) + 1e-12
    return 10 * math.log10(p_signal / p_noise)


def psnr_db(original: np.ndarray, reconstructed: np.ndarray) -> float:
    mse = np.mean((original - reconstructed) ** 2) + 1e-12
    peak = np.max(np.abs(original)) + 1e-12
    return 20 * math.log10(peak / math.sqrt(mse))


def compute_compression_metrics(original: np.ndarray, compressed_bytes: bytes, reconstructed: Optional[np.ndarray], compression_time_s: float) -> Dict[str, Any]:
    original_size = original.nbytes
    compressed_size = len(compressed_bytes)
    ratio = float(original_size) / float(compressed_size) if compressed_size > 0 else 1.0
    metrics: Dict[str, Any] = {
        "compression_ratio": ratio,
        "latency_ms": compression_time_s * 1000.0,
        "original_size": original_size,
        "compressed_size": compressed_size,
    }
    if reconstructed is not None and reconstructed.shape == original.shape:
        try:
            metrics["snr_db"] = snr_db(original, reconstructed)
            metrics["psnr_db"] = psnr_db(original, reconstructed)
        except Exception:  # pragma: no cover
            pass
    return metrics

# --------------------------------------------------------------------------------------
# Plugin discovery / fallback
# --------------------------------------------------------------------------------------


FALLBACK_PLUGINS = [
    {"name": "dummy_lz", "modes": ["fast", "balanced", "quality"], "lossless": True, "lossy": False, "streaming": False, "gpu_accel": False, "version": "0.1"},
    {"name": "dummy_delta", "modes": ["fast", "balanced", "quality"], "lossless": False, "lossy": True, "streaming": True, "gpu_accel": False, "version": "0.1"},
]


class FallbackCompressor:
    def __init__(self, name: str, mode: str = "balanced", quality_level: float = 0.8):
        self.name = name
        self.mode = mode
        self.quality_level = quality_level

    def compress(self, data: np.ndarray):
        if self.name == "dummy_lz":
            reduced = data.astype(np.float16).tobytes()
            meta = {"method": "float16_cast", "original_shape": data.shape, "dtype": str(data.dtype)}
            return reduced, meta
        else:  # delta
            delta = np.diff(data, axis=1, prepend=data[:, :1])
            quant = (delta / (np.std(delta) + 1e-9) * 127).astype(np.int8)
            meta = {"method": "delta_int8", "original_shape": data.shape, "dtype": str(data.dtype)}
            return quant.tobytes(), meta

    def decompress(self, payload: bytes):  # pragma: no cover - dummy
        return None

# --------------------------------------------------------------------------------------
# WebSocket connection manager
# --------------------------------------------------------------------------------------


class ConnectionManager:
    def __init__(self):
        self.connections: List[WebSocket] = []
        self._stream_task: Optional[asyncio.Task] = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.append(websocket)
        if self._stream_task is None or self._stream_task.done():
            self._stream_task = asyncio.create_task(self._stream_metrics())

    def disconnect(self, websocket: WebSocket):
        if websocket in self.connections:
            self.connections.remove(websocket)
        if not self.connections and self._stream_task:
            self._stream_task.cancel()
            self._stream_task = None

    async def broadcast(self, payload: Dict[str, Any]):
        disconnected = []
        for ws in self.connections:
            try:
                await ws.send_text(json.dumps(payload))
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.disconnect(ws)

    async def _stream_metrics(self):
        logger.info("Starting metrics stream task")
        while self.connections:
            try:
                now = time.time()
                metrics = {
                    "type": "metrics",
                    "ts": now,
                    "compression_ratio": float(2.5 + 0.3 * math.sin(now * 0.2) + 0.1 * np.random.randn()),
                    "latency_ms": float(0.5 + abs(0.2 * np.random.randn())),
                    "snr_db": float(25 + 5 * np.random.randn()),
                    "psnr_db": float(35 + 5 * np.random.randn()),
                    "spectral_coherence_error": abs(float(0.05 * np.random.randn())),
                    "gpu_available": GPU_AVAILABLE,
                }
                if np.random.rand() < 0.25:
                    metrics["spike_f1"] = float(0.85 + 0.1 * np.random.randn())
                    metrics["jitter_ms"] = abs(float(0.1 * np.random.randn()))
                await self.broadcast(metrics)
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:  # pragma: no cover
                break
            except Exception as e:  # pragma: no cover
                logger.error(f"Streaming error: {e}")
                await asyncio.sleep(1.0)
        logger.info("Metrics stream task terminated")


manager = ConnectionManager()

# --------------------------------------------------------------------------------------
# Helper functions for artifacts
# --------------------------------------------------------------------------------------


def ensure_session_dir(session_id: str) -> Path:
    d = GUI_ARTIFACTS_DIR / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_artifact(session_id: str, compressed: bytes, original: np.ndarray, plugin: str, mode: str, quality: float, options: Dict[str, Any], comp_meta: Dict[str, Any], metrics: Dict[str, Any]) -> Tuple[Path, Path]:
    session_dir = ensure_session_dir(session_id)
    artifact_path = session_dir / "compressed.npz"
    metadata_path = session_dir / "metadata.json"

    metadata_json = {
        "session_id": session_id,
        "plugin": plugin,
        "mode": mode,
        "quality_level": quality,
        "options": options,
        "original_shape": original.shape,
        "dtype": str(original.dtype),
        "compression_metadata": comp_meta,
        "metrics": metrics,
        "gpu_available": GPU_AVAILABLE,
    }

    np.savez_compressed(
        artifact_path,
        payload=np.frombuffer(compressed, dtype=np.uint8),
        original_shape=np.array(original.shape, dtype=np.int64),
        dtype=np.array(str(original.dtype)),
        metadata_json=np.array(json.dumps(metadata_json))
    )
    with open(metadata_path, 'w') as f:
        json.dump(metadata_json, f, indent=2, default=str)
    return artifact_path, metadata_path

# --------------------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------------------


@app.get("/api/plugins", response_model=List[PluginDescriptor])
async def api_plugins():
    plugins: List[PluginDescriptor] = []
    if BCI_AVAILABLE:
        try:
            names = list_plugins() or []
            for name in names:
                try:
                    plugin_cls = get_plugin(name)
                    caps = getattr(plugin_cls, 'capabilities', lambda: {})()
                    modes = caps.get("modes", ["fast", "balanced", "quality"])
                    if isinstance(modes, str):
                        modes = [modes]
                    plugins.append(PluginDescriptor(
                        name=name,
                        modes=modes,
                        lossless=bool(caps.get("lossless", True)),
                        lossy=bool(caps.get("lossy", False)),
                        streaming=bool(caps.get("streaming", False)),
                        gpu_accel=bool(caps.get("gpu", False) and GPU_AVAILABLE),
                        version=str(getattr(plugin_cls, '__version__', 'unknown'))
                    ))
                except Exception as e:  # pragma: no cover
                    logger.warning(f"Plugin introspection failed for {name}: {e}")
        except Exception as e:  # pragma: no cover
            logger.warning(f"list_plugins failed: {e}")
    if not plugins:
        for p in FALLBACK_PLUGINS:
            plugins.append(PluginDescriptor(**p))
    return plugins


@app.post("/api/compress", response_model=CompressResponse)
async def api_compress(
    plugin: str = Form(...),
    mode: str = Form("balanced"),
    quality_level: float = Form(0.8),
    options: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    filePath: Optional[str] = Form(None)
):
    if not file and not filePath:
        raise HTTPException(status_code=400, detail="Either file or filePath required")
    if file and filePath:
        raise HTTPException(status_code=400, detail="Provide only one of file or filePath")

    try:
        if file:
            content = await file.read()
            if file.filename.endswith('.npy'):
                with tempfile.NamedTemporaryFile(suffix='.npy') as tmp:
                    tmp.write(content)
                    tmp.flush()
                    data = np.load(tmp.name)
            elif H5_AVAILABLE and file.filename.endswith(('.h5', '.hdf5')):
                with tempfile.NamedTemporaryFile(suffix='.h5') as tmp:
                    tmp.write(content)
                    tmp.flush()
                    with h5py.File(tmp.name, 'r') as hf:
                        first_key = list(hf.keys())[0]
                        data = hf[first_key][:]
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format (use .npy or .h5)")
        else:
            if not os.path.exists(filePath):
                raise HTTPException(status_code=404, detail="filePath not found")
            if filePath.endswith('.npy'):
                data = np.load(filePath)
            elif H5_AVAILABLE and filePath.endswith(('.h5', '.hdf5')):
                with h5py.File(filePath, 'r') as hf:
                    first_key = list(hf.keys())[0]
                    data = hf[first_key][:]
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format")

        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim > 2:
            data = data.reshape(data.shape[0], -1)

        opts_dict: Dict[str, Any] = {}
        if options:
            try:
                opts_dict = json.loads(options)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON in options")

        session_id = session_id or str(uuid.uuid4())

        if BCI_AVAILABLE:
            try:
                compressor = create_compressor(plugin)
            except Exception:
                compressor = FallbackCompressor(plugin, mode, quality_level)
        else:
            compressor = FallbackCompressor(plugin, mode, quality_level)

        t0 = time.time()
        if hasattr(compressor, 'compress'):
            compressed, comp_meta = compressor.compress(data)
        else:  # pragma: no cover
            raise HTTPException(status_code=400, detail="Compressor lacks compress method")
        compression_time = time.time() - t0

        if isinstance(compressed, (bytes, bytearray)):
            compressed_bytes = bytes(compressed)
        elif isinstance(compressed, np.ndarray):
            compressed_bytes = compressed.tobytes()
        else:
            compressed_bytes = bytes(compressed)

        reconstructed = None
        if hasattr(compressor, 'decompress'):
            try:
                tmp = compressor.decompress(compressed_bytes)
                if isinstance(tmp, np.ndarray) and tmp.shape == data.shape:
                    reconstructed = tmp
            except Exception:  # pragma: no cover
                pass

        metrics = compute_compression_metrics(data, compressed_bytes, reconstructed, compression_time)
        artifact_path, metadata_path = save_artifact(session_id, compressed_bytes, data, plugin, mode, quality_level, opts_dict, comp_meta if isinstance(comp_meta, dict) else {}, metrics)

        return CompressResponse(
            session_id=session_id,
            artifactPath=str(artifact_path),
            metadataPath=str(metadata_path),
            metrics=metrics
        )
    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover
        logger.exception("Compression error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/decompress", response_model=DecompressResponse)
async def api_decompress(req: DecompressRequest):
    if not os.path.exists(req.artifactPath):
        raise HTTPException(status_code=404, detail="Artifact not found")
    try:
        with np.load(req.artifactPath) as npz:
            payload = npz['payload'].tobytes()
            original_shape = tuple(npz['original_shape'].tolist())
            dtype = str(npz['dtype'].item()) if hasattr(npz['dtype'], 'item') else str(npz['dtype'])
            meta_json = json.loads(str(npz['metadata_json']))
            plugin = meta_json.get("plugin", "dummy_lz")

        if BCI_AVAILABLE:
            try:
                compressor = create_compressor(plugin)
            except Exception:
                compressor = FallbackCompressor(plugin)
        else:
            compressor = FallbackCompressor(plugin)

        t0 = time.time()
        reconstructed = None
        if hasattr(compressor, 'decompress'):
            try:
                reconstructed = compressor.decompress(payload)
            except Exception:  # pragma: no cover
                reconstructed = None
        dec_time = (time.time() - t0) * 1000.0

        if isinstance(reconstructed, np.ndarray) and reconstructed.size == int(np.prod(original_shape)):
            try:
                reconstructed = reconstructed.reshape(original_shape)
            except Exception:  # pragma: no cover
                pass
        else:
            reconstructed = np.zeros(original_shape, dtype=dtype)

        session_id = Path(req.artifactPath).parent.name
        out_path = ensure_session_dir(session_id) / "reconstructed.npy"
        np.save(out_path, reconstructed)
        metrics = {"decompression_time_ms": dec_time, "reconstructed_shape": reconstructed.shape, "reconstructed_dtype": str(reconstructed.dtype)}
        return DecompressResponse(reconstructedPath=str(out_path), metrics=metrics)
    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover
        logger.exception("Decompression error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/benchmark")
async def api_benchmark(req: BenchmarkRequest):
    if req.dataset == "synthetic":
        samples = req.sample_rate * req.duration_s
        data = generate_synthetic_neural_data(req.channels, samples, req.sample_rate)
    elif req.dataset == "file":
        if not req.filePath or not os.path.exists(req.filePath):
            raise HTTPException(status_code=400, detail="filePath required for file dataset")
        if req.filePath.endswith('.npy'):
            data = np.load(req.filePath)
        elif H5_AVAILABLE and req.filePath.endswith(('.h5', '.hdf5')):
            with h5py.File(req.filePath, 'r') as hf:
                first_key = list(hf.keys())[0]
                data = hf[first_key][:]
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format for benchmark")
        if data.ndim == 1:
            data = data.reshape(1, -1)
    else:
        raise HTTPException(status_code=400, detail="Unsupported dataset type")

    results: List[BenchmarkResult] = []
    for plugin in req.plugins:
        ratios: List[float] = []
        latencies: List[float] = []
        snrs: List[float] = []
        psnrs: List[float] = []
        for _ in range(req.trials):
            if BCI_AVAILABLE:
                try:
                    compressor = create_compressor(plugin)
                except Exception:
                    compressor = FallbackCompressor(plugin)
            else:
                compressor = FallbackCompressor(plugin)
            t0 = time.time()
            compressed, _meta = compressor.compress(data)
            elapsed = time.time() - t0
            if isinstance(compressed, (bytes, bytearray)):
                compressed_bytes = bytes(compressed)
            elif isinstance(compressed, np.ndarray):
                compressed_bytes = compressed.tobytes()
            else:
                compressed_bytes = bytes(compressed)
            ratio = data.nbytes / max(1, len(compressed_bytes))
            ratios.append(ratio)
            latencies.append(elapsed * 1000.0)
            if hasattr(compressor, 'decompress'):
                try:
                    recon = compressor.decompress(compressed_bytes)
                    if isinstance(recon, np.ndarray) and recon.shape == data.shape:
                        snrs.append(snr_db(data, recon))
                        psnrs.append(psnr_db(data, recon))
                except Exception:  # pragma: no cover
                    pass
        aggregated: Dict[str, Dict[str, float]] = {}
        if "compression_ratio" in req.metrics:
            aggregated["compression_ratio"] = stats_summary(ratios)
        if "latency_ms" in req.metrics:
            aggregated["latency_ms"] = stats_summary(latencies)
        if "snr_db" in req.metrics and snrs:
            aggregated["snr_db"] = stats_summary(snrs)
        if "psnr_db" in req.metrics and psnrs:
            aggregated["psnr_db"] = stats_summary(psnrs)
        results.append(BenchmarkResult(plugin=plugin, aggregated_metrics=aggregated))

    return {
        "dataset": req.dataset,
        "shape": data.shape,
        "results": [r.dict() for r in results]
    }


def stats_summary(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


@app.get("/api/logs")
async def api_logs(tail: int = Query(100, ge=1, le=1000)):
    if not LOG_FILE.exists():
        return {"logs": [], "tail": tail, "total_lines": 0}
    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
        return {
            "logs": [line.rstrip('\n') for line in lines[-tail:]],
            "tail": tail,
            "total_lines": len(lines)
        }
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-data", response_model=GenerateDataResponse)
async def api_generate_data(req: GenerateDataRequest):
    data = generate_synthetic_neural_data(req.channels, req.samples, req.sample_rate, req.noise_level)
    gen_dir = GUI_ARTIFACTS_DIR / "generated"
    gen_dir.mkdir(exist_ok=True)
    fname = f"synthetic_{req.channels}ch_{req.samples}samp_{int(time.time())}.npy"
    fpath = gen_dir / fname
    np.save(fpath, data)
    return GenerateDataResponse(file_path=str(fpath), shape=data.shape, dtype=str(data.dtype), size_bytes=data.nbytes)


@app.websocket("/ws/metrics")
async def ws_metrics(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                await ws.send_text(json.dumps({"type": "ping", "ts": time.time()}))
            except WebSocketDisconnect:
                break
    finally:
        manager.disconnect(ws)


@app.get("/health")
async def health():
    return {"status": "ok", "gpu_available": GPU_AVAILABLE, "artifact_dir": str(GUI_ARTIFACTS_DIR)}

# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    logger.info("Starting server on 0.0.0.0:8000")
    uvicorn.run("scripts.telemetry_server:app", host="0.0.0.0", port=8000, reload=True)
    return {"status": "ok", "gpu_available": GPU_AVAILABLE, "artifact_dir": str(GUI_ARTIFACTS_DIR)}

# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    logger.info("Starting server on 0.0.0.0:8000")
    uvicorn.run("scripts.telemetry_server:app", host="0.0.0.0", port=8000, reload=True)
