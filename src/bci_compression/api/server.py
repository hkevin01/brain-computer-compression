"""BCI Compression API Server."""

import os
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="BCI Compression API",
    description="Neural data compression toolkit for brain-computer interfaces",
    version="0.8.0",
)

# Enable CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Data models
class CompressionRequest(BaseModel):
    data: List[List[float]]  # 2D array: channels x samples
    algorithm: str = "lz4"
    quality: Optional[float] = None


class CompressionResult(BaseModel):
    compressed_size: int
    original_size: int
    compression_ratio: float
    compression_time: float
    algorithm: str


class SystemInfo(BaseModel):
    backend: str
    device: str
    available_algorithms: List[str]
    gpu_info: Optional[Dict] = None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Detect backend
    backend = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            backend = "cuda"
    except ImportError:
        pass

    try:
        # Check for ROCm
        import subprocess

        result = subprocess.run(["rocm-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            backend = "rocm"
    except (ImportError, FileNotFoundError):
        pass

    return {"status": "healthy", "backend": backend}


@app.get("/info", response_model=SystemInfo)
async def get_system_info():
    """Get system information and capabilities."""
    algorithms = ["lz4", "zstd", "blosc", "transformer", "vae"]

    # Detect backend
    backend = "cpu"
    device = "cpu"
    gpu_info = None

    try:
        import torch

        if torch.cuda.is_available():
            backend = "cuda"
            device = f"cuda:{torch.cuda.current_device()}"
            gpu_info = {
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(),
            }
    except ImportError:
        pass

    try:
        # Check for ROCm
        import subprocess

        result = subprocess.run(["rocm-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            backend = "rocm"
            device = "rocm"
    except (ImportError, FileNotFoundError):
        pass

    return SystemInfo(
        backend=backend,
        device=device,
        available_algorithms=algorithms,
        gpu_info=gpu_info,
    )


@app.post("/compress", response_model=CompressionResult)
async def compress_data(request: CompressionRequest):
    """Compress neural data."""
    try:
        import time

        import numpy as np

        # Convert input to numpy array
        data = np.array(request.data, dtype=np.float32)
        original_size = data.nbytes

        start_time = time.time()

        # Simple compression simulation (replace with actual algorithms)
        if request.algorithm == "lz4":
            import zlib

            compressed = zlib.compress(data.tobytes())
            compressed_size = len(compressed)
        else:
            # Simulate compression for other algorithms
            compressed_size = int(original_size * 0.3)  # 30% compression ratio

        compression_time = time.time() - start_time
        compression_ratio = (
            original_size / compressed_size if compressed_size > 0 else 1.0
        )

        return CompressionResult(
            compressed_size=compressed_size,
            original_size=original_size,
            compression_ratio=compression_ratio,
            compression_time=compression_time,
            algorithm=request.algorithm,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/algorithms")
async def get_algorithms():
    """Get available compression algorithms."""
    return {
        "traditional": ["lz4", "zstd", "blosc", "gzip"],
        "neural": ["transformer", "vae", "autoencoder"],
        "specialized": ["spike_codec", "wavelets", "pca"],
    }


@app.get("/benchmark")
async def run_benchmark(algorithms: str = "lz4,zstd", samples: int = 1000):
    """Run compression benchmark."""
    import time

    import numpy as np

    # Generate synthetic neural data
    channels = 32
    data = np.random.randn(channels, samples).astype(np.float32)

    results = []
    for algo in algorithms.split(","):
        start_time = time.time()

        # Simulate compression
        original_size = data.nbytes
        if algo == "lz4":
            import zlib

            compressed = zlib.compress(data.tobytes())
            compressed_size = len(compressed)
        else:
            compressed_size = int(original_size * np.random.uniform(0.2, 0.5))

        compression_time = time.time() - start_time

        results.append(
            {
                "algorithm": algo,
                "original_size": original_size,
                "compressed_size": compressed_size,
                "compression_ratio": original_size / compressed_size,
                "compression_time": compression_time,
                "throughput_mbps": (original_size / 1024 / 1024) / compression_time,
            }
        )

    return {"benchmark_results": results}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
