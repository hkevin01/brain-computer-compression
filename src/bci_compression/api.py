# Requires FastAPI and Uvicorn for API server
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .plugins import PLUGIN_REGISTRY

app = FastAPI(title="BCI Compression API", description="API for dashboard/backend integration, including plugin listing.")

# Allow dashboard frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/plugins", summary="List available compressor plugins")
def list_plugins():
    """
    Returns a list of registered plugin names and their docstrings for dashboard integration.
    """
    return [
        {"name": name, "doc": (cls.__doc__ or "").strip()} for name, cls in PLUGIN_REGISTRY.items()
    ]


@app.get("/metrics", summary="Get live compression metrics")
def get_metrics():
    """
    Returns mock (or real) compression metrics for dashboard live metrics display.
    """
    return {
        "compression_ratio": 2.7,
        "latency_ms": 1.3,
        "snr_db": 22.5,
        "power_mw": 85.0
    }

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .plugins import PLUGIN_REGISTRY

app = FastAPI(title="BCI Compression API", description="API for dashboard/backend integration, including plugin listing.")

# Allow dashboard frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/plugins", summary="List available compressor plugins")
def list_plugins():
    """
    Returns a list of registered plugin names and their docstrings for dashboard integration.
    """
    return [
        {"name": name, "doc": (cls.__doc__ or "").strip()} for name, cls in PLUGIN_REGISTRY.items()
    ]


@app.get("/metrics", summary="Get live compression metrics")
def get_metrics():
    """
    Returns mock (or real) compression metrics for dashboard live metrics display.
    """
    return {
        "compression_ratio": 2.7,
        "latency_ms": 1.3,
        "snr_db": 22.5,
        "power_mw": 85.0
    }

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .plugins import PLUGIN_REGISTRY

app = FastAPI(title="BCI Compression API", description="API for dashboard/backend integration, including plugin listing.")

# Allow dashboard frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/plugins", summary="List available compressor plugins")
def list_plugins():
    """
    Returns a list of registered plugin names and their docstrings for dashboard integration.
    """
    return [
        {"name": name, "doc": (cls.__doc__ or "").strip()} for name, cls in PLUGIN_REGISTRY.items()
    ]


@app.get("/metrics", summary="Get live compression metrics")
def get_metrics():
    """
    Returns mock (or real) compression metrics for dashboard live metrics display.
    """
    return {
        "compression_ratio": 2.7,
        "latency_ms": 1.3,
        "snr_db": 22.5,
        "power_mw": 85.0
    }

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .plugins import PLUGIN_REGISTRY

app = FastAPI(title="BCI Compression API", description="API for dashboard/backend integration, including plugin listing.")

# Allow dashboard frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/plugins", summary="List available compressor plugins")
def list_plugins():
    """
    Returns a list of registered plugin names and their docstrings for dashboard integration.
    """
    return [
        {"name": name, "doc": (cls.__doc__ or "").strip()} for name, cls in PLUGIN_REGISTRY.items()
    ]


@app.get("/metrics", summary="Get live compression metrics")
def get_metrics():
    """
    Returns mock (or real) compression metrics for dashboard live metrics display.
    """
    return {
        "compression_ratio": 2.7,
        "latency_ms": 1.3,
        "snr_db": 22.5,
        "power_mw": 85.0
    }

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .plugins import PLUGIN_REGISTRY

app = FastAPI(title="BCI Compression API", description="API for dashboard/backend integration, including plugin listing.")

# Allow dashboard frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/plugins", summary="List available compressor plugins")
def list_plugins():
    """
    Returns a list of registered plugin names and their docstrings for dashboard integration.
    """
    return [
        {"name": name, "doc": (cls.__doc__ or "").strip()} for name, cls in PLUGIN_REGISTRY.items()
    ]


@app.get("/metrics", summary="Get live compression metrics")
def get_metrics():
    """
    Returns mock (or real) compression metrics for dashboard live metrics display.
    """
    return {
        "compression_ratio": 2.7,
        "latency_ms": 1.3,
        "snr_db": 22.5,
        "power_mw": 85.0
    }
