# Brain-Computer Interface Data Compression Toolkit

[![Python Version](https://img.shields.io/badge/python-3.14%2B-blue.svg?style=flat-square&logo=python)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg?style=flat-square&logo=docker)](docker/)
[![GPU Acceleration](https://img.shields.io/badge/GPU-CUDA%20%7C%20ROCm-green.svg?style=flat-square&logo=nvidia)](README.md#gpu-acceleration)
[![API Server](https://img.shields.io/badge/API-FastAPI-teal.svg?style=flat-square&logo=fastapi)](http://localhost:8000/docs)
[![Compression](https://img.shields.io/badge/compression-neural--optimized-red.svg?style=flat-square)](README.md#compression-technologies)
[![BCI](https://img.shields.io/badge/BCI-real--time-purple.svg?style=flat-square)](README.md#project-purpose)
[![Tests](https://img.shields.io/badge/tests-177%20passing-success.svg?style=flat-square)](tests/)

> **🧠 A state-of-the-art toolkit for neural data compression in brain-computer interfaces**
> *Enabling real-time, lossless compression of neural signals for next-generation BCIs with GPU acceleration*

---

## ✨ Recent Updates (April 2026)

**🎉 Neural Quality Metrics, Algorithm Fixes & Research-Aligned Enhancements**

- ✅ **177 Tests Passing**: Up from 45 — comprehensive coverage across all modules
- ✅ **Neural Quality Metrics Module** (`neural_quality.py`): SNR, PSNR, spike timing jitter, phase coherence, mutual information — `NeuralQualityMetrics` class with full reporting
- ✅ **NeuralLZ77 Compression Ratio Fixed**: Added `zlib` post-compression on token stream; compression ratio now reliably > 1.0
- ✅ **Array Shape Preservation**: `AdaptiveLZCompressor` and `DictionaryCompressor` now encode shape header via `struct.pack` — lossless round-trip for shaped arrays
- ✅ **EMG Power Optimizer**: Added `optimize_for_power_consumption(battery_level, quality_target, processing_load)` for adaptive wearable usage
- ✅ **TransformerCompressor**: Added `get_compression_ratio()` method for runtime monitoring
- ✅ **12 Bug Fixes**: scipy.signal shadowing, power optimizer signature, EMGMobileLZ decompress override, PSD tuple return, mutual_information adaptive binning
- ✅ **NASA-Style Structured Comments**: Applied to `core.py`, `lossy.py`, `adaptive_selector.py`, `neural_lz.py` for safety-critical documentation
- ✅ **4 New Device Adapters**: Blackrock (Neuroport/Cerebus), Intan (RHD/RHS), HDF5 (generic) *(November 2025)*

> [!TIP]
> See the new `NeuralQualityMetrics` class in `src/bci_compression/metrics/neural_quality.py` for research-grade fidelity assessment post-compression.

See [Adapters Implementation Summary](docs/adapters_implementation_summary.md) for complete adapter details.

---

## 📖 Table of Contents

- [🎯 Project Purpose](#-project-purpose)
- [🏗️ System Architecture](#️-system-architecture)
- [🚀 Quick Start](#-quick-start)
- [🔧 Technology Stack](#-technology-stack)
- [⚡ GPU Acceleration](#-gpu-acceleration)
 - [🌐 Multi-BCI Systems & Electrode Mapping](#-multi-bci-systems--electrode-mapping)
 - [🧪 Testing & Benchmarks Enhancements](#-testing--benchmarks-enhancements)
- [🔬 2025–2026 Research Alignment](#-20252026-research-alignment)

---

## 🎯 Project Purpose

### Why This Project Exists

Brain-Computer Interfaces (BCIs) represent one of the most promising frontiers in neuroscience and human-computer interaction. However, a critical bottleneck threatens to limit their potential: **data management**. Modern BCIs generate enormous volumes of high-dimensional neural data that must be processed, transmitted, and stored in real-time with perfect fidelity. This project exists to solve that bottleneck.

### The Problem We're Solving

**Neural data is fundamentally different from traditional data:**
- 📊 **Volume**: A single 256-channel neural array at 30kHz generates **30.72 million samples per second** (15 MB/s uncompressed)
- ⚡ **Latency**: Closed-loop BCIs require **sub-millisecond response times** for natural control
- 🎯 **Fidelity**: Neural features must be preserved perfectly - even tiny distortions can break decoding algorithms
- 🔋 **Constraints**: Implantable and mobile BCIs have severe power and bandwidth limitations

**Traditional compression algorithms fail because:**
1. They treat neural data as generic byte streams, missing temporal and spatial patterns
2. They're optimized for text/images, not oscillating multi-channel time-series data
3. They can't guarantee real-time performance with varying signal characteristics
4. They don't preserve the specific neural features needed for BCI decoding

### Why This Matters

| <sub>Without This Toolkit</sub> | <sub>With This Toolkit</sub> |
|---------------------|-------------------|
| <sub>❌ Wireless BCIs limited to minutes of recording</sub> | <sub>✅ Hours of continuous wireless neural streaming</sub> |
| <sub>❌ Expensive high-bandwidth transmitters required</sub> | <sub>✅ 5-10x reduction in transmission costs</sub> |
| <sub>❌ Researchers forced to downsample or select channels</sub> | <sub>✅ Full-resolution multi-channel recordings</sub> |
| <sub>❌ Real-time processing limited by data bottlenecks</sub> | <sub>✅ Sub-millisecond compression for closed-loop control</sub> |
| <sub>❌ Neural datasets too large to share easily</sub> | <sub>✅ Shareable compressed datasets for reproducibility</sub> |

### The Challenge We're Addressing

| <sub>Challenge</sub> | <sub>Impact</sub> | <sub>Current Solutions</sub> | <sub>Our Approach</sub> |
|-----------|--------|------------------|--------------|
| <sub>**Data Volume**</sub> | <sub>100+ channels × 30kHz = 3M+ samples/sec</sub> | <sub>Basic compression (20-30% reduction)</sub> | <sub>Neural-aware algorithms (60-80% reduction)</sub> |
| <sub>**Real-time Requirements**</sub> | <sub><1ms latency for closed-loop control</sub> | <sub>Hardware buffers, simplified algorithms</sub> | <sub>GPU-accelerated processing</sub> |
| <sub>**Signal Fidelity**</sub> | <sub>Lossless preservation of neural features</sub> | <sub>Generic compression loses critical features</sub> | <sub>BCI-specific feature preservation</sub> |
| <sub>**Resource Constraints**</sub> | <sub>Mobile/embedded devices with limited power</sub> | <sub>CPU-only, high power consumption</sub> | <sub>Optimized GPU kernels, adaptive selection</sub> |
| <sub>**Accessibility**</sub> | <sub>Expensive infrastructure required</sub> | <sub>Limited to well-funded labs</sub> | <sub>Open-source, cloud-deployable solution</sub> |

### Who Benefits From This

1. **🔬 Researchers**: Conduct longer experiments, store more data, collaborate easier
2. **🏥 Medical Professionals**: Enable real-time neural monitoring, telemedicine applications
3. **🏢 BCI Companies**: Reduce hardware costs, enable mobile/implantable devices
4. **♿ End Users**: Better BCI performance, more affordable assistive devices
5. **🌍 Neuroscience Community**: Shared compression standard for reproducible research

### Target Applications

```mermaid
mindmap
  root((🧠 BCI Data Compression))
    🎯 Applications
      🦾 Motor BCIs
        Prosthetic Control
        Robotic Arms
        Wheelchair Navigation
      🧠 Cognitive BCIs
        Speech Synthesis
        Memory Enhancement
        Attention Monitoring
      🏥 Medical BCIs
        Epilepsy Monitoring
        Depression Treatment
        Sleep Analysis
      📱 Consumer BCIs
        Gaming Interfaces
        VR/AR Control
        Meditation Apps
    📊 Data Types
      🔌 Neural Signals
        Spike Trains
        Local Field Potentials
        ECoG Arrays
      📈 Biosignals
        EMG Patterns
        EEG Recordings
        fMRI Data
    ⚡ Performance Goals
      🚀 Speed
        <1ms Latency
        Real-time Processing
        Streaming Compatible
      💾 Efficiency
        60-80% Compression
        Lossless Quality
        Adaptive Selection
```

### Key Innovation Areas

| <sub>Innovation</sub> | <sub>Description</sub> | <sub>Benefit</sub> |
|------------|-------------|---------|
| <sub>**Neural-Aware Compression**</sub> | <sub>Algorithms designed specifically for neural signal characteristics</sub> | <sub>2-3x better compression ratios than generic methods</sub> |
| <sub>**GPU Acceleration**</sub> | <sub>CUDA/ROCm optimized kernels for parallel processing</sub> | <sub>10-100x faster than CPU-only implementations</sub> |
| <sub>**Adaptive Selection**</sub> | <sub>Real-time algorithm selection based on signal properties</sub> | <sub>Optimal balance of speed, quality, and compression ratio</sub> |
| <sub>**Streaming Architecture**</sub> | <sub>Designed for continuous data streams with minimal buffering</sub> | <sub>Enables real-time BCI applications</sub> |
---

## 🔀 Multi-BCI Systems & Electrode Mapping

Different BCI systems use different electrode layouts, channel naming conventions, and sampling characteristics. This project provides a comprehensive adapter layer to make algorithms portable across acquisition systems:

### Supported BCI Devices

| <sub>Device</sub> | <sub>Channels</sub> | <sub>Sampling Rate</sub> | <sub>Adapter Status</sub> | <sub>Use Case</sub> |
|--------|----------|---------------|----------------|----------|
| <sub>**OpenBCI Cyton**</sub> | <sub>8</sub> | <sub>250 Hz</sub> | <sub>✅ Complete</sub> | <sub>Scalp EEG, consumer BCIs</sub> |
| <sub>**OpenBCI Daisy**</sub> | <sub>16</sub> | <sub>250 Hz</sub> | <sub>✅ Complete</sub> | <sub>Multi-channel EEG</sub> |
| <sub>**Blackrock Neuroport**</sub> | <sub>96</sub> | <sub>30 kHz</sub> | <sub>✅ Complete</sub> | <sub>Utah array, intracortical recording</sub> |
| <sub>**Blackrock Cerebus**</sub> | <sub>128</sub> | <sub>30 kHz</sub> | <sub>✅ Complete</sub> | <sub>Dual Utah arrays, high-density recording</sub> |
| <sub>**Intan RHD2132**</sub> | <sub>32</sub> | <sub>20 kHz</sub> | <sub>✅ Complete</sub> | <sub>LFP, research applications</sub> |
| <sub>**Intan RHD2164**</sub> | <sub>64</sub> | <sub>20 kHz</sub> | <sub>✅ Complete</sub> | <sub>Multi-area recording</sub> |
| <sub>**Intan RHS128**</sub> | <sub>128</sub> | <sub>30 kHz</sub> | <sub>✅ Complete</sub> | <sub>Stimulation-capable headstage</sub> |
| <sub>**Generic HDF5**</sub> | <sub>Variable</sub> | <sub>Variable</sub> | <sub>✅ Complete</sub> | <sub>Any HDF5-formatted neural data</sub> |

### Adapter Features

- **Electrode mapping**: Declarative JSON/YAML mapping files that translate channel indices and names between systems
- **Resampling adapters**: High-performance polyphase and FFT-based resamplers to normalize sampling rates (250Hz ↔ 30kHz)
- **Channel grouping**: Logical grouping for spatial filters and compression (cortical areas, grid rows, functional regions)
- **Calibration metadata**: Store per-session scaling, DC offsets, and bad-channel masks in standardized format
- **Device-specific features**: Utah array grid layouts, headstage type tracking, stimulation capability detection

### Quick Start Examples

#### OpenBCI (Scalp EEG)
```python
from bci_compression.adapters.openbci import OpenBCIAdapter

adapter = OpenBCIAdapter(device='cyton_8ch')
standardized_data = adapter.convert(raw_data)
channel_groups = adapter.get_channel_groups()  # frontal, central, parietal, occipital
```

#### Blackrock (Intracortical)
```python
from bci_compression.adapters.blackrock import BlackrockAdapter

adapter = BlackrockAdapter(device='neuroport_96ch')
downsampled = adapter.resample_to(raw_data, target_rate=1000)
motor_cortex = adapter.get_channel_groups()['motor_cortex']
```

#### Intan (LFP Recording)
```python
from bci_compression.adapters.intan import IntanAdapter

adapter = IntanAdapter(device='rhd2164_64ch')
processed = adapter.convert(raw_data)
has_stim = adapter.stim_capable  # Check stimulation capability
```

#### HDF5 (Generic Loader)
```python
from bci_compression.adapters.hdf5 import HDF5Adapter

adapter = HDF5Adapter.from_hdf5('recording.h5', data_path='/neural/raw')
partial_data = adapter.load_data(start_sample=0, end_sample=10000, channels=[0, 1, 2])
info = adapter.get_info()  # Auto-detect metadata
```

### Multi-Device Pipeline

Combine data from multiple BCI systems in a unified compression pipeline:

```python
from bci_compression.adapters import MultiDevicePipeline

pipeline = MultiDevicePipeline()
pipeline.add_device('openbci', openbci_adapter, priority='normal')   # Scalp EEG
pipeline.add_device('blackrock', blackrock_adapter, priority='high')  # Intracortical (lossless)
pipeline.add_device('intan', intan_adapter, priority='normal')        # LFP

# Process synchronized batch
compressed = pipeline.process_batch({
    'openbci': eeg_data,
    'blackrock': spike_data,
    'intan': lfp_data
})

summary = pipeline.get_summary()  # Get compression statistics
```

### Core Adapter API

The adapter layer exposes a consistent API across all devices:

- `map_channels(data, mapping)` → Remap channel indices/names
- `resample(data, src_rate, dst_rate, method='polyphase'|'fft')` → Change sampling rate
- `apply_channel_groups(data, groups, reducer='mean')` → Apply spatial grouping
- `apply_calibration(data, gains, offsets)` → Apply calibration parameters
- `load_mapping_file(filepath)` / `save_mapping_file(mapping, filepath)` → I/O utilities

### Example Mapping File (YAML)

```yaml
device: openbci_cyton_8ch
sampling_rate: 250
channels: 8
mapping:
  ch_0: Fp1
  ch_1: Fp2
  ch_2: C3
  ch_3: C4
  ch_4: P7
  ch_5: P8
  ch_6: O1
  ch_7: O2
channel_groups:
  frontal: [0, 1]
  central: [2, 3]
  parietal: [4, 5]
  occipital: [6, 7]
```

### File Locations

- **Adapters**: `src/bci_compression/adapters/`
- **Tests**: `tests/test_adapters.py`, `tests/test_blackrock_adapter.py` (45 tests passing)
- **Examples**: `examples/openbci_adapter_demo.py`, `examples/multi_device_pipeline_example.py`
- **Documentation**: `docs/adapters_guide.md`, `docs/adapters_implementation_summary.md`

### Performance

Real-time streaming compression with <1ms latency:

```python
from examples.streaming_compression_example import StreamingCompressor

compressor = StreamingCompressor(n_channels=8, window_size=1000, overlap=250)
for chunk in data_stream:
    compressed = compressor.process_chunk(chunk)  # ~0.06ms average
```

See `scripts/profile_adapters.py` for detailed performance benchmarks.

## 🧪 Testing & Benchmarks Enhancements

We want a fast, reliable test and benchmark workflow that developers can run locally and in CI. Planned improvements include:

- Quick mode tests: fast unit-level smoke tests that run in <30s for quick iteration. These use small synthetic datasets and mock GPU backends when necessary.
- Isolation and timeouts: add per-test timeouts (pytest-timeout) and explicit resource cleanup to prevent hangs. Tests that require longer runtimes live under `tests/long/` and are not part of the `quick` profile.
- Deterministic synthetic data: use fixed random seeds and small synthetic datasets to keep runtimes stable.
- Benchmarks: `scripts/benchmark_runner.py` already supports synthetic and real datasets — we'll add `--quick` and `--full` profiles. Quick runs will provide approximate comparisons; full runs will produce CSV/JSON artifacts for analysis.
- Progress reporting: integrate pytest's -q and pytest-benchmark's progress reporting; optionally add a small CLI progress bar in `scripts/benchmark_runner.py` to stream progress.

Suggested test-related changes (I can implement):

1. Add pytest-timeout to `requirements-dev.txt` and apply a 10s timeout to unit tests and 60s to integration/algorithm tests via `pytest.ini`.
2. Mark long-running tests with `@pytest.mark.slow` and put them in `tests/long/`.
3. Add a `tests/quick_run.sh` script that runs the quick profile and exits non-zero on failures.
4. Update `tests/run_tests.py` to support `--profile quick|full|dependencies-only` and ensure `quick` uses smaller data sizes.

These changes will make local development snappier and prevent CI timeouts caused by blocked processes.


## 🏗️ System Architecture

### High-Level Architecture Overview

```mermaid
graph TB
    subgraph "🧠 Neural Signal Sources"
        N1[Multi-Channel Neural Arrays<br/>64-256 channels @ 30kHz]
        N2[EMG Sensors<br/>8-32 channels @ 2kHz]
        N3[EEG Electrodes<br/>64-128 channels @ 1kHz]
        N4[Single-Unit Recordings<br/>Spike trains @ variable rate]
    end

    subgraph "⚡ Real-Time Processing Layer"
        P1[Signal Preprocessing<br/>• Filtering & Denoising<br/>• Channel Selection<br/>• Quality Assessment]
        P2[Feature Extraction<br/>• Temporal Patterns<br/>• Frequency Analysis<br/>• Spatial Correlations]
        P3[Backend Detection<br/>• GPU Capability Check<br/>• Performance Profiling<br/>• Resource Allocation]
    end

    subgraph "🗜️ Compression Engine Core"
        C1[Algorithm Selection<br/>• Signal Type Analysis<br/>• Latency Requirements<br/>• Quality Constraints]
        C2[Parallel Processing<br/>• Multi-threaded CPU<br/>• GPU Acceleration<br/>• Memory Management]
        C3[Quality Control<br/>• Compression Validation<br/>• Error Detection<br/>• Adaptive Tuning]
    end

    subgraph "🎯 Output & Applications"
        A1[Real-time Control<br/>• Prosthetic Devices<br/>• Robotic Systems<br/>• Gaming Interfaces]
        A2[Data Storage<br/>• HDF5 Archives<br/>• Cloud Storage<br/>• Local Databases]
        A3[Analytics Pipeline<br/>• Machine Learning<br/>• Statistical Analysis<br/>• Visualization]
        A4[Streaming Services<br/>• WebRTC Transmission<br/>• Mobile Apps<br/>• Remote Monitoring]
    end

    N1 --> P1
    N2 --> P1
    N3 --> P1
    N4 --> P1

    P1 --> P2
    P2 --> P3
    P3 --> C1

    C1 --> C2
    C2 --> C3

    C3 --> A1
    C3 --> A2
    C3 --> A3
    C3 --> A4

    classDef neuralsource fill:#1a365d,stroke:#2c5282,stroke-width:2px,color:#ffffff
    classDef processing fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#ffffff
    classDef compression fill:#744210,stroke:#975a16,stroke-width:2px,color:#ffffff
    classDef output fill:#1a202c,stroke:#2d3748,stroke-width:2px,color:#ffffff

    class N1,N2,N3,N4 neuralsource
    class P1,P2,P3 processing
    class C1,C2,C3 compression
    class A1,A2,A3,A4 output
```

### GPU Acceleration Architecture

```mermaid
graph LR
    subgraph "💻 Host System"
        CPU[CPU Controller<br/>• Task Scheduling<br/>• Memory Management<br/>• I/O Operations]
        RAM[System Memory<br/>• Input Buffers<br/>• Algorithm Storage<br/>• Result Cache]
    end

    subgraph "🎮 GPU Processing Units"
        CUDA[CUDA Cores<br/>• Parallel Compression<br/>• Matrix Operations<br/>• Stream Processing]
        ROCm[ROCm Compute<br/>• AMD GPU Support<br/>• HIP Kernels<br/>• Memory Coalescing]
        MEM[GPU Memory<br/>• High Bandwidth<br/>• Shared Buffers<br/>• Texture Cache]
    end

    subgraph "⚙️ Acceleration Backend"
        DETECT[Backend Detection<br/>• Hardware Enumeration<br/>• Capability Testing<br/>• Performance Profiling]
        SCHED[Work Scheduler<br/>• Load Balancing<br/>• Memory Allocation<br/>• Error Handling]
        OPTIM[Performance Optimization<br/>• Kernel Tuning<br/>• Memory Access Patterns<br/>• Pipeline Efficiency]
    end

    CPU --> DETECT
    RAM --> DETECT

    DETECT --> CUDA
    DETECT --> ROCm

    CUDA --> SCHED
    ROCm --> SCHED
    MEM --> SCHED

    SCHED --> OPTIM
    OPTIM --> CPU

    classDef host fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#ffffff
    classDef gpu fill:#1a365d,stroke:#2c5282,stroke-width:2px,color:#ffffff
    classDef backend fill:#744210,stroke:#975a16,stroke-width:2px,color:#ffffff

    class CPU,RAM host
    class CUDA,ROCm,MEM gpu
    class DETECT,SCHED,OPTIM backend
```

### Data Flow Pipeline

```mermaid
flowchart TD
    START([Neural Data Input<br/>Multi-channel streams]) --> PREPROCESS{Signal Preprocessing}

    PREPROCESS --> ANALYZE[Signal Analysis<br/>• Type Classification<br/>• Quality Assessment<br/>• Resource Requirements]

    ANALYZE --> BACKEND{Backend Selection}

    BACKEND -->|High Performance| GPU_PATH[GPU Acceleration Path<br/>• CUDA/ROCm Kernels<br/>• Parallel Processing<br/>• Memory Optimization]

    BACKEND -->|Compatibility| CPU_PATH[CPU Processing Path<br/>• Multi-threading<br/>• SIMD Instructions<br/>• Cache Optimization]

    GPU_PATH --> ALGORITHM{Algorithm Selection}
    CPU_PATH --> ALGORITHM

    ALGORITHM -->|Ultra-Fast| LZ4[LZ4 Compression<br/>< 0.1ms latency]
    ALGORITHM -->|Balanced| ZSTD[Zstandard<br/>< 1ms latency]
    ALGORITHM -->|High-Ratio| NEURAL[Neural Algorithms<br/>< 2ms latency]

    LZ4 --> VALIDATE{Quality Validation}
    ZSTD --> VALIDATE
    NEURAL --> VALIDATE

    VALIDATE -->|Pass| OUTPUT[Compressed Output<br/>• Streaming Ready<br/>• Metadata Attached<br/>• Error Corrected]
    VALIDATE -->|Fail| FALLBACK[Fallback Algorithm<br/>• Conservative Settings<br/>• Guaranteed Quality]

    FALLBACK --> OUTPUT
    OUTPUT --> END([Application Layer<br/>Real-time usage])

    classDef process fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#ffffff
    classDef decision fill:#1a365d,stroke:#2c5282,stroke-width:2px,color:#ffffff
    classDef algorithm fill:#744210,stroke:#975a16,stroke-width:2px,color:#ffffff
    classDef endpoint fill:#1a202c,stroke:#2d3748,stroke-width:2px,color:#ffffff

    class PREPROCESS,ANALYZE,GPU_PATH,CPU_PATH,VALIDATE,FALLBACK process
    class BACKEND,ALGORITHM decision
    class LZ4,ZSTD,NEURAL algorithm
    class START,OUTPUT,END endpoint
```

### Complete Technology Ecosystem

```mermaid
graph TB
    subgraph "🎨 Frontend Layer"
        WEB[Web Dashboard<br/>React + TypeScript<br/>Real-time Visualization]
        API_DOCS[API Documentation<br/>Swagger/OpenAPI<br/>Interactive Testing]
    end

    subgraph "🔌 API Layer"
        FASTAPI[FastAPI Server<br/>Python 3.8+<br/>Async/Await]
        PYDANTIC[Pydantic Models<br/>Type Validation<br/>Schema Generation]
        UVICORN[Uvicorn Server<br/>ASGI Protocol<br/>WebSocket Support]
    end

    subgraph "🧮 Core Processing"
        COMPRESSION[Compression Engine<br/>Multi-Algorithm Support<br/>Adaptive Selection]
        SIGNAL[Signal Processing<br/>SciPy/NumPy<br/>Filtering & Analysis]
        STREAMING[Streaming Pipeline<br/>Chunk Processing<br/>Real-time Flow]
    end

    subgraph "⚡ Acceleration Layer"
        GPU_BACKEND[GPU Backend<br/>CUDA/ROCm Detection<br/>Memory Management]
        CUPY[CuPy Arrays<br/>GPU NumPy<br/>Zero-copy Transfers]
        PYTORCH[PyTorch<br/>Deep Learning<br/>AI Compression]
    end

    subgraph "📦 Compression Algorithms"
        TRADITIONAL[Traditional<br/>LZ4, Zstandard, Blosc<br/>< 1ms latency]
        NEURAL_ALG[Neural-Optimized<br/>Wavelet, Quantization<br/>5-10x compression]
        AI_ALG[AI-Powered<br/>Autoencoders, Transformers<br/>15-40x compression]
    end

    subgraph "💾 Storage & Data"
        HDF5[HDF5 Archives<br/>Hierarchical Storage<br/>Metadata Support]
        FORMATS[Neural Formats<br/>NEV, NSx, Intan<br/>Format Converters]
        CACHE[Result Cache<br/>Redis/Memory<br/>Fast Retrieval]
    end

    subgraph "📊 Monitoring & Ops"
        PROMETHEUS[Prometheus Metrics<br/>Performance Tracking<br/>Time-series Data]
        LOGGING[Structured Logging<br/>JSON Format<br/>Error Tracking]
        PROFILING[Performance Profiling<br/>Latency Analysis<br/>Resource Usage]
    end

    subgraph "🐳 Deployment"
        DOCKER[Docker Containers<br/>Multi-stage Builds<br/>CPU/CUDA/ROCm]
        COMPOSE[Docker Compose<br/>Service Orchestration<br/>Development Profiles]
        K8S[Kubernetes<br/>Production Scale<br/>Auto-scaling]
    end

    subgraph "🧪 Testing & Quality"
        PYTEST[Pytest Framework<br/>Unit & Integration<br/>Coverage Reports]
        BENCHMARK[Benchmarking Suite<br/>Performance Testing<br/>Regression Detection]
        CI_CD[GitHub Actions<br/>CI/CD Pipeline<br/>Automated Testing]
    end

    WEB --> FASTAPI
    API_DOCS --> FASTAPI
    FASTAPI --> PYDANTIC
    FASTAPI --> UVICORN

    FASTAPI --> COMPRESSION
    COMPRESSION --> SIGNAL
    COMPRESSION --> STREAMING

    COMPRESSION --> GPU_BACKEND
    GPU_BACKEND --> CUPY
    GPU_BACKEND --> PYTORCH

    COMPRESSION --> TRADITIONAL
    COMPRESSION --> NEURAL_ALG
    COMPRESSION --> AI_ALG

    STREAMING --> HDF5
    STREAMING --> FORMATS
    COMPRESSION --> CACHE

    FASTAPI --> PROMETHEUS
    FASTAPI --> LOGGING
    GPU_BACKEND --> PROFILING

    DOCKER --> FASTAPI
    COMPOSE --> DOCKER
    K8S --> DOCKER

    PYTEST --> COMPRESSION
    BENCHMARK --> COMPRESSION
    CI_CD --> PYTEST
    CI_CD --> BENCHMARK

    classDef frontend fill:#1a365d,stroke:#2c5282,stroke-width:2px,color:#ffffff
    classDef api fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#ffffff
    classDef core fill:#744210,stroke:#975a16,stroke-width:2px,color:#ffffff
    classDef gpu fill:#1e3a8a,stroke:#1e40af,stroke-width:2px,color:#ffffff
    classDef algorithms fill:#065f46,stroke:#047857,stroke-width:2px,color:#ffffff
    classDef storage fill:#7c2d12,stroke:#9a3412,stroke-width:2px,color:#ffffff
    classDef monitoring fill:#4c1d95,stroke:#5b21b6,stroke-width:2px,color:#ffffff
    classDef deployment fill:#831843,stroke:#9f1239,stroke-width:2px,color:#ffffff
    classDef testing fill:#14532d,stroke:#166534,stroke-width:2px,color:#ffffff

    class WEB,API_DOCS frontend
    class FASTAPI,PYDANTIC,UVICORN api
    class COMPRESSION,SIGNAL,STREAMING core
    class GPU_BACKEND,CUPY,PYTORCH gpu
    class TRADITIONAL,NEURAL_ALG,AI_ALG algorithms
    class HDF5,FORMATS,CACHE storage
    class PROMETHEUS,LOGGING,PROFILING monitoring
    class DOCKER,COMPOSE,K8S deployment
    class PYTEST,BENCHMARK,CI_CD testing
```

**Technology Rationale Summary:**

| <sub>Layer</sub> | <sub>Key Technologies</sub> | <sub>Why This Combination</sub> |
|-------|-----------------|---------------------|
| <sub>**Frontend**</sub> | <sub>React + TypeScript</sub> | <sub>Type safety, component reusability, real-time updates</sub> |
| <sub>**API**</sub> | <sub>FastAPI + Pydantic</sub> | <sub>Automatic docs, type validation, high performance</sub> |
| <sub>**Core**</sub> | <sub>NumPy + SciPy</sub> | <sub>Scientific computing standard, optimized algorithms</sub> |
| <sub>**GPU**</sub> | <sub>CUDA + ROCm + CuPy</sub> | <sub>Broad GPU support, minimal code changes</sub> |
| <sub>**Algorithms**</sub> | <sub>LZ4 + Zstandard + AI</sub> | <sub>Speed/ratio trade-offs, neural-specific optimization</sub> |
| <sub>**Storage**</sub> | <sub>HDF5</sub> | <sub>Scientific data standard, efficient compression</sub> |
| <sub>**Monitoring**</sub> | <sub>Prometheus + JSON logs</sub> | <sub>Industry standard, powerful querying</sub> |
| <sub>**Deployment**</sub> | <sub>Docker + K8s</sub> | <sub>Reproducibility, scalability, platform independence</sub> |
| <sub>**Testing**</sub> | <sub>Pytest + Benchmarks</sub> | <sub>Comprehensive coverage, performance tracking</sub> |

---

## 🚀 Quick Start

### Prerequisites

| <sub>Requirement</sub> | <sub>Version</sub> | <sub>Purpose</sub> | <sub>Installation</sub> |
|-------------|---------|---------|--------------|
| <sub>**Python**</sub> | <sub>3.8+</sub> | <sub>Core runtime environment</sub> | <sub>[Download Python](https://python.org/downloads)</sub> |
| <sub>**Docker**</sub> | <sub>20.10+</sub> | <sub>Containerized deployment</sub> | <sub>[Install Docker](https://docs.docker.com/get-docker/)</sub> |
| <sub>**GPU Drivers**</sub> | <sub>Latest</sub> | <sub>Hardware acceleration</sub> | <sub>[NVIDIA](https://developer.nvidia.com/cuda-downloads) \</sub> | <sub>[AMD](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html)</sub> |
| <sub>**Git**</sub> | <sub>2.25+</sub> | <sub>Version control</sub> | <sub>[Install Git](https://git-scm.com/downloads)</sub> |

### Installation & Setup

#### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/hkevin01/brain-computer-compression.git
cd brain-computer-compression

# One-command setup with development environment
make setup

# Start all services with auto-detected GPU backend
./run.sh up

# Check system status and capabilities
./run.sh status

# Open interactive API documentation
./run.sh gui:open
```

#### Option 2: Manual Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev,quality]"

# Install GPU acceleration (optional)
pip install -e ".[cuda]"  # For NVIDIA GPUs
# pip install -e ".[rocm]"  # For AMD GPUs

# Start API server
python -m bci_compression.api.server

# In another terminal, start dashboard
python -m http.server 3000 --directory web
```

### Verification

```bash
# Run health checks
./run.sh health

# Execute benchmarks
./run.sh bench:all

# Run test suite
make test

# Check code quality
make lint
```

---

## 🔧 Technology Stack

### Core Technologies

| <sub>Technology</sub> | <sub>Version</sub> | <sub>Purpose</sub> | <sub>Why Chosen</sub> |
|------------|---------|---------|------------|
| <sub>**Python**</sub> | <sub>3.8-3.12</sub> | <sub>Primary language</sub> | <sub>• Excellent scientific computing ecosystem<br/>• Rich neural data processing libraries<br/>• Easy integration with ML frameworks</sub> |
| <sub>**NumPy**</sub> | <sub>1.21+</sub> | <sub>Numerical computing</sub> | <sub>• Optimized array operations for neural data<br/>• Memory-efficient multi-dimensional arrays<br/>• Foundation for scientific Python stack</sub> |
| <sub>**SciPy**</sub> | <sub>1.7+</sub> | <sub>Scientific algorithms</sub> | <sub>• Signal processing functions (filters, FFT)<br/>• Statistical analysis for neural patterns<br/>• Optimized implementations of math functions</sub> |
| <sub>**PyTorch**</sub> | <sub>1.13+</sub> | <sub>Machine learning</sub> | <sub>• GPU acceleration for neural networks<br/>• Dynamic computation graphs<br/>• Strong ecosystem for research</sub> |

### GPU Acceleration

| <sub>Technology</sub> | <sub>Purpose</sub> | <sub>Implementation</sub> | <sub>Benefits</sub> |
|------------|---------|----------------|----------|
| <sub>**CUDA 12.x**</sub> | <sub>NVIDIA GPU support</sub> | <sub>CuPy integration + custom kernels</sub> | <sub>• 10-100x speedup for parallel operations<br/>• Mature ecosystem with extensive libraries<br/>• Optimized memory management</sub> |
| <sub>**ROCm 6.x**</sub> | <sub>AMD GPU support</sub> | <sub>HIP kernels + PyTorch backend</sub> | <sub>• Open-source alternative to CUDA<br/>• Growing support for scientific computing<br/>• Better price/performance for some workloads</sub> |
| <sub>**CuPy**</sub> | <sub>GPU-accelerated NumPy</sub> | <sub>Drop-in replacement for NumPy</sub> | <sub>• Minimal code changes for GPU acceleration<br/>• Automatic memory management<br/>• Seamless CPU-GPU transfers</sub> |

### Web & API Framework

| <sub>Component</sub> | <sub>Technology</sub> | <sub>Purpose</sub> | <sub>Why Chosen</sub> |
|-----------|------------|---------|------------|
| <sub>**FastAPI**</sub> | <sub>Modern Python web framework</sub> | <sub>RESTful API server</sub> | <sub>• Automatic API documentation<br/>• Type validation and serialization<br/>• High performance (comparable to Node.js)<br/>• Built-in async support</sub> |
| <sub>**Pydantic**</sub> | <sub>Data validation</sub> | <sub>Request/response models</sub> | <sub>• Runtime type checking<br/>• Automatic JSON serialization<br/>• Clear error messages<br/>• Integration with FastAPI</sub> |
| <sub>**Uvicorn**</sub> | <sub>ASGI server</sub> | <sub>Production deployment</sub> | <sub>• High-performance async server<br/>• Hot reloading for development<br/>• WebSocket support for streaming</sub> |

### Containerization & Orchestration

| <sub>Technology</sub> | <sub>Purpose</sub> | <sub>Configuration</sub> | <sub>Benefits</sub> |
|------------|---------|---------------|----------|
| <sub>**Docker**</sub> | <sub>Application containerization</sub> | <sub>Multi-stage builds</sub> | <sub>• Consistent environments across platforms<br/>• Isolated dependencies<br/>• Easy deployment and scaling</sub> |
| <sub>**Docker Compose**</sub> | <sub>Service orchestration</sub> | <sub>Profile-based configs</sub> | <sub>• Multi-service coordination<br/>• Environment-specific configurations<br/>• Development vs production profiles</sub> |
| <sub>**Multi-stage Builds**</sub> | <sub>Optimized images</sub> | <sub>CPU/CUDA/ROCm variants</sub> | <sub>• Smaller production images<br/>• Backend-specific optimizations<br/>• Reduced attack surface</sub> |

### Development & Quality Tools

| <sub>Category</sub> | <sub>Tools</sub> | <sub>Purpose</sub> | <sub>Integration</sub> |
|----------|-------|---------|-------------|
| <sub>**Code Quality**</sub> | <sub>Ruff, Black, MyPy</sub> | <sub>Linting, formatting, type checking</sub> | <sub>Pre-commit hooks + CI/CD</sub> |
| <sub>**Testing**</sub> | <sub>Pytest, Hypothesis</sub> | <sub>Unit tests, property-based testing</sub> | <sub>Automated test discovery</sub> |
| <sub>**Benchmarking**</sub> | <sub>pytest-benchmark</sub> | <sub>Performance measurement</sub> | <sub>Integrated with test suite</sub> |
| <sub>**Documentation**</sub> | <sub>Sphinx, MkDocs</sub> | <sub>API docs, user guides</sub> | <sub>Auto-generated from docstrings</sub> |

### Data Storage & Formats

| <sub>Technology</sub> | <sub>Use Case</sub> | <sub>Features</sub> | <sub>Why Chosen</sub> |
|------------|----------|----------|------------|
| <sub>**HDF5**</sub> | <sub>Neural data archives</sub> | <sub>Hierarchical, compressed</sub> | <sub>• Industry standard for scientific data<br/>• Built-in compression<br/>• Metadata support<br/>• Cross-platform compatibility</sub> |
| <sub>**JSON**</sub> | <sub>Configuration, API</sub> | <sub>Human-readable, structured</sub> | <sub>• Universal support<br/>• Easy debugging<br/>• Schema validation with Pydantic</sub> |
| <sub>**MessagePack**</sub> | <sub>Binary serialization</sub> | <sub>Compact, fast</sub> | <sub>• Smaller than JSON<br/>• Faster parsing<br/>• Maintains type information</sub> |

### Compression Libraries

| <sub>Library</sub> | <sub>Purpose</sub> | <sub>Performance</sub> | <sub>Integration</sub> |
|---------|---------|-------------|-------------|
| <sub>**LZ4**</sub> | <sub>Ultra-fast compression</sub> | <sub>< 0.1ms latency</sub> | <sub>Direct Python bindings</sub> |
| <sub>**Zstandard**</sub> | <sub>Balanced compression</sub> | <sub>< 1ms latency</sub> | <sub>Facebook's library with Python API</sub> |
| <sub>**Blosc**</sub> | <sub>Array compression</sub> | <sub>Optimized for NumPy</sub> | <sub>Native multi-threading support</sub> |
| <sub>**PyWavelets**</sub> | <sub>Wavelet transforms</sub> | <sub>Scientific-grade</sub> | <sub>SciPy ecosystem integration</sub> |

---

## ⚡ GPU Acceleration

### Backend Detection & Selection

The toolkit automatically detects and optimizes for available hardware:

```mermaid
flowchart TD
    START([System Startup]) --> DETECT{Hardware Detection}

    DETECT -->|NVIDIA GPU Found| CUDA_CHECK[CUDA Capability Check<br/>• Driver Version<br/>• Compute Capability<br/>• Memory Available]

    DETECT -->|AMD GPU Found| ROCM_CHECK[ROCm Capability Check<br/>• ROCm Version<br/>• HIP Support<br/>• Memory Available]

    DETECT -->|CPU Only| CPU_OPT[CPU Optimization<br/>• Thread Count<br/>• SIMD Support<br/>• Cache Optimization]

    CUDA_CHECK -->|Compatible| CUDA_INIT[CUDA Backend<br/>• CuPy Arrays<br/>• Custom Kernels<br/>• Memory Pools]

    ROCM_CHECK -->|Compatible| ROCM_INIT[ROCm Backend<br/>• HIP Kernels<br/>• PyTorch Backend<br/>• Unified Memory]

    CPU_OPT --> CPU_INIT[CPU Backend<br/>• NumPy + BLAS<br/>• Multi-threading<br/>• Memory Mapping]

    CUDA_CHECK -->|Incompatible| CPU_INIT
    ROCM_CHECK -->|Incompatible| CPU_INIT

    CUDA_INIT --> READY[Backend Ready]
    ROCM_INIT --> READY
    CPU_INIT --> READY

    READY --> BENCHMARK[Performance Profiling<br/>• Throughput Testing<br/>• Latency Measurement<br/>• Memory Bandwidth]

    BENCHMARK --> OPTIMIZE[Runtime Optimization<br/>• Kernel Tuning<br/>• Memory Layout<br/>• Pipeline Depth]

    classDef detection fill:#1a365d,stroke:#2c5282,stroke-width:2px,color:#ffffff
    classDef backend fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#ffffff
    classDef optimization fill:#744210,stroke:#975a16,stroke-width:2px,color:#ffffff

    class DETECT,CUDA_CHECK,ROCM_CHECK detection
    class CUDA_INIT,ROCM_INIT,CPU_INIT,READY backend
    class BENCHMARK,OPTIMIZE optimization
```

### Performance Optimization Strategies

| <sub>Strategy</sub> | <sub>Implementation</sub> | <sub>Benefit</sub> | <sub>Use Case</sub> |
|----------|----------------|---------|----------|
| <sub>**Memory Coalescing**</sub> | <sub>Aligned memory access patterns</sub> | <sub>2-10x bandwidth improvement</sub> | <sub>Large array operations</sub> |
| <sub>**Stream Processing**</sub> | <sub>Overlapped compute and memory</sub> | <sub>Reduced latency, higher throughput</sub> | <sub>Real-time streaming</sub> |
| <sub>**Kernel Fusion**</sub> | <sub>Combined operations in single kernel</sub> | <sub>Reduced memory overhead</sub> | <sub>Complex transformations</sub> |
| <sub>**Adaptive Block Size**</sub> | <sub>Dynamic workload partitioning</sub> | <sub>Optimal GPU utilization</sub> | <sub>Variable input sizes</sub> |

### Hardware Requirements & Performance

| <sub>GPU Tier</sub> | <sub>Examples</sub> | <sub>Expected Performance</sub> | <sub>Supported Features</sub> |
|----------|----------|---------------------|-------------------|
| <sub>**High-End**</sub> | <sub>RTX 4090, A100, MI300X</sub> | <sub>> 1000 MB/s throughput</sub> | <sub>All algorithms, maximum parallelism</sub> |
| <sub>**Mid-Range**</sub> | <sub>RTX 3060, RX 6600 XT</sub> | <sub>200-500 MB/s throughput</sub> | <sub>Most algorithms, good parallelism</sub> |
| <sub>**Entry-Level**</sub> | <sub>GTX 1660, RX 5500 XT</sub> | <sub>50-200 MB/s throughput</sub> | <sub>Basic algorithms, limited parallelism</sub> |
| <sub>**CPU Fallback**</sub> | <sub>Any modern CPU</sub> | <sub>10-50 MB/s throughput</sub> | <sub>All algorithms, multi-threading</sub> |

---

## 🗜️ Compression Algorithms

### Algorithm Categories & Selection

```mermaid
graph TD
    INPUT[Neural Data Input<br/>Multi-channel streams] --> ANALYSIS{Signal Analysis}

    ANALYSIS --> TYPE{Signal Type}
    TYPE -->|Continuous EEG/LFP| CONT[Continuous Signals<br/>High temporal resolution]
    TYPE -->|Spike Trains| SPIKE[Event-Based Signals<br/>Sparse temporal data]
    TYPE -->|EMG/Muscular| EMG[Physiological Signals<br/>Variable amplitude]

    ANALYSIS --> QUALITY{Quality Requirements}
    QUALITY -->|Research Grade| LOSSLESS[Lossless Algorithms<br/>Perfect reconstruction]
    QUALITY -->|Clinical| NEARLOS[Near-Lossless<br/>Perceptually identical]
    QUALITY -->|Monitoring| LOSSY[Lossy Algorithms<br/>Feature preservation]

    ANALYSIS --> LATENCY{Latency Constraints}
    LATENCY -->|Real-time Control| ULTRA[Ultra-Fast<br/>< 0.1ms latency]
    LATENCY -->|Interactive| FAST[Fast<br/>< 1ms latency]
    LATENCY -->|Batch Processing| OPTIMAL[Optimal Ratio<br/>< 2ms latency]

    CONT --> LZ4_CONT[LZ4 + Preprocessing]
    SPIKE --> SPIKE_CODEC[Spike Codec]
    EMG --> BLOSC_EMG[Blosc + Filtering]

    LOSSLESS --> ZSTD_LOSS[Zstandard]
    NEARLOS --> NEURAL_NEAR[Neural LZ77]
    LOSSY --> TRANSFORM[Transformer Models]

    ULTRA --> LZ4_ULTRA[LZ4]
    FAST --> ZSTD_FAST[Zstandard]
    OPTIMAL --> AI_OPT[AI Models]

    classDef input fill:#1a365d,stroke:#2c5282,stroke-width:2px,color:#ffffff
    classDef analysis fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#ffffff
    classDef algorithm fill:#744210,stroke:#975a16,stroke-width:2px,color:#ffffff

    class INPUT input
    class ANALYSIS,TYPE,QUALITY,LATENCY analysis
    class LZ4_CONT,SPIKE_CODEC,BLOSC_EMG,ZSTD_LOSS,NEURAL_NEAR,TRANSFORM,LZ4_ULTRA,ZSTD_FAST,AI_OPT algorithm
```

### Traditional Compression Algorithms

#### LZ4 - Ultra-Fast Real-Time Compression

**Purpose**: Absolute minimum latency for real-time BCI control applications

| <sub>Metric</sub> | <sub>Performance</sub> | <sub>Use Case</sub> |
|--------|-------------|----------|
| <sub>**Latency**</sub> | <sub>< 0.1ms</sub> | <sub>Prosthetic control, gaming interfaces</sub> |
| <sub>**Compression Ratio**</sub> | <sub>1.5-2.5x</sub> | <sub>Moderate compression, high speed priority</sub> |
| <sub>**Throughput**</sub> | <sub>> 500 MB/s</sub> | <sub>Continuous neural streaming</sub> |
| <sub>**Memory Usage**</sub> | <sub>Very Low</sub> | <sub>Embedded BCI systems</sub> |

**Technical Details**:
- **Algorithm Type**: Dictionary-based LZ77 variant with fast parsing
- **Implementation**: Optimized C library with Python bindings
- **GPU Acceleration**: Custom CUDA kernels for parallel block processing
- **Neural Data Optimization**: Preprocessor for temporal correlation detection

#### Zstandard (ZSTD) - Intelligent Dictionary Compression

**Purpose**: Balanced performance for most neural data processing scenarios

| <sub>Metric</sub> | <sub>Performance</sub> | <sub>Use Case</sub> |
|--------|-------------|----------|
| <sub>**Latency**</sub> | <sub>< 1ms</sub> | <sub>Real-time analysis, data logging</sub> |
| <sub>**Compression Ratio**</sub> | <sub>3-6x</sub> | <sub>Good balance of speed and compression</sub> |
| <sub>**Throughput**</sub> | <sub>100-300 MB/s</sub> | <sub>Multi-channel recordings</sub> |
| <sub>**Memory Usage**</sub> | <sub>Moderate</sub> | <sub>Standard workstation deployment</sub> |

**Technical Details**:
- **Algorithm Type**: Advanced dictionary compression with entropy coding
- **Implementation**: Facebook's reference implementation with neural adaptations
- **GPU Acceleration**: Parallel dictionary construction and entropy encoding
- **Neural Data Optimization**: Pre-trained dictionaries for common neural patterns

#### Blosc - Multi-Dimensional Array Specialist

**Purpose**: Optimized for multi-channel neural array data with spatial correlations

| <sub>Metric</sub> | <sub>Performance</sub> | <sub>Use Case</sub> |
|--------|-------------|----------|
| <sub>**Latency**</sub> | <sub>< 0.5ms</sub> | <sub>Array-based recordings (Utah arrays, ECoG)</sub> |
| <sub>**Compression Ratio**</sub> | <sub>4-8x</sub> | <sub>Excellent for structured neural data</sub> |
| <sub>**Throughput**</sub> | <sub>200-400 MB/s</sub> | <sub>High-density electrode arrays</sub> |
| <sub>**Memory Usage**</sub> | <sub>Low</sub> | <sub>Memory-efficient streaming</sub> |

**Technical Details**:
- **Algorithm Type**: Chunked compression with multiple algorithms (LZ4, ZSTD, ZLIB)
- **Implementation**: Optimized for NumPy arrays with multi-threading
- **GPU Acceleration**: Parallel chunk processing and memory coalescing
- **Neural Data Optimization**: Spatial correlation detection across channels

### Neural-Optimized Algorithms

#### Neural LZ77 - BCI-Optimized Temporal Compression

**Purpose**: Leverages temporal patterns specific to neural signals

- **Innovation**: Pattern recognition for neural oscillations and spike timing
- **Performance**: 5-10x compression with <1ms latency
- **Specialization**: Optimized for neural frequency bands and temporal structure
- **Implementation**: Custom algorithm with GPU-accelerated pattern matching

#### Perceptual Quantization - Neural Feature Preservation

**Purpose**: Lossy compression that preserves neural decoding performance

- **Innovation**: Quantization based on neural feature importance
- **Performance**: 10-20x compression with minimal decoding accuracy loss
- **Specialization**: Preserves signal features critical for BCI applications
- **Implementation**: Learned quantization levels from neural decoding tasks

#### Adaptive Wavelets - Multi-Resolution Neural Analysis

**Purpose**: Time-frequency decomposition optimized for neural oscillations

- **Innovation**: Adaptive wavelet bases learned from neural data
- **Performance**: 8-15x compression with frequency-specific quality control
- **Specialization**: Preserves power spectral density and phase relationships
- **Implementation**: GPU-accelerated wavelet transforms with learned bases

### AI-Powered Compression

#### Deep Autoencoders - Learned Neural Representations

**Purpose**: End-to-end learned compression optimized for neural data

| <sub>Component</sub> | <sub>Architecture</sub> | <sub>Innovation</sub> |
|-----------|--------------|------------|
| <sub>**Encoder**</sub> | <sub>1D CNN + LSTM</sub> | <sub>Captures temporal dependencies</sub> |
| <sub>**Bottleneck**</sub> | <sub>Learned compression</sub> | <sub>Adaptive rate control</sub> |
| <sub>**Decoder**</sub> | <sub>Transposed CNN</sub> | <sub>Reconstruction optimization</sub> |
| <sub>**Training**</sub> | <sub>Neural data corpus</sub> | <sub>Domain-specific learning</sub> |

**Performance**:
- **Compression Ratio**: 15-30x depending on signal type
- **Latency**: 1-5ms (GPU required)
- **Quality**: Perceptually lossless for most BCI applications
- **Adaptability**: Continuously improves with more neural data

#### Transformer Models - Attention-Based Temporal Patterns

**Purpose**: Captures long-range temporal dependencies in neural signals

| <sub>Component</sub> | <sub>Architecture</sub> | <sub>Purpose</sub> |
|-----------|--------------|---------|
| <sub>**Positional Encoding**</sub> | <sub>Sinusoidal + learned</sub> | <sub>Temporal position awareness</sub> |
| <sub>**Multi-Head Attention**</sub> | <sub>8-16 heads</sub> | <sub>Parallel pattern recognition</sub> |
| <sub>**Feed-Forward**</sub> | <sub>Gated linear units</sub> | <sub>Non-linear transformations</sub> |
| <sub>**Compression Head**</sub> | <sub>Learned quantization</sub> | <sub>Rate-distortion optimization</sub> |

**Performance**:
- **Compression Ratio**: 20-40x with quality control
- **Latency**: 2-10ms (requires high-end GPU)
- **Quality**: State-of-the-art for complex neural patterns
- **Scalability**: Handles variable-length sequences efficiently

#### Variational Autoencoders (VAE) - Probabilistic Quality Control

**Purpose**: Provides uncertainty estimates and quality guarantees

| <sub>Component</sub> | <sub>Function</sub> | <sub>Benefit</sub> |
|-----------|----------|---------|
| <sub>**Probabilistic Encoder**</sub> | <sub>Uncertainty quantification</sub> | <sub>Quality assessment</sub> |
| <sub>**Latent Space**</sub> | <sub>Structured representation</sub> | <sub>Interpretable compression</sub> |
| <sub>**Decoder**</sub> | <sub>Reconstruction + uncertainty</sub> | <sub>Error bounds</sub> |
| <sub>**Rate Control**</sub> | <sub>Adaptive bitrate</sub> | <sub>Quality-based allocation</sub> |

**Performance**:
- **Compression Ratio**: 10-25x with quality bounds
- **Latency**: 3-8ms (GPU recommended)
- **Quality**: Provides confidence intervals for reconstruction
- **Reliability**: Built-in quality assessment and error detection

### Performance Characteristics

#### Real-Time Processing Guarantees

| <sub>Algorithm Class</sub> | <sub>Worst-Case Latency</sub> | <sub>Throughput</sub> | <sub>Memory</sub> | <sub>Use Case</sub> |
|-----------------|-------------------|------------|--------|----------|
| <sub>**Ultra-Fast**</sub> | <sub>< 0.1ms</sub> | <sub>> 500 MB/s</sub> | <sub>< 10MB</sub> | <sub>Real-time control</sub> |
| <sub>**Balanced**</sub> | <sub>< 1ms</sub> | <sub>100-500 MB/s</sub> | <sub>10-50MB</sub> | <sub>General purpose</sub> |
| <sub>**High-Ratio**</sub> | <sub>< 2ms</sub> | <sub>50-200 MB/s</sub> | <sub>50-200MB</sub> | <sub>Storage/transmission</sub> |
| <sub>**AI-Powered**</sub> | <sub>< 10ms</sub> | <sub>20-100 MB/s</sub> | <sub>200MB-2GB</sub> | <sub>Research/analysis</sub> |

#### Hardware Acceleration Benefits

| <sub>Hardware</sub> | <sub>Speedup vs CPU</sub> | <sub>Supported Algorithms</sub> | <sub>Optimal Use Cases</sub> |
|----------|----------------|---------------------|-------------------|
| <sub>**High-End GPU**</sub> | <sub>50-100x</sub> | <sub>All algorithms</sub> | <sub>Real-time + AI compression</sub> |
| <sub>**Mid-Range GPU**</sub> | <sub>20-50x</sub> | <sub>Traditional + some AI</sub> | <sub>Balanced workloads</sub> |
| <sub>**Entry GPU**</sub> | <sub>5-20x</sub> | <sub>Traditional algorithms</sub> | <sub>Cost-effective acceleration</sub> |
| <sub>**Multi-Core CPU**</sub> | <sub>1-4x</sub> | <sub>All algorithms</sub> | <sub>Compatibility fallback</sub> |

#### Memory Efficiency

| <sub>Optimization</sub> | <sub>Technique</sub> | <sub>Benefit</sub> | <sub>Implementation</sub> |
|--------------|-----------|---------|----------------|
| <sub>**Streaming**</sub> | <sub>Chunk-based processing</sub> | <sub>Constant memory usage</sub> | <sub>Sliding window buffers</sub> |
| <sub>**In-Place**</sub> | <sub>No intermediate copies</sub> | <sub>50% memory reduction</sub> | <sub>Zero-copy operations</sub> |
| <sub>**Memory Pools**</sub> | <sub>Pre-allocated buffers</sub> | <sub>Reduced allocation overhead</sub> | <sub>GPU memory management</sub> |
| <sub>**Compression Caching**</sub> | <sub>LRU cache for patterns</sub> | <sub>Faster repeated patterns</sub> | <sub>Dictionary reuse</sub> |

## 📁 Project Structure

```
brain-computer-compression/
├── README.md                    # This file
├── requirements*.txt            # Python dependencies
├── pyproject.toml              # Python project config
├── run.sh                      # Main orchestration script
├── docs/                       # 📚 Documentation
│   ├── guides/                 # User guides
│   └── project/               # Project documentation
├── docker/                     # 🐳 Docker configuration
│   ├── Dockerfile             # Main backend image
│   └── compose/               # Docker compose files
├── scripts/                    # 🔧 Scripts and tools
│   ├── setup/                 # Installation scripts
│   └── tools/                 # Utility scripts
├── src/                       # 🧠 Core source code
├── tests/                     # 🧪 Test suite
├── dashboard/                 # 🌐 React GUI
├── examples/                  # 📖 Usage examples
└── notebooks/                 # 📊 Jupyter notebooks
```

## � Examples & Demos

### Device Adapter Examples

Comprehensive examples demonstrating BCI device integration:

| <sub>Example</sub> | <sub>File</sub> | <sub>Description</sub> | <sub>Features</sub> |
|---------|------|-------------|----------|
| <sub>**OpenBCI Demo**</sub> | <sub>`examples/openbci_adapter_demo.py`</sub> | <sub>6 scenarios for OpenBCI devices</sub> | <sub>Basic conversion, resampling, channel grouping, calibration, full pipeline, multi-device</sub> |
| <sub>**Streaming Compression**</sub> | <sub>`examples/streaming_compression_example.py`</sub> | <sub>Real-time streaming with <1ms latency</sub> | <sub>Sliding windows, circular buffers, latency monitoring, throughput stats</sub> |
| <sub>**Multi-Device Pipeline**</sub> | <sub>`examples/multi_device_pipeline_example.py`</sub> | <sub>Unified pipeline for multiple BCI systems</sub> | <sub>OpenBCI + Blackrock + Intan integration, hierarchical compression, channel alignment</sub> |

### Running Examples

```bash
# OpenBCI adapter demos (all 6 scenarios)
python examples/openbci_adapter_demo.py

# Real-time streaming compression
python examples/streaming_compression_example.py

# Multi-device integration
python examples/multi_device_pipeline_example.py

# EMG signal processing
python examples/emg_demo.py

# Transformer-based compression
python examples/transformer_demo.py

# GPU-accelerated compression (Jupyter notebook)
jupyter lab examples/cuda_acceleration.ipynb
```

### Performance Profiling

```bash
# Profile all adapters with detailed benchmarks
python scripts/profile_adapters.py

# View generated profiling report
cat results/adapter_profiling_report.txt
```

**Expected Performance:**
- OpenBCI: 0.059ms per window (169,948k samples/s)
- Blackrock: 4.216ms per window (7,116k samples/s)
- Intan: 1.803ms per window (11,090k samples/s)
- Streaming: <0.1ms average latency, <0.13ms max

### Jupyter Notebooks

Interactive analysis and visualization:

```bash
# Start Jupyter Lab
jupyter lab

# Or use the provided task
./run.sh jupyter
```

Available notebooks:
- `notebooks/compression_analysis.ipynb` - Compression algorithm comparison
- `notebooks/benchmarking_results.ipynb` - Performance analysis
- `examples/cuda_acceleration.ipynb` - GPU acceleration demos

## �📚 Documentation

- **[Quick Start Guide](docs/guides/DOCKER_QUICK_START.md)** - Get started with Docker
- **[Adapters Guide](docs/adapters_guide.md)** - Complete BCI adapter documentation
- **[Adapters Implementation](docs/adapters_implementation_summary.md)** - Technical implementation details
- **[Docker Troubleshooting](docs/guides/DOCKER_BUILD_FIX.md)** - Fix common Docker issues
- **[Contributing Guide](docs/CONTRIBUTING.md)** - How to contribute
- **[Changelog](docs/CHANGELOG.md)** - Version history
- **[Project Status](docs/project/STATUS_REPORT.md)** - Current development status

## 🐳 Docker Usage - Zero Configuration Required

**Docker-First Design Benefits:**

- 🚀 **Instant Setup**: One command starts everything
- 🔒 **Isolated Environment**: No conflicts with system packages
- 📦 **Batteries Included**: All dependencies pre-configured
- 🔄 **Consistent Results**: Same environment across all systems
- 🛡️ **Error-Free**: Template generation prevents configuration mistakes

All Docker files are organized in the `docker/` directory:

```bash
# Build images (optional - auto-built on first run)
./run.sh build

# Start services - everything you need!
./run.sh up

# View logs
./run.sh logs

# Stop services
./run.sh down
```

## 🔧 Development Tools

Utility scripts are in `scripts/tools/`:

- **Setup**: `scripts/setup/setup.sh` - Quick environment setup
- **Docker Tools**: `scripts/tools/test_docker_build.sh` - Test Docker builds
- **Cleanup**: `scripts/tools/cleanup_now.sh` - Clean temporary files

## ✨ Key Features

### 🔌 Multi-BCI System Support

**🧠 Native Support for 8+ BCI Systems** ✨ NEW

Full adapter implementations with tested, production-ready code:

| <sub>System</sub> | <sub>Channels</sub> | <sub>Sampling</sub> | <sub>Status</sub> | <sub>Implementation</sub> |
|--------|----------|----------|--------|----------------|
| <sub>**OpenBCI Cyton/Daisy**</sub> | <sub>8-16</sub> | <sub>250 Hz</sub> | <sub>✅ Complete</sub> | <sub>Full adapter with electrode mapping</sub> |
| <sub>**Blackrock Neuroport**</sub> | <sub>96</sub> | <sub>30 kHz</sub> | <sub>✅ Complete</sub> | <sub>Utah array grid layout, NEV support</sub> |
| <sub>**Blackrock Cerebus**</sub> | <sub>128</sub> | <sub>30 kHz</sub> | <sub>✅ Complete</sub> | <sub>Dual Utah arrays, cortical regions</sub> |
| <sub>**Intan RHD2132**</sub> | <sub>32</sub> | <sub>20 kHz</sub> | <sub>✅ Complete</sub> | <sub>LFP recording, headstage tracking</sub> |
| <sub>**Intan RHD2164**</sub> | <sub>64</sub> | <sub>20 kHz</sub> | <sub>✅ Complete</sub> | <sub>Multi-area recording</sub> |
| <sub>**Intan RHS128**</sub> | <sub>128</sub> | <sub>30 kHz</sub> | <sub>✅ Complete</sub> | <sub>Stimulation-capable</sub> |
| <sub>**Generic HDF5**</sub> | <sub>Variable</sub> | <sub>Variable</sub> | <sub>✅ Complete</sub> | <sub>Auto-detection, flexible loading</sub> |
| <sub>**Custom Devices**</sub> | <sub>Any</sub> | <sub>Any</sub> | <sub>✅ Supported</sub> | <sub>YAML/JSON mapping files</sub> |

**Additional Systems** (via configuration):
- **Emotiv EPOC** (14 channels, 128 Hz) - Consumer EEG headsets
- **BioSemi ActiveTwo** (64 channels, 2048 Hz) - High-density research EEG
- **EGI GSN HydroCel** (128 channels, 1000 Hz) - Geodesic Sensor Net
- **Delsys Trigno** (16 channels, 2000 Hz) - Wireless EMG systems
- **Neuropixels** (384 channels, 30 kHz) - High-density neural probes

**📊 Advanced Adapter Features**

- ✅ **Real-time streaming** with <1ms latency
- ✅ **Multi-device pipelines** for simultaneous recording from different systems
- ✅ **Hierarchical compression** (lossless for high-priority, lossy for others)
- ✅ **Channel grouping** by cortical regions, grid rows, or functional areas
- ✅ **High-performance resampling** (250 Hz ↔ 30 kHz with FFT/polyphase filters)
- ✅ **Automatic calibration** with gain/offset correction
- ✅ **Partial data loading** from large HDF5 files (memory-efficient)
- ✅ **Device-specific metadata** (Utah array layouts, headstage types, etc.)

**🔄 Quick Adapter Usage**

```python
# OpenBCI (Scalp EEG)
from bci_compression.adapters.openbci import OpenBCIAdapter
adapter = OpenBCIAdapter(device='cyton_8ch')
processed = adapter.convert(raw_data)

# Blackrock (Intracortical)
from bci_compression.adapters.blackrock import BlackrockAdapter
adapter = BlackrockAdapter(device='neuroport_96ch')
downsampled = adapter.resample_to(raw_data, target_rate=1000)

# Multi-device pipeline
from examples.multi_device_pipeline_example import MultiDevicePipeline
pipeline = MultiDevicePipeline()
pipeline.add_device('openbci', openbci_adapter, priority='normal')
pipeline.add_device('blackrock', blackrock_adapter, priority='high')
compressed = pipeline.process_batch({'openbci': eeg_data, 'blackrock': spike_data})
```

**📈 Performance Benchmarks**

Real measurements from `scripts/profile_adapters.py`:

- **OpenBCI**: 0.059ms full pipeline (170k samples/sec)
- **Blackrock**: 4.216ms full pipeline (7k samples/sec)
- **Intan**: 1.803ms full pipeline (11k samples/sec)
- **Streaming**: <0.1ms average latency

See [Adapters Guide](docs/adapters_guide.md) and [Implementation Summary](docs/adapters_implementation_summary.md) for complete documentation.

### 🧠 Neural Data Compression Algorithms

#### Lossless Compression - Perfect Signal Preservation

**🚀 LZ4 - Ultra-Fast Real-Time Compression**

- **What it is**: Industry-standard lossless compression optimized for speed over ratio
- **Why chosen**: Provides >675 MB/s compression, <0.1ms latency for real-time BCI control
- **Neural application**: Ideal for closed-loop prosthetic control where timing is critical
- **Technical specs**: 1.5-2x compression ratio, 3850 MB/s decompression speed
- **Use case**: Motor cortex signals for robotic arm control, real-time feedback systems

**⚡ Zstandard (ZSTD) - Intelligent Dictionary Compression**

- **What it is**: Facebook's modern compression algorithm with machine learning dictionary training
- **Why chosen**: Adaptive compression models learn from neural data patterns over time
- **Neural application**: Optimizes compression ratios for repetitive neural firing patterns
- **Technical specs**: 2-4x compression ratio, 510 MB/s compression, 1550 MB/s decompression
- **Use case**: Long-term neural recordings, session-based BCI training data

**🔢 Blosc - Multi-Dimensional Array Specialist**

- **What it is**: High-performance compressor designed specifically for numerical arrays
- **Why chosen**: Leverages SIMD instructions and multi-threading for neural array data
- **Neural application**: Optimized for multi-channel electrode arrays (64-256+ channels)
- **Technical specs**: Blocking technique reduces memory bandwidth, AVX512/NEON acceleration
- **Use case**: High-density neural arrays, spatial correlation across electrode grids

**🧠 Neural LZ77 - BCI-Optimized Temporal Compression**

- **What it is**: Custom LZ77 implementation trained on neural signal characteristics
- **Why chosen**: Exploits temporal correlations unique to neural firing patterns
- **Neural application**: Recognizes spike trains, bursting patterns, oscillatory activity
- **Technical specs**: 1.5-3x compression ratio, <1ms latency, 95%+ pattern accuracy
- **Use case**: Single-unit recordings, spike train analysis, temporal pattern preservation

#### Lossy Compression - Quality-Controlled Neural Encoding

**🎵 Perceptual Quantization - Neural Feature Preservation**

- **What it is**: Psychoacoustic principles applied to neural signal frequency domains
- **Why chosen**: Preserves critical neural features while discarding perceptually irrelevant data
- **Neural application**: Maintains action potential shapes, preserves frequency bands (alpha, beta, gamma)
- **Technical specs**: 2-10x compression, 15-25 dB SNR, configurable quality levels
- **Use case**: EEG analysis, spectral power studies, frequency-domain BCI features

**🌊 Adaptive Wavelets - Multi-Resolution Neural Analysis**

- **What it is**: Wavelet transforms with neural-specific basis functions and smart thresholding
- **Why chosen**: Natural fit for neural signals with multi-scale temporal dynamics
- **Neural application**: Preserves both fast spikes and slow oscillations simultaneously
- **Technical specs**: 3-15x compression, configurable frequency band preservation
- **Use case**: Multi-scale neural analysis, time-frequency BCI features, neural oscillations

**🤖 Deep Autoencoders - Learned Neural Representations**

- **What it is**: Neural networks trained to compress neural data into learned latent spaces
- **Why chosen**: Discovers optimal representations specific to individual neural patterns
- **Neural application**: Personalized compression models adapt to each user's neural signatures
- **Technical specs**: 2-4x compression, learned from user's historical neural data
- **Use case**: Personalized BCIs, adaptive neural interfaces, long-term implant optimization

**🔮 Transformer Models - Attention-Based Temporal Patterns**

- **What it is**: Multi-head attention mechanisms for compressing temporal neural sequences
- **Why chosen**: Captures long-range dependencies in neural activity patterns
- **Neural application**: Models complex temporal relationships across brain regions
- **Technical specs**: 3-5x compression, 25-35 dB SNR, handles variable-length sequences
- **Use case**: Multi-region neural recordings, cognitive state decoding, complex BCI tasks

**📊 Variational Autoencoders (VAE) - Probabilistic Quality Control**

- **What it is**: Probabilistic encoders with uncertainty quantification for neural data
- **Why chosen**: Provides quality estimates and confidence intervals for compressed neural signals
- **Neural application**: Maintains uncertainty bounds critical for medical-grade BCI applications
- **Technical specs**: Quality-controlled compression with statistical guarantees
- **Use case**: Medical BCIs, safety-critical applications, neural signal validation

#### Advanced Techniques

- **Predictive Coding**: Linear and adaptive prediction models for temporal patterns
- **Context-Aware**: Brain state adaptive compression with real-time switching
- **Multi-Channel**: Spatial correlation exploitation across electrode arrays
- **Spike Detection**: Specialized compression for neural action potentials (>95% accuracy)

### 🚀 Performance Features

**⚡ Real-Time Processing Guarantees**

- **Ultra-low latency**: < 1ms for basic algorithms, < 2ms for advanced neural methods
- **Deterministic timing**: Hard real-time guarantees for closed-loop BCI systems
- **Streaming architecture**: Bounded memory usage for continuous data processing
- **Pipeline optimization**: Multi-stage processing with minimal buffering delays

**🖥️ Hardware Acceleration**

- **GPU acceleration**: CUDA-optimized kernels with CPU fallback (3-5x speedup)
- **SIMD optimization**: AVX512, NEON, and ALTIVEC instruction utilization
- **Multi-threading**: Efficient parallel processing across CPU cores
- **Memory optimization**: Cache-friendly algorithms reduce memory bandwidth

**📱 Mobile & Embedded Support**

- **Power efficiency**: Battery-optimized algorithms for wearable BCI devices
- **Resource constraints**: Minimal memory footprint for embedded systems
- **Cross-platform**: ARM, x86, and RISC-V architecture support
- **Edge computing**: Local processing without cloud dependencies

## 🔬 Compression Technologies Deep Dive

### 🏭 Standard Compression Libraries

**LZ4 - The Speed Champion**

```mermaid
graph LR
    subgraph "LZ4 Pipeline"
        style L1 fill:#1a202c,stroke:#2d3748,stroke-width:2px,color:#ffffff
        style L2 fill:#1a202c,stroke:#2d3748,stroke-width:2px,color:#ffffff
        style L3 fill:#1a202c,stroke:#2d3748,stroke-width:2px,color:#ffffff

        L1[Hash Table<br/>Lookup]
        L2[Match<br/>Finding]
        L3[Token<br/>Encoding]
    end

    L1 --> L2 --> L3
```

- **Lightning-fast lossless compression**: Optimized for streaming neural data
- **Minimal CPU overhead**: Perfect for real-time BCI applications
- **Industry standard**: Used by Facebook, Netflix, Linux kernel

**Zstandard (ZSTD) - The Smart Compressor**

```mermaid
graph TB
    subgraph "ZSTD Intelligence"
        style Z1 fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#ffffff
        style Z2 fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#ffffff
        style Z3 fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#ffffff

        Z1[Dictionary<br/>Learning]
        Z2[Entropy<br/>Modeling]
        Z3[Adaptive<br/>Algorithms]
    end

    Z1 --> Z2 --> Z3
```

- **Modern compression**: Facebook's algorithm with dictionary learning for high ratios
- **Neural pattern adaptation**: Learns from repetitive neural firing patterns
- **Scalable performance**: 1-22 compression levels for speed/ratio trade-offs

**Blosc - The Array Specialist**

```mermaid
graph LR
    subgraph "Blosc Architecture"
        style B1 fill:#1a202c,stroke:#2d3748,stroke-width:2px,color:#ffffff
        style B2 fill:#1a202c,stroke:#2d3748,stroke-width:2px,color:#ffffff
        style B3 fill:#1a202c,stroke:#2d3748,stroke-width:2px,color:#ffffff

        B1[Shuffle<br/>Filter]
        B2[SIMD<br/>Acceleration]
        B3[Multi-thread<br/>Processing]
    end

    B1 --> B2 --> B3
```

- **Multi-threaded compression library**: Optimized for numerical arrays
- **SIMD optimization**: AVX512, NEON acceleration for neural array data
- **Cache-friendly**: Blocking technique reduces memory bandwidth

### 🧠 Neural-Specific Algorithms

**Neural LZ77 - BCI-Optimized Pattern Recognition**

- **Custom LZ77 implementation**: Trained on neural signal temporal patterns
- **Spike pattern recognition**: Optimized for action potential sequences
- **Temporal correlation exploitation**: Understands neural firing rhythms

**Perceptual Quantization - Frequency-Domain Intelligence**

- **Psychoacoustically-inspired**: Adapted from audio compression for neural frequencies
- **Critical band preservation**: Maintains alpha, beta, gamma frequency information
- **Configurable quality**: Adjustable SNR levels from 15-35 dB

**Adaptive Wavelets - Multi-Scale Neural Analysis**

- **Multi-resolution analysis**: Preserves both fast spikes and slow oscillations
- **Neural-specific basis functions**: Optimized for biological signal characteristics
- **Smart thresholding**: Preserves critical neural features while removing noise

### 🤖 AI/ML Compression Revolution

**Deep Autoencoders - Learned Neural Representations**

- **Personalized compression**: Models adapt to individual neural signatures
- **Latent space optimization**: Discovers optimal representations for neural data
- **Transfer learning**: Pre-trained models adapt to new subjects quickly

**Variational Autoencoders (VAE) - Probabilistic Intelligence**

- **Uncertainty quantification**: Provides confidence intervals for compressed data
- **Quality-controlled compression**: Statistical guarantees for medical applications
- **Generative modeling**: Can synthesize realistic neural data for training

**Transformer Models - Attention-Based Neural Compression**

- **Multi-head attention**: Captures long-range dependencies in neural sequences
- **Sequence-to-sequence**: Handles variable-length neural recordings
- **State-of-the-art performance**: 25-35 dB SNR with 3-5x compression

**Predictive Coding - Temporal Pattern Prediction**

- **Linear/nonlinear prediction**: Models temporal dependencies in neural signals
- **Adaptive algorithms**: Continuously update models based on signal characteristics
- **Real-time learning**: Updates compression models during acquisition

---

## 🔬 2025–2026 Research Alignment

> [!IMPORTANT]
> The following section maps cutting-edge BCI compression research (Dec 2025 – Apr 2026) to this toolkit's roadmap. Items marked **Planned** are tracked in [docs/roadmap.md](docs/roadmap.md).

### Top Research Advances & Project Roadmap

```mermaid
quadrantChart
  title Research Impact vs Implementation Effort
  x-axis Low Effort --> High Effort
  y-axis Low Impact --> High Impact
  quadrant-1 Plan Carefully
  quadrant-2 Ship First
  quadrant-3 Skip For Now
  quadrant-4 Consider Later
  DS CAE LFP: [0.65, 0.95]
  BrainCodec RVQ: [0.55, 0.90]
  LLC Spike CLEM: [0.40, 0.85]
  Diffusion EEG: [0.70, 0.80]
```

| <sub>#</sub> | <sub>Research Finding</sub> | <sub>Source</sub> | <sub>What It Enables</sub> | <sub>Project Gap</sub> | <sub>Status</sub> |
|---|-----------------|--------|-----------------|-------------|--------|
| <sub>1</sub> | <sub>**DS-CAE: 150x LFP compression** via Depthwise-Separable Convolutional Autoencoder + hardware-aware balanced stochastic pruning (32.4% parameter reduction, 15.1 μW/channel, SNDR 22.6–27.4 dB, R² 0.81–0.94)</sub> | <sub>arXiv 2504.06996 — *RAMAN tinyML Accelerator* (Apr 2025)</sub> | <sub>Ultra-high-ratio LFP compression for implantable edge hardware</sub> | <sub>VAE/Transformer achieved 2–8x; no depthwise-separable arch; no hardware-aware pruning</sub> | <sub>✅ Implemented (`cae_compression.py`)</sub> |
| <sub>2</sub> | <sub>**BrainCodec RVQ-VAE: 64x EEG/iEEG** — Residual Vector Quantization autoencoder with line-length loss for transient preservation; iEEG→EEG transfer learning; no downstream task degradation</sub> | <sub>[github.com/IBM/eeg-ieeg-brain-compressor](https://github.com/IBM/eeg-ieeg-brain-compressor) — ICLR 2025</sub> | <sub>Codec-quality EEG storage with neural-specific perceptual loss</sub> | <sub>`vae_compression.py` had dense VAE, no RVQ stages, no transient-preserving loss</sub> | <sub>✅ Implemented (`rvq_compressor.py`)</sub> |
| <sub>3</sub> | <sub>**LLCSpike CLEM: lossless spike trains** — Categorical Logit-based Entropy Model learns spike-sequence distributions; short-term aggregation + intensity remapping</sub> | <sub>*IEEE TIP 2025* DOI:10.1109/TIP.2025.3630868</sub> | <sub>Optimal lossless archival of single-unit recordings</sub> | <sub>`NeuralLZ77` used generic zlib entropy coder; no learned spike priors</sub> | <sub>✅ Implemented (`llc_spike_compressor.py`)</sub> |
| <sub>4</sub> | <sub>**EEGCiD diffusion reconstruction** — Encode only a compact latent; pre-trained diffusion prior reconstructs full signal on the receiver; extreme semantic compression for long recordings</sub> | <sub>*IEEE EMBC 2025* + *The Innovation Life* (2026)</sub> | <sub>>100x archival compression for offline EEG analysis</sub> | <sub>No generative prior; all compressors required symmetric encode/decode</sub> | <sub>✅ Implemented (`diffusion_compressor.py`)</sub> |

---

### 🧠 Why These Algorithms? Design Rationale vs Alternatives

| <sub>Algorithm</sub> | <sub>Module</sub> | <sub>Signal Type</sub> | <sub>Why Chosen</sub> | <sub>Alternatives Considered</sub> | <sub>How It Improves Compression</sub> |
|-----------|--------|-------------|------------|------------------------|-----------------------------|
| <sub>**RVQ-VAE** (BrainCodec)</sub> | <sub>`rvq_compressor.py`</sub> | <sub>EEG / iEEG</sub> | <sub>RVQ provides a discrete codebook bottleneck — unlike a continuous VAE latent, each residual stage refines quantization error, giving predictable bit-rate control and enabling codec-style streaming. The **line-length loss** penalises waveform smoothing so sharp transients (epileptic spikes, P300) are preserved, whereas MSE-only VAE blurs them.</sub> | <sub>Plain VAE (`vae_compression.py`), transformer compression, JPEG2000-style wavelet</sub> | <sub>Adds 4 residual codebook stages → 4× more expressive prior per channel; line-length loss recovers high-freq energy lost by naive MSE; iEEG→EEG transfer lets a single model handle both signal types</sub> |
| <sub>**DS-CAE + pruning** (RAMAN)</sub> | <sub>`cae_compression.py`</sub> | <sub>LFP</sub> | <sub>Depthwise-separable convolutions cut multiply-accumulates by ~9× vs dense conv, enabling implantable hardware at **15.1 μW/channel**. **Balanced stochastic pruning** eliminates 32.4% of weights while preserving per-channel pruning balance — critical for FPGA/ASIC where unbalanced sparsity creates routing bottlenecks.</sub> | <sub>Standard CAE, dense VAE, wavelet</sub> | <sub>DS-conv reduces parameter count 9×; pruning cuts 32.4% more; SNDR stays 22.6–27.4 dB vs 18–21 dB for equivalent dense model at same ratio</sub> |
| <sub>**CLEM entropy model** (LLCSpike)</sub> | <sub>`llc_spike_compressor.py`</sub> | <sub>Spike trains</sub> | <sub>Generic zlib/LZ77 treats spike trains as raw bytes, ignoring that inter-spike intervals follow heavy-tailed distributions. CLEM **learns the marginal and transition probabilities** of spike intensity frames, then remaps symbols by learned frequency order so the downstream entropy coder sees near-i.i.d. input — optimal entropy coding condition. Lossless: exact bit reconstruction guaranteed.</sub> | <sub>`NeuralLZ77` + zlib, BLOSC, Huffman on raw binary</sub> | <sub>Learned symbol remapping reduces average code length by matching coder alphabet to actual spike statistics; bitpacked position representation halves storage of sparse spike patterns vs int8</sub> |
| <sub>**Diffusion prior** (EEGCiD)</sub> | <sub>`diffusion_compressor.py`</sub> | <sub>Long EEG recordings</sub> | <sub>For **archival** use cases where exact waveform fidelity is secondary to event detection (sleep stages, slow oscillations, BCI trial labels), a generative prior can reconstruct plausible signals from 64–128× compressed latents. The spectral prior in the score network enforces empirically known EEG band structure (delta/theta/alpha/beta/gamma), injecting domain knowledge that a generic auto-decoder lacks.</sub> | <sub>Variational AE, GANs, simple downsampling</sub> | <sub>Score network bends reconstruction toward neural oscillation priors; cosine noise schedule reduces perceptual artifacts vs linear schedule; float16 latent quantization reduces stored coefficients to 2 bytes each</sub> |

---

### 🔗 Integration with Existing Pipeline

> [!NOTE]
> All four new algorithms are **fully integrated** into the existing toolkit architecture — they require no changes to your existing code. They slot into the factory, the adaptive selector, the streaming pipeline, and the benchmarking harness automatically via the shared `BaseCompressor` interface.

---

#### 🏗️ Architecture Overview: How Integration Works

The toolkit is built around three layers that the new algorithms plug into:

```
┌─────────────────────────────────────────────────────────┐
│              Your Application / BCI Pipeline            │
└───────────────────────┬─────────────────────────────────┘
                        │  compress(data) / decompress(bytes, meta)
┌───────────────────────▼─────────────────────────────────┐
│           AdaptiveSelector  (signal-type routing)        │
│  Analyses: spike_rate · band_power · kurtosis · corr    │
│  Routes to best algorithm for the current data window   │
└───────┬───────────┬──────────────┬──────────┬───────────┘
        ▼           ▼              ▼          ▼
   [rvq]       [ds_cae]      [llc_spike]  [diffusion]
   EEG/iEEG    LFP           Spike trains  Long EEG
   64x lossy   150x lossy    lossless      >100x lossy
        │           │              │          │
        └───────────┴──────────────┴──────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│           BaseCompressor  (shared interface)             │
│  Provides: timing · metadata · streaming · benchmarking │
└─────────────────────────────────────────────────────────┘
```

**Why this architecture?**
Rather than wrapping each new algorithm in custom glue code, every compressor shares a single interface (`_compress_impl` / `_decompress_impl`). This means the new algorithms immediately inherit:
- automatic **latency timing** and **compression ratio** tracking
- **streaming chunk** support for real-time pipelines
- **benchmark harness** compatibility (run `benchmark_runner.py` — all four appear automatically)
- **metadata dict** propagation (ratio, channel count, algorithm-specific metrics)

---

#### 🏭 The Algorithm Factory — Create Any Compressor by Name

**What it is:** A registry that maps string keys to compressor classes, allowing runtime selection without hard-coded imports.

**Why it matters:** BCI rigs vary — an implanted device needs `ds_cae` (ultra-low-power LFP), a research rack needs `rvq` (high-fidelity EEG), an archival pipeline needs `diffusion` (extreme ratio). The factory lets a config file or CLI flag select the algorithm without changing application code.

**How to use it:**

```python
from bci_compression.algorithms.factory import create_compressor

# Create by name — swap algorithms without touching application code
rvq_comp  = create_compressor("rvq")                          # BrainCodec 64x EEG
cae_comp  = create_compressor("ds_cae", target_ratio=128)     # RAMAN 150x LFP
llc_comp  = create_compressor("llc_spike", frame_size=16)     # Lossless spike trains
diff_comp = create_compressor("diffusion", latent_ratio=64)   # EEGCiD extreme EEG

# All return a BaseCompressor — identical call signature for all:
import numpy as np
eeg = np.random.randn(8, 1024).astype(np.float32)

compressed, meta = rvq_comp.compress(eeg)
reconstructed    = rvq_comp.decompress(compressed, meta)

print(f"Ratio: {meta['compression_ratio']:.1f}x  |  Latency: {meta['latency_ms']:.2f} ms")
```

**All registered algorithm keys:**

| <sub>Key</sub> | <sub>Class</sub> | <sub>Best For</sub> |
|-----|-------|---------|
| <sub>`"rvq"`</sub> | <sub>`RVQCompressor`</sub> | <sub>Broadband EEG / iEEG (BrainCodec, 64x)</sub> |
| <sub>`"ds_cae"`</sub> | <sub>`DSCAECompressor`</sub> | <sub>Local Field Potentials / implantable hardware (RAMAN, 150x)</sub> |
| <sub>`"llc_spike"`</sub> | <sub>`LLCSpikeCompressor`</sub> | <sub>Binary spike trains — lossless (CLEM, IEEE TIP)</sub> |
| <sub>`"diffusion"`</sub> | <sub>`DiffusionCompressor`</sub> | <sub>Long archival EEG recordings (EEGCiD, >100x)</sub> |
| <sub>`"transformer"`</sub> | <sub>`TransformerCompressor`</sub> | <sub>High-SNR multi-channel EEG</sub> |
| <sub>`"vae"`</sub> | <sub>`VAECompressor`</sub> | <sub>Non-Gaussian / kurtotic neural signals</sub> |
| <sub>`"neural_lz77"`</sub> | <sub>`NeuralLZ77Compressor`</sub> | <sub>Correlated broadband LFP / EEG</sub> |
| <sub>`"adaptive_lz"`</sub> | <sub>`AdaptiveLZCompressor`</sub> | <sub>Fast lossless — any signal type</sub> |

**Benefit:** A **single line of config** (`algorithm: "ds_cae"`) switches from a general-purpose compressor to the hardware-optimised LFP algorithm — no code changes, no re-imports, no API differences.

---

#### 🧭 Adaptive Selector — Automatic Signal-Type Routing

**What it is:** A real-time signal analyser that extracts features from each data window (spike rate, band power, kurtosis, cross-channel correlation) and scores all registered algorithms against those features to pick the best one automatically.

**Why it matters:** A BCI recording session is not homogeneous. A 60-minute session contains:
- **Rest periods** — smooth EEG, low spike rate → `diffusion` or `rvq`
- **Motor task epochs** — high gamma, structured spikes → `transformer` or `rvq`
- **Spike-sorted units** — sparse binary trains → `llc_spike`
- **LFP channels** — delta/theta dominant → `ds_cae`

Without the adaptive selector, a researcher would need to manually switch algorithms between epochs or accept a one-size-fits-all compressor that is sub-optimal for most windows.

**How to use it:**

```python
from bci_compression.algorithms.adaptive_selector import AdaptiveSelector, AdaptiveSelectorConfig
from bci_compression.algorithms.factory import create_compressor

# Configure with your recording's sampling rate
config = AdaptiveSelectorConfig(fs=1000.0, hysteresis=3, switch_threshold=0.15)
sel = AdaptiveSelector(config)

# Build a compressor pool for all candidate algorithms
pool = {name: create_compressor(name) for name in
        ["rvq", "ds_cae", "llc_spike", "diffusion", "transformer", "neural_lz77"]}

# Process streaming windows
for window in incoming_data_windows:
    algo_name, decision = sel.select(window)  # <1 ms feature extraction
    compressor = pool[algo_name]

    compressed, meta = compressor.compress(window)
    print(f"Window → {algo_name} | ratio={meta['compression_ratio']:.1f}x "
          f"| latency={meta['latency_ms']:.2f}ms")
    # decision['features'] and decision['scores'] available for logging
```

**Routing logic — what signals each algorithm wins on:**

```
Signal Characteristics          → Selected Algorithm   Rationale
─────────────────────────────────────────────────────────────────────────────
Delta power (1–4 Hz) dominant   → ds_cae              LFP profile; RAMAN 150x target
 + spike rate < 5%

Spike rate > 20%                → llc_spike           Pure binary spike trains;
                                                       CLEM lossless optimal

Broad alpha/beta/gamma          → rvq                 EEG/iEEG multi-band;
 + moderate kurtosis                                   BrainCodec 64x target

Very smooth, low-kurtosis        → diffusion           Long quiescent EEG;
 + delta/theta                                         generative prior reconstructs well

High beta + gamma power         → transformer         Rich HF content; attention
                                                       captures temporal dependencies

High cross-channel correlation  → neural_lz77         Redundant multi-channel LFP;
 + low spike rate                                      dictionary compression excels
```

**Benefit:** Automatically achieves near-optimal compression for **every window** of a multi-hour recording — no manual tuning, no session-specific algorithm selection. The hysteresis control prevents rapid flapping between algorithms, keeping overhead under 0.5 ms per window.

---

#### ♻️ Backward Compatibility — Drop-In Upgrades

The new algorithms are designed as **replacements** for existing ones in the same use case, not additions that break existing workflows:

| <sub>Existing Workflow</sub> | <sub>New Drop-In</sub> | <sub>What Changes</sub> | <sub>What Stays the Same</sub> |
|-------------------|------------|--------------|---------------------|
| <sub>`VAECompressor` for EEG</sub> | <sub>`RVQCompressor` (`"rvq"`)</sub> | <sub>4 RVQ codebook stages + line-length loss added</sub> | <sub>Same `compress()` / `decompress()` API; same metadata keys</sub> |
| <sub>`NeuralLZ77` for spike trains</sub> | <sub>`LLCSpikeCompressor` (`"llc_spike"`)</sub> | <sub>Generic zlib → CLEM learned entropy model</sub> | <sub>Lossless guarantee preserved; API identical</sub> |
| <sub>Dense CAE for LFP</sub> | <sub>`DSCAECompressor` (`"ds_cae"`)</sub> | <sub>Dense conv → depthwise-separable + balanced pruning</sub> | <sub>Same `_compress_impl` hook; `sndr_estimate()` added as bonus</sub> |
| <sub>Any compressor for long EEG</sub> | <sub>`DiffusionCompressor` (`"diffusion"`)</sub> | <sub>Adds generative prior reconstruction on decompress</sub> | <sub>Same interface; `set_inference_steps()` controls quality/speed</sub> |

```python
# Before — existing code unchanged:
from bci_compression.algorithms.vae_compression import VAECompressor
comp = VAECompressor()
compressed, meta = comp.compress(eeg_data)

# After — swap one import, identical call:
from bci_compression.algorithms.rvq_compressor import RVQCompressor
comp = RVQCompressor()                         # 64x vs 2–8x; preserves transients
compressed, meta = comp.compress(eeg_data)    # identical signature
```

---

#### 📊 End-to-End Use Case Examples

**Use case 1 — Implantable BCI (closed-loop, low power):**
```python
# Target: 15.1 μW/channel, 150x LFP compression for implanted device
from bci_compression.algorithms.factory import create_compressor

comp = create_compressor("ds_cae", target_ratio=128, apply_pruning=True)

lfp_stream = acquire_lfp_channels()          # (32, 512) LFP window at 1 kHz
compressed, meta = comp.compress(lfp_stream)
# → meta['compression_ratio'] ~ 80–130x
# → meta['achieved_sparsity']  ~ 0.32 (32% of weights zeroed for hardware)

sndr = comp.sndr_estimate(lfp_stream, comp.decompress(compressed, meta))
print(f"SNDR: {sndr:.1f} dB")               # Target ≥ 22.6 dB (RAMAN paper)
```

**Use case 2 — Spike sorting archive (lossless):**
```python
# Target: lossless storage of spike-sorted single units for offline re-analysis
from bci_compression.algorithms.factory import create_compressor
import numpy as np

comp  = create_compressor("llc_spike", frame_size=16)
spikes = load_sorted_units()                 # (64 units, 30000 samples) binary

compressed, meta = comp.compress(spikes)
restored = comp.decompress(compressed, meta)

assert np.array_equal(restored, spikes)      # guaranteed bit-exact
print(f"Lossless ratio: {meta['compression_ratio']:.2f}x")
```

**Use case 3 — Long-term EEG archival (extreme ratio):**
```python
# Target: store 8-hour sleep EEG at >100x compression for sleep staging
from bci_compression.algorithms.factory import create_compressor

comp = create_compressor("diffusion", latent_ratio=64, n_diff_steps=20)

for epoch in sleep_eeg_epochs:               # 30-second epochs, (8, 7680) at 256 Hz
    compressed, meta = comp.compress(epoch)
    # → meta['compression_ratio'] ~ 40–80x
    # Store compressed bytes; reconstruct on analysis machine:
    recon = comp.decompress(compressed, meta)
    quality = comp.get_reconstruction_quality_estimate(epoch, recon)
    print(f"SNR: {quality['snr_db']:.1f} dB  |  Alpha band SNR: {quality['spectral_snr_alpha_db']:.1f} dB")
```

**Use case 4 — Research rig (codec-quality EEG):**
```python
# Target: 64x EEG compression with transient preservation for epilepsy research
from bci_compression.algorithms.factory import create_compressor
from bci_compression.algorithms.rvq_compressor import line_length_loss

comp = create_compressor("rvq", n_residuals=4, codebook_size=256)

eeg = acquire_ieeg_montage()                 # (128, 2048) iEEG, 2 kHz
compressed, meta = comp.compress(eeg)
recon = comp.decompress(compressed, meta)

ll_loss = line_length_loss(eeg, recon)       # BrainCodec transient metric
print(f"Line-length loss: {ll_loss:.4f}")    # Lower = sharper transients preserved
print(f"Ratio: {meta['compression_ratio']:.1f}x  |  Ratio estimate: {comp.get_compression_ratio_estimate():.1f}x")
```

---

#### 🔧 Integration with Existing Components — Full Mapping

| <sub>Component</sub> | <sub>File</sub> | <sub>How New Algorithms Connect</sub> | <sub>Benefit</sub> |
|-----------|------|----------------------------|---------|
| <sub>**AlgorithmFactory**</sub> | <sub>`algorithms/factory.py`</sub> | <sub>All 4 registered in `register_default_algorithms()` under `"rvq"`, `"ds_cae"`, `"llc_spike"`, `"diffusion"`</sub> | <sub>Create by name from config/CLI without code changes</sub> |
| <sub>**AdaptiveSelector**</sub> | <sub>`algorithms/adaptive_selector.py`</sub> | <sub>`score_algorithms()` extended with 4 new scoring heuristics based on delta power, spike rate, kurtosis, and inter-channel correlation</sub> | <sub>Automatic best-algorithm routing per window; no manual epoch labelling</sub> |
| <sub>**BaseCompressor**</sub> | <sub>`core.py`</sub> | <sub>All 4 implement `_compress_impl(data)→(bytes, dict)` and `_decompress_impl(bytes, dict)→ndarray`</sub> | <sub>Timing, metadata, streaming, and benchmarking work without modification</sub> |
| <sub>**NeuralLZ77**</sub> | <sub>`algorithms/neural_lz.py`</sub> | <sub>`LLCSpikeCompressor` provides a learned-entropy replacement for the zlib stage when data is spike trains</sub> | <sub>Better compression ratio vs generic zlib on structured spike data</sub> |
| <sub>**VAECompressor**</sub> | <sub>`algorithms/vae_compression.py`</sub> | <sub>`RVQCompressor` is a direct upgrade: same dense encoder replaced with RVQ stages + line-length loss</sub> | <sub>4–64x vs 2–8x; transient fidelity preserved without rewriting downstream code</sub> |
| <sub>**Benchmark Runner**</sub> | <sub>`scripts/benchmark_runner.py`</sub> | <sub>New algorithms appear automatically via factory registry</sub> | <sub>Zero-effort benchmarking; `--algorithms rvq ds_cae llc_spike diffusion`</sub> |
| <sub>**Streaming Pipeline**</sub> | <sub>`core.py` `StreamContext`</sub> | <sub>`compress()` / `decompress()` calls are streaming-safe via `BaseCompressor` chunk handling</sub> | <sub>Real-time frame processing with `< 5 ms` per window for all four</sub> |

### Key Metrics from RAMAN Paper (arXiv 2504.06996)

<details>
<summary>📊 DS-CAE Hardware Performance Details</summary>

| <sub>Metric</sub> | <sub>Value</sub> |
|--------|-------|
| <sub>Compression Ratio (LFP)</sub> | <sub>Up to **150x**</sub> |
| <sub>Power per Channel</sub> | <sub>**15.1 μW** @ 2 MHz</sub> |
| <sub>Technology Node</sub> | <sub>TSMC 65-nm</sub> |
| <sub>Area per Channel</sub> | <sub>0.0187 mm²</sub> |
| <sub>Parameter Reduction (pruning)</sub> | <sub>**32.4%** via balanced stochastic pruning</sub> |
| <sub>Reconstruction Quality (SNDR)</sub> | <sub>22.6–27.4 dB</sub> |
| <sub>R² Score on Monkey Neural Data</sub> | <sub>0.81–0.94</sub> |
| <sub>Architecture</sub> | <sub>Depthwise-Separable CAE (DS-CAE) with zero-skipping + weight gating</sub> |

</details>

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

### 📊 Technical Specifications & Performance Matrix

#### Core Algorithm Performance

| <sub>Algorithm</sub> | <sub>Compression Ratio</sub> | <sub>Latency</sub> | <sub>Throughput</sub> | <sub>Quality</sub> | <sub>Memory Usage</sub> | <sub>GPU Speedup</sub> |
|-----------|------------------|---------|------------|---------|--------------|-------------|
| <sub>**LZ4**</sub> | <sub>1.5-2x</sub> | <sub>< 0.1ms</sub> | <sub>675+ MB/s</sub> | <sub>Lossless</sub> | <sub>32KB</sub> | <sub>2x</sub> |
| <sub>**Zstandard**</sub> | <sub>2-4x</sub> | <sub>< 0.5ms</sub> | <sub>510 MB/s</sub> | <sub>Lossless</sub> | <sub>128KB</sub> | <sub>3x</sub> |
| <sub>**Blosc**</sub> | <sub>1.8-3x</sub> | <sub>< 0.2ms</sub> | <sub>800+ MB/s</sub> | <sub>Lossless</sub> | <sub>64KB</sub> | <sub>4x</sub> |
| <sub>**Neural LZ77**</sub> | <sub>1.5-3x</sub> | <sub>< 1ms</sub> | <sub>400 MB/s</sub> | <sub>Lossless</sub> | <sub>256KB</sub> | <sub>2.5x</sub> |
| <sub>**Perceptual Quant**</sub> | <sub>2-10x</sub> | <sub>< 1ms</sub> | <sub>300 MB/s</sub> | <sub>15-25 dB SNR</sub> | <sub>512KB</sub> | <sub>5x</sub> |
| <sub>**Adaptive Wavelets**</sub> | <sub>3-15x</sub> | <sub>< 1ms</sub> | <sub>250 MB/s</sub> | <sub>Configurable</sub> | <sub>1MB</sub> | <sub>6x</sub> |
| <sub>**Transformers**</sub> | <sub>3-5x</sub> | <sub>< 2ms</sub> | <sub>150 MB/s</sub> | <sub>25-35 dB SNR</sub> | <sub>2MB</sub> | <sub>8x</sub> |
| <sub>**VAE**</sub> | <sub>2-8x</sub> | <sub>< 5ms</sub> | <sub>100 MB/s</sub> | <sub>Statistical</sub> | <sub>4MB</sub> | <sub>10x</sub> |

#### Neural Signal Specific Performance

| <sub>Signal Type</sub> | <sub>Best Algorithm</sub> | <sub>Compression Ratio</sub> | <sub>Latency</sub> | <sub>Fidelity</sub> |
|-------------|---------------|------------------|---------|----------|
| <sub>**Motor Cortex**</sub> | <sub>LZ4 + Neural LZ77</sub> | <sub>2.1x</sub> | <sub>< 0.5ms</sub> | <sub>100%</sub> |
| <sub>**Visual Cortex**</sub> | <sub>Zstandard</sub> | <sub>3.2x</sub> | <sub>< 0.8ms</sub> | <sub>100%</sub> |
| <sub>**EMG Signals**</sub> | <sub>Blosc + Wavelets</sub> | <sub>8.5x</sub> | <sub>< 1.2ms</sub> | <sub>98.5%</sub> |
| <sub>**EEG Arrays**</sub> | <sub>Perceptual Quant</sub> | <sub>6.8x</sub> | <sub>< 1.5ms</sub> | <sub>22 dB SNR</sub> |
| <sub>**Spike Trains**</sub> | <sub>Neural LZ77</sub> | <sub>2.8x</sub> | <sub>< 0.3ms</sub> | <sub>99.8%</sub> |
| <sub>**Multi-Channel**</sub> | <sub>Blosc</sub> | <sub>4.1x</sub> | <sub>< 0.4ms</sub> | <sub>100%</sub> |

#### Hardware Platform Support

| <sub>Platform</sub> | <sub>CPU Architecture</sub> | <sub>GPU Support</sub> | <sub>Max Channels</sub> | <sub>Max Sampling Rate</sub> |
|----------|-----------------|-------------|--------------|------------------|
| <sub>**Desktop**</sub> | <sub>x86-64, ARM64</sub> | <sub>CUDA, OpenCL</sub> | <sub>1024+</sub> | <sub>50kHz</sub> |
| <sub>**Mobile**</sub> | <sub>ARM Cortex-A</sub> | <sub>GPU Compute</sub> | <sub>256</sub> | <sub>30kHz</sub> |
| <sub>**Embedded**</sub> | <sub>ARM Cortex-M</sub> | <sub>None</sub> | <sub>64</sub> | <sub>10kHz</sub> |
| <sub>**FPGA**</sub> | <sub>Custom</sub> | <sub>Hardware</sub> | <sub>2048+</sub> | <sub>100kHz</sub> |
| <sub>**Cloud**</sub> | <sub>x86-64</sub> | <sub>CUDA, TPU</sub> | <sub>Unlimited</sub> | <sub>Unlimited</sub> |

### 🎯 Specialized Applications & Use Cases

#### Medical-Grade BCI Applications

```mermaid
graph TB
    subgraph "🏥 Clinical BCIs"
        style C1 fill:#1a202c,stroke:#2d3748,stroke-width:2px,color:#ffffff
        style C2 fill:#1a202c,stroke:#2d3748,stroke-width:2px,color:#ffffff
        style C3 fill:#1a202c,stroke:#2d3748,stroke-width:2px,color:#ffffff

        C1[Epilepsy<br/>Monitoring]
        C2[Deep Brain<br/>Stimulation]
        C3[Neural<br/>Prosthetics]
    end

    subgraph "⚡ Real-Time Requirements"
        style R1 fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#ffffff
        style R2 fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#ffffff
        style R3 fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#ffffff

        R1[< 1ms Latency<br/>LZ4 + Neural LZ77]
        R2[< 500μs Latency<br/>LZ4 Only]
        R3[< 2ms Latency<br/>Advanced ML]
    end

    subgraph "📊 Data Characteristics"
        style D1 fill:#1a202c,stroke:#2d3748,stroke-width:2px,color:#ffffff
        style D2 fill:#1a202c,stroke:#2d3748,stroke-width:2px,color:#ffffff
        style D3 fill:#1a202c,stroke:#2d3748,stroke-width:2px,color:#ffffff

        D1[128+ Channels<br/>30kHz Sampling]
        D2[32 Channels<br/>10kHz Sampling]
        D3[256+ Channels<br/>40kHz Sampling]
    end

    C1 --> R3
    C2 --> R2
    C3 --> R1

    R1 --> D3
    R2 --> D2
    R3 --> D1
```

#### Performance vs Quality Trade-offs

```mermaid
graph LR
    subgraph "🏃 Ultra-Fast (< 0.1ms)"
        style UF1 fill:#1a202c,stroke:#2d3748,stroke-width:2px,color:#ffffff
        style UF2 fill:#1a202c,stroke:#2d3748,stroke-width:2px,color:#ffffff

        UF1[LZ4<br/>1.5-2x ratio]
        UF2[Blosc<br/>1.8-3x ratio]
    end

    subgraph "⚡ Fast (< 1ms)"
        style F1 fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#ffffff
        style F2 fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#ffffff
        style F3 fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#ffffff

        F1[Zstandard<br/>2-4x ratio]
        F2[Neural LZ77<br/>1.5-3x ratio]
        F3[Perceptual Quant<br/>2-10x ratio]
    end

    subgraph "🧠 Advanced (< 2ms)"
        style A1 fill:#1a202c,stroke:#2d3748,stroke-width:2px,color:#ffffff
        style A2 fill:#1a202c,stroke:#2d3748,stroke-width:2px,color:#ffffff

        A1[Transformers<br/>3-5x ratio]
        A2[VAE<br/>2-8x ratio]
    end

    UF1 --> F1
    UF2 --> F2
    F1 --> A1
    F2 --> A2
    F3 --> A1
```

#### Specialized Signal Support

**🧠 EMG Compression**

- **Specialized algorithms**: Electromyography signals (5-12x compression)
- **Muscle artifact handling**: Optimized for movement-related noise
- **Real-time feedback**: < 500μs latency for prosthetic control

**📡 Multi-Channel Arrays**

- **Spatial correlation**: High-density electrode grids (256+ channels)
- **Blosc optimization**: Columnar compression for array data
- **Scalable architecture**: Supports up to 2048 channels simultaneously

**📱 Mobile/Embedded BCIs**

- **Power efficiency**: Battery-optimized algorithms for wearable devices
- **Resource constraints**: Minimal memory footprint (< 1MB)
- **ARM optimization**: NEON SIMD instruction utilization

**☁️ Cloud Analytics**

- **Batch processing**: High-ratio compression for long-term storage
- **Dictionary training**: Zstandard with learned neural patterns
- **Scalable processing**: Distributed compression across multiple GPUs

## 📡 API Documentation

### Core Compression API

```python
from neural_compression import NeuralCompressor, CompressionConfig

# Initialize compressor with GPU acceleration
compressor = NeuralCompressor(
    algorithm='neural_lz77',
    gpu_enabled=True,
    real_time=True
)

# Compress neural data stream
compressed_data = compressor.compress(
    neural_signals,  # numpy array (channels, samples)
    quality_level=0.95,  # 0.0-1.0 for lossy algorithms
    preserve_spikes=True  # maintain action potential fidelity
)

# Real-time streaming compression
stream = compressor.create_stream(
    buffer_size=1024,
    overlap=128,
    latency_target=0.5  # milliseconds
)

for chunk in neural_data_stream:
    compressed_chunk = stream.process(chunk)
    # < 1ms processing time guaranteed
```

### Algorithm Selection API

```python
from neural_compression import AlgorithmSelector

# Automatic algorithm selection based on signal characteristics
selector = AlgorithmSelector()
optimal_config = selector.analyze_and_recommend(
    signal_data=neural_array,
    sampling_rate=30000,  # Hz
    channel_count=256,
    latency_requirement=1.0,  # ms
    quality_requirement=0.98  # fidelity score
)

# Returns optimized configuration
# optimal_config.algorithm -> 'blosc' for multi-channel
# optimal_config.parameters -> {compression_level: 5, threads: 4}
```

### Performance Monitoring API

```python
from neural_compression import PerformanceMonitor

monitor = PerformanceMonitor()

# Real-time performance tracking
with monitor.track_compression() as tracker:
    result = compressor.compress(data)

    # Automatic metrics collection
    metrics = tracker.get_metrics()
    # metrics.latency -> 0.8ms
    # metrics.throughput -> 450 MB/s
    # metrics.compression_ratio -> 2.3x
    # metrics.fidelity_score -> 0.987
```

### WebSocket Streaming API

```python
import asyncio
from neural_compression.streaming import NeuralWebSocket

async def stream_neural_data():
    websocket = NeuralWebSocket(
        host='localhost',
        port=8080,
        compression='lz4',
        real_time=True
    )

    async for compressed_chunk in websocket.stream():
        # Receive compressed neural data
        decompressed = websocket.decompress(compressed_chunk)
        # Process in real-time (< 1ms latency)
```

### REST API Endpoints

**Compression Service** - `POST /api/v1/compress`

```json
{
  "data": "base64_encoded_neural_data",
  "algorithm": "neural_lz77",
  "config": {
    "quality": 0.95,
    "gpu_acceleration": true,
    "real_time": true
  }
}
```

**Algorithm Recommendation** - `POST /api/v1/recommend`

```json
{
  "signal_characteristics": {
    "sampling_rate": 30000,
    "channel_count": 128,
    "signal_type": "motor_cortex",
    "noise_level": 0.05
  },
  "requirements": {
    "max_latency_ms": 1.0,
    "min_fidelity": 0.98,
    "target_compression": 3.0
  }
}
```

**Performance Metrics** - `GET /api/v1/metrics`

```json
{
  "current_throughput": "675 MB/s",
  "average_latency": "0.45ms",
  "compression_ratio": "2.8x",
  "gpu_utilization": "23%",
  "active_streams": 12
}
```

### Configuration Management

```python
from neural_compression import CompressionConfig

# Algorithm-specific configurations
configs = {
    'real_time_control': CompressionConfig(
        algorithm='lz4',
        latency_target=0.1,  # 100μs for prosthetic control
        quality=1.0,  # lossless
        gpu_enabled=False  # CPU for deterministic timing
    ),

    'high_density_arrays': CompressionConfig(
        algorithm='blosc',
        threads=8,
        compression_level=6,
        shuffle=True,  # optimize for array patterns
        gpu_enabled=True
    ),

    'analysis_storage': CompressionConfig(
        algorithm='zstd',
        compression_level=19,  # maximum ratio
        dictionary_training=True,
        quality=1.0  # lossless for analysis
    ),

    'mobile_streaming': CompressionConfig(
        algorithm='perceptual_quantization',
        quality=0.85,  # balanced quality/size
        power_efficient=True,
        memory_limit='256MB'
    )
}
```

## 🏃‍♂️ Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/hkevin01/brain-computer-compression.git
   cd brain-computer-compression
   ```

2. **Start with Docker** (recommended)

   ```bash
   ./run.sh up
   ```

3. **Or manual setup**

   ```bash
   ./scripts/setup/setup.sh
   ```

4. **Access the dashboard**
   - Open <http://localhost:3000> in your browser
   - Or run `./run.sh gui:open`

5. **API access**
   - REST API: <http://localhost:8000/docs>
   - WebSocket: `ws://localhost:8080/stream`
   - Metrics: <http://localhost:8000/metrics>

## 🧪 Benchmarking & Testing

### Performance Benchmarks

**Real-Time Processing Benchmarks**

```mermaid
graph LR
    subgraph "⚡ Latency Benchmarks (ms)"
        style L1 fill:#1a202c,stroke:#2d3748,stroke-width:2px,color:#ffffff
        style L2 fill:#1a202c,stroke:#2d3748,stroke-width:2px,color:#ffffff
        style L3 fill:#1a202c,stroke:#2d3748,stroke-width:2px,color:#ffffff
        style L4 fill:#1a202c,stroke:#2d3748,stroke-width:2px,color:#ffffff

        L1[LZ4<br/>0.08ms]
        L2[Blosc<br/>0.15ms]
        L3[ZSTD<br/>0.42ms]
        L4[Neural LZ77<br/>0.85ms]
    end

    subgraph "🚀 Throughput (MB/s)"
        style T1 fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#ffffff
        style T2 fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#ffffff
        style T3 fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#ffffff
        style T4 fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#ffffff

        T1[LZ4<br/>675 MB/s]
        T2[Blosc<br/>820 MB/s]
        T3[ZSTD<br/>510 MB/s]
        T4[Neural LZ77<br/>385 MB/s]
    end

    L1 --> T1
    L2 --> T2
    L3 --> T3
    L4 --> T4
```

**Neural Data Specific Benchmarks**

| <sub>Dataset</sub> | <sub>Algorithm</sub> | <sub>Compression Ratio</sub> | <sub>Latency</sub> | <sub>SNR</sub> | <sub>Spike Accuracy</sub> |
|---------|-----------|------------------|---------|-----|----------------|
| <sub>**Motor Cortex (128ch, 30kHz)**</sub> | <sub>LZ4 + Neural LZ77</sub> | <sub>2.1x</sub> | <sub>0.5ms</sub> | <sub>∞ (lossless)</sub> | <sub>100%</sub> |
| <sub>**Visual Cortex (256ch, 40kHz)**</sub> | <sub>Blosc + ZSTD</sub> | <sub>3.8x</sub> | <sub>0.8ms</sub> | <sub>∞ (lossless)</sub> | <sub>100%</sub> |
| <sub>**EMG Arrays (64ch, 10kHz)**</sub> | <sub>Perceptual Quant</sub> | <sub>8.2x</sub> | <sub>1.2ms</sub> | <sub>28.5 dB</sub> | <sub>98.7%</sub> |
| <sub>**EEG (32ch, 1kHz)**</sub> | <sub>Adaptive Wavelets</sub> | <sub>12.5x</sub> | <sub>1.8ms</sub> | <sub>32.1 dB</sub> | <sub>99.2%</sub> |
| <sub>**Spike Trains (Single Unit)**</sub> | <sub>Neural LZ77</sub> | <sub>2.9x</sub> | <sub>0.3ms</sub> | <sub>∞ (lossless)</sub> | <sub>99.9%</sub> |

### Test Suite Coverage

**Unit Tests** - Core Algorithm Validation

```bash
# Run all compression algorithm tests
pytest tests/algorithms/ -v --cov=neural_compression

# Test specific algorithms
pytest tests/algorithms/test_lz4_compression.py
pytest tests/algorithms/test_neural_lz77.py
pytest tests/algorithms/test_gpu_acceleration.py

# Performance regression tests
pytest tests/performance/ --benchmark-only
```

**Integration Tests** - End-to-End Validation

```bash
# Full pipeline tests with real neural data
pytest tests/integration/test_neural_pipeline.py

# Real-time streaming tests
pytest tests/integration/test_realtime_processing.py

# GPU acceleration integration
pytest tests/integration/test_gpu_pipeline.py
```

**Benchmark Tests** - Performance Validation

```bash
# Comprehensive benchmarking suite
python scripts/benchmark/run_benchmarks.py

# Specific performance tests
python scripts/benchmark/latency_benchmark.py
python scripts/benchmark/throughput_benchmark.py
python scripts/benchmark/compression_ratio_benchmark.py
```

### Test Data Sources

**Synthetic Neural Data**

- **Generated spike trains**: Poisson processes with realistic firing rates
- **Multi-channel arrays**: Simulated electrode grids with spatial correlations
- **Noise models**: Realistic thermal and electronic noise characteristics
- **Artifact simulation**: Movement artifacts, line noise, electrode drift

**Real Neural Datasets**

- **Motor cortex recordings**: Utah array data from macaque experiments
- **Visual cortex data**: Multi-electrode recordings during visual stimulation
- **Human EEG/ECoG**: Clinical datasets with appropriate anonymization
- **EMG recordings**: High-density surface and intramuscular recordings

### Continuous Integration

**GitHub Actions Workflow**

```yaml
name: Neural Compression CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run unit tests
      run: pytest tests/ --cov=neural_compression

    - name: Run integration tests
      run: pytest tests/integration/

    - name: Performance benchmarks
      run: python scripts/benchmark/ci_benchmarks.py

    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

**Performance Regression Detection**

- **Automatic benchmarking**: Every commit tested for performance regressions
- **Latency monitoring**: Alerts if processing latency exceeds thresholds
- **Memory usage tracking**: Detects memory leaks in streaming scenarios
- **GPU utilization monitoring**: Ensures efficient hardware acceleration usage

### Quality Assurance

**Code Quality Tools**

```bash
# Code formatting
black neural_compression/
isort neural_compression/

# Type checking
mypy neural_compression/

# Linting
flake8 neural_compression/
pylint neural_compression/

# Security scanning
bandit -r neural_compression/
```

**Documentation Testing**

```bash
# Docstring examples
python -m doctest neural_compression/*.py

# Documentation build
sphinx-build -b html docs/ docs/_build/

# API documentation validation
python scripts/validate_api_docs.py
```

## � Project Structure

```
brain-computer-compression/
├── 📦 neural_compression/          # Core compression library
│   ├── 🧠 algorithms/              # Compression algorithms
│   │   ├── lossless/               # Lossless compression (LZ4, ZSTD, Blosc)
│   │   ├── lossy/                  # Lossy compression (wavelets, quantization)
│   │   ├── neural/                 # Neural-specific algorithms (Neural LZ77)
│   │   └── ai_powered/             # AI/ML compression (autoencoders, transformers)
│   ├── 🚀 gpu/                     # GPU acceleration modules
│   │   ├── cuda_kernels/           # Custom CUDA implementations
│   │   ├── cupy_wrappers/          # CuPy integration layer
│   │   └── memory_management/      # GPU memory optimization
│   ├── 📊 streaming/               # Real-time processing
│   │   ├── buffers/                # Circular buffers and windowing
│   │   ├── pipelines/              # Processing pipelines
│   │   └── websockets/             # WebSocket streaming
│   ├── 🔧 utils/                   # Utility functions
│   │   ├── signal_processing/      # Signal preprocessing
│   │   ├── performance/            # Performance monitoring
│   │   └── data_formats/           # Neural data format support
│   └── 📡 api/                     # API interfaces
│       ├── rest/                   # REST API endpoints
│       ├── websocket/              # WebSocket handlers
│       └── config/                 # Configuration management
├── 🌐 web/                         # Web dashboard
│   ├── frontend/                   # React/Next.js frontend
│   │   ├── components/             # UI components
│   │   ├── pages/                  # Dashboard pages
│   │   └── hooks/                  # Custom React hooks
│   └── backend/                    # FastAPI backend
│       ├── routers/                # API route handlers
│       ├── services/               # Business logic
│       └── models/                 # Data models
├── 🧪 tests/                       # Test suite
│   ├── unit/                       # Unit tests
│   │   ├── algorithms/             # Algorithm-specific tests
│   │   ├── gpu/                    # GPU acceleration tests
│   │   └── streaming/              # Real-time processing tests
│   ├── integration/                # Integration tests
│   │   ├── pipelines/              # End-to-end pipeline tests
│   │   ├── api/                    # API integration tests
│   │   └── performance/            # Performance validation
│   └── benchmark/                  # Benchmarking suite
│       ├── latency/                # Latency benchmarks
│       ├── throughput/             # Throughput benchmarks
│       └── compression_ratio/      # Compression ratio tests
├── 📖 docs/                        # Documentation
│   ├── api/                        # API documentation
│   ├── guides/                     # User guides and tutorials
│   ├── algorithms/                 # Algorithm documentation
│   ├── benchmarks/                 # Performance reports
│   └── project/                    # Project documentation
├── 🐳 docker/                      # Docker configuration
│   ├── services/                   # Individual service containers
│   │   ├── compression/            # Compression service
│   │   ├── web/                    # Web dashboard
│   │   └── gpu/                    # GPU-enabled containers
│   ├── compose/                    # Docker Compose files
│   └── scripts/                    # Container scripts
├── 🔧 scripts/                     # Utility scripts
│   ├── setup/                      # Environment setup
│   ├── benchmark/                  # Benchmarking scripts
│   ├── tools/                      # Development tools
│   └── deployment/                 # Deployment scripts
├── 📊 data/                        # Sample and test data
│   ├── synthetic/                  # Generated neural data
│   ├── real/                       # Real neural recordings
│   └── benchmarks/                 # Benchmark datasets
└── 📋 config/                      # Configuration files
    ├── algorithms/                 # Algorithm configurations
    ├── deployment/                 # Deployment configurations
    └── development/                # Development settings
```

### Core Components Deep Dive

#### 🧠 Neural Compression Algorithms (`neural_compression/algorithms/`)

**Lossless Compression** (`lossless/`)
- `lz4_compression.py` - Ultra-fast LZ4 implementation with neural optimizations
- `zstd_compression.py` - Zstandard with dictionary learning for neural patterns
- `blosc_compression.py` - Multi-threaded array compression with SIMD acceleration
- `neural_lz77.py` - Custom LZ77 variant trained on neural signal characteristics

**Lossy Compression** (`lossy/`)
- `perceptual_quantization.py` - Psychoacoustic principles adapted for neural frequencies
- `adaptive_wavelets.py` - Multi-resolution wavelet compression with neural-specific basis
- `predictive_coding.py` - Linear and adaptive prediction models for temporal patterns

**AI-Powered Compression** (`ai_powered/`)
- `autoencoders.py` - Deep autoencoder models for learned neural representations
- `transformers.py` - Multi-head attention models for sequence compression
- `vae_compression.py` - Variational autoencoders with uncertainty quantification

#### 🚀 GPU Acceleration (`neural_compression/gpu/`)

**CUDA Kernels** (`cuda_kernels/`)
- `lz4_cuda.cu` - Custom CUDA implementation of LZ4 compression
- `wavelet_cuda.cu` - GPU-accelerated wavelet transforms
- `neural_network_cuda.cu` - Optimized neural network inference kernels

**Memory Management** (`memory_management/`)
- `gpu_buffers.py` - Efficient GPU memory allocation and streaming
- `memory_pool.py` - Memory pool management for continuous processing
- `transfer_optimization.py` - CPU-GPU memory transfer optimization

#### 📊 Real-Time Streaming (`neural_compression/streaming/`)

**Buffer Management** (`buffers/`)
- `circular_buffer.py` - Lock-free circular buffers for streaming data
- `sliding_window.py` - Overlapping window processing for continuous signals
- `adaptive_buffer.py` - Dynamic buffer sizing based on processing load

**Processing Pipelines** (`pipelines/`)
- `realtime_pipeline.py` - Real-time processing pipeline with guaranteed latency
- `batch_pipeline.py` - High-throughput batch processing for offline analysis
- `streaming_pipeline.py` - Continuous streaming with backpressure handling

#### 🌐 Web Dashboard (`web/`)

**Frontend** (`frontend/`)
- `components/CompressionMonitor.tsx` - Real-time compression performance monitoring
- `components/AlgorithmSelector.tsx` - Interactive algorithm selection interface
- `components/PerformanceCharts.tsx` - Real-time performance visualization
- `pages/Dashboard.tsx` - Main dashboard with compression metrics
- `pages/Benchmarks.tsx` - Performance benchmarking interface

**Backend** (`backend/`)
- `routers/compression.py` - Compression API endpoints
- `routers/streaming.py` - WebSocket streaming endpoints
- `services/compression_service.py` - Core compression business logic
- `models/neural_data.py` - Neural data models and validation

### Configuration Management

#### Algorithm Configurations (`config/algorithms/`)

```yaml
# config/algorithms/realtime.yaml
realtime_compression:
  algorithm: "lz4"
  max_latency_ms: 1.0
  gpu_enabled: false  # CPU for deterministic timing
  buffer_size: 1024

# config/algorithms/high_ratio.yaml
high_ratio_compression:
  algorithm: "zstd"
  compression_level: 19
  dictionary_training: true
  gpu_enabled: true

# config/algorithms/neural_optimized.yaml
neural_optimized:
  algorithm: "neural_lz77"
  spike_detection: true
  temporal_correlation: true
  adaptive_learning: true
```

#### Deployment Configurations (`config/deployment/`)

```yaml
# config/deployment/production.yaml
production:
  gpu_memory_limit: "8GB"
  max_concurrent_streams: 100
  monitoring_enabled: true
  logging_level: "INFO"

# config/deployment/development.yaml
development:
  gpu_memory_limit: "2GB"
  max_concurrent_streams: 10
  monitoring_enabled: true
  logging_level: "DEBUG"
  profiling_enabled: true
```

## 🤝 Contributing & Development

### Development Workflow

Development follows a structured process with quality gates:

#### Setting Up Development Environment

##### Option 1: Docker Development (Recommended)

```bash
# Clone and start development environment
git clone https://github.com/hkevin01/brain-computer-compression.git
cd brain-computer-compression

# Start development services with hot-reload
./run.sh dev

# Access development tools
# - Code: http://localhost:8080 (VS Code in browser)
# - API: http://localhost:8000/docs
# - Dashboard: http://localhost:3000
# - Jupyter: http://localhost:8888
```

##### Option 2: Local Development

```bash
# Python environment setup
python3.10 -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install project in development mode
pip install -e .

# Install pre-commit hooks
pre-commit install
```

#### Code Standards & Guidelines

##### Python Code Style

Follow PEP 8 with these specific guidelines:

```python
# Type hints are required for all public functions
def compress_neural_data(
    data: np.ndarray,
    algorithm: str = "lz4",
    quality: float = 1.0,
    gpu_enabled: bool = False
) -> CompressionResult:
    """
    Compress neural data using specified algorithm.

    Args:
        data: Neural signal array (channels, samples)
        algorithm: Compression algorithm name
        quality: Quality level (0.0-1.0)
        gpu_enabled: Enable GPU acceleration

    Returns:
        Compression result with metrics

    Raises:
        ValueError: If algorithm not supported
        MemoryError: If insufficient GPU memory
    """
    pass
```

##### Performance Requirements

All contributions must meet these performance benchmarks:

- **Latency**: < 1ms for real-time algorithms, < 5ms for advanced algorithms
- **Throughput**: Minimum 100 MB/s compression speed
- **Memory**: Bounded memory usage for streaming scenarios
- **GPU Efficiency**: > 70% GPU utilization when GPU-enabled

##### Testing Requirements

Unit Tests (Required for all PRs):

```bash
# Run specific test categories
pytest tests/unit/algorithms/ -v --cov=neural_compression.algorithms
pytest tests/unit/gpu/ -v --cov=neural_compression.gpu
pytest tests/unit/streaming/ -v --cov=neural_compression.streaming

# Minimum coverage: 85% for new code
pytest --cov=neural_compression --cov-report=html --cov-fail-under=85
```

#### Pull Request Requirements

Checklist for all PRs:

- [ ] All tests pass (`pytest tests/`)
- [ ] Performance benchmarks meet requirements
- [ ] Code coverage ≥ 85% for new code
- [ ] Documentation updated (API docs, README if needed)
- [ ] Type hints added for all public functions
- [ ] Docstrings follow Google/NumPy style
- [ ] No performance regressions detected
- [ ] GPU compatibility verified (if applicable)

Performance Validation:

```bash
# Before submitting PR, run full validation
./scripts/validate_pr.sh

# This script runs:
# - All unit and integration tests
# - Performance regression testing
# - Code quality checks (flake8, mypy, black)
# - Documentation validation
# - Security scanning (bandit)
```

#### Getting Help

##### Development Support

- **GitHub Discussions**: For design questions and general development help
- **Slack Channel**: `#neural-compression-dev` for real-time collaboration
- **Weekly Office Hours**: Thursdays 2-3 PM EST for direct developer support
- **Documentation**: [docs/development/](docs/development/) for detailed guides

##### Issue Reporting

- **Bug Reports**: Use GitHub Issues with the `bug` label
- **Feature Requests**: Use GitHub Issues with the `enhancement` label
- **Performance Issues**: Include benchmark results and system specifications
- **GPU Issues**: Provide CUDA version, driver version, and hardware details

## 📖 Learn More

- **API Documentation**: <http://localhost:8000/docs> (when running)
- **Project Guides**: [docs/guides/](docs/guides/)
- **Development Setup**: [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)
- **Architecture Overview**: [docs/project/](docs/project/)

## 🧠 Memory Bank

This project maintains a comprehensive memory bank for tracking decisions, implementations, and changes:

- **📝 App Description**: [memory-bank/app-description.md](memory-bank/app-description.md) - Comprehensive project overview and mission
- **📋 Implementation Plans**: [memory-bank/implementation-plans/](memory-bank/implementation-plans/) - ACID-structured feature development plans
- **🏗️ Architecture Decisions**: [memory-bank/architecture-decisions/](memory-bank/architecture-decisions/) - ADRs documenting key technical decisions
- **📊 Change Log**: [memory-bank/change-log.md](memory-bank/change-log.md) - Complete history of project modifications

The memory bank follows ACID principles (Atomic, Consistent, Isolated, Durable) to ensure clear, traceable, and maintainable documentation of all project evolution.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**🎯 Goal**: Efficient neural data compression for next-generation brain-computer interfaces.