# Multi-BCI System Support Guide

The BCI Compression Toolkit now supports multiple BCI systems with different configurations, including various electrode layouts, sampling rates, and data types.

## Supported BCI Systems

### EEG Systems

| System | Channels | Sampling Rate | Manufacturer | Data Type |
|--------|----------|---------------|--------------|-----------|
| OpenBCI Ganglion | 8 | 200 Hz | OpenBCI | EEG |
| OpenBCI Cyton | 16 | 250 Hz | OpenBCI | EEG |
| Emotiv EPOC | 14 | 128 Hz | Emotiv | EEG |
| BioSemi ActiveTwo 64 | 64 | 2,048 Hz | BioSemi | EEG |
| EGI GSN HydroCel 128 | 128 | 1,000 Hz | EGI | EEG |

### EMG Systems

| System | Channels | Sampling Rate | Manufacturer | Data Type |
|--------|----------|---------------|--------------|-----------|
| Delsys Trigno | 16 | 2,000 Hz | Delsys | EMG |

### Neural Recording Systems

| System | Channels | Sampling Rate | Manufacturer | Data Type |
|--------|----------|---------------|--------------|-----------|
| Blackrock Cerebus | 96 | 30,000 Hz | Blackrock Neurotech | Spikes |
| Intan RHD2000 | 64 | 20,000 Hz | Intan Technologies | Neural |
| Neuropixels 1.0 | 384 | 30,000 Hz | IMEC | Spikes |

## Quick Start

### List All Supported Systems

```python
from bci_compression.formats import list_supported_systems

systems = list_supported_systems()
for sys in systems:
    print(f"{sys['name']}: {sys['channels']} channels @ {sys['sampling_rate']} Hz")
```

### Get System Profile

```python
from bci_compression.formats import get_system_profile

# Load a specific system profile
profile = get_system_profile('openbci_16')

print(f"Channels: {profile.num_channels}")
print(f"Sampling Rate: {profile.sampling_rate} Hz")
print(f"Recommended Compression: {profile.recommended_compression}")
```

### Adapt Data Between Systems

```python
import numpy as np
from bci_compression.formats import adapt_data

# Simulate OpenBCI data (8 channels, 200 Hz)
data = np.random.randn(8, 2000)  # 10 seconds

# Adapt to 1000 Hz for standard processing
adapted_data, settings = adapt_data(
    data,
    source_system='openbci_8',
    target_sampling_rate=1000
)

print(f"Original: {data.shape}")
print(f"Adapted: {adapted_data.shape}")
print(f"Recommended algorithm: {settings['algorithm']}")
```

## System Profiles

Each system profile includes:

- **Channel Count**: Number of recording channels
- **Sampling Rate**: Data acquisition rate in Hz
- **Electrode Standard**: Electrode placement system (10-20, 10-10, custom)
- **Bit Depth**: ADC resolution (12, 16, or 24 bits)
- **Voltage Range**: Dynamic range in microvolts
- **Recommended Compression**: Suggested compression algorithm
- **Data Type**: Type of neural data (EEG, EMG, spikes, etc.)

## Data Adaptation

The data adapter can:

1. **Resample** data to different sampling rates
2. **Scale** voltages between different dynamic ranges
3. **Normalize** data to [-1, 1] range
4. **Convert** data types based on bit depth

### Example: Resampling

```python
from bci_compression.formats import BCIDataAdapter

# Create adapter
adapter = BCIDataAdapter(
    source_profile='openbci_8',
    target_sampling_rate=1000,
    normalize=True
)

# Adapt data
adapted = adapter.adapt(original_data)
settings = adapter.get_compression_settings()
```

## Compression Workflow

### Step 1: Load System Profile

```python
from bci_compression.formats import get_system_profile

profile = get_system_profile('biosemi_64')
```

### Step 2: Adapt Data (if needed)

```python
from bci_compression.formats import adapt_data

adapted_data, settings = adapt_data(
    raw_data,
    source_system='biosemi_64',
    target_sampling_rate=1000
)
```

### Step 3: Compress Using Recommended Algorithm

```python
from bci_compression.algorithms.emg_compression import EMGLZCompressor

compressor = EMGLZCompressor()
compressed = compressor.compress(adapted_data)

stats = compressor.compression_stats
print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
```

## Adding Custom Systems

You can create custom system profiles:

```python
from bci_compression.formats import BCISystemProfile, ElectrodeStandard

custom_system = BCISystemProfile(
    name="My Custom BCI",
    num_channels=32,
    sampling_rate=512,
    electrode_standard=ElectrodeStandard.INTERNATIONAL_10_20,
    bit_depth=16,
    voltage_range=(-100, 100),
    recommended_compression="emg_lz",
    data_type="EEG",
    manufacturer="Custom Lab",
    description="32-channel custom EEG system"
)

# Use custom profile
from bci_compression.formats import BCIDataAdapter

adapter = BCIDataAdapter(source_profile=custom_system)
```

## Electrode Standards

The toolkit supports several electrode placement standards:

- **10-20 System**: Standard 21-electrode system
- **10-10 System**: Extended 74-electrode system
- **10-5 System**: High-density 345-electrode system
- **GSN HydroCel**: EGI Geodesic Sensor Net
- **BioSemi**: BioSemi electrode system
- **Custom**: Custom electrode layouts

## Data Rate Calculations

For high-density systems, data rates can be substantial:

```python
from bci_compression.formats import get_system_profile

profile = get_system_profile('neuropixels')

bytes_per_sample = profile.bit_depth / 8
data_rate_mbps = (profile.num_channels * profile.sampling_rate * 
                  bytes_per_sample) / (1024 * 1024)

print(f"Uncompressed data rate: {data_rate_mbps:.2f} MB/s")
```

**Example Output:**
- **Neuropixels**: 13.7 MB/s → 1,158 GB/day
- **Blackrock 96**: 5.5 MB/s → 463 GB/day
- **Intan 64**: 2.4 MB/s → 206 GB/day

## Performance Recommendations

### Low-Density Systems (8-32 channels)
- **Algorithm**: EMG LZ or Adaptive LZ
- **Latency**: < 1 ms
- **Compression**: 2-4x typical

### Medium-Density Systems (64-128 channels)
- **Algorithm**: Neural LZ or Wavelet Transform
- **Latency**: < 5 ms
- **Compression**: 3-6x typical

### High-Density Systems (256+ channels)
- **Algorithm**: Neural LZ with GPU acceleration
- **Latency**: < 10 ms
- **Compression**: 4-8x typical

## Examples

See `examples/multi_system_demo.py` for a comprehensive demonstration of:

- Listing all supported systems
- Loading system profiles
- Adapting data between systems
- Compressing data from different sources
- Calculating data rates and storage requirements

## API Reference

### `get_system_profile(system_name: str) -> BCISystemProfile`

Load a predefined system profile.

**Parameters:**
- `system_name`: Name of the system (case-insensitive)

**Returns:**
- `BCISystemProfile` object

**Raises:**
- `ValueError` if system name is not found

### `list_supported_systems() -> List[Dict]`

List all supported BCI systems.

**Returns:**
- List of dictionaries with system information

### `adapt_data(data, source_system, target_system=None, target_sampling_rate=None, normalize=False) -> Tuple[ndarray, dict]`

Adapt data between BCI systems.

**Parameters:**
- `data`: Input data (channels, samples)
- `source_system`: Source system name
- `target_system`: Target system name (optional)
- `target_sampling_rate`: Override sampling rate (optional)
- `normalize`: Normalize to [-1, 1] range

**Returns:**
- `adapted_data`: Transformed data
- `settings`: Recommended compression settings

### `BCIDataAdapter`

Class for converting BCI data between formats.

**Methods:**
- `adapt(data)`: Adapt data to target format
- `get_compression_settings()`: Get recommended settings

## Troubleshooting

### Window Size Errors

For very short data segments, filter window sizes may need adjustment. The toolkit automatically handles this for segments as short as 5 samples.

### Resampling Quality

The default adapter uses linear interpolation for speed. For production use with critical applications, consider using `scipy.signal.resample_poly` for better quality.

### Memory Usage

High-density systems with many channels may require significant memory. Process data in chunks if memory is limited:

```python
chunk_size = 1000  # samples
for i in range(0, data.shape[1], chunk_size):
    chunk = data[:, i:i+chunk_size]
    compressed_chunk = compressor.compress(chunk)
    # Process compressed_chunk
```

## Future Extensions

Planned additions:

- [ ] Support for more BCI systems (g.tec, ANT Neuro, etc.)
- [ ] EDF/BDF file format readers
- [ ] Automatic system detection from file headers
- [ ] Channel remapping between electrode standards
- [ ] Artifact detection profiles per system
- [ ] System-specific filtering recommendations

## References

- OpenBCI Documentation: https://docs.openbci.com/
- BioSemi System Specifications: https://www.biosemi.com/
- EGI Technical Specifications: https://www.egi.com/
- Neuropixels Probe Specifications: https://www.neuropixels.org/

---

For more information, see:
- [User Guide](user_guide.md)
- [API Documentation](api_documentation.md)
- [Benchmarking Guide](benchmarking_guide.md)
