# BCI Device Adapters Guide

## Overview

The BCI device adapters module provides utilities to work with data from different brain-computer interface acquisition systems. The module handles electrode mapping, resampling, channel grouping, and calibration to create a standardized data processing pipeline.

## Motivation

Different BCI systems use different:
- **Electrode layouts** (10-20 system, custom arrays, etc.)
- **Channel naming** (numeric indices, anatomical names, custom labels)
- **Sampling rates** (250 Hz, 1000 Hz, 30 kHz, etc.)
- **Calibration parameters** (gain, offset, impedance)

The adapters module provides a consistent interface for working with all these systems.

## Quick Start

### Basic Channel Mapping

```python
from bci_compression.adapters import map_channels

# Define mapping from device channels to standard electrode names
mapping = {
    'ch_0': 'Fp1',
    'ch_1': 'Fp2',
    'ch_2': 'C3',
    'ch_3': 'C4'
}

# Apply mapping (data is channels x samples)
standardized_data = map_channels(raw_data, mapping)
```

### Resampling

```python
from bci_compression.adapters import resample

# Downsample from 30kHz to 10kHz
resampled = resample(data, src_rate=30000, dst_rate=10000, method='polyphase')

# Upsample from 250Hz to 1000Hz
resampled = resample(data, src_rate=250, dst_rate=1000, method='fft')
```

### Channel Grouping

```python
from bci_compression.adapters import apply_channel_groups

# Define channel groups (e.g., by brain region)
groups = {
    'motor_strip': [8, 9, 10, 11],
    'visual_cortex': [20, 21, 22, 23],
    'emg_pair': [30, 31]
}

# Average channels within each group
grouped = apply_channel_groups(data, groups, reducer='mean')

# Access grouped data
motor_data = grouped['motor_strip']  # Shape: (1 x samples)
```

### Calibration

```python
from bci_compression.adapters import apply_calibration

# Define calibration parameters
calibration = {
    'scale': [1.0, 1.2, 0.95, 1.1],  # Per-channel gain
    'offset': [0.0, -0.5, 0.3, 0.0],  # Per-channel DC offset
    'bad_channels': [2]  # Channels to zero out
}

# Apply calibration
calibrated = apply_calibration(data, calibration)
```

## OpenBCI Adapter

The OpenBCI adapter provides pre-configured mappings for OpenBCI devices.

### Cyton 8-Channel Board

```python
from bci_compression.adapters.openbci import OpenBCIAdapter

# Create adapter for Cyton 8-channel
adapter = OpenBCIAdapter(device='cyton_8ch')

# Convert raw data to standardized format
standard_data = adapter.convert(raw_data)

# Resample to target rate
resampled = adapter.resample_to(standard_data, target_rate=1000)

# Get channel groups
groups = adapter.get_channel_groups()
# Returns: {'frontal': [0,1], 'central': [2,3], ...}
```

### Daisy 16-Channel Board

```python
# Create adapter for Daisy 16-channel
adapter = OpenBCIAdapter(device='daisy_16ch')

# Convert and resample in one step
from bci_compression.adapters.openbci import convert_openbci_to_standard

standard_data = convert_openbci_to_standard(
    raw_data,
    device='daisy_16ch',
    target_rate=1000
)
```

## Complete Pipeline Example

```python
from bci_compression.adapters.openbci import OpenBCIAdapter
from bci_compression.adapters import apply_calibration
from bci_compression.algorithms.lossless import NeuralLZ77Compressor

# 1. Create adapter
adapter = OpenBCIAdapter(device='cyton_8ch')

# 2. Convert to standard format
standard_data = adapter.convert(raw_data)

# 3. Resample if needed
resampled_data = adapter.resample_to(standard_data, target_rate=1000)

# 4. Apply calibration
calibration = {
    'scale': [1.0] * 8,
    'offset': [0.0] * 8,
    'bad_channels': []
}
calibrated_data = apply_calibration(resampled_data, calibration)

# 5. Compress
compressor = NeuralLZ77Compressor()
compressed = compressor.compress(calibrated_data)

print(f"Compression ratio: {calibrated_data.nbytes / len(compressed):.2f}x")
```

## Mapping File Format

Mappings can be saved and loaded from YAML or JSON files:

### YAML Example

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

### Using Mapping Files

```python
from bci_compression.adapters import load_mapping_file, save_mapping_file
from bci_compression.adapters.openbci import OpenBCIAdapter

# Save a mapping
adapter = OpenBCIAdapter(device='cyton_8ch')
adapter.save_mapping('my_device_mapping.yaml')

# Load and use a mapping
adapter = OpenBCIAdapter.from_file('my_device_mapping.yaml')
```

## API Reference

### Core Functions

#### `map_channels(data, mapping, input_format='index', output_format='index')`
Map channels from one naming/indexing convention to another.

**Parameters:**
- `data`: Input neural data (channels x samples)
- `mapping`: Dictionary mapping source to target channels
- `input_format`: 'index' or 'name'
- `output_format`: 'index' or 'name'

**Returns:** Remapped neural data

#### `resample(data, src_rate, dst_rate, method='polyphase', axis=-1)`
Resample neural data with anti-aliasing filter.

**Parameters:**
- `data`: Input neural data
- `src_rate`: Source sampling rate in Hz
- `dst_rate`: Destination sampling rate in Hz
- `method`: 'fft' or 'polyphase'
- `axis`: Axis along which to resample

**Returns:** Resampled neural data

#### `apply_channel_groups(data, groups, reducer='mean')`
Apply logical channel grouping for spatial filtering.

**Parameters:**
- `data`: Input neural data (channels x samples)
- `groups`: Dictionary mapping group names to channel indices
- `reducer`: 'mean', 'median', 'first', or 'concat'

**Returns:** Dictionary mapping group names to processed data

#### `apply_calibration(data, calibration, channels=None)`
Apply per-channel calibration (scaling, offsets, bad channel masking).

**Parameters:**
- `data`: Input neural data (channels x samples)
- `calibration`: Dictionary with 'scale', 'offset', 'bad_channels' keys
- `channels`: Optional list of channel indices to calibrate

**Returns:** Calibrated neural data

### OpenBCI Adapter

#### `OpenBCIAdapter(device='cyton_8ch', custom_mapping=None)`
Adapter for OpenBCI device data.

**Parameters:**
- `device`: 'cyton_8ch' or 'daisy_16ch'
- `custom_mapping`: Optional custom mapping dictionary

**Methods:**
- `convert(data, apply_mapping=True)`: Convert to standardized format
- `resample_to(data, target_rate, method='polyphase')`: Resample data
- `get_channel_groups()`: Get channel groupings
- `save_mapping(filepath)`: Save mapping to file
- `from_file(filepath)`: Load adapter from mapping file

## Best Practices

1. **Always specify data format**: Make it clear whether your data is channels x samples or samples x channels
2. **Use mapping files**: Store device configurations in version-controlled YAML files
3. **Validate after resampling**: Check that signal characteristics are preserved
4. **Document calibration parameters**: Keep records of gain/offset values for reproducibility
5. **Group related channels**: Use channel grouping to reduce dimensionality before compression

## Extending the Module

### Adding a New Device Adapter

```python
from bci_compression.adapters.openbci import OpenBCIAdapter

# Define your device mapping
MY_DEVICE_MAPPING = {
    'device': 'my_custom_array',
    'sampling_rate': 2000,
    'channels': 32,
    'mapping': {
        'ch_0': 'electrode_A1',
        'ch_1': 'electrode_A2',
        # ... more mappings
    },
    'channel_groups': {
        'region_1': [0, 1, 2, 3],
        'region_2': [4, 5, 6, 7],
    }
}

# Create adapter
adapter = OpenBCIAdapter(custom_mapping=MY_DEVICE_MAPPING)
```

## Performance Considerations

- **Polyphase resampling** is faster than FFT for integer rate ratios
- **FFT resampling** provides higher quality for arbitrary rate ratios
- **Channel grouping** can significantly reduce data size before compression
- **Bad channel masking** should be done before compression to avoid wasting bits

## See Also

- [OpenBCI Adapter Demo](../examples/openbci_adapter_demo.py)
- [Test Suite](../tests/test_adapters.py)
- [Main README](../README.md)
