#!/usr/bin/env python3
"""
Quick debug script for signal processing issue.
"""

import sys
import os
import numpy as np

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bci_compression.data_processing.signal_processing import NeuralSignalProcessor

# Create processor
processor = NeuralSignalProcessor(sampling_rate=1000.0)

# Create test data
n_channels, n_samples = 8, 1000
test_data = np.random.randn(n_channels, n_samples)

print(f"Input data shape: {test_data.shape}")
print(f"Input data type: {type(test_data)}")

try:
    # Test bandpass filter
    filtered = processor.bandpass_filter(test_data, 8.0, 30.0)
    print(f"Filtered data: {type(filtered)}")
    if hasattr(filtered, 'shape'):
        print(f"Filtered shape: {filtered.shape}")
    else:
        print(f"Filtered data: {filtered}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
