"""
Tests for Blackrock adapter.
"""

import numpy as np
import pytest
from bci_compression.adapters.blackrock import (
    BlackrockAdapter,
    BLACKROCK_NEUROPORT_96CH_MAPPING,
    BLACKROCK_CEREBUS_128CH_MAPPING,
    convert_blackrock_to_standard
)


class TestBlackrockMappings:
    """Test predefined Blackrock mappings."""

    def test_neuroport_mapping_structure(self):
        """Test Neuroport 96-channel mapping structure."""
        assert 'device' in BLACKROCK_NEUROPORT_96CH_MAPPING
        assert 'sampling_rate' in BLACKROCK_NEUROPORT_96CH_MAPPING
        assert 'channels' in BLACKROCK_NEUROPORT_96CH_MAPPING
        assert 'mapping' in BLACKROCK_NEUROPORT_96CH_MAPPING

        assert BLACKROCK_NEUROPORT_96CH_MAPPING['device'] == 'blackrock_neuroport_96ch'
        assert BLACKROCK_NEUROPORT_96CH_MAPPING['sampling_rate'] == 30000
        assert BLACKROCK_NEUROPORT_96CH_MAPPING['channels'] == 96

    def test_neuroport_channel_mapping(self):
        """Test Neuroport channel names."""
        mapping = BLACKROCK_NEUROPORT_96CH_MAPPING['mapping']

        # Should have 96 channels
        assert len(mapping) == 96

        # Check naming convention
        assert 'ch_0' in mapping
        assert 'ch_95' in mapping
        assert mapping['ch_0'] == 'electrode_001'
        assert mapping['ch_95'] == 'electrode_096'

    def test_cerebus_mapping_structure(self):
        """Test Cerebus 128-channel mapping structure."""
        assert BLACKROCK_CEREBUS_128CH_MAPPING['device'] == 'blackrock_cerebus_128ch'
        assert BLACKROCK_CEREBUS_128CH_MAPPING['sampling_rate'] == 30000
        assert BLACKROCK_CEREBUS_128CH_MAPPING['channels'] == 128

    def test_cerebus_channel_mapping(self):
        """Test Cerebus channel names."""
        mapping = BLACKROCK_CEREBUS_128CH_MAPPING['mapping']

        # Should have 128 channels
        assert len(mapping) == 128

        # Check naming convention
        assert 'ch_0' in mapping
        assert 'ch_127' in mapping

    def test_channel_groups(self):
        """Test channel grouping."""
        assert 'channel_groups' in BLACKROCK_NEUROPORT_96CH_MAPPING

        groups = BLACKROCK_NEUROPORT_96CH_MAPPING['channel_groups']

        # Check for expected groups
        assert 'grid_row_0' in groups
        assert 'grid_row_9' in groups
        assert 'motor_cortex' in groups
        assert 'sensory_cortex' in groups

        # Check grid row sizes (10x10 array, but last row has only 6)
        for i in range(9):
            assert len(groups[f'grid_row_{i}']) == 10
        # Last row has only 6 electrodes (96 total = 9*10 + 6)
        assert len(groups['grid_row_9']) == 6


class TestBlackrockAdapter:
    """Test BlackrockAdapter class."""

    def test_initialization_neuroport(self):
        """Test Neuroport adapter initialization."""
        adapter = BlackrockAdapter(device='neuroport_96ch')

        assert adapter.device == 'neuroport_96ch'
        assert adapter.mapping == BLACKROCK_NEUROPORT_96CH_MAPPING

    def test_initialization_cerebus(self):
        """Test Cerebus adapter initialization."""
        adapter = BlackrockAdapter(device='cerebus_128ch')

        assert adapter.device == 'cerebus_128ch'
        assert adapter.mapping == BLACKROCK_CEREBUS_128CH_MAPPING

    def test_initialization_invalid(self):
        """Test initialization with invalid device."""
        with pytest.raises(ValueError):
            BlackrockAdapter(device='invalid_device')

    def test_convert_basic(self):
        """Test basic data conversion."""
        adapter = BlackrockAdapter(device='neuroport_96ch')
        data = np.random.randn(96, 1000)

        converted = adapter.convert(data, apply_mapping=False)

        assert converted.shape == data.shape
        assert np.issubdtype(converted.dtype, np.floating)

    def test_resample_to(self):
        """Test resampling."""
        adapter = BlackrockAdapter(device='neuroport_96ch')
        data = np.random.randn(96, 30000)  # 1 second @ 30kHz

        resampled = adapter.resample_to(data, target_rate=1000)

        # Should have 1000 samples
        assert resampled.shape == (96, 1000)

    def test_get_channel_groups(self):
        """Test channel group retrieval."""
        adapter = BlackrockAdapter(device='neuroport_96ch')

        groups = adapter.get_channel_groups()

        assert isinstance(groups, dict)
        assert 'grid_row_0' in groups
        assert 'motor_cortex' in groups

        # Check indices are valid
        for indices in groups.values():
            assert all(0 <= i < 96 for i in indices)


class TestBlackrockConverter:
    """Test standalone converter function."""

    def test_convert_neuroport(self):
        """Test Neuroport conversion."""
        data = np.random.randn(96, 1000)

        converted = convert_blackrock_to_standard(
            data,
            device='neuroport_96ch'
        )

        assert converted.shape == data.shape
        assert np.issubdtype(converted.dtype, np.floating)

    def test_convert_cerebus(self):
        """Test Cerebus conversion."""
        data = np.random.randn(128, 1000)

        converted = convert_blackrock_to_standard(
            data,
            device='cerebus_128ch'
        )

        assert converted.shape == data.shape
        assert np.issubdtype(converted.dtype, np.floating)

    def test_convert_with_resampling(self):
        """Test conversion with resampling."""
        data = np.random.randn(96, 30000)

        converted = convert_blackrock_to_standard(
            data,
            device='neuroport_96ch',
            target_rate=1000
        )

        assert converted.shape == (96, 1000)
class TestBlackrockDataProcessing:
    """Test data processing operations."""

    def test_high_frequency_data(self):
        """Test processing high sampling rate data."""
        adapter = BlackrockAdapter(device='neuroport_96ch')

        # Generate 1 second of data at 30kHz
        t = np.linspace(0, 1, 30000)
        data = np.zeros((96, 30000))

        # Add some frequency components
        for ch in range(96):
            data[ch] = np.sin(2 * np.pi * 100 * t)  # 100 Hz component

        converted = adapter.convert(data, apply_mapping=False)

        assert converted.shape == data.shape
        assert not np.isnan(converted).any()
        assert not np.isinf(converted).any()

    def test_channel_extraction(self):
        """Test extracting specific channel groups."""
        adapter = BlackrockAdapter(device='neuroport_96ch')
        data = np.random.randn(96, 1000)

        groups = adapter.get_channel_groups()
        motor_indices = groups['motor_cortex']

        motor_data = data[motor_indices, :]

        assert motor_data.shape[0] == len(motor_indices)
        assert motor_data.shape[1] == 1000

    def test_multirate_processing(self):
        """Test processing at multiple sampling rates."""
        adapter = BlackrockAdapter(device='neuroport_96ch')
        data = np.random.randn(96, 30000)

        # Resample to different rates
        rates = [10000, 5000, 2000, 1000]

        for rate in rates:
            resampled = adapter.resample_to(data, target_rate=rate)
            expected_samples = int(30000 * rate / 30000)
            assert resampled.shape == (96, expected_samples)


class TestBlackrockMetadata:
    """Test metadata handling."""

    def test_device_info(self):
        """Test device information retrieval."""
        adapter = BlackrockAdapter(device='neuroport_96ch')

        assert adapter.mapping['device'] == 'blackrock_neuroport_96ch'
        assert adapter.mapping['sampling_rate'] == 30000
        assert adapter.mapping['channels'] == 96
        assert 'utah' in adapter.mapping.get('array_layout', '')

    def test_grid_layout(self):
        """Test Utah array grid layout."""
        adapter = BlackrockAdapter(device='neuroport_96ch')

        # Neuroport uses 10x10 grid (96 active electrodes)
        groups = adapter.get_channel_groups()

        # Should have 10 rows
        rows = [k for k in groups.keys() if k.startswith('grid_row_')]
        assert len(rows) == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
