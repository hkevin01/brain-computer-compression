"""
Tests for BCI device adapters module.
"""

import pytest
import numpy as np
import tempfile
import os
from bci_compression.adapters import (
    map_channels,
    resample,
    apply_channel_groups,
    load_mapping_file,
    save_mapping_file,
    apply_calibration,
)
from bci_compression.adapters.openbci import (
    OpenBCIAdapter,
    convert_openbci_to_standard,
    OPENBCI_CYTON_8CH_MAPPING,
)


class TestChannelMapping:
    """Test channel mapping functionality."""

    def test_basic_channel_mapping(self):
        """Test basic index-based channel mapping."""
        # Create test data: 4 channels x 100 samples
        data = np.random.randn(4, 100)
        mapping = {'ch_0': 'A', 'ch_1': 'B', 'ch_2': 'C', 'ch_3': 'D'}

        result = map_channels(data, mapping)
        assert result.shape == data.shape

    def test_channel_reordering(self):
        """Test that channel mapping can reorder channels."""
        # Create distinct data for each channel
        data = np.array([
            [1, 1, 1],  # channel 0
            [2, 2, 2],  # channel 1
            [3, 3, 3],  # channel 2
        ], dtype=float)

        # Map channel 0 -> 2, channel 2 -> 0
        mapping = {0: 2, 1: 1, 2: 0}
        result = map_channels(data, mapping)

        # Check that channels were swapped
        np.testing.assert_array_equal(result[0], [3, 3, 3])
        np.testing.assert_array_equal(result[2], [1, 1, 1])

    def test_transposed_input(self):
        """Test handling of transposed input (samples x channels)."""
        # samples x channels format
        data = np.random.randn(100, 4)
        mapping = {'ch_0': 'A', 'ch_1': 'B', 'ch_2': 'C', 'ch_3': 'D'}

        result = map_channels(data, mapping)
        # Should return in same format
        assert result.shape == data.shape


class TestResampling:
    """Test resampling functionality."""

    def test_no_resampling_needed(self):
        """Test that same rate returns unchanged data."""
        data = np.random.randn(4, 1000)
        result = resample(data, 1000, 1000)
        np.testing.assert_array_equal(result, data)

    def test_downsampling(self):
        """Test downsampling reduces sample count."""
        data = np.random.randn(4, 1000)
        result = resample(data, 1000, 500, method='polyphase')

        # Should have ~half the samples
        assert result.shape[0] == 4
        assert 480 <= result.shape[1] <= 520  # Allow some rounding

    def test_upsampling(self):
        """Test upsampling increases sample count."""
        data = np.random.randn(4, 1000)
        result = resample(data, 1000, 2000, method='polyphase')

        # Should have ~double the samples
        assert result.shape[0] == 4
        assert 1980 <= result.shape[1] <= 2020  # Allow some rounding

    def test_fft_method(self):
        """Test FFT-based resampling."""
        data = np.random.randn(4, 1000)
        result = resample(data, 1000, 500, method='fft')

        assert result.shape[0] == 4
        assert result.shape[1] == 500

    def test_polyphase_method(self):
        """Test polyphase resampling."""
        data = np.random.randn(4, 1000)
        result = resample(data, 1000, 750, method='polyphase')

        assert result.shape[0] == 4
        assert 740 <= result.shape[1] <= 760


class TestChannelGrouping:
    """Test channel grouping functionality."""

    def test_mean_reducer(self):
        """Test mean reduction of channel groups."""
        # Create data in channels x samples format (4 channels x 3 samples)
        data = np.array([
            [1.0, 2.0, 3.0],  # ch 0
            [2.0, 3.0, 4.0],  # ch 1
            [3.0, 4.0, 5.0],  # ch 2
            [4.0, 5.0, 6.0],  # ch 3
        ])

        groups = {
            'group1': [0, 1],
            'group2': [2, 3],
        }

        result = apply_channel_groups(data, groups, reducer='mean')

        assert 'group1' in result
        assert 'group2' in result
        np.testing.assert_array_almost_equal(result['group1'][0], [1.5, 2.5, 3.5])
        np.testing.assert_array_almost_equal(result['group2'][0], [3.5, 4.5, 5.5])

    def test_median_reducer(self):
        """Test median reduction of channel groups."""
        data = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [10.0, 10.0, 10.0],  # outlier
        ])

        groups = {'all': [0, 1, 2]}
        result = apply_channel_groups(data, groups, reducer='median')

        np.testing.assert_array_equal(result['all'][0], [2.0, 3.0, 4.0])

    def test_first_reducer(self):
        """Test first channel selection."""
        data = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])

        groups = {'group': [0, 1]}
        result = apply_channel_groups(data, groups, reducer='first')

        np.testing.assert_array_equal(result['group'][0], [1.0, 2.0, 3.0])

    def test_concat_reducer(self):
        """Test concatenation of channels."""
        data = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ])

        groups = {'group': [0, 1]}
        result = apply_channel_groups(data, groups, reducer='concat')

        assert result['group'].shape == (2, 2)


class TestMappingFiles:
    """Test mapping file I/O."""

    def test_save_and_load_yaml(self):
        """Test saving and loading YAML mapping files."""
        mapping = {
            'device': 'test_device',
            'sampling_rate': 1000,
            'mapping': {'ch_0': 'A', 'ch_1': 'B'},
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            filepath = f.name

        try:
            save_mapping_file(mapping, filepath)
            loaded = load_mapping_file(filepath)

            assert loaded['device'] == 'test_device'
            assert loaded['sampling_rate'] == 1000
            assert loaded['mapping']['ch_0'] == 'A'
        finally:
            os.unlink(filepath)

    def test_save_and_load_json(self):
        """Test saving and loading JSON mapping files."""
        mapping = {
            'device': 'test_device',
            'channels': [0, 1, 2],
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            save_mapping_file(mapping, filepath)
            loaded = load_mapping_file(filepath)

            assert loaded['device'] == 'test_device'
            assert loaded['channels'] == [0, 1, 2]
        finally:
            os.unlink(filepath)


class TestCalibration:
    """Test calibration functionality."""

    def test_scaling(self):
        """Test per-channel scaling."""
        data = np.ones((3, 10))
        calibration = {'scale': [1.0, 2.0, 0.5]}

        result = apply_calibration(data, calibration)

        np.testing.assert_array_equal(result[0], 1.0)
        np.testing.assert_array_equal(result[1], 2.0)
        np.testing.assert_array_equal(result[2], 0.5)

    def test_offset(self):
        """Test per-channel offset."""
        data = np.zeros((3, 10))
        calibration = {'offset': [1.0, -1.0, 0.5]}

        result = apply_calibration(data, calibration)

        np.testing.assert_array_equal(result[0], 1.0)
        np.testing.assert_array_equal(result[1], -1.0)
        np.testing.assert_array_equal(result[2], 0.5)

    def test_bad_channels(self):
        """Test bad channel masking."""
        data = np.ones((4, 10))
        calibration = {'bad_channels': [1, 3]}

        result = apply_calibration(data, calibration)

        np.testing.assert_array_equal(result[0], 1.0)
        np.testing.assert_array_equal(result[1], 0.0)
        np.testing.assert_array_equal(result[2], 1.0)
        np.testing.assert_array_equal(result[3], 0.0)

    def test_combined_calibration(self):
        """Test combined scaling, offset, and bad channel masking."""
        data = np.ones((3, 10))
        calibration = {
            'scale': [2.0, 1.0, 1.0],
            'offset': [1.0, 0.0, 0.0],
            'bad_channels': [2],
        }

        result = apply_calibration(data, calibration)

        np.testing.assert_array_equal(result[0], 3.0)  # 1.0 * 2.0 + 1.0
        np.testing.assert_array_equal(result[1], 1.0)
        np.testing.assert_array_equal(result[2], 0.0)  # masked


class TestOpenBCIAdapter:
    """Test OpenBCI adapter."""

    def test_cyton_8ch_initialization(self):
        """Test Cyton 8-channel adapter initialization."""
        adapter = OpenBCIAdapter(device='cyton_8ch')

        assert adapter.device == 'cyton_8ch'
        assert adapter.sampling_rate == 250
        assert len(adapter.mapping['mapping']) == 8

    def test_daisy_16ch_initialization(self):
        """Test Daisy 16-channel adapter initialization."""
        adapter = OpenBCIAdapter(device='daisy_16ch')

        assert adapter.device == 'daisy_16ch'
        assert adapter.sampling_rate == 250
        assert len(adapter.mapping['mapping']) == 16

    def test_convert(self):
        """Test data conversion."""
        adapter = OpenBCIAdapter(device='cyton_8ch')
        data = np.random.randn(8, 1000)

        result = adapter.convert(data)
        assert result.shape == data.shape

    def test_resample_to(self):
        """Test resampling through adapter."""
        adapter = OpenBCIAdapter(device='cyton_8ch')
        data = np.random.randn(8, 1000)

        result = adapter.resample_to(data, target_rate=500)

        assert result.shape[0] == 8
        assert 1900 <= result.shape[1] <= 2100  # ~2000 samples at 500Hz

    def test_get_channel_groups(self):
        """Test getting channel groups."""
        adapter = OpenBCIAdapter(device='cyton_8ch')
        groups = adapter.get_channel_groups()

        assert 'frontal' in groups
        assert 'central' in groups
        assert len(groups['frontal']) == 2

    def test_save_and_load_mapping(self):
        """Test saving and loading adapter mapping."""
        adapter = OpenBCIAdapter(device='cyton_8ch')

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            filepath = f.name

        try:
            adapter.save_mapping(filepath)
            loaded_adapter = OpenBCIAdapter.from_file(filepath)

            assert loaded_adapter.device == adapter.device
            assert loaded_adapter.sampling_rate == adapter.sampling_rate
        finally:
            os.unlink(filepath)

    def test_quick_converter(self):
        """Test quick converter function."""
        data = np.random.randn(8, 1000)
        result = convert_openbci_to_standard(data, device='cyton_8ch')

        assert result.shape == data.shape

    def test_quick_converter_with_resampling(self):
        """Test quick converter with resampling."""
        data = np.random.randn(8, 1000)
        result = convert_openbci_to_standard(
            data,
            device='cyton_8ch',
            target_rate=500
        )

        assert result.shape[0] == 8
        assert 1900 <= result.shape[1] <= 2100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
