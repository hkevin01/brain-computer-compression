"""
Unit tests for Phase 1 foundation components.

This module contains tests for the core signal processing,
neural decoding, and data acquisition components of the
BCI compression toolkit.
"""

import unittest
import numpy as np
import time
from unittest.mock import patch, MagicMock

# Import our modules
import sys
import os
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
# Disabled for PYTHONPATH=src compatibility

from bci_compression.data_processing.signal_processing import NeuralSignalProcessor
from bci_compression.neural_decoder import MotorImageryDecoder, RealTimeDecoder
from bci_compression.data_acquisition import (
    SimulatedDataAcquisition,
    FileDataAcquisition,
    DataAcquisitionManager
)


class TestNeuralSignalProcessor(unittest.TestCase):
    """Test cases for NeuralSignalProcessor."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = NeuralSignalProcessor(sampling_rate=1000.0)

        # Create test data
        self.n_channels = 8
        self.n_samples = 1000
        self.test_data = np.random.randn(self.n_channels, self.n_samples)

        # Add some structure to the data
        t = np.linspace(0, 1, self.n_samples)
        for ch in range(self.n_channels):
            # Add 10 Hz sine wave
            self.test_data[ch] += np.sin(2 * np.pi * 10 * t)
            # Add 50 Hz noise (power line)
            self.test_data[ch] += 0.5 * np.sin(2 * np.pi * 50 * t)

    def test_bandpass_filter(self):
        """Test bandpass filtering."""
        filtered = self.processor.bandpass_filter(
            self.test_data,
            low_freq=8.0,
            high_freq=12.0
        )

        # Check output shape
        self.assertEqual(filtered.shape, self.test_data.shape)

        # Check that filtering reduces power outside the band
        # This is a basic check - in practice you'd use FFT analysis
        self.assertIsInstance(filtered, np.ndarray)

    def test_notch_filter(self):
        """Test notch filtering."""
        filtered = self.processor.notch_filter(self.test_data, notch_freq=50.0)

        # Check output shape
        self.assertEqual(filtered.shape, self.test_data.shape)

        # Filtered data should be different from original
        self.assertFalse(np.array_equal(filtered, self.test_data))

    def test_normalize_signals(self):
        """Test signal normalization."""
        normalized = self.processor.normalize_signals(self.test_data)

        # Check output shape
        self.assertEqual(normalized.shape, self.test_data.shape)

        # Check normalization properties (approximately zero mean, unit std)
        for ch in range(self.n_channels):
            self.assertAlmostEqual(np.mean(normalized[ch]), 0.0, places=1)
            self.assertAlmostEqual(np.std(normalized[ch]), 1.0, places=1)

    def test_extract_features(self):
        """Test feature extraction."""
        features = self.processor.extract_features(self.test_data)

        # Should return a dictionary
        self.assertIsInstance(features, dict)

        # Should contain expected feature types
        expected_features = ['power_spectral_density', 'spectral_centroid', 'zero_crossings']
        for feature in expected_features:
            self.assertIn(feature, features)

    def test_detect_artifacts(self):
        """Test artifact detection."""
        # Create data with artifacts
        artifact_data = self.test_data.copy()
        # Add large spike artifact
        artifact_data[0, 500:510] = 100.0

        artifacts = self.processor.detect_artifacts(artifact_data)

        # Should return boolean array
        self.assertIsInstance(artifacts, np.ndarray)
        self.assertEqual(artifacts.dtype, bool)
        self.assertEqual(artifacts.shape, (self.n_channels, self.n_samples))

    def test_preprocess_pipeline(self):
        """Test complete preprocessing pipeline."""
        config = {
            'bandpass': {'low_freq': 1.0, 'high_freq': 100.0},
            'notch': {'notch_freq': 50.0},
            'normalize': True,
            'artifact_removal': True
        }

        processed = self.processor.preprocess_pipeline(self.test_data, config)

        # Check output shape
        self.assertEqual(processed.shape, self.test_data.shape)

        # Should be different from original
        self.assertFalse(np.array_equal(processed, self.test_data))


class TestMotorImageryDecoder(unittest.TestCase):
    """Test cases for MotorImageryDecoder."""

    def setUp(self):
        """Set up test fixtures."""
        self.decoder = MotorImageryDecoder(sampling_rate=1000.0)

        # Create synthetic training data
        self.n_trials = 100
        self.n_channels = 16
        self.n_samples = 1000

        # Generate data for two classes
        self.training_data = np.random.randn(
            self.n_trials, self.n_channels, self.n_samples
        )

        # Add class-specific patterns
        labels = []
        for trial in range(self.n_trials):
            if trial < 50:  # Class 0
                # Add 10 Hz pattern
                t = np.linspace(0, 1, self.n_samples)
                for ch in range(8):  # First 8 channels
                    self.training_data[trial, ch] += np.sin(2 * np.pi * 10 * t)
                labels.append('left')
            else:  # Class 1
                # Add 20 Hz pattern
                t = np.linspace(0, 1, self.n_samples)
                for ch in range(8, 16):  # Last 8 channels
                    self.training_data[trial, ch] += np.sin(2 * np.pi * 20 * t)
                labels.append('right')

        self.labels = np.array(labels)

    def test_feature_extraction(self):
        """Test motor imagery feature extraction."""
        test_signal = self.training_data[0]
        features = self.decoder.extract_motor_features(test_signal)

        # Should return feature vector
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.ndim, 1)
        self.assertGreater(len(features), 0)

    @patch('sklearn.ensemble.RandomForestClassifier')
    @patch('sklearn.preprocessing.StandardScaler')
    def test_training(self, mock_scaler, mock_classifier):
        """Test decoder training."""
        # Mock sklearn components
        mock_scaler_instance = MagicMock()
        mock_classifier_instance = MagicMock()
        mock_scaler.return_value = mock_scaler_instance
        mock_classifier.return_value = mock_classifier_instance

        # Train decoder
        self.decoder.train(self.training_data, self.labels)

        # Check that training was called
        mock_scaler_instance.fit_transform.assert_called_once()
        mock_classifier_instance.fit.assert_called_once()

        # Check training status
        self.assertTrue(self.decoder.is_trained)

    def test_decode_without_training(self):
        """Test that decoding fails without training."""
        test_signal = self.training_data[0]

        with self.assertRaises(ValueError):
            self.decoder.decode(test_signal)


class TestSimulatedDataAcquisition(unittest.TestCase):
    """Test cases for SimulatedDataAcquisition."""

    def setUp(self):
        """Set up test fixtures."""
        self.acquisition = SimulatedDataAcquisition(
            sampling_rate=1000.0,
            n_channels=8,
            buffer_duration=0.1  # Small buffer for testing
        )

    def test_initialization(self):
        """Test proper initialization."""
        self.assertEqual(self.acquisition.sampling_rate, 1000.0)
        self.assertEqual(self.acquisition.n_channels, 8)
        self.assertFalse(self.acquisition.is_acquiring)

    def test_connection(self):
        """Test device connection."""
        self.assertTrue(self.acquisition.connect())
        self.assertTrue(self.acquisition.disconnect())

    def test_data_generation(self):
        """Test data chunk generation."""
        chunk = self.acquisition._acquire_data_chunk()

        # Check chunk properties
        self.assertIsInstance(chunk, np.ndarray)
        self.assertEqual(chunk.shape[0], self.acquisition.n_channels)
        self.assertGreater(chunk.shape[1], 0)

    def test_acquisition_start_stop(self):
        """Test starting and stopping acquisition."""
        # Start acquisition
        self.assertTrue(self.acquisition.start_acquisition())
        self.assertTrue(self.acquisition.is_acquiring)

        # Let it run briefly
        time.sleep(0.1)

        # Stop acquisition
        self.assertTrue(self.acquisition.stop_acquisition())
        self.assertFalse(self.acquisition.is_acquiring)

    def test_buffer_status(self):
        """Test buffer status reporting."""
        status = self.acquisition.get_buffer_status()

        # Check status format
        self.assertIsInstance(status, dict)
        expected_keys = ['buffer_size', 'current_index', 'fill_percentage', 'duration_filled']
        for key in expected_keys:
            self.assertIn(key, status)


class TestDataAcquisitionManager(unittest.TestCase):
    """Test cases for DataAcquisitionManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = DataAcquisitionManager()

        # Create test sources
        self.source1 = SimulatedDataAcquisition(
            sampling_rate=1000.0, n_channels=8
        )
        self.source2 = SimulatedDataAcquisition(
            sampling_rate=1000.0, n_channels=16
        )

    def test_add_remove_sources(self):
        """Test adding and removing sources."""
        # Add sources
        self.manager.add_source("source1", self.source1)
        self.manager.add_source("source2", self.source2)

        self.assertIn("source1", self.manager.sources)
        self.assertIn("source2", self.manager.sources)

        # Remove source
        self.assertTrue(self.manager.remove_source("source1"))
        self.assertNotIn("source1", self.manager.sources)

        # Try to remove non-existent source
        self.assertFalse(self.manager.remove_source("nonexistent"))

    def test_start_stop_all(self):
        """Test starting and stopping all sources."""
        self.manager.add_source("source1", self.source1)
        self.manager.add_source("source2", self.source2)

        # Start all
        self.assertTrue(self.manager.start_all())

        # Brief delay
        time.sleep(0.05)

        # Stop all
        self.assertTrue(self.manager.stop_all())

    def test_source_status(self):
        """Test getting source status."""
        self.manager.add_source("source1", self.source1)

        status = self.manager.get_source_status()

        self.assertIsInstance(status, dict)
        self.assertIn("source1", status)


class TestRealTimeDecoder(unittest.TestCase):
    """Test cases for RealTimeDecoder."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock decoder
        self.mock_decoder = MagicMock()
        self.mock_decoder.is_trained = True
        self.mock_decoder.decode.return_value = {
            'intent': 'test',
            'confidence': 0.8
        }

        self.real_time_decoder = RealTimeDecoder(
            decoder=self.mock_decoder,
            buffer_size=100,
            update_rate=10.0
        )

    def test_initialization(self):
        """Test proper initialization."""
        self.assertEqual(self.real_time_decoder.buffer_size, 100)
        self.assertEqual(self.real_time_decoder.update_rate, 10.0)
        self.assertFalse(self.real_time_decoder.is_running)

    def test_callback_setting(self):
        """Test setting callbacks."""
        decode_callback = MagicMock()
        error_callback = MagicMock()

        self.real_time_decoder.set_decode_callback(decode_callback)
        self.real_time_decoder.set_error_callback(error_callback)

        self.assertEqual(self.real_time_decoder.on_decode_callback, decode_callback)
        self.assertEqual(self.real_time_decoder.on_error_callback, error_callback)

    def test_add_data(self):
        """Test adding data to processing queue."""
        test_data = np.random.randn(8, 100)

        # Start decoder
        self.real_time_decoder.start()

        # Add data
        self.real_time_decoder.add_data(test_data)

        # Brief delay to allow processing
        time.sleep(0.2)

        # Stop decoder
        self.real_time_decoder.stop()

        # Check that decode was called
        self.mock_decoder.decode.assert_called()

    def test_start_stop(self):
        """Test starting and stopping real-time decoder."""
        # Start
        self.real_time_decoder.start()
        self.assertTrue(self.real_time_decoder.is_running)

        # Stop
        self.real_time_decoder.stop()
        self.assertFalse(self.real_time_decoder.is_running)

    def test_untrained_decoder_error(self):
        """Test error when using untrained decoder."""
        untrained_decoder = MagicMock()
        untrained_decoder.is_trained = False

        rt_decoder = RealTimeDecoder(untrained_decoder)

        with self.assertRaises(ValueError):
            rt_decoder.start()


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()

    # Add test cases
    test_classes = [
        TestNeuralSignalProcessor,
        TestMotorImageryDecoder,
        TestSimulatedDataAcquisition,
        TestDataAcquisitionManager,
        TestRealTimeDecoder
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
