#!/usr/bin/env python3
"""
Simple validation script for Phase 1 foundation components.

This script tests our core BCI toolkit components without requiring
external dependencies that may not be installed.
"""

import sys
import os
import numpy as np
import time

# Add source to path
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
# Disabled for PYTHONPATH=src compatibility

def test_signal_processing():
    """Test basic signal processing functionality."""
    print("Testing NeuralSignalProcessor...")

    try:
        from bci_compression.data_processing.signal_processing import NeuralSignalProcessor

        # Create processor
        processor = NeuralSignalProcessor(sampling_rate=1000.0)

        # Create test data
        n_channels, n_samples = 8, 1000
        test_data = np.random.randn(n_channels, n_samples)

        # Test bandpass filter
        print("  Testing bandpass filter...")
        filtered = processor.bandpass_filter(test_data, 8.0, 30.0)
        assert filtered.shape == test_data.shape, f"Bandpass filter shape mismatch: {filtered.shape} vs {test_data.shape}"

        # Test normalization
        print("  Testing normalization...")
        normalized, params = processor.normalize_signals(test_data)
        assert normalized.shape == test_data.shape, f"Normalization shape mismatch: {normalized.shape} vs {test_data.shape}"
        assert isinstance(params, dict), f"Normalization params should be dict, got {type(params)}"

        # Test feature extraction
        print("  Testing feature extraction...")
        features = processor.extract_features(test_data)
        assert isinstance(features, dict), f"Features should be a dictionary, got {type(features)}"

        print("✅ NeuralSignalProcessor tests passed")
        return True

    except Exception as e:
        print(f"❌ NeuralSignalProcessor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_acquisition():
    """Test data acquisition functionality."""
    print("Testing SimulatedDataAcquisition...")

    try:
        from bci_compression.data_acquisition import SimulatedDataAcquisition

        # Create acquisition system
        acq = SimulatedDataAcquisition(
            sampling_rate=1000.0,
            n_channels=8,
            buffer_duration=0.1
        )

        # Test connection
        assert acq.connect(), "Connection failed"
        assert acq.disconnect(), "Disconnection failed"

        # Test data generation
        chunk = acq._acquire_data_chunk()
        assert chunk is not None, "Data chunk is None"
        assert chunk.shape[0] == 8, "Wrong number of channels"
        assert chunk.shape[1] > 0, "Empty data chunk"

        # Test acquisition start/stop
        assert acq.start_acquisition(), "Failed to start acquisition"
        time.sleep(0.1)  # Let it run briefly
        assert acq.stop_acquisition(), "Failed to stop acquisition"

        print("✅ SimulatedDataAcquisition tests passed")
        return True

    except Exception as e:
        print(f"❌ SimulatedDataAcquisition test failed: {e}")
        return False


def test_neural_decoder():
    """Test neural decoder functionality."""
    print("Testing MotorImageryDecoder...")

    try:
        from bci_compression.neural_decoder import MotorImageryDecoder

        # Create decoder
        decoder = MotorImageryDecoder(sampling_rate=1000.0)

        # Test feature extraction
        test_signal = np.random.randn(8, 1000)
        features = decoder.extract_motor_features(test_signal)

        assert isinstance(features, np.ndarray), "Features should be numpy array"
        assert features.ndim == 1, "Features should be 1D"
        assert len(features) > 0, "Empty feature vector"

        print("✅ MotorImageryDecoder tests passed")
        return True

    except Exception as e:
        print(f"❌ MotorImageryDecoder test failed: {e}")
        return False


def test_real_time_decoder():
    """Test real-time decoder functionality."""
    print("Testing RealTimeDecoder...")

    try:
        from bci_compression.neural_decoder import RealTimeDecoder
        from unittest.mock import MagicMock

        # Create mock decoder
        mock_decoder = MagicMock()
        mock_decoder.is_trained = True
        mock_decoder.decode.return_value = {'intent': 'test', 'confidence': 0.8}

        # Create real-time decoder
        rt_decoder = RealTimeDecoder(
            decoder=mock_decoder,
            buffer_size=100,
            update_rate=10.0
        )

        # Test initialization
        assert rt_decoder.buffer_size == 100, "Wrong buffer size"
        assert rt_decoder.update_rate == 10.0, "Wrong update rate"
        assert not rt_decoder.is_running, "Should not be running initially"

        # Test start/stop
        rt_decoder.start()
        assert rt_decoder.is_running, "Should be running after start"

        time.sleep(0.1)  # Brief delay

        rt_decoder.stop()
        assert not rt_decoder.is_running, "Should not be running after stop"

        print("✅ RealTimeDecoder tests passed")
        return True

    except Exception as e:
        print(f"❌ RealTimeDecoder test failed: {e}")
        return False


def test_device_controller():
    """Test device controller functionality."""
    print("Testing DeviceController...")

    try:
        from bci_compression.neural_decoder import DeviceController

        # Create controller
        controller = DeviceController()

        # Create mock device function
        def mock_device_control(command, **kwargs):
            return f"Executed {command}"

        # Register device
        controller.register_device("test_device", mock_device_control)
        assert "test_device" in controller.connected_devices, "Device not registered"

        # Test command execution
        result = controller.execute_command("test_device", "test_command")
        assert result, "Command execution failed"

        # Check command history
        assert len(controller.command_history) > 0, "No command history"

        print("✅ DeviceController tests passed")
        return True

    except Exception as e:
        print(f"❌ DeviceController test failed: {e}")
        return False


def test_factory_functions():
    """Test factory functions."""
    print("Testing factory functions...")

    try:
        from bci_compression.neural_decoder import create_motor_imagery_system
        from bci_compression.data_acquisition import create_test_acquisition_system

        # Test motor imagery system creation
        decoder, rt_system, controller = create_motor_imagery_system()
        assert decoder is not None, "Decoder is None"
        assert rt_system is not None, "Real-time system is None"
        assert controller is not None, "Controller is None"

        # Test acquisition system creation
        acq_manager = create_test_acquisition_system(n_channels=16)
        assert acq_manager is not None, "Acquisition manager is None"
        assert "simulated" in acq_manager.sources, "Simulated source not found"

        print("✅ Factory function tests passed")
        return True

    except Exception as e:
        print(f"❌ Factory function test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Brain-Computer Interface Toolkit - Phase 1 Validation")
    print("=" * 60)

    tests = [
        test_signal_processing,
        test_data_acquisition,
        test_neural_decoder,
        test_real_time_decoder,
        test_device_controller,
        test_factory_functions
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()  # Add spacing between tests

    print("=" * 60)
    print(f"PHASE 1 VALIDATION SUMMARY")
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")

    if passed == total:
        print("🎉 All Phase 1 foundation components are working correctly!")
        print("✅ Ready to proceed to Phase 2: Core Compression Algorithms")
    else:
        print("⚠️  Some tests failed. Please review the implementation.")

    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
