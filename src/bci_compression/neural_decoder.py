"""
Real-time neural decoder framework for brain-computer interfaces.

This module provides the foundation for real-time neural signal decoding,
including pattern recognition, feature extraction, and device control
interfaces for BCI applications.
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Any
import time
import threading
from queue import Queue
from abc import ABC, abstractmethod


class NeuralDecoder(ABC):
    """
    Abstract base class for neural signal decoders.
    
    This class defines the interface for real-time neural decoders
    that can interpret brain signals and translate them into control
    commands for external devices.
    """
    
    def __init__(self, sampling_rate: float = 30000.0):
        """
        Initialize the neural decoder.
        
        Parameters
        ----------
        sampling_rate : float, default=30000.0
            Sampling rate of the input signals in Hz
        """
        self.sampling_rate = sampling_rate
        self.is_trained = False
        self.is_running = False
        
    @abstractmethod
    def train(self, training_data: np.ndarray, labels: np.ndarray) -> None:
        """Train the decoder with labeled neural data."""
        pass
    
    @abstractmethod
    def decode(self, neural_data: np.ndarray) -> Dict[str, Any]:
        """Decode neural signals into control commands."""
        pass
    
    @abstractmethod
    def predict_intent(self, features: np.ndarray) -> str:
        """Predict user intent from extracted features."""
        pass


class MotorImageryDecoder(NeuralDecoder):
    """
    Motor imagery decoder for movement-based BCI control.
    
    This decoder interprets motor imagery signals (imagined movements)
    and translates them into control commands for prosthetics or
    computer interfaces.
    """
    
    def __init__(
        self, 
        sampling_rate: float = 30000.0,
        frequency_bands: Dict[str, tuple] = None
    ):
        """
        Initialize motor imagery decoder.
        
        Parameters
        ----------
        sampling_rate : float, default=30000.0
            Sampling rate in Hz
        frequency_bands : dict, optional
            Frequency bands for feature extraction
        """
        super().__init__(sampling_rate)
        
        # Default frequency bands for motor imagery
        if frequency_bands is None:
            self.frequency_bands = {
                'mu': (8, 12),      # Mu rhythm
                'beta': (13, 30),   # Beta rhythm
                'gamma': (30, 100)  # Gamma rhythm
            }
        else:
            self.frequency_bands = frequency_bands
            
        self.classifier = None
        self.feature_extractor = None
        
    def extract_motor_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract motor imagery features from neural signals.
        
        Parameters
        ----------
        data : np.ndarray
            Neural data with shape (channels, samples)
            
        Returns
        -------
        np.ndarray
            Extracted features
        """
        from scipy import signal
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
            
        n_channels, n_samples = data.shape
        features = []
        
        for ch in range(n_channels):
            channel_features = []
            
            for band_name, (low_freq, high_freq) in self.frequency_bands.items():
                # Bandpass filter for specific frequency band
                nyquist = self.sampling_rate / 2
                low_norm = low_freq / nyquist
                high_norm = high_freq / nyquist
                
                try:
                    b, a = signal.butter(4, [low_norm, high_norm], btype='band')
                    filtered_signal = signal.filtfilt(b, a, data[ch])
                    
                    # Power in frequency band
                    power = np.mean(filtered_signal**2)
                    channel_features.append(power)
                    
                    # Spectral entropy
                    freqs, psd = signal.welch(filtered_signal, self.sampling_rate)
                    psd_norm = psd / np.sum(psd)
                    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
                    channel_features.append(spectral_entropy)
                    
                except Exception:
                    # Fallback values if filtering fails
                    channel_features.extend([0.0, 0.0])
            
            features.extend(channel_features)
        
        return np.array(features)
    
    def train(self, training_data: np.ndarray, labels: np.ndarray) -> None:
        """
        Train the motor imagery decoder.
        
        Parameters
        ----------
        training_data : np.ndarray
            Training neural data with shape (trials, channels, samples)
        labels : np.ndarray
            Labels for each trial
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError("scikit-learn is required for training")
        
        # Extract features from training data
        n_trials = training_data.shape[0]
        feature_vectors = []
        
        for trial in range(n_trials):
            features = self.extract_motor_features(training_data[trial])
            feature_vectors.append(features)
        
        X = np.array(feature_vectors)
        y = labels
        
        # Normalize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train classifier
        self.classifier = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            max_depth=10
        )
        self.classifier.fit(X_scaled, y)
        
        self.is_trained = True
        print(f"Motor imagery decoder trained with {n_trials} trials")
    
    def decode(self, neural_data: np.ndarray) -> Dict[str, Any]:
        """
        Decode neural signals into movement intentions.
        
        Parameters
        ----------
        neural_data : np.ndarray
            Neural data to decode
            
        Returns
        -------
        dict
            Decoded movement intention and confidence
        """
        if not self.is_trained:
            raise ValueError("Decoder must be trained before use")
        
        # Extract features
        features = self.extract_motor_features(neural_data)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get prediction and confidence
        prediction = self.classifier.predict(features_scaled)[0]
        prediction_proba = self.classifier.predict_proba(features_scaled)[0]
        confidence = np.max(prediction_proba)
        
        return {
            'intent': prediction,
            'confidence': confidence,
            'probabilities': dict(zip(self.classifier.classes_, prediction_proba)),
            'features': features
        }
    
    def predict_intent(self, features: np.ndarray) -> str:
        """Predict movement intent from features."""
        if not self.is_trained:
            raise ValueError("Decoder must be trained before use")
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        return self.classifier.predict(features_scaled)[0]


class RealTimeDecoder:
    """
    Real-time neural decoding system with streaming capabilities.
    
    This class manages real-time processing of neural signals,
    including buffering, preprocessing, and continuous decoding.
    """
    
    def __init__(
        self,
        decoder: NeuralDecoder,
        buffer_size: int = 3000,  # 100ms at 30kHz
        update_rate: float = 10.0,  # Hz
        preprocessing_pipeline: Optional[Callable] = None
    ):
        """
        Initialize real-time decoder.
        
        Parameters
        ----------
        decoder : NeuralDecoder
            Trained neural decoder
        buffer_size : int, default=3000
            Size of the processing buffer
        update_rate : float, default=10.0
            Decoding update rate in Hz
        preprocessing_pipeline : callable, optional
            Preprocessing function to apply to incoming data
        """
        self.decoder = decoder
        self.buffer_size = buffer_size
        self.update_rate = update_rate
        self.update_interval = 1.0 / update_rate
        self.preprocessing_pipeline = preprocessing_pipeline
        
        # Threading and buffering
        self.data_queue = Queue()
        self.result_queue = Queue()
        self.is_running = False
        self.processing_thread = None
        
        # Callbacks for real-time events
        self.on_decode_callback: Optional[Callable] = None
        self.on_error_callback: Optional[Callable] = None
        
    def set_decode_callback(self, callback: Callable[[Dict], None]) -> None:
        """Set callback function for decode results."""
        self.on_decode_callback = callback
    
    def set_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Set callback function for errors."""
        self.on_error_callback = callback
    
    def add_data(self, neural_data: np.ndarray) -> None:
        """
        Add new neural data to the processing queue.
        
        Parameters
        ----------
        neural_data : np.ndarray
            New neural data samples
        """
        if self.is_running:
            self.data_queue.put(neural_data)
    
    def _processing_loop(self) -> None:
        """Main processing loop for real-time decoding."""
        buffer = []
        last_update = time.time()
        
        while self.is_running:
            try:
                # Check for new data
                while not self.data_queue.empty():
                    new_data = self.data_queue.get_nowait()
                    buffer.append(new_data)
                
                # Check if it's time to process
                current_time = time.time()
                if current_time - last_update >= self.update_interval:
                    if buffer:
                        # Concatenate buffer data
                        if len(buffer) == 1:
                            combined_data = buffer[0]
                        else:
                            combined_data = np.concatenate(buffer, axis=-1)
                        
                        # Take last buffer_size samples
                        if combined_data.shape[-1] >= self.buffer_size:
                            processing_data = combined_data[..., -self.buffer_size:]
                            
                            # Apply preprocessing if available
                            if self.preprocessing_pipeline:
                                processing_data = self.preprocessing_pipeline(processing_data)
                            
                            # Decode
                            result = self.decoder.decode(processing_data)
                            result['timestamp'] = current_time
                            
                            # Send result
                            self.result_queue.put(result)
                            
                            # Trigger callback
                            if self.on_decode_callback:
                                self.on_decode_callback(result)
                        
                        # Keep only recent data
                        buffer = [combined_data[..., -self.buffer_size//2:]]
                        last_update = current_time
                
                # Small sleep to prevent CPU overload
                time.sleep(0.001)  # 1ms
                
            except Exception as e:
                if self.on_error_callback:
                    self.on_error_callback(e)
                else:
                    print(f"Error in processing loop: {e}")
    
    def start(self) -> None:
        """Start real-time decoding."""
        if not self.decoder.is_trained:
            raise ValueError("Decoder must be trained before starting")
        
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
        print("Real-time decoding started")
    
    def stop(self) -> None:
        """Stop real-time decoding."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
        
        print("Real-time decoding stopped")
    
    def get_latest_result(self) -> Optional[Dict]:
        """Get the most recent decoding result."""
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None


class DeviceController:
    """
    Interface for controlling external devices based on neural commands.
    
    This class provides a framework for translating decoded neural
    intentions into device control signals.
    """
    
    def __init__(self):
        """Initialize device controller."""
        self.connected_devices = {}
        self.command_history = []
        
    def register_device(self, device_name: str, control_function: Callable) -> None:
        """
        Register a controllable device.
        
        Parameters
        ----------
        device_name : str
            Name of the device
        control_function : callable
            Function to call for device control
        """
        self.connected_devices[device_name] = control_function
    
    def execute_command(self, device_name: str, command: str, **kwargs) -> bool:
        """
        Execute a command on a registered device.
        
        Parameters
        ----------
        device_name : str
            Name of the target device
        command : str
            Command to execute
        **kwargs
            Additional command parameters
            
        Returns
        -------
        bool
            Success status
        """
        if device_name not in self.connected_devices:
            print(f"Device '{device_name}' not registered")
            return False
        
        try:
            control_function = self.connected_devices[device_name]
            result = control_function(command, **kwargs)
            
            # Log command
            self.command_history.append({
                'timestamp': time.time(),
                'device': device_name,
                'command': command,
                'kwargs': kwargs,
                'success': True
            })
            
            return True
        except Exception as e:
            print(f"Error executing command on {device_name}: {e}")
            self.command_history.append({
                'timestamp': time.time(),
                'device': device_name,
                'command': command,
                'kwargs': kwargs,
                'success': False,
                'error': str(e)
            })
            return False


def create_motor_imagery_system(
    sampling_rate: float = 30000.0,
    buffer_size: int = 3000,
    update_rate: float = 10.0
) -> tuple:
    """
    Factory function to create a complete motor imagery BCI system.
    
    Parameters
    ----------
    sampling_rate : float, default=30000.0
        Sampling rate in Hz
    buffer_size : int, default=3000
        Processing buffer size
    update_rate : float, default=10.0
        Decoding update rate in Hz
        
    Returns
    -------
    tuple
        (decoder, real_time_system, device_controller)
    """
    # Create components
    decoder = MotorImageryDecoder(sampling_rate=sampling_rate)
    real_time_system = RealTimeDecoder(
        decoder=decoder,
        buffer_size=buffer_size,
        update_rate=update_rate
    )
    device_controller = DeviceController()
    
    return decoder, real_time_system, device_controller
