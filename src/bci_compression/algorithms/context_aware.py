"""
Context-aware compression methods for neural signals.

This module implements sophisticated context modeling that adapts to neural
signal characteristics including brain states, spatial relationships, and
temporal patterns.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import logging


@dataclass
class ContextMetadata:
    """Metadata for context-aware compression."""
    context_type: str
    brain_states: List[str]
    spatial_groups: Dict
    temporal_contexts: Dict
    compression_ratio: float
    adaptation_time: float
    context_switches: int


class BrainStateDetector:
    """
    Detect different brain states from neural signals.
    
    Classifies neural activity into states like rest, active, motor preparation,
    and adapts compression accordingly.
    """
    
    def __init__(self, sampling_rate: float = 30000.0):
        """
        Initialize brain state detector.
        
        Args:
            sampling_rate: Signal sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.states = ['rest', 'active', 'motor', 'cognitive']
        self.state_models = {}
        self.current_state = 'rest'
        self.state_history = deque(maxlen=100)
        
        # Feature extraction parameters
        self.window_size = int(0.1 * sampling_rate)  # 100ms windows
        self.overlap = 0.5
        
    def extract_features(self, data: np.ndarray) -> Dict:
        """
        Extract features for brain state classification.
        
        Args:
            data: Multi-channel neural data
            
        Returns:
            Feature dictionary
        """
        features = {}
        
        # Power spectral features
        freqs = np.fft.fftfreq(data.shape[1], 1/self.sampling_rate)
        
        for ch_idx in range(data.shape[0]):
            fft_data = np.fft.fft(data[ch_idx])
            power_spectrum = np.abs(fft_data)**2
            
            # Band power features
            alpha_band = np.logical_and(freqs >= 8, freqs <= 12)
            beta_band = np.logical_and(freqs >= 13, freqs <= 30)
            gamma_band = np.logical_and(freqs >= 30, freqs <= 100)
            
            features[f'ch{ch_idx}_alpha_power'] = np.mean(power_spectrum[alpha_band])
            features[f'ch{ch_idx}_beta_power'] = np.mean(power_spectrum[beta_band])
            features[f'ch{ch_idx}_gamma_power'] = np.mean(power_spectrum[gamma_band])
            
            # Statistical features
            features[f'ch{ch_idx}_variance'] = np.var(data[ch_idx])
            features[f'ch{ch_idx}_kurtosis'] = self._kurtosis(data[ch_idx])
            features[f'ch{ch_idx}_zero_crossings'] = self._zero_crossings(data[ch_idx])
        
        # Cross-channel features
        features['coherence'] = self._compute_coherence(data)
        features['spatial_complexity'] = self._spatial_complexity(data)
        
        return features
    
    def _kurtosis(self, signal: np.ndarray) -> float:
        """Compute kurtosis of signal."""
        if np.std(signal) == 0:
            return 0.0
        normalized = (signal - np.mean(signal)) / np.std(signal)
        return np.mean(normalized**4) - 3.0
    
    def _zero_crossings(self, signal: np.ndarray) -> int:
        """Count zero crossings in signal."""
        return np.sum(np.diff(np.signbit(signal - np.mean(signal))))
    
    def _compute_coherence(self, data: np.ndarray) -> float:
        """Compute average coherence between channels."""
        n_channels = data.shape[0]
        coherences = []
        
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                corr = np.corrcoef(data[i], data[j])[0, 1]
                coherences.append(abs(corr))
        
        return np.mean(coherences) if coherences else 0.0
    
    def _spatial_complexity(self, data: np.ndarray) -> float:
        """Compute spatial complexity measure."""
        # Principal component analysis for spatial complexity
        try:
            cov_matrix = np.cov(data)
            eigenvals = np.linalg.eigvals(cov_matrix)
            eigenvals = eigenvals[eigenvals > 0]  # Remove zero eigenvalues
            
            if len(eigenvals) > 1:
                # Normalized entropy of eigenvalues
                eigenvals_norm = eigenvals / np.sum(eigenvals)
                entropy = -np.sum(eigenvals_norm * np.log(eigenvals_norm + 1e-10))
                return entropy / np.log(len(eigenvals))
            else:
                return 0.0
        except:
            return 0.0
    
    def classify_state(self, data: np.ndarray) -> str:
        """
        Classify current brain state.
        
        Args:
            data: Current data window
            
        Returns:
            Detected brain state
        """
        features = self.extract_features(data)
        
        # Simple rule-based classifier (could be replaced with ML model)
        avg_gamma = np.mean([v for k, v in features.items() if 'gamma_power' in k])
        avg_beta = np.mean([v for k, v in features.items() if 'beta_power' in k])
        coherence = features['coherence']
        spatial_complexity = features['spatial_complexity']
        
        # State classification rules
        if avg_gamma > 0.5 and coherence > 0.3:
            state = 'motor'
        elif avg_beta > 0.3 and spatial_complexity > 0.5:
            state = 'cognitive'
        elif coherence > 0.4:
            state = 'active'
        else:
            state = 'rest'
        
        self.current_state = state
        self.state_history.append(state)
        
        return state


class HierarchicalContextModel:
    """
    Multi-level context modeling for neural signals.
    
    Implements hierarchical context trees that capture patterns at different
    temporal scales and complexity levels.
    """
    
    def __init__(self, max_depth: int = 5, alphabet_size: int = 256):
        """
        Initialize hierarchical context model.
        
        Args:
            max_depth: Maximum context depth
            alphabet_size: Size of symbol alphabet
        """
        self.max_depth = max_depth
        self.alphabet_size = alphabet_size
        self.context_trees = {}
        self.symbol_counts = defaultdict(int)
        self.total_symbols = 0
        
        # Initialize context trees for different levels
        for level in range(max_depth + 1):
            self.context_trees[level] = defaultdict(lambda: defaultdict(int))
    
    def update_context(self, symbols: List[int], level: int = 0):
        """
        Update context model with new symbols.
        
        Args:
            symbols: List of symbols to add
            level: Context tree level to update
        """
        if level > self.max_depth:
            return
            
        for i in range(len(symbols)):
            # Update symbol counts
            symbol = symbols[i]
            self.symbol_counts[symbol] += 1
            self.total_symbols += 1
            
            # Update context trees
            for depth in range(min(level + 1, i + 1, self.max_depth + 1)):
                if depth == 0:
                    context = ()
                else:
                    context = tuple(symbols[max(0, i - depth):i])
                
                self.context_trees[depth][context][symbol] += 1
    
    def get_conditional_probability(self, symbol: int, context: Tuple) -> float:
        """
        Get conditional probability of symbol given context.
        
        Args:
            symbol: Symbol to predict
            context: Context tuple
            
        Returns:
            Conditional probability
        """
        context_depth = len(context)
        
        if context_depth > self.max_depth:
            context = context[-self.max_depth:]
            context_depth = self.max_depth
        
        # Try different context depths (backing off if needed)
        for depth in range(context_depth, -1, -1):
            current_context = context[-depth:] if depth > 0 else ()
            
            if current_context in self.context_trees[depth]:
                context_counts = self.context_trees[depth][current_context]
                total_count = sum(context_counts.values())
                
                if total_count > 0:
                    # Laplace smoothing
                    count = context_counts.get(symbol, 0)
                    return (count + 1) / (total_count + self.alphabet_size)
        
        # Fallback to uniform distribution
        return 1.0 / self.alphabet_size
    
    def get_model_statistics(self) -> Dict:
        """Get context model statistics."""
        stats = {
            'total_symbols': self.total_symbols,
            'unique_symbols': len(self.symbol_counts),
            'context_trees': len(self.context_trees),
            'average_context_size': 0.0,
            'memory_usage_estimate': 0
        }
        
        total_contexts = 0
        for level, tree in self.context_trees.items():
            total_contexts += len(tree)
            stats['memory_usage_estimate'] += len(tree) * self.alphabet_size * 4  # bytes
        
        if len(self.context_trees) > 0:
            stats['average_context_size'] = total_contexts / len(self.context_trees)
        
        return stats


class SpatialContextModel:
    """
    Model spatial relationships between electrode channels.
    
    Uses electrode positions and functional connectivity to create
    spatial context for compression.
    """
    
    def __init__(self, n_channels: int):
        """
        Initialize spatial context model.
        
        Args:
            n_channels: Number of electrode channels
        """
        self.n_channels = n_channels
        self.electrode_positions = {}
        self.connectivity_matrix = np.eye(n_channels)
        self.spatial_groups = {}
        self.neighborhood_map = {}
        
    def set_electrode_layout(self, positions: Dict[int, Tuple[float, float]]):
        """
        Set electrode positions for spatial modeling.
        
        Args:
            positions: Dictionary mapping channel ID to (x, y) position
        """
        self.electrode_positions = positions
        self._compute_spatial_neighborhoods()
    
    def _compute_spatial_neighborhoods(self):
        """Compute spatial neighborhoods based on electrode positions."""
        if not self.electrode_positions:
            return
            
        for ch_id, pos in self.electrode_positions.items():
            neighbors = []
            distances = []
            
            for other_id, other_pos in self.electrode_positions.items():
                if ch_id != other_id:
                    dist = np.sqrt((pos[0] - other_pos[0])**2 + 
                                 (pos[1] - other_pos[1])**2)
                    neighbors.append(other_id)
                    distances.append(dist)
            
            # Sort by distance and take closest neighbors
            sorted_neighbors = [n for _, n in sorted(zip(distances, neighbors))]
            self.neighborhood_map[ch_id] = sorted_neighbors[:4]  # 4 nearest neighbors
    
    def compute_functional_connectivity(self, data: np.ndarray, 
                                      method: str = 'correlation'):
        """
        Compute functional connectivity between channels.
        
        Args:
            data: Multi-channel neural data
            method: Connectivity method ('correlation', 'coherence')
        """
        if method == 'correlation':
            self.connectivity_matrix = np.corrcoef(data)
        elif method == 'coherence':
            # Simplified coherence calculation
            n_channels = data.shape[0]
            coherence_matrix = np.zeros((n_channels, n_channels))
            
            for i in range(n_channels):
                for j in range(n_channels):
                    if i == j:
                        coherence_matrix[i, j] = 1.0
                    else:
                        # Cross-correlation in frequency domain
                        fft_i = np.fft.fft(data[i])
                        fft_j = np.fft.fft(data[j])
                        cross_power = fft_i * np.conj(fft_j)
                        coherence = np.abs(np.mean(cross_power)) / (
                            np.sqrt(np.mean(np.abs(fft_i)**2)) * 
                            np.sqrt(np.mean(np.abs(fft_j)**2))
                        )
                        coherence_matrix[i, j] = coherence
            
            self.connectivity_matrix = coherence_matrix
    
    def create_spatial_groups(self, threshold: float = 0.3) -> Dict:
        """
        Create spatial groups based on connectivity.
        
        Args:
            threshold: Connectivity threshold for grouping
            
        Returns:
            Dictionary of spatial groups
        """
        # Find strongly connected channel groups
        strong_connections = self.connectivity_matrix > threshold
        
        # Simple clustering based on connectivity
        visited = set()
        groups = {}
        group_id = 0
        
        for ch_id in range(self.n_channels):
            if ch_id not in visited:
                group = [ch_id]
                queue = [ch_id]
                visited.add(ch_id)
                
                while queue:
                    current = queue.pop(0)
                    for neighbor in range(self.n_channels):
                        if (neighbor not in visited and 
                            strong_connections[current, neighbor]):
                            group.append(neighbor)
                            queue.append(neighbor)
                            visited.add(neighbor)
                
                groups[f'group_{group_id}'] = group
                group_id += 1
        
        self.spatial_groups = groups
        return groups


class ContextAwareCompressor:
    """
    Main context-aware compression system.
    
    Combines brain state detection, hierarchical context modeling, and
    spatial relationships for adaptive neural signal compression.
    """
    
    def __init__(self, sampling_rate: float = 30000.0):
        """
        Initialize context-aware compressor.
        
        Args:
            sampling_rate: Signal sampling rate
        """
        self.sampling_rate = sampling_rate
        self.brain_state_detector = BrainStateDetector(sampling_rate)
        self.hierarchical_model = HierarchicalContextModel()
        self.spatial_model = None
        
        # State-specific compression parameters
        self.state_parameters = {
            'rest': {'quantization_bits': 10, 'context_depth': 3},
            'active': {'quantization_bits': 12, 'context_depth': 4},
            'motor': {'quantization_bits': 14, 'context_depth': 5},
            'cognitive': {'quantization_bits': 12, 'context_depth': 4}
        }
        
        # Performance tracking
        self.context_switches = 0
        self.adaptation_times = []
        
    def setup_spatial_model(self, n_channels: int, 
                           electrode_positions: Optional[Dict] = None):
        """
        Set up spatial context model.
        
        Args:
            n_channels: Number of channels
            electrode_positions: Optional electrode positions
        """
        self.spatial_model = SpatialContextModel(n_channels)
        if electrode_positions:
            self.spatial_model.set_electrode_layout(electrode_positions)
    
    def compress(self, data: np.ndarray, 
                window_size: Optional[int] = None) -> Tuple[List[bytes], ContextMetadata]:
        """
        Compress data using context-aware methods.
        
        Args:
            data: Multi-channel neural data
            window_size: Processing window size
            
        Returns:
            Compressed data and metadata
        """
        start_time = time.time()
        n_channels, n_samples = data.shape
        
        if window_size is None:
            window_size = min(1000, n_samples // 4)
        
        # Set up spatial model if needed
        if self.spatial_model is None:
            self.setup_spatial_model(n_channels)
        
        # Analyze spatial connectivity
        self.spatial_model.compute_functional_connectivity(data)
        spatial_groups = self.spatial_model.create_spatial_groups()
        
        compressed_data = []
        brain_states_detected = []
        
        # Process data in windows
        for start_idx in range(0, n_samples, window_size):
            end_idx = min(start_idx + window_size, n_samples)
            window_data = data[:, start_idx:end_idx]
            
            # Detect brain state
            current_state = self.brain_state_detector.classify_state(window_data)
            brain_states_detected.append(current_state)
            
            if (len(brain_states_detected) > 1 and 
                current_state != brain_states_detected[-2]):
                self.context_switches += 1
            
            # Get state-specific parameters
            params = self.state_parameters.get(current_state, 
                                             self.state_parameters['rest'])
            
            # Quantize data based on current state
            quantized_window = self._quantize_adaptive(window_data, 
                                                     params['quantization_bits'])
            
            # Update hierarchical context model
            flat_data = quantized_window.flatten().astype(int)
            self.hierarchical_model.update_context(flat_data.tolist(), 
                                                 params['context_depth'])
            
            # Simple encoding (placeholder for full implementation)
            encoded_window = self._encode_window(quantized_window, current_state)
            compressed_data.append(encoded_window)
        
        compression_time = time.time() - start_time
        self.adaptation_times.append(compression_time)
        
        # Calculate compression ratio
        original_bits = data.size * 16  # Assuming 16-bit input
        compressed_bits = sum(len(chunk) * 8 for chunk in compressed_data)
        compression_ratio = original_bits / compressed_bits if compressed_bits > 0 else 1.0
        
        metadata = ContextMetadata(
            context_type="hierarchical_spatial_state_aware",
            brain_states=list(set(brain_states_detected)),
            spatial_groups=spatial_groups,
            temporal_contexts=self.hierarchical_model.get_model_statistics(),
            compression_ratio=compression_ratio,
            adaptation_time=compression_time,
            context_switches=self.context_switches
        )
        
        return compressed_data, metadata
    
    def _quantize_adaptive(self, data: np.ndarray, bits: int) -> np.ndarray:
        """Adaptive quantization based on signal characteristics."""
        # Per-channel adaptive quantization
        quantized = np.zeros_like(data, dtype=np.int16)
        
        for ch in range(data.shape[0]):
            signal = data[ch]
            signal_range = np.ptp(signal)  # Peak-to-peak range
            
            if signal_range > 0:
                # Adaptive scaling based on signal dynamics
                levels = 2**bits - 1
                scale_factor = levels / signal_range
                offset = np.min(signal)
                
                quantized[ch] = np.round((signal - offset) * scale_factor).astype(np.int16)
            
        return quantized
    
    def _encode_window(self, data: np.ndarray, state: str) -> bytes:
        """Encode quantized window data."""
        # Simple encoding - could be enhanced with entropy coding
        encoded = bytearray()
        
        # Add state information
        state_map = {'rest': 0, 'active': 1, 'motor': 2, 'cognitive': 3}
        encoded.append(state_map.get(state, 0))
        
        # Flatten and encode data
        flat_data = data.flatten().astype(np.int16)
        encoded.extend(flat_data.tobytes())
        
        return bytes(encoded)
    
    def get_compression_statistics(self) -> Dict:
        """Get comprehensive compression statistics."""
        stats = {
            'brain_states_detected': len(set(self.brain_state_detector.state_history)),
            'context_switches': self.context_switches,
            'average_adaptation_time': np.mean(self.adaptation_times) if self.adaptation_times else 0.0,
            'hierarchical_model_stats': self.hierarchical_model.get_model_statistics()
        }
        
        if self.spatial_model:
            stats['spatial_groups'] = len(self.spatial_model.spatial_groups)
            stats['average_connectivity'] = np.mean(self.spatial_model.connectivity_matrix)
        
        return stats


def create_context_aware_compressor(mode: str = "adaptive") -> ContextAwareCompressor:
    """
    Factory function for context-aware compressor.
    
    Args:
        mode: Compression mode ("adaptive", "spatial", "temporal")
        
    Returns:
        Configured context-aware compressor
    """
    if mode == "spatial":
        compressor = ContextAwareCompressor()
        # Enhanced spatial processing
        return compressor
    elif mode == "temporal":
        compressor = ContextAwareCompressor()
        # Enhanced temporal context
        compressor.hierarchical_model.max_depth = 8
        return compressor
    else:  # adaptive
        return ContextAwareCompressor()


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Context-Aware Compression...")
    
    # Generate test data
    n_channels, n_samples = 8, 2000
    test_data = np.random.randn(n_channels, n_samples) * 100
    
    # Add some structure to simulate neural patterns
    for ch in range(n_channels):
        # Add periodic components
        t = np.linspace(0, 1, n_samples)
        test_data[ch] += 50 * np.sin(2 * np.pi * 10 * t)  # 10 Hz
        test_data[ch] += 20 * np.sin(2 * np.pi * 50 * t)  # 50 Hz
    
    # Test compressor
    compressor = create_context_aware_compressor("adaptive")
    
    compressed, metadata = compressor.compress(test_data)
    
    print(f"Compression ratio: {metadata.compression_ratio:.2f}x")
    print(f"Brain states detected: {metadata.brain_states}")
    print(f"Context switches: {metadata.context_switches}")
    print(f"Spatial groups: {len(metadata.spatial_groups)}")
    
    stats = compressor.get_compression_statistics()
    print(f"Statistics: {stats}")
