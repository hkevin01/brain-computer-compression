"""
Neural-optimized LZ compression variants for brain-computer interface data.

This module implements LZ77/LZ78 variants specifically optimized for the
characteristics of neural data, including temporal correlations and
multi-channel redundancy.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Union
import struct
from collections import defaultdict
import time


class NeuralLZ77Compressor:
    """
    LZ77 variant optimized for neural signal compression.
    
    This compressor takes advantage of temporal correlations in neural
    signals and optimizes the sliding window and lookahead buffer sizes
    for typical neural data characteristics.
    """
    
    def __init__(
        self,
        window_size: int = 4096,  # Optimized for neural sampling rates
        lookahead_size: int = 256,
        min_match_length: int = 3,
        quantization_bits: int = 16
    ):
        """
        Initialize Neural LZ77 compressor.
        
        Parameters
        ----------
        window_size : int, default=4096
            Size of the sliding window (optimized for ~100ms at 30kHz)
        lookahead_size : int, default=256
            Size of lookahead buffer
        min_match_length : int, default=3
            Minimum length for pattern matching
        quantization_bits : int, default=16
            Bit depth for neural signal quantization
        """
        self.window_size = window_size
        self.lookahead_size = lookahead_size
        self.min_match_length = min_match_length
        self.quantization_bits = quantization_bits
        self.quantization_levels = 2 ** quantization_bits
        
        # Statistics for optimization
        self.compression_stats = {
            'matches_found': 0,
            'total_symbols': 0,
            'compression_ratio': 0.0,
            'processing_time': 0.0
        }
    
    def _quantize_signal(self, data: np.ndarray) -> np.ndarray:
        """
        Quantize neural signal to specified bit depth.
        
        Parameters
        ----------
        data : np.ndarray
            Input neural data
            
        Returns
        -------
        np.ndarray
            Quantized data as integers
        """
        # Normalize to [0, 1] range
        data_min, data_max = np.min(data), np.max(data)
        if data_max > data_min:
            normalized = (data - data_min) / (data_max - data_min)
        else:
            normalized = np.zeros_like(data)
        
        # Quantize to integer levels
        quantized = np.round(normalized * (self.quantization_levels - 1))
        return quantized.astype(np.uint16), (data_min, data_max)
    
    def _dequantize_signal(
        self, 
        quantized: np.ndarray, 
        scale_params: Tuple[float, float]
    ) -> np.ndarray:
        """
        Dequantize signal back to original range.
        
        Parameters
        ----------
        quantized : np.ndarray
            Quantized integer data
        scale_params : tuple
            (min_val, max_val) for rescaling
            
        Returns
        -------
        np.ndarray
            Dequantized signal
        """
        data_min, data_max = scale_params
        
        # Convert back to [0, 1] range
        normalized = quantized.astype(np.float64) / (self.quantization_levels - 1)
        
        # Rescale to original range
        return normalized * (data_max - data_min) + data_min
    
    def _find_longest_match(
        self, 
        data: np.ndarray, 
        pos: int, 
        window_start: int
    ) -> Tuple[int, int]:
        """
        Find the longest match in the sliding window.
        
        Parameters
        ----------
        data : np.ndarray
            Input data array
        pos : int
            Current position in data
        window_start : int
            Start of sliding window
            
        Returns
        -------
        tuple
            (offset, length) of longest match
        """
        best_offset = 0
        best_length = 0
        
        # Lookahead buffer
        lookahead_end = min(pos + self.lookahead_size, len(data))
        
        # Search in sliding window
        for i in range(window_start, pos):
            match_length = 0
            
            # Count matching symbols
            while (pos + match_length < lookahead_end and
                   i + match_length < pos and
                   data[i + match_length] == data[pos + match_length]):
                match_length += 1
            
            # Update best match if longer and meets minimum length
            if match_length >= self.min_match_length and match_length > best_length:
                best_offset = pos - i
                best_length = match_length
        
        return best_offset, best_length
    
    def compress_channel(self, data: np.ndarray) -> Tuple[bytes, Dict]:
        """
        Compress a single channel of neural data.
        
        Parameters
        ----------
        data : np.ndarray
            Single channel neural data
            
        Returns
        -------
        tuple
            (compressed_data, metadata)
        """
        start_time = time.time()
        
        # Quantize the signal
        quantized_data, scale_params = self._quantize_signal(data)
        
        # Initialize compression
        compressed = []
        pos = 0
        matches_found = 0
        
        while pos < len(quantized_data):
            # Define sliding window
            window_start = max(0, pos - self.window_size)
            
            # Find longest match
            offset, length = self._find_longest_match(
                quantized_data, pos, window_start
            )
            
            if length >= self.min_match_length:
                # Encode as (offset, length) pair
                compressed.append(('match', offset, length))
                pos += length
                matches_found += 1
            else:
                # Encode as literal symbol
                compressed.append(('literal', int(quantized_data[pos])))
                pos += 1
        
        # Convert to binary format
        binary_data = self._encode_tokens(compressed)
        
        # Update statistics
        processing_time = time.time() - start_time
        original_size = len(data) * 4  # Assuming 32-bit floats
        compressed_size = len(binary_data)
        
        self.compression_stats.update({
            'matches_found': matches_found,
            'total_symbols': len(data),
            'compression_ratio': original_size / compressed_size if compressed_size > 0 else 0,
            'processing_time': processing_time
        })
        
        metadata = {
            'scale_params': scale_params,
            'original_length': len(data),
            'quantization_bits': self.quantization_bits,
            'window_size': self.window_size,
            'lookahead_size': self.lookahead_size,
            'compression_stats': self.compression_stats.copy()
        }
        
        return binary_data, metadata
    
    def _encode_tokens(self, tokens: List) -> bytes:
        """
        Encode tokens to binary format.
        
        Parameters
        ----------
        tokens : list
            List of tokens (literals or matches)
            
        Returns
        -------
        bytes
            Binary encoded data
        """
        binary_data = bytearray()
        
        for token in tokens:
            if token[0] == 'literal':
                # Flag (1 bit) + literal value
                flag_and_value = 0x80000000 | token[1]  # Set MSB for literal
                binary_data.extend(struct.pack('>I', flag_and_value))
            else:  # match
                # Flag (1 bit) + offset (15 bits) + length (16 bits)
                flag_offset_length = (token[1] << 16) | token[2]  # Clear MSB for match
                binary_data.extend(struct.pack('>I', flag_offset_length))
        
        return bytes(binary_data)
    
    def decompress_channel(self, compressed_data: bytes, metadata: Dict) -> np.ndarray:
        """
        Decompress a single channel of neural data.
        
        Parameters
        ----------
        compressed_data : bytes
            Compressed binary data
        metadata : dict
            Compression metadata
            
        Returns
        -------
        np.ndarray
            Decompressed neural data
        """
        # Decode tokens
        tokens = self._decode_tokens(compressed_data)
        
        # Reconstruct quantized data
        reconstructed = []
        
        for token in tokens:
            if token[0] == 'literal':
                reconstructed.append(token[1])
            else:  # match
                offset, length = token[1], token[2]
                start_pos = len(reconstructed) - offset
                
                for i in range(length):
                    reconstructed.append(reconstructed[start_pos + i])
        
        # Convert to numpy array and dequantize
        quantized_array = np.array(reconstructed, dtype=np.uint16)
        
        # Pad or truncate to original length
        original_length = metadata['original_length']
        if len(quantized_array) > original_length:
            quantized_array = quantized_array[:original_length]
        elif len(quantized_array) < original_length:
            # This shouldn't happen in correct implementation
            padding = np.zeros(original_length - len(quantized_array), dtype=np.uint16)
            quantized_array = np.concatenate([quantized_array, padding])
        
        # Dequantize
        decompressed = self._dequantize_signal(
            quantized_array, 
            metadata['scale_params']
        )
        
        return decompressed
    
    def _decode_tokens(self, binary_data: bytes) -> List:
        """
        Decode binary data to tokens.
        
        Parameters
        ----------
        binary_data : bytes
            Binary encoded data
            
        Returns
        -------
        list
            List of decoded tokens
        """
        tokens = []
        pos = 0
        
        while pos < len(binary_data):
            if pos + 4 > len(binary_data):
                break
                
            # Unpack 32-bit value
            value = struct.unpack('>I', binary_data[pos:pos+4])[0]
            pos += 4
            
            if value & 0x80000000:  # MSB set -> literal
                literal_value = value & 0x7FFFFFFF
                tokens.append(('literal', literal_value))
            else:  # MSB clear -> match
                offset = (value >> 16) & 0x7FFF
                length = value & 0xFFFF
                tokens.append(('match', offset, length))
        
        return tokens


class MultiChannelNeuralLZ:
    """
    Multi-channel neural LZ compressor that exploits spatial correlations.
    
    This compressor processes multiple neural channels simultaneously,
    taking advantage of correlations between nearby electrodes.
    """
    
    def __init__(
        self,
        single_channel_compressor: Optional[NeuralLZ77Compressor] = None,
        use_channel_prediction: bool = True,
        prediction_order: int = 2
    ):
        """
        Initialize multi-channel compressor.
        
        Parameters
        ----------
        single_channel_compressor : NeuralLZ77Compressor, optional
            Single channel compressor to use
        use_channel_prediction : bool, default=True
            Whether to use inter-channel prediction
        prediction_order : int, default=2
            Order of linear prediction between channels
        """
        if single_channel_compressor is None:
            self.channel_compressor = NeuralLZ77Compressor()
        else:
            self.channel_compressor = single_channel_compressor
            
        self.use_channel_prediction = use_channel_prediction
        self.prediction_order = prediction_order
    
    def _predict_channel(
        self, 
        target_channel: np.ndarray, 
        reference_channels: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict target channel from reference channels.
        
        Parameters
        ----------
        target_channel : np.ndarray
            Channel to predict
        reference_channels : list
            List of reference channel data
            
        Returns
        -------
        tuple
            (prediction, residual)
        """
        if not reference_channels or not self.use_channel_prediction:
            return np.zeros_like(target_channel), target_channel
        
        # Simple linear prediction using reference channels
        n_samples = len(target_channel)
        n_refs = min(len(reference_channels), self.prediction_order)
        
        if n_refs == 0:
            return np.zeros_like(target_channel), target_channel
        
        # Stack reference channels
        ref_matrix = np.column_stack(reference_channels[:n_refs])
        
        # Solve least squares for prediction coefficients
        try:
            # Add small regularization to avoid singular matrices
            A = ref_matrix.T @ ref_matrix + 1e-6 * np.eye(n_refs)
            b = ref_matrix.T @ target_channel
            coeffs = np.linalg.solve(A, b)
            
            # Generate prediction
            prediction = ref_matrix @ coeffs
            residual = target_channel - prediction
            
            return prediction, residual
        except np.linalg.LinAlgError:
            # Fallback if matrix is singular
            return np.zeros_like(target_channel), target_channel
    
    def compress(self, data: np.ndarray) -> Tuple[List[bytes], Dict]:
        """
        Compress multi-channel neural data.
        
        Parameters
        ----------
        data : np.ndarray
            Multi-channel data with shape (channels, samples)
            
        Returns
        -------
        tuple
            (list_of_compressed_channels, metadata)
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        n_channels, n_samples = data.shape
        compressed_channels = []
        channel_metadata = []
        total_start_time = time.time()
        
        for ch in range(n_channels):
            current_channel = data[ch]
            
            if self.use_channel_prediction and ch > 0:
                # Use previous channels for prediction
                reference_channels = [data[i] for i in range(ch)]
                prediction, residual = self._predict_channel(
                    current_channel, reference_channels
                )
                
                # Compress the residual instead of original signal
                compressed_data, metadata = self.channel_compressor.compress_channel(residual)
                metadata['has_prediction'] = True
                metadata['prediction_coeffs'] = prediction  # Store prediction for decompression
            else:
                # Compress original signal
                compressed_data, metadata = self.channel_compressor.compress_channel(current_channel)
                metadata['has_prediction'] = False
            
            compressed_channels.append(compressed_data)
            channel_metadata.append(metadata)
        
        total_processing_time = time.time() - total_start_time
        
        # Calculate overall statistics
        original_size = data.size * 4  # 32-bit floats
        compressed_size = sum(len(ch_data) for ch_data in compressed_channels)
        
        global_metadata = {
            'n_channels': n_channels,
            'n_samples': n_samples,
            'channel_metadata': channel_metadata,
            'overall_compression_ratio': original_size / compressed_size if compressed_size > 0 else 0,
            'total_processing_time': total_processing_time,
            'use_channel_prediction': self.use_channel_prediction
        }
        
        return compressed_channels, global_metadata
    
    def decompress(
        self, 
        compressed_channels: List[bytes], 
        metadata: Dict
    ) -> np.ndarray:
        """
        Decompress multi-channel neural data.
        
        Parameters
        ----------
        compressed_channels : list
            List of compressed channel data
        metadata : dict
            Compression metadata
            
        Returns
        -------
        np.ndarray
            Decompressed multi-channel data
        """
        n_channels = metadata['n_channels']
        n_samples = metadata['n_samples']
        
        decompressed_data = np.zeros((n_channels, n_samples))
        
        for ch in range(n_channels):
            ch_metadata = metadata['channel_metadata'][ch]
            
            # Decompress channel
            if ch_metadata['has_prediction']:
                # Decompress residual
                residual = self.channel_compressor.decompress_channel(
                    compressed_channels[ch], ch_metadata
                )
                
                # Reconstruct original signal
                if ch > 0:
                    reference_channels = [decompressed_data[i] for i in range(ch)]
                    prediction, _ = self._predict_channel(
                        residual, reference_channels  # Use same prediction method
                    )
                    decompressed_data[ch] = residual + prediction
                else:
                    decompressed_data[ch] = residual
            else:
                # Direct decompression
                decompressed_data[ch] = self.channel_compressor.decompress_channel(
                    compressed_channels[ch], ch_metadata
                )
        
        return decompressed_data


def create_neural_lz_compressor(
    optimization_preset: str = 'balanced'
) -> MultiChannelNeuralLZ:
    """
    Factory function to create optimized neural LZ compressor.
    
    Parameters
    ----------
    optimization_preset : str, default='balanced'
        Optimization preset ('speed', 'balanced', 'compression')
        
    Returns
    -------
    MultiChannelNeuralLZ
        Configured compressor
    """
    if optimization_preset == 'speed':
        # Fast compression with larger quantization
        base_compressor = NeuralLZ77Compressor(
            window_size=1024,
            lookahead_size=64,
            min_match_length=2,
            quantization_bits=12
        )
    elif optimization_preset == 'compression':
        # Maximum compression with fine quantization
        base_compressor = NeuralLZ77Compressor(
            window_size=8192,
            lookahead_size=512,
            min_match_length=4,
            quantization_bits=16
        )
    else:  # balanced
        # Default balanced settings
        base_compressor = NeuralLZ77Compressor()
    
    return MultiChannelNeuralLZ(
        single_channel_compressor=base_compressor,
        use_channel_prediction=True,
        prediction_order=2
    )
