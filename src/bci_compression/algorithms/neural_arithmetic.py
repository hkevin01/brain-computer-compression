"""
Arithmetic coding with neural data models for BCI compression.

This module implements context-aware arithmetic coding specifically
designed for neural signal characteristics, including adaptive
probability models and multi-scale temporal contexts.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque


class NeuralArithmeticModel:
    """
    Adaptive probability model for neural data arithmetic coding.

    This model maintains context-dependent probability distributions
    that adapt to the statistical characteristics of neural signals.
    """

    def __init__(
        self,
        alphabet_size: int = 65536,  # 16-bit quantization
        context_length: int = 4,
        adaptation_rate: float = 0.1,
        min_frequency: int = 1
    ):
        """
        Initialize neural arithmetic model.

        Parameters
        ----------
        alphabet_size : int, default=65536
            Size of symbol alphabet (2^quantization_bits)
        context_length : int, default=4
            Length of context for probability estimation
        adaptation_rate : float, default=0.1
            Rate of adaptation to new symbols
        min_frequency : int, default=1
            Minimum frequency count for symbols
        """
        self.alphabet_size = alphabet_size
        self.context_length = context_length
        self.adaptation_rate = adaptation_rate
        self.min_frequency = min_frequency

        # Context-dependent frequency tables
        self.context_frequencies = defaultdict(
            lambda: np.ones(alphabet_size, dtype=np.int32) * min_frequency
        )

        # Cumulative frequency tables for encoding
        self.context_cumulative = defaultdict(
            lambda: np.cumsum(
                np.ones(alphabet_size + 1, dtype=np.int32) * min_frequency
            )
        )

        # Context history
        self.context_history = deque(maxlen=context_length)

        # Statistics
        self.total_symbols = 0
        self.context_hits = 0

    def _get_context_key(self) -> tuple:
        """Get current context as a hashable key."""
        return tuple(self.context_history)

    def get_symbol_probability(self, symbol: int) -> Tuple[int, int, int]:
        """
        Get probability information for a symbol.

        Parameters
        ----------
        symbol : int
            Symbol to get probability for

        Returns
        -------
        tuple
            (low, high, total) cumulative frequencies
        """
        context_key = self._get_context_key()

        if context_key in self.context_cumulative:
            cumulative = self.context_cumulative[context_key]
            self.context_hits += 1
        else:
            # Use uniform distribution for unseen contexts
            frequencies = np.ones(self.alphabet_size, dtype=np.int32) * self.min_frequency
            cumulative = np.cumsum(np.concatenate([[0], frequencies]))
            self.context_cumulative[context_key] = cumulative

        low = cumulative[symbol]
        high = cumulative[symbol + 1]
        total = cumulative[-1]

        return int(low), int(high), int(total)

    def update_model(self, symbol: int) -> None:
        """
        Update model with observed symbol.

        Parameters
        ----------
        symbol : int
            Observed symbol
        """
        context_key = self._get_context_key()

        # Update frequency
        self.context_frequencies[context_key][symbol] += 1

        # Update cumulative frequencies
        frequencies = self.context_frequencies[context_key]
        self.context_cumulative[context_key] = np.cumsum(
            np.concatenate([[0], frequencies])
        )

        # Add symbol to context history
        self.context_history.append(symbol)
        self.total_symbols += 1

    def get_model_statistics(self) -> Dict:
        """Get model statistics."""
        return {
            'total_symbols': self.total_symbols,
            'context_hits': self.context_hits,
            'context_hit_rate': self.context_hits / max(1, self.total_symbols),
            'unique_contexts': len(self.context_frequencies),
            'alphabet_size': self.alphabet_size
        }


class NeuralArithmeticCoder:
    """
    Arithmetic coder optimized for neural signals.

    This implementation uses adaptive probability models and
    precision management optimized for neural data patterns.
    """

    def __init__(
        self,
        precision_bits: int = 32,
        quantization_bits: int = 16,
        model: Optional[NeuralArithmeticModel] = None
    ):
        """
        Initialize neural arithmetic coder.

        Parameters
        ----------
        precision_bits : int, default=32
            Precision for arithmetic coding (must be > quantization_bits + 8)
        quantization_bits : int, default=16
            Quantization bits for neural signals
        model : NeuralArithmeticModel, optional
            Probability model to use
        """
        self.precision_bits = precision_bits
        self.quantization_bits = quantization_bits
        self.alphabet_size = 2 ** quantization_bits

        # Arithmetic coding range
        self.max_range = (1 << precision_bits) - 1
        self.quarter_range = 1 << (precision_bits - 2)
        self.half_range = 2 * self.quarter_range
        self.three_quarter_range = 3 * self.quarter_range

        # Initialize model
        if model is None:
            self.model = NeuralArithmeticModel(alphabet_size=self.alphabet_size)
        else:
            self.model = model

    def _quantize_neural_data(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Quantize neural data for arithmetic coding.

        Parameters
        ----------
        data : np.ndarray
            Input neural data

        Returns
        -------
        tuple
            (quantized_data, scaling_info)
        """
        # Find data range
        data_min, data_max = np.min(data), np.max(data)

        if data_max > data_min:
            # Normalize to [0, 1]
            normalized = (data - data_min) / (data_max - data_min)
        else:
            normalized = np.zeros_like(data)

        # Quantize to integer levels
        quantized = np.round(normalized * (self.alphabet_size - 1))
        quantized = np.clip(quantized, 0, self.alphabet_size - 1)

        scaling_info = {
            'data_min': data_min,
            'data_max': data_max,
            'quantization_bits': self.quantization_bits
        }

        return quantized.astype(np.int32), scaling_info

    def _dequantize_neural_data(
        self,
        quantized: np.ndarray,
        scaling_info: Dict
    ) -> np.ndarray:
        """
        Dequantize neural data after arithmetic decoding.

        Parameters
        ----------
        quantized : np.ndarray
            Quantized integer data
        scaling_info : dict
            Scaling information from quantization

        Returns
        -------
        np.ndarray
            Dequantized neural data
        """
        data_min = scaling_info['data_min']
        data_max = scaling_info['data_max']

        # Normalize to [0, 1]
        normalized = quantized.astype(np.float64) / (self.alphabet_size - 1)

        # Scale back to original range
        if data_max > data_min:
            return normalized * (data_max - data_min) + data_min
        else:
            return np.full_like(normalized, data_min)

    def encode(self, data: np.ndarray) -> Tuple[bytes, Dict]:
        """
        Encode neural data using arithmetic coding.

        Parameters
        ----------
        data : np.ndarray
            Neural data to encode

        Returns
        -------
        tuple
            (encoded_bytes, metadata)
        """
        # Quantize input data
        quantized, scaling_info = self._quantize_neural_data(data)
        symbols = quantized.flatten()

        # Initialize encoding state
        low = 0
        high = self.max_range
        pending_bits = 0
        encoded_bits = []

        # Reset model context
        self.model.context_history.clear()

        for symbol in symbols:
            # Get symbol probability
            sym_low, sym_high, sym_total = self.model.get_symbol_probability(symbol)

            # Update range
            range_size = high - low + 1
            high = low + (range_size * sym_high) // sym_total - 1
            low = low + (range_size * sym_low) // sym_total

            # Handle range renormalization
            while True:
                if high < self.half_range:
                    # Output 0 and pending 1s
                    encoded_bits.append(0)
                    for _ in range(pending_bits):
                        encoded_bits.append(1)
                    pending_bits = 0
                elif low >= self.half_range:
                    # Output 1 and pending 0s
                    encoded_bits.append(1)
                    for _ in range(pending_bits):
                        encoded_bits.append(0)
                    pending_bits = 0
                    low -= self.half_range
                    high -= self.half_range
                elif (low >= self.quarter_range and
                      high < self.three_quarter_range):
                    # Handle convergence to middle
                    pending_bits += 1
                    low -= self.quarter_range
                    high -= self.quarter_range
                else:
                    break

                # Scale range
                low = 2 * low
                high = 2 * high + 1

                # Ensure range doesn't exceed precision
                if high > self.max_range:
                    high = self.max_range

            # Update model
            self.model.update_model(symbol)

        # Final bits
        if low < self.quarter_range:
            encoded_bits.append(0)
            for _ in range(pending_bits):
                encoded_bits.append(1)
        else:
            encoded_bits.append(1)
            for _ in range(pending_bits):
                encoded_bits.append(0)

        # Convert bits to bytes
        encoded_bytes = self._bits_to_bytes(encoded_bits)

        # Prepare metadata
        metadata = {
            'original_shape': data.shape,
            'scaling_info': scaling_info,
            'model_stats': self.model.get_model_statistics(),
            'encoded_length': len(symbols),
            'compressed_bits': len(encoded_bits)
        }

        return encoded_bytes, metadata

    def decode(self, encoded_bytes: bytes, metadata: Dict) -> np.ndarray:
        """
        Decode neural data using arithmetic coding.

        Parameters
        ----------
        encoded_bytes : bytes
            Encoded data
        metadata : dict
            Encoding metadata

        Returns
        -------
        np.ndarray
            Decoded neural data
        """
        # Convert bytes to bits
        encoded_bits = self._bytes_to_bits(encoded_bytes)

        # Initialize decoding state
        low = 0
        high = self.max_range
        value = 0

        # Read initial value
        for i in range(min(self.precision_bits, len(encoded_bits))):
            value = (value << 1) + encoded_bits[i]

        bit_index = self.precision_bits
        decoded_symbols = []

        # Reset model context
        self.model.context_history.clear()

        # Decode symbols
        for _ in range(metadata['encoded_length']):
            # Find symbol for current value
            range_size = high - low + 1
            symbol = self._find_symbol(value, low, range_size)
            decoded_symbols.append(symbol)

            # Get symbol probability
            sym_low, sym_high, sym_total = self.model.get_symbol_probability(symbol)

            # Update range
            high = low + (range_size * sym_high) // sym_total - 1
            low = low + (range_size * sym_low) // sym_total

            # Handle range renormalization
            while True:
                if high < self.half_range:
                    pass  # No adjustment needed
                elif low >= self.half_range:
                    low -= self.half_range
                    high -= self.half_range
                    value -= self.half_range
                elif (low >= self.quarter_range and
                      high < self.three_quarter_range):
                    low -= self.quarter_range
                    high -= self.quarter_range
                    value -= self.quarter_range
                else:
                    break

                # Scale range and read next bit
                low = 2 * low
                high = 2 * high + 1
                value = 2 * value

                if bit_index < len(encoded_bits):
                    value += encoded_bits[bit_index]
                    bit_index += 1

                # Ensure range doesn't exceed precision
                if high > self.max_range:
                    high = self.max_range

            # Update model
            self.model.update_model(symbol)

        # Reshape and dequantize
        quantized = np.array(decoded_symbols, dtype=np.int32)
        quantized = quantized.reshape(metadata['original_shape'])

        decoded_data = self._dequantize_neural_data(
            quantized,
            metadata['scaling_info']
        )

        return decoded_data

    def _find_symbol(self, value: int, low: int, range_size: int) -> int:
        """
        Find symbol corresponding to current arithmetic coding value.

        Parameters
        ----------
        value : int
            Current arithmetic coding value
        low : int
            Current low bound
        range_size : int
            Current range size

        Returns
        -------
        int
            Corresponding symbol
        """
        # Calculate normalized value
        normalized_value = ((value - low + 1) * self.alphabet_size - 1) // range_size

        # Binary search in cumulative frequencies
        context_key = self.model._get_context_key()
        if context_key in self.model.context_cumulative:
            cumulative = self.model.context_cumulative[context_key]
        else:
            # Use uniform distribution
            frequencies = np.ones(self.alphabet_size, dtype=np.int32)
            cumulative = np.cumsum(np.concatenate([[0], frequencies]))

        # Find symbol
        symbol = 0
        for i in range(self.alphabet_size):
            if cumulative[i + 1] > normalized_value:
                symbol = i
                break

        return symbol

    def _bits_to_bytes(self, bits: List[int]) -> bytes:
        """Convert list of bits to bytes."""
        # Pad to byte boundary
        while len(bits) % 8 != 0:
            bits.append(0)

        bytes_data = bytearray()
        for i in range(0, len(bits), 8):
            byte_val = 0
            for j in range(8):
                if i + j < len(bits):
                    byte_val |= (bits[i + j] << (7 - j))
            bytes_data.append(byte_val)

        return bytes(bytes_data)

    def _bytes_to_bits(self, bytes_data: bytes) -> List[int]:
        """Convert bytes to list of bits."""
        bits = []
        for byte_val in bytes_data:
            for i in range(8):
                bits.append((byte_val >> (7 - i)) & 1)
        return bits


class MultiChannelArithmeticCoder:
    """
    Multi-channel arithmetic coder for neural data.

    This coder processes multiple channels and can exploit
    inter-channel correlations for improved compression.
    """

    def __init__(
        self,
        single_channel_coder: Optional[NeuralArithmeticCoder] = None,
        use_channel_modeling: bool = True
    ):
        """
        Initialize multi-channel coder.

        Parameters
        ----------
        single_channel_coder : NeuralArithmeticCoder, optional
            Single channel coder to use
        use_channel_modeling : bool, default=True
            Whether to use inter-channel modeling
        """
        if single_channel_coder is None:
            self.channel_coder = NeuralArithmeticCoder()
        else:
            self.channel_coder = single_channel_coder

        self.use_channel_modeling = use_channel_modeling

    def encode(self, data: np.ndarray) -> Tuple[List[bytes], Dict]:
        """
        Encode multi-channel neural data.

        Parameters
        ----------
        data : np.ndarray
            Multi-channel data with shape (channels, samples)

        Returns
        -------
        tuple
            (list_of_encoded_channels, metadata)
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_channels, n_samples = data.shape
        encoded_channels = []
        channel_metadata = []

        for ch in range(n_channels):
            # Create fresh coder for each channel
            if self.use_channel_modeling and ch > 0:
                # Use adaptive model that considers previous channels
                # This is a simplified approach - could be enhanced
                model = NeuralArithmeticModel(
                    context_length=6,  # Longer context for inter-channel patterns
                    adaptation_rate=0.05  # Slower adaptation
                )
                coder = NeuralArithmeticCoder(model=model)
            else:
                coder = NeuralArithmeticCoder()

            encoded_data, metadata = coder.encode(data[ch])
            encoded_channels.append(encoded_data)
            channel_metadata.append(metadata)

        global_metadata = {
            'n_channels': n_channels,
            'n_samples': n_samples,
            'channel_metadata': channel_metadata,
            'use_channel_modeling': self.use_channel_modeling
        }

        return encoded_channels, global_metadata

    def decode(
        self,
        encoded_channels: List[bytes],
        metadata: Dict
    ) -> np.ndarray:
        """
        Decode multi-channel neural data.

        Parameters
        ----------
        encoded_channels : list
            List of encoded channel data
        metadata : dict
            Encoding metadata

        Returns
        -------
        np.ndarray
            Decoded multi-channel data
        """
        n_channels = metadata['n_channels']
        n_samples = metadata['n_samples']

        decoded_data = np.zeros((n_channels, n_samples))

        for ch in range(n_channels):
            ch_metadata = metadata['channel_metadata'][ch]

            # Create appropriate decoder
            if metadata['use_channel_modeling'] and ch > 0:
                model = NeuralArithmeticModel(
                    context_length=6,
                    adaptation_rate=0.05
                )
                decoder = NeuralArithmeticCoder(model=model)
            else:
                decoder = NeuralArithmeticCoder()

            decoded_data[ch] = decoder.decode(encoded_channels[ch], ch_metadata)

        return decoded_data


def create_neural_arithmetic_coder(
    optimization_preset: str = 'balanced'
) -> MultiChannelArithmeticCoder:
    """
    Factory function for neural arithmetic coder.

    Parameters
    ----------
    optimization_preset : str, default='balanced'
        Optimization preset ('speed', 'balanced', 'compression')

    Returns
    -------
    MultiChannelArithmeticCoder
        Configured arithmetic coder
    """
    if optimization_preset == 'speed':
        # Fast coding with reduced precision
        base_coder = NeuralArithmeticCoder(
            precision_bits=24,
            quantization_bits=12
        )
    elif optimization_preset == 'compression':
        # Maximum compression with high precision
        base_coder = NeuralArithmeticCoder(
            precision_bits=32,
            quantization_bits=16
        )
    else:  # balanced
        base_coder = NeuralArithmeticCoder()

    return MultiChannelArithmeticCoder(
        single_channel_coder=base_coder,
        use_channel_modeling=True
    )
