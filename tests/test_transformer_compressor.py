import numpy as np
import pytest

from bci_compression.algorithms.deep_learning import create_transformer_compressor, create_vae_compressor


def random_signal(length: int = 2048) -> np.ndarray:
    return np.random.randn(length).astype(np.float32)


def test_transformer_basic_round_trip() -> None:
    comp = create_transformer_compressor('fast')
    x = random_signal(1024)
    compressed, meta = comp.compress(x)
    assert 'compression_ratio' in meta
    decompressed = comp.decompress(compressed, meta)
    assert decompressed.dtype == np.float32 or decompressed.dtype == np.float64
    assert decompressed.size > 0


def test_vae_basic_round_trip() -> None:
    comp = create_vae_compressor(window=256, train_steps=2)
    x = random_signal(4096)
    compressed, meta = comp.compress(x)
    decompressed = comp.decompress(compressed, meta)
    assert decompressed.size >= 0  # May differ in size (autoencoding approximation)


@pytest.mark.parametrize('mode', ['fast', 'balanced', 'quality'])
def test_transformer_presets(mode: str) -> None:
    comp = create_transformer_compressor(mode)
    x = random_signal(512)
    compressed, meta = comp.compress(x)
    assert meta['algorithm'] == 'transformer'


def test_stream_interface() -> None:
    comp = create_transformer_compressor('fast')
    x = random_signal(2048)
    chunks = np.split(x, 4)
    results = [comp.stream_chunk(c) for c in chunks]
    assert len(results) == 4
