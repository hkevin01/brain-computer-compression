import numpy as np
from bci_compression.algorithms.adaptive_selector import create_adaptive_selector, AdaptiveSelectorConfig


def test_adaptive_selector_basic() -> None:
    cfg = AdaptiveSelectorConfig(fs=1000.0, hysteresis=2, switch_threshold=0.05)
    selector = create_adaptive_selector(config=cfg)
    # Simulate windows with varying spectral content
    fs = 1000
    t = np.arange(0, 1.0, 1 / fs)
    win1 = np.sin(2 * np.pi * 10 * t)[None, :]  # alpha
    win2 = np.sin(2 * np.pi * 25 * t)[None, :]  # beta
    win3 = np.sin(2 * np.pi * 40 * t)[None, :]  # gamma (partial beyond 30)
    for w in [win1, win2, win3, win2, win1]:
        choice, meta = selector.select(w)
        assert 'scores' in meta and 'features' in meta
        assert choice in meta['scores']


def test_hysteresis_behavior() -> None:
    cfg = AdaptiveSelectorConfig(hysteresis=2, switch_threshold=0.10)
    selector = create_adaptive_selector(config=cfg)
    fs = 1000
    t = np.arange(0, 1.0, 1 / fs)
    base = np.sin(2 * np.pi * 12 * t)[None, :]

    last = None
    for _ in range(5):
        choice, _ = selector.select(base)
        if last is not None:
            # Should stabilize quickly
            assert choice == last
        last = choice
