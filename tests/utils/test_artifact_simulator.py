import numpy as np
from src.utils import artifact_simulator

def test_inject_spike():
    signal = np.zeros((2, 100))
    out = artifact_simulator.inject_spike(signal, severity=1.0)
    assert np.any(out != 0)

def test_inject_noise():
    signal = np.zeros((2, 100))
    out = artifact_simulator.inject_noise(signal, severity=1.0)
    assert np.std(out) > 0

def test_inject_drift():
    signal = np.zeros((2, 100))
    out = artifact_simulator.inject_drift(signal, severity=1.0)
    assert np.allclose(out[:, -1], 1.0, atol=0.1)
