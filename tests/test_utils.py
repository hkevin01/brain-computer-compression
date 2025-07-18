"""
Unit tests for utility functions and data structures (Phase 1)
"""
import pytest
from bci_compression.data_processing.utils import flatten, chunk_data

def test_flatten_basic():
    assert flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]

def test_chunk_data_basic():
    data = list(range(10))
    chunks = list(chunk_data(data, 3))
    assert chunks == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
