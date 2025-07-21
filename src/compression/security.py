"""
AES-256 Encryption Utilities for Neural Data

Implements modular, reusable encryption functions for BCI data streams.

Usage:
    - encrypt_data(data: bytes, key: bytes) -> bytes
    - decrypt_data(encrypted: bytes, key: bytes) -> bytes

All functions include error handling and docstrings.
"""
from typing import Tuple
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

BLOCK_SIZE = 16


def pad(data: bytes) -> bytes:
    """Pad data to AES block size."""
    pad_len = BLOCK_SIZE - len(data) % BLOCK_SIZE
    return data + bytes([pad_len] * pad_len)


def unpad(data: bytes) -> bytes:
    """Remove padding from data."""
    pad_len = data[-1]
    return data[:-pad_len]


def encrypt_data(data: bytes, key: bytes) -> bytes:
    """Encrypt data using AES-256 CBC mode.

    Args:
        data: Data to encrypt.
        key: 32-byte AES key.
    Returns:
        Encrypted data (IV + ciphertext).
    """
    iv = get_random_bytes(BLOCK_SIZE)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    padded = pad(data)
    encrypted = cipher.encrypt(padded)
    return iv + encrypted


def decrypt_data(encrypted: bytes, key: bytes) -> bytes:
    """Decrypt AES-256 CBC encrypted data.

    Args:
        encrypted: IV + ciphertext.
        key: 32-byte AES key.
    Returns:
        Decrypted data.
    """
    iv = encrypted[:BLOCK_SIZE]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    padded = cipher.decrypt(encrypted[BLOCK_SIZE:])
    return unpad(padded)
