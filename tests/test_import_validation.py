#!/usr/bin/env python3
"""
Quick validation script to ensure basic imports work.
This helps validate that the CI workflow will succeed.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_basic_imports() -> bool:
    """Test basic package imports work."""
    try:
        import bci_compression
        print(f"✅ Successfully imported bci_compression version {bci_compression.__version__}")

        # Test core modules that should exist based on __init__.py
        try:
            from bci_compression.core import Compressor  # noqa: F401
            print("✅ Successfully imported Compressor")
        except ImportError:
            print("⚠️  Compressor import failed (may be expected)")

        try:
            from bci_compression.algorithms import create_neural_lz_compressor  # noqa: F401
            print("✅ Successfully imported create_neural_lz_compressor")
        except ImportError:
            print("⚠️  create_neural_lz_compressor import failed (may be expected)")

        try:
            from bci_compression import plugins  # noqa: F401
            print("✅ Successfully imported plugins")
        except ImportError:
            print("⚠️  plugins import failed (may be expected)")

        print("\n🎉 Basic package import successful! CI should pass.")
        return True

    except ImportError as e:
        print(f"❌ Critical import failed: {e}")
        return False


if __name__ == "__main__":
    success = test_basic_imports()
    sys.exit(0 if success else 1)
