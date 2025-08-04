"""Setup configuration for BCI Compression Toolkit."""
import codecs
import os
from setuptools import setup, find_packages


# Get the absolute path to the directory containing this file
here = os.path.abspath(os.path.dirname(__file__))


def read_version():
    """Read package version from __init__.py."""
    init_path = os.path.join(here, "src", "bci_compression", "__init__.py")
    with codecs.open(init_path, "r", "utf-8") as init_file:
        for line in init_file:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    raise RuntimeError("Unable to find version string.")


def read_description():
    """Read long description from README.md."""
    with codecs.open(os.path.join(here, "README.md"), "r", "utf-8") as readme_file:
        return readme_file.read()


# Get package version and description
version = read_version()
long_description = read_description()

setup(
    name="bci-compression",
    version=version,
    author="Kevin",
    author_email="contact@bci-compression.org",
    description="Neural data compression toolkit for brain-computer interfaces",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hkevin01/brain-computer-compression",
    project_urls={
        "Documentation": "https://brain-computer-compression.readthedocs.io/",
        "Bug Reports": "https://github.com/hkevin01/brain-computer-compression/issues",
        "Source Code": "https://github.com/hkevin01/brain-computer-compression",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0,<1.25.0",
        "scipy>=1.7.0,<1.11.0",
        "torch>=1.13.0,<2.1.0",
        "transformers>=4.20.0,<5.0.0",
        "scikit-learn>=1.1.0,<1.4.0",
        "matplotlib>=3.5.0,<3.8.0",
        "h5py>=3.7.0,<3.10.0",
        "pywavelets>=1.1.1,<1.2.0",
        "pandas>=1.3.0,<2.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0,<8.0.0",
            "pytest-cov>=2.12.0,<3.0.0",
            "pytest-benchmark>=3.4.0,<4.0.0",
            "coverage>=6.0.0,<8.0.0",
            "black>=22.0.0,<23.0.0",
            "flake8>=4.0.0,<5.0.0",
            "mypy>=0.991,<1.0.0",
            "pre-commit>=2.17.0,<3.0.0",
            "hypothesis>=6.0.0,<7.0.0",
        ],
        "gpu": [
            "cupy-cuda11x>=12.0.0",
        ],
        "docs": [
            "sphinx>=4.5.0,<5.0.0",
            "sphinx-rtd-theme>=1.0.0,<2.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bci-compress=bci_compression.cli:main",
            "bci-benchmark=bci_compression.benchmarking.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
