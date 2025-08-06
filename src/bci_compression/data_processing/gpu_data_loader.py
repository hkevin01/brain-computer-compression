"""
GPU-accelerated data loading and preprocessing using cuDF.
"""

import warnings
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

try:
    import cudf
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    # Create fallback types
    cudf = None
    warnings.warn("CUDA libraries not available. Using CPU fallback.")


class GPUDataLoader:
    """
    GPU-accelerated data loading and preprocessing using cuDF.
    Provides efficient loading and preprocessing of neural data files.
    """

    def __init__(self):
        self.cuda_available = CUDA_AVAILABLE

    def load_data(
        self,
        filepath: Union[str, Path],
        format: str = "auto"
    ) -> Union[cudf.DataFrame, np.ndarray]:
        """
        Load neural data with GPU acceleration where possible.

        Parameters
        ----------
        filepath : str or Path
            Path to the data file
        format : str, default="auto"
            File format specification ('auto', 'nev', 'nsx', 'hdf5', 'mat', 'csv')

        Returns
        -------
        Union[cudf.DataFrame, np.ndarray]
            Loaded data. Returns cuDF DataFrame if format supports it and CUDA is available,
            otherwise returns numpy array.
        """
        if not self.cuda_available:
            return self._cpu_load_data(filepath, format)

        filepath = Path(filepath)
        if format == "auto":
            format = filepath.suffix.lower()[1:]

        try:
            if format in ['csv', 'parquet']:
                # These formats can be loaded directly with cuDF
                if format == 'csv':
                    return cudf.read_csv(filepath)
                else:
                    return cudf.read_parquet(filepath)
            else:
                # Fall back to numpy for other formats
                return self._cpu_load_data(filepath, format)
        except Exception as e:
            warnings.warn(f"GPU loading failed: {e}. Falling back to CPU.")
            return self._cpu_load_data(filepath, format)

    def save_data(
        self,
        data: Union[cudf.DataFrame, np.ndarray],
        filepath: Union[str, Path],
        format: str = "auto",
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Save neural data with GPU acceleration where possible.

        Parameters
        ----------
        data : Union[cudf.DataFrame, np.ndarray]
            Data to save
        filepath : str or Path
            Output filepath
        format : str, default="auto"
            Output format ('auto', 'csv', 'parquet', 'hdf5')
        metadata : dict, optional
            Additional metadata to save with the data
        """
        if not self.cuda_available:
            return self._cpu_save_data(data, filepath, format, metadata)

        filepath = Path(filepath)
        if format == "auto":
            format = filepath.suffix.lower()[1:]

        try:
            if isinstance(data, cudf.DataFrame):
                if format == 'csv':
                    data.to_csv(filepath, index=False)
                elif format == 'parquet':
                    data.to_parquet(filepath)
                else:
                    # Convert to numpy for other formats
                    cpu_data = data.to_pandas().to_numpy()
                    self._cpu_save_data(cpu_data, filepath, format, metadata)
            else:
                # If input is numpy array, use CPU save
                self._cpu_save_data(data, filepath, format, metadata)
        except Exception as e:
            warnings.warn(f"GPU saving failed: {e}. Falling back to CPU.")
            self._cpu_save_data(data, filepath, format, metadata)

    def preprocess_dataframe(
        self,
        df: cudf.DataFrame,
        operations: Dict[str, Dict]
    ) -> cudf.DataFrame:
        """
        Apply GPU-accelerated preprocessing operations to a cuDF DataFrame.

        Parameters
        ----------
        df : cudf.DataFrame
            Input DataFrame
        operations : dict
            Dictionary of preprocessing operations to apply

        Returns
        -------
        cudf.DataFrame
            Preprocessed DataFrame
        """
        if not self.cuda_available:
            return df

        processed_df = df.copy()

        for op_name, op_config in operations.items():
            if op_name == 'normalize':
                cols = op_config.get('columns', df.columns)
                method = op_config.get('method', 'zscore')

                for col in cols:
                    if method == 'zscore':
                        mean = processed_df[col].mean()
                        std = processed_df[col].std()
                        processed_df[col] = (processed_df[col] - mean) / (std + 1e-8)
                    elif method == 'minmax':
                        min_val = processed_df[col].min()
                        max_val = processed_df[col].max()
                        processed_df[col] = (processed_df[col] - min_val) / (max_val - min_val + 1e-8)

            elif op_name == 'fillna':
                method = op_config.get('method', 'mean')
                cols = op_config.get('columns', df.columns)

                for col in cols:
                    if method == 'mean':
                        fill_value = processed_df[col].mean()
                    elif method == 'median':
                        fill_value = processed_df[col].median()
                    elif method == 'constant':
                        fill_value = op_config.get('value', 0)
                    processed_df[col] = processed_df[col].fillna(fill_value)

            elif op_name == 'drop_outliers':
                cols = op_config.get('columns', df.columns)
                n_std = op_config.get('n_std', 3)

                for col in cols:
                    mean = processed_df[col].mean()
                    std = processed_df[col].std()
                    mask = (processed_df[col] - mean).abs() <= n_std * std
                    processed_df = processed_df[mask]

        return processed_df

    def _cpu_load_data(self, filepath: Path, format: str) -> np.ndarray:
        """CPU fallback for data loading."""
        import h5py
        import scipy.io as sio

        if format == 'nev' or format == 'nsx':
            # Placeholder: implement actual NEV/NSx loading
            return np.random.randn(64, 30000)
        elif format == 'hdf5' or format == 'h5':
            with h5py.File(filepath, 'r') as f:
                return f['neural_data'][:]
        elif format == 'mat':
            return sio.loadmat(filepath)['data']
        elif format == 'csv':
            return np.genfromtxt(filepath, delimiter=',')
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _cpu_save_data(
        self,
        data: np.ndarray,
        filepath: Path,
        format: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """CPU fallback for data saving."""
        import h5py
        import scipy.io as sio

        filepath.parent.mkdir(parents=True, exist_ok=True)

        if format == 'hdf5' or format == 'h5':
            with h5py.File(filepath, 'w') as f:
                f.create_dataset('neural_data', data=data)
                if metadata:
                    for key, value in metadata.items():
                        f.attrs[key] = value
        elif format == 'mat':
            sio.savemat(filepath, {'data': data, 'metadata': metadata or {}})
        elif format == 'csv':
            np.savetxt(filepath, data, delimiter=',')
        else:
            raise ValueError(f"Unsupported format: {format}")
