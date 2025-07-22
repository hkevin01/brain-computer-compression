from typing import Dict, Any, Optional
import numpy as np
from src.utils.metrics_helper import snr as calculate_snr, compression_ratio
from src.config.pipeline_config_manager import PipelineConfigManager
from src.utils.artifact_detector import ArtifactDetector
import logging

class PipelineConnector:
    def __init__(self):
        self.config_manager = PipelineConfigManager()
        self.logger = logging.getLogger("PipelineConnector")
        self.logger.setLevel(logging.INFO)
        self.artifact_detector = ArtifactDetector()

    def get_live_metrics(self, num_channels: int = 64, sample_size: int = 1000, use_gpu: bool = False, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Simulates real neural data metrics for multiple channels.
        Args:
            num_channels: Number of neural channels (default: 64)
            sample_size: Number of samples per channel (default: 1000)
            use_gpu: Whether to use GPU acceleration (default: False)
            seed: Optional random seed for reproducibility
        Returns:
            Dictionary of metrics: compression_ratio, snr_db
        """
        try:
            if seed is not None:
                np.random.seed(seed)
            signal = np.random.normal(0, 1, (num_channels, sample_size))
            noise = np.random.normal(0, 0.1, (num_channels, sample_size))
            snr_db = calculate_snr(signal, noise)
            compression_ratio_val = np.random.uniform(2.5, 4.0)
            metrics = {
                "compression_ratio": round(compression_ratio_val, 2),
                "snr_db": round(snr_db, 2)
            }
            self.logger.info(f"Live metrics: {metrics}")
            return metrics
        except Exception as e:
            self.logger.error(f"Error in get_live_metrics: {e}")
            return {"error": str(e)}

    def simulate_compression(self, original_size: int, compressed_size: int) -> float:
        """
        Calculates compression ratio from original and compressed sizes.
        Args:
            original_size: Size of original data (bytes)
            compressed_size: Size after compression (bytes)
        Returns:
            Compression ratio (float)
        """
        return round(compression_ratio(original_size, compressed_size), 2)

    def inject_artifacts(self, signal: np.ndarray, artifact_type: str = "spike", severity: float = 0.5) -> np.ndarray:
        """
        Injects artifacts into neural signal for simulation.
        Args:
            signal: Neural signal array (channels x samples)
            artifact_type: Type of artifact ("spike", "noise", "drift")
            severity: Severity of artifact (0.0–1.0)
        Returns:
            Modified signal array
        """
        try:
            signal = signal.copy()
            if artifact_type == "spike":
                num_spikes = int(severity * signal.size * 0.01)
                idx = np.random.choice(signal.size, num_spikes, replace=False)
                signal.flat[idx] += np.random.uniform(5, 10, num_spikes)
            elif artifact_type == "noise":
                signal += np.random.normal(0, severity, signal.shape)
            elif artifact_type == "drift":
                drift = np.linspace(0, severity, signal.shape[1])
                signal += drift
            self.logger.info(f"Injected {artifact_type} artifact with severity {severity}")
            return signal
        except Exception as e:
            self.logger.error(f"Error in inject_artifacts: {e}")
            return signal

    def simulate_multimodal_fusion(self, eeg: np.ndarray, fmri: np.ndarray) -> np.ndarray:
        """
        Simulates multi-modal fusion of EEG and fMRI data.
        Args:
            eeg: EEG signal array (channels x samples)
            fmri: fMRI signal array (channels x samples)
        Returns:
            Fused signal array (channels x samples)
        """
        try:
            eeg_weight = 0.7
            fmri_weight = 0.3
            min_shape = (min(eeg.shape[0], fmri.shape[0]), min(eeg.shape[1], fmri.shape[1]))
            eeg = eeg[:min_shape[0], :min_shape[1]]
            fmri = fmri[:min_shape[0], :min_shape[1]]
            fused = eeg_weight * eeg + fmri_weight * fmri
            self.logger.info(f"Fused EEG and fMRI with weights {eeg_weight}, {fmri_weight}")
            return fused
        except Exception as e:
            self.logger.error(f"Error in simulate_multimodal_fusion: {e}")
            return eeg

    def update_pipeline_config(self, new_config: Dict[str, Any]) -> bool:
        """
        Updates the pipeline configuration using PipelineConfigManager.
        Args:
            new_config: Dictionary of new configuration parameters
        Returns:
            Success status (bool)
        """
        try:
            self.config_manager.update_config(new_config)
            self.logger.info(f"Pipeline config updated: {new_config}")
            return True
        except Exception as e:
            self.logger.error(f"Config update failed: {e}")
            return False

    def analyze_artifacts(self, signal: np.ndarray) -> Dict[str, Any]:
        """
        Analyzes neural signal for artifacts using ArtifactDetector.
        Args:
            signal: Neural signal array (channels x samples)
        Returns:
            Dictionary summarizing detected artifacts
        """
        try:
            summary = self.artifact_detector.detect_all(signal)
            self.logger.info(f"Artifact analysis: {summary}")
            return summary
        except Exception as e:
            self.logger.error(f"Error in analyze_artifacts: {e}")
            return {"error": str(e)}



