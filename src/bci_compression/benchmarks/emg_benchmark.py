"""
EMG Compression Benchmarking Suite

This module provides benchmarking utilities specifically designed for EMG
compression algorithms, including dataset handling, performance evaluation,
and comparison with baseline methods.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging

from ..algorithms.emg_compression import (
    EMGLZCompressor, EMGPerceptualQuantizer, EMGPredictiveCompressor
)
from ..metrics.emg_quality import EMGQualityMetrics, evaluate_emg_compression_quality
from ..mobile.emg_mobile import MobileEMGCompressor

logger = logging.getLogger(__name__)


class EMGBenchmarkSuite:
    """
    Comprehensive benchmarking suite for EMG compression algorithms.
    
    Provides standardized evaluation of compression performance on EMG data,
    including clinical relevance metrics and comparison with baseline methods.
    """
    
    def __init__(
        self,
        sampling_rate: float = 2000.0,
        output_dir: Optional[str] = None
    ):
        """
        Initialize EMG benchmark suite.
        
        Parameters
        ----------
        sampling_rate : float, default=2000.0
            EMG sampling rate in Hz
        output_dir : str, optional
            Directory for saving benchmark results
        """
        self.sampling_rate = sampling_rate
        self.output_dir = Path(output_dir) if output_dir else Path("emg_benchmark_results")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize quality metrics
        self.quality_metrics = EMGQualityMetrics(sampling_rate)
        
        # Benchmark results storage
        self.results = {}
        
        # EMG compressors to benchmark
        self.compressors = {
            'EMG_LZ': EMGLZCompressor(sampling_rate=sampling_rate),
            'EMG_Perceptual': EMGPerceptualQuantizer(sampling_rate=sampling_rate),
            'EMG_Predictive': EMGPredictiveCompressor(),
            'Mobile_EMG': MobileEMGCompressor(emg_sampling_rate=sampling_rate)
        }
        
        # Baseline compressors for comparison
        self.baseline_compressors = {}
        
    def run_comprehensive_benchmark(
        self,
        test_datasets: Dict[str, np.ndarray],
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive benchmark on EMG datasets.
        
        Parameters
        ----------
        test_datasets : dict
            Dictionary of test datasets {name: data}
        save_results : bool, default=True
            Whether to save results to disk
            
        Returns
        -------
        dict
            Comprehensive benchmark results
        """
        logger.info("Starting EMG compression benchmark...")
        
        all_results = {}
        
        for dataset_name, data in test_datasets.items():
            logger.info(f"Benchmarking dataset: {dataset_name}")
            
            dataset_results = {}
            
            # Test each compressor
            for compressor_name, compressor in self.compressors.items():
                logger.info(f"  Testing {compressor_name}...")
                
                try:
                    result = self._benchmark_single_compressor(
                        compressor, data, f"{dataset_name}_{compressor_name}"
                    )
                    dataset_results[compressor_name] = result
                    
                except Exception as e:
                    logger.error(f"Error benchmarking {compressor_name}: {e}")
                    dataset_results[compressor_name] = {'error': str(e)}
            
            all_results[dataset_name] = dataset_results
        
        # Calculate summary statistics
        summary = self._calculate_summary_statistics(all_results)
        all_results['summary'] = summary
        
        # Save results
        if save_results:
            self._save_benchmark_results(all_results)
        
        # Generate plots
        self._generate_benchmark_plots(all_results)
        
        self.results = all_results
        return all_results
    
    def _benchmark_single_compressor(
        self,
        compressor: Any,
        data: np.ndarray,
        test_id: str
    ) -> Dict[str, Any]:
        """Benchmark single compressor on data."""
        results = {
            'test_id': test_id,
            'data_shape': data.shape,
            'data_size_bytes': data.nbytes
        }
        
        # Compression timing
        start_time = time.time()
        try:
            compressed_data = compressor.compress(data)
            compression_time = time.time() - start_time
            
            results.update({
                'compression_successful': True,
                'compression_time_ms': compression_time * 1000,
                'compressed_size_bytes': len(compressed_data),
                'compression_ratio': data.nbytes / len(compressed_data)
            })
            
        except Exception as e:
            results.update({
                'compression_successful': False,
                'compression_error': str(e)
            })
            return results
        
        # Decompression timing
        start_time = time.time()
        try:
            decompressed_data = compressor.decompress(compressed_data)
            decompression_time = time.time() - start_time
            
            results.update({
                'decompression_successful': True,
                'decompression_time_ms': decompression_time * 1000,
                'total_latency_ms': (compression_time + decompression_time) * 1000
            })
            
        except Exception as e:
            results.update({
                'decompression_successful': False,
                'decompression_error': str(e)
            })
            return results
        
        # Quality metrics
        try:
            quality_results = evaluate_emg_compression_quality(
                data, decompressed_data, self.sampling_rate
            )
            results['quality_metrics'] = quality_results
            
        except Exception as e:
            results['quality_error'] = str(e)
        
        # Additional EMG-specific metrics
        try:
            emg_metrics = self._calculate_emg_specific_metrics(data, decompressed_data)
            results['emg_specific_metrics'] = emg_metrics
            
        except Exception as e:
            results['emg_metrics_error'] = str(e)
        
        return results
    
    def _calculate_emg_specific_metrics(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray
    ) -> Dict[str, float]:
        """Calculate EMG-specific performance metrics."""
        metrics = {}
        
        # Muscle activation detection accuracy
        activation_metrics = self.quality_metrics.muscle_activation_detection_accuracy(
            original, reconstructed
        )
        metrics.update(activation_metrics)
        
        # Envelope correlation
        envelope_metrics = self.quality_metrics.emg_envelope_correlation(
            original, reconstructed
        )
        metrics.update(envelope_metrics)
        
        # Spectral fidelity
        spectral_metrics = self.quality_metrics.emg_spectral_fidelity(
            original, reconstructed
        )
        metrics.update(spectral_metrics)
        
        # Timing precision
        timing_metrics = self.quality_metrics.emg_timing_precision(
            original, reconstructed
        )
        metrics.update(timing_metrics)
        
        return metrics
    
    def _calculate_summary_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics across all benchmarks."""
        summary = {
            'compressor_rankings': {},
            'average_metrics': {},
            'best_performers': {}
        }
        
        # Collect metrics across all datasets
        all_metrics = {}
        for dataset_name, dataset_results in results.items():
            if dataset_name == 'summary':  # Skip summary if already exists
                continue
                
            for compressor_name, compressor_results in dataset_results.items():
                if 'error' in compressor_results:
                    continue
                    
                if compressor_name not in all_metrics:
                    all_metrics[compressor_name] = {
                        'compression_ratios': [],
                        'compression_times': [],
                        'quality_scores': [],
                        'envelope_correlations': []
                    }
                
                # Collect performance metrics
                if 'compression_ratio' in compressor_results:
                    all_metrics[compressor_name]['compression_ratios'].append(
                        compressor_results['compression_ratio']
                    )
                
                if 'compression_time_ms' in compressor_results:
                    all_metrics[compressor_name]['compression_times'].append(
                        compressor_results['compression_time_ms']
                    )
                
                # Quality metrics
                if 'quality_metrics' in compressor_results:
                    quality = compressor_results['quality_metrics']
                    if 'overall_quality_score' in quality:
                        all_metrics[compressor_name]['quality_scores'].append(
                            quality['overall_quality_score']
                        )
                    
                    if 'envelope_preservation' in quality:
                        env_corr = quality['envelope_preservation'].get('envelope_correlation', 0)
                        all_metrics[compressor_name]['envelope_correlations'].append(env_corr)
        
        # Calculate averages
        for compressor_name, metrics in all_metrics.items():
            summary['average_metrics'][compressor_name] = {
                'avg_compression_ratio': np.mean(metrics['compression_ratios']) if metrics['compression_ratios'] else 0,
                'avg_compression_time_ms': np.mean(metrics['compression_times']) if metrics['compression_times'] else 0,
                'avg_quality_score': np.mean(metrics['quality_scores']) if metrics['quality_scores'] else 0,
                'avg_envelope_correlation': np.mean(metrics['envelope_correlations']) if metrics['envelope_correlations'] else 0
            }
        
        # Determine best performers
        if summary['average_metrics']:
            # Best compression ratio
            best_compression = max(
                summary['average_metrics'].items(),
                key=lambda x: x[1]['avg_compression_ratio']
            )
            summary['best_performers']['compression_ratio'] = best_compression[0]
            
            # Best quality
            best_quality = max(
                summary['average_metrics'].items(),
                key=lambda x: x[1]['avg_quality_score']
            )
            summary['best_performers']['quality'] = best_quality[0]
            
            # Fastest compression
            best_speed = min(
                summary['average_metrics'].items(),
                key=lambda x: x[1]['avg_compression_time_ms']
            )
            summary['best_performers']['speed'] = best_speed[0]
        
        return summary
    
    def _generate_benchmark_plots(self, results: Dict[str, Any]):
        """Generate visualization plots for benchmark results."""
        if 'summary' not in results or not results['summary']['average_metrics']:
            logger.warning("No summary data available for plotting")
            return
        
        avg_metrics = results['summary']['average_metrics']
        compressor_names = list(avg_metrics.keys())
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('EMG Compression Benchmark Results', fontsize=16)
        
        # Plot 1: Compression Ratio
        compression_ratios = [avg_metrics[name]['avg_compression_ratio'] for name in compressor_names]
        axes[0, 0].bar(compressor_names, compression_ratios)
        axes[0, 0].set_title('Average Compression Ratio')
        axes[0, 0].set_ylabel('Compression Ratio')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Compression Time
        compression_times = [avg_metrics[name]['avg_compression_time_ms'] for name in compressor_names]
        axes[0, 1].bar(compressor_names, compression_times)
        axes[0, 1].set_title('Average Compression Time')
        axes[0, 1].set_ylabel('Time (ms)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Quality Score
        quality_scores = [avg_metrics[name]['avg_quality_score'] for name in compressor_names]
        axes[1, 0].bar(compressor_names, quality_scores)
        axes[1, 0].set_title('Average Quality Score')
        axes[1, 0].set_ylabel('Quality Score (0-1)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Envelope Correlation
        envelope_corrs = [avg_metrics[name]['avg_envelope_correlation'] for name in compressor_names]
        axes[1, 1].bar(compressor_names, envelope_corrs)
        axes[1, 1].set_title('Average Envelope Correlation')
        axes[1, 1].set_ylabel('Correlation')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'emg_benchmark_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Performance trade-off plot
        self._plot_performance_tradeoffs(avg_metrics)
    
    def _plot_performance_tradeoffs(self, avg_metrics: Dict[str, Dict[str, float]]):
        """Plot performance trade-offs between compression and quality."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        compression_ratios = []
        quality_scores = []
        compressor_names = []
        
        for name, metrics in avg_metrics.items():
            compression_ratios.append(metrics['avg_compression_ratio'])
            quality_scores.append(metrics['avg_quality_score'])
            compressor_names.append(name)
        
        # Scatter plot
        scatter = ax.scatter(compression_ratios, quality_scores, s=100, alpha=0.7)
        
        # Annotate points
        for i, name in enumerate(compressor_names):
            ax.annotate(name, (compression_ratios[i], quality_scores[i]),
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Compression Ratio')
        ax.set_ylabel('Quality Score')
        ax.set_title('EMG Compression: Quality vs Compression Trade-off')
        ax.grid(True, alpha=0.3)
        
        plt.savefig(self.output_dir / 'emg_quality_vs_compression.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_benchmark_results(self, results: Dict[str, Any]):
        """Save benchmark results to file."""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(results)
        
        # Save JSON results
        with open(self.output_dir / 'emg_benchmark_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save summary report
        self._generate_text_report(results)
        
        logger.info(f"Benchmark results saved to {self.output_dir}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, tuple):
            return list(obj)
        else:
            return obj
    
    def _generate_text_report(self, results: Dict[str, Any]):
        """Generate human-readable text report."""
        report_path = self.output_dir / 'emg_benchmark_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("EMG Compression Benchmark Report\n")
            f.write("=" * 40 + "\n\n")
            
            if 'summary' in results and 'best_performers' in results['summary']:
                best = results['summary']['best_performers']
                f.write("Best Performers:\n")
                f.write(f"  Best Compression Ratio: {best.get('compression_ratio', 'N/A')}\n")
                f.write(f"  Best Quality: {best.get('quality', 'N/A')}\n")
                f.write(f"  Fastest: {best.get('speed', 'N/A')}\n\n")
            
            if 'summary' in results and 'average_metrics' in results['summary']:
                f.write("Average Performance Metrics:\n")
                f.write("-" * 30 + "\n")
                
                for compressor, metrics in results['summary']['average_metrics'].items():
                    f.write(f"\n{compressor}:\n")
                    f.write(f"  Compression Ratio: {metrics['avg_compression_ratio']:.2f}\n")
                    f.write(f"  Compression Time: {metrics['avg_compression_time_ms']:.2f} ms\n")
                    f.write(f"  Quality Score: {metrics['avg_quality_score']:.3f}\n")
                    f.write(f"  Envelope Correlation: {metrics['avg_envelope_correlation']:.3f}\n")


def create_synthetic_emg_datasets() -> Dict[str, np.ndarray]:
    """
    Create synthetic EMG datasets for benchmarking.
    
    Returns
    -------
    dict
        Dictionary of synthetic EMG datasets
    """
    datasets = {}
    sampling_rate = 2000.0
    
    # Dataset 1: Single muscle activation
    duration = 5.0  # seconds
    n_samples = int(duration * sampling_rate)
    time = np.linspace(0, duration, n_samples)
    
    # Simulate muscle activation burst
    activation_start = 1.0
    activation_end = 3.0
    activation_mask = (time >= activation_start) & (time <= activation_end)
    
    # Base noise
    emg_signal = np.random.normal(0, 0.05, n_samples)
    
    # Add muscle activation
    activation_amplitude = 0.5
    activation_freq = 150  # Hz, typical EMG frequency content
    activation_signal = activation_amplitude * np.sin(2 * np.pi * activation_freq * time)
    activation_signal *= np.exp(-((time - 2.0) ** 2) / 0.5)  # Gaussian envelope
    
    emg_signal[activation_mask] += activation_signal[activation_mask]
    
    datasets['single_activation'] = emg_signal.reshape(1, -1)
    
    # Dataset 2: Multi-channel with different activation patterns
    n_channels = 4
    multi_channel_emg = np.random.normal(0, 0.05, (n_channels, n_samples))
    
    for ch in range(n_channels):
        # Different activation timing for each channel
        start_time = 0.5 + ch * 0.5
        end_time = start_time + 1.5
        ch_mask = (time >= start_time) & (time <= end_time)
        
        if np.any(ch_mask):
            ch_activation = 0.3 * (ch + 1) * np.sin(2 * np.pi * (100 + ch * 50) * time)
            ch_activation *= np.exp(-((time - (start_time + end_time) / 2) ** 2) / 0.3)
            multi_channel_emg[ch, ch_mask] += ch_activation[ch_mask]
    
    datasets['multi_channel'] = multi_channel_emg
    
    # Dataset 3: Fatigue simulation (changing frequency content)
    fatigue_signal = np.random.normal(0, 0.05, n_samples)
    
    # Simulate muscle fatigue - decreasing mean frequency over time
    for i in range(n_samples):
        if time[i] > 1.0 and time[i] < 4.0:
            # Frequency decreases with fatigue
            freq = 200 - 50 * (time[i] - 1.0) / 3.0
            amplitude = 0.4 * np.exp(-(time[i] - 2.5) ** 2 / 1.0)
            fatigue_signal[i] += amplitude * np.sin(2 * np.pi * freq * time[i])
    
    datasets['fatigue_simulation'] = fatigue_signal.reshape(1, -1)
    
    return datasets


def run_emg_benchmark_example():
    """Run an example EMG compression benchmark."""
    # Create synthetic datasets
    datasets = create_synthetic_emg_datasets()
    
    # Initialize benchmark suite
    benchmark = EMGBenchmarkSuite(sampling_rate=2000.0)
    
    # Run benchmark
    results = benchmark.run_comprehensive_benchmark(datasets)
    
    print("EMG Benchmark completed!")
    print(f"Results saved to: {benchmark.output_dir}")
    
    # Print summary
    if 'summary' in results and 'best_performers' in results['summary']:
        best = results['summary']['best_performers']
        print(f"\nBest Performers:")
        print(f"  Compression: {best.get('compression_ratio', 'N/A')}")
        print(f"  Quality: {best.get('quality', 'N/A')}")
        print(f"  Speed: {best.get('speed', 'N/A')}")
    
    return results


if __name__ == "__main__":
    # Run example benchmark
    results = run_emg_benchmark_example()
