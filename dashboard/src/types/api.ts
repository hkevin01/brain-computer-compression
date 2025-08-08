export interface PluginInfo {
  name: string
  modes: string[]
  capabilities: Record<string, any>
  is_lossless: boolean
  is_lossy: boolean
  supports_streaming: boolean
  supports_gpu: boolean
}

export interface CompressRequest {
  plugin: string
  mode?: string
  quality_level?: number
  options?: Record<string, any>
}

export interface BenchmarkRequest {
  dataset: string
  plugin: string
  duration_s: number
  channels: number
  sample_rate: number
  metrics: string[]
}

export interface MetricsTelemetry {
  timestamp: number
  session_id: string
  compression_ratio: number
  latency_ms: number
  snr_db?: number
  psnr_db?: number
  spectral_coherence_error?: number
  spike_f1?: number
  memory_usage_mb?: number
  gpu_available: boolean
}

export interface CompressionResult {
  compression_ratio: number
  compression_time_ms: number
  original_size: number
  compressed_size: number
  output_filename: string
  snr_db?: number
  psnr_db?: number
}

export interface DecompressionResult {
  decompression_time_ms: number
  decompressed_size: number
  output_filename: string
  snr_db?: number
  psnr_db?: number
}

export interface BenchmarkResult {
  avg_compression_ratio: number
  avg_latency_ms: number
  avg_snr_db?: number
  avg_psnr_db?: number
  num_trials: number
  plugin: string
  mode: string
  dataset_info: {
    channels: number
    samples: number
    duration_s: number
    sample_rate: number
  }
  results: Record<string, number>
  gpu_available: boolean
}
