import { useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts'
import { BenchmarkResult, PluginInfo } from '../types/api'

interface BenchmarkComparison {
  plugin: string
  mode: string
  compression_ratio: number
  latency_ms: number
  snr_db?: number
  psnr_db?: number
  quality_score: number
}

export default function Benchmarks() {
  const [plugins, setPlugins] = useState<PluginInfo[]>([])
  const [selectedPlugins, setSelectedPlugins] = useState<string[]>([])
  const [selectedMode, setSelectedMode] = useState('balanced')
  const [isBenchmarking, setIsBenchmarking] = useState(false)
  const [benchmarkResults, setBenchmarkResults] = useState<BenchmarkComparison[]>([])
  const [currentTest, setCurrentTest] = useState('')
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    fetchPlugins()
  }, [])

  const fetchPlugins = async () => {
    try {
      const response = await fetch('/api/plugins')
      if (response.ok) {
        const pluginData = await response.json()
        setPlugins(pluginData)
      }
    } catch (error) {
      console.error('Failed to fetch plugins:', error)
    }
  }

  const runBenchmark = async () => {
    if (selectedPlugins.length === 0) {
      alert('Please select at least one plugin to benchmark')
      return
    }

    setIsBenchmarking(true)
    setBenchmarkResults([])
    setProgress(0)

    const qualityLevels = [0.5, 0.7, 0.9]
    const totalTests = selectedPlugins.length * qualityLevels.length
    let completed = 0

    const results: BenchmarkComparison[] = []

    for (const plugin of selectedPlugins) {
      for (const quality of qualityLevels) {
        setCurrentTest(`Testing ${plugin} at quality ${quality}`)

        try {
          const response = await fetch('/api/benchmark', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              plugin,
              mode: selectedMode,
              quality,
              num_trials: 5
            }),
          })

          if (response.ok) {
            const result: BenchmarkResult = await response.json()

            // Calculate quality score (higher is better)
            let qualityScore = result.avg_compression_ratio * 50
            if (result.avg_snr_db) {
              qualityScore += result.avg_snr_db
            }
            if (result.avg_psnr_db) {
              qualityScore += result.avg_psnr_db * 0.1
            }
            qualityScore -= result.avg_latency_ms * 0.01 // Penalize high latency

            results.push({
              plugin: `${plugin} (q=${quality})`,
              mode: selectedMode,
              compression_ratio: result.avg_compression_ratio,
              latency_ms: result.avg_latency_ms,
              snr_db: result.avg_snr_db,
              psnr_db: result.avg_psnr_db,
              quality_score: qualityScore
            })
          }
        } catch (error) {
          console.error(`Benchmark failed for ${plugin}:`, error)
        }

        completed++
        setProgress((completed / totalTests) * 100)
      }
    }

    setBenchmarkResults(results)
    setIsBenchmarking(false)
    setCurrentTest('')
    setProgress(0)
  }

  const togglePluginSelection = (pluginName: string) => {
    setSelectedPlugins(prev =>
      prev.includes(pluginName)
        ? prev.filter(p => p !== pluginName)
        : [...prev, pluginName]
    )
  }

  const generateSyntheticData = async () => {
    try {
      const response = await fetch('/api/generate-data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          num_channels: 64,
          duration_seconds: 10,
          sampling_rate: 1000
        }),
      })

      if (response.ok) {
        alert('Synthetic data generated successfully')
      }
    } catch (error) {
      console.error('Failed to generate synthetic data:', error)
      alert('Failed to generate synthetic data')
    }
  }

  const formatLatency = (latency: number) => {
    if (latency < 1) return `${(latency * 1000).toFixed(0)}μs`
    return `${latency.toFixed(1)}ms`
  }

  return (
    <div>
      <div className="card">
        <div className="card-header">Benchmark Configuration</div>
        <div className="card-content">
          <div className="row">
            <div className="col-md-6">
              <div className="form-group">
                <label>Select Plugins to Benchmark</label>
                <div className="plugin-selection">
                  {plugins.map(plugin => (
                    <div key={plugin.name} className="plugin-checkbox">
                      <label>
                        <input
                          type="checkbox"
                          checked={selectedPlugins.includes(plugin.name)}
                          onChange={() => togglePluginSelection(plugin.name)}
                        />
                        <span className="ml-2">{plugin.name}</span>
                        <small className="text-muted d-block">
                          {plugin.capabilities.join(', ')}
                        </small>
                      </label>
                    </div>
                  ))}
                </div>
              </div>

              <div className="form-group">
                <label>Compression Mode</label>
                <select
                  value={selectedMode}
                  onChange={(e) => setSelectedMode(e.target.value)}
                  className="form-control"
                >
                  <option value="fast">Fast</option>
                  <option value="balanced">Balanced</option>
                  <option value="quality">Quality</option>
                </select>
              </div>

              <div className="d-flex gap-2">
                <button
                  onClick={runBenchmark}
                  disabled={isBenchmarking || selectedPlugins.length === 0}
                  className="btn btn-primary"
                >
                  {isBenchmarking ? 'Benchmarking...' : 'Run Benchmark'}
                </button>

                <button
                  onClick={generateSyntheticData}
                  className="btn btn-secondary"
                >
                  Generate Test Data
                </button>
              </div>

              {isBenchmarking && (
                <div className="progress-container mt-3">
                  <label>{currentTest}</label>
                  <div className="progress">
                    <div
                      className="progress-bar"
                      style={{ width: `${progress}%` }}
                    >
                      {progress.toFixed(0)}%
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="col-md-6">
              {benchmarkResults.length > 0 && (
                <div className="results-summary">
                  <h4>Quick Summary</h4>
                  <div className="summary-grid">
                    <div className="summary-item">
                      <strong>Best Compression:</strong>
                      <br />
                      {benchmarkResults.reduce((best, current) =>
                        current.compression_ratio > best.compression_ratio ? current : best
                      ).plugin}
                      <br />
                      <small>{benchmarkResults.reduce((best, current) =>
                        current.compression_ratio > best.compression_ratio ? current : best
                      ).compression_ratio.toFixed(2)}x ratio</small>
                    </div>
                    <div className="summary-item">
                      <strong>Fastest:</strong>
                      <br />
                      {benchmarkResults.reduce((fastest, current) =>
                        current.latency_ms < fastest.latency_ms ? current : fastest
                      ).plugin}
                      <br />
                      <small>{formatLatency(benchmarkResults.reduce((fastest, current) =>
                        current.latency_ms < fastest.latency_ms ? current : fastest
                      ).latency_ms)}</small>
                    </div>
                    <div className="summary-item">
                      <strong>Best Quality:</strong>
                      <br />
                      {benchmarkResults.reduce((best, current) =>
                        (current.quality_score || 0) > (best.quality_score || 0) ? current : best
                      ).plugin}
                      <br />
                      <small>Score: {(benchmarkResults.reduce((best, current) =>
                        (current.quality_score || 0) > (best.quality_score || 0) ? current : best
                      ).quality_score || 0).toFixed(1)}</small>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {benchmarkResults.length > 0 && (
        <>
          <div className="card">
            <div className="card-header">Compression Ratio Comparison</div>
            <div className="card-content">
              <div className="chart-container">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={benchmarkResults}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="plugin"
                      angle={-45}
                      textAnchor="end"
                      height={100}
                    />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="compression_ratio" fill="#3498db" name="Compression Ratio" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-header">Latency vs Compression Ratio</div>
            <div className="card-content">
              <div className="chart-container">
                <ResponsiveContainer width="100%" height={300}>
                  <ScatterChart data={benchmarkResults}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="latency_ms"
                      name="Latency (ms)"
                      type="number"
                    />
                    <YAxis
                      dataKey="compression_ratio"
                      name="Compression Ratio"
                      type="number"
                    />
                    <Tooltip
                      formatter={(value, name) => [value, name]}
                      labelFormatter={() => ''}
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          const data = payload[0].payload
                          return (
                            <div className="tooltip">
                              <p>{data.plugin}</p>
                              <p>Ratio: {data.compression_ratio.toFixed(2)}x</p>
                              <p>Latency: {formatLatency(data.latency_ms)}</p>
                              {data.snr_db && <p>SNR: {data.snr_db.toFixed(1)}dB</p>}
                            </div>
                          )
                        }
                        return null
                      }}
                    />
                    <Scatter dataKey="compression_ratio" fill="#e74c3c" />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {benchmarkResults.some(r => r.snr_db) && (
            <div className="card">
              <div className="card-header">Signal Quality (SNR)</div>
              <div className="card-content">
                <div className="chart-container">
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={benchmarkResults.filter(r => r.snr_db)}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        dataKey="plugin"
                        angle={-45}
                        textAnchor="end"
                        height={100}
                      />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="snr_db" fill="#27ae60" name="SNR (dB)" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
