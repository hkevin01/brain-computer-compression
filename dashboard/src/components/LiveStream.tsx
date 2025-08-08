import { useEffect, useRef, useState } from 'react'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { MetricsTelemetry } from '../types/api'
import PluginPicker from './PluginPicker'

interface ChartData {
  timestamp: number
  compression_ratio: number
  latency_ms: number
  snr_db?: number
}

export default function LiveStream() {
  const [selectedPlugin, setSelectedPlugin] = useState('')
  const [selectedMode, setSelectedMode] = useState('balanced')
  const [quality, setQuality] = useState(0.8)
  const [isStreaming, setIsStreaming] = useState(false)
  const [metrics, setMetrics] = useState<ChartData[]>([])
  const [currentMetrics, setCurrentMetrics] = useState<MetricsTelemetry | null>(null)
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected')

  const wsRef = useRef<WebSocket | null>(null)
  const sessionIdRef = useRef<string | null>(null)

  const connectWebSocket = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    setConnectionStatus('connecting')
    const ws = new WebSocket('ws://localhost:8000/ws/metrics')

    ws.onopen = () => {
      setConnectionStatus('connected')
      console.log('WebSocket connected')
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.type === 'metrics') {
        const telemetry: MetricsTelemetry = data.data
        setCurrentMetrics(telemetry)

        // Add to chart data (keep last 50 points)
        setMetrics(prev => {
          const newData = {
            timestamp: telemetry.timestamp,
            compression_ratio: telemetry.compression_ratio,
            latency_ms: telemetry.latency_ms,
            snr_db: telemetry.snr_db
          }
          const updated = [...prev, newData]
          return updated.slice(-50) // Keep last 50 points
        })
      }
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      setConnectionStatus('disconnected')
    }

    ws.onclose = () => {
      setConnectionStatus('disconnected')
      console.log('WebSocket closed')
    }

    wsRef.current = ws
  }

  const startStreaming = () => {
    if (!selectedPlugin) {
      alert('Please select a plugin first')
      return
    }

    connectWebSocket()

    // Wait for connection then start
    const startTimer = setTimeout(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        const message = {
          command: 'start_stream',
          plugin: selectedPlugin,
          mode: selectedMode,
          quality: quality
        }
        wsRef.current.send(JSON.stringify(message))
        setIsStreaming(true)
        setMetrics([]) // Clear previous data
      }
    }, 500)

    return () => clearTimeout(startTimer)
  }

  const stopStreaming = () => {
    if (sessionIdRef.current && wsRef.current?.readyState === WebSocket.OPEN) {
      const message = {
        command: 'stop_stream',
        session_id: sessionIdRef.current
      }
      wsRef.current.send(JSON.stringify(message))
    }
    setIsStreaming(false)
    sessionIdRef.current = null
  }

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  useEffect(() => {
    if (currentMetrics && !sessionIdRef.current) {
      sessionIdRef.current = currentMetrics.session_id
    }
  }, [currentMetrics])

  return (
    <div>
      <div className="card">
        <div className="card-header">Live Stream Compression</div>
        <div className="card-content">
          <div className="row">
            <div className="col-md-6">
              <PluginPicker
                selectedPlugin={selectedPlugin}
                onPluginChange={setSelectedPlugin}
                selectedMode={selectedMode}
                onModeChange={setSelectedMode}
              />

              <div className="form-group">
                <label htmlFor="quality-slider">Quality Level: {quality.toFixed(2)}</label>
                <input
                  id="quality-slider"
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.1"
                  value={quality}
                  onChange={(e) => setQuality(parseFloat(e.target.value))}
                  className="form-control"
                />
              </div>

              <div className="d-flex gap-2 align-center">
                <button
                  onClick={startStreaming}
                  disabled={isStreaming || !selectedPlugin || connectionStatus !== 'connected'}
                  className="btn btn-success"
                >
                  {isStreaming ? 'Streaming...' : 'Start Stream'}
                </button>

                <button
                  onClick={stopStreaming}
                  disabled={!isStreaming}
                  className="btn btn-danger"
                >
                  Stop Stream
                </button>

                <span className={`status ${
                  connectionStatus === 'connected' ? 'status-success' :
                  connectionStatus === 'connecting' ? 'status-warning' : 'status-danger'
                }`}>
                  WebSocket: {connectionStatus}
                </span>
              </div>
            </div>

            <div className="col-md-6">
              {currentMetrics && (
                <div className="metrics-grid">
                  <div className="metric-card">
                    <div className="metric-value">{currentMetrics.compression_ratio.toFixed(2)}x</div>
                    <div className="metric-label">Compression Ratio</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-value">{currentMetrics.latency_ms.toFixed(1)}ms</div>
                    <div className="metric-label">Latency</div>
                  </div>
                  {currentMetrics.snr_db && (
                    <div className="metric-card">
                      <div className="metric-value">{currentMetrics.snr_db.toFixed(1)}dB</div>
                      <div className="metric-label">SNR</div>
                    </div>
                  )}
                  <div className="metric-card">
                    <div className="metric-value">{currentMetrics.gpu_available ? 'Yes' : 'No'}</div>
                    <div className="metric-label">GPU Available</div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {metrics.length > 0 && (
        <div className="card">
          <div className="card-header">Real-Time Metrics</div>
          <div className="card-content">
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={metrics}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="timestamp"
                    type="number"
                    scale="time"
                    domain={['dataMin', 'dataMax']}
                    tickFormatter={(value) => new Date(value * 1000).toLocaleTimeString()}
                  />
                  <YAxis yAxisId="ratio" orientation="left" />
                  <YAxis yAxisId="latency" orientation="right" />
                  <Tooltip
                    labelFormatter={(value) => new Date(value * 1000).toLocaleTimeString()}
                    formatter={(value, name) => [
                      typeof value === 'number' ? value.toFixed(2) : value,
                      name
                    ]}
                  />
                  <Line
                    yAxisId="ratio"
                    type="monotone"
                    dataKey="compression_ratio"
                    stroke="#3498db"
                    name="Compression Ratio"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    yAxisId="latency"
                    type="monotone"
                    dataKey="latency_ms"
                    stroke="#e74c3c"
                    name="Latency (ms)"
                    strokeWidth={2}
                    dot={false}
                  />
                  {metrics.some(m => m.snr_db) && (
                    <Line
                      yAxisId="ratio"
                      type="monotone"
                      dataKey="snr_db"
                      stroke="#27ae60"
                      name="SNR (dB)"
                      strokeWidth={2}
                      dot={false}
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
