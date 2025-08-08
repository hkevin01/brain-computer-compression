import { usePlugins } from '../contexts/PluginContext'
import { PluginInfo } from '../types/api'

interface PluginPickerProps {
  selectedPlugin: string
  onPluginChange: (plugin: string) => void
  selectedMode?: string
  onModeChange?: (mode: string) => void
  filter?: (plugin: PluginInfo) => boolean
}

export default function PluginPicker({
  selectedPlugin,
  onPluginChange,
  selectedMode,
  onModeChange,
  filter
}: PluginPickerProps) {
  const { plugins, loading, error } = usePlugins()

  if (loading) {
    return <div>Loading plugins...</div>
  }

  if (error) {
    return <div className="text-danger">Error: {error}</div>
  }

  const filteredPlugins = filter ? plugins.filter(filter) : plugins
  const currentPlugin = plugins.find(p => p.name === selectedPlugin)

  return (
    <div>
      <div className="form-group">
        <label htmlFor="plugin-select">Compression Plugin</label>
        <select
          id="plugin-select"
          className="form-select"
          value={selectedPlugin}
          onChange={(e) => onPluginChange(e.target.value)}
        >
          <option value="">Select a plugin...</option>
          {filteredPlugins.map((plugin) => (
            <option key={plugin.name} value={plugin.name}>
              {plugin.name} ({plugin.is_lossless ? 'Lossless' : 'Lossy'})
              {plugin.supports_gpu ? ' [GPU]' : ''}
            </option>
          ))}
        </select>
      </div>

      {currentPlugin && currentPlugin.modes.length > 1 && onModeChange && (
        <div className="form-group">
          <label htmlFor="mode-select">Mode</label>
          <select
            id="mode-select"
            className="form-select"
            value={selectedMode || currentPlugin.modes[0]}
            onChange={(e) => onModeChange(e.target.value)}
          >
            {currentPlugin.modes.map((mode) => (
              <option key={mode} value={mode}>
                {mode}
              </option>
            ))}
          </select>
        </div>
      )}

      {currentPlugin && (
        <div className="mb-2">
          <small className="text-muted">
            <strong>Capabilities:</strong>
            {currentPlugin.is_lossless && <span className="status status-success">Lossless</span>}
            {currentPlugin.is_lossy && <span className="status status-warning">Lossy</span>}
            {currentPlugin.supports_streaming && <span className="status status-success">Streaming</span>}
            {currentPlugin.supports_gpu && <span className="status status-success">GPU</span>}
          </small>
        </div>
      )}
    </div>
  )
}
