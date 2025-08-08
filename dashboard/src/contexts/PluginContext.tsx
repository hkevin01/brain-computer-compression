import React, { createContext, useContext, useEffect, useState } from 'react'
import { PluginInfo } from '../types/api'

interface PluginContextType {
  plugins: PluginInfo[]
  loading: boolean
  error: string | null
  refreshPlugins: () => Promise<void>
}

const PluginContext = createContext<PluginContextType | undefined>(undefined)

export function usePlugins() {
  const context = useContext(PluginContext)
  if (context === undefined) {
    throw new Error('usePlugins must be used within a PluginProvider')
  }
  return context
}

export function PluginProvider({ children }: { children: React.ReactNode }) {
  const [plugins, setPlugins] = useState<PluginInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchPlugins = async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await fetch('http://localhost:8000/api/plugins')
      if (!response.ok) {
        throw new Error(`Failed to fetch plugins: ${response.statusText}`)
      }
      const data = await response.json()
      setPlugins(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch plugins')
      setPlugins([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchPlugins()
  }, [])

  const value = {
    plugins,
    loading,
    error,
    refreshPlugins: fetchPlugins
  }

  return (
    <PluginContext.Provider value={value}>
      {children}
    </PluginContext.Provider>
  )
}
