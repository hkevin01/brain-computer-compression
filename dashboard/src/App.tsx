import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { Link, Route, BrowserRouter as Router, Routes, useLocation } from 'react-router-dom'

// Import components
import Benchmarks from './components/Benchmarks'
import FileCompression from './components/FileCompression'
import LiveStream from './components/LiveStream'
import { PluginProvider } from './contexts/PluginContext'

const queryClient = new QueryClient()

function Navigation() {
  const location = useLocation()

  return (
    <nav className="navbar">
      <h1>BCI Compression Dashboard</h1>
      <div className="nav-links">
        <Link
          to="/"
          className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}
        >
          Live Stream
        </Link>
        <Link
          to="/compression"
          className={`nav-link ${location.pathname === '/compression' ? 'active' : ''}`}
        >
          File Compression
        </Link>
        <Link
          to="/benchmarks"
          className={`nav-link ${location.pathname === '/benchmarks' ? 'active' : ''}`}
        >
          Benchmarks
        </Link>
      </div>
    </nav>
  )
}

function App() {
  const [serverStatus, setServerStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking')

  useEffect(() => {
    // Check server health on startup
    const checkHealth = async () => {
      try {
        const response = await fetch('http://localhost:8000/health')
        if (response.ok) {
          setServerStatus('connected')
        } else {
          setServerStatus('disconnected')
        }
      } catch (error) {
        setServerStatus('disconnected')
      }
    }

    checkHealth()
    const interval = setInterval(checkHealth, 30000) // Check every 30 seconds

    return () => clearInterval(interval)
  }, [])

  return (
    <QueryClientProvider client={queryClient}>
      <PluginProvider>
        <Router>
          <div className="app">
            <Navigation />

            {/* Server status indicator */}
            <div style={{
              padding: '0.5rem 2rem',
              background: serverStatus === 'connected' ? '#d4edda' : '#f8d7da',
              color: serverStatus === 'connected' ? '#155724' : '#721c24',
              fontSize: '0.875rem'
            }}>
              Server Status: {
                serverStatus === 'checking' ? 'Checking...' :
                serverStatus === 'connected' ? 'Connected' :
                'Disconnected - Make sure the FastAPI server is running on port 8000'
              }
            </div>

            <main className="main-content">
              <Routes>
                <Route path="/" element={<LiveStream />} />
                <Route path="/compression" element={<FileCompression />} />
                <Route path="/benchmarks" element={<Benchmarks />} />
              </Routes>
            </main>
          </div>
        </Router>
      </PluginProvider>
    </QueryClientProvider>
  )
}

export default App
