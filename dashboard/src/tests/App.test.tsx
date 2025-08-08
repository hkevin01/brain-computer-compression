import '@testing-library/jest-dom'
import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import App from '../src/App'

// Mock react-router-dom
vi.mock('react-router-dom', () => ({
  BrowserRouter: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Routes: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Route: ({ element }: { element: React.ReactNode }) => <div>{element}</div>,
  Link: ({ children, to }: { children: React.ReactNode; to: string }) => (
    <a href={to}>{children}</a>
  ),
  useLocation: () => ({ pathname: '/' })
}))

// Mock fetch
global.fetch = vi.fn(() =>
  Promise.resolve({
    ok: true,
    json: () => Promise.resolve({ status: 'healthy' })
  })
) as any

describe('App Component', () => {
  it('renders the navigation', () => {
    render(<App />)
    expect(screen.getByText('BCI Compression Dashboard')).toBeInTheDocument()
    expect(screen.getByText('Live Stream')).toBeInTheDocument()
    expect(screen.getByText('File Compression')).toBeInTheDocument()
    expect(screen.getByText('Benchmarks')).toBeInTheDocument()
  })

  it('shows server status', () => {
    render(<App />)
    expect(screen.getByText(/Server:/)).toBeInTheDocument()
  })
})
