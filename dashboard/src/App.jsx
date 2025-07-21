import axios from 'axios';
import React, { useEffect, useState } from 'react';
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

const mockMetrics = [
  { time: '10:00', ratio: 2.1 },
  { time: '10:01', ratio: 2.3 },
  { time: '10:02', ratio: 2.5 },
  { time: '10:03', ratio: 2.4 },
  { time: '10:04', ratio: 2.6 },
  { time: '10:05', ratio: 2.7 },
  { time: '10:06', ratio: 2.8 },
];

function LiveMetrics() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let isMounted = true;
    const fetchMetrics = () => {
      axios.get('http://localhost:8000/metrics')
        .then(res => {
          if (isMounted) {
            setMetrics(res.data);
            setLoading(false);
          }
        })
        .catch(() => {
          if (isMounted) {
            setError('Failed to fetch metrics');
            setLoading(false);
          }
        });
    };
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 3000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  if (loading) return <div>Loading live metrics...</div>;
  if (error) return <div style={{ color: 'red' }}>{error}</div>;
  return (
    <div style={{ border: '1px solid #ccc', padding: 16, borderRadius: 8, marginBottom: 32 }}>
      <h2>Live Metrics</h2>
      <ul>
        <li>Compression Ratio: {metrics.compression_ratio}</li>
        <li>Latency: {metrics.latency_ms} ms</li>
        <li>SNR: {metrics.snr_db} dB</li>
        <li>Power Consumption: {metrics.power_mw} mW</li>
      </ul>
    </div>
  );
}

function PluginList({ onSelect, selected }) {
  const [plugins, setPlugins] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    axios.get('http://localhost:8000/plugins')
      .then(res => {
        setPlugins(res.data);
        setLoading(false);
      })
      .catch(err => {
        setError('Failed to fetch plugins');
        setLoading(false);
      });
  }, []);

  if (loading) return <div>Loading plugins...</div>;
  if (error) return <div style={{ color: 'red' }}>{error}</div>;
  return (
    <div style={{ border: '1px solid #ccc', padding: 16, borderRadius: 8, marginBottom: 32 }}>
      <h2>Available Compression Plugins</h2>
      <ul style={{ listStyle: 'none', padding: 0 }}>
        {plugins.map((p) => (
          <li key={p.name} style={{ marginBottom: 8 }}>
            <button
              style={{
                background: selected && selected.name === p.name ? '#e0e7ff' : '#f9f9f9',
                border: '1px solid #888',
                borderRadius: 4,
                padding: '6px 12px',
                cursor: 'pointer',
                fontWeight: selected && selected.name === p.name ? 'bold' : 'normal',
                width: '100%',
                textAlign: 'left',
              }}
              onClick={() => onSelect(p)}
            >
              {p.name}
            </button>
            <div style={{ fontSize: 13, color: '#555', marginTop: 2 }}>{p.doc}</div>
          </li>
        ))}
      </ul>
    </div>
  );
}

function App() {
  const [selectedPlugin, setSelectedPlugin] = useState(null);
  return (
    <div style={{ padding: 32 }}>
      <h1>BCI Compression Dashboard</h1>
      <p>Real-time visualization and monitoring for neural data compression.</p>
      <PluginList onSelect={setSelectedPlugin} selected={selectedPlugin} />
      {selectedPlugin && (
        <div style={{ background: '#e0e7ff', border: '2px solid #6366f1', borderRadius: 8, padding: 16, marginBottom: 32 }}>
          <h3>Selected Plugin: {selectedPlugin.name}</h3>
          <div style={{ fontSize: 15 }}>{selectedPlugin.doc}</div>
        </div>
      )}
      <LiveMetrics />
      <div style={{ marginTop: 32 }}>
        <div style={{ border: '1px solid #ccc', padding: 16, borderRadius: 8, marginBottom: 32 }}>
          <h2>Live Metrics (Placeholder)</h2>
          <ul>
            <li>Compression Ratio: --</li>
            <li>Latency: -- ms</li>
            <li>SNR: -- dB</li>
            <li>Power Consumption: -- mW</li>
          </ul>
        </div>
        <div style={{ border: '1px solid #ccc', padding: 16, borderRadius: 8 }}>
          <h2>Compression Ratio Over Time</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={mockMetrics} margin={{ top: 16, right: 32, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis domain={[2, 3]} label={{ value: 'Ratio', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="ratio" stroke="#8884d8" strokeWidth={2} dot={true} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

export default App;

import React from 'react';

const mockMetrics = [
  { time: '10:00', ratio: 2.1 },
  { time: '10:01', ratio: 2.3 },
  { time: '10:02', ratio: 2.5 },
  { time: '10:03', ratio: 2.4 },
  { time: '10:04', ratio: 2.6 },
  { time: '10:05', ratio: 2.7 },
  { time: '10:06', ratio: 2.8 },
];

function LiveMetrics() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let isMounted = true;
    const fetchMetrics = () => {
      axios.get('http://localhost:8000/metrics')
        .then(res => {
          if (isMounted) {
            setMetrics(res.data);
            setLoading(false);
          }
        })
        .catch(() => {
          if (isMounted) {
            setError('Failed to fetch metrics');
            setLoading(false);
          }
        });
    };
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 3000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  if (loading) return <div>Loading live metrics...</div>;
  if (error) return <div style={{ color: 'red' }}>{error}</div>;
  return (
    <div style={{ border: '1px solid #ccc', padding: 16, borderRadius: 8, marginBottom: 32 }}>
      <h2>Live Metrics</h2>
      <ul>
        <li>Compression Ratio: {metrics.compression_ratio}</li>
        <li>Latency: {metrics.latency_ms} ms</li>
        <li>SNR: {metrics.snr_db} dB</li>
        <li>Power Consumption: {metrics.power_mw} mW</li>
      </ul>
    </div>
  );
}

function PluginList({ onSelect, selected }) {
  const [plugins, setPlugins] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    axios.get('http://localhost:8000/plugins')
      .then(res => {
        setPlugins(res.data);
        setLoading(false);
      })
      .catch(err => {
        setError('Failed to fetch plugins');
        setLoading(false);
      });
  }, []);

  if (loading) return <div>Loading plugins...</div>;
  if (error) return <div style={{ color: 'red' }}>{error}</div>;
  return (
    <div style={{ border: '1px solid #ccc', padding: 16, borderRadius: 8, marginBottom: 32 }}>
      <h2>Available Compression Plugins</h2>
      <ul style={{ listStyle: 'none', padding: 0 }}>
        {plugins.map((p) => (
          <li key={p.name} style={{ marginBottom: 8 }}>
            <button
              style={{
                background: selected && selected.name === p.name ? '#e0e7ff' : '#f9f9f9',
                border: '1px solid #888',
                borderRadius: 4,
                padding: '6px 12px',
                cursor: 'pointer',
                fontWeight: selected && selected.name === p.name ? 'bold' : 'normal',
                width: '100%',
                textAlign: 'left',
              }}
              onClick={() => onSelect(p)}
            >
              {p.name}
            </button>
            <div style={{ fontSize: 13, color: '#555', marginTop: 2 }}>{p.doc}</div>
          </li>
        ))}
      </ul>
    </div>
  );
}

function App() {
  const [selectedPlugin, setSelectedPlugin] = useState(null);
  return (
    <div style={{ padding: 32 }}>
      <h1>BCI Compression Dashboard</h1>
      <p>Real-time visualization and monitoring for neural data compression.</p>
      <PluginList onSelect={setSelectedPlugin} selected={selectedPlugin} />
      {selectedPlugin && (
        <div style={{ background: '#e0e7ff', border: '2px solid #6366f1', borderRadius: 8, padding: 16, marginBottom: 32 }}>
          <h3>Selected Plugin: {selectedPlugin.name}</h3>
          <div style={{ fontSize: 15 }}>{selectedPlugin.doc}</div>
        </div>
      )}
      <LiveMetrics />
      <div style={{ marginTop: 32 }}>
        <div style={{ border: '1px solid #ccc', padding: 16, borderRadius: 8, marginBottom: 32 }}>
          <h2>Live Metrics (Placeholder)</h2>
          <ul>
            <li>Compression Ratio: --</li>
            <li>Latency: -- ms</li>
            <li>SNR: -- dB</li>
            <li>Power Consumption: -- mW</li>
          </ul>
        </div>
        <div style={{ border: '1px solid #ccc', padding: 16, borderRadius: 8 }}>
          <h2>Compression Ratio Over Time</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={mockMetrics} margin={{ top: 16, right: 32, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis domain={[2, 3]} label={{ value: 'Ratio', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="ratio" stroke="#8884d8" strokeWidth={2} dot={true} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

export default App;

import React from 'react';

const mockMetrics = [
  { time: '10:00', ratio: 2.1 },
  { time: '10:01', ratio: 2.3 },
  { time: '10:02', ratio: 2.5 },
  { time: '10:03', ratio: 2.4 },
  { time: '10:04', ratio: 2.6 },
  { time: '10:05', ratio: 2.7 },
  { time: '10:06', ratio: 2.8 },
];

function LiveMetrics() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let isMounted = true;
    const fetchMetrics = () => {
      axios.get('http://localhost:8000/metrics')
        .then(res => {
          if (isMounted) {
            setMetrics(res.data);
            setLoading(false);
          }
        })
        .catch(() => {
          if (isMounted) {
            setError('Failed to fetch metrics');
            setLoading(false);
          }
        });
    };
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 3000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  if (loading) return <div>Loading live metrics...</div>;
  if (error) return <div style={{ color: 'red' }}>{error}</div>;
  return (
    <div style={{ border: '1px solid #ccc', padding: 16, borderRadius: 8, marginBottom: 32 }}>
      <h2>Live Metrics</h2>
      <ul>
        <li>Compression Ratio: {metrics.compression_ratio}</li>
        <li>Latency: {metrics.latency_ms} ms</li>
        <li>SNR: {metrics.snr_db} dB</li>
        <li>Power Consumption: {metrics.power_mw} mW</li>
      </ul>
    </div>
  );
}

function PluginList({ onSelect, selected }) {
  const [plugins, setPlugins] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    axios.get('http://localhost:8000/plugins')
      .then(res => {
        setPlugins(res.data);
        setLoading(false);
      })
      .catch(err => {
        setError('Failed to fetch plugins');
        setLoading(false);
      });
  }, []);

  if (loading) return <div>Loading plugins...</div>;
  if (error) return <div style={{ color: 'red' }}>{error}</div>;
  return (
    <div style={{ border: '1px solid #ccc', padding: 16, borderRadius: 8, marginBottom: 32 }}>
      <h2>Available Compression Plugins</h2>
      <ul style={{ listStyle: 'none', padding: 0 }}>
        {plugins.map((p) => (
          <li key={p.name} style={{ marginBottom: 8 }}>
            <button
              style={{
                background: selected && selected.name === p.name ? '#e0e7ff' : '#f9f9f9',
                border: '1px solid #888',
                borderRadius: 4,
                padding: '6px 12px',
                cursor: 'pointer',
                fontWeight: selected && selected.name === p.name ? 'bold' : 'normal',
                width: '100%',
                textAlign: 'left',
              }}
              onClick={() => onSelect(p)}
            >
              {p.name}
            </button>
            <div style={{ fontSize: 13, color: '#555', marginTop: 2 }}>{p.doc}</div>
          </li>
        ))}
      </ul>
    </div>
  );
}

function App() {
  const [selectedPlugin, setSelectedPlugin] = useState(null);
  return (
    <div style={{ padding: 32 }}>
      <h1>BCI Compression Dashboard</h1>
      <p>Real-time visualization and monitoring for neural data compression.</p>
      <PluginList onSelect={setSelectedPlugin} selected={selectedPlugin} />
      {selectedPlugin && (
        <div style={{ background: '#e0e7ff', border: '2px solid #6366f1', borderRadius: 8, padding: 16, marginBottom: 32 }}>
          <h3>Selected Plugin: {selectedPlugin.name}</h3>
          <div style={{ fontSize: 15 }}>{selectedPlugin.doc}</div>
        </div>
      )}
      <LiveMetrics />
      <div style={{ marginTop: 32 }}>
        <div style={{ border: '1px solid #ccc', padding: 16, borderRadius: 8, marginBottom: 32 }}>
          <h2>Live Metrics (Placeholder)</h2>
          <ul>
            <li>Compression Ratio: --</li>
            <li>Latency: -- ms</li>
            <li>SNR: -- dB</li>
            <li>Power Consumption: -- mW</li>
          </ul>
        </div>
        <div style={{ border: '1px solid #ccc', padding: 16, borderRadius: 8 }}>
          <h2>Compression Ratio Over Time</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={mockMetrics} margin={{ top: 16, right: 32, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis domain={[2, 3]} label={{ value: 'Ratio', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="ratio" stroke="#8884d8" strokeWidth={2} dot={true} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

export default App;

import React from 'react';

const mockMetrics = [
  { time: '10:00', ratio: 2.1 },
  { time: '10:01', ratio: 2.3 },
  { time: '10:02', ratio: 2.5 },
  { time: '10:03', ratio: 2.4 },
  { time: '10:04', ratio: 2.6 },
  { time: '10:05', ratio: 2.7 },
  { time: '10:06', ratio: 2.8 },
];

function LiveMetrics() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let isMounted = true;
    const fetchMetrics = () => {
      axios.get('http://localhost:8000/metrics')
        .then(res => {
          if (isMounted) {
            setMetrics(res.data);
            setLoading(false);
          }
        })
        .catch(() => {
          if (isMounted) {
            setError('Failed to fetch metrics');
            setLoading(false);
          }
        });
    };
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 3000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  if (loading) return <div>Loading live metrics...</div>;
  if (error) return <div style={{ color: 'red' }}>{error}</div>;
  return (
    <div style={{ border: '1px solid #ccc', padding: 16, borderRadius: 8, marginBottom: 32 }}>
      <h2>Live Metrics</h2>
      <ul>
        <li>Compression Ratio: {metrics.compression_ratio}</li>
        <li>Latency: {metrics.latency_ms} ms</li>
        <li>SNR: {metrics.snr_db} dB</li>
        <li>Power Consumption: {metrics.power_mw} mW</li>
      </ul>
    </div>
  );
}

function PluginList({ onSelect, selected }) {
  const [plugins, setPlugins] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    axios.get('http://localhost:8000/plugins')
      .then(res => {
        setPlugins(res.data);
        setLoading(false);
      })
      .catch(err => {
        setError('Failed to fetch plugins');
        setLoading(false);
      });
  }, []);

  if (loading) return <div>Loading plugins...</div>;
  if (error) return <div style={{ color: 'red' }}>{error}</div>;
  return (
    <div style={{ border: '1px solid #ccc', padding: 16, borderRadius: 8, marginBottom: 32 }}>
      <h2>Available Compression Plugins</h2>
      <ul style={{ listStyle: 'none', padding: 0 }}>
        {plugins.map((p) => (
          <li key={p.name} style={{ marginBottom: 8 }}>
            <button
              style={{
                background: selected && selected.name === p.name ? '#e0e7ff' : '#f9f9f9',
                border: '1px solid #888',
                borderRadius: 4,
                padding: '6px 12px',
                cursor: 'pointer',
                fontWeight: selected && selected.name === p.name ? 'bold' : 'normal',
                width: '100%',
                textAlign: 'left',
              }}
              onClick={() => onSelect(p)}
            >
              {p.name}
            </button>
            <div style={{ fontSize: 13, color: '#555', marginTop: 2 }}>{p.doc}</div>
          </li>
        ))}
      </ul>
    </div>
  );
}

function App() {
  const [selectedPlugin, setSelectedPlugin] = useState(null);
  return (
    <div style={{ padding: 32 }}>
      <h1>BCI Compression Dashboard</h1>
      <p>Real-time visualization and monitoring for neural data compression.</p>
      <PluginList onSelect={setSelectedPlugin} selected={selectedPlugin} />
      {selectedPlugin && (
        <div style={{ background: '#e0e7ff', border: '2px solid #6366f1', borderRadius: 8, padding: 16, marginBottom: 32 }}>
          <h3>Selected Plugin: {selectedPlugin.name}</h3>
          <div style={{ fontSize: 15 }}>{selectedPlugin.doc}</div>
        </div>
      )}
      <LiveMetrics />
      <div style={{ marginTop: 32 }}>
        <div style={{ border: '1px solid #ccc', padding: 16, borderRadius: 8, marginBottom: 32 }}>
          <h2>Live Metrics (Placeholder)</h2>
          <ul>
            <li>Compression Ratio: --</li>
            <li>Latency: -- ms</li>
            <li>SNR: -- dB</li>
            <li>Power Consumption: -- mW</li>
          </ul>
        </div>
        <div style={{ border: '1px solid #ccc', padding: 16, borderRadius: 8 }}>
          <h2>Compression Ratio Over Time</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={mockMetrics} margin={{ top: 16, right: 32, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis domain={[2, 3]} label={{ value: 'Ratio', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="ratio" stroke="#8884d8" strokeWidth={2} dot={true} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

export default App;

import React from 'react';

const mockMetrics = [
  { time: '10:00', ratio: 2.1 },
  { time: '10:01', ratio: 2.3 },
  { time: '10:02', ratio: 2.5 },
  { time: '10:03', ratio: 2.4 },
  { time: '10:04', ratio: 2.6 },
  { time: '10:05', ratio: 2.7 },
  { time: '10:06', ratio: 2.8 },
];

function LiveMetrics() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let isMounted = true;
    const fetchMetrics = () => {
      axios.get('http://localhost:8000/metrics')
        .then(res => {
          if (isMounted) {
            setMetrics(res.data);
            setLoading(false);
          }
        })
        .catch(() => {
          if (isMounted) {
            setError('Failed to fetch metrics');
            setLoading(false);
          }
        });
    };
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 3000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  if (loading) return <div>Loading live metrics...</div>;
  if (error) return <div style={{ color: 'red' }}>{error}</div>;
  return (
    <div style={{ border: '1px solid #ccc', padding: 16, borderRadius: 8, marginBottom: 32 }}>
      <h2>Live Metrics</h2>
      <ul>
        <li>Compression Ratio: {metrics.compression_ratio}</li>
        <li>Latency: {metrics.latency_ms} ms</li>
        <li>SNR: {metrics.snr_db} dB</li>
        <li>Power Consumption: {metrics.power_mw} mW</li>
      </ul>
    </div>
  );
}

function PluginList({ onSelect, selected }) {
  const [plugins, setPlugins] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    axios.get('http://localhost:8000/plugins')
      .then(res => {
        setPlugins(res.data);
        setLoading(false);
      })
      .catch(err => {
        setError('Failed to fetch plugins');
        setLoading(false);
      });
  }, []);

  if (loading) return <div>Loading plugins...</div>;
  if (error) return <div style={{ color: 'red' }}>{error}</div>;
  return (
    <div style={{ border: '1px solid #ccc', padding: 16, borderRadius: 8, marginBottom: 32 }}>
      <h2>Available Compression Plugins</h2>
      <ul style={{ listStyle: 'none', padding: 0 }}>
        {plugins.map((p) => (
          <li key={p.name} style={{ marginBottom: 8 }}>
            <button
              style={{
                background: selected && selected.name === p.name ? '#e0e7ff' : '#f9f9f9',
                border: '1px solid #888',
                borderRadius: 4,
                padding: '6px 12px',
                cursor: 'pointer',
                fontWeight: selected && selected.name === p.name ? 'bold' : 'normal',
                width: '100%',
                textAlign: 'left',
              }}
              onClick={() => onSelect(p)}
            >
              {p.name}
            </button>
            <div style={{ fontSize: 13, color: '#555', marginTop: 2 }}>{p.doc}</div>
          </li>
        ))}
      </ul>
    </div>
  );
}

function App() {
  const [selectedPlugin, setSelectedPlugin] = useState(null);
  return (
    <div style={{ padding: 32 }}>
      <h1>BCI Compression Dashboard</h1>
      <p>Real-time visualization and monitoring for neural data compression.</p>
      <PluginList onSelect={setSelectedPlugin} selected={selectedPlugin} />
      {selectedPlugin && (
        <div style={{ background: '#e0e7ff', border: '2px solid #6366f1', borderRadius: 8, padding: 16, marginBottom: 32 }}>
          <h3>Selected Plugin: {selectedPlugin.name}</h3>
          <div style={{ fontSize: 15 }}>{selectedPlugin.doc}</div>
        </div>
      )}
      <LiveMetrics />
      <div style={{ marginTop: 32 }}>
        <div style={{ border: '1px solid #ccc', padding: 16, borderRadius: 8, marginBottom: 32 }}>
          <h2>Live Metrics (Placeholder)</h2>
          <ul>
            <li>Compression Ratio: --</li>
            <li>Latency: -- ms</li>
            <li>SNR: -- dB</li>
            <li>Power Consumption: -- mW</li>
          </ul>
        </div>
        <div style={{ border: '1px solid #ccc', padding: 16, borderRadius: 8 }}>
          <h2>Compression Ratio Over Time</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={mockMetrics} margin={{ top: 16, right: 32, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis domain={[2, 3]} label={{ value: 'Ratio', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="ratio" stroke="#8884d8" strokeWidth={2} dot={true} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

export default App;
