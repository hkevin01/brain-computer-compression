import React from 'react';
import Dashboard from './Dashboard';
import MetricsPanel from './MetricsPanel';
import AlertsPanel from './AlertsPanel';
import HealthPanel from './HealthPanel';
import LogsPanel from './LogsPanel';

// Main application file integrating dashboard components
const App: React.FC = () => {
  return (
    <div>
      <h1>BCI Compression Dashboard</h1>
      <Dashboard />
      <MetricsPanel />
      <AlertsPanel />
      <HealthPanel />
      <LogsPanel />
    </div>
  );
};

export default App;
