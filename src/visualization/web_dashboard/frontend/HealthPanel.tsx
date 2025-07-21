import React, { useEffect, useState } from 'react';
import { fetchHealth } from './api_client';
import { formatMetric } from './format_utils';

// HealthPanel component for system health metrics display
const HealthPanel: React.FC = () => {
  const [health, setHealth] = useState<any>({});

  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await fetchHealth();
        setHealth(data);
      } catch (error) {
        console.error('Error fetching health metrics:', error);
      }
    };
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      <h3>System Health</h3>
      <ul>
        <li>Memory Usage (MB): {formatMetric(health.memory_usage_mb)}</li>
        <li>GPU Utilization (%): {formatMetric(health.gpu_utilization_pct)}</li>
        <li>Error Rate (%): {formatMetric(health.error_rate_pct)}</li>
      </ul>
    </div>
  );
};

export default HealthPanel;
