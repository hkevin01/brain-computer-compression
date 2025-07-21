import React, { useEffect, useState } from 'react';
import { fetch } from 'cross-fetch';

// PerformancePanel component for backend performance metrics
const PerformancePanel: React.FC = () => {
  const [metrics, setMetrics] = useState<any>({});

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('/performance');
        const data = await response.json();
        setMetrics(data);
      } catch (error) {
        // Optionally handle error
      }
    };
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      <h3>Backend Performance</h3>
      <ul>
        <li>Average Latency (ms): {metrics.avg_latency_ms}</li>
        <li>Average Throughput (req/s): {metrics.avg_throughput_rps}</li>
      </ul>
    </div>
  );
};

export default PerformancePanel;
