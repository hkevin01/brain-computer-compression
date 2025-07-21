import React, { useEffect, useState } from 'react';
import { fetchMetricsLive } from './api_client';
import { formatMetric } from './format_utils';

// Dashboard component for live metrics display
const Dashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<any>({});

  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await fetchMetricsLive();
        setMetrics(data);
      } catch (error) {
        console.error('Error fetching metrics:', error);
      }
    };
    fetchData();
    const interval = setInterval(fetchData, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      <h2>Live Compression Metrics</h2>
      <ul>
        <li>Compression Ratio: {formatMetric(metrics.compression_ratio)}</li>
        <li>Latency (ms): {formatMetric(metrics.latency_ms)}</li>
        <li>SNR (dB): {formatMetric(metrics.snr_db)}</li>
        <li>Power (mW): {formatMetric(metrics.power_mw)}</li>
      </ul>
    </div>
  );
};

export default Dashboard;
