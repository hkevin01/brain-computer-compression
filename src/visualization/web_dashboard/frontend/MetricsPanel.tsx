import React, { useEffect, useState } from 'react';
import { fetchMetricsAverage } from './api_client';
import { formatMetric } from './format_utils';

// MetricsPanel component for average metrics display
const MetricsPanel: React.FC = () => {
  const [metrics, setMetrics] = useState<any>({});

  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await fetchMetricsAverage();
        setMetrics(data);
      } catch (error) {
        console.error('Error fetching average metrics:', error);
      }
    };
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      <h3>Average Compression Metrics</h3>
      <ul>
        <li>Compression Ratio: {formatMetric(metrics.compression_ratio)}</li>
        <li>Latency (ms): {formatMetric(metrics.latency_ms)}</li>
        <li>SNR (dB): {formatMetric(metrics.snr_db)}</li>
        <li>Power (mW): {formatMetric(metrics.power_mw)}</li>
      </ul>
    </div>
  );
};

export default MetricsPanel;
