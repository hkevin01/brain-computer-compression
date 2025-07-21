import React, { useEffect, useState } from 'react';
import { fetchLogs } from './api_client';
import { formatLog } from './format_utils';

// LogsPanel component for recent log events display
const LogsPanel: React.FC = () => {
  const [logs, setLogs] = useState<any[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await fetchLogs();
        setLogs(data);
      } catch (error) {
        console.error('Error fetching logs:', error);
      }
    };
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      <h3>Recent Log Events</h3>
      <ul>
        {logs.map((log, idx) => (
          <li key={idx}>{formatLog(log)}</li>
        ))}
      </ul>
    </div>
  );
};

export default LogsPanel;
