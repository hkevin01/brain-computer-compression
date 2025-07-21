import React, { useEffect, useState } from 'react';
import { fetchAlerts } from './api_client';

// AlertsPanel component for system alerts display
const AlertsPanel: React.FC = () => {
  const [alerts, setAlerts] = useState<any[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await fetchAlerts();
        setAlerts(data);
      } catch (error) {
        console.error('Error fetching alerts:', error);
      }
    };
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      <h3>System Alerts</h3>
      <ul>
        {alerts.map((alert, idx) => (
          <li key={idx}>{alert.type}: {alert.message}</li>
        ))}
      </ul>
    </div>
  );
};

export default AlertsPanel;
