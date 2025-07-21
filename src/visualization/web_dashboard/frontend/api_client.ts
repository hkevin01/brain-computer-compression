"""
API client utility for frontend-backend communication.
Provides functions to fetch metrics, alerts, health, and logs from backend API.

References:
- API endpoints in dashboard backend
- PEP 8, type hints, and docstring standards
"""
export async function fetchMetricsLive(): Promise<any> {
  const response = await fetch('/metrics/live');
  return response.json();
}

export async function fetchMetricsAverage(): Promise<any> {
  const response = await fetch('/metrics/average');
  return response.json();
}

export async function fetchAlerts(): Promise<any> {
  const response = await fetch('/alerts');
  return response.json();
}

export async function fetchHealth(): Promise<any> {
  const response = await fetch('/health');
  return response.json();
}

export async function fetchLogs(): Promise<any> {
  const response = await fetch('/logs');
  return response.json();
}
