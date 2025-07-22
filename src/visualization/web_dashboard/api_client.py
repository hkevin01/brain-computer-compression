"""
Dashboard API client for frontend integration.
Fetches metrics, alerts, and config from backend API.

References:
- Backend/frontend communication
"""
import requests

API_BASE = "http://localhost:8000/api"

def get_metrics():
    resp = requests.get(f"{API_BASE}/metrics")
    return resp.json()

def get_alerts():
    resp = requests.get(f"{API_BASE}/alerts")
    return resp.json()

def get_config():
    resp = requests.get(f"{API_BASE}/config")
    return resp.json()
