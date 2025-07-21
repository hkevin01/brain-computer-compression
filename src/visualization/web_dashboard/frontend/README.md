# BCI Dashboard Frontend (React)

This folder contains the React frontend for real-time dashboard visualization.

## Setup Instructions
1. Install dependencies: `npm install`
2. Start development server: `npm start`
3. Connect to backend API endpoints for live metrics, alerts, health, logs

## Component Outline
- Dashboard.tsx: Main dashboard view
- MetricsPanel.tsx: Displays live and average metrics
- AlertsPanel.tsx: Shows system alerts
- HealthPanel.tsx: System health metrics
- LogsPanel.tsx: Recent log events

## API Integration
- `/metrics/live` for live metrics
- `/metrics/average` for average metrics
- `/alerts` for alerts
- `/health` for health metrics
- `/logs` for log events

See project_plan.md for more details.
