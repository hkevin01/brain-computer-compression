# BCI Dashboard Frontend Integration

This folder will contain the frontend code for the real-time dashboard.

## Recommended Stack
- React (TypeScript) or Dash (Python)
- WebSocket integration for live metrics
- Modular components: Metrics, Alerts, Health, Logs

## Integration Steps
1. Start backend API (see api.py)
2. Connect frontend to API endpoints (/metrics/live, /alerts, /health, /logs)
3. Implement real-time updates and visualization
4. Add authentication and access control as needed

## Architecture Diagram
```
[Compression Pipeline] <--> [FastAPI Backend] <--> [Frontend (React/Dash)] <--> [User]
```

See project_plan.md for more details.

## Security & Compliance Workflow

### Audit Logging
- All access and modification events are logged using `audit_utils.py`.
- Audit logs include user, event type, and details for compliance.

### Access Control
- Roles and permissions are defined in `access_control.yaml`.
- Sensitive endpoints (e.g., /logs) require appropriate permissions.
- Unauthorized access attempts are logged and denied.

### Next Steps
- Integrate role-based authentication for API endpoints.
- Expand audit logging to all sensitive actions.
- Validate compliance with HIPAA/GDPR/FDA as outlined in project_plan.md.
