"""
PermissionChecker utility for endpoint access validation.
Checks if a user has permission for a given action based on role.

References:
- access_control.yaml
- PEP 8, type hints, and docstring standards
"""
from typing import Dict, List
import yaml

class PermissionChecker:
    """
    Checks user permissions for dashboard API endpoints.
    """
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.role_permissions: Dict[str, List[str]] = {
            role["name"]: role["permissions"] for role in self.config["roles"]
        }

    def has_permission(self, role: str, permission: str) -> bool:
        """
        Returns True if the role has the specified permission.
        """
        return permission in self.role_permissions.get(role, [])
