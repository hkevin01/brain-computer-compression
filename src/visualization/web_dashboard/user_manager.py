"""
UserManager class for authentication and role management.
Supports user creation, authentication, and role assignment.

References:
- Security & compliance requirements (see project_plan.md)
- PEP 8, type hints, and docstring standards
"""
from typing import Dict, Optional
import hashlib

class UserManager:
    """
    Manages users, authentication, and roles for dashboard access control.
    """
    def __init__(self):
        self.users: Dict[str, Dict[str, str]] = {}
        # Example: {"admin": {"password": "hashed_pw", "role": "admin"}}

    def create_user(self, username: str, password: str, role: str = "user") -> None:
        """
        Creates a new user with hashed password and assigned role.
        """
        hashed_pw = hashlib.sha256(password.encode()).hexdigest()
        self.users[username] = {"password": hashed_pw, "role": role}

    def authenticate(self, username: str, password: str) -> bool:
        """
        Authenticates a user by verifying password hash.
        """
        user = self.users.get(username)
        if not user:
            return False
        return user["password"] == hashlib.sha256(password.encode()).hexdigest()

    def get_role(self, username: str) -> Optional[str]:
        """
        Returns the role of the user, or None if not found.
        """
        user = self.users.get(username)
        return user["role"] if user else None
