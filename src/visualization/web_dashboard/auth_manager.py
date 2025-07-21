"""
AuthManager for user authentication and session management.
Provides methods for login, logout, and session validation.

References:
- Authentication & security (see project_plan.md)
- PEP 8, type hints, and docstring standards
"""
from typing import Dict, Optional
import hashlib
import uuid

class AuthManager:
    """
    Manages user authentication and sessions for dashboard access.
    """
    def __init__(self):
        self.users: Dict[str, str] = {"admin": hashlib.sha256("adminpass".encode()).hexdigest()}
        self.sessions: Dict[str, str] = {}  # session_id -> username

    def login(self, username: str, password: str) -> Optional[str]:
        hashed_pw = hashlib.sha256(password.encode()).hexdigest()
        if self.users.get(username) == hashed_pw:
            session_id = str(uuid.uuid4())
            self.sessions[session_id] = username
            return session_id
        return None

    def logout(self, session_id: str) -> None:
        if session_id in self.sessions:
            del self.sessions[session_id]

    def validate_session(self, session_id: str) -> bool:
        return session_id in self.sessions

    def get_user(self, session_id: str) -> Optional[str]:
        return self.sessions.get(session_id)
