"""
Pipeline status monitoring and reporting utility.
Tracks pipeline state, errors, and progress.

References:
- Status monitoring
- Error reporting
"""
from typing import Dict, Any

class PipelineStatus:
    def __init__(self):
        self.state = "initialized"
        self.errors = []
        self.progress = 0.0

    def set_state(self, state: str):
        self.state = state

    def add_error(self, error: str):
        self.errors.append(error)

    def set_progress(self, progress: float):
        self.progress = progress

    def get_status(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "errors": self.errors,
            "progress": self.progress
        }
