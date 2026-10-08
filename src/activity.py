"""Process-local admission tied to actual worker lifetime, not HTTP lifetime."""

import threading
import uuid


class ProcessingActivity:
    """Atomically reserve one operation and publish the current process identity."""

    def __init__(self) -> None:
        self.instance_id = str(uuid.uuid4())
        self._lock = threading.Lock()
        self._operation: dict | None = None

    def claim(self, kind: str) -> str | None:
        with self._lock:
            if self._operation is not None:
                return None
            claim_id = str(uuid.uuid4())
            self._operation = {"claim_id": claim_id, "job_id": None, "kind": kind}
            return claim_id

    def attach(self, claim_id: str, job_id: str) -> None:
        with self._lock:
            if self._operation and self._operation["claim_id"] == claim_id:
                self._operation["job_id"] = job_id

    def release(self, claim_id: str) -> None:
        with self._lock:
            if self._operation and self._operation["claim_id"] == claim_id:
                self._operation = None

    def snapshot(self) -> dict | None:
        with self._lock:
            return dict(self._operation) if self._operation else None
