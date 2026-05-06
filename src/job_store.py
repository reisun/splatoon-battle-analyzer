"""In-memory job store for async highlight detection jobs."""

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobProgress:
    phase: int = 0
    phase_total: int = 1
    frames_done: int = 0
    frames_total: int = 0


@dataclass
class Job:
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    progress: JobProgress = field(default_factory=JobProgress)
    result: dict | None = None
    error: str | None = None
    started_at: float | None = None
    completed_at: float | None = None


class JobStore:
    """In-memory thread-safe job store."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self) -> Job:
        job_id = str(uuid.uuid4())
        job = Job(job_id=job_id)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update_progress(
        self, job_id: str, phase: int, frames_done: int, frames_total: int
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.progress.phase = phase
                job.progress.frames_done = frames_done
                job.progress.frames_total = frames_total

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.RUNNING
                job.started_at = time.time()

    def mark_completed(self, job_id: str, result: dict) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.COMPLETED
                job.result = result
                job.completed_at = time.time()

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.FAILED
                job.error = error
                job.completed_at = time.time()

    def cleanup_old(self, max_age_seconds: float = 3600) -> None:
        now = time.time()
        with self._lock:
            to_remove = [
                jid
                for jid, j in self._jobs.items()
                if j.completed_at and (now - j.completed_at) > max_age_seconds
            ]
            for jid in to_remove:
                del self._jobs[jid]
