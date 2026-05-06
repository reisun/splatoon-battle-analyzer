"""Tests for the job store module."""

import time

from src.job_store import Job, JobProgress, JobStatus, JobStore


class TestJobProgress:
    """Tests for JobProgress dataclass."""

    def test_defaults(self) -> None:
        p = JobProgress()
        assert p.phase == 0
        assert p.phase_total == 1
        assert p.frames_done == 0
        assert p.frames_total == 0


class TestJob:
    """Tests for Job dataclass."""

    def test_defaults(self) -> None:
        job = Job(job_id="test-id")
        assert job.job_id == "test-id"
        assert job.status == JobStatus.QUEUED
        assert job.result is None
        assert job.error is None
        assert job.started_at is None
        assert job.completed_at is None


class TestJobStore:
    """Tests for JobStore."""

    def test_create_returns_job_with_uuid(self) -> None:
        store = JobStore()
        job = store.create()
        assert len(job.job_id) == 36  # UUID format
        assert job.status == JobStatus.QUEUED

    def test_get_existing_job(self) -> None:
        store = JobStore()
        job = store.create()
        fetched = store.get(job.job_id)
        assert fetched is not None
        assert fetched.job_id == job.job_id

    def test_get_nonexistent_returns_none(self) -> None:
        store = JobStore()
        assert store.get("nonexistent") is None

    def test_mark_running(self) -> None:
        store = JobStore()
        job = store.create()
        store.mark_running(job.job_id)
        fetched = store.get(job.job_id)
        assert fetched is not None
        assert fetched.status == JobStatus.RUNNING
        assert fetched.started_at is not None

    def test_mark_completed(self) -> None:
        store = JobStore()
        job = store.create()
        store.mark_running(job.job_id)
        store.mark_completed(job.job_id, {"key": "value"})
        fetched = store.get(job.job_id)
        assert fetched is not None
        assert fetched.status == JobStatus.COMPLETED
        assert fetched.result == {"key": "value"}
        assert fetched.completed_at is not None

    def test_mark_failed(self) -> None:
        store = JobStore()
        job = store.create()
        store.mark_running(job.job_id)
        store.mark_failed(job.job_id, "something went wrong")
        fetched = store.get(job.job_id)
        assert fetched is not None
        assert fetched.status == JobStatus.FAILED
        assert fetched.error == "something went wrong"
        assert fetched.completed_at is not None

    def test_update_progress(self) -> None:
        store = JobStore()
        job = store.create()
        store.update_progress(job.job_id, phase=1, frames_done=5, frames_total=10)
        fetched = store.get(job.job_id)
        assert fetched is not None
        assert fetched.progress.phase == 1
        assert fetched.progress.frames_done == 5
        assert fetched.progress.frames_total == 10

    def test_cleanup_old_removes_completed_jobs(self) -> None:
        store = JobStore()
        job = store.create()
        store.mark_running(job.job_id)
        store.mark_completed(job.job_id, {"done": True})

        # Force completed_at to be old
        fetched = store.get(job.job_id)
        assert fetched is not None
        fetched.completed_at = time.time() - 7200

        store.cleanup_old(max_age_seconds=3600)
        assert store.get(job.job_id) is None

    def test_cleanup_old_keeps_recent_jobs(self) -> None:
        store = JobStore()
        job = store.create()
        store.mark_running(job.job_id)
        store.mark_completed(job.job_id, {"done": True})

        store.cleanup_old(max_age_seconds=3600)
        assert store.get(job.job_id) is not None

    def test_cleanup_old_keeps_running_jobs(self) -> None:
        store = JobStore()
        job = store.create()
        store.mark_running(job.job_id)

        store.cleanup_old(max_age_seconds=0)
        assert store.get(job.job_id) is not None

    def test_mark_running_nonexistent_is_noop(self) -> None:
        store = JobStore()
        store.mark_running("nonexistent")  # should not raise

    def test_mark_completed_nonexistent_is_noop(self) -> None:
        store = JobStore()
        store.mark_completed("nonexistent", {})  # should not raise

    def test_mark_failed_nonexistent_is_noop(self) -> None:
        store = JobStore()
        store.mark_failed("nonexistent", "err")  # should not raise

    def test_update_progress_nonexistent_is_noop(self) -> None:
        store = JobStore()
        store.update_progress("nonexistent", 1, 1, 10)  # should not raise
