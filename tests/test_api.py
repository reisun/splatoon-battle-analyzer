"""Tests for the FastAPI application."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api import app, job_store
from src.highlight_detector import HighlightSegment

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health(self) -> None:
        """Health check returns ok."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestAnalyzeHighlightsEndpoint:
    """Tests for /analyze/highlights endpoint."""

    def test_file_not_found(self) -> None:
        """Return 404 when video file does not exist."""
        response = client.post(
            "/analyze/highlights",
            json={"file_path": "/nonexistent/video.mp4"},
        )
        assert response.status_code == 404

    @patch("src.api.check_api_key_available", return_value=False)
    def test_cli_unavailable(self, mock_check: MagicMock, tmp_path: pytest.TempPathFactory) -> None:
        """Return 503 when Claude CLI is unavailable."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")
        response = client.post(
            "/analyze/highlights",
            json={"file_path": str(video)},
        )
        assert response.status_code == 503

    @patch("src.api.check_api_key_available", return_value=True)
    @patch("src.api.HighlightDetector.detect")
    def test_successful_detection(
        self, mock_detect: MagicMock, mock_check: MagicMock, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Return highlights on successful detection."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")

        mock_detect.return_value = [
            HighlightSegment(
                start_seconds=100.0,
                end_seconds=130.0,
                peak_intensity=8,
            ),
        ]

        response = client.post(
            "/analyze/highlights",
            json={
                "file_path": str(video),
                "start": 60.0,
                "end": 180.0,
                "max_highlights": 4,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["video"] == "test.mp4"
        assert len(data["highlights"]) == 1
        assert data["highlights"][0]["peak_intensity"] == 8
        assert data["highlights"][0]["start_seconds"] == 100.0

    @patch("src.api.check_api_key_available", return_value=True)
    @patch("src.api.HighlightDetector.detect")
    def test_no_highlights(
        self, mock_detect: MagicMock, mock_check: MagicMock, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Return empty highlights list when none detected."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")

        mock_detect.return_value = []

        response = client.post(
            "/analyze/highlights",
            json={"file_path": str(video)},
        )
        assert response.status_code == 200
        assert response.json()["highlights"] == []

    def test_invalid_request(self) -> None:
        """Return 422 on invalid request body."""
        response = client.post("/analyze/highlights", json={})
        assert response.status_code == 422


class TestCreateHighlightJob:
    """Tests for POST /analyze/highlights/jobs endpoint."""

    def test_file_not_found(self) -> None:
        """Return 404 when video file does not exist."""
        response = client.post(
            "/analyze/highlights/jobs",
            json={"file_path": "/nonexistent/video.mp4"},
        )
        assert response.status_code == 404

    @patch("src.api.check_api_key_available", return_value=False)
    def test_cli_unavailable(self, mock_check: MagicMock, tmp_path: pytest.TempPathFactory) -> None:
        """Return 503 when Claude CLI is unavailable."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")
        response = client.post(
            "/analyze/highlights/jobs",
            json={"file_path": str(video)},
        )
        assert response.status_code == 503

    @patch("src.api.check_api_key_available", return_value=True)
    @patch("src.api._run_job")
    def test_creates_job(
        self, mock_run: MagicMock, mock_check: MagicMock, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Return job_id on successful job creation."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")
        response = client.post(
            "/analyze/highlights/jobs",
            json={"file_path": str(video)},
        )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert len(data["job_id"]) == 36

    def test_invalid_request(self) -> None:
        """Return 422 on invalid request body."""
        response = client.post("/analyze/highlights/jobs", json={})
        assert response.status_code == 422


class TestGetJobStatus:
    """Tests for GET /analyze/highlights/jobs/{job_id} endpoint."""

    def test_job_not_found(self) -> None:
        """Return 404 for nonexistent job."""
        response = client.get("/analyze/highlights/jobs/nonexistent-id")
        assert response.status_code == 404

    def test_queued_job(self) -> None:
        """Return queued status for newly created job."""
        job = job_store.create()
        response = client.get(f"/analyze/highlights/jobs/{job.job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job.job_id
        assert data["status"] == "queued"
        assert data["progress"] is None
        assert data["result"] is None
        assert data["error"] is None

    def test_running_job_with_progress(self) -> None:
        """Return running status with progress info."""
        job = job_store.create()
        job_store.mark_running(job.job_id)
        job_store.update_progress(job.job_id, phase=1, frames_done=5, frames_total=10)

        response = client.get(f"/analyze/highlights/jobs/{job.job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["progress"]["phase"] == 1
        assert data["progress"]["frames_done"] == 5
        assert data["progress"]["frames_total"] == 10
        assert data["started_at"] is not None

    def test_completed_job_with_result(self) -> None:
        """Return completed status with result."""
        job = job_store.create()
        job_store.mark_running(job.job_id)
        result = {
            "video": "test.mp4",
            "model": "claude-sonnet-4-20250514",
            "highlights": [
                {
                    "start_seconds": 100.0,
                    "end_seconds": 130.0,
                    "peak_intensity": 8,
                }
            ],
            "scan_summary": {"total_frames": 10, "battle_frames": 5, "candidate_frames": 1},
        }
        job_store.mark_completed(job.job_id, result)

        response = client.get(f"/analyze/highlights/jobs/{job.job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["result"]["video"] == "test.mp4"
        assert len(data["result"]["highlights"]) == 1
        assert data["progress"] is not None

    def test_failed_job(self) -> None:
        """Return failed status with error message."""
        job = job_store.create()
        job_store.mark_running(job.job_id)
        job_store.mark_failed(job.job_id, "Something went wrong")

        response = client.get(f"/analyze/highlights/jobs/{job.job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert data["error"] == "Something went wrong"
        assert data["result"] is None


class TestCreateMatchScanJob:
    """Tests for POST /analyze/matches/scan/jobs endpoint."""

    def test_file_not_found(self) -> None:
        """Return 404 when video file does not exist."""
        response = client.post(
            "/analyze/matches/scan/jobs",
            json={"file_path": "/nonexistent/video.mp4"},
        )
        assert response.status_code == 404

    @patch("src.api.check_api_key_available", return_value=False)
    def test_cli_unavailable(self, mock_check: MagicMock, tmp_path: pytest.TempPathFactory) -> None:
        """Return 503 when Claude CLI is unavailable."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")
        response = client.post(
            "/analyze/matches/scan/jobs",
            json={"file_path": str(video)},
        )
        assert response.status_code == 503

    @patch("src.api.check_api_key_available", return_value=True)
    @patch("src.api._run_scan_job")
    def test_creates_job(
        self, mock_run: MagicMock, mock_check: MagicMock, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Return job_id on successful job creation."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")
        response = client.post(
            "/analyze/matches/scan/jobs",
            json={"file_path": str(video)},
        )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert len(data["job_id"]) == 36

    def test_invalid_request(self) -> None:
        """Return 422 on invalid request body."""
        response = client.post("/analyze/matches/scan/jobs", json={})
        assert response.status_code == 422


class TestGetMatchScanJobStatus:
    """Tests for GET /analyze/matches/scan/jobs/{job_id} endpoint."""

    def test_job_not_found(self) -> None:
        """Return 404 for nonexistent job."""
        response = client.get("/analyze/matches/scan/jobs/nonexistent-id")
        assert response.status_code == 404

    def test_queued_job(self) -> None:
        """Return queued status for newly created job."""
        job = job_store.create()
        response = client.get(f"/analyze/matches/scan/jobs/{job.job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job.job_id
        assert data["status"] == "queued"
        assert data["progress"] is None
        assert data["result"] is None
        assert data["error"] is None

    def test_running_job_with_progress(self) -> None:
        """Return running status with progress info."""
        job = job_store.create()
        job_store.mark_running(job.job_id)
        job_store.update_progress(job.job_id, phase=1, frames_done=5, frames_total=20)

        response = client.get(f"/analyze/matches/scan/jobs/{job.job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["progress"]["frames_done"] == 5
        assert data["progress"]["frames_total"] == 20
        assert data["started_at"] is not None

    def test_completed_job_with_results(self) -> None:
        """Return completed status with match results."""
        job = job_store.create()
        job_store.mark_running(job.job_id)
        result = {
            "matches": [
                {
                    "start_seconds": 10.0,
                    "duration_seconds": 300,
                    "duration_type": "5min",
                },
                {
                    "start_seconds": 400.0,
                    "duration_seconds": 180,
                    "duration_type": "3min",
                },
            ],
        }
        job_store.mark_completed(job.job_id, result)

        response = client.get(f"/analyze/matches/scan/jobs/{job.job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert len(data["result"]["matches"]) == 2
        assert data["result"]["matches"][0]["duration_type"] == "5min"
        assert data["result"]["matches"][1]["duration_type"] == "3min"
        assert data["progress"] is not None

    def test_failed_job(self) -> None:
        """Return failed status with error message."""
        job = job_store.create()
        job_store.mark_running(job.job_id)
        job_store.mark_failed(job.job_id, "Scan failed")

        response = client.get(f"/analyze/matches/scan/jobs/{job.job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert data["error"] == "Scan failed"
        assert data["result"] is None
