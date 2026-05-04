"""Tests for the FastAPI application."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api import app
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
                description="Intense battle",
            ),
        ]

        response = client.post(
            "/analyze/highlights",
            json={
                "file_path": str(video),
                "start": 60.0,
                "end": 180.0,
                "threshold": 5,
                "max_highlights": 3,
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
