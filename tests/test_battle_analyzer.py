"""Tests for the battle analysis module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.battle_analyzer import BattleAnalyzer, check_api_key_available


class TestBattleAnalyzer:
    """Tests for BattleAnalyzer class."""

    def test_init_default_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Initialize with default Ollama base URL."""
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        analyzer = BattleAnalyzer()
        assert analyzer.base_url == "http://ollama:11434"

    def test_init_with_env_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Initialize with base URL from environment variable."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://custom:11434")
        analyzer = BattleAnalyzer()
        assert analyzer.base_url == "http://custom:11434"

    def test_init_with_explicit_base_url(self) -> None:
        """Initialize with explicitly provided base URL."""
        analyzer = BattleAnalyzer(base_url="http://localhost:11434")
        assert analyzer.base_url == "http://localhost:11434"

    def test_init_concurrency_default(self) -> None:
        """Default concurrency is 1."""
        analyzer = BattleAnalyzer()
        assert analyzer.concurrency == 1

    def test_init_concurrency_custom(self) -> None:
        """Custom concurrency value."""
        analyzer = BattleAnalyzer(concurrency=8)
        assert analyzer.concurrency == 8

    def test_analyze_frame_file_not_found(self) -> None:
        """Raise FileNotFoundError when image does not exist."""
        analyzer = BattleAnalyzer()
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            analyzer.analyze_frame("/nonexistent/frame.jpg")

    @patch("src.battle_analyzer.requests.post")
    def test_analyze_frame_success(self, mock_post: MagicMock, tmp_path: Path) -> None:
        """Successfully analyze a frame image."""
        image_file = tmp_path / "frame_00m10s.jpg"
        image_file.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-data")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "Game Mode: Turf War\nScore: 45% vs 55%"}
        }
        mock_post.return_value = mock_response

        analyzer = BattleAnalyzer(base_url="http://localhost:11434")
        result = analyzer.analyze_frame(image_file)

        assert "Turf War" in result
        assert "Score" in result
        mock_post.assert_called_once()

        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"]
        assert payload["model"] == "llava-llama3"
        assert payload["stream"] is False
        assert len(payload["messages"][0]["images"]) == 1

    @patch("src.battle_analyzer.requests.post")
    def test_analyze_frame_from_memory(self, mock_post: MagicMock) -> None:
        """Analyze a frame from memory (numpy array)."""
        import numpy as np

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": {"content": "Analysis from memory"}}
        mock_post.return_value = mock_response

        analyzer = BattleAnalyzer(base_url="http://localhost:11434")
        result = analyzer.analyze_frame_from_memory(frame, "01m30s")

        assert result == "Analysis from memory"
        mock_post.assert_called_once()

    @patch("src.battle_analyzer.requests.post")
    def test_analyze_frames_multiple(self, mock_post: MagicMock, tmp_path: Path) -> None:
        """Analyze multiple frames and return structured results."""
        paths = []
        for name in ["frame_00m00s.jpg", "frame_00m10s.jpg", "frame_00m20s.jpg"]:
            p = tmp_path / name
            p.write_bytes(b"\xff\xd8\xff\xe0fake")
            paths.append(p)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": {"content": "Battle analysis"}}
        mock_post.return_value = mock_response

        analyzer = BattleAnalyzer(base_url="http://localhost:11434")
        results = analyzer.analyze_frames(paths)

        assert len(results) == 3
        assert results[0]["timestamp"] == "00m00s"
        assert results[1]["timestamp"] == "00m10s"
        assert results[2]["timestamp"] == "00m20s"
        assert all(r["analysis"] == "Battle analysis" for r in results)

    @patch("src.battle_analyzer.requests.post")
    def test_analyze_frames_handles_error(self, mock_post: MagicMock, tmp_path: Path) -> None:
        """Handle analysis errors gracefully for individual frames."""
        good_frame = tmp_path / "frame_00m00s.jpg"
        good_frame.write_bytes(b"\xff\xd8\xff\xe0fake")

        bad_frame = tmp_path / "frame_00m10s.jpg"
        bad_frame.write_bytes(b"\xff\xd8\xff\xe0fake")

        good_response = MagicMock()
        good_response.status_code = 200
        good_response.json.return_value = {"message": {"content": "OK"}}
        good_response.raise_for_status.return_value = None

        bad_response = MagicMock()
        bad_response.raise_for_status.side_effect = Exception("API error")

        mock_post.side_effect = [good_response, bad_response]

        analyzer = BattleAnalyzer(base_url="http://localhost:11434", concurrency=1)
        results = analyzer.analyze_frames([good_frame, bad_frame])

        assert len(results) == 2
        assert results[0]["analysis"] == "OK"
        assert "[Error]" in results[1]["analysis"]

    @patch("src.battle_analyzer.requests.post")
    def test_analyze_frames_preserves_order(self, mock_post: MagicMock, tmp_path: Path) -> None:
        """Results are returned in the same order as input paths."""
        paths = []
        for name in ["frame_00m00s.jpg", "frame_00m10s.jpg", "frame_00m20s.jpg"]:
            p = tmp_path / name
            p.write_bytes(b"\xff\xd8\xff\xe0fake")
            paths.append(p)

        call_count = [0]

        def mock_post_fn(*args, **kwargs):
            call_count[0] += 1
            response = MagicMock()
            response.status_code = 200
            response.json.return_value = {"message": {"content": f"Result {call_count[0]}"}}
            response.raise_for_status.return_value = None
            return response

        mock_post.side_effect = mock_post_fn

        analyzer = BattleAnalyzer(base_url="http://localhost:11434", concurrency=1)
        results = analyzer.analyze_frames(paths)

        # With concurrency=1, order should be deterministic
        assert len(results) == 3
        assert results[0]["timestamp"] == "00m00s"
        assert results[1]["timestamp"] == "00m10s"
        assert results[2]["timestamp"] == "00m20s"


class TestCheckApiKeyAvailable:
    """Tests for check_api_key_available function."""

    @patch("src.battle_analyzer.requests.get")
    def test_ollama_reachable(self, mock_get: MagicMock) -> None:
        """Return True when Ollama is reachable."""
        mock_get.return_value = MagicMock(status_code=200)
        assert check_api_key_available() is True

    @patch("src.battle_analyzer.requests.get")
    def test_ollama_not_reachable(self, mock_get: MagicMock) -> None:
        """Return False when Ollama is not reachable."""
        import requests as req

        mock_get.side_effect = req.ConnectionError("Connection refused")
        assert check_api_key_available() is False

    @patch("src.battle_analyzer.requests.get")
    def test_ollama_timeout(self, mock_get: MagicMock) -> None:
        """Return False when Ollama times out."""
        import requests as req

        mock_get.side_effect = req.Timeout("Timed out")
        assert check_api_key_available() is False

    @patch("src.battle_analyzer.requests.get")
    def test_ollama_error_status(self, mock_get: MagicMock) -> None:
        """Return False when Ollama returns non-200 status."""
        mock_get.return_value = MagicMock(status_code=500)
        assert check_api_key_available() is False
