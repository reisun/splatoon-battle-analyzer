"""Tests for the battle analysis module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.battle_analyzer import BattleAnalyzer, check_api_key_available


class TestBattleAnalyzer:
    """Tests for BattleAnalyzer class."""

    def test_init_with_explicit_key(self) -> None:
        """Initialize with an explicitly provided API key."""
        analyzer = BattleAnalyzer(api_key="test-key-123")
        assert analyzer.api_key == "test-key-123"

    def test_init_with_env_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Initialize with API key from environment variable."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key-456")
        analyzer = BattleAnalyzer()
        assert analyzer.api_key == "env-key-456"

    def test_init_without_key_raises_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raise ValueError when no API key is available."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is required"):
            BattleAnalyzer()

    def test_init_concurrency_default(self) -> None:
        """Default concurrency is 4."""
        analyzer = BattleAnalyzer(api_key="test-key")
        assert analyzer.concurrency == 4

    def test_init_concurrency_custom(self) -> None:
        """Custom concurrency value."""
        analyzer = BattleAnalyzer(api_key="test-key", concurrency=8)
        assert analyzer.concurrency == 8

    def test_analyze_frame_file_not_found(self) -> None:
        """Raise FileNotFoundError when image does not exist."""
        analyzer = BattleAnalyzer(api_key="test-key")
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            analyzer.analyze_frame("/nonexistent/frame.jpg")

    def test_analyze_frame_success(self, tmp_path: Path) -> None:
        """Successfully analyze a frame image."""
        image_file = tmp_path / "frame_00m10s.jpg"
        image_file.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-data")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Game Mode: Turf War\nScore: 45% vs 55%")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        analyzer = BattleAnalyzer(api_key="test-key")
        analyzer._client = mock_client

        result = analyzer.analyze_frame(image_file)

        assert "Turf War" in result
        assert "Score" in result
        mock_client.messages.create.assert_called_once()

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert call_kwargs["max_tokens"] == 1024

    def test_analyze_frame_png_media_type(self, tmp_path: Path) -> None:
        """Use correct media type for PNG images."""
        image_file = tmp_path / "frame_00m00s.png"
        image_file.write_bytes(b"\x89PNGfake-png-data")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Analysis result")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        analyzer = BattleAnalyzer(api_key="test-key")
        analyzer._client = mock_client

        analyzer.analyze_frame(image_file)

        call_kwargs = mock_client.messages.create.call_args[1]
        image_source = call_kwargs["messages"][0]["content"][0]["source"]
        assert image_source["media_type"] == "image/png"

    def test_analyze_frames_multiple(self, tmp_path: Path) -> None:
        """Analyze multiple frames and return structured results."""
        paths = []
        for name in ["frame_00m00s.jpg", "frame_00m10s.jpg", "frame_00m20s.jpg"]:
            p = tmp_path / name
            p.write_bytes(b"\xff\xd8\xff\xe0fake")
            paths.append(p)

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Battle analysis")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        analyzer = BattleAnalyzer(api_key="test-key")
        analyzer._client = mock_client

        results = analyzer.analyze_frames(paths)

        assert len(results) == 3
        assert results[0]["timestamp"] == "00m00s"
        assert results[1]["timestamp"] == "00m10s"
        assert results[2]["timestamp"] == "00m20s"
        assert all(r["analysis"] == "Battle analysis" for r in results)

    def test_analyze_frames_handles_error(self, tmp_path: Path) -> None:
        """Handle analysis errors gracefully for individual frames."""
        good_frame = tmp_path / "frame_00m00s.jpg"
        good_frame.write_bytes(b"\xff\xd8\xff\xe0fake")

        bad_frame = tmp_path / "frame_00m10s.jpg"
        bad_frame.write_bytes(b"\xff\xd8\xff\xe0fake")

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            MagicMock(content=[MagicMock(text="OK")]),
            Exception("API error"),
        ]

        analyzer = BattleAnalyzer(api_key="test-key", concurrency=1)
        analyzer._client = mock_client

        results = analyzer.analyze_frames([good_frame, bad_frame])

        assert len(results) == 2
        assert results[0]["analysis"] == "OK"
        assert "[Error]" in results[1]["analysis"]

    def test_analyze_frames_preserves_order(self, tmp_path: Path) -> None:
        """Results are returned in the same order as input paths."""
        paths = []
        for name in ["frame_00m00s.jpg", "frame_00m10s.jpg", "frame_00m20s.jpg"]:
            p = tmp_path / name
            p.write_bytes(b"\xff\xd8\xff\xe0fake")
            paths.append(p)

        call_count = [0]

        def mock_create(**kwargs):
            call_count[0] += 1
            response = MagicMock()
            response.content = [MagicMock(text=f"Result {call_count[0]}")]
            return response

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = mock_create

        analyzer = BattleAnalyzer(api_key="test-key", concurrency=1)
        analyzer._client = mock_client

        results = analyzer.analyze_frames(paths)

        # With concurrency=1, order should be deterministic
        assert len(results) == 3
        assert results[0]["timestamp"] == "00m00s"
        assert results[1]["timestamp"] == "00m10s"
        assert results[2]["timestamp"] == "00m20s"

    def test_lazy_client_initialization(self) -> None:
        """Client is lazily initialized on first access."""
        analyzer = BattleAnalyzer(api_key="test-key")
        assert analyzer._client is None

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            client = analyzer.client
            assert client is not None
            mock_anthropic.Anthropic.assert_called_once_with(api_key="test-key")


class TestCheckApiKeyAvailable:
    """Tests for check_api_key_available function."""

    def test_key_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Return True when API key is set."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "some-key")
        assert check_api_key_available() is True

    def test_key_not_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Return False when API key is not set."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert check_api_key_available() is False

    def test_key_empty_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Return False when API key is an empty string."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        assert check_api_key_available() is False
