"""Tests for the battle analysis module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.battle_analyzer import BattleAnalyzer, check_api_key_available, parse_llm_response


class TestParseLlmResponse:
    """Tests for parse_llm_response function."""

    def test_valid_json(self) -> None:
        """Parse valid JSON string."""
        result = parse_llm_response('{"game_mode": "ナワバリバトル"}')
        assert result == {"game_mode": "ナワバリバトル"}

    def test_json_in_code_block(self) -> None:
        """Parse JSON wrapped in ```json code block."""
        text = '```json\n{"highlight_score": 7}\n```'
        result = parse_llm_response(text)
        assert result == {"highlight_score": 7}

    def test_json_in_plain_code_block(self) -> None:
        """Parse JSON wrapped in ``` code block without language tag."""
        text = '```\n{"game_mode": "不明"}\n```'
        result = parse_llm_response(text)
        assert result == {"game_mode": "不明"}

    def test_invalid_json_returns_raw(self) -> None:
        """Return raw string when JSON parsing fails."""
        text = "This is not JSON at all"
        result = parse_llm_response(text)
        assert result == text

    def test_empty_string(self) -> None:
        """Return empty string as-is."""
        result = parse_llm_response("")
        assert result == ""


def _make_mock_client(response_text: str) -> MagicMock:
    """Create a mock Gemini client that returns the given text."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = response_text
    mock_client.models.generate_content.return_value = mock_response
    return mock_client


class TestBattleAnalyzer:
    """Tests for BattleAnalyzer class."""

    @patch("src.battle_analyzer._create_client")
    def test_init_default_model(
        self, mock_create: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Initialize with default model."""
        monkeypatch.delenv("GEMINI_MODEL", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        mock_create.return_value = MagicMock()
        analyzer = BattleAnalyzer()
        assert analyzer.model == "gemini-2.5-flash-lite"

    @patch("src.battle_analyzer._create_client")
    def test_init_with_env_model(
        self, mock_create: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Initialize with model from environment variable."""
        monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        mock_create.return_value = MagicMock()
        analyzer = BattleAnalyzer()
        assert analyzer.model == "gemini-2.5-flash"

    @patch("src.battle_analyzer._create_client")
    def test_init_with_explicit_model(
        self, mock_create: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Initialize with explicitly provided model."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        mock_create.return_value = MagicMock()
        analyzer = BattleAnalyzer(model="gemini-2.0-flash")
        assert analyzer.model == "gemini-2.0-flash"

    @patch("src.battle_analyzer._create_client")
    def test_init_concurrency_default(
        self, mock_create: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default concurrency is 4."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        mock_create.return_value = MagicMock()
        analyzer = BattleAnalyzer()
        assert analyzer.concurrency == 4

    @patch("src.battle_analyzer._create_client")
    def test_init_concurrency_custom(
        self, mock_create: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Custom concurrency value."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        mock_create.return_value = MagicMock()
        analyzer = BattleAnalyzer(concurrency=8)
        assert analyzer.concurrency == 8

    @patch("src.battle_analyzer._create_client")
    def test_init_model_explicit_overrides_env(
        self, mock_create: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit model parameter overrides environment variable."""
        monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        mock_create.return_value = MagicMock()
        analyzer = BattleAnalyzer(model="gemini-2.0-flash")
        assert analyzer.model == "gemini-2.0-flash"

    @patch("src.battle_analyzer._create_client")
    def test_analyze_frame_file_not_found(
        self, mock_create: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Raise FileNotFoundError when image does not exist."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        mock_create.return_value = MagicMock()
        analyzer = BattleAnalyzer()
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            analyzer.analyze_frame("/nonexistent/frame.jpg")

    @patch("src.battle_analyzer._create_client")
    def test_analyze_frame_success(
        self, mock_create: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Successfully analyze a frame image."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        image_file = tmp_path / "frame_00m10s.jpg"
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        image_file.write_bytes(buf.tobytes())

        mock_client = _make_mock_client('{"game_mode": "ナワバリバトル", "highlight_score": 5}')
        mock_create.return_value = mock_client

        analyzer = BattleAnalyzer(model="gemini-2.5-flash-lite")
        result = analyzer.analyze_frame(image_file)

        assert isinstance(result, dict)
        assert result["game_mode"] == "ナワバリバトル"
        assert result["highlight_score"] == 5
        mock_client.models.generate_content.assert_called_once()

        call_kwargs = mock_client.models.generate_content.call_args
        assert call_kwargs[1]["model"] == "gemini-2.5-flash-lite"

    @patch("src.battle_analyzer._create_client")
    def test_analyze_frame_returns_raw_on_parse_failure(
        self, mock_create: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Return raw string when LLM response is not valid JSON."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        image_file = tmp_path / "frame_00m10s.jpg"
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        image_file.write_bytes(buf.tobytes())

        mock_client = _make_mock_client("I cannot analyze this image clearly.")
        mock_create.return_value = mock_client

        analyzer = BattleAnalyzer(model="gemini-2.5-flash-lite")
        result = analyzer.analyze_frame(image_file)

        assert isinstance(result, str)
        assert "cannot analyze" in result

    @patch("src.battle_analyzer._create_client")
    def test_analyze_frame_api_error(
        self, mock_create: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Raise RuntimeError when Gemini API call fails."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        image_file = tmp_path / "frame_00m10s.jpg"
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        image_file.write_bytes(buf.tobytes())

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API error")
        mock_create.return_value = mock_client

        analyzer = BattleAnalyzer(model="gemini-2.5-flash-lite")
        with pytest.raises(RuntimeError, match="Gemini API call failed"):
            analyzer.analyze_frame(image_file)

    @patch("src.battle_analyzer._create_client")
    def test_analyze_frame_from_memory(
        self, mock_create: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Analyze a frame from memory (numpy array)."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        mock_client = _make_mock_client('{"game_mode": "不明", "highlight_score": 1}')
        mock_create.return_value = mock_client

        analyzer = BattleAnalyzer(model="gemini-2.5-flash-lite")
        result = analyzer.analyze_frame_from_memory(frame, "01m30s")

        assert isinstance(result, dict)
        assert result["game_mode"] == "不明"
        mock_client.models.generate_content.assert_called_once()

    @patch("src.battle_analyzer._create_client")
    def test_analyze_frames_multiple(
        self, mock_create: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Analyze multiple frames and return structured results."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        paths = []
        for name in ["frame_00m00s.jpg", "frame_00m10s.jpg", "frame_00m20s.jpg"]:
            p = tmp_path / name
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            _, buf = cv2.imencode(".jpg", img)
            p.write_bytes(buf.tobytes())
            paths.append(p)

        mock_client = _make_mock_client('{"game_mode": "ナワバリバトル"}')
        mock_create.return_value = mock_client

        analyzer = BattleAnalyzer(model="gemini-2.5-flash-lite", concurrency=1)
        results = analyzer.analyze_frames(paths)

        assert len(results) == 3
        assert results[0]["timestamp"] == "00m00s"
        assert results[1]["timestamp"] == "00m10s"
        assert results[2]["timestamp"] == "00m20s"
        assert all(isinstance(r["analysis"], dict) for r in results)

    @patch("src.battle_analyzer._create_client")
    def test_analyze_frames_handles_error(
        self, mock_create: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Handle analysis errors gracefully for individual frames."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)

        good_frame = tmp_path / "frame_00m00s.jpg"
        good_frame.write_bytes(buf.tobytes())

        bad_frame = tmp_path / "frame_00m10s.jpg"
        bad_frame.write_bytes(buf.tobytes())

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API error")
        mock_create.return_value = mock_client

        analyzer = BattleAnalyzer(model="gemini-2.5-flash-lite", concurrency=1)
        results = analyzer.analyze_frames([good_frame, bad_frame])

        assert len(results) == 2
        assert "[Error]" in results[0]["analysis"]
        assert "[Error]" in results[1]["analysis"]

    @patch("src.battle_analyzer._create_client")
    def test_analyze_frames_preserves_order(
        self, mock_create: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Results are returned in the same order as input paths."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        paths = []
        for name in ["frame_00m00s.jpg", "frame_00m10s.jpg", "frame_00m20s.jpg"]:
            p = tmp_path / name
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            _, buf = cv2.imencode(".jpg", img)
            p.write_bytes(buf.tobytes())
            paths.append(p)

        call_count = [0]

        def mock_generate(*args, **kwargs):
            call_count[0] += 1
            resp = MagicMock()
            resp.text = f'{{"result": {call_count[0]}}}'
            return resp

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = mock_generate
        mock_create.return_value = mock_client

        analyzer = BattleAnalyzer(model="gemini-2.5-flash-lite", concurrency=1)
        results = analyzer.analyze_frames(paths)

        assert len(results) == 3
        assert results[0]["timestamp"] == "00m00s"
        assert results[1]["timestamp"] == "00m10s"
        assert results[2]["timestamp"] == "00m20s"


class TestCheckApiKeyAvailable:
    """Tests for check_api_key_available function."""

    def test_key_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Return True when GEMINI_API_KEY is set."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        assert check_api_key_available() is True

    def test_key_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Return False when GEMINI_API_KEY is not set."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        assert check_api_key_available() is False

    def test_key_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Return False when GEMINI_API_KEY is empty string."""
        monkeypatch.setenv("GEMINI_API_KEY", "")
        assert check_api_key_available() is False
