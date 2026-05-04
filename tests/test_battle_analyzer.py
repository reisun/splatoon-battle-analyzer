"""Tests for the battle analysis module."""

import subprocess
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


class TestBattleAnalyzer:
    """Tests for BattleAnalyzer class."""

    def test_init_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Initialize with default model."""
        monkeypatch.delenv("CLAUDE_MODEL", raising=False)
        analyzer = BattleAnalyzer()
        assert analyzer.model == "haiku"

    def test_init_with_env_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Initialize with model from environment variable."""
        monkeypatch.setenv("CLAUDE_MODEL", "sonnet")
        analyzer = BattleAnalyzer()
        assert analyzer.model == "sonnet"

    def test_init_with_explicit_model(self) -> None:
        """Initialize with explicitly provided model."""
        analyzer = BattleAnalyzer(model="opus")
        assert analyzer.model == "opus"

    def test_init_concurrency_default(self) -> None:
        """Default concurrency is 4."""
        analyzer = BattleAnalyzer()
        assert analyzer.concurrency == 4

    def test_init_concurrency_custom(self) -> None:
        """Custom concurrency value."""
        analyzer = BattleAnalyzer(concurrency=8)
        assert analyzer.concurrency == 8

    def test_init_model_explicit_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit model parameter overrides environment variable."""
        monkeypatch.setenv("CLAUDE_MODEL", "sonnet")
        analyzer = BattleAnalyzer(model="opus")
        assert analyzer.model == "opus"

    def test_analyze_frame_file_not_found(self) -> None:
        """Raise FileNotFoundError when image does not exist."""
        analyzer = BattleAnalyzer()
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            analyzer.analyze_frame("/nonexistent/frame.jpg")

    @patch("src.battle_analyzer.subprocess.run")
    def test_analyze_frame_success(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Successfully analyze a frame image."""
        image_file = tmp_path / "frame_00m10s.jpg"
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        image_file.write_bytes(buf.tobytes())

        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"game_mode": "ナワバリバトル", "highlight_score": 5}',
            stderr="",
        )

        analyzer = BattleAnalyzer(model="haiku")
        result = analyzer.analyze_frame(image_file)

        assert isinstance(result, dict)
        assert result["game_mode"] == "ナワバリバトル"
        assert result["highlight_score"] == 5
        mock_run.assert_called_once()

        call_args = mock_run.call_args[0][0]
        assert "claude" in call_args
        assert "--model" in call_args
        assert "haiku" in call_args

    @patch("src.battle_analyzer.subprocess.run")
    def test_analyze_frame_returns_raw_on_parse_failure(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """Return raw string when LLM response is not valid JSON."""
        image_file = tmp_path / "frame_00m10s.jpg"
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        image_file.write_bytes(buf.tobytes())

        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="I cannot analyze this image clearly.",
            stderr="",
        )

        analyzer = BattleAnalyzer(model="haiku")
        result = analyzer.analyze_frame(image_file)

        assert isinstance(result, str)
        assert "cannot analyze" in result

    @patch("src.battle_analyzer.subprocess.run")
    def test_analyze_frame_cli_error(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Raise RuntimeError when CLI fails."""
        image_file = tmp_path / "frame_00m10s.jpg"
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        image_file.write_bytes(buf.tobytes())

        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="Authentication failed",
        )

        analyzer = BattleAnalyzer(model="haiku")
        with pytest.raises(RuntimeError, match="Claude CLI failed"):
            analyzer.analyze_frame(image_file)

    @patch("src.battle_analyzer.subprocess.run")
    def test_analyze_frame_from_memory(self, mock_run: MagicMock) -> None:
        """Analyze a frame from memory (numpy array)."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"game_mode": "不明", "highlight_score": 1}',
            stderr="",
        )

        analyzer = BattleAnalyzer(model="haiku")
        result = analyzer.analyze_frame_from_memory(frame, "01m30s")

        assert isinstance(result, dict)
        assert result["game_mode"] == "不明"
        mock_run.assert_called_once()

    @patch("src.battle_analyzer.subprocess.run")
    def test_analyze_frames_multiple(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Analyze multiple frames and return structured results."""
        paths = []
        for name in ["frame_00m00s.jpg", "frame_00m10s.jpg", "frame_00m20s.jpg"]:
            p = tmp_path / name
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            _, buf = cv2.imencode(".jpg", img)
            p.write_bytes(buf.tobytes())
            paths.append(p)

        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"game_mode": "ナワバリバトル"}',
            stderr="",
        )

        analyzer = BattleAnalyzer(model="haiku", concurrency=1)
        results = analyzer.analyze_frames(paths)

        assert len(results) == 3
        assert results[0]["timestamp"] == "00m00s"
        assert results[1]["timestamp"] == "00m10s"
        assert results[2]["timestamp"] == "00m20s"
        assert all(isinstance(r["analysis"], dict) for r in results)

    @patch("src.battle_analyzer.subprocess.run")
    def test_analyze_frames_handles_error(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Handle analysis errors gracefully for individual frames."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)

        good_frame = tmp_path / "frame_00m00s.jpg"
        good_frame.write_bytes(buf.tobytes())

        bad_frame = tmp_path / "frame_00m10s.jpg"
        bad_frame.write_bytes(buf.tobytes())

        good_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout='{"game_mode": "不明"}', stderr=""
        )
        bad_result = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="API error"
        )

        mock_run.side_effect = [good_result, bad_result]

        analyzer = BattleAnalyzer(model="haiku", concurrency=1)
        results = analyzer.analyze_frames([good_frame, bad_frame])

        assert len(results) == 2
        assert isinstance(results[0]["analysis"], dict)
        assert "[Error]" in results[1]["analysis"]

    @patch("src.battle_analyzer.subprocess.run")
    def test_analyze_frames_preserves_order(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Results are returned in the same order as input paths."""
        paths = []
        for name in ["frame_00m00s.jpg", "frame_00m10s.jpg", "frame_00m20s.jpg"]:
            p = tmp_path / name
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            _, buf = cv2.imencode(".jpg", img)
            p.write_bytes(buf.tobytes())
            paths.append(p)

        call_count = [0]

        def mock_run_fn(*args, **kwargs):
            call_count[0] += 1
            return subprocess.CompletedProcess(
                args=[], returncode=0, stdout=f'{{"result": {call_count[0]}}}', stderr=""
            )

        mock_run.side_effect = mock_run_fn

        analyzer = BattleAnalyzer(model="haiku", concurrency=1)
        results = analyzer.analyze_frames(paths)

        assert len(results) == 3
        assert results[0]["timestamp"] == "00m00s"
        assert results[1]["timestamp"] == "00m10s"
        assert results[2]["timestamp"] == "00m20s"


class TestCheckApiKeyAvailable:
    """Tests for check_api_key_available function."""

    @patch("src.battle_analyzer.subprocess.run")
    def test_cli_available(self, mock_run: MagicMock) -> None:
        """Return True when Claude CLI is available."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="1.0.0", stderr=""
        )
        assert check_api_key_available() is True

    @patch("src.battle_analyzer.subprocess.run")
    def test_cli_not_found(self, mock_run: MagicMock) -> None:
        """Return False when Claude CLI is not installed."""
        mock_run.side_effect = FileNotFoundError("No such file")
        assert check_api_key_available() is False

    @patch("src.battle_analyzer.subprocess.run")
    def test_cli_timeout(self, mock_run: MagicMock) -> None:
        """Return False when CLI times out."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=10)
        assert check_api_key_available() is False

    @patch("src.battle_analyzer.subprocess.run")
    def test_cli_error(self, mock_run: MagicMock) -> None:
        """Return False when CLI returns non-zero exit code."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="error"
        )
        assert check_api_key_available() is False
