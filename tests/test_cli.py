"""Tests for the CLI module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.cli import format_timeline, parse_args, run


class TestParseArgs:
    """Tests for argument parsing."""

    def test_required_input(self) -> None:
        """--input is required."""
        with pytest.raises(SystemExit):
            parse_args([])

    def test_input_only(self) -> None:
        """Parse with only --input."""
        args = parse_args(["--input", "video.mp4"])
        assert args.input == "video.mp4"
        assert args.interval == 10.0
        assert args.output_dir == "./output"
        assert args.frames_only is False
        assert args.max_frames is None
        assert args.start is None
        assert args.end is None
        assert args.no_save is False
        assert args.concurrency == 4

    def test_all_arguments(self) -> None:
        """Parse all arguments."""
        args = parse_args(
            [
                "--input",
                "game.mkv",
                "--interval",
                "5",
                "--output-dir",
                "/tmp/frames",
                "--frames-only",
                "--verbose",
                "--max-frames",
                "20",
                "--start",
                "30",
                "--end",
                "120",
                "--concurrency",
                "8",
            ]
        )
        assert args.input == "game.mkv"
        assert args.interval == 5.0
        assert args.output_dir == "/tmp/frames"
        assert args.frames_only is True
        assert args.verbose is True
        assert args.max_frames == 20
        assert args.start == 30.0
        assert args.end == 120.0
        assert args.concurrency == 8

    def test_default_interval(self) -> None:
        """Default interval is 10 seconds."""
        args = parse_args(["--input", "v.mp4"])
        assert args.interval == 10.0

    def test_default_output_dir(self) -> None:
        """Default output directory is ./output."""
        args = parse_args(["--input", "v.mp4"])
        assert args.output_dir == "./output"

    def test_no_save_option(self) -> None:
        """Parse --no-save option."""
        args = parse_args(["--input", "v.mp4", "--no-save"])
        assert args.no_save is True

    def test_max_frames_option(self) -> None:
        """Parse --max-frames option."""
        args = parse_args(["--input", "v.mp4", "--max-frames", "50"])
        assert args.max_frames == 50


class TestFormatTimeline:
    """Tests for timeline formatting."""

    def test_empty_results(self) -> None:
        """Format empty results list."""
        result = format_timeline([])
        assert "Total frames analyzed: 0" in result
        assert "SPLATOON BATTLE TIMELINE" in result

    def test_single_result(self) -> None:
        """Format a single analysis result."""
        results = [{"timestamp": "00m10s", "analysis": "Game Mode: Turf War"}]
        result = format_timeline(results)
        assert "[00m10s]" in result
        assert "Game Mode: Turf War" in result
        assert "Total frames analyzed: 1" in result

    def test_multiple_results(self) -> None:
        """Format multiple analysis results."""
        results = [
            {"timestamp": "00m00s", "analysis": "Start"},
            {"timestamp": "00m10s", "analysis": "Mid"},
            {"timestamp": "00m20s", "analysis": "End"},
        ]
        result = format_timeline(results)
        assert "[00m00s]" in result
        assert "[00m10s]" in result
        assert "[00m20s]" in result
        assert "Total frames analyzed: 3" in result


class TestRun:
    """Tests for the main run function."""

    @patch("src.cli.extract_frames")
    def test_frames_only_mode(self, mock_extract: MagicMock, tmp_path: Path) -> None:
        """Run in frames-only mode without API."""
        frame_paths = [tmp_path / "frame_00m00s.jpg", tmp_path / "frame_00m10s.jpg"]
        mock_extract.return_value = frame_paths

        video = tmp_path / "test.mp4"
        video.touch()

        result = run(
            [
                "--input",
                str(video),
                "--interval",
                "10",
                "--output-dir",
                str(tmp_path),
                "--frames-only",
            ]
        )

        assert result == 0
        mock_extract.assert_called_once()

    @patch("src.cli.extract_frames")
    def test_extraction_failure(self, mock_extract: MagicMock, tmp_path: Path) -> None:
        """Return error code when extraction fails."""
        mock_extract.side_effect = FileNotFoundError("Video not found")

        result = run(["--input", "/no/video.mp4", "--frames-only"])

        assert result == 1

    @patch("src.cli.extract_frames")
    def test_no_frames_extracted(self, mock_extract: MagicMock, tmp_path: Path) -> None:
        """Return success when no frames are extracted."""
        mock_extract.return_value = []

        result = run(["--input", str(tmp_path / "test.mp4"), "--frames-only"])

        assert result == 0

    @patch("src.cli.check_api_key_available")
    @patch("src.cli.extract_frames")
    def test_missing_api_key_without_frames_only(
        self,
        mock_extract: MagicMock,
        mock_check_key: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Return error when API key is missing and not in frames-only mode."""
        mock_extract.return_value = [tmp_path / "frame_00m00s.jpg"]
        mock_check_key.return_value = False

        result = run(["--input", str(tmp_path / "test.mp4")])

        assert result == 1

    @patch("src.cli.BattleAnalyzer")
    @patch("src.cli.check_api_key_available")
    @patch("src.cli.extract_frames")
    def test_full_pipeline(
        self,
        mock_extract: MagicMock,
        mock_check_key: MagicMock,
        mock_analyzer_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Run the full pipeline with API analysis."""
        frame_paths = [tmp_path / "frame_00m00s.jpg"]
        mock_extract.return_value = frame_paths
        mock_check_key.return_value = True

        mock_analyzer = MagicMock()
        mock_analyzer.analyze_frames.return_value = [
            {"timestamp": "00m00s", "analysis": "Turf War in progress"}
        ]
        mock_analyzer_cls.return_value = mock_analyzer

        result = run(["--input", str(tmp_path / "test.mp4")])

        assert result == 0
        mock_analyzer.analyze_frames.assert_called_once_with(frame_paths)

    def test_no_save_with_frames_only_returns_error(self) -> None:
        """Return error when --no-save is used with --frames-only."""
        result = run(["--input", "video.mp4", "--no-save", "--frames-only"])
        assert result == 1

    @patch("src.cli.extract_frames")
    def test_max_frames_passed_to_extractor(self, mock_extract: MagicMock, tmp_path: Path) -> None:
        """Pass max_frames to extract_frames."""
        mock_extract.return_value = []

        run(
            [
                "--input",
                str(tmp_path / "test.mp4"),
                "--frames-only",
                "--max-frames",
                "5",
            ]
        )

        call_kwargs = mock_extract.call_args[1]
        assert call_kwargs["max_frames"] == 5

    @patch("src.cli.extract_frames")
    def test_start_end_passed_to_extractor(self, mock_extract: MagicMock, tmp_path: Path) -> None:
        """Pass start and end seconds to extract_frames."""
        mock_extract.return_value = []

        run(
            [
                "--input",
                str(tmp_path / "test.mp4"),
                "--frames-only",
                "--start",
                "30",
                "--end",
                "90",
            ]
        )

        call_kwargs = mock_extract.call_args[1]
        assert call_kwargs["start_seconds"] == 30.0
        assert call_kwargs["end_seconds"] == 90.0

    @patch("src.cli.BattleAnalyzer")
    @patch("src.cli.check_api_key_available")
    @patch("src.cli.extract_frames")
    def test_concurrency_passed_to_analyzer(
        self,
        mock_extract: MagicMock,
        mock_check_key: MagicMock,
        mock_analyzer_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Pass concurrency to BattleAnalyzer."""
        mock_extract.return_value = [tmp_path / "frame_00m00s.jpg"]
        mock_check_key.return_value = True

        mock_analyzer = MagicMock()
        mock_analyzer.analyze_frames.return_value = [{"timestamp": "00m00s", "analysis": "test"}]
        mock_analyzer_cls.return_value = mock_analyzer

        run(["--input", str(tmp_path / "test.mp4"), "--concurrency", "8"])

        mock_analyzer_cls.assert_called_once_with(concurrency=8)
