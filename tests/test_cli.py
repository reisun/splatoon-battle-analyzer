"""Tests for the CLI module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.cli import (
    _timestamp_to_seconds,
    format_highlight_json,
    format_highlight_text,
    format_json_output,
    format_timeline,
    parse_args,
    run,
)
from src.highlight_detector import HighlightSegment


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
        assert args.model is None
        assert args.output_format == "text"
        assert args.output_file is None

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
                "--model",
                "gemma3:12b",
                "--output-format",
                "json",
                "--output-file",
                "/tmp/result.json",
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
        assert args.model == "gemma3:12b"
        assert args.output_format == "json"
        assert args.output_file == "/tmp/result.json"

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

    def test_model_option(self) -> None:
        """Parse --model option."""
        args = parse_args(["--input", "v.mp4", "--model", "llava:7b"])
        assert args.model == "llava:7b"

    def test_output_format_json(self) -> None:
        """Parse --output-format json."""
        args = parse_args(["--input", "v.mp4", "--output-format", "json"])
        assert args.output_format == "json"

    def test_output_format_invalid(self) -> None:
        """Invalid --output-format raises error."""
        with pytest.raises(SystemExit):
            parse_args(["--input", "v.mp4", "--output-format", "xml"])

    def test_output_file_option(self) -> None:
        """Parse --output-file option."""
        args = parse_args(["--input", "v.mp4", "--output-file", "out.json"])
        assert args.output_file == "out.json"


class TestTimestampToSeconds:
    """Tests for _timestamp_to_seconds helper."""

    def test_zero(self) -> None:
        assert _timestamp_to_seconds("00m00s") == 0

    def test_minutes_and_seconds(self) -> None:
        assert _timestamp_to_seconds("02m30s") == 150

    def test_large_timestamp(self) -> None:
        assert _timestamp_to_seconds("30m00s") == 1800

    def test_invalid_format(self) -> None:
        assert _timestamp_to_seconds("invalid") == 0


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

    def test_dict_analysis_formatted_as_json(self) -> None:
        """Dict analysis is formatted as JSON in text output."""
        results = [{"timestamp": "00m10s", "analysis": {"game_mode": "ナワバリバトル"}}]
        result = format_timeline(results)
        assert "ナワバリバトル" in result


class TestFormatJsonOutput:
    """Tests for JSON output formatting."""

    def test_basic_json_output(self) -> None:
        """Generate valid JSON output."""
        results = [
            {"timestamp": "00m00s", "analysis": {"game_mode": "ナワバリバトル"}},
            {"timestamp": "01m00s", "analysis": {"game_mode": "不明"}},
        ]
        args = parse_args(["--input", "video.mp4", "--interval", "60"])
        output = format_json_output(results, args, "llava-llama3")
        data = json.loads(output)

        assert data["video"] == "video.mp4"
        assert data["model"] == "llava-llama3"
        assert data["interval_seconds"] == 60.0
        assert data["frames_analyzed"] == 2
        assert len(data["timeline"]) == 2
        assert data["timeline"][0]["seconds"] == 0
        assert data["timeline"][1]["seconds"] == 60
        assert data["timeline"][0]["analysis"]["game_mode"] == "ナワバリバトル"

    def test_json_with_raw_string_analysis(self) -> None:
        """Handle raw string analysis in JSON output."""
        results = [{"timestamp": "02m30s", "analysis": "Could not parse"}]
        args = parse_args(["--input", "game.mp4", "--interval", "10"])
        output = format_json_output(results, args, "llava-llama3")
        data = json.loads(output)

        assert data["timeline"][0]["analysis"] == "Could not parse"
        assert data["timeline"][0]["seconds"] == 150


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
    def test_ollama_not_reachable_without_frames_only(
        self,
        mock_extract: MagicMock,
        mock_check_key: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Return error when Ollama is not reachable and not in frames-only mode."""
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
        mock_analyzer.model = "llava-llama3"
        mock_analyzer.analyze_frames.return_value = [
            {"timestamp": "00m00s", "analysis": {"game_mode": "ナワバリバトル"}}
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
        mock_analyzer.model = "llava-llama3"
        mock_analyzer.analyze_frames.return_value = [
            {"timestamp": "00m00s", "analysis": {"game_mode": "test"}}
        ]
        mock_analyzer_cls.return_value = mock_analyzer

        run(["--input", str(tmp_path / "test.mp4"), "--concurrency", "8"])

        mock_analyzer_cls.assert_called_once_with(concurrency=8, model=None)

    @patch("src.cli.BattleAnalyzer")
    @patch("src.cli.check_api_key_available")
    @patch("src.cli.extract_frames")
    def test_model_passed_to_analyzer(
        self,
        mock_extract: MagicMock,
        mock_check_key: MagicMock,
        mock_analyzer_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Pass model to BattleAnalyzer."""
        mock_extract.return_value = [tmp_path / "frame_00m00s.jpg"]
        mock_check_key.return_value = True

        mock_analyzer = MagicMock()
        mock_analyzer.model = "gemma3:12b"
        mock_analyzer.analyze_frames.return_value = [
            {"timestamp": "00m00s", "analysis": {"game_mode": "test"}}
        ]
        mock_analyzer_cls.return_value = mock_analyzer

        run(["--input", str(tmp_path / "test.mp4"), "--model", "gemma3:12b"])

        mock_analyzer_cls.assert_called_once_with(concurrency=4, model="gemma3:12b")

    @patch("src.cli.BattleAnalyzer")
    @patch("src.cli.check_api_key_available")
    @patch("src.cli.extract_frames")
    def test_json_output_format(
        self,
        mock_extract: MagicMock,
        mock_check_key: MagicMock,
        mock_analyzer_cls: MagicMock,
        tmp_path: Path,
        capsys,
    ) -> None:
        """JSON output format produces valid JSON to stdout."""
        mock_extract.return_value = [tmp_path / "frame_00m00s.jpg"]
        mock_check_key.return_value = True

        mock_analyzer = MagicMock()
        mock_analyzer.model = "llava-llama3"
        mock_analyzer.analyze_frames.return_value = [
            {"timestamp": "00m00s", "analysis": {"game_mode": "ナワバリバトル"}}
        ]
        mock_analyzer_cls.return_value = mock_analyzer

        result = run(
            ["--input", str(tmp_path / "test.mp4"), "--output-format", "json", "--interval", "60"]
        )

        assert result == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["video"] == "test.mp4"
        assert data["model"] == "llava-llama3"
        assert data["frames_analyzed"] == 1

    @patch("src.cli.BattleAnalyzer")
    @patch("src.cli.check_api_key_available")
    @patch("src.cli.extract_frames")
    def test_output_file_writes_to_file(
        self,
        mock_extract: MagicMock,
        mock_check_key: MagicMock,
        mock_analyzer_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--output-file writes output to specified file."""
        mock_extract.return_value = [tmp_path / "frame_00m00s.jpg"]
        mock_check_key.return_value = True

        mock_analyzer = MagicMock()
        mock_analyzer.model = "llava-llama3"
        mock_analyzer.analyze_frames.return_value = [
            {"timestamp": "00m00s", "analysis": {"game_mode": "test"}}
        ]
        mock_analyzer_cls.return_value = mock_analyzer

        output_file = tmp_path / "result.json"
        result = run(
            [
                "--input",
                str(tmp_path / "test.mp4"),
                "--output-format",
                "json",
                "--output-file",
                str(output_file),
            ]
        )

        assert result == 0
        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert data["frames_analyzed"] == 1


class TestHighlightModeArgs:
    """Tests for highlight mode argument parsing."""

    def test_mode_default_is_timeline(self) -> None:
        args = parse_args(["--input", "v.mp4"])
        assert args.mode == "timeline"

    def test_mode_highlight(self) -> None:
        args = parse_args(["--input", "v.mp4", "--mode", "highlight"])
        assert args.mode == "highlight"

    def test_highlight_interval_default(self) -> None:
        args = parse_args(["--input", "v.mp4"])
        assert args.highlight_interval == 5.0

    def test_threshold_default(self) -> None:
        args = parse_args(["--input", "v.mp4"])
        assert args.threshold == 100

    def test_custom_highlight_args(self) -> None:
        args = parse_args(
            [
                "--input",
                "v.mp4",
                "--mode",
                "highlight",
                "--highlight-interval",
                "10",
                "--threshold",
                "200",
            ]
        )
        assert args.highlight_interval == 10.0
        assert args.threshold == 200


class TestHighlightFormatting:
    """Tests for highlight output formatting."""

    def test_format_highlight_json(self) -> None:
        highlights = [
            HighlightSegment(
                start_seconds=120.0,
                end_seconds=155.0,
                peak_intensity=8,
                description="Multiple kills",
            )
        ]
        summary = {"total_frames": 40, "battle_frames": 30}
        args = parse_args(["--input", "gameplay.mp4", "--mode", "highlight"])
        output = format_highlight_json(highlights, summary, args, "llava-llama3")
        data = json.loads(output)

        assert data["video"] == "gameplay.mp4"
        assert data["model"] == "llava-llama3"
        assert data["mode"] == "highlight"
        assert len(data["highlights"]) == 1
        assert data["highlights"][0]["start_seconds"] == 120.0
        assert data["highlights"][0]["peak_intensity"] == 8
        assert data["scan_summary"]["total_frames"] == 40

    def test_format_highlight_text_no_highlights(self) -> None:
        summary = {"total_frames": 10, "battle_frames": 0}
        output = format_highlight_text([], summary)
        assert "No highlights detected." in output
        assert "10 frames scanned" in output

    def test_format_highlight_text_with_highlights(self) -> None:
        highlights = [
            HighlightSegment(
                start_seconds=60.0,
                end_seconds=90.0,
                peak_intensity=9,
                description="Team wipe",
            )
        ]
        summary = {"total_frames": 20, "battle_frames": 15}
        output = format_highlight_text(highlights, summary)
        assert "Highlight #1" in output
        assert "60s - 90s" in output
        assert "9" in output


class TestHighlightModeRun:
    """Tests for highlight mode execution."""

    @patch("src.cli.HighlightDetector")
    @patch("src.cli.check_api_key_available")
    def test_highlight_mode_full_pipeline(
        self,
        mock_check: MagicMock,
        mock_detector_cls: MagicMock,
        tmp_path: Path,
        capsys,
    ) -> None:
        mock_check.return_value = True

        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            HighlightSegment(
                start_seconds=30.0,
                end_seconds=60.0,
                peak_intensity=8,
                description="Big play",
            )
        ]
        mock_detector.scan_summary = {
            "total_frames": 10,
            "battle_frames": 8,
        }
        mock_detector_cls.return_value = mock_detector

        result = run(
            [
                "--input",
                str(tmp_path / "test.mp4"),
                "--mode",
                "highlight",
                "--output-format",
                "json",
            ]
        )

        assert result == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["mode"] == "highlight"
        assert len(data["highlights"]) == 1
        assert data["highlights"][0]["peak_intensity"] == 8

    @patch("src.cli.check_api_key_available")
    def test_highlight_mode_ollama_unreachable(self, mock_check: MagicMock, tmp_path: Path) -> None:
        mock_check.return_value = False
        result = run(["--input", str(tmp_path / "test.mp4"), "--mode", "highlight"])
        assert result == 1
