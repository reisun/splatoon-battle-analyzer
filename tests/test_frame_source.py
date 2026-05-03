"""Tests for the frame source module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.frame_source import FileFrameSource, StreamFrameSource


class TestFileFrameSource:
    """Tests for FileFrameSource."""

    def test_file_not_found_raises_error(self) -> None:
        """Raise FileNotFoundError when video file does not exist."""
        with pytest.raises(FileNotFoundError, match="Video file not found"):
            FileFrameSource("/nonexistent/video.mp4")

    def test_invalid_interval_raises_error(self, tmp_path: Path) -> None:
        """Raise ValueError when interval is not positive."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        with pytest.raises(ValueError, match="Interval must be positive"):
            FileFrameSource(video_file, interval_seconds=0)

    def test_negative_interval_raises_error(self, tmp_path: Path) -> None:
        """Raise ValueError when interval is negative."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        with pytest.raises(ValueError, match="Interval must be positive"):
            FileFrameSource(video_file, interval_seconds=-5)

    def test_cannot_open_video_raises_error(self, tmp_path: Path) -> None:
        """Raise RuntimeError when video cannot be opened."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        with patch("src.frame_source.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cv2.VideoCapture.return_value = mock_cap

            source = FileFrameSource(video_file)
            with pytest.raises(RuntimeError, match="Cannot open video file"):
                list(source.frames())

    def test_invalid_fps_raises_error(self, tmp_path: Path) -> None:
        """Raise RuntimeError when video has invalid FPS."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        with patch("src.frame_source.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                mock_cv2.CAP_PROP_FPS: 0.0,
                mock_cv2.CAP_PROP_FRAME_COUNT: 100.0,
            }.get(prop, 0.0)
            mock_cv2.VideoCapture.return_value = mock_cap

            source = FileFrameSource(video_file)
            with pytest.raises(RuntimeError, match="Invalid FPS"):
                list(source.frames())

    def test_yields_frames_at_interval(self, tmp_path: Path) -> None:
        """Yield frames at the specified interval."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        with patch("src.frame_source.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                mock_cv2.CAP_PROP_FPS: 30.0,
                mock_cv2.CAP_PROP_FRAME_COUNT: 900.0,
            }.get(prop, 0.0)

            fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame_reads = [(True, fake_frame)] * 900 + [(False, None)]
            mock_cap.read.side_effect = frame_reads
            mock_cv2.VideoCapture.return_value = mock_cap

            source = FileFrameSource(video_file, interval_seconds=10)
            result = list(source.frames())

            # 30fps * 10s = 300 frame interval -> frames at 0, 300, 600
            assert len(result) == 3
            assert result[0][0] == pytest.approx(0.0)
            assert result[1][0] == pytest.approx(10.0)
            assert result[2][0] == pytest.approx(20.0)
            for _, frame in result:
                assert isinstance(frame, np.ndarray)

            mock_cap.release.assert_called_once()

    def test_releases_capture_on_error(self, tmp_path: Path) -> None:
        """Release VideoCapture even when an error occurs during reading."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        with patch("src.frame_source.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                mock_cv2.CAP_PROP_FPS: 30.0,
                mock_cv2.CAP_PROP_FRAME_COUNT: 100.0,
            }.get(prop, 0.0)
            mock_cap.read.side_effect = Exception("Read error")
            mock_cv2.VideoCapture.return_value = mock_cap

            source = FileFrameSource(video_file, interval_seconds=10)
            with pytest.raises(Exception, match="Read error"):
                list(source.frames())

            mock_cap.release.assert_called_once()


class TestStreamFrameSource:
    """Tests for StreamFrameSource."""

    def test_empty_url_raises_error(self) -> None:
        """Raise ValueError when stream URL is empty."""
        with pytest.raises(ValueError, match="Stream URL must not be empty"):
            StreamFrameSource("")

    def test_invalid_interval_raises_error(self) -> None:
        """Raise ValueError when interval is not positive."""
        with pytest.raises(ValueError, match="Interval must be positive"):
            StreamFrameSource("rtmp://localhost/live/stream", interval_seconds=0)

    def test_connection_failure_raises_error(self) -> None:
        """Raise RuntimeError when stream connection fails after retries."""
        with patch("src.frame_source.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cv2.VideoCapture.return_value = mock_cap

            with patch("src.frame_source.time") as mock_time:
                mock_time.monotonic.return_value = 0.0
                mock_time.sleep = MagicMock()

                source = StreamFrameSource(
                    "rtmp://localhost/live/stream",
                    max_retries=2,
                    retry_delay=0.1,
                )
                with pytest.raises(RuntimeError, match="Cannot connect to stream after 2"):
                    list(source.frames())

                # Should have tried twice and slept once between attempts
                assert mock_cv2.VideoCapture.call_count == 2
                mock_time.sleep.assert_called_once_with(0.1)

    def test_yields_frames_from_stream(self) -> None:
        """Yield frames from the RTMP stream at specified intervals."""
        with patch("src.frame_source.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 30.0  # FPS

            fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Simulate 300 frames (10 seconds at 30fps) then stream ends
            frame_reads = [(True, fake_frame)] * 300 + [(False, None)]
            mock_cap.read.side_effect = frame_reads
            mock_cv2.VideoCapture.return_value = mock_cap

            with patch("src.frame_source.time") as mock_time:
                mock_time.monotonic.return_value = 0.0
                mock_time.sleep = MagicMock()

                source = StreamFrameSource(
                    "rtmp://localhost/live/stream",
                    interval_seconds=10,
                    max_retries=1,
                )
                # After 300 frames, read fails -> reconnect attempt fails -> stops
                # With reconnect failing (isOpened returns False on second cap)
                second_cap = MagicMock()
                second_cap.isOpened.return_value = False
                mock_cv2.VideoCapture.side_effect = [mock_cap, second_cap]

                result = list(source.frames())

            # 30fps * 10s = 300 frame interval -> frame at 0 only (frame 300 fails)
            assert len(result) == 1
            assert isinstance(result[0][1], np.ndarray)

    def test_graceful_shutdown(self) -> None:
        """Stop capturing when shutdown is requested."""
        with patch("src.frame_source.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 30.0

            fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

            call_count = 0

            def read_side_effect() -> tuple[bool, np.ndarray | None]:
                nonlocal call_count
                call_count += 1
                if call_count > 5:
                    # Simulate shutdown after a few reads
                    source._shutdown = True
                return (True, fake_frame)

            mock_cap.read.side_effect = read_side_effect
            mock_cv2.VideoCapture.return_value = mock_cap

            with patch("src.frame_source.time") as mock_time:
                mock_time.monotonic.return_value = 0.0
                mock_time.sleep = MagicMock()

                with patch("src.frame_source.signal"):
                    source = StreamFrameSource(
                        "rtmp://localhost/live/stream",
                        interval_seconds=1,
                        max_retries=1,
                    )
                    result = list(source.frames())

            # Should have captured frame at index 0 only (interval=1s, fps=30 -> every 30 frames)
            # But shutdown after 5 reads, so only frame 0
            assert len(result) == 1
            mock_cap.release.assert_called_once()

    def test_reconnect_on_read_failure(self) -> None:
        """Reconnect when stream read fails mid-capture."""
        with patch("src.frame_source.cv2") as mock_cv2:
            fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # First capture: connected, yields 1 frame, then fails
            first_cap = MagicMock()
            first_cap.isOpened.return_value = True
            first_cap.get.return_value = 30.0
            first_cap.read.side_effect = [(True, fake_frame), (False, None)]

            # Second capture (reconnect): connected, yields 1 frame, then fails
            second_cap = MagicMock()
            second_cap.isOpened.return_value = True
            second_cap.get.return_value = 30.0
            second_cap.read.side_effect = [(True, fake_frame), (False, None)]

            # Third capture (second reconnect fails)
            third_cap = MagicMock()
            third_cap.isOpened.return_value = False

            mock_cv2.VideoCapture.side_effect = [first_cap, second_cap, third_cap]

            with patch("src.frame_source.time") as mock_time:
                mock_time.monotonic.return_value = 0.0
                mock_time.sleep = MagicMock()

                with patch("src.frame_source.signal"):
                    source = StreamFrameSource(
                        "rtmp://localhost/live/stream",
                        interval_seconds=1,
                        max_retries=1,
                    )
                    result = list(source.frames())

            # Frame at index 0 from first cap, frame at index 0 (continue) from second cap
            # interval = 30 frames, so only frame_number % 30 == 0 yields
            # first_cap: frame 0 yields, frame 1 fails -> reconnect
            # second_cap: continue from frame_number=1, reads frame -> frame_number=2, fails
            # third_cap: reconnect fails -> stop
            # Only frame 0 from first capture yielded
            assert len(result) == 1

    def test_default_fps_when_unavailable(self) -> None:
        """Use default FPS when stream does not report FPS."""
        with patch("src.frame_source.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 0.0  # Invalid FPS

            fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cap.read.side_effect = [(True, fake_frame), (False, None)]
            mock_cv2.VideoCapture.return_value = mock_cap

            with patch("src.frame_source.time") as mock_time:
                mock_time.monotonic.return_value = 0.0
                mock_time.sleep = MagicMock()

                # Reconnect after read failure fails
                second_cap = MagicMock()
                second_cap.isOpened.return_value = False
                mock_cv2.VideoCapture.side_effect = [mock_cap, second_cap]

                with patch("src.frame_source.signal"):
                    source = StreamFrameSource(
                        "rtmp://localhost/live/stream",
                        interval_seconds=1,
                        max_retries=1,
                    )
                    result = list(source.frames())

            # Should still work with default FPS (30.0)
            assert len(result) == 1


class TestCliStreamIntegration:
    """Tests for CLI --stream and --input mutual exclusion."""

    def test_input_and_stream_mutually_exclusive(self) -> None:
        """Error when both --input and --stream are specified."""
        from src.cli import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--input", "video.mp4", "--stream", "rtmp://localhost/live"])

    def test_neither_input_nor_stream_errors(self) -> None:
        """Error when neither --input nor --stream is specified."""
        from src.cli import parse_args

        with pytest.raises(SystemExit):
            parse_args([])

    def test_stream_only(self) -> None:
        """Parse with only --stream."""
        from src.cli import parse_args

        args = parse_args(["--stream", "rtmp://localhost:1935/live/stream"])
        assert args.stream == "rtmp://localhost:1935/live/stream"
        assert args.input is None
        assert args.interval == 10.0

    def test_input_only(self) -> None:
        """Parse with only --input (backward compatible)."""
        from src.cli import parse_args

        args = parse_args(["--input", "video.mp4"])
        assert args.input == "video.mp4"
        assert args.stream is None

    def test_stream_with_options(self) -> None:
        """Parse --stream with other options."""
        from src.cli import parse_args

        args = parse_args(
            [
                "--stream",
                "rtmp://host.docker.internal:1935/live/stream",
                "--interval",
                "5",
                "--output-dir",
                "/tmp/frames",
                "--frames-only",
            ]
        )
        assert args.stream == "rtmp://host.docker.internal:1935/live/stream"
        assert args.interval == 5.0
        assert args.output_dir == "/tmp/frames"
        assert args.frames_only is True

    @patch("src.cli.save_frames")
    @patch("src.cli.create_frame_source")
    def test_run_with_stream(
        self,
        mock_create_source: MagicMock,
        mock_save_frames: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Run pipeline with --stream in frames-only mode."""
        from src.cli import run

        mock_source = MagicMock()
        mock_create_source.return_value = mock_source
        mock_save_frames.return_value = [tmp_path / "frame_00m00s.jpg"]

        result = run(
            [
                "--stream",
                "rtmp://localhost/live/stream",
                "--output-dir",
                str(tmp_path),
                "--frames-only",
            ]
        )

        assert result == 0
        mock_create_source.assert_called_once()
        mock_save_frames.assert_called_once()
