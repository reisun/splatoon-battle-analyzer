"""Tests for the frame extraction module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.frame_extractor import extract_frames


class TestExtractFrames:
    """Tests for extract_frames function."""

    def test_file_not_found_raises_error(self, tmp_path: Path) -> None:
        """Raise FileNotFoundError when video file does not exist."""
        with pytest.raises(FileNotFoundError, match="Video file not found"):
            extract_frames("/nonexistent/video.mp4", output_dir=tmp_path)

    def test_invalid_interval_raises_error(self, tmp_path: Path) -> None:
        """Raise ValueError when interval is not positive."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        with patch("src.frame_extractor.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cv2.VideoCapture.return_value = mock_cap

            with pytest.raises(ValueError, match="Interval must be positive"):
                extract_frames(video_file, interval_seconds=0, output_dir=tmp_path)

    def test_negative_interval_raises_error(self, tmp_path: Path) -> None:
        """Raise ValueError when interval is negative."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        with pytest.raises(ValueError, match="Interval must be positive"):
            extract_frames(video_file, interval_seconds=-5, output_dir=tmp_path)

    def test_cannot_open_video_raises_error(self, tmp_path: Path) -> None:
        """Raise RuntimeError when video cannot be opened."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        with patch("src.frame_extractor.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cv2.VideoCapture.return_value = mock_cap

            with pytest.raises(RuntimeError, match="Cannot open video file"):
                extract_frames(video_file, output_dir=tmp_path)

    def test_extracts_frames_at_interval(self, tmp_path: Path) -> None:
        """Extract frames at the specified interval using seek."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()
        output_dir = tmp_path / "output"

        with patch("src.frame_extractor.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                mock_cv2.CAP_PROP_FPS: 30.0,
                mock_cv2.CAP_PROP_FRAME_COUNT: 900.0,  # 30 seconds at 30fps
            }.get(prop, 0.0)

            # With seek-based approach, cap.read() is called once per target frame
            fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cap.read.return_value = (True, fake_frame)
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cv2.imwrite.return_value = True
            mock_cv2.CAP_PROP_POS_FRAMES = 1  # cv2 constant

            result = extract_frames(video_file, interval_seconds=10, output_dir=output_dir)

            # 30 seconds at 10s interval -> frames at 0, 10, 20 = 3 frames
            assert len(result) == 3
            assert result[0] == output_dir / "frame_00m00s.jpg"
            assert result[1] == output_dir / "frame_00m10s.jpg"
            assert result[2] == output_dir / "frame_00m20s.jpg"
            assert mock_cv2.imwrite.call_count == 3
            # Verify seek was called for each target frame
            assert mock_cap.set.call_count == 3
            mock_cap.release.assert_called_once()

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        """Create output directory if it does not exist."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()
        output_dir = tmp_path / "nested" / "output"

        with patch("src.frame_extractor.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                mock_cv2.CAP_PROP_FPS: 30.0,
                mock_cv2.CAP_PROP_FRAME_COUNT: 30.0,
            }.get(prop, 0.0)
            fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cap.read.return_value = (True, fake_frame)
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cv2.imwrite.return_value = True
            mock_cv2.CAP_PROP_POS_FRAMES = 1

            extract_frames(video_file, interval_seconds=1, output_dir=output_dir)

            assert output_dir.exists()

    def test_invalid_fps_raises_error(self, tmp_path: Path) -> None:
        """Raise RuntimeError when video has invalid FPS."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        with patch("src.frame_extractor.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                mock_cv2.CAP_PROP_FPS: 0.0,
                mock_cv2.CAP_PROP_FRAME_COUNT: 100.0,
            }.get(prop, 0.0)
            mock_cv2.VideoCapture.return_value = mock_cap

            with pytest.raises(RuntimeError, match="Invalid FPS"):
                extract_frames(video_file, output_dir=tmp_path)

    def test_max_frames_limits_output(self, tmp_path: Path) -> None:
        """Limit number of extracted frames with max_frames parameter."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()
        output_dir = tmp_path / "output"

        with patch("src.frame_extractor.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                mock_cv2.CAP_PROP_FPS: 30.0,
                mock_cv2.CAP_PROP_FRAME_COUNT: 9000.0,  # 5 minutes
            }.get(prop, 0.0)

            fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cap.read.return_value = (True, fake_frame)
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cv2.imwrite.return_value = True
            mock_cv2.CAP_PROP_POS_FRAMES = 1

            result = extract_frames(
                video_file, interval_seconds=10, output_dir=output_dir, max_frames=2
            )

            assert len(result) == 2

    def test_start_end_seconds(self, tmp_path: Path) -> None:
        """Extract frames only within the specified time range."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()
        output_dir = tmp_path / "output"

        with patch("src.frame_extractor.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                mock_cv2.CAP_PROP_FPS: 30.0,
                mock_cv2.CAP_PROP_FRAME_COUNT: 3600.0,  # 120 seconds
            }.get(prop, 0.0)

            fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cap.read.return_value = (True, fake_frame)
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cv2.imwrite.return_value = True
            mock_cv2.CAP_PROP_POS_FRAMES = 1

            # Start at 30s, end at 60s, interval 10s -> frames at 30, 40, 50 = 3 frames
            result = extract_frames(
                video_file,
                interval_seconds=10,
                output_dir=output_dir,
                start_seconds=30,
                end_seconds=60,
            )

            assert len(result) == 3
            assert result[0] == output_dir / "frame_00m30s.jpg"
            assert result[1] == output_dir / "frame_00m40s.jpg"
            assert result[2] == output_dir / "frame_00m50s.jpg"

    def test_no_save_returns_numpy_arrays(self, tmp_path: Path) -> None:
        """Return numpy arrays when no_save=True."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        with patch("src.frame_extractor.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                mock_cv2.CAP_PROP_FPS: 30.0,
                mock_cv2.CAP_PROP_FRAME_COUNT: 900.0,
            }.get(prop, 0.0)

            fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cap.read.return_value = (True, fake_frame)
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cv2.CAP_PROP_POS_FRAMES = 1

            result = extract_frames(
                video_file, interval_seconds=10, output_dir=tmp_path, no_save=True
            )

            assert len(result) == 3
            assert all(isinstance(f, np.ndarray) for f in result)
            # imwrite should not be called
            mock_cv2.imwrite.assert_not_called()

    def test_seek_based_extraction_skips_failed_reads(self, tmp_path: Path) -> None:
        """Skip frames that fail to read after seek."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()
        output_dir = tmp_path / "output"

        with patch("src.frame_extractor.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                mock_cv2.CAP_PROP_FPS: 30.0,
                mock_cv2.CAP_PROP_FRAME_COUNT: 900.0,
            }.get(prop, 0.0)

            fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # First read succeeds, second fails, third succeeds
            mock_cap.read.side_effect = [
                (True, fake_frame),
                (False, None),
                (True, fake_frame),
            ]
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cv2.imwrite.return_value = True
            mock_cv2.CAP_PROP_POS_FRAMES = 1

            result = extract_frames(video_file, interval_seconds=10, output_dir=output_dir)

            # 3 target frames, but one fails -> 2 results
            assert len(result) == 2
