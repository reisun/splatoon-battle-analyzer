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
        """Extract frames at the specified interval."""
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

            # Simulate 900 frames (30 seconds at 30fps)
            # With 10 second interval, frame_interval = 300
            # Frames captured at: 0, 300, 600 -> 3 frames
            fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame_reads = [(True, fake_frame)] * 900 + [(False, None)]
            mock_cap.read.side_effect = frame_reads
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cv2.imwrite.return_value = True

            result = extract_frames(video_file, interval_seconds=10, output_dir=output_dir)

            assert len(result) == 3
            assert result[0] == output_dir / "frame_00m00s.jpg"
            assert result[1] == output_dir / "frame_00m10s.jpg"
            assert result[2] == output_dir / "frame_00m20s.jpg"
            assert mock_cv2.imwrite.call_count == 3
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
            mock_cap.read.side_effect = [(True, np.zeros((480, 640, 3), dtype=np.uint8))] * 30 + [
                (False, None)
            ]
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cv2.imwrite.return_value = True

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
