"""Frame extraction module for video files.

Extracts frames from video files at specified intervals using OpenCV.
"""

import logging
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str | Path,
    interval_seconds: float = 10.0,
    output_dir: str | Path = "./output",
) -> list[Path]:
    """Extract frames from a video file at specified intervals.

    Args:
        video_path: Path to the input video file (mp4/mkv).
        interval_seconds: Time interval between frame captures in seconds.
        output_dir: Directory to save extracted frame images.

    Returns:
        List of paths to saved frame images.

    Raises:
        FileNotFoundError: If video file does not exist.
        ValueError: If interval_seconds is not positive.
        RuntimeError: If video file cannot be opened.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if interval_seconds <= 0:
        raise ValueError(f"Interval must be positive, got {interval_seconds}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid FPS value from video: {fps}")

    frame_interval = int(fps * interval_seconds)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    logger.info(
        "Video: %s (%.1f fps, %.1f sec, %d frames)",
        video_path.name,
        fps,
        duration,
        total_frames,
    )

    saved_paths: list[Path] = []
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_interval == 0:
            timestamp = frame_number / fps
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            filename = f"frame_{minutes:02d}m{seconds:02d}s.jpg"
            output_path = output_dir / filename

            cv2.imwrite(str(output_path), frame)
            saved_paths.append(output_path)
            logger.info("Saved frame at %02d:%02d -> %s", minutes, seconds, filename)

        frame_number += 1

    cap.release()
    logger.info("Extracted %d frames from %s", len(saved_paths), video_path.name)
    return saved_paths
