"""Frame extraction module for video files.

Extracts frames from video files at specified intervals using OpenCV.
Supports seek-based extraction for performance and time range filtering.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str | Path,
    interval_seconds: float = 10.0,
    output_dir: str | Path = "./output",
    max_frames: int | None = None,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    no_save: bool = False,
) -> list[Path] | list[np.ndarray]:
    """Extract frames from a video file at specified intervals using seek.

    Args:
        video_path: Path to the input video file (mp4/mkv).
        interval_seconds: Time interval between frame captures in seconds.
        output_dir: Directory to save extracted frame images.
        max_frames: Maximum number of frames to extract. None means no limit.
        start_seconds: Start time in seconds. None means from the beginning.
        end_seconds: End time in seconds. None means until the end.
        no_save: If True, return frames as numpy arrays instead of saving to disk.

    Returns:
        List of paths to saved frame images, or list of numpy arrays if no_save=True.

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

    if not no_save:
        output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid FPS value from video: {fps}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    logger.info(
        "Video: %s (%.1f fps, %.1f sec, %d frames)",
        video_path.name,
        fps,
        duration,
        total_frames,
    )

    # Determine start and end frame numbers
    start_frame = int(start_seconds * fps) if start_seconds is not None else 0
    end_frame = int(end_seconds * fps) if end_seconds is not None else total_frames

    # Clamp to valid range
    start_frame = max(0, start_frame)
    end_frame = min(total_frames, end_frame)

    # Compute target frame numbers using seek
    frame_interval = int(fps * interval_seconds)
    target_frames = list(range(start_frame, end_frame, frame_interval))

    # Apply max_frames limit
    if max_frames is not None and max_frames > 0:
        target_frames = target_frames[:max_frames]

    logger.info(
        "Extracting %d frames (start=%.1fs, end=%.1fs, interval=%.1fs)",
        len(target_frames),
        start_frame / fps,
        end_frame / fps,
        interval_seconds,
    )

    saved_paths: list[Path] = []
    memory_frames: list[np.ndarray] = []

    for target_frame_number in target_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_number)
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame at position %d", target_frame_number)
            continue

        timestamp = target_frame_number / fps
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)

        if no_save:
            memory_frames.append(frame)
            logger.debug("Captured frame at %02d:%02d (in memory)", minutes, seconds)
        else:
            filename = f"frame_{minutes:02d}m{seconds:02d}s.jpg"
            output_path = output_dir / filename
            cv2.imwrite(str(output_path), frame)
            saved_paths.append(output_path)
            logger.info("Saved frame at %02d:%02d -> %s", minutes, seconds, filename)

    cap.release()

    if no_save:
        logger.info("Extracted %d frames from %s (in memory)", len(memory_frames), video_path.name)
        return memory_frames
    else:
        logger.info("Extracted %d frames from %s", len(saved_paths), video_path.name)
        return saved_paths
