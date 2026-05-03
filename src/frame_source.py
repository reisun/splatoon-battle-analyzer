"""Frame source abstraction for video file and RTMP stream inputs.

Provides a unified interface for extracting frames from different sources.
"""

import logging
import signal
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameSource(ABC):
    """Abstract base class for frame sources."""

    @abstractmethod
    def frames(self) -> Iterator[tuple[float, np.ndarray]]:
        """Yield (timestamp, frame) pairs.

        Yields:
            Tuples of (timestamp_in_seconds, frame_as_numpy_array).
        """
        ...


class FileFrameSource(FrameSource):
    """Extract frames from a local video file at specified intervals."""

    def __init__(
        self,
        video_path: str | Path,
        interval_seconds: float = 10.0,
    ) -> None:
        """Initialize the file frame source.

        Args:
            video_path: Path to the input video file.
            interval_seconds: Time interval between frame captures in seconds.

        Raises:
            FileNotFoundError: If video file does not exist.
            ValueError: If interval_seconds is not positive.
        """
        self.video_path = Path(video_path)
        self.interval_seconds = interval_seconds

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        if self.interval_seconds <= 0:
            raise ValueError(f"Interval must be positive, got {self.interval_seconds}")

    def frames(self) -> Iterator[tuple[float, np.ndarray]]:
        """Yield (timestamp, frame) pairs from the video file.

        Yields:
            Tuples of (timestamp_in_seconds, frame_as_numpy_array).

        Raises:
            RuntimeError: If video file cannot be opened or has invalid FPS.
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {self.video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                raise RuntimeError(f"Invalid FPS value from video: {fps}")

            frame_interval = int(fps * self.interval_seconds)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            logger.info(
                "Video: %s (%.1f fps, %.1f sec, %d frames)",
                self.video_path.name,
                fps,
                duration,
                total_frames,
            )

            frame_number = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_number % frame_interval == 0:
                    timestamp = frame_number / fps
                    yield (timestamp, frame)

                frame_number += 1
        finally:
            cap.release()


class StreamFrameSource(FrameSource):
    """Extract frames from an RTMP stream at specified intervals."""

    def __init__(
        self,
        stream_url: str,
        interval_seconds: float = 10.0,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> None:
        """Initialize the stream frame source.

        Args:
            stream_url: RTMP stream URL (e.g., rtmp://host:1935/live/stream).
            interval_seconds: Time interval between frame captures in seconds.
            max_retries: Maximum number of reconnection attempts.
            retry_delay: Delay in seconds between retry attempts.

        Raises:
            ValueError: If interval_seconds is not positive or stream_url is empty.
        """
        if not stream_url:
            raise ValueError("Stream URL must not be empty")

        if interval_seconds <= 0:
            raise ValueError(f"Interval must be positive, got {interval_seconds}")

        self.stream_url = stream_url
        self.interval_seconds = interval_seconds
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._shutdown = False

    def _connect(self) -> cv2.VideoCapture:
        """Attempt to connect to the RTMP stream with retries.

        Returns:
            An opened VideoCapture object.

        Raises:
            RuntimeError: If connection fails after all retries.
        """
        for attempt in range(1, self.max_retries + 1):
            if self._shutdown:
                raise RuntimeError("Shutdown requested during connection")

            logger.info(
                "Connecting to stream: %s (attempt %d/%d)",
                self.stream_url,
                attempt,
                self.max_retries,
            )
            cap = cv2.VideoCapture(self.stream_url)
            if cap.isOpened():
                logger.info("Connected to stream: %s", self.stream_url)
                return cap

            cap.release()
            if attempt < self.max_retries:
                logger.warning(
                    "Connection failed, retrying in %.1f seconds...",
                    self.retry_delay,
                )
                time.sleep(self.retry_delay)

        raise RuntimeError(
            f"Cannot connect to stream after {self.max_retries} attempts: {self.stream_url}"
        )

    def frames(self) -> Iterator[tuple[float, np.ndarray]]:
        """Yield (timestamp, frame) pairs from the RTMP stream.

        Captures frames at the specified interval. Handles graceful shutdown
        via Ctrl+C (SIGINT).

        Yields:
            Tuples of (timestamp_in_seconds, frame_as_numpy_array).

        Raises:
            RuntimeError: If stream connection fails after retries.
        """
        self._shutdown = False
        original_handler = signal.getsignal(signal.SIGINT)

        def _signal_handler(signum: int, frame: object) -> None:
            logger.info("Shutdown requested (SIGINT received)")
            self._shutdown = True

        signal.signal(signal.SIGINT, _signal_handler)

        try:
            cap = self._connect()
            try:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30.0
                    logger.warning("Could not determine stream FPS, defaulting to %.1f", fps)

                frame_interval = int(fps * self.interval_seconds)
                if frame_interval < 1:
                    frame_interval = 1

                logger.info(
                    "Stream: %s (%.1f fps, capturing every %d frames / %.1fs)",
                    self.stream_url,
                    fps,
                    frame_interval,
                    self.interval_seconds,
                )

                frame_number = 0
                start_time = time.monotonic()

                while not self._shutdown:
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning("Stream read failed, attempting reconnect...")
                        cap.release()
                        try:
                            cap = self._connect()
                            continue
                        except RuntimeError:
                            logger.error("Reconnection failed, stopping stream capture")
                            break

                    if frame_number % frame_interval == 0:
                        timestamp = time.monotonic() - start_time
                        yield (timestamp, frame)

                    frame_number += 1

                if self._shutdown:
                    logger.info("Stream capture stopped by user")
            finally:
                cap.release()
        finally:
            signal.signal(signal.SIGINT, original_handler)
