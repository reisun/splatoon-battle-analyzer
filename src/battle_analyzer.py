"""Battle analysis module using Claude Vision API.

Sends frame images to Claude Vision API and extracts battle status information.
Supports concurrent API calls for improved throughput.
"""

import base64
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ANALYSIS_PROMPT = """You are analyzing a screenshot from the Nintendo game Splatoon.
Extract the following battle information visible on screen:

1. **Game Mode**: Turf War, Splat Zones, Tower Control, Rainmaker, or Clam Blitz
2. **Score/Objective Status**: Current score, zone control percentage, tower position, etc.
3. **Time Remaining**: Match timer if visible
4. **Player Team Status**: Number of active players, any splatted indicators
5. **Enemy Team Status**: Number of active players, any splatted indicators
6. **Special Gauges**: Special weapon charge status if visible
7. **Map Control**: General ink coverage assessment (which team controls more territory)
8. **Notable Events**: Any significant events visible (splats, specials being used, objectives being captured)

Respond in a structured format. If any element is not visible or unclear, note it as "Not visible".
Keep the response concise - one line per category."""


class BattleAnalyzer:
    """Analyzes Splatoon battle frames using Claude Vision API."""

    def __init__(self, api_key: str | None = None, concurrency: int = 4) -> None:
        """Initialize the analyzer with an Anthropic API key.

        Args:
            api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
            concurrency: Maximum number of concurrent API calls.

        Raises:
            ValueError: If no API key is available.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required. "
                "Set it via environment variable or pass it directly."
            )
        self._client: object | None = None
        self.concurrency = concurrency

    @property
    def client(self) -> object:
        """Lazy-initialize the Anthropic client."""
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def analyze_frame(self, image_path: str | Path) -> str:
        """Analyze a single frame image using Claude Vision API.

        Args:
            image_path: Path to the frame image file.

        Returns:
            Analysis text describing the battle status.

        Raises:
            FileNotFoundError: If image file does not exist.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image_data = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")

        suffix = image_path.suffix.lower()
        media_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_type_map.get(suffix, "image/jpeg")

        logger.info("Analyzing frame: %s", image_path.name)

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": ANALYSIS_PROMPT,
                        },
                    ],
                }
            ],
        )

        result = response.content[0].text
        logger.info("Analysis complete for: %s", image_path.name)
        return result

    def analyze_frame_from_memory(self, frame: np.ndarray, timestamp: str) -> str:
        """Analyze a frame held in memory (numpy array) using Claude Vision API.

        Args:
            frame: Frame image as a numpy array (BGR format from OpenCV).
            timestamp: Timestamp label for logging.

        Returns:
            Analysis text describing the battle status.
        """
        import cv2

        _, buffer = cv2.imencode(".jpg", frame)
        image_data = base64.standard_b64encode(buffer.tobytes()).decode("utf-8")

        logger.info("Analyzing frame at %s (from memory)", timestamp)

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": ANALYSIS_PROMPT,
                        },
                    ],
                }
            ],
        )

        result = response.content[0].text
        logger.info("Analysis complete for frame at %s", timestamp)
        return result

    def analyze_frames(self, image_paths: list[Path]) -> list[dict[str, str]]:
        """Analyze multiple frame images concurrently.

        Args:
            image_paths: List of paths to frame images.

        Returns:
            List of dicts with 'timestamp' (from filename) and 'analysis' keys.
            Results are returned in the same order as input paths.
        """
        results: list[dict[str, str]] = [{}] * len(image_paths)

        def _analyze_one(index: int, path: Path) -> tuple[int, dict[str, str]]:
            timestamp = path.stem.replace("frame_", "")
            try:
                analysis = self.analyze_frame(path)
                return index, {"timestamp": timestamp, "analysis": analysis}
            except Exception:
                logger.exception("Failed to analyze %s", path.name)
                return index, {
                    "timestamp": timestamp,
                    "analysis": f"[Error] Failed to analyze {path.name}",
                }

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = [
                executor.submit(_analyze_one, i, path) for i, path in enumerate(image_paths)
            ]
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return results


def check_api_key_available() -> bool:
    """Check if ANTHROPIC_API_KEY is set in environment.

    Returns:
        True if the key is available, False otherwise.
    """
    return bool(os.environ.get("ANTHROPIC_API_KEY"))
