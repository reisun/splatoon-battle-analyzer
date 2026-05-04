"""Battle analysis module using Ollama Vision API (llava-llama3).

Sends frame images to Ollama REST API and extracts battle status information.
Supports concurrent API calls for improved throughput.
"""

import base64
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

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

DEFAULT_OLLAMA_BASE_URL = "http://ollama:11434"
OLLAMA_MODEL = "llava-llama3"


class BattleAnalyzer:
    """Analyzes Splatoon battle frames using Ollama Vision API."""

    def __init__(self, base_url: str | None = None, concurrency: int = 4) -> None:
        """Initialize the analyzer with Ollama endpoint.

        Args:
            base_url: Ollama API base URL. Falls back to OLLAMA_BASE_URL env var.
            concurrency: Maximum number of concurrent API calls.
        """
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        self.concurrency = concurrency

    def _call_ollama(self, prompt: str, image_base64: str) -> str:
        """Call Ollama /api/chat with an image.

        Args:
            prompt: Text prompt to send.
            image_base64: Base64-encoded image data.

        Returns:
            Response text from the model.

        Raises:
            requests.HTTPError: If the API returns a non-2xx status.
        """
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_base64],
                }
            ],
            "stream": False,
        }

        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()

        data = response.json()
        return data["message"]["content"]

    def analyze_frame(self, image_path: str | Path) -> str:
        """Analyze a single frame image using Ollama Vision API.

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

        logger.info("Analyzing frame: %s", image_path.name)

        result = self._call_ollama(ANALYSIS_PROMPT, image_data)
        logger.info("Analysis complete for: %s", image_path.name)
        return result

    def analyze_frame_from_memory(self, frame: np.ndarray, timestamp: str) -> str:
        """Analyze a frame held in memory (numpy array) using Ollama Vision API.

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

        result = self._call_ollama(ANALYSIS_PROMPT, image_data)
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
    """Check if Ollama is reachable by calling /api/tags.

    Returns:
        True if Ollama responds successfully, False otherwise.
    """
    base_url = os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False
