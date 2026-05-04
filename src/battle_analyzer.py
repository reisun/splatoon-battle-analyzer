"""Battle analysis module using Ollama Vision API (llava-llama3).

Sends frame images to Ollama REST API and extracts battle status information.
Supports concurrent API calls for improved throughput.
"""

import base64
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

logger = logging.getLogger(__name__)

ANALYSIS_PROMPT = """あなたはスプラトゥーンのゲーム画面を分析するアシスタントです。
以下のスクリーンショットから戦況を読み取り、JSON形式で回答してください。

回答フォーマット:
{
  "game_mode": "ナワバリバトル/ガチエリア/ガチヤグラ/ガチホコ/ガチアサリ のいずれか、または不明",
  "time_remaining": "残り時間（秒数）。不明なら null",
  "score": {"player_team": null, "enemy_team": null},
  "team_status": {
    "player_team": {"alive": 0, "splatted": 0},
    "enemy_team": {"alive": 0, "splatted": 0}
  },
  "map_control": "味方優勢/互角/敵優勢/不明",
  "special_gauge": "0-100の数値。不明なら null",
  "events": ["発生中のイベントを列挙"],
  "highlight_score": "1-10の整数（10が最も盛り上がっている）",
  "highlight_reason": "スコアの理由を短く"
}

見えない項目は null としてください。推測せず、見えるものだけ報告してください。"""

STAGE1_PROMPT = """Analyze this Splatoon gameplay screenshot. Answer in JSON only:
{"scene": "battle", "intensity": 5, "reason": "brief description"}
scene must be one of: battle, lobby, result, other
intensity must be 1-10. If unsure, use lower intensity."""

STAGE2_PROMPT = """Analyze this Splatoon battle screenshot in detail. Answer in JSON only:
{"kills_visible": false, "deaths_visible": false, "special_active": false, "score_change": false, "team_wipe": false, "intensity": 5, "description": "what is happening"}
Only report what you clearly see. intensity must be 1-10."""

DEFAULT_OLLAMA_BASE_URL = "http://ollama:11434"
DEFAULT_OLLAMA_MODEL = "llava-llama3"


def parse_llm_response(text: str) -> dict | str:
    """Parse JSON from LLM response text.

    Handles responses wrapped in ```json ... ``` code blocks.
    Returns parsed dict on success, raw string on failure.
    """
    cleaned = text.strip()

    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()

    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return text


class BattleAnalyzer:
    """Analyzes Splatoon battle frames using Ollama Vision API."""

    def __init__(
        self, base_url: str | None = None, model: str | None = None, concurrency: int = 1
    ) -> None:
        """Initialize the analyzer with Ollama endpoint.

        Args:
            base_url: Ollama API base URL. Falls back to OLLAMA_BASE_URL env var.
            model: Model name. Falls back to OLLAMA_MODEL env var, then default.
            concurrency: Maximum number of concurrent API calls.
        """
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        if model:
            self.model = model
        else:
            self.model = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
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
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_base64],
                }
            ],
            "stream": False,
        }

        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()

        data = response.json()
        return data["message"]["content"]

    def analyze_frame(self, image_path: str | Path) -> dict | str:
        """Analyze a single frame image using Ollama Vision API.

        Args:
            image_path: Path to the frame image file.

        Returns:
            Parsed analysis dict or raw string if parsing fails.

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
        return parse_llm_response(result)

    def analyze_frame_from_memory(self, frame: np.ndarray, timestamp: str) -> dict | str:
        """Analyze a frame held in memory (numpy array) using Ollama Vision API.

        Args:
            frame: Frame image as a numpy array (BGR format from OpenCV).
            timestamp: Timestamp label for logging.

        Returns:
            Parsed analysis dict or raw string if parsing fails.
        """
        import cv2

        _, buffer = cv2.imencode(".jpg", frame)
        image_data = base64.standard_b64encode(buffer.tobytes()).decode("utf-8")

        logger.info("Analyzing frame at %s (from memory)", timestamp)

        result = self._call_ollama(ANALYSIS_PROMPT, image_data)
        logger.info("Analysis complete for frame at %s", timestamp)
        return parse_llm_response(result)

    def analyze_frame_from_memory_with_prompt(
        self, frame: np.ndarray, prompt: str, timestamp: str
    ) -> dict | str:
        """Analyze a frame in memory using a custom prompt.

        Args:
            frame: Frame image as a numpy array (BGR format from OpenCV).
            prompt: Custom prompt to send to the model.
            timestamp: Timestamp label for logging.

        Returns:
            Parsed analysis dict or raw string if parsing fails.
        """
        import cv2

        _, buffer = cv2.imencode(".jpg", frame)
        image_data = base64.standard_b64encode(buffer.tobytes()).decode("utf-8")

        logger.info("Analyzing frame at %s with custom prompt", timestamp)

        result = self._call_ollama(prompt, image_data)
        logger.info("Analysis complete for frame at %s", timestamp)
        return parse_llm_response(result)

    def analyze_frames(self, image_paths: list[Path]) -> list[dict[str, str]]:
        """Analyze multiple frame images concurrently.

        Args:
            image_paths: List of paths to frame images.

        Returns:
            List of dicts with 'timestamp' (from filename) and 'analysis' keys.
            Results are returned in the same order as input paths.
        """
        results: list[dict[str, str]] = [{}] * len(image_paths)

        def _analyze_one(index: int, path: Path) -> tuple[int, dict]:
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
