"""Battle analysis module using Claude Code CLI.

Sends frame images to Claude Vision via CLI subprocess and extracts battle status.
Supports concurrent calls for improved throughput.
"""

import json
import logging
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "haiku"


def _save_temp_frame(frame: np.ndarray) -> str:
    """Save frame to a temporary JPEG file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    cv2.imwrite(path, frame)
    return path


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

STAGE1_PROMPT = """Analyze this Splatoon gameplay screenshot. Key UI elements:
- Top center: match timer
- Left of timer: 4 ally icons (grayed out = dead)
- Right of timer: 4 enemy icons (grayed out = dead)
- Below timer: game progress bars (lower count = winning)
- Bottom center: kill log (multiple names = multi-kill streak)

Score each factor from 1 (nothing notable) to 10 (extremely intense):
- kills: Did the player eliminate enemies? Check kill log at bottom center.
- assists: Did the player assist in eliminating enemies?
- score_gain: Is the player's team score increasing noticeably?
- clutch: Is the team losing AND the score is NOT improving? (high = desperate/tense situation)
- special: Is a special weapon being activated or its effects visible?

Answer in JSON only:
{"scene": "battle", "kills": 1, "assists": 1, "score_gain": 1, "clutch": 1, "special": 1, "reason": "brief description"}
scene must be one of: battle, lobby, result, other
All scores must be 1-10. If unsure, use 1."""

STAGE2_PROMPT = """Analyze this Splatoon battle screenshot in detail. Key UI elements:
- Top center: match timer
- Left of timer: 4 ally icons (grayed out = dead)
- Right of timer: 4 enemy icons (grayed out = dead)
- Below timer: game progress bars (lower count = winning)
- Bottom center: kill log (multiple names = multi-kill streak)

Score each factor from 1 (nothing notable) to 10 (extremely intense):
- kills: Did the player eliminate enemies? Check kill log at bottom center.
- assists: Did the player assist in eliminating enemies?
- score_gain: Is the player's team score increasing noticeably?
- clutch: Is the team losing AND the score is NOT improving? (high = desperate/tense situation)
- special: Is a special weapon being activated or its effects visible?

Answer in JSON only:
{"kills": 1, "assists": 1, "score_gain": 1, "clutch": 1, "special": 1, "description": "what is happening"}
All scores must be 1-10. If unsure, use 1."""


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
    """Analyzes Splatoon battle frames using Claude Code CLI."""

    def __init__(
        self, model: str | None = None, concurrency: int = 4, timeout: int = 120
    ) -> None:
        """Initialize the analyzer.

        Args:
            model: Claude model name (default: env CLAUDE_MODEL or "haiku").
            concurrency: Maximum number of concurrent CLI calls.
            timeout: Timeout in seconds for each CLI call.
        """
        if model:
            self.model = model
        else:
            self.model = os.environ.get("CLAUDE_MODEL", DEFAULT_MODEL)
        self.concurrency = concurrency
        self.timeout = timeout

    def _call_cli(self, prompt: str, image_path: str) -> str:
        """Call Claude Code CLI with an image file.

        Args:
            prompt: Text prompt to send.
            image_path: Path to the image file for analysis.

        Returns:
            Response text from the model.

        Raises:
            RuntimeError: If the CLI call fails.
        """
        full_prompt = (
            f"Read the image file at {image_path} and analyze it. "
            f"Respond with ONLY the requested format, no extra text.\n\n{prompt}"
        )

        result = subprocess.run(
            [
                "claude",
                "-p",
                full_prompt,
                "--dangerously-skip-permissions",
                "--model",
                self.model,
            ],
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip() or "Unknown error"
            raise RuntimeError(f"Claude CLI failed: {error_msg}")

        return result.stdout.strip()

    def analyze_frame(self, image_path: str | Path) -> dict | str:
        """Analyze a single frame image.

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

        logger.info("Analyzing frame: %s", image_path.name)
        result = self._call_cli(ANALYSIS_PROMPT, str(image_path.resolve()))
        logger.info("Analysis complete for: %s", image_path.name)
        return parse_llm_response(result)

    def analyze_frame_from_memory(self, frame: np.ndarray, timestamp: str) -> dict | str:
        """Analyze a frame held in memory (numpy array).

        Args:
            frame: Frame image as a numpy array (BGR format from OpenCV).
            timestamp: Timestamp label for logging.

        Returns:
            Parsed analysis dict or raw string if parsing fails.
        """
        tmp_path = _save_temp_frame(frame)
        try:
            logger.info("Analyzing frame at %s", timestamp)
            result = self._call_cli(ANALYSIS_PROMPT, tmp_path)
            logger.info("Analysis complete for frame at %s", timestamp)
            return parse_llm_response(result)
        finally:
            os.unlink(tmp_path)

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
        tmp_path = _save_temp_frame(frame)
        try:
            logger.info("Analyzing frame at %s with custom prompt", timestamp)
            result = self._call_cli(prompt, tmp_path)
            logger.info("Analysis complete for frame at %s", timestamp)
            return parse_llm_response(result)
        finally:
            os.unlink(tmp_path)

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
    """Check if Claude Code CLI is available and authenticated.

    Returns:
        True if CLI is available, False otherwise.
    """
    try:
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
