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


FRAME_ANALYSIS_PROMPT = """スプラトゥーンのゲーム画面スクリーンショットを詳細に分析してください。

■ UI要素の位置:
- 画面上部中央: 試合タイマー（残り時間）
- タイマーの左側: 自チームの色のイカランプ4つ（グレーアウト＝デス中）
- タイマーの右側: 相手チーム色のイカランプ4つ（グレーアウト＝デス中）
- タイマーの下: ゲームカウント。自チームの色と相手チームの色の２つ
（カウントの上に小さく「のこり」と表示されている。先に0にすると勝ち=カウントが少ない方が勝っている）
- ゲームカウント回りの小さな数字: ルールごとに仕様が異なるため割愛。無視する
- 画面下部中央: 直近で倒したプレイヤーの名前（複数の名前＝連続キル）

■ チームカラーの確認方法:
自チームの色はタイマー左側のイカランプの色で確認してください。
相手チームの色はタイマー右側のイカランプの色で確認してください。

■ 各項目を1（特になし）〜10（非常に激しい）でスコアリング:
- kills: プレイヤーが敵を倒したか？画面下部の倒したプレイヤーの名前を確認
- assists: プレイヤーがキルをアシストしたか？
- special: スペシャルウェポンが発動中、またはその効果が見えるか？
- is_dead: 自プレイヤーがデス中か？（画面が暗転・復帰待ち状態ならtrue）

JSONのみで回答してください:
{"kills": 1, "assists": 1, "special": 1, "is_dead": false, "my_team_color": "自チームの色", "enemy_team_color": "相手チームの色", "my_team_count": null, "enemy_team_count": null, "description": "現在の状況の説明"}
スコアは全て1-10。不明なら1。is_deadはtrue/false。ゲームカウントが不明ならnull。"""


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

    def __init__(self, model: str | None = None, concurrency: int = 4, timeout: int = 120) -> None:
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
        result = self._call_cli(FRAME_ANALYSIS_PROMPT, str(image_path.resolve()))
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
            result = self._call_cli(FRAME_ANALYSIS_PROMPT, tmp_path)
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
            futures = [executor.submit(_analyze_one, i, path) for i, path in enumerate(image_paths)]
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
