"""Battle analysis module using Agent Gateway HTTP API.

Sends frame images to Claude Vision via Agent Gateway and extracts battle status.
Supports concurrent calls for improved throughput.
"""

import json
import logging
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "haiku"
DEFAULT_AGENT_GATEWAY_URL = "http://llm-internal-proxy/agent"
POLL_INTERVAL = 2.0
MAX_POLL_ATTEMPTS = 900


SHARED_TEMP_DIR = os.environ.get("SHARED_TEMP_DIR", "/shared-data/tmp")


def _half_resize(frame: np.ndarray) -> np.ndarray:
    """Resize frame to half resolution for token reduction."""
    h, w = frame.shape[:2]
    return cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA)


def _save_temp_frame(frame: np.ndarray) -> str:
    """Save frame to a shared temporary JPEG file accessible by Agent Gateway."""
    os.makedirs(SHARED_TEMP_DIR, exist_ok=True)
    fd, path = tempfile.mkstemp(suffix=".jpg", dir=SHARED_TEMP_DIR)
    os.close(fd)
    cv2.imwrite(path, frame)
    return path


UPPER_HALF_SYSTEM_PROMPT = """\
あなたはスプラトゥーンのゲーム画面の上部を分析する専門AIです。
この画像はゲーム画面の上部30%のみをクロップしたものです。
以下のルールに従ってJSON形式で回答してください。

■ UI要素の位置:
- 画面上部中央:
    試合タイマー
- タイマーの左側:
    自チームの色のイカランプ4つ
- タイマーの右側:
    相手チーム色のイカランプ4つ
- タイマーの下:
    ゲームカウント。自チームの色と相手チームの色の２つ
    カウントの上に小さく「のこり」と表示されている
    ※ ヤグラ・ホコルールの場合:
      自チームのカウントバーは中央から右へ進行する
      敵チームのカウントバーは中央から左へ進行する
      （オブジェクトを敵陣に押し込むゲーム性のため）
- ゲームカウント回りの小さな数字:
    ルールごとに仕様が異なり複雑なため無視する。混同注意。

■ 各項目:
- my_team_count / enemy_team_count: (null, 0~100) 自チーム・相手チームのカウント。不明瞭な場合はnull

■ 出力フォーマット（JSONのみ、他のテキスト不可）:
{
  "my_team_count": number | null,
  "enemy_team_count": number | null
}
"""

LOWER_HALF_SYSTEM_PROMPT = """\
あなたはスプラトゥーンのゲーム画面の下部を分析する専門AIです。
この画像はゲーム画面の下部30%のみをクロップしたものです。
以下のルールに従ってJSON形式で回答してください。

■ UI要素の位置:
- 画面下部中央:
    直近で倒したプレイヤーの名前「◯◯ をたおした！」と表示される
    薄い透過黒背景に白文字
    周囲に別の表示が重なることが多いため混同注意。
    「◯◯ をたおした！」の完全一致で判断すること
- 画面右下:
    デス中は「復活まであと◯◯秒」と表示される

■ 各項目:
- kills: (0~4)「◯◯ をたおした！」の完全一致の数で判断する。複数ある場合はその数をカウントする。不明瞭な場合はカウントしない
- is_dead: (true/false) 自プレイヤーがデス中か？「復活まであと」の表示や画面暗転があればtrue。不明瞭な場合はfalse

■ 出力フォーマット（JSONのみ、他のテキスト不可）:
{
  "kills": number,
  "is_dead": boolean
}
"""

UPPER_HALF_USER_PROMPT = "この画像（ゲーム画面の上部30%）を分析してJSON形式で回答してください。"
LOWER_HALF_USER_PROMPT = "この画像（ゲーム画面の下部30%）を分析してJSON形式で回答してください。"

FRAME_ANALYSIS_SYSTEM_PROMPT = UPPER_HALF_SYSTEM_PROMPT
FULL_FRAME_USER_PROMPT = "この画像を分析してJSON形式で回答してください。"


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


def _get_gateway_url() -> str:
    """Get Agent Gateway base URL from environment."""
    return os.environ.get("AGENT_GATEWAY_URL", DEFAULT_AGENT_GATEWAY_URL)


class BattleAnalyzer:
    """Analyzes Splatoon battle frames using Agent Gateway HTTP API."""

    def __init__(self, model: str | None = None, concurrency: int = 4, timeout: int = 120) -> None:
        """Initialize the analyzer.

        Args:
            model: Claude model name (default: env CLAUDE_MODEL or "haiku").
            concurrency: Maximum number of concurrent API calls.
            timeout: Timeout in seconds for each API call.
        """
        if model:
            self.model = model
        else:
            self.model = os.environ.get("CLAUDE_MODEL", DEFAULT_MODEL)
        self.concurrency = concurrency
        self.timeout = timeout
        self.gateway_url = _get_gateway_url()

    def _call_agent_gateway(
        self,
        prompt: str,
        image_path: str,
        system_prompt: str | None = None,
    ) -> str:
        """Call Agent Gateway API with an image file.

        Args:
            prompt: Text prompt to send.
            image_path: Path to the image file for analysis.
            system_prompt: Optional system prompt for the model.

        Returns:
            Response text from the model.

        Raises:
            RuntimeError: If the API call fails.
        """
        run_url = f"{self.gateway_url}/run"
        payload: dict = {
            "agent": "claude",
            "prompt": prompt,
            "model": self.model,
            "timeout": self.timeout,
            "image_path": image_path,
            "agent_options": {"allowed_tools": ["Read"]},
        }
        if system_prompt:
            payload["system_prompt"] = system_prompt

        try:
            response = requests.post(run_url, json=payload, timeout=30)
        except requests.RequestException as e:
            raise RuntimeError(f"Agent Gateway request failed: {e}")

        if response.status_code != 202:
            raise RuntimeError(
                f"Agent Gateway returned status {response.status_code}: {response.text}"
            )

        job_data = response.json()
        job_id = job_data["job_id"]

        return self._poll_job(job_id)

    def _poll_job(self, job_id: str) -> str:
        """Poll Agent Gateway for job completion.

        Args:
            job_id: The job ID to poll.

        Returns:
            Result text from the completed job.

        Raises:
            RuntimeError: If the job fails or times out.
        """
        status_url = f"{self.gateway_url}/jobs/{job_id}"

        for _ in range(MAX_POLL_ATTEMPTS):
            time.sleep(POLL_INTERVAL)

            try:
                response = requests.get(status_url, timeout=10)
            except requests.RequestException as e:
                raise RuntimeError(f"Agent Gateway poll failed: {e}")

            if response.status_code != 200:
                raise RuntimeError(
                    f"Agent Gateway status check returned {response.status_code}: {response.text}"
                )

            data = response.json()
            status = data["status"]

            if status == "done":
                return data["result"]
            elif status == "failed":
                error_msg = data.get("error", "Unknown error")
                raise RuntimeError(f"Agent Gateway job failed: {error_msg}")

        raise RuntimeError("Agent Gateway job timed out waiting for completion")

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
        result = self._call_agent_gateway(
            FULL_FRAME_USER_PROMPT,
            str(image_path.resolve()),
            system_prompt=FRAME_ANALYSIS_SYSTEM_PROMPT,
        )
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
            result = self._call_agent_gateway(
                FULL_FRAME_USER_PROMPT,
                tmp_path,
                system_prompt=FRAME_ANALYSIS_SYSTEM_PROMPT,
            )
            logger.info("Analysis complete for frame at %s", timestamp)
            return parse_llm_response(result)
        finally:
            os.unlink(tmp_path)

    def _analyze_cropped(
        self,
        frame: np.ndarray,
        user_prompt: str,
        system_prompt: str,
        timestamp: str,
    ) -> dict | str:
        """Analyze a cropped frame region."""
        tmp_path = _save_temp_frame(frame)
        try:
            result = self._call_agent_gateway(user_prompt, tmp_path, system_prompt=system_prompt)
            return parse_llm_response(result)
        finally:
            os.unlink(tmp_path)

    @staticmethod
    def _merge_results(upper: dict | str, lower: dict | str) -> dict:
        """Merge upper/lower half analysis results into a single dict."""
        merged: dict = {}
        if isinstance(upper, dict):
            merged.update(upper)
        else:
            merged.update({"my_team_count": None, "enemy_team_count": None})
        if isinstance(lower, dict):
            merged.update(lower)
        else:
            merged.update({"kills": 0, "is_dead": False})
        return merged

    def analyze_frame_split(self, frame: np.ndarray, timestamp: str) -> dict:
        """Analyze a frame by splitting into upper/lower halves in parallel."""
        frame = _half_resize(frame)
        h = frame.shape[0]
        upper_half = frame[: int(h * 0.3), :, :]
        lower_half = frame[int(h * 0.7) :, :, :]

        logger.info("Analyzing frame at %s (split mode)", timestamp)
        with ThreadPoolExecutor(max_workers=2) as executor:
            upper_future = executor.submit(
                self._analyze_cropped,
                upper_half,
                UPPER_HALF_USER_PROMPT,
                UPPER_HALF_SYSTEM_PROMPT,
                timestamp,
            )
            lower_future = executor.submit(
                self._analyze_cropped,
                lower_half,
                LOWER_HALF_USER_PROMPT,
                LOWER_HALF_SYSTEM_PROMPT,
                timestamp,
            )
            upper_result = upper_future.result()
            lower_result = lower_future.result()

        merged = self._merge_results(upper_result, lower_result)
        logger.info("Analysis complete for frame at %s (split mode)", timestamp)
        return merged

    def analyze_frame_upper_only(self, frame: np.ndarray, timestamp: str) -> dict:
        """Analyze only the upper half (game count only, skip kills/death)."""
        frame = _half_resize(frame)
        h = frame.shape[0]
        upper_half = frame[: int(h * 0.3), :, :]

        logger.info("Analyzing frame at %s (upper-only mode)", timestamp)
        upper_result = self._analyze_cropped(
            upper_half,
            UPPER_HALF_USER_PROMPT,
            UPPER_HALF_SYSTEM_PROMPT,
            timestamp,
        )
        merged = self._merge_results(upper_result, {})
        logger.info("Analysis complete for frame at %s (upper-only mode)", timestamp)
        return merged

    def analyze_frame_lower_only(self, frame: np.ndarray, timestamp: str) -> dict:
        """Analyze only the lower half (skip game count for Nawabari)."""
        frame = _half_resize(frame)
        h = frame.shape[0]
        lower_half = frame[int(h * 0.7) :, :, :]

        logger.info("Analyzing frame at %s (lower-only mode)", timestamp)
        lower_result = self._analyze_cropped(
            lower_half,
            LOWER_HALF_USER_PROMPT,
            LOWER_HALF_SYSTEM_PROMPT,
            timestamp,
        )
        merged = self._merge_results({}, lower_result)
        logger.info("Analysis complete for frame at %s (lower-only mode)", timestamp)
        return merged

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
            result = self._call_agent_gateway(
                prompt, tmp_path, system_prompt=FRAME_ANALYSIS_SYSTEM_PROMPT
            )
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
    """Check if Agent Gateway is available.

    Returns:
        True if Agent Gateway is reachable, False otherwise.
    """
    gateway_url = _get_gateway_url()
    health_url = gateway_url.rsplit("/agent", 1)[0] + "/health"

    try:
        response = requests.get(health_url, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False
