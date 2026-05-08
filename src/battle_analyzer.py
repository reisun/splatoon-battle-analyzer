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


def _save_temp_frame(frame: np.ndarray) -> str:
    """Save frame to a shared temporary JPEG file accessible by Agent Gateway."""
    os.makedirs(SHARED_TEMP_DIR, exist_ok=True)
    fd, path = tempfile.mkstemp(suffix=".jpg", dir=SHARED_TEMP_DIR)
    os.close(fd)
    cv2.imwrite(path, frame)
    return path


FRAME_ANALYSIS_SYSTEM_PROMPT = """\
あなたはスプラトゥーンのゲーム画面を分析する専門AIです。
画像を受け取り、以下のルールに従ってJSON形式で回答してください。

■ UI要素の位置:
- 画面上部中央: 試合タイマー（残り時間）
- タイマーの左側: 自チームの色のイカランプ4つ（グレーアウト＝デス中）
- タイマーの右側: 相手チーム色のイカランプ4つ（グレーアウト＝デス中）
- 画面の右上端：スペシャルゲージ
- タイマーの下: ゲームカウント。自チームの色と相手チームの色の２つ\
補足：カウントの上に小さく「のこり」と表示されている。先に0にすると勝ち=カウントが少ない方が勝っている
- ゲームカウント回りの小さな数字: ルールごとに仕様が異なり複雑なため無視する。混同注意。
- 画面下部中央: 直近で倒したプレイヤーの名前「◯◯ をたおした！」と表示される（複数の名前＝連続キル）\
補足：味方の名前が常時表示されているため、誤認しないこと。「◯◯ をたおした！」の表示があるかどうかでキルを判断すること。

■ チームカラーの確認方法:
自チームの色はタイマー左側のイカランプの色で確認してください。
相手チームの色はタイマー右側のイカランプの色で確認してください。

■ 各項目:
- kills: 「◯◯ をたおした！」の表示が複数ある場合はその数をカウントする。表示がない場合は0
- special: 自プレイヤーがスペシャルウェポンが発動中、またはその効果が見えるか？
- is_dead: 自プレイヤーがデス中か？(true/false)（画面が暗転・復帰待ち状態ならtrue）
- remaining_time: 画面上部中央のタイマーに表示されている残り時間を必ず「M:SS」形式で記録すること
- my_team_color および enemy_team_color: タイマー左側と右側のイカランプの色を記録すること（例: "オレンジ", "ブルー"）
- my_team_count および enemy_team_count: 自チームと相手チームのカウントを記録すること(ゲームカウントが不明ならnull。)

■ 出力フォーマット（JSONのみ、他のテキスト不可）:
{"kills": 1, "special": false, "is_dead": false, "remaining_time": "4:30",
"my_team_color": "オレンジ", "enemy_team_color": "ブルー",
"my_team_count": 85, "enemy_team_count": 72}
"""

FRAME_ANALYSIS_PROMPT = FRAME_ANALYSIS_SYSTEM_PROMPT


def build_frame_prompt(match_duration: str | None = None) -> str:
    """試合時間コンテキスト付きのフレーム分析プロンプトを生成する."""
    if match_duration is None:
        return "この画像を分析してJSON形式で回答してください。"
    return f"この試合は{match_duration}マッチです。この画像を分析してJSON形式で回答してください。"


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
            build_frame_prompt(),
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
                build_frame_prompt(),
                tmp_path,
                system_prompt=FRAME_ANALYSIS_SYSTEM_PROMPT,
            )
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
