"""Battle analysis module using Gemini Vision API.

Sends frame images to Gemini 2.5 Flash-Lite and extracts battle status.
Supports concurrent calls for improved throughput.
"""

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash-lite"


def _half_resize(frame: np.ndarray) -> np.ndarray:
    """Resize frame to half resolution for token reduction."""
    h, w = frame.shape[:2]
    return cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA)


def _encode_frame_jpeg(frame: np.ndarray) -> bytes:
    """Encode a numpy frame to JPEG bytes."""
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Failed to encode frame to JPEG")
    return buf.tobytes()


UPPER_HALF_SYSTEM_PROMPT = """\
あなたはスプラトゥーンのゲーム画面の上部を分析する専門AIです。
この画像はゲーム画面の上部30%・中央60%幅をクロップしたものです。
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
- ゲームカウント回りの小さな数字:
    ルールごとに仕様が異なり複雑なため無視する。混同注意。
- タイマーの下（ゲームカウントの周辺）:
    ヤグラ・ホコルールでは、ゲームカウントが動くためのレール（横棒）が表示される
    エリア・ナワバリでは表示されない

■ 各項目:
- my_team_count / enemy_team_count: (null, 0~100) 自チーム・相手チームのカウント。不明瞭な場合はnull
- has_count_rail: (true/false) ゲームカウントのレール（横棒）が表示されているか。不明瞭な場合はfalse

■ 出力フォーマット（JSONのみ、他のテキスト不可）:
{
  "my_team_count": number | null,
  "enemy_team_count": number | null,
  "has_count_rail": boolean
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

UPPER_HALF_USER_PROMPT = "この画像（ゲーム画面の上部中央）を分析してJSON形式で回答してください。"
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


def _create_client() -> genai.Client:
    """Create a Gemini API client."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")
    return genai.Client(api_key=api_key)


class BattleAnalyzer:
    """Analyzes Splatoon battle frames using Gemini Vision API."""

    def __init__(self, model: str | None = None, concurrency: int = 4, timeout: int = 120) -> None:
        if model:
            self.model = model
        else:
            self.model = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        self.concurrency = concurrency
        self.timeout = timeout
        self.client = _create_client()

    def _call_gemini(
        self,
        prompt: str,
        image_bytes: bytes | None = None,
        system_prompt: str | None = None,
        image_bytes_list: list[bytes] | None = None,
    ) -> str:
        """Call Gemini API with image data.

        Returns:
            Response text from the model.

        Raises:
            RuntimeError: If the API call fails.
        """
        contents: list = []

        if image_bytes_list:
            for img_data in image_bytes_list:
                contents.append(types.Part.from_bytes(data=img_data, mime_type="image/jpeg"))
        elif image_bytes:
            contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))

        contents.append(prompt)

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            http_options=types.HttpOptions(timeout=self.timeout * 1000),
        )
        if system_prompt:
            config.system_instruction = system_prompt

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {e}")

        return response.text or ""

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
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        image_bytes = _encode_frame_jpeg(frame)
        result = self._call_gemini(
            FULL_FRAME_USER_PROMPT,
            image_bytes,
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
        image_bytes = _encode_frame_jpeg(frame)
        logger.info("Analyzing frame at %s", timestamp)
        result = self._call_gemini(
            FULL_FRAME_USER_PROMPT,
            image_bytes,
            system_prompt=FRAME_ANALYSIS_SYSTEM_PROMPT,
        )
        logger.info("Analysis complete for frame at %s", timestamp)
        return parse_llm_response(result)

    def _analyze_cropped(
        self,
        frame: np.ndarray,
        user_prompt: str,
        system_prompt: str,
        timestamp: str,
    ) -> dict | str:
        """Analyze a cropped frame region."""
        image_bytes = _encode_frame_jpeg(frame)
        result = self._call_gemini(user_prompt, image_bytes, system_prompt=system_prompt)
        return parse_llm_response(result)

    def _analyze_cropped_multi(
        self,
        frames: list[np.ndarray],
        user_prompt: str,
        system_prompt: str,
        timestamp: str,
    ) -> dict | str:
        """Analyze multiple cropped frame regions in a single request."""
        image_bytes_list = [_encode_frame_jpeg(f) for f in frames]
        result = self._call_gemini(
            user_prompt, system_prompt=system_prompt, image_bytes_list=image_bytes_list
        )
        return parse_llm_response(result)

    @staticmethod
    def _merge_results(upper: dict | str, lower: dict | str) -> dict:
        """Merge upper/lower half analysis results into a single dict."""
        merged: dict = {}
        if isinstance(upper, dict):
            merged.update(upper)
        else:
            merged.update({"my_team_count": None, "enemy_team_count": None, "has_count_rail": False})
        if isinstance(lower, dict):
            merged.update(lower)
        else:
            merged.update({"kills": 0, "is_dead": False})
        return merged

    def analyze_frame_split(self, frame: np.ndarray, timestamp: str) -> dict:
        """Analyze a frame by splitting into upper/lower halves in parallel."""
        frame = _half_resize(frame)
        h, w = frame.shape[:2]
        upper = frame[: int(h * 0.3), :, :]
        upper_half = upper[:, int(w * 0.2) : int(w * 0.8), :]
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
        h, w = frame.shape[:2]
        upper = frame[: int(h * 0.3), :, :]
        upper_half = upper[:, int(w * 0.2) : int(w * 0.8), :]

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
        image_bytes = _encode_frame_jpeg(frame)
        logger.info("Analyzing frame at %s with custom prompt", timestamp)
        result = self._call_gemini(
            prompt, image_bytes, system_prompt=FRAME_ANALYSIS_SYSTEM_PROMPT
        )
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
            futures = [executor.submit(_analyze_one, i, path) for i, path in enumerate(image_paths)]
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return results


def check_api_key_available() -> bool:
    """Check if Gemini API key is configured.

    Returns:
        True if GEMINI_API_KEY is set, False otherwise.
    """
    return bool(os.environ.get("GEMINI_API_KEY"))
