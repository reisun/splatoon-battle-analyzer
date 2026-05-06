"""Single-pass highlight detection for Splatoon gameplay videos."""

import logging
import math
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from src.battle_analyzer import FRAME_ANALYSIS_PROMPT, BattleAnalyzer
from src.frame_extractor import extract_frames

ProgressCallback = Callable[[int, int, int], None]  # (phase, frames_done, frames_total)

logger = logging.getLogger(__name__)

MAX_CLIP_SECONDS = 15
MAX_TOTAL_SECONDS = 60
SCORE_GAIN_WINDOW_SECONDS = 40


def _calc_score_gain(prev_count: int | None, cur_count: int | None) -> int:
    """前フレームとのゲームカウント差分から score_gain を計算."""
    if prev_count is None or cur_count is None:
        return 1
    gain = (prev_count - cur_count) / 10 + 1
    return max(1, min(10, int(gain)))


FIRST_VALUE_FLOOR = 50
IQR_MULTIPLIER = 1.5
CONSECUTIVE_RUN_MIN = 2


def _compute_iqr_fence(deltas: list[float]) -> float:
    """減少幅リストの IQR 上限フェンスを算出する.

    Q3 + 1.5 * IQR を返す。データが不足の場合は inf を返す（全て正常扱い）。
    """
    if len(deltas) < 4:
        return math.inf
    sorted_d = sorted(deltas)
    n = len(sorted_d)
    q1 = sorted_d[n // 4]
    q3 = sorted_d[(3 * n) // 4]
    iqr = q3 - q1
    return q3 + IQR_MULTIPLIER * iqr


def _normalize_counts(
    results: list[tuple[float, dict | str]],
) -> None:
    """ゲームカウントを正規化し、raw値と正規化値の両方をdictに格納する.

    2パスアプローチ:
    Pass 1: raw値から隣接フレーム間の減少幅を全て収集し、IQR で外れ値閾値を算出
    Pass 2: 閾値を超える減少でも連続(2フレーム以上)している場合は正常と判定
            孤立した単発の急落のみ異常値として直前値で置換
    """
    for field in ("my_team_count", "enemy_team_count"):
        raw_field = f"{field}_raw"

        # --- Pass 1: raw値を収集し、隣接間の減少幅からIQRフェンスを算出 ---
        raw_values: list[tuple[int, int]] = []  # (index, value)
        dict_indices: list[int] = []
        for i, (_ts, result) in enumerate(results):
            if not isinstance(result, dict):
                continue
            dict_indices.append(i)
            val = result.get(field)
            if val is not None:
                raw_values.append((i, val))

        deltas: list[float] = []
        for j in range(1, len(raw_values)):
            prev_val = raw_values[j - 1][1]
            cur_val = raw_values[j][1]
            if cur_val < prev_val:
                deltas.append(prev_val - cur_val)

        fence = _compute_iqr_fence(deltas)

        # --- Pass 2: 正規化適用 ---
        # まず各フレームの「フェンス超え減少」フラグを計算
        raw_sequence: list[tuple[int, int | None]] = []
        for i in dict_indices:
            val = results[i][1].get(field)  # type: ignore[union-attr]
            raw_sequence.append((i, val))

        # 隣接減少幅がフェンスを超えるインデックスを特定
        large_drop_indices: set[int] = set()
        prev_val_for_flag: int | None = None
        for seq_idx, (_, val) in enumerate(raw_sequence):
            if val is None:
                continue
            if prev_val_for_flag is not None and val < prev_val_for_flag:
                delta = prev_val_for_flag - val
                if delta > fence:
                    large_drop_indices.add(seq_idx)
            prev_val_for_flag = val

        # 連続するフェンス超えを正常と判定するためのセット構築
        sustained_indices: set[int] = set()
        sorted_large = sorted(large_drop_indices)
        run_start = 0
        for k in range(1, len(sorted_large) + 1):
            if k == len(sorted_large) or sorted_large[k] - sorted_large[k - 1] > 1:
                run_length = k - run_start
                if run_length >= CONSECUTIVE_RUN_MIN:
                    for m in range(run_start, k):
                        sustained_indices.add(sorted_large[m])
                run_start = k

        # 最終的な正規化
        prev_normalized: int | None = None
        prev_raw: int | None = None
        for seq_idx, (i, _) in enumerate(raw_sequence):
            result = results[i][1]
            raw_value = result.get(field)  # type: ignore[union-attr]
            result[raw_field] = raw_value  # type: ignore[union-attr]

            if raw_value is None:
                result[field] = prev_normalized  # type: ignore[union-attr]
                continue

            if prev_normalized is None:
                normalized = 100 if raw_value < FIRST_VALUE_FLOOR else raw_value
            else:
                if raw_value > prev_normalized:
                    normalized = prev_normalized
                elif seq_idx in large_drop_indices and seq_idx not in sustained_indices:
                    normalized = prev_normalized
                else:
                    normalized = raw_value

            result[field] = normalized  # type: ignore[union-attr]
            prev_normalized = normalized
            prev_raw = raw_value


def _compute_score(result: dict) -> int:
    """4項目の掛け算でスコアを計算。デス中は半減."""
    kills = max(1, min(10, result.get("kills", 1)))
    assists = max(1, min(10, result.get("assists", 1)))
    score_gain = max(1, min(10, result.get("score_gain", 1)))
    special = max(1, min(10, result.get("special", 1)))
    score = kills * assists * score_gain * special
    if result.get("is_dead", False):
        score //= 2
    return score


@dataclass
class _ScoredFrame:
    timestamp: float
    score: int
    description: str
    raw: dict


@dataclass
class FrameAnalysis:
    timestamp_seconds: float
    score: int
    kills: int
    assists: int
    score_gain: int
    special: int
    is_dead: bool
    description: str
    my_team_color: str
    enemy_team_color: str
    my_team_count: int | None
    enemy_team_count: int | None
    my_team_count_raw: int | None
    enemy_team_count_raw: int | None


@dataclass
class HighlightSegment:
    start_seconds: float
    end_seconds: float
    peak_intensity: int
    description: str


class HighlightDetector:
    """Detect highlight segments using single-pass parallel analysis."""

    def __init__(
        self,
        analyzer: BattleAnalyzer,
        interval: float = 5,
        threshold: int = 100,
    ) -> None:
        self.analyzer = analyzer
        self.interval = interval
        self.threshold = threshold
        self.scan_summary: dict = {}
        self.all_frames: list[FrameAnalysis] = []

    def detect(
        self,
        video_path: str | Path,
        start_seconds: float | None = None,
        end_seconds: float | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> list[HighlightSegment]:
        video_path = Path(video_path)

        frames = extract_frames(
            video_path=video_path,
            interval_seconds=self.interval,
            no_save=True,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )

        scan_start = start_seconds or 0.0
        total_frames = len(frames)
        results: list[tuple[float, dict | str]] = [None] * total_frames  # type: ignore[list-item]
        done_count = [0]

        def _analyze_one(index: int) -> None:
            timestamp_sec = scan_start + index * self.interval
            ts_label = self._format_timestamp(timestamp_sec)
            try:
                result = self.analyzer.analyze_frame_from_memory_with_prompt(
                    frames[index], FRAME_ANALYSIS_PROMPT, ts_label
                )
            except Exception:
                logger.exception("Analysis failed for frame at %s", ts_label)
                result = {"kills": 1, "description": "analysis failed"}
            results[index] = (timestamp_sec, result)

            done_count[0] += 1
            if progress_callback:
                progress_callback(1, done_count[0], total_frames)

        with ThreadPoolExecutor(max_workers=self.analyzer.concurrency) as executor:
            futures = [executor.submit(_analyze_one, i) for i in range(total_frames)]
            for future in as_completed(futures):
                future.result()

        battle_count = sum(
            1 for _, r in results if isinstance(r, dict) and r.get("scene") != "other"
        )
        self.scan_summary = {
            "total_frames": total_frames,
            "battle_frames": battle_count,
        }

        scored = self._score_frames(results)
        self.all_frames = [
            FrameAnalysis(
                timestamp_seconds=f.timestamp,
                score=f.score,
                kills=max(1, min(10, f.raw.get("kills", 1))),
                assists=max(1, min(10, f.raw.get("assists", 1))),
                score_gain=max(1, min(10, f.raw.get("score_gain", 1))),
                special=max(1, min(10, f.raw.get("special", 1))),
                is_dead=f.raw.get("is_dead", False),
                description=f.raw.get("description", ""),
                my_team_color=f.raw.get("my_team_color", ""),
                enemy_team_color=f.raw.get("enemy_team_color", ""),
                my_team_count=f.raw.get("my_team_count"),
                enemy_team_count=f.raw.get("enemy_team_count"),
                my_team_count_raw=f.raw.get("my_team_count_raw"),
                enemy_team_count_raw=f.raw.get("enemy_team_count_raw"),
            )
            for f in scored
        ]
        segments = self._select_windows(scored)

        remaining = MAX_TOTAL_SECONDS
        budget_segments: list[HighlightSegment] = []
        for s in segments:
            dur = s.end_seconds - s.start_seconds
            if dur <= remaining:
                budget_segments.append(s)
                remaining -= dur

        return budget_segments

    def _score_frames(self, results: list[tuple[float, dict | str]]) -> list[_ScoredFrame]:
        sorted_results = sorted(results, key=lambda x: x[0])
        _normalize_counts(sorted_results)
        window_size = max(1, int(SCORE_GAIN_WINDOW_SECONDS / self.interval))
        recent_counts: deque[int] = deque(maxlen=window_size)
        scored: list[_ScoredFrame] = []
        for timestamp, result in sorted_results:
            score = 0
            description = ""
            raw: dict = {}
            if isinstance(result, dict):
                raw = result
                cur_count = raw.get("my_team_count")
                avg_prev = int(sum(recent_counts) / len(recent_counts)) if recent_counts else None
                raw["score_gain"] = _calc_score_gain(avg_prev, cur_count)
                if cur_count is not None:
                    recent_counts.append(cur_count)
                score = _compute_score(raw)
                description = raw.get("description", "")
            scored.append(_ScoredFrame(timestamp, score, description, raw))
        return scored

    def _select_windows(self, scored: list[_ScoredFrame]) -> list[HighlightSegment]:
        """Select best non-overlapping windows by sliding window score sum."""
        if not scored:
            return []

        window_size = max(1, int(MAX_CLIP_SECONDS / self.interval))
        n = len(scored)

        windows: list[tuple[int, int, int]] = []
        for i in range(n - window_size + 1):
            window = scored[i : i + window_size]
            total = sum(f.score for f in window)
            peak = max(f.score for f in window)
            if peak >= self.threshold:
                windows.append((i, total, peak))

        if not windows:
            for i, frame in enumerate(scored):
                if frame.score >= self.threshold:
                    windows.append((i, frame.score, frame.score))

        if not windows:
            return []

        windows.sort(key=lambda w: w[1], reverse=True)

        selected: list[int] = []
        used: set[int] = set()
        for start_idx, _total, _peak in windows:
            end_idx = min(start_idx + window_size - 1, n - 1)
            frame_indices = set(range(start_idx, end_idx + 1))
            if frame_indices & used:
                continue
            selected.append(start_idx)
            used.update(frame_indices)

        selected.sort()

        segments: list[HighlightSegment] = []
        for start_idx in selected:
            end_idx = min(start_idx + window_size - 1, n - 1)
            window = scored[start_idx : end_idx + 1]
            descriptions = [f.description for f in window if f.description]
            segments.append(
                HighlightSegment(
                    start_seconds=scored[start_idx].timestamp,
                    end_seconds=scored[end_idx].timestamp + self.interval,
                    peak_intensity=sum(f.score for f in window),
                    description=self._summarize_descriptions(descriptions),
                )
            )

        return segments

    @staticmethod
    def _summarize_descriptions(descriptions: list[str]) -> str:
        if not descriptions:
            return ""
        unique = list(dict.fromkeys(descriptions))
        return "; ".join(unique[:3])

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}m{secs:02d}s"
