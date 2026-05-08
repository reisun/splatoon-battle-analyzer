"""Single-pass highlight detection for Splatoon gameplay videos."""

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


from src.battle_analyzer import BattleAnalyzer
from src.frame_extractor import extract_frames
from src.scoring_config import ScoringConfig, load_scoring_config

ProgressCallback = Callable[[int, int, int], None]  # (phase, frames_done, frames_total)

logger = logging.getLogger(__name__)

MAX_CLIP_SECONDS = 15
MAX_TOTAL_SECONDS = 60


def _calc_score_count_gain(cur_count: int | None, future_avg: int | None) -> int:
    """現時点と未来平均のゲームカウント差分から score_count_gain を計算."""
    if cur_count is None or future_avg is None:
        return 0
    gain = (cur_count - future_avg) / 10
    return max(0, min(10, int(gain)))


FIRST_VALUE_FLOOR = 50
MEDIAN_RADIUS = 2


def _median_smooth(
    values: list[int | None], radius: int = MEDIAN_RADIUS
) -> list[int | None]:
    """スライディングウィンドウのメディアンで外れ値を平滑化する.

    None値はスキップ（出力もNone）。各位置で前後radius個の範囲内の
    非null値からメディアンを計算して置換する。
    """
    n = len(values)
    result: list[int | None] = list(values)
    for i in range(n):
        if values[i] is None:
            continue
        window: list[int] = []
        for j in range(max(0, i - radius), min(n, i + radius + 1)):
            if values[j] is not None:
                window.append(values[j])
        window.sort()
        result[i] = window[len(window) // 2]
    return result


def _isotonic_decreasing(values: list[int | None]) -> list[int | None]:
    """PAVアルゴリズムによる単調非増加回帰（L2最適）.

    None値はスキップし、非null値に対してのみ回帰を適用。
    ブロック結合時に加重平均を使用し、全体として二乗誤差最小の
    単調非増加系列を返す。
    """
    non_null = [(i, v) for i, v in enumerate(values) if v is not None]
    if not non_null:
        return list(values)

    blocks: list[tuple[float, int, list[int]]] = []
    for idx, val in non_null:
        blocks.append((float(val), 1, [idx]))
        while len(blocks) >= 2:
            if blocks[-1][0] / blocks[-1][1] <= blocks[-2][0] / blocks[-2][1]:
                break
            s1, c1, i1 = blocks.pop()
            s2, c2, i2 = blocks.pop()
            blocks.append((s1 + s2, c1 + c2, i2 + i1))

    result: list[int | None] = list(values)
    for total, count, indices in blocks:
        fitted = int(round(total / count))
        for i in indices:
            result[i] = fitted
    return result


def _normalize_counts(
    results: list[tuple[float, dict | str]],
) -> None:
    """ゲームカウントを正規化し、raw値と正規化値の両方をdictに格納する.

    3ステップ:
    1. 全フレームの raw 値を収集
    2. メディアンフィルタで外れ値を平滑化
    3. 先頭値補正 + PAV等張回帰で単調非増加を保証 + null埋め
    """
    for field in ("my_team_count", "enemy_team_count"):
        raw_field = f"{field}_raw"

        dict_indices: list[int] = []
        raw_values: list[int | None] = []
        for i, (_ts, result) in enumerate(results):
            if not isinstance(result, dict):
                continue
            dict_indices.append(i)
            raw_values.append(result.get(field))

        smoothed = _median_smooth(raw_values)

        for k, v in enumerate(smoothed):
            if v is not None:
                if v < FIRST_VALUE_FLOOR:
                    smoothed[k] = 100
                break

        fitted = _isotonic_decreasing(smoothed)

        prev_val: int | None = None
        for seq_idx, di in enumerate(dict_indices):
            result = results[di][1]
            result[raw_field] = raw_values[seq_idx]
            val = fitted[seq_idx]
            if val is not None:
                prev_val = val
                result[field] = val
            else:
                result[field] = prev_val


@dataclass
class ScoreBreakdown:
    score: int
    score_kills: int
    score_count_gain: int
    score_dead: int


def _compute_score(result: dict, cfg: ScoringConfig | None = None) -> ScoreBreakdown:
    """重み付き加算でスコアを計算し内訳を返す."""
    if cfg is None:
        cfg = load_scoring_config()
    if result.get("is_dead", False):
        score_dead = int(cfg.death_penalty)
        return ScoreBreakdown(score=score_dead, score_kills=0, score_count_gain=0, score_dead=score_dead)
    kills = max(0, min(4, result.get("kills", 0)))
    gain = max(0, min(10, result.get("score_count_gain", 0)))
    score_kills = int(kills * cfg.weights.kills)
    score_count_gain = int(gain * cfg.weights.score_count_gain)
    return ScoreBreakdown(
        score=score_kills + score_count_gain, score_kills=score_kills,
        score_count_gain=score_count_gain, score_dead=0,
    )


@dataclass
class _ScoredFrame:
    timestamp: float
    score: int
    breakdown: ScoreBreakdown
    raw: dict


@dataclass
class FrameAnalysis:
    timestamp_seconds: float
    score: int
    score_kills: int
    score_count_gain: int
    score_dead: int
    my_team_count: int | None
    enemy_team_count: int | None
    kills: int
    is_dead: bool
    my_team_count_raw: int | None
    enemy_team_count_raw: int | None


@dataclass
class HighlightSegment:
    start_seconds: float
    end_seconds: float
    peak_intensity: int


class HighlightDetector:
    """Detect highlight segments using single-pass parallel analysis."""

    def __init__(
        self,
        analyzer: BattleAnalyzer,
        interval: float = 5,
    ) -> None:
        self.analyzer = analyzer
        self.interval = interval
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
                result = self.analyzer.analyze_frame_split(frames[index], ts_label)
            except Exception:
                logger.exception("Analysis failed for frame at %s", ts_label)
                result = {"kills": 0}
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
                score_kills=f.breakdown.score_kills,
                score_count_gain=f.breakdown.score_count_gain,
                score_dead=f.breakdown.score_dead,
                my_team_count=f.raw.get("my_team_count"),
                enemy_team_count=f.raw.get("enemy_team_count"),
                kills=max(0, min(4, f.raw.get("kills", 0))),
                is_dead=f.raw.get("is_dead", False),
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
        cfg = load_scoring_config()
        sorted_results = sorted(results, key=lambda x: x[0])
        _normalize_counts(sorted_results)
        window_size = max(1, int(cfg.score_count_gain_window_seconds / self.interval))

        counts: list[int | None] = []
        for _, result in sorted_results:
            if isinstance(result, dict):
                counts.append(result.get("my_team_count"))
            else:
                counts.append(None)

        scored: list[_ScoredFrame] = []
        for i, (timestamp, result) in enumerate(sorted_results):
            breakdown = ScoreBreakdown(score=0, score_kills=0, score_count_gain=0, score_dead=0)
            raw: dict = {}
            if isinstance(result, dict):
                raw = result
                cur_count = counts[i]
                future_slice = [c for c in counts[i + 1 : i + 1 + window_size] if c is not None]
                future_avg = int(sum(future_slice) / len(future_slice)) if future_slice else None
                raw["score_count_gain"] = _calc_score_count_gain(cur_count, future_avg)
                breakdown = _compute_score(raw, cfg)
            scored.append(_ScoredFrame(timestamp, breakdown.score, breakdown, raw))
        return scored

    def _select_windows(self, scored: list[_ScoredFrame]) -> list[HighlightSegment]:
        """1区間選定→スコア0化→先頭から再スキャンを繰り返して区間を選出."""
        if not scored:
            return []

        window_size = max(1, int(MAX_CLIP_SECONDS / self.interval))
        n = len(scored)
        max_segments = int(MAX_TOTAL_SECONDS // MAX_CLIP_SECONDS)
        scores = [f.score for f in scored]

        segments: list[HighlightSegment] = []
        for _ in range(max_segments):
            best_idx = -1
            best_total = 0
            for i in range(n - window_size + 1):
                total = sum(scores[i : i + window_size])
                if total > best_total:
                    best_total = total
                    best_idx = i

            if best_idx < 0 or best_total <= 0:
                break

            end_idx = min(best_idx + window_size - 1, n - 1)
            segments.append(
                HighlightSegment(
                    start_seconds=scored[best_idx].timestamp,
                    end_seconds=scored[end_idx].timestamp + self.interval,
                    peak_intensity=best_total,
                )
            )

            for i in range(best_idx, end_idx + 1):
                scores[i] = 0

        segments.sort(key=lambda s: s.start_seconds)
        return segments

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}m{secs:02d}s"
