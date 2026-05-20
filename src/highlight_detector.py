"""Highlight detection for Splatoon gameplay videos.

Ranked (5min): Phase A (upper 15s) -> Phase B (lower 5s, gain>1 regions).
Nawabari (3min): Single pass lower-only at 5s intervals.
"""

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


def _calc_score_count_gain(cur_count: int | None, future_avg: float | None) -> float:
    """現時点と未来平均のゲームカウント差分を返す."""
    if cur_count is None or future_avg is None:
        return 0.0
    return max(0.0, float(cur_count - future_avg))


FIRST_VALUE_FLOOR = 50
MEDIAN_RADIUS = 2


def _median_smooth(values: list[int | None], radius: int = MEDIAN_RADIUS) -> list[int | None]:
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


DROP_THRESHOLD = 10


def _pav_segment(non_null: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """単一区間に対するPAV単調非増加回帰."""
    blocks: list[tuple[float, int, list[int]]] = []
    for idx, val in non_null:
        blocks.append((float(val), 1, [idx]))
        while len(blocks) >= 2:
            if blocks[-1][0] / blocks[-1][1] <= blocks[-2][0] / blocks[-2][1]:
                break
            s1, c1, i1 = blocks.pop()
            s2, c2, i2 = blocks.pop()
            blocks.append((s1 + s2, c1 + c2, i2 + i1))
    fitted: list[tuple[int, int]] = []
    for total, count, indices in blocks:
        v = int(round(total / count))
        for i in indices:
            fitted.append((i, v))
    return fitted


def _isotonic_decreasing(values: list[int | None]) -> list[int | None]:
    """PAVアルゴリズムによる単調非増加回帰（L2最適）.

    None値はスキップし、非null値に対してのみ回帰を適用。
    隣接する非null値の差がDROP_THRESHOLDを超える降下点で区間を分割し、
    各区間で独立にPAVを適用することで急降下を追従する。
    """
    non_null = [(i, v) for i, v in enumerate(values) if v is not None]
    if not non_null:
        return list(values)

    segments: list[list[tuple[int, int]]] = [[non_null[0]]]
    for k in range(1, len(non_null)):
        prev_val = non_null[k - 1][1]
        cur_val = non_null[k][1]
        if prev_val - cur_val > DROP_THRESHOLD:
            segments.append([])
        segments[-1].append(non_null[k])

    result: list[int | None] = list(values)
    for seg in segments:
        for idx, fitted_val in _pav_segment(seg):
            result[idx] = fitted_val
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
    score: float
    score_kills: float
    score_count_gain: float
    score_dead: float
    enemy_score_gain: float = 0.0


def _compute_score(result: dict, cfg: ScoringConfig | None = None) -> ScoreBreakdown:
    """重み付き加算でスコアを計算し内訳を返す."""
    if cfg is None:
        cfg = load_scoring_config()
    enemy_score_gain = float(result.get("enemy_score_gain", 0.0))
    if result.get("is_dead", False):
        score_dead = cfg.death_penalty
        return ScoreBreakdown(
            score=score_dead,
            score_kills=0.0,
            score_count_gain=0.0,
            score_dead=score_dead,
            enemy_score_gain=enemy_score_gain,
        )
    kills = max(0, min(4, result.get("kills", 0)))
    gain = max(0.0, float(result.get("score_count_gain", 0)))
    score_kills = kills * cfg.weights.kills
    score_count_gain = 1 + gain * cfg.weights.score_count_gain
    score = score_kills * score_count_gain
    return ScoreBreakdown(
        score=score,
        score_kills=score_kills,
        score_count_gain=score_count_gain,
        score_dead=0.0,
        enemy_score_gain=enemy_score_gain,
    )


@dataclass
class _ScoredFrame:
    timestamp: float
    score: float
    breakdown: ScoreBreakdown
    raw: dict


@dataclass
class FrameAnalysis:
    timestamp_seconds: float
    score: float
    score_kills: float
    score_count_gain: float
    score_dead: float
    my_team_count: int | None
    enemy_team_count: int | None
    kills: int
    is_dead: bool
    my_team_count_raw: int | None
    enemy_team_count_raw: int | None
    enemy_score_gain: float = 0.0
    has_count_rail: bool | None = None


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
        duration_type: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> list[HighlightSegment]:
        video_path = Path(video_path)
        scan_start = start_seconds or 0.0

        if duration_type == "3min":
            return self._detect_nawabari(
                video_path,
                scan_start,
                end_seconds,
                progress_callback,
            )
        return self._detect_ranked(
            video_path,
            scan_start,
            end_seconds,
            progress_callback,
        )

    def _detect_nawabari(
        self,
        video_path: Path,
        scan_start: float,
        end_seconds: float | None,
        progress_callback: ProgressCallback | None,
    ) -> list[HighlightSegment]:
        """ナワバリ: 全区間で下半分のみ5秒間隔."""
        frames = extract_frames(
            video_path=video_path,
            interval_seconds=self.interval,
            no_save=True,
            start_seconds=scan_start if scan_start else None,
            end_seconds=end_seconds,
        )
        total = len(frames)
        done_count = [0]
        results: list[tuple[float, dict | str]] = []

        def _analyze(index: int) -> tuple[float, dict]:
            ts = scan_start + index * self.interval
            label = self._format_timestamp(ts)
            try:
                result = self.analyzer.analyze_frame_lower_only(frames[index], label)
            except Exception:
                logger.exception("Analysis failed for frame at %s", label)
                result = {"kills": 0}
            return ts, result

        with ThreadPoolExecutor(max_workers=self.analyzer.concurrency) as executor:
            futures = {executor.submit(_analyze, i): i for i in range(total)}
            for future in as_completed(futures):
                ts, result = future.result()
                results.append((ts, result))
                done_count[0] += 1
                if progress_callback:
                    progress_callback(1, done_count[0], total)

        self.scan_summary = {
            "phase_a_frames": total,
            "phase_b_frames": 0,
            "total_frames": total,
            "count_swapped": False,
        }
        return self._finalize(results)

    def _detect_ranked(
        self,
        video_path: Path,
        scan_start: float,
        end_seconds: float | None,
        progress_callback: ProgressCallback | None,
    ) -> list[HighlightSegment]:
        """ガチルール: Phase A (上半分15秒) -> Phase B (下半分5秒, gain>1区間)."""
        phase_a_interval = 15.0

        # --- Phase A: upper-only at 15s intervals ---
        frames_a = extract_frames(
            video_path=video_path,
            interval_seconds=phase_a_interval,
            no_save=True,
            start_seconds=scan_start if scan_start else None,
            end_seconds=end_seconds,
        )
        total_a = len(frames_a)
        done_count = [0]
        results_a: list[tuple[float, dict | str]] = [None] * total_a  # type: ignore[list-item]

        def _analyze_upper(index: int) -> None:
            ts = scan_start + index * phase_a_interval
            label = self._format_timestamp(ts)
            try:
                result = self.analyzer.analyze_frame_upper_only(frames_a[index], label)
            except Exception:
                logger.exception("Phase A failed for frame at %s", label)
                result = {"my_team_count": None, "enemy_team_count": None, "kills": 0}
            results_a[index] = (ts, result)
            done_count[0] += 1
            if progress_callback:
                progress_callback(1, done_count[0], total_a)

        with ThreadPoolExecutor(max_workers=self.analyzer.concurrency) as executor:
            futures = [executor.submit(_analyze_upper, i) for i in range(total_a)]
            for future in as_completed(futures):
                future.result()

        # Score Phase A to get score_count_gain values
        scored_a = self._score_frames(list(results_a))

        # --- Count rail detection -> swap my/enemy counts ---
        rail_count = sum(
            1 for _, r in results_a
            if isinstance(r, dict) and r.get("has_count_rail", False)
        )
        count_swapped = rail_count > total_a / 2
        if count_swapped:
            for _, r in results_a:
                if isinstance(r, dict):
                    my = r.get("my_team_count")
                    enemy = r.get("enemy_team_count")
                    r["my_team_count"] = enemy
                    r["enemy_team_count"] = my
                    my_raw = r.get("my_team_count_raw")
                    enemy_raw = r.get("enemy_team_count_raw")
                    r["my_team_count_raw"] = enemy_raw
                    r["enemy_team_count_raw"] = my_raw
            # Re-score after swap
            scored_a = self._score_frames(list(results_a))

        # --- Determine Phase B regions (score_count_gain > 1 or enemy_score_gain > 1) ---
        interesting_timestamps: set[float] = set()
        for sf in scored_a:
            if sf.raw.get("score_count_gain", 0) > 1 or sf.raw.get("enemy_score_gain", 0) > 1:
                interesting_timestamps.add(sf.timestamp)

        # Phase A timestamps on 15s grid (also on 5s grid)
        phase_a_ts_set: set[float] = {sf.timestamp for sf in scored_a}

        # Build 5s grid timestamps and determine which need Phase B analysis
        frames_b = extract_frames(
            video_path=video_path,
            interval_seconds=self.interval,
            no_save=True,
            start_seconds=scan_start if scan_start else None,
            end_seconds=end_seconds,
        )
        total_b_grid = len(frames_b)

        # Map 5s grid indices to timestamps
        b_timestamps = [scan_start + i * self.interval for i in range(total_b_grid)]

        # Determine which 5s frames need lower-half analysis
        # A frame needs Phase B if any interesting Phase A timestamp is within +-15s
        phase_b_indices: list[int] = []
        for i, ts in enumerate(b_timestamps):
            if ts in phase_a_ts_set:
                # Phase A frame: reuse upper data, still need lower if interesting
                pass
            for interesting_ts in interesting_timestamps:
                if abs(ts - interesting_ts) <= phase_a_interval:
                    phase_b_indices.append(i)
                    break

        # Remove duplicates and sort
        phase_b_indices = sorted(set(phase_b_indices))
        phase_b_total = len(phase_b_indices)

        logger.info(
            "Phase A done: %d frames. Phase B: %d frames needed.",
            total_a,
            phase_b_total,
        )

        # --- Phase B: lower-only at 5s intervals for interesting regions ---
        done_count[0] = 0
        phase_b_results: dict[float, dict] = {}

        def _analyze_lower(index: int) -> None:
            ts = b_timestamps[index]
            label = self._format_timestamp(ts)
            try:
                result = self.analyzer.analyze_frame_lower_only(frames_b[index], label)
            except Exception:
                logger.exception("Phase B failed for frame at %s", label)
                result = {"kills": 0, "is_dead": False}
            phase_b_results[ts] = result
            done_count[0] += 1
            if progress_callback:
                progress_callback(2, done_count[0], phase_b_total)

        if phase_b_indices:
            with ThreadPoolExecutor(max_workers=self.analyzer.concurrency) as executor:
                futures = [executor.submit(_analyze_lower, i) for i in phase_b_indices]
                for future in as_completed(futures):
                    future.result()

        # --- Merge Phase A + Phase B ---
        # Build Phase A lookup
        phase_a_data: dict[float, dict] = {}
        for sf in scored_a:
            phase_a_data[sf.timestamp] = sf.raw

        # Build merged results on the 5s grid
        merged_results: list[tuple[float, dict | str]] = []
        for i, ts in enumerate(b_timestamps):
            merged: dict = {}
            # Upper data: from Phase A if on 15s grid, otherwise empty
            if ts in phase_a_data:
                upper = phase_a_data[ts]
                merged.update(upper)
            else:
                merged.update({"my_team_count": None, "enemy_team_count": None})

            # Lower data: from Phase B if analyzed, otherwise defaults
            if ts in phase_b_results:
                lower = phase_b_results[ts]
                if isinstance(lower, dict):
                    merged.update(lower)
                else:
                    merged.update({"kills": 0, "is_dead": False})
            else:
                merged.update({"kills": 0, "is_dead": False})

            merged_results.append((ts, merged))

        self.scan_summary = {
            "phase_a_frames": total_a,
            "phase_b_frames": phase_b_total,
            "total_frames": len(merged_results),
            "count_swapped": count_swapped,
        }
        return self._finalize(merged_results)

    def _finalize(self, results: list[tuple[float, dict | str]]) -> list[HighlightSegment]:
        """Score, build all_frames, select windows, apply budget."""
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
                enemy_score_gain=f.breakdown.enemy_score_gain,
                has_count_rail=f.raw.get("has_count_rail"),
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

        my_counts: list[int | None] = []
        enemy_counts: list[int | None] = []
        for _, result in sorted_results:
            if isinstance(result, dict):
                my_counts.append(result.get("my_team_count"))
                enemy_counts.append(result.get("enemy_team_count"))
            else:
                my_counts.append(None)
                enemy_counts.append(None)

        scored: list[_ScoredFrame] = []
        for i, (timestamp, result) in enumerate(sorted_results):
            breakdown = ScoreBreakdown(score=0.0, score_kills=0.0, score_count_gain=0.0, score_dead=0.0)
            raw: dict = {}
            if isinstance(result, dict):
                raw = result
                # my_team score_count_gain
                cur_count = my_counts[i]
                future_slice = [c for c in my_counts[i + 1 : i + 1 + window_size] if c is not None]
                future_avg = sum(future_slice) / len(future_slice) if future_slice else None
                raw["score_count_gain"] = _calc_score_count_gain(cur_count, future_avg)
                # enemy_team score_gain
                enemy_cur = enemy_counts[i]
                enemy_future = [
                    c for c in enemy_counts[i + 1 : i + 1 + window_size] if c is not None
                ]
                enemy_avg = sum(enemy_future) / len(enemy_future) if enemy_future else None
                raw["enemy_score_gain"] = _calc_score_count_gain(enemy_cur, enemy_avg)
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
                    peak_intensity=int(best_total),
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
