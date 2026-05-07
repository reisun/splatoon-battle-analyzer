"""Single-pass highlight detection for Splatoon gameplay videos."""

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from src.battle_analyzer import FRAME_ANALYSIS_PROMPT, BattleAnalyzer
from src.frame_extractor import extract_frames
from src.scoring_config import ScoringConfig, load_scoring_config

ProgressCallback = Callable[[int, int, int], None]  # (phase, frames_done, frames_total)

logger = logging.getLogger(__name__)

MAX_CLIP_SECONDS = 15
MAX_TOTAL_SECONDS = 60


def _calc_score_gain(cur_count: int | None, future_avg: int | None) -> int:
    """現時点と未来平均のゲームカウント差分から score_gain を計算."""
    if cur_count is None or future_avg is None:
        return 1
    gain = (cur_count - future_avg) / 10 + 1
    return max(1, min(10, int(gain)))


FIRST_VALUE_FLOOR = 50
LOOK_RANGE = 10
DROP_THRESHOLD = 50
BIN_WIDTH = 10


def _frequency_filter(
    values: list[int | None],
    look_range: int = LOOK_RANGE,
    bin_width: int = BIN_WIDTH,
    drop_threshold: int = DROP_THRESHOLD,
) -> list[int | None]:
    """全データの出現頻度から支配的レベルを決定し、一時的な誤読値を除去する.

    1. 全非null値をビンに分類し、最頻ビンの中心を「支配的レベル」とする
    2. 支配的レベルから大きく乖離した各値について、前後look_range個の非null値を確認
    3. 前後の値が支配的レベル付近に戻っている場合、その値は一時的な誤読としてNoneに置換
    """
    non_null = [(i, v) for i, v in enumerate(values) if v is not None]
    n_vals = len(non_null)
    if n_vals < 5:
        return list(values)

    bins: dict[int, int] = {}
    for _, v in non_null:
        b = v // bin_width
        bins[b] = bins.get(b, 0) + 1
    dominant_bin = max(bins, key=bins.get)
    dominant_level = dominant_bin * bin_width + bin_width // 2

    result: list[int | None] = list(values)

    for idx in range(n_vals):
        i, v = non_null[idx]

        if dominant_level - v <= drop_threshold:
            continue

        past = [pv for _, pv in non_null[max(0, idx - look_range) : idx]]
        future = [fv for _, fv in non_null[idx + 1 : idx + 1 + look_range]]

        past_at_dominant = sum(1 for pv in past if dominant_level - pv <= drop_threshold)
        future_at_dominant = sum(1 for fv in future if dominant_level - fv <= drop_threshold)

        is_outlier = False
        if len(past) >= 2 and len(future) >= 2:
            if past_at_dominant > len(past) * 0.4 and future_at_dominant > len(future) * 0.4:
                is_outlier = True
        elif len(past) < 2 and len(future) >= 8:
            if future_at_dominant > len(future) * 0.75:
                is_outlier = True
        elif len(future) < 2 and len(past) >= 8:
            if past_at_dominant > len(past) * 0.75:
                is_outlier = True

        if is_outlier:
            result[i] = None

    return result


def _normalize_counts(
    results: list[tuple[float, dict | str]],
) -> None:
    """ゲームカウントを正規化し、raw値と正規化値の両方をdictに格納する.

    3ステップ:
    1. 全フレームの raw 値を収集
    2. 出現頻度フィルタで一時的な誤読値を除去
    3. 初回値の補正 + 単調減少制約 + null 埋めを適用
    """
    for field in ("my_team_count", "enemy_team_count"):
        raw_field = f"{field}_raw"

        # Step 1: raw値を収集
        dict_indices: list[int] = []
        raw_values: list[int | None] = []
        for i, (_ts, result) in enumerate(results):
            if not isinstance(result, dict):
                continue
            dict_indices.append(i)
            raw_values.append(result.get(field))

        # Step 2: 出現頻度フィルタ
        filtered = _frequency_filter(raw_values)

        # Step 3: 初回値補正 + 単調減少 + null埋め
        prev_normalized: int | None = None
        for seq_idx, di in enumerate(dict_indices):
            result = results[di][1]
            result[raw_field] = raw_values[seq_idx]  # type: ignore[union-attr]
            val = filtered[seq_idx]

            if val is None:
                result[field] = prev_normalized  # type: ignore[union-attr]
                continue

            if prev_normalized is None:
                normalized = 100 if val < FIRST_VALUE_FLOOR else val
            elif val > prev_normalized:
                normalized = prev_normalized
            else:
                normalized = val

            result[field] = normalized  # type: ignore[union-attr]
            prev_normalized = normalized


def _compute_score(result: dict, cfg: ScoringConfig | None = None) -> int:
    """4項目の重み付き掛け算でスコアを計算。デス中はペナルティ適用."""
    if cfg is None:
        cfg = load_scoring_config()
    kills = max(1, min(10, result.get("kills", 1)))
    assists = max(1, min(10, result.get("assists", 1)))
    score_gain = max(1, min(10, result.get("score_gain", 1)))
    special = max(1, min(10, result.get("special", 1)))
    score = (
        kills
        * cfg.weights.kills
        * assists
        * cfg.weights.assists
        * score_gain
        * cfg.weights.score_gain
        * special
        * cfg.weights.special
    )
    if result.get("is_dead", False):
        score *= cfg.death_penalty
    return int(score)


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
    description: str | None
    my_team_color: str | None
    enemy_team_color: str | None
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
        cfg = load_scoring_config()
        sorted_results = sorted(results, key=lambda x: x[0])
        _normalize_counts(sorted_results)
        window_size = max(1, int(cfg.score_gain_window_seconds / self.interval))

        counts: list[int | None] = []
        for _, result in sorted_results:
            if isinstance(result, dict):
                counts.append(result.get("my_team_count"))
            else:
                counts.append(None)

        scored: list[_ScoredFrame] = []
        for i, (timestamp, result) in enumerate(sorted_results):
            score = 0
            description = ""
            raw: dict = {}
            if isinstance(result, dict):
                raw = result
                cur_count = counts[i]
                future_slice = [c for c in counts[i + 1 : i + 1 + window_size] if c is not None]
                future_avg = int(sum(future_slice) / len(future_slice)) if future_slice else None
                raw["score_gain"] = _calc_score_gain(cur_count, future_avg)
                score = _compute_score(raw, cfg)
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
