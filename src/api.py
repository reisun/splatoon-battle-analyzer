"""FastAPI application for Splatoon battle highlight detection."""

import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.battle_analyzer import BattleAnalyzer, check_api_key_available
from src.highlight_detector import FrameAnalysis, HighlightDetector
from src.job_store import JobStatus, JobStore
from src.match_scanner import MatchInfo, MatchScanner, ScanResult
from src.scoring_config import load_scoring_config

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Splatoon Battle Analyzer",
    description="Highlight detection API for Splatoon gameplay videos",
    version="0.2.0",
)

job_store = JobStore()


class HighlightRequest(BaseModel):
    file_path: str = Field(description="Absolute path to the video file on server")
    start: float | None = Field(default=None, description="Start time in seconds")
    end: float | None = Field(default=None, description="End time in seconds")
    interval: float = Field(default=5.0, description="Frame scan interval (seconds)")
    model: str | None = Field(default=None, description="Claude model name")
    concurrency: int = Field(default=4, description="Concurrent API calls")
    duration_type: str | None = Field(default=None, description="Match rule type (5min/3min)")


class SegmentResult(BaseModel):
    start_seconds: float
    end_seconds: float
    peak_intensity: int


class FrameResult(BaseModel):
    model_config = {"coerce_numbers_to_str": False}

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

    @field_validator("my_team_count_raw", "enemy_team_count_raw", mode="before")
    @classmethod
    def _round_count_raw(cls, v: object) -> int | None:
        if v is None:
            return None
        return round(float(v))


class ScoringInfo(BaseModel):
    description: str = "カウント上昇に絡むキルがより選定されやすいスコア計算としています"
    score: str = "score_kills * score_count_gain + score_dead"
    score_kills: str = "kills * kills_weight"
    score_count_gain: str = "1 + count_gain * count_gain_weight"
    score_dead: str = "is_dead ? death_penalty : 0"
    weights: dict = Field(default_factory=dict)
    death_penalty: float = 0
    count_gain_window_seconds: int = 30


class HighlightResponse(BaseModel):
    video: str
    model: str
    highlights: list[SegmentResult]
    scoring: ScoringInfo = Field(default_factory=ScoringInfo)
    frames: list[FrameResult] = Field(default_factory=list)
    scan_summary: dict


class JobCreateResponse(BaseModel):
    job_id: str


class JobProgressResponse(BaseModel):
    phase: int
    phase_total: int
    frames_done: int
    frames_total: int


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: JobProgressResponse | None = None
    result: HighlightResponse | None = None
    error: str | None = None
    started_at: float | None = None


class HealthResponse(BaseModel):
    status: str = "ok"


def _build_scoring_info() -> ScoringInfo:
    cfg = load_scoring_config()
    return ScoringInfo(
        weights={"kills": cfg.weights.kills, "score_count_gain": cfg.weights.score_count_gain},
        death_penalty=cfg.death_penalty,
        count_gain_window_seconds=cfg.score_count_gain_window_seconds,
    )


def _to_frame_results(frames: list[FrameAnalysis]) -> list[FrameResult]:
    return [
        FrameResult(
            timestamp_seconds=f.timestamp_seconds,
            score=f.score,
            score_kills=f.score_kills,
            score_count_gain=f.score_count_gain,
            score_dead=f.score_dead,
            my_team_count=f.my_team_count,
            enemy_team_count=f.enemy_team_count,
            kills=f.kills,
            is_dead=f.is_dead,
            my_team_count_raw=f.my_team_count_raw,
            enemy_team_count_raw=f.enemy_team_count_raw,
        )
        for f in frames
    ]


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse()


@app.post("/analyze/highlights", response_model=HighlightResponse)
async def analyze_highlights(request: HighlightRequest) -> HighlightResponse:
    """Detect highlights in a Splatoon gameplay video."""
    video_path = Path(request.file_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {request.file_path}")

    if not check_api_key_available():
        raise HTTPException(status_code=503, detail="Claude CLI is not available")

    analyzer = BattleAnalyzer(
        model=request.model,
        concurrency=request.concurrency,
    )
    detector = HighlightDetector(
        analyzer=analyzer,
        interval=request.interval,
    )

    try:
        highlights = detector.detect(
            video_path=video_path,
            start_seconds=request.start,
            end_seconds=request.end,
            duration_type=request.duration_type,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        raise HTTPException(status_code=500, detail=str(e))

    return HighlightResponse(
        video=video_path.name,
        model=analyzer.model,
        highlights=[
            SegmentResult(
                start_seconds=h.start_seconds,
                end_seconds=h.end_seconds,
                peak_intensity=h.peak_intensity,
            )
            for h in highlights
        ],
        scoring=_build_scoring_info(),
        frames=_to_frame_results(detector.all_frames),
        scan_summary=detector.scan_summary,
    )


@app.post("/analyze/highlights/jobs", response_model=JobCreateResponse)
async def create_highlight_job(request: HighlightRequest) -> JobCreateResponse:
    """Create an async highlight detection job."""
    video_path = Path(request.file_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {request.file_path}")

    if not check_api_key_available():
        raise HTTPException(status_code=503, detail="Claude CLI is not available")

    job = job_store.create()
    asyncio.get_event_loop().run_in_executor(None, _run_job, job.job_id, request)
    return JobCreateResponse(job_id=job.job_id)


def _run_job(job_id: str, request: HighlightRequest) -> None:
    job_store.mark_running(job_id)
    try:
        analyzer = BattleAnalyzer(model=request.model, concurrency=request.concurrency)
        detector = HighlightDetector(
            analyzer=analyzer,
            interval=request.interval,
        )

        def on_progress(phase: int, frames_done: int, frames_total: int) -> None:
            job_store.update_progress(job_id, phase, frames_done, frames_total, phase_total=2)

        highlights = detector.detect(
            video_path=Path(request.file_path),
            start_seconds=request.start,
            end_seconds=request.end,
            duration_type=request.duration_type,
            progress_callback=on_progress,
        )

        result = HighlightResponse(
            video=Path(request.file_path).name,
            model=analyzer.model,
            highlights=[
                SegmentResult(
                    start_seconds=h.start_seconds,
                    end_seconds=h.end_seconds,
                    peak_intensity=h.peak_intensity,
                )
                for h in highlights
            ],
            scoring=_build_scoring_info(),
            frames=_to_frame_results(detector.all_frames),
            scan_summary=detector.scan_summary,
        )
        job_store.mark_completed(job_id, result.model_dump())
    except Exception as e:
        job_store.mark_failed(job_id, str(e))


@app.get("/analyze/highlights/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Get the status of a highlight detection job."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    progress = None
    result = None
    if job.status in (JobStatus.RUNNING, JobStatus.COMPLETED):
        progress = JobProgressResponse(
            phase=job.progress.phase,
            phase_total=job.progress.phase_total,
            frames_done=job.progress.frames_done,
            frames_total=job.progress.frames_total,
        )
    if job.status == JobStatus.COMPLETED and job.result:
        result = HighlightResponse(**job.result)

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        progress=progress,
        result=result,
        error=job.error,
        started_at=job.started_at,
    )


# --- Match Boundary Scan ---


class MatchScanRequest(BaseModel):
    file_path: str = Field(description="Absolute path to the video file on server")
    interval: float = Field(default=30.0, description="Frame scan interval (seconds)")
    model: str | None = Field(default=None, description="Claude model name")
    concurrency: int = Field(default=4, description="Concurrent API calls")


class MatchResult(BaseModel):
    start_seconds: float
    duration_seconds: int
    duration_type: str


class TimerReadingResult(BaseModel):
    frame_timestamp: float
    timer_seconds: float
    total_duration: int
    duration_type: str
    match_start: float


class MatchScanResponse(BaseModel):
    matches: list[MatchResult]
    readings: list[TimerReadingResult] = Field(default_factory=list)


class MatchScanJobCreateResponse(BaseModel):
    job_id: str


class MatchScanJobProgressResponse(BaseModel):
    frames_done: int
    frames_total: int


class MatchScanJobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: MatchScanJobProgressResponse | None = None
    result: MatchScanResponse | None = None
    error: str | None = None
    started_at: float | None = None


@app.post("/analyze/matches/scan/jobs", response_model=MatchScanJobCreateResponse)
async def create_match_scan_job(request: MatchScanRequest) -> MatchScanJobCreateResponse:
    """Create an async match boundary scan job."""
    video_path = Path(request.file_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {request.file_path}")

    if not check_api_key_available():
        raise HTTPException(status_code=503, detail="Claude CLI is not available")

    job = job_store.create()
    asyncio.get_event_loop().run_in_executor(None, _run_scan_job, job.job_id, request)
    return MatchScanJobCreateResponse(job_id=job.job_id)


def _run_scan_job(job_id: str, request: MatchScanRequest) -> None:
    job_store.mark_running(job_id)
    try:
        analyzer = BattleAnalyzer(model=request.model, concurrency=request.concurrency)
        scanner = MatchScanner(analyzer=analyzer, interval=request.interval)

        def on_progress(frames_done: int, frames_total: int) -> None:
            job_store.update_progress(job_id, 1, frames_done, frames_total)

        scan_result = scanner.scan(
            video_path=Path(request.file_path),
            progress_callback=on_progress,
        )

        result = MatchScanResponse(
            matches=[
                MatchResult(
                    start_seconds=m.start_seconds,
                    duration_seconds=m.duration_seconds,
                    duration_type=m.duration_type,
                )
                for m in scan_result.matches
            ],
            readings=[
                TimerReadingResult(
                    frame_timestamp=r.frame_timestamp,
                    timer_seconds=r.timer_seconds,
                    total_duration=r.total_duration,
                    duration_type=r.duration_type,
                    match_start=r.match_start,
                )
                for r in scan_result.readings
            ],
        )
        job_store.mark_completed(job_id, result.model_dump())
    except Exception as e:
        job_store.mark_failed(job_id, str(e))


@app.get("/analyze/matches/scan/jobs/{job_id}", response_model=MatchScanJobStatusResponse)
async def get_match_scan_job_status(job_id: str) -> MatchScanJobStatusResponse:
    """Get the status of a match boundary scan job."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    progress = None
    result = None
    if job.status in (JobStatus.RUNNING, JobStatus.COMPLETED):
        progress = MatchScanJobProgressResponse(
            frames_done=job.progress.frames_done,
            frames_total=job.progress.frames_total,
        )
    if job.status == JobStatus.COMPLETED and job.result:
        result = MatchScanResponse(**job.result)

    return MatchScanJobStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        progress=progress,
        result=result,
        error=job.error,
        started_at=job.started_at,
    )
