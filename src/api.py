"""FastAPI application for Splatoon battle highlight detection."""

import asyncio
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.battle_analyzer import BattleAnalyzer, check_api_key_available
from src.highlight_detector import HighlightDetector
from src.job_store import JobStatus, JobStore

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Splatoon Battle Analyzer",
    description="Highlight detection API for Splatoon gameplay videos",
    version="0.1.0",
)

job_store = JobStore()


class HighlightRequest(BaseModel):
    file_path: str = Field(description="Absolute path to the video file on server")
    start: float | None = Field(default=None, description="Start time in seconds")
    end: float | None = Field(default=None, description="End time in seconds")
    stage1_interval: float = Field(default=30.0, description="Stage 1 scan interval (seconds)")
    stage2_interval: float = Field(default=5.0, description="Stage 2 scan interval (seconds)")
    threshold: int = Field(default=5, description="Intensity threshold (1-10)")
    max_highlights: int = Field(default=3, description="Max highlight regions")
    model: str | None = Field(default=None, description="Claude model name")
    concurrency: int = Field(default=4, description="Concurrent API calls")


class SegmentResult(BaseModel):
    start_seconds: float
    end_seconds: float
    peak_intensity: int
    description: str


class HighlightResponse(BaseModel):
    video: str
    model: str
    highlights: list[SegmentResult]
    stage1_summary: dict


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
        stage1_interval=request.stage1_interval,
        stage2_interval=request.stage2_interval,
        threshold=request.threshold,
        max_highlights=request.max_highlights,
    )

    try:
        highlights = detector.detect(
            video_path=video_path,
            start_seconds=request.start,
            end_seconds=request.end,
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
                description=h.description,
            )
            for h in highlights
        ],
        stage1_summary=detector.stage1_summary,
    )


@app.post("/analyze/highlights/jobs", response_model=JobCreateResponse)
async def create_highlight_job(request: HighlightRequest) -> JobCreateResponse:
    """Create an async highlight detection job."""
    video_path = Path(request.file_path)
    if not video_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Video file not found: {request.file_path}"
        )

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
            stage1_interval=request.stage1_interval,
            stage2_interval=request.stage2_interval,
            threshold=request.threshold,
            max_highlights=request.max_highlights,
        )

        def on_progress(phase: int, frames_done: int, frames_total: int) -> None:
            job_store.update_progress(job_id, phase, frames_done, frames_total)

        highlights = detector.detect(
            video_path=Path(request.file_path),
            start_seconds=request.start,
            end_seconds=request.end,
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
                    description=h.description,
                )
                for h in highlights
            ],
            stage1_summary=detector.stage1_summary,
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
