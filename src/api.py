"""FastAPI application for Splatoon battle highlight detection."""

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.battle_analyzer import BattleAnalyzer, check_api_key_available
from src.highlight_detector import HighlightDetector

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Splatoon Battle Analyzer",
    description="Highlight detection API for Splatoon gameplay videos",
    version="0.1.0",
)


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
