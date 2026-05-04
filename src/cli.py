"""CLI entry point for Splatoon Battle Analyzer.

Provides command-line interface to extract frames from gameplay videos
or RTMP streams and optionally analyze them using Ollama Vision API (llava-llama3).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from src.battle_analyzer import BattleAnalyzer, check_api_key_available
from src.frame_extractor import extract_frames
from src.frame_source import FileFrameSource, FrameSource, StreamFrameSource
from src.highlight_detector import HighlightDetector

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: List of arguments (defaults to sys.argv[1:]).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="splatoon-battle-analyzer",
        description="Analyze Splatoon gameplay videos using frame extraction and Ollama Vision API.",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--input",
        default=None,
        help="Path to the input video file (mp4/mkv)",
    )
    source_group.add_argument(
        "--stream",
        default=None,
        help="RTMP stream URL (e.g., rtmp://host:1935/live/stream)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Frame extraction interval in seconds (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Directory for extracted frame images (default: ./output)",
    )
    parser.add_argument(
        "--frames-only",
        action="store_true",
        help="Extract frames only without API analysis",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to extract (default: no limit)",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="Start time in seconds (default: beginning of video)",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="End time in seconds (default: end of video)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Process frames in memory without saving to disk (requires API analysis)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of concurrent API calls (default: 4)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ollama model name (default: env OLLAMA_MODEL or llava-llama3)",
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Write output to file instead of stdout",
    )
    parser.add_argument(
        "--mode",
        choices=["timeline", "highlight"],
        default="timeline",
        help="Analysis mode (default: timeline)",
    )
    parser.add_argument(
        "--stage1-interval",
        type=float,
        default=30.0,
        help="Stage 1 frame interval in seconds for highlight mode (default: 30)",
    )
    parser.add_argument(
        "--stage2-interval",
        type=float,
        default=5.0,
        help="Stage 2 frame interval in seconds for highlight mode (default: 5)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=5,
        help="Intensity threshold for highlight candidates (default: 5)",
    )
    return parser.parse_args(argv)


def create_frame_source(args: argparse.Namespace) -> FrameSource:
    """Create the appropriate FrameSource based on CLI arguments."""
    if args.stream:
        return StreamFrameSource(stream_url=args.stream, interval_seconds=args.interval)
    return FileFrameSource(video_path=args.input, interval_seconds=args.interval)


def save_frames(source: FrameSource, output_dir: Path) -> list[Path]:
    """Save frames from a FrameSource to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for timestamp, frame in source.frames():
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        filename = f"frame_{minutes:02d}m{seconds:02d}s.jpg"
        output_path = output_dir / filename
        cv2.imwrite(str(output_path), frame)
        saved_paths.append(output_path)
        logger.info("Saved frame at %02d:%02d -> %s", minutes, seconds, filename)
    return saved_paths


def _timestamp_to_seconds(timestamp: str) -> int:
    """Convert timestamp like '02m30s' to seconds."""
    import re

    match = re.match(r"(\d+)m(\d+)s", timestamp)
    if match:
        return int(match.group(1)) * 60 + int(match.group(2))
    return 0


def format_timeline(results: list[dict]) -> str:
    """Format analysis results as a timeline string.

    Args:
        results: List of dicts with 'timestamp' and 'analysis' keys.

    Returns:
        Formatted timeline string.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("SPLATOON BATTLE TIMELINE")
    lines.append("=" * 60)

    for entry in results:
        lines.append("")
        lines.append(f"[{entry['timestamp']}]")
        lines.append("-" * 40)
        analysis = entry["analysis"]
        if isinstance(analysis, dict):
            lines.append(json.dumps(analysis, ensure_ascii=False, indent=2))
        else:
            lines.append(str(analysis))

    lines.append("")
    lines.append("=" * 60)
    lines.append(f"Total frames analyzed: {len(results)}")
    lines.append("=" * 60)

    return "\n".join(lines)


def format_json_output(results: list[dict], args: argparse.Namespace, model: str) -> str:
    """Format analysis results as JSON.

    Args:
        results: List of dicts with 'timestamp' and 'analysis' keys.
        args: Parsed CLI arguments.
        model: Model name used for analysis.

    Returns:
        JSON string.
    """
    timeline = []
    for entry in results:
        timeline.append(
            {
                "timestamp": entry["timestamp"],
                "seconds": _timestamp_to_seconds(entry["timestamp"]),
                "analysis": entry["analysis"],
            }
        )

    output = {
        "video": Path(args.input).name,
        "model": model,
        "interval_seconds": args.interval,
        "frames_analyzed": len(results),
        "timeline": timeline,
    }

    return json.dumps(output, ensure_ascii=False, indent=2)


def run(argv: list[str] | None = None) -> int:
    """Execute the main CLI pipeline.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success, 1 for error).
    """
    args = parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir)

    # Validate: --no-save requires API analysis
    if args.no_save and args.frames_only:
        logger.error("--no-save cannot be used with --frames-only")
        return 1

    # Stream mode: use FrameSource pipeline
    if args.stream:
        source = create_frame_source(args)
        frame_paths = save_frames(source, output_dir)
        if not frame_paths:
            logger.warning("No frames were captured from stream.")
            return 0
        if args.frames_only:
            print(f"\nExtracted {len(frame_paths)} frames to {output_dir}")
            for p in frame_paths:
                print(f"  {p}")
            return 0
        # TODO: stream + analysis mode
        return 0

    video_path = Path(args.input)

    # Highlight mode: use 2-stage pipeline
    if args.mode == "highlight":
        return _run_highlight_mode(args)

    # Step 1: Extract frames
    logger.info("Extracting frames from: %s (interval: %.1fs)", video_path, args.interval)
    try:
        frame_paths = extract_frames(
            video_path=video_path,
            interval_seconds=args.interval,
            output_dir=output_dir,
            max_frames=args.max_frames,
            start_seconds=args.start,
            end_seconds=args.end,
            no_save=args.no_save,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error("Frame extraction failed: %s", e)
        return 1

    if not frame_paths:
        logger.warning("No frames were extracted.")
        return 0

    logger.info("Extracted %d frames", len(frame_paths))

    # Step 2: Analyze frames (unless --frames-only)
    if args.frames_only:
        logger.info("Frames-only mode: skipping analysis.")
        print(f"\nExtracted {len(frame_paths)} frames to {output_dir}")
        for p in frame_paths:
            print(f"  {p}")
        return 0

    if not check_api_key_available():
        logger.warning(
            "Ollama is not reachable. Use --frames-only or check OLLAMA_BASE_URL in .env"
        )
        if not args.no_save:
            print(f"\nExtracted {len(frame_paths)} frames to {output_dir}")
        print("Ensure Ollama is running and OLLAMA_BASE_URL is correctly configured.")
        return 1

    # Step 3: Analyze and output timeline
    try:
        analyzer = BattleAnalyzer(concurrency=args.concurrency, model=args.model)
        if args.no_save:
            results = _analyze_memory_frames(analyzer, frame_paths, args)
        else:
            results = analyzer.analyze_frames(frame_paths)
    except Exception:
        logger.exception("Analysis failed")
        return 1

    # Step 4: Output results
    if args.output_format == "json":
        output_text = format_json_output(results, args, analyzer.model)
    else:
        output_text = format_timeline(results)

    if args.output_file:
        Path(args.output_file).write_text(output_text, encoding="utf-8")
        logger.info("Output written to: %s", args.output_file)
    else:
        print(output_text)

    return 0


def _run_highlight_mode(args: argparse.Namespace) -> int:
    """Run the 2-stage highlight detection pipeline."""
    if not check_api_key_available():
        logger.error("Ollama is not reachable. Check OLLAMA_BASE_URL.")
        return 1

    analyzer = BattleAnalyzer(concurrency=args.concurrency, model=args.model)
    detector = HighlightDetector(
        analyzer=analyzer,
        stage1_interval=args.stage1_interval,
        stage2_interval=args.stage2_interval,
        threshold=args.threshold,
    )

    video_path = Path(args.input)
    try:
        highlights = detector.detect(
            video_path=video_path,
            start_seconds=args.start,
            end_seconds=args.end,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error("Highlight detection failed: %s", e)
        return 1

    if args.output_format == "json":
        output_text = format_highlight_json(highlights, detector.stage1_summary, args, analyzer.model)
    else:
        output_text = format_highlight_text(highlights, detector.stage1_summary)

    if args.output_file:
        Path(args.output_file).write_text(output_text, encoding="utf-8")
        logger.info("Output written to: %s", args.output_file)
    else:
        print(output_text)

    return 0


def format_highlight_json(
    highlights: list,
    stage1_summary: dict,
    args: argparse.Namespace,
    model: str,
) -> str:
    """Format highlight results as JSON."""
    output = {
        "video": Path(args.input).name,
        "model": model,
        "mode": "highlight",
        "highlights": [
            {
                "start_seconds": h.start_seconds,
                "end_seconds": h.end_seconds,
                "peak_intensity": h.peak_intensity,
                "description": h.description,
            }
            for h in highlights
        ],
        "stage1_summary": stage1_summary,
    }
    return json.dumps(output, ensure_ascii=False, indent=2)


def format_highlight_text(highlights: list, stage1_summary: dict) -> str:
    """Format highlight results as plain text."""
    lines = []
    lines.append("=" * 60)
    lines.append("SPLATOON HIGHLIGHT DETECTION")
    lines.append("=" * 60)
    lines.append("")
    lines.append(
        f"Stage 1: {stage1_summary.get('total_frames', 0)} frames scanned, "
        f"{stage1_summary.get('battle_frames', 0)} battle, "
        f"{stage1_summary.get('candidate_frames', 0)} candidates"
    )
    lines.append("")

    if not highlights:
        lines.append("No highlights detected.")
    else:
        for i, h in enumerate(highlights, 1):
            lines.append(f"Highlight #{i}:")
            lines.append(f"  Time: {h.start_seconds:.0f}s - {h.end_seconds:.0f}s")
            lines.append(f"  Peak intensity: {h.peak_intensity}")
            lines.append(f"  Description: {h.description}")
            lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def _analyze_memory_frames(
    analyzer: BattleAnalyzer,
    frames: list,
    args: argparse.Namespace,
) -> list[dict]:
    """Analyze frames held in memory using concurrent API calls.

    Args:
        analyzer: BattleAnalyzer instance.
        frames: List of numpy arrays (BGR frames).
        args: Parsed CLI arguments for computing timestamps.

    Returns:
        List of analysis result dicts.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    interval = args.interval
    start = args.start or 0.0

    results: list[dict] = [{}] * len(frames)

    def _analyze_one(index: int, frame) -> tuple[int, dict]:
        timestamp_sec = start + index * interval
        minutes = int(timestamp_sec // 60)
        seconds = int(timestamp_sec % 60)
        timestamp = f"{minutes:02d}m{seconds:02d}s"
        try:
            analysis = analyzer.analyze_frame_from_memory(frame, timestamp)
            return index, {"timestamp": timestamp, "analysis": analysis}
        except Exception:
            logger.exception("Failed to analyze frame at %s", timestamp)
            return index, {
                "timestamp": timestamp,
                "analysis": f"[Error] Failed to analyze frame at {timestamp}",
            }

    with ThreadPoolExecutor(max_workers=analyzer.concurrency) as executor:
        futures = [executor.submit(_analyze_one, i, f) for i, f in enumerate(frames)]
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    return results


if __name__ == "__main__":
    sys.exit(run())
