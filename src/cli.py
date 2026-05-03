"""CLI entry point for Splatoon Battle Analyzer.

Provides command-line interface to extract frames from gameplay videos
or RTMP streams and optionally analyze them using Claude Vision API.
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2

from src.battle_analyzer import BattleAnalyzer, check_api_key_available
from src.frame_source import FileFrameSource, FrameSource, StreamFrameSource

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: List of arguments (defaults to sys.argv[1:]).

    Returns:
        Parsed arguments namespace.

    Raises:
        SystemExit: If argument validation fails.
    """
    parser = argparse.ArgumentParser(
        prog="splatoon-battle-analyzer",
        description="Analyze Splatoon gameplay videos using frame extraction and Claude Vision API.",
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--input",
        help="Path to the input video file (mp4/mkv)",
    )
    source_group.add_argument(
        "--stream",
        help="RTMP stream URL (e.g., rtmp://host.docker.internal:1935/live/stream)",
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
    return parser.parse_args(argv)


def create_frame_source(args: argparse.Namespace) -> FrameSource:
    """Create the appropriate FrameSource based on CLI arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A FrameSource instance (FileFrameSource or StreamFrameSource).
    """
    if args.stream:
        return StreamFrameSource(
            stream_url=args.stream,
            interval_seconds=args.interval,
        )
    return FileFrameSource(
        video_path=args.input,
        interval_seconds=args.interval,
    )


def save_frames(
    source: FrameSource,
    output_dir: Path,
) -> list[Path]:
    """Extract and save frames from a FrameSource.

    Args:
        source: The frame source to extract from.
        output_dir: Directory to save extracted frame images.

    Returns:
        List of paths to saved frame images.
    """
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


def format_timeline(results: list[dict[str, str]]) -> str:
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
        lines.append(entry["analysis"])

    lines.append("")
    lines.append("=" * 60)
    lines.append(f"Total frames analyzed: {len(results)}")
    lines.append("=" * 60)

    return "\n".join(lines)


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
    source_label = args.stream if args.stream else args.input

    # Step 1: Create frame source and extract frames
    logger.info("Extracting frames from: %s (interval: %.1fs)", source_label, args.interval)
    try:
        source = create_frame_source(args)
        frame_paths = save_frames(source, output_dir)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error("Frame extraction failed: %s", e)
        return 1

    if not frame_paths:
        logger.warning("No frames were extracted.")
        return 0

    logger.info("Extracted %d frames to %s", len(frame_paths), output_dir)

    # Step 2: Analyze frames (unless --frames-only)
    if args.frames_only:
        logger.info("Frames-only mode: skipping analysis.")
        print(f"\nExtracted {len(frame_paths)} frames to {output_dir}")
        for p in frame_paths:
            print(f"  {p}")
        return 0

    if not check_api_key_available():
        logger.warning(
            "ANTHROPIC_API_KEY is not set. Use --frames-only or set the API key in .env"
        )
        print(f"\nExtracted {len(frame_paths)} frames to {output_dir}")
        print("Set ANTHROPIC_API_KEY to enable battle analysis.")
        return 1

    # Step 3: Analyze and output timeline
    try:
        analyzer = BattleAnalyzer()
        results = analyzer.analyze_frames(frame_paths)
    except Exception:
        logger.exception("Analysis failed")
        return 1

    timeline = format_timeline(results)
    print(timeline)

    return 0


if __name__ == "__main__":
    sys.exit(run())
