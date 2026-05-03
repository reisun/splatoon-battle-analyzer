"""CLI entry point for Splatoon Battle Analyzer.

Provides command-line interface to extract frames from gameplay videos
and optionally analyze them using Claude Vision API.
"""

import argparse
import logging
import sys
from pathlib import Path

from src.battle_analyzer import BattleAnalyzer, check_api_key_available
from src.frame_extractor import extract_frames

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
        description="Analyze Splatoon gameplay videos using frame extraction and Claude Vision API.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input video file (mp4/mkv)",
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

    video_path = Path(args.input)
    output_dir = Path(args.output_dir)

    # Step 1: Extract frames
    logger.info("Extracting frames from: %s (interval: %.1fs)", video_path, args.interval)
    try:
        frame_paths = extract_frames(
            video_path=video_path,
            interval_seconds=args.interval,
            output_dir=output_dir,
        )
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
