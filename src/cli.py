"""CLI entry point for Splatoon Battle Analyzer.

Provides command-line interface to extract frames from gameplay videos
and optionally analyze them using Ollama Vision API (llava-llama3).
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
        description="Analyze Splatoon gameplay videos using frame extraction and Ollama Vision API.",
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

    # Validate: --no-save requires API analysis
    if args.no_save and args.frames_only:
        logger.error("--no-save cannot be used with --frames-only")
        return 1

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
        analyzer = BattleAnalyzer(concurrency=args.concurrency)
        if args.no_save:
            # Frames are numpy arrays in memory
            results = _analyze_memory_frames(analyzer, frame_paths, args)
        else:
            results = analyzer.analyze_frames(frame_paths)
    except Exception:
        logger.exception("Analysis failed")
        return 1

    timeline = format_timeline(results)
    print(timeline)

    return 0


def _analyze_memory_frames(
    analyzer: BattleAnalyzer,
    frames: list,
    args: argparse.Namespace,
) -> list[dict[str, str]]:
    """Analyze frames held in memory using concurrent API calls.

    Args:
        analyzer: BattleAnalyzer instance.
        frames: List of numpy arrays (BGR frames).
        args: Parsed CLI arguments for computing timestamps.

    Returns:
        List of analysis result dicts.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    fps_estimate = 30.0  # Approximate; actual fps used during extraction
    interval = args.interval
    start = args.start or 0.0

    results: list[dict[str, str]] = [{}] * len(frames)

    def _analyze_one(index: int, frame) -> tuple[int, dict[str, str]]:
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
