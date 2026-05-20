"""CV-based kill/death detection and OCR-based timer reading.

Replaces LLM Vision calls for Phase B (kills/deaths) and timer scanning
with OpenCV template matching and Tesseract OCR respectively.
"""

import logging
import re
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_SAMPLES_DIR = Path(__file__).parent.parent / "samples"

# --- Template loading (once at module level) ---


def _load_template(name: str) -> np.ndarray | None:
    """Load a template image as grayscale, returning None if not found."""
    path = _SAMPLES_DIR / name
    if not path.exists():
        logger.warning("Template not found: %s", path)
        return None
    tmpl = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if tmpl is None:
        logger.warning("Failed to read template: %s", path)
        return None
    return tmpl


_KILL_TEMPLATE = _load_template("をたおした！.png")
_DEATH_TEMPLATE_YELLOW = _load_template("復活_黄.png")
_DEATH_TEMPLATE_BLACK = _load_template("復活_黒.png")

# Threshold for template matching confidence
_MATCH_THRESHOLD = 0.55

# Scales to try for multi-scale template matching
_SCALES = [0.8, 1.0, 1.2]


def _multi_scale_match(
    gray_frame: np.ndarray,
    template: np.ndarray,
    threshold: float = _MATCH_THRESHOLD,
) -> list[tuple[int, int, float]]:
    """Run template matching at multiple scales, return all matches above threshold.

    Returns list of (x, y, confidence) tuples.
    """
    matches: list[tuple[int, int, float]] = []
    fh, fw = gray_frame.shape[:2]

    for scale in _SCALES:
        th, tw = template.shape[:2]
        new_h = int(th * scale)
        new_w = int(tw * scale)
        if new_h <= 0 or new_w <= 0 or new_h > fh or new_w > fw:
            continue
        scaled = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(gray_frame, scaled, cv2.TM_CCOEFF_NORMED)
        locs = np.where(result >= threshold)
        for pt_y, pt_x in zip(locs[0], locs[1]):
            conf = float(result[pt_y, pt_x])
            matches.append((int(pt_x), int(pt_y), conf))

    return matches


def detect_kills(frame: np.ndarray) -> int:
    """Detect kill count from the lower 30% crop of a half-resized frame.

    Uses template matching with the 'をたおした！' template.
    Multiple kills appear as vertically stacked text.

    Args:
        frame: Lower 30% crop of a half-resized game frame (BGR).

    Returns:
        Number of unique kill detections (0-4).
    """
    if _KILL_TEMPLATE is None:
        return 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    matches = _multi_scale_match(gray, _KILL_TEMPLATE)

    if not matches:
        return 0

    # Non-maximum suppression: filter overlapping detections vertically
    # Sort by confidence descending
    matches.sort(key=lambda m: m[2], reverse=True)
    tmpl_h = _KILL_TEMPLATE.shape[0]

    kept: list[tuple[int, int, float]] = []
    for mx, my, mc in matches:
        is_dup = False
        for kx, ky, _kc in kept:
            if abs(my - ky) < tmpl_h:
                is_dup = True
                break
        if not is_dup:
            kept.append((mx, my, mc))

    return min(4, len(kept))


def detect_death(frame: np.ndarray) -> bool:
    """Detect whether the player is dead from the lower 30% crop.

    Currently disabled due to high false-positive rate with template matching.
    Returns False unconditionally. Kill detection alone provides sufficient
    highlight scoring signal.
    """
    return False


def read_timer(frame: np.ndarray) -> str | None:
    """Read the timer text from a timer crop using Tesseract OCR.

    Args:
        frame: Timer crop (top 15% x center 40% of half-resized frame, BGR).

    Returns:
        Timer string in M:SS format (e.g. '4:52'), or None if not found.
    """
    try:
        import pytesseract
    except ImportError:
        logger.warning("pytesseract not installed, cannot read timer")
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

    # Threshold to isolate white text on dark background
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Slight dilation to connect broken character segments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    config = "--psm 7 -c tessedit_char_whitelist=0123456789:"
    text = pytesseract.image_to_string(dilated, config=config).strip()

    # Look for M:SS pattern
    match = re.search(r"(\d+:\d{1,2})", text)
    if match:
        timer_str = match.group(1)
        # Validate: seconds part must be < 60
        parts = timer_str.split(":")
        if len(parts) == 2 and int(parts[1]) < 60:
            return timer_str

    return None
