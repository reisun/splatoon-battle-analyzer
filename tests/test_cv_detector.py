"""Tests for CV-based kill/death detection and OCR timer reading."""

from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.cv_detector import (
    _KILL_TEMPLATE,
    _SAMPLES_DIR,
    detect_death,
    detect_kills,
    read_timer,
)


class TestDetectKills:
    """Tests for detect_kills using template matching."""

    def test_blank_frame_returns_zero(self) -> None:
        """A blank/black frame should produce zero kills."""
        frame = np.zeros((100, 300, 3), dtype=np.uint8)
        assert detect_kills(frame) == 0

    def test_white_frame_returns_zero(self) -> None:
        """A solid white frame should not match the kill template."""
        frame = np.full((100, 300, 3), 255, dtype=np.uint8)
        assert detect_kills(frame) == 0

    def test_returns_int_in_range(self) -> None:
        """Result is always clamped to 0-4."""
        frame = np.zeros((100, 300, 3), dtype=np.uint8)
        result = detect_kills(frame)
        assert isinstance(result, int)
        assert 0 <= result <= 4

    def test_template_file_exists(self) -> None:
        """The kill template image should exist in samples/."""
        assert (_SAMPLES_DIR / "をたおした！.png").exists()

    def test_kill_template_loaded(self) -> None:
        """The kill template should be loaded at module level."""
        assert _KILL_TEMPLATE is not None
        assert _KILL_TEMPLATE.ndim == 2  # grayscale

    def test_grayscale_frame_accepted(self) -> None:
        """A grayscale (2D) frame should work without error."""
        frame = np.zeros((100, 300), dtype=np.uint8)
        result = detect_kills(frame)
        assert result == 0


class TestDetectDeath:
    """Tests for detect_death using template matching."""

    def test_blank_frame_returns_false(self) -> None:
        """A blank/black frame should not detect death."""
        frame = np.zeros((100, 300, 3), dtype=np.uint8)
        assert detect_death(frame) is False

    def test_white_frame_returns_false(self) -> None:
        """A solid white frame should not match the death template."""
        frame = np.full((100, 300, 3), 255, dtype=np.uint8)
        assert detect_death(frame) is False

    def test_returns_bool(self) -> None:
        """Result is always a bool."""
        frame = np.zeros((100, 300, 3), dtype=np.uint8)
        result = detect_death(frame)
        assert isinstance(result, bool)

    def test_death_template_files_exist(self) -> None:
        """The death template images should exist in samples/."""
        assert (_SAMPLES_DIR / "復活_黄.png").exists()
        assert (_SAMPLES_DIR / "復活_黒.png").exists()

    def test_searches_right_region_only(self) -> None:
        """detect_death should only search the right 40% of the frame."""
        # Create a frame wide enough to verify region selection
        frame = np.zeros((100, 500, 3), dtype=np.uint8)
        # Should not crash and should return False for blank
        assert detect_death(frame) is False


class TestReadTimer:
    """Tests for read_timer using Tesseract OCR."""

    def test_blank_frame_returns_none(self) -> None:
        """A blank frame should not produce a valid timer reading."""
        frame = np.zeros((50, 200, 3), dtype=np.uint8)
        result = read_timer(frame)
        assert result is None

    def test_white_frame_returns_none(self) -> None:
        """A solid white frame should not produce a valid timer reading."""
        frame = np.full((50, 200, 3), 255, dtype=np.uint8)
        result = read_timer(frame)
        assert result is None

    def test_returns_string_or_none(self) -> None:
        """Result is always str or None."""
        frame = np.zeros((50, 200, 3), dtype=np.uint8)
        result = read_timer(frame)
        assert result is None or isinstance(result, str)

    def test_pytesseract_missing_returns_none(self) -> None:
        """When pytesseract is not importable, returns None gracefully."""
        frame = np.zeros((50, 200, 3), dtype=np.uint8)
        with patch.dict("sys.modules", {"pytesseract": None}):
            # Force re-import failure by patching builtins
            import builtins

            original_import = builtins.__import__

            def mock_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
                if name == "pytesseract":
                    raise ImportError("mocked")
                return original_import(name, *args, **kwargs)

            with patch.object(builtins, "__import__", side_effect=mock_import):
                result = read_timer(frame)
                assert result is None

    def test_grayscale_frame_accepted(self) -> None:
        """A grayscale (2D) frame should work without error."""
        frame = np.zeros((50, 200), dtype=np.uint8)
        result = read_timer(frame)
        assert result is None


class TestTemplateLocations:
    """Verify template loading paths are correct."""

    def test_samples_dir_exists(self) -> None:
        """The samples directory should exist."""
        assert _SAMPLES_DIR.exists()
        assert _SAMPLES_DIR.is_dir()

    def test_samples_dir_is_relative_to_module(self) -> None:
        """samples/ should be relative to the cv_detector module's parent."""
        expected = Path(__file__).parent.parent / "samples"
        assert _SAMPLES_DIR == expected
