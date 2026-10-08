"""Test the real Clef adapter's request contract, validation and safe failures."""

import base64
import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.battle_analyzer import LOWER_HALF_SYSTEM_PROMPT, BattleAnalyzer
from src.clef_client import analyze_lower_frame, lower_configuration_error


def _response(kills: str = "2", probability: float = 0.8) -> dict:
    return {
        "success": True,
        "result": {
            "answers": {
                "kills": {"type": "choice", "choice": kills},
                "is_dead": {"type": "noul", "noul": probability},
            }
        },
    }


@pytest.mark.parametrize("probability,expected", [(0.49, False), (0.5, True), (1, True)])
def test_clef_request_and_decisions(probability: float, expected: bool) -> None:
    """Send both questions with the original prompt, and threshold Noul at 0.5."""
    state = LOWER_HALF_SYSTEM_PROMPT.split("■ 出力フォーマット")[0]
    with patch("src.clef_client.urllib.request.urlopen") as urlopen:
        urlopen.return_value = io.BytesIO(json.dumps(_response("2", probability)).encode())
        result = analyze_lower_frame(b"jpeg-bytes", state, 12)
    assert result == {"kills": 2, "is_dead": expected}
    request = urlopen.call_args.args[0]
    assert request.full_url.endswith("/accounts/test-account/ai/run/@cf/cloudflare/clef-flash")
    assert request.get_header("Authorization") == "Bearer test-token"
    assert urlopen.call_args.kwargs["timeout"] == 12
    payload = json.loads(request.data)
    assert payload["state"] == state
    assert "JSON形式で回答" in payload["state"]
    assert payload["images"] == [
        {"content_type": "image/jpeg", "base64": base64.b64encode(b"jpeg-bytes").decode()}
    ]
    assert payload["questions"]["kills"]["criteria"] == {str(i): str(i) for i in range(5)}
    assert payload["questions"]["is_dead"]["type"] == "noul"


@pytest.mark.parametrize("choice", ["5", "-1", "unknown", 2, None, {}])
def test_invalid_kill_choice(choice: object) -> None:
    payload = _response()
    payload["result"]["answers"]["kills"]["choice"] = choice
    with patch(
        "src.clef_client.urllib.request.urlopen",
        return_value=io.BytesIO(json.dumps(payload).encode()),
    ):
        with pytest.raises(RuntimeError, match="invalid kills/death"):
            analyze_lower_frame(b"jpeg", "state", 12)


@pytest.mark.parametrize("probability", [-0.1, 1.1, "0.8", True, None, float("nan")])
def test_invalid_death_probability(probability: object) -> None:
    payload = _response()
    payload["result"]["answers"]["is_dead"]["noul"] = probability
    with patch(
        "src.clef_client.urllib.request.urlopen",
        return_value=io.BytesIO(json.dumps(payload).encode()),
    ):
        with pytest.raises(RuntimeError, match="invalid kills/death"):
            analyze_lower_frame(b"jpeg", "state", 12)


@pytest.mark.parametrize(
    "payload", [{"success": False}, {}, [], {"success": True, "result": {"answers": {}}}]
)
def test_invalid_envelope(payload: object) -> None:
    with patch(
        "src.clef_client.urllib.request.urlopen",
        return_value=io.BytesIO(json.dumps(payload).encode()),
    ):
        with pytest.raises(RuntimeError, match="invalid kills/death"):
            analyze_lower_frame(b"jpeg", "state", 12)


def test_invalid_json() -> None:
    with patch("src.clef_client.urllib.request.urlopen", return_value=io.BytesIO(b"not json")):
        with pytest.raises(RuntimeError, match="invalid JSON"):
            analyze_lower_frame(b"jpeg", "state", 12)


@pytest.mark.parametrize(
    "error,expected",
    [
        (urllib.error.HTTPError("secret-account", 401, "secret-token", None, None), "HTTP 401"),
        (urllib.error.URLError("secret-account secret-token"), "connection failed"),
        (TimeoutError("secret-token"), "connection failed"),
    ],
)
def test_failures_do_not_expose_credentials(error: Exception, expected: str) -> None:
    with patch("src.clef_client.urllib.request.urlopen", side_effect=error):
        with pytest.raises(RuntimeError, match=expected) as captured:
            analyze_lower_frame(b"jpeg", "state", 12)
    assert "secret" not in str(captured.value)
    assert captured.value.__suppress_context__


def test_missing_configuration_fails_before_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN")
    with patch("src.clef_client.urllib.request.urlopen") as urlopen:
        with pytest.raises(RuntimeError, match="CLOUDFLARE_API_TOKEN"):
            analyze_lower_frame(b"jpeg", "state", 12)
    urlopen.assert_not_called()


def test_explicit_gemini_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOWER_ANALYSIS_PROVIDER", "gemini")
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN")
    assert lower_configuration_error() is None


def test_invalid_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOWER_ANALYSIS_PROVIDER", "unknown")
    assert lower_configuration_error() == "LOWER_ANALYSIS_PROVIDER must be clef or gemini"


@patch("src.battle_analyzer._create_client")
def test_lower_only_routes_to_clef_with_production_crop(create_client: MagicMock) -> None:
    analyzer = BattleAnalyzer(timeout=15)
    frame = np.zeros((400, 800, 3), dtype=np.uint8)
    with patch(
        "src.battle_analyzer.analyze_lower_frame", return_value={"kills": 2, "is_dead": True}
    ) as clef:
        assert analyzer.analyze_frame_lower_only(frame, "00m01s") == {"kills": 2, "is_dead": True}
    image, state, timeout = clef.call_args.args
    decoded = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (60, 400, 3)
    assert state == LOWER_HALF_SYSTEM_PROMPT.split("■ 出力フォーマット")[0]
    assert timeout == 15
    create_client.return_value.models.generate_content.assert_not_called()


@patch("src.battle_analyzer._create_client")
def test_split_uses_gemini_upper_and_clef_lower(create_client: MagicMock) -> None:
    create_client.return_value.models.generate_content.return_value.text = '{"my_team_count": 50}'
    analyzer = BattleAnalyzer()
    with patch(
        "src.battle_analyzer.analyze_lower_frame", return_value={"kills": 1, "is_dead": False}
    ) as clef:
        result = analyzer.analyze_frame_split(np.zeros((400, 800, 3), dtype=np.uint8), "00m01s")
    assert result == {"my_team_count": 50, "kills": 1, "is_dead": False}
    create_client.return_value.models.generate_content.assert_called_once()
    clef.assert_called_once()


@patch("src.battle_analyzer._create_client")
def test_upper_does_not_require_cloudflare(
    create_client: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN")
    create_client.return_value.models.generate_content.return_value.text = '{"my_team_count": 50}'
    analyzer = BattleAnalyzer()
    with patch("src.battle_analyzer.analyze_lower_frame") as clef:
        assert analyzer.analyze_frame_upper_only(
            np.zeros((400, 800, 3), dtype=np.uint8), "00m01s"
        ) == {"my_team_count": 50}
    clef.assert_not_called()


@patch("src.battle_analyzer._create_client")
def test_explicit_gemini_lower_rollback(
    create_client: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOWER_ANALYSIS_PROVIDER", "gemini")
    create_client.return_value.models.generate_content.return_value.text = (
        '{"kills": 1, "is_dead": false}'
    )
    analyzer = BattleAnalyzer()
    with patch("src.battle_analyzer.analyze_lower_frame") as clef:
        assert analyzer.analyze_frame_lower_only(
            np.zeros((400, 800, 3), dtype=np.uint8), "00m01s"
        ) == {"kills": 1, "is_dead": False}
    create_client.return_value.models.generate_content.assert_called_once()
    clef.assert_not_called()
