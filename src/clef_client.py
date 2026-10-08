"""Cloudflare Clef-flash adapter for lower-frame kills/death decisions."""

import base64
import json
import os
import urllib.error
import urllib.request


def lower_configuration_error() -> str | None:
    """Describe missing lower-analysis configuration without exposing values."""
    provider = os.environ.get("LOWER_ANALYSIS_PROVIDER", "clef")
    if provider == "gemini":
        return None
    if provider != "clef":
        return "LOWER_ANALYSIS_PROVIDER must be clef or gemini"
    missing = [
        key
        for key in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN")
        if not os.environ.get(key, "").strip()
    ]
    if missing:
        return "Clef lower analysis requires " + ", ".join(missing)
    return None


def analyze_lower_frame(image_bytes: bytes, state: str, timeout: int) -> dict:
    """Return the existing kills/is_dead fields using the benchmarked prompt.

    Error messages exclude request URLs, response bodies and exception chains,
    which can contain the Cloudflare account ID or authentication headers.
    """
    error = lower_configuration_error()
    if error:
        raise RuntimeError(error)
    account = os.environ["CLOUDFLARE_ACCOUNT_ID"].strip()
    token = os.environ["CLOUDFLARE_API_TOKEN"].strip()
    questions = {
        "kills": {
            "type": "choice",
            "instructions": "「◯◯ をたおした！」の完全一致の表示数を選択。不明瞭な場合は0。",
            "criteria": {str(i): str(i) for i in range(5)},
        },
        "is_dead": {
            "type": "noul",
            "instructions": (
                "自プレイヤーはデス中ですか？「復活まであと」の表示や画面暗転があればtrue。"
                "不明瞭な場合はfalse。"
            ),
        },
    }
    body = {
        "model": "clef-flash",
        "state": state,
        "images": [
            {"content_type": "image/jpeg", "base64": base64.b64encode(image_bytes).decode()}
        ],
        "questions": questions,
    }
    request = urllib.request.Request(
        f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/cloudflare/clef-flash",
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Clef API call failed (HTTP {exc.code})") from None
    except (urllib.error.URLError, OSError):
        raise RuntimeError("Clef API connection failed") from None
    except (ValueError, UnicodeError):
        raise RuntimeError("Clef API returned invalid JSON") from None

    try:
        if payload["success"] is not True:
            raise ValueError
        answers = payload["result"]["answers"]
        kills = answers["kills"]
        death = answers["is_dead"]
        if kills["type"] != "choice" or death["type"] != "noul":
            raise ValueError
        choice = kills["choice"]
        probability = death["noul"]
        if not isinstance(choice, str) or choice not in questions["kills"]["criteria"]:
            raise ValueError
        if type(probability) not in (float, int) or not 0 <= probability <= 1:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise RuntimeError("Clef API returned invalid kills/death decisions") from None

    return {"kills": int(choice), "is_dead": probability >= 0.5}
