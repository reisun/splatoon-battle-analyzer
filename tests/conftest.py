# pytest-ruff plugin handles ruff check and ruff format --check
# via the --ruff and --ruff-format flags in pyproject.toml [tool.pytest.ini_options]

import os

import pytest

os.environ.setdefault("SHARED_TEMP_DIR", "/tmp")


@pytest.fixture(autouse=True)
def clef_test_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use dummy Cloudflare settings independently of developer credentials."""
    monkeypatch.setenv("LOWER_ANALYSIS_PROVIDER", "clef")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "test-account")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "test-token")
