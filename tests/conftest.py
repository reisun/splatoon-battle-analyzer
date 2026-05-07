# pytest-ruff plugin handles ruff check and ruff format --check
# via the --ruff and --ruff-format flags in pyproject.toml [tool.pytest.ini_options]

import os

os.environ.setdefault("SHARED_TEMP_DIR", "/tmp")
