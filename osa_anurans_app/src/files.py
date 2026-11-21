"""Write local JSON reports and model metadata without replacing partial files."""

import json
from pathlib import Path


def write_json(path: Path, value, *, default=None) -> None:
    """Serialize JSON, write a temporary sibling file, then replace the target.

    Rules:
    - Create parent directories when needed.
    - Reject NaN/infinity and unsupported objects before touching the target.
    - Pass default=str explicitly when a report should stringify paths/dates.
    - Add a trailing newline; preserve the existing target if serialization fails.
    """
    contents = json.dumps(value, indent=2, allow_nan=False, default=default) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)
