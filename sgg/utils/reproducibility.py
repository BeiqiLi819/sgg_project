from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch


def save_run_manifest(
    cfg: Mapping[str, Any],
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    filename: str = "run_manifest.json",
) -> Path:
    """Persist the resolved config and launch metadata beside experiment output."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "argv": list(sys.argv),
        "cwd": str(Path.cwd()),
        "config_path": "" if config_path is None else str(config_path),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cfg": cfg,
    }
    path = output_dir / filename
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    if config_path is not None:
        source = Path(config_path)
        if source.is_file():
            shutil.copy2(source, output_dir / "source_config.py")
    return path
