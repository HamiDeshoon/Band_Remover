"""Config persistence — reads/writes config.json in the user's AppData folder."""
from __future__ import annotations
import json
import os
from pathlib import Path

_CONFIG_DIR = Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming")) / "DeStripe"
CONFIG_PATH = _CONFIG_DIR / "config.json"

_DEFAULTS: dict = {
    "model_path": "",
    "last_input_dir": "",
    "last_output_dir": "",
    "device_pref": "auto",
}


def load() -> dict:
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            return {**_DEFAULTS, **data}
        except Exception:
            pass
    return dict(_DEFAULTS)


def save(cfg: dict) -> None:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
