"""설정 파일 로더"""
from pathlib import Path
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
