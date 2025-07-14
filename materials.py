# materials.py — source of alloy end-member formulas

import yaml
from pathlib import Path

def load_end_members(path="data/end_members.yaml"):
    """Load perovskite end-member formulas from YAML file."""
    with open(Path(path), "r") as f:
        data = yaml.safe_load(f)
    return data.get("formulas", [])
