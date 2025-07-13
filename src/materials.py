# src/materials.py
"""
Single source of truth for end-member properties.
Loads data/end_members.yaml once and caches it.
"""

import yaml
import pathlib
import functools

@functools.lru_cache()
def load_end_members(path: str = "data/end_members.yaml") -> dict:
    with open(pathlib.Path(path)) as f:
        rows = yaml.safe_load(f)
    # map "CsSnBr3" → {Eg:…, Eox_e:…, Ehull:…}
    return {row["formula"]: row for row in rows}
