import yaml, pathlib, functools

@functools.lru_cache()
def load_end_members(path="data/end_members.yaml"):
    """Return {formula: property-dict} mapping."""
    with open(path) as f:
        rows = yaml.safe_load(f)
    return {row["formula"]: row for row in rows}
