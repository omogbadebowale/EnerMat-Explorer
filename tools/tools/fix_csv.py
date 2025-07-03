"""
tools/fix_csv.py – convert the legacy pipe-table to a real CSV.

• pulls all rows that start with “|  1 | …” etc.
• converts Eg_eV:
   – plain number  → float
   – range “1.27–1.38” or “1.27-1.38” → midpoint 1.325
• drops anything that still isn’t numeric
• writes back comma-separated CSV
"""

import re, pathlib, pandas as pd, numpy as np

path = pathlib.Path("backend/data/perovskite_bandgap_merged.csv")
text = path.read_text(encoding="utf-8")

rows = []
for line in text.splitlines():
    # recognise a markdown row like: | 12 | CsSn0.4Pb0.6I3 | 1.30 | … |
    if re.match(r"\s*\|\s*\d+\s*\|", line):
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) == 5:
            rows.append(cells)

df = pd.DataFrame(rows,
                  columns=["ID", "Composition", "Eg_eV",
                           "Measurement_Comment", "Reference_URL"])

def to_float(val):
    val = val.replace(" ", "")
    if "–" in val or "-" in val:
        lo, hi = re.split("[–-]", val)
        try:
            return (float(lo) + float(hi)) / 2
        except ValueError:
            return np.nan
    try:
        return float(val)
    except ValueError:
        return np.nan

df["Eg_eV"] = df["Eg_eV"].apply(to_float)
df = df.dropna(subset=["Eg_eV"]).astype({"ID": int, "Eg_eV": float})

df.to_csv(path, index=False)
print(f"✅ wrote {len(df)} numeric rows to {path}")
