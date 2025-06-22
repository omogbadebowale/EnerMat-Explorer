# lead_free_screening.py – automated binary screens for EnerMat manuscript
"""
Run high‑throughput binary (A–B) screenings for the four Pb‑containing systems
specified by the PI and for two exemplar lead‑free pairs.  The script saves
CSV outputs for each system, exports high‑resolution Plotly figures, and
writes a concise summary table ready to paste into the manuscript.

Usage
-----
$ export MP_API_KEY="<32‑character key>"
$ python lead_free_screening.py

Outputs (created under ./results/)
----------------------------------
├── csv/
│   ├── CsPbBr3_CsPbI3.csv
│   ├── CsSnBr3_CsPbBr3.csv
│   ├── CsSnBr3_CsPbI3.csv
│   ├── CsSnBr3_CsSnCl3.csv
│   ├── CsSnBr3_CsSnI3.csv
│   └── Cs2AgBiBr6_Cs2AgInCl6.csv
└── figs/
    ├── scatter_CsPbBr3_CsPbI3.svg
    ├── scatter_CsSnBr3_CsPbBr3.svg
    ├── scatter_CsSnBr3_CsPbI3.svg
    ├── scatter_CsSnBr3_CsSnCl3.svg
    ├── scatter_CsSnBr3_CsSnI3.svg
    └── scatter_Cs2AgBiBr6_Cs2AgInCl6.svg

A final file "summary_table.csv" collects the top‑ranked composition(s) from
all systems with stability ≥ 0.99.
"""
from __future__ import annotations

import os
from pathlib import Path
import datetime as dt

import pandas as pd
import plotly.express as px

# ---- EnerMat backend ---------------------------------------------------------
# Assumes this script sits at repo root and the backend module is importable.
from backend.perovskite_utils import mix_abx3  # type: ignore

# ---- User‑tunable knobs ------------------------------------------------------
RH = 50        # relative humidity [%]
TEMP = 25      # temperature [°C]
BG_WINDOW = (1.0, 1.4)  # eV
BOWING = 0.30  # eV
DX = 0.05      # composition step
STABILITY_THRESHOLD = 0.99
TOP_N_EXPORT = 10        # rows to keep in per‑system CSV

# Binary systems to screen -----------------------------------------------------
SYSTEMS_PB = [
    ("CsPbBr3", "CsPbI3"),
    ("CsSnBr3", "CsPbBr3"),
    ("CsSnBr3", "CsPbI3"),
    ("CsSnBr3", "CsSnCl3"),
]
SYSTEMS_LEADFREE = [
    ("CsSnBr3", "CsSnI3"),
    ("Cs2AgBiBr6", "Cs2AgInCl6"),
]
SYSTEMS = SYSTEMS_PB + SYSTEMS_LEADFREE

# Output folders ---------------------------------------------------------------
ROOT = Path("results")
CSV_DIR = ROOT / "csv"
FIG_DIR = ROOT / "figs"
for d in (CSV_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Convenience ------------------------------------------------------------------
def nice_tag(a: str, b: str) -> str:
    """Return a file‑friendly tag like CsSnBr3_CsPbI3."""
    return f"{a}_{b}".replace("/", "-")


def run_and_save(a: str, b: str) -> pd.DataFrame:
    """Run EnerMat binary screen for (A, B) and write CSV + figure."""
    tag = nice_tag(a, b)
    print(f"▶ Screening {tag} …", flush=True)
    df = mix_abx3(
        formula_A=a,
        formula_B=b,
        rh=RH,
        temp=TEMP,
        bg_window=BG_WINDOW,
        bowing=BOWING,
        dx=DX,
    )

    if df.empty:
        print(f"  ⚠️  No data returned for {tag} – skipping.")
        return df

    # Save trimmed CSV -------------------------------------------------------
    top = (
        df.query("stability >= @STABILITY_THRESHOLD")
        .nlargest(TOP_N_EXPORT, "score")
        .reset_index(drop=True)
    )
    csv_path = CSV_DIR / f"{tag}.csv"
    top.to_csv(csv_path, index=False)
    print(f"   ✔ CSV → {csv_path.relative_to(ROOT)} (n={len(top)})")

    # Plot scatter -----------------------------------------------------------
    fig = px.scatter(
        df,
        x="stability",
        y="Eg",
        color="score",
        color_continuous_scale="Turbo",
        hover_data=["formula", "x", "Eg", "stability", "score"],
        width=800,
        height=600,
    )
    # emphasise top 20 % halo
    q80 = df["score"].quantile(0.8)
    mask = df["score"] >= q80
    fig.add_scatter(
        x=df.loc[mask, "stability"],
        y=df.loc[mask, "Eg"],
        mode="markers",
        marker=dict(size=12, symbol="circle-open", line=dict(width=2, color="black")),
        hoverinfo="skip",
        showlegend=False,
    )
    fig.update_layout(template="plotly_white", title=tag)

    fig_path = FIG_DIR / f"scatter_{tag}.svg"
    fig.write_image(fig_path, scale=2)
    print(f"   ✔ Figure → {fig_path.relative_to(ROOT)}")

    return top


def main() -> None:
    summary_rows: list[dict] = []

    for a, b in SYSTEMS:
        top_rows = run_and_save(a, b)
        if top_rows.empty:
            continue
        best = top_rows.iloc[0]
        summary_rows.append({
            "system": f"{a}/{b}",
            "best_x": best.get("x", "-"),
            "Eg": best.Eg,
            "stability": best.stability,
            "score": best.score,
            "formula": best.formula,
        })

    # Collate summary --------------------------------------------------------
    if summary_rows:
        summary = pd.DataFrame(summary_rows).sort_values("score", ascending=False)
        summary_path = ROOT / "summary_table.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\n📊 Summary table → {summary_path.relative_to(ROOT)}")

    print("\nDone –", dt.datetime.now().strftime("%Y-%m-%d %H:%M"))


if __name__ == "__main__":
    main()
