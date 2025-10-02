# src/fppull/audit_scoring_table.py
"""
Quick sanity auditor for ESPN scoring_table.csv

Usage:
  python src/fppull/audit_scoring_table.py
"""
import os, sys
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
IN_SCORING_TPL = "data/processed/season_{season}/espn/scoring_table.csv"

# The same mapping used in compute_points_dynamic
STATID_TO_METRIC = {
    3: ("pass_yds", "per_unit"),
    25: ("pass_td",  "count"),
    26: ("pass_int", "count"),
    24: ("rush_yds", "per_unit"),
    27: ("rush_td",  "count"),
    42: ("rec_rec",  "count"),
    43: ("rec_yds",  "per_unit"),
    44: ("rec_td",   "count"),
    52: ("fum_lost", "count"),
    86: ("xp_made",  "count"),
}

DEFAULTS = {
    "pass_yds": 0.04, "rush_yds": 0.10, "rec_yds": 0.10,
    "rec_rec": 1.0, "pass_td": 6.0, "rush_td": 6.0, "rec_td": 6.0,
    "pass_int": -2.0, "fum_lost": -2.0, "xp_made": 1.0,
}

def main():
    load_dotenv(ROOT / ".env")
    season = int(os.getenv("SEASON", "0") or "0")
    if not season:
        print("Set SEASON in .env", file=sys.stderr)
        sys.exit(1)

    p = ROOT / IN_SCORING_TPL.format(season=season)
    if not p.exists():
        print(f"Missing scoring_table: {p}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(p)
    df = df[pd.to_numeric(df["statId"], errors="coerce").notnull()].copy()
    df["statId"] = df["statId"].astype(int)
    df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0.0)

    rows = []
    for _, r in df.iterrows():
        sid = int(r["statId"]); pts = float(r["points"])
        if sid in STATID_TO_METRIC:
            metric, mode = STATID_TO_METRIC[sid]
            rows.append((sid, metric, mode, pts, DEFAULTS.get(metric)))
        else:
            rows.append((sid, "(unknown)", "", pts, None))

    out = pd.DataFrame(rows, columns=["statId","metric","mode","espn_points","default_if_clamped"])
    # Flag “danger”
    def flag(row):
        m = row["metric"]
        v = row["espn_points"]
        if m in ("pass_yds","rush_yds","rec_yds"):
            if v <= 0 or v > 0.5: return "PER_YD_BAD"
        if m == "rec_rec":
            if v <= 0.25: return "REC_TINY"
        if m in ("pass_td","rush_td","rec_td"):
            if v < 3 or v > 12: return "TD_OUTLIER"
        if m == "pass_int":
            if v >= 0: return "INT_NONNEG"
        if m == "(unknown)":
            return "UNKNOWN_ID"
        return ""
    out["flag"] = out.apply(flag, axis=1)

    # Show all rows with flags up top
    out = out.sort_values(["flag","metric","statId"], ascending=[False, True, True])
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()