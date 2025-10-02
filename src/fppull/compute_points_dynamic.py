# src/fppull/compute_points_dynamic.py
"""
Compute player fantasy points from public wide stats, loading weights
dynamically from ESPN league scoring (scoring_table.csv).

Coverage (given current wide schema):
- Passing: yards, TD, INT
- Rushing: yards, TD
- Receiving: receptions, yards, TD
- Fumbles lost
- Kicker XP made (FG buckets intentionally not applied yet)

Normalization:
- Coerce numeric columns
- Collapse to one row per (season, week, team_abbr, athlete_name) using MAX
  (handles cumulative/incremental feeds without inflating totals)
- Detect yardage stored in tenths and downscale by 10

Outputs:
- data/processed/season_{SEASON}/player_week_points.csv

Usage:
  python src/fppull/compute_points_dynamic.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Paths / constants
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
IN_WIDE_TPL = "data/processed/season_{season}/player_week_stats_wide.csv"
IN_SCORING_TPL = "data/processed/season_{season}/espn/scoring_table.csv"
OUT_POINTS_TPL = "data/processed/season_{season}/player_week_points.csv"

# Minimal mapping from ESPN statId -> our internal wide-stat keys or roles
STATID_TO_METRIC: Dict[int, tuple[str, str, float]] = {
    # Passing
    3:  ("pass_yds", "per_unit", 1.0),
    25: ("pass_td",  "count",    1.0),
    26: ("pass_int", "count",    1.0),
    # Rushing
    24: ("rush_yds", "per_unit", 1.0),
    27: ("rush_td",  "count",    1.0),
    # Receiving
    42: ("rec_rec",  "count",    1.0),
    43: ("rec_yds",  "per_unit", 1.0),
    44: ("rec_td",   "count",    1.0),
    # Fumbles
    52: ("fum_lost", "count",    1.0),
    # Kicking (only XP supported in current wide)
    86: ("xp_made",  "count",    1.0),
    # NOTE: FG buckets (74,77,80,83,88) not applied until wide supports distance bins
}

WIDE_NUMERIC_COLS: List[str] = [
    # passing
    "pass_yds", "pass_td", "pass_int",
    # rushing
    "rush_yds", "rush_td",
    # receiving
    "rec_rec", "rec_yds", "rec_td",
    # misc
    "fum_lost",
    # kicking (coarse)
    "xp_made", "xp_att", "k_fgm", "k_fga",
]

YARD_COLS = ["pass_yds", "rush_yds", "rec_yds"]
KEY_COLS = ["season", "week", "team_abbr", "athlete_name"]  # do NOT include 'position' in grouping


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _load_env() -> int:
    load_dotenv(ROOT / ".env")
    s = os.getenv("SEASON", "").strip()
    if not s:
        print("Set SEASON in .env", file=sys.stderr)
        sys.exit(1)
    try:
        return int(s)
    except Exception:
        print("SEASON must be an integer.", file=sys.stderr)
        sys.exit(1)


def _load_scoring(season: int) -> pd.DataFrame:
    scoring_path = ROOT / IN_SCORING_TPL.format(season=season)
    if not scoring_path.exists():
        print(f"⚠️  {scoring_path} not found. Run fetch_espn_scoring.py first.", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(scoring_path)
    if "statId" not in df.columns or "points" not in df.columns:
        print("⚠️  scoring_table.csv missing required columns [statId, points].", file=sys.stderr)
        sys.exit(1)
    df = df[pd.to_numeric(df["statId"], errors="coerce").notnull()].copy()
    df["statId"] = df["statId"].astype(int)
    df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0.0)
    return df


def _build_weight_config(scoring_df: pd.DataFrame) -> Dict[str, float]:
    """Translate ESPN statId weights into a dict keyed by our 'wide' metric names."""
    weights: Dict[str, float] = {}
    unknown = []

    for _, row in scoring_df.iterrows():
        sid = int(row["statId"])
        pts = float(row["points"])
        meta = STATID_TO_METRIC.get(sid)
        if not meta:
            unknown.append(sid)
            continue
        metric, mode, scale = meta
        if mode not in ("per_unit", "count"):
            continue
        weights[metric] = pts * scale

    if unknown:
        print("ℹ️  Scoring statIds not used in this compute (unknown mapping):")
        print("    ", ", ".join(str(s) for s in sorted(set(unknown))))

    # Safe defaults so compute never crashes if some keys missing
    defaults = {
        "pass_yds": 0.04, "pass_td": 4.0, "pass_int": -2.0,
        "rush_yds": 0.1,  "rush_td": 6.0,
        "rec_rec": 1.0,   "rec_yds": 0.1, "rec_td": 6.0,
        "fum_lost": -2.0,
        "xp_made": 1.0,
    }
    for k, v in defaults.items():
        weights.setdefault(k, v)
    return weights


def _safe(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0.0)


def _ensure_columns(wide: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist and are properly typed."""
    w = wide.copy()

    # ID columns
    for k in KEY_COLS:
        if k not in w.columns:
            w[k] = ""
        w[k] = w[k].fillna("").astype(str)

    # optional position column (not part of key)
    if "position" not in w.columns:
        w["position"] = ""

    # numeric columns
    for c in WIDE_NUMERIC_COLS:
        if c not in w.columns:
            w[c] = 0.0
        w[c] = pd.to_numeric(w[c], errors="coerce").fillna(0.0)

    return w


def _collapse_player_weeks(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse to ONE row per (season, week, team_abbr, athlete_name) by taking MAX of
    numeric columns, which matches final box totals for cumulative feeds.
    """
    try:
        return (wide.groupby(KEY_COLS, as_index=False, dropna=False)[WIDE_NUMERIC_COLS].max())
    except TypeError:
        # Older pandas without dropna= argument
        return (wide.groupby(KEY_COLS, as_index=False)[WIDE_NUMERIC_COLS].max())


def _normalize_yard_units(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Detect yardage stored in tenths (common in some public feeds) and divide by 10.
    Heuristic: if any yardage column's max > 1000 for a single player-week, treat as tenths.
    """
    w = wide.copy()
    try:
        yard_max = max(float(w[c].max()) for c in YARD_COLS if c in w.columns)
    except Exception:
        yard_max = 0.0

    if yard_max > 1000.0:
        for c in YARD_COLS:
            if c in w.columns:
                w[c] = pd.to_numeric(w[c], errors="coerce").fillna(0.0) / 10.0
        print("🔧 Normalized yardage columns from tenths → yards (÷10).")
    return w


def _debug_preview(tag: str, df: pd.DataFrame, cols: list[str]) -> None:
    if os.getenv("DEBUG_POINTS", "").strip() != "1":
        return
    try:
        print(f"\n🔎 DEBUG {tag}:")
        print(df[cols].head(12).to_string(index=False))
    except Exception as e:
        print(f"⚠️ DEBUG preview failed: {e}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    season = _load_env()

    # Load inputs
    wide_path = ROOT / IN_WIDE_TPL.format(season=season)
    if not wide_path.exists():
        print(f"Missing wide CSV: {wide_path}. Run build_player_week_wide.py first.", file=sys.stderr)
        sys.exit(1)
    wide_raw = pd.read_csv(wide_path)

    # Normalize schema + types
    wide = _ensure_columns(wide_raw)

    # Collapse duplicates (cumulative feeds) → one row per player-week
    wide = _collapse_player_weeks(wide)

    # Ensure optional 'position' column exists after collapse (groupby drops non-numeric)
    if "position" not in wide.columns:
        wide["position"] = ""
    else:
        wide["position"] = wide["position"].fillna("").astype(str)

    # Yardage unit normalization (tenths → yards)
    wide = _normalize_yard_units(wide)

    # Extra diagnostics: show top raw yardage leaders to confirm plausibility
    if os.getenv("DEBUG_POINTS", "").strip() == "1":
        def _show_top(df, col, extra_cols=None, n=10):
            extra_cols = extra_cols or []
            keep = [c for c in (["season","week","team_abbr","athlete_name",col] + extra_cols) if c in df.columns]
            try:
                print(f"\n🔎 Top {n} by {col}:")
                print(df.sort_values(col, ascending=False)[keep].head(n).to_string(index=False))
            except Exception as e:
                print(f"⚠️ DEBUG top-{col} failed: {e}")

        # Print tops after normalization, before scoring, so we see raw inputs
        if "rec_yds" in wide.columns:
            _show_top(wide, "rec_yds", extra_cols=["rec_rec"])
        if "rec_rec" in wide.columns:
            _show_top(wide, "rec_rec", extra_cols=["rec_yds"])
        if "rush_yds" in wide.columns:
            _show_top(wide, "rush_yds")
        if "pass_yds" in wide.columns:
            _show_top(wide, "pass_yds")

    # Optional debug preview
    _debug_preview("player-week (post-normalization)", wide, KEY_COLS + ["pass_yds","rush_yds","rec_rec","rec_yds"])

    # Load scoring weights
    scoring_df = _load_scoring(season)
    weights = _build_weight_config(scoring_df)

    # Compute subtotal components
    pts_pass = (
        _safe(wide, "pass_yds") * weights.get("pass_yds", 0.0)
        + _safe(wide, "pass_td") * weights.get("pass_td", 0.0)
        + _safe(wide, "pass_int") * weights.get("pass_int", 0.0)
    )
    pts_rush = (
        _safe(wide, "rush_yds") * weights.get("rush_yds", 0.0)
        + _safe(wide, "rush_td") * weights.get("rush_td", 0.0)
    )
    pts_rec = (
        _safe(wide, "rec_rec") * weights.get("rec_rec", 0.0)
        + _safe(wide, "rec_yds") * weights.get("rec_yds", 0.0)
        + _safe(wide, "rec_td") * weights.get("rec_td", 0.0)
    )
    pts_kick = (
        _safe(wide, "xp_made") * weights.get("xp_made", 0.0)
    )
    pts_misc = _safe(wide, "fum_lost") * weights.get("fum_lost", 0.0)

    # Assemble output
    out = wide[["season", "week", "team_abbr", "athlete_name", "position"]].copy()
    out["pts_pass"] = pts_pass.round(2)
    out["pts_rush"] = pts_rush.round(2)
    out["pts_rec"]  = pts_rec.round(2)
    out["pts_kick"] = pts_kick.round(2)
    out["pts_misc"] = pts_misc.round(2)
    out["pts_ppr"]  = (out["pts_pass"] + out["pts_rush"] + out["pts_rec"] + out["pts_kick"] + out["pts_misc"]).round(2)

    # Post-scoring quick look (optional)
    if os.getenv("DEBUG_POINTS", "").strip() == "1":
        try:
            print("\n🔎 Top 10 by pts_ppr (post-scoring):")
            print(out.sort_values("pts_ppr", ascending=False)[
                ["season","week","team_abbr","athlete_name","position","pts_ppr","pts_pass","pts_rush","pts_rec"]
            ].head(10).to_string(index=False))
        except Exception as e:
            print(f"⚠️ DEBUG pts_ppr failed: {e}")

    # Save
    out_path = ROOT / OUT_POINTS_TPL.format(season=season)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"✅ Wrote {out_path} with {len(out):,} player-week rows.")
    if os.getenv("DEBUG_POINTS", "").strip() == "1":
        print("\n🔎 DEBUG sample output:")
        try:
            print(out.head(12).to_string(index=False))
        except Exception:
            pass


if __name__ == "__main__":
    main()