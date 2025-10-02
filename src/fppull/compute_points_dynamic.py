# src/fppull/compute_points_dynamic.py
"""
Compute player fantasy points from public wide stats, loading weights
dynamically from ESPN league scoring (scoring_table.csv).

Assumptions (locked):
- FULL PPR (1.0 per reception) regardless of ESPN CSV quirks
- Sane guardrails for yards/TD/INT so a bad scoring row can't nuke results

Normalization:
- Coerce numeric columns
- Collapse to ONE row per (season, week, team_abbr, athlete_name) via MAX
  (handles cumulative feeds without double-counting)
- Detect yardage stored in tenths and divide by 10

Outputs:
- data/processed/season_{SEASON}/player_week_points.csv

Usage:
  export DEBUG_POINTS=1  # optional
  python src/fppull/compute_points_dynamic.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, List

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
    # NOTE: FG buckets (74,77,80,83,88) intentionally not applied until wide supports distance bins
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
    return weights


def _normalize_weights_full_ppr(weights: Dict[str, float]) -> Dict[str, float]:
    """
    LOCK full PPR and apply guardrails so bad ESPN rows can't corrupt scoring.
    """
    w = dict(weights)
    defaults = {
        "pass_yds": 0.04, "rush_yds": 0.10, "rec_yds": 0.10,
        "pass_td": 6.0,  "rush_td": 6.0,   "rec_td": 6.0,
        "rec_rec": 1.0,  # FULL PPR
        "pass_int": -2.0, "fum_lost": -2.0, "xp_made": 1.0,
    }

    # force PPR
    w["rec_rec"] = 1.0

    # per-yard guardrails (0 < per_yd <= 0.5)
    for k in ("pass_yds", "rush_yds", "rec_yds"):
        v = float(w.get(k, defaults[k]))
        if v <= 0 or v > 0.5:
            v = defaults[k]
        w[k] = v

    # TD guardrails (3..12, default 6 if too small)
    for k in ("pass_td", "rush_td", "rec_td"):
        v = float(w.get(k, defaults[k]))
        if v < 3.0: v = 6.0
        if v > 12.0: v = 12.0
        w[k] = v

    # interceptions must be negative
    pint = float(w.get("pass_int", defaults["pass_int"]))
    if pint > 0:
        pint = -abs(pint)
    w["pass_int"] = pint

    # ensure other defaults exist
    for k, v in defaults.items():
        w.setdefault(k, v)

    return w


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
    else:
        w["position"] = w["position"].fillna("").astype(str)

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


def _debug_preview(tag: str, df: pd.DataFrame, cols: List[str]) -> None:
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

    # Yardage unit normalization (tenths → yards)
    wide = _normalize_yard_units(wide)

    # Extra diagnostics (optional)
    if os.getenv("DEBUG_POINTS", "").strip() == "1":
        def _show_top(df, col, extra_cols=None, n=10):
            extra_cols = extra_cols or []
            keep = [c for c in (["season","week","team_abbr","athlete_name",col] + extra_cols) if c in df.columns]
            try:
                print(f"\n🔎 Top {n} by {col}:")
                print(df.sort_values(col, ascending=False)[keep].head(n).to_string(index=False))
            except Exception as e:
                print(f"⚠️ DEBUG top-{col} failed: {e}")

        if "rec_yds" in wide.columns: _show_top(wide, "rec_yds", extra_cols=["rec_rec"])
        if "rec_rec" in wide.columns: _show_top(wide, "rec_rec", extra_cols=["rec_yds"])
        if "rush_yds" in wide.columns: _show_top(wide, "rush_yds")
        if "pass_yds" in wide.columns: _show_top(wide, "pass_yds")

    _debug_preview("player-week (post-normalization)", wide, KEY_COLS + ["pass_yds","rush_yds","rec_rec","rec_yds"])

    # Load scoring weights and LOCK to full PPR with guardrails
    scoring_df = _load_scoring(season)
    raw_weights = _build_weight_config(scoring_df)
    if os.getenv("DEBUG_POINTS", "").strip() == "1":
        keyw = {k: raw_weights.get(k) for k in ["pass_yds","rush_yds","rec_yds","rec_rec","pass_td","rush_td","rec_td","pass_int","xp_made","fum_lost"]}
        print("\n🔎 Effective weights (raw):", {k: round(v, 4) if v is not None else None for k, v in keyw.items()})

    weights = _normalize_weights_full_ppr(raw_weights)

    if os.getenv("DEBUG_POINTS", "").strip() == "1":
        keyw = {k: weights.get(k) for k in ["pass_yds","rush_yds","rec_yds","rec_rec","pass_td","rush_td","rec_td","pass_int","xp_made","fum_lost"]}
        print("🔎 Effective weights (final):", {k: round(v, 4) if v is not None else None for k, v in keyw.items()})

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
    pts_kick = _safe(wide, "xp_made") * weights.get("xp_made", 0.0)
    pts_misc = _safe(wide, "fum_lost") * weights.get("fum_lost", 0.0)

    # Assemble output
    # NOTE: groupby drops non-numeric; ensure 'position' exists now
    if "position" not in wide.columns:
        wide["position"] = ""
    out = wide[["season", "week", "team_abbr", "athlete_name", "position"]].copy()
    out["pts_pass"] = pts_pass.round(2)
    out["pts_rush"] = pts_rush.round(2)
    out["pts_rec"]  = pts_rec.round(2)
    out["pts_kick"] = pts_kick.round(2)
    out["pts_misc"] = pts_misc.round(2)
    out["pts_ppr"]  = (out["pts_pass"] + out["pts_rush"] + out["pts_rec"] + out["pts_kick"] + out["pts_misc"]).round(2)

    # Post-scoring peek (optional)
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


if __name__ == "__main__":
    main()