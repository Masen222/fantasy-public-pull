# src/fppull/compute_points_dynamic.py
"""
Compute player fantasy points from public wide stats, but load weights
dynamically from ESPN league scoring (scoring_table.csv).

Current coverage (per available public wide stats):
- Passing: yards, TD, INT
- Rushing: yards, TD
- Receiving: receptions, yards, TD
- Fumbles lost (if available)
- Kicker XP made (if available in wide)
- Kicker FG made (no distance buckets yet — flagged as unsupported)

Not yet covered (flagged; will be added next iteration):
- FG distance buckets (0-39, 40-49, 50-59, 60+) and FG missed
- 2-pt conversions (pass/rush/rec)
- DST/Team Defense stats
- Return TD variants (KRTD, PRTD, etc.)

Outputs:
- data/processed/season_{SEASON}/player_week_points.csv

Usage:
  python src/fppull/compute_points_dynamic.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"

IN_WIDE_TPL = "data/processed/season_{season}/player_week_stats_wide.csv"
IN_SCORING_TPL = "data/processed/season_{season}/espn/scoring_table.csv"
OUT_POINTS_TPL = "data/processed/season_{season}/player_week_points.csv"

# --- Minimal mapping from ESPN statId -> our internal wide-stat keys or roles ---
# NOTE: These ids are commonly seen; your league's ids may differ. We match dynamically
# by ids found in scoring_table.csv; unknown ids are reported.
STATID_TO_METRIC = {
    # Passing
    3:  ("pass_yds", "per_unit", 1.0),   # we will multiply by the points value per yard
    25: ("pass_td",  "count",   1.0),
    26: ("pass_int", "count",   1.0),
    29: ("pass_2pt", "count",   1.0),    # not present in wide yet (placeholder)
    # Rushing
    24: ("rush_yds", "per_unit", 1.0),
    27: ("rush_td",  "count",   1.0),
    32: ("rush_2pt", "count",   1.0),    # placeholder; not in wide yet
    # Receiving
    42: ("rec_rec",  "count",   1.0),    # receptions (PPR)
    43: ("rec_yds",  "per_unit", 1.0),
    44: ("rec_td",   "count",   1.0),
    45: ("rec_2pt",  "count",   1.0),    # placeholder; not in wide yet
    # Fumbles
    52: ("fum_lost", "count",   1.0),
    # Kicking (placeholders — our wide has only total FGM/XP, not distance buckets)
    86: ("xp_made",  "count",   1.0),
    # FG buckets (unsupported for now — we’ll flag if present in scoring):
    # 74: FG 0-39, 77: FG 40-49, 80: FG 50-59, 83: FG 60+, 88: FG Missed
}

# Keys present in our public wide table
WIDE_COLUMNS = {
    "pass_yds", "pass_td", "pass_int",
    "rush_yds", "rush_td",
    "rec_rec", "rec_yds", "rec_td",
    "fum_lost",
    "k_fgm", "k_fga", "xp_made", "xp_att"
}


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
    # Normalize expected columns
    if "statId" not in df.columns or "points" not in df.columns:
        print("⚠️  scoring_table.csv missing required columns [statId, points].", file=sys.stderr)
        sys.exit(1)
    # Keep only numeric statId rows
    df = df[pd.to_numeric(df["statId"], errors="coerce").notnull()].copy()
    df["statId"] = df["statId"].astype(int)
    df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0.0)
    return df


def _build_weight_config(scoring_df: pd.DataFrame) -> Dict[str, float]:
    """
    Translate ESPN statId weights into a dict keyed by our 'wide' metric names.
      - 'per_unit' stats (yards): keep raw points as per-yard factor (e.g., 0.04)
      - 'count' stats (TD/INT/etc.): keep raw points per event
    We ignore items that map to placeholders not present in the wide CSV (2PT, FG buckets).
    """
    weights: Dict[str, float] = {}

    unsupported: Dict[int, float] = {}
    unknown: Dict[int, float] = {}

    for _, row in scoring_df.iterrows():
        sid = int(row["statId"])
        pts = float(row["points"])
        meta = STATID_TO_METRIC.get(sid)
        if not meta:
            # Unknown to our mapper — keep list so we can decide next step
            unknown[sid] = pts
            continue

        metric, mode, scale = meta
        if metric not in WIDE_COLUMNS:
            # Placeholder or not-yet-parsed column
            unsupported[sid] = pts
            continue

        if mode == "per_unit":
            weights[metric] = pts * scale
        else:
            weights[metric] = pts * scale

    # Console diagnostics (short, clear)
    if unsupported:
        print("⚠️  Scoring items present but not supported by current wide schema:")
        for sid, pts in sorted(unsupported.items()):
            print(f"    • statId {sid}: weight={pts}  (needs additional parsing; e.g., 2PT or FG buckets)")
    if unknown:
        # Quite normal — leagues often have many statIds we don't need for player offense
        print("ℹ️  Scoring statIds not used in this compute (unknown mapping):")
        print("    ", ", ".join(str(s) for s in sorted(unknown.keys())))

    # Fallback defaults for any commonly expected metric not provided in scoring
    # (use PPR baseline so compute never crashes)
    defaults = {
        "pass_yds": 0.04,
        "pass_td": 4.0,
        "pass_int": -2.0,
        "rush_yds": 0.1,
        "rush_td": 6.0,
        "rec_rec": 1.0,
        "rec_yds": 0.1,
        "rec_td": 6.0,
        "fum_lost": -2.0,
        "xp_made": 1.0,     # sane default
        # "k_fgm": 3.0  # we cannot honor distance buckets yet; leave out to avoid double-counting
    }
    for k, v in defaults.items():
        weights.setdefault(k, v)

    return weights


def _safe(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0.0)


def main():
    season = _load_env()

    # Input tables
    wide_path = ROOT / IN_WIDE_TPL.format(season=season)
    if not wide_path.exists():
        print(f"Missing wide CSV: {wide_path}. Run build_player_week_wide.py first.", file=sys.stderr)
        sys.exit(1)
    wide = pd.read_csv(wide_path)

    # === BEGIN: week-level normalization (dedupe + sanity) ===
    # Goal: ensure ONE row per (season, week, player) before computing points.
    # This prevents inflated totals when the "wide" source contains duplicate
    # rows (e.g., multiple stat snapshots, merges, or season-to-date artifacts).

    # 1) Identify the key and the numeric stat columns we care about
    _key = ["season", "week", "team_abbr", "athlete_name", "position"]
    _numeric_cols = [
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

    for c in _numeric_cols:
        if c not in wide.columns:
            wide[c] = 0.0

    # 2) Coerce numeric columns safely
    for c in _numeric_cols:
        wide[c] = pd.to_numeric(wide[c], errors="coerce").fillna(0.0)

    # 3) Group to one row per player-week and SUM numeric stats
    #    (If your source was already unique, this is a no-op.)
    wide = (
        wide.groupby(_key, as_index=False)[_numeric_cols]
            .sum()
    )

    # 4) Sanity warnings (non-fatal): flag impossible lines to guide upstream fixes
    _suspicious = wide[
        (wide["rec_yds"] > 300) |
        (wide["rec_rec"] > 25)  |
        (wide["rush_yds"] > 300)|
        (wide["pass_yds"] > 700)
    ]
    if len(_suspicious) > 0:
        print(f"⚠️  Sanity: {len(_suspicious)} player-week rows exceed plausible ranges. Example:")
        try:
            print(_suspicious[_key + ["pass_yds","rush_yds","rec_rec","rec_yds"]].head(8).to_string(index=False))
        except Exception:
            pass
    # === END: week-level normalization (dedupe + sanity) ===

    scoring_df = _load_scoring(season)
    weights = _build_weight_config(scoring_df)

    # Compute subtotals with loaded weights
    pts_pass = (
        _safe(wide, "pass_yds") * weights.get("pass_yds", 0.0)
        + _safe(wide, "pass_td") * weights.get("pass_td", 0.0)
        + _safe(wide, "pass_int") * weights.get("pass_int", 0.0)
        # 2-pt pass placeholder: wide has no column yet
    )

    pts_rush = (
        _safe(wide, "rush_yds") * weights.get("rush_yds", 0.0)
        + _safe(wide, "rush_td") * weights.get("rush_td", 0.0)
        # 2-pt rush placeholder
    )

    pts_rec = (
        _safe(wide, "rec_rec") * weights.get("rec_rec", 0.0)
        + _safe(wide, "rec_yds") * weights.get("rec_yds", 0.0)
        + _safe(wide, "rec_td") * weights.get("rec_td", 0.0)
        # 2-pt rec placeholder
    )

    # Kicking: we can honor XP made; FG distance buckets not available in wide (yet)
    pts_kick = (
        _safe(wide, "xp_made") * weights.get("xp_made", 0.0)
        # For now, DO NOT credit k_fgm to avoid mis-scoring vs distance buckets.
        # We will add FG bucket parsing in a follow-up.
    )

    pts_misc = _safe(wide, "fum_lost") * weights.get("fum_lost", 0.0)

    out = wide[[
        "season", "week", "team_abbr", "athlete_name", "position"
    ]].copy()

    out["pts_pass"] = pts_pass.round(2)
    out["pts_rush"] = pts_rush.round(2)
    out["pts_rec"]  = pts_rec.round(2)
    out["pts_kick"] = pts_kick.round(2)
    out["pts_misc"] = pts_misc.round(2)
    out["pts_ppr"]  = (out["pts_pass"] + out["pts_rush"] + out["pts_rec"] + out["pts_kick"] + out["pts_misc"]).round(2)

    out_path = ROOT / OUT_POINTS_TPL.format(season=season)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"✅ Wrote {out_path} with {len(out):,} player-week rows.")
    print("\nSample:")
    print(out.head(12).to_string(index=False))


if __name__ == "__main__":
    main()