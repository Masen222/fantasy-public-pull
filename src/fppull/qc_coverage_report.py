#!/usr/bin/env python3
"""
Quick coverage report so we can see whether our inputs support all roster types.

- Summarizes league_player_points.csv by lineup_slot and position
- Flags if IDP (slot 17) appears but we have no IDP stat columns in player_week_points
- Exits with nonzero status if gaps are detected (so CI can catch it)

Usage:
  python src/fppull/qc_coverage_report.py
"""

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"

def _latest_season_folder() -> Path:
    seasons = sorted(p for p in PROC.glob("season_*") if p.is_dir())
    if not seasons:
        print("No processed/season_* folder found.", file=sys.stderr)
        sys.exit(1)
    return seasons[-1]

def main():
    season_dir = _latest_season_folder()
    league_player = season_dir / "league_player_points.csv"
    player_points = season_dir / "player_week_points.csv"

    if not league_player.exists():
        print(f"Missing {league_player}", file=sys.stderr)
        sys.exit(1)
    if not player_points.exists():
        print(f"Missing {player_points}", file=sys.stderr)
        sys.exit(1)

    df_lp = pd.read_csv(league_player)
    df_pp = pd.read_csv(player_points)

    # ---- Part 1: coverage by lineup_slot ----
    ls = pd.to_numeric(df_lp.get("lineup_slot"), errors="coerce")
    by_slot = (df_lp.assign(lineup_slot=ls)
                     .groupby("lineup_slot", dropna=False)
                     .agg(rows=("athlete_name","size"),
                          sum_pts=("pts_ppr","sum"))
                     .sort_index())

    print("\n=== Coverage by lineup_slot ===")
    print(by_slot.to_string())

    # ---- Part 2: coverage by position (if present) ----
    if "position" in df_pp.columns:
        by_pos = (df_pp.fillna({"position": ""})
                        .groupby("position", dropna=False)
                        .agg(rows=("athlete_name","size"),
                             max_pts=("pts_ppr","max"),
                             sum_pts=("pts_ppr","sum"))
                        .sort_values("sum_pts", ascending=False))
        print("\n=== Coverage by position (from player_week_points) ===")
        print(by_pos.to_string())
    else:
        print("\n(position column missing in player_week_points.csv)")

        # ---- Part 3: IDP plumbing check ----
    # ESPN common: lineup_slot 17 is an IDP slot in many leagues.
    # If slot 17 exists with nonzero rows, but our player_week_points doesn't
    # have any obvious IDP stat columns, we raise a clear error.
    idp_present = 17.0 in by_slot.index and by_slot.loc[17.0, "rows"] > 0
    
    # Very conservative: look for typical offensive columns; if those are the *only* sources,
    # we assume no IDP support yet.
    offensive_cols = {"pass_yds","rush_yds","rec_yds","rec_rec","pass_td","rush_td","rec_td","pass_int"}
    stat_cols = set(c for c in df_pp.columns if c.startswith(("pass_","rush_","rec_","xp_","k_","fum_")))
    has_only_offense = (stat_cols != set()) and stat_cols.issubset(offensive_cols.union({"xp_made","k_fga","k_fgm","fum_lost"}))

    if idp_present and has_only_offense:
        print("\n❌ IDP lineup slot detected (slot 17) but no IDP stat support present in player_week_points.")
        print("   Next step: add IDP wide stats + statId→metric mapping, then compute pts_idp per your league scoring.")
        # Nonzero exit so CI / scripts can stop and surface the problem.
        sys.exit(2)

    print("\n✅ Coverage check passed.")
    sys.exit(0)

if __name__ == "__main__":
    main()