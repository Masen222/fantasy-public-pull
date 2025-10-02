# src/fppull/join_league_points.py
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
PROC_DIR_TPL = ROOT / "data" / "processed" / "season_{season}"


def _read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _standardize_columns(df: pd.DataFrame, mapping_options: List[Dict[str, str]]) -> pd.DataFrame:
    """Try mappings until one fits; return df with standardized names."""
    cols = {c.lower(): c for c in df.columns}
    for mapping in mapping_options:
        can_use = all(src.lower() in cols for src in mapping.values())
        if can_use:
            rename_dict = {cols[src.lower()]: dst for dst, src in mapping.items()}
            return df.rename(columns=rename_dict)
    return df


def _coerce_int(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df


def _norm_name(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.normalize("NFKC")
         .str.replace(r"\s+", " ", regex=True)
         .str.strip()
    )


def _ensure_athlete_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coalesce any columns that look like athlete_id (athlete_id, athlete_id_x/y)
    into a single integer 'athlete_id' column, then drop the rest.
    Always returns a df with 'athlete_id'.
    """
    # find all candidate columns
    cand = [c for c in df.columns if c.lower().startswith("athlete_id")]
    if not cand:
        df = df.copy()
        df["athlete_id"] = pd.NA
        df["athlete_id"] = pd.to_numeric(df["athlete_id"], errors="coerce")
        return df

    # start with the first, then fillna from the rest
    out = pd.to_numeric(df[cand[0]], errors="coerce")
    for c in cand[1:]:
        out = out.fillna(pd.to_numeric(df[c], errors="coerce"))

    df = df.copy()
    df["athlete_id"] = out
    # drop any extra athlete_id* columns, keeping only the canonical one
    drop_cols = [c for c in cand if c != "athlete_id"]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def main() -> None:
    load_dotenv(ROOT / ".env")
    season = os.getenv("SEASON")
    if not season:
        print("Set SEASON in .env", file=sys.stderr)
        sys.exit(1)

    proc_dir = Path(str(PROC_DIR_TPL).format(season=season))

    # Inputs
    points_csv = proc_dir / "player_week_points.csv"           # from compute_points.py
    wide_csv   = proc_dir / "player_week_stats_wide.csv"       # to recover athlete_id for joining
    roster_csv = proc_dir / "espn" / "roster_week.csv"         # from league_context.py
    teams_csv  = proc_dir / "espn" / "teams.csv"               # optional

    # Outputs
    out_player_csv = proc_dir / "league_player_points.csv"
    out_totals_csv = proc_dir / "league_team_week_totals.csv"

    # Read
    points = _read_csv(points_csv)
    wide   = _read_csv(wide_csv)
    roster = _read_csv(roster_csv)
    teams  = _read_csv(teams_csv, required=False)

    if points.empty or wide.empty or roster.empty:
        raise SystemExit(
            "Missing one of the required inputs: "
            "player_week_points.csv, player_week_stats_wide.csv, espn/roster_week.csv"
        )

    # Points: normalize
    points.columns = [c.strip() for c in points.columns]
    points["week"] = pd.to_numeric(points["week"], errors="coerce").astype(int)
    points["athlete_name"] = _norm_name(points["athlete_name"])
    if "team_abbr" in points.columns:
        points["team_abbr"] = points["team_abbr"].astype(str).str.upper()

    # Wide: ensure keys present
    wide_needed = {"season", "week", "team_abbr", "athlete_id", "athlete_name"}
    if not wide_needed.issubset(set(wide.columns)):
        raise SystemExit(f"{wide_csv} is missing required columns to recover athlete_id: need {sorted(wide_needed)}")
    wide["athlete_name"] = _norm_name(wide["athlete_name"])
    wide["team_abbr"] = wide["team_abbr"].astype(str).str.upper()
    wide = wide.drop_duplicates(subset=["season", "week", "team_abbr", "athlete_name", "athlete_id"])

    # Roster standardization (accept multiple shapes)
    roster_std_options = [
        # Your current file (seen earlier):
        {
            "week": "week",
            "athlete_id": "player_id",
            "athlete_name": "player_name",
            "fantasy_team_id": "team_id",
            "lineup_slot": "lineup_slot",
        },
        # Other variants:
        {
            "week": "week",
            "athlete_id": "athlete_id",
            "athlete_name": "athlete",
            "fantasy_team_id": "fantasy_team_id",
            "fantasy_team_name": "fantasy_team_name",
            "lineup_slot": "lineup_slot",
        },
        {
            "week": "week",
            "athlete_id": "player_id",
            "athlete_name": "player_name",
            "fantasy_team_id": "team_id",
            "fantasy_team_name": "team_name",
            "lineup_slot": "slot",
        },
    ]
    roster_std = _standardize_columns(roster, roster_std_options)

    required_roster_min = {"week", "athlete_id", "athlete_name", "fantasy_team_id"}
    if not required_roster_min.issubset(set(roster_std.columns)):
        raise SystemExit(
            f"{roster_csv} lacks required columns after standardization. "
            f"Needed at least: {sorted(required_roster_min)}. Found: {list(roster_std.columns)}"
        )

    # Fill fantasy_team_name if missing
    if "fantasy_team_name" not in roster_std.columns:
        if not teams.empty and {"team_id", "team_name"}.issubset(set(teams.columns)):
            teams_norm = teams.rename(columns={"team_id": "fantasy_team_id", "team_name": "fantasy_team_name"})
            teams_norm["fantasy_team_id"] = pd.to_numeric(teams_norm["fantasy_team_id"], errors="coerce").fillna(0).astype(int)
            roster_std = roster_std.merge(
                teams_norm[["fantasy_team_id", "fantasy_team_name"]],
                on="fantasy_team_id",
                how="left",
                validate="m:1",
            )
        if "fantasy_team_name" not in roster_std.columns:
            roster_std["fantasy_team_name"] = roster_std["fantasy_team_id"].apply(lambda x: f"Team {int(x)}")

    _coerce_int(roster_std, ["week", "athlete_id", "fantasy_team_id"])
    roster_std["athlete_name"] = _norm_name(roster_std["athlete_name"])

    # --- 1) Merge points→wide on (season, week, team_abbr, athlete_name)
    merge_keys = ["season", "week", "team_abbr", "athlete_name"]
    pts_with_id = points.merge(
        wide[merge_keys + ["athlete_id"]],
        on=merge_keys,
        how="left",
        validate="m:1"
    )
    pts_with_id = _ensure_athlete_id(pts_with_id)

    # --- 2) Fallback: merge without team_abbr
    fallback2 = points.merge(
        wide[["season", "week", "athlete_name", "athlete_id"]].drop_duplicates(),
        on=["season", "week", "athlete_name"],
        how="left",
    )
    fallback2 = _ensure_athlete_id(fallback2)

    # choose better resolution
    if fallback2["athlete_id"].notna().sum() > pts_with_id["athlete_id"].notna().sum():
        pts_with_id = fallback2

    # --- 3) Fallback via roster names (recover athlete_id from roster if still missing)
    if pts_with_id["athlete_id"].isna().sum() > 0:
        roster_name_map = (
            roster_std[["week", "athlete_id", "athlete_name"]]
            .drop_duplicates(subset=["week", "athlete_name"])
        )
        fallback3 = pts_with_id.merge(
            roster_name_map,
            on=["week", "athlete_name"],
            how="left",
            suffixes=("", "_from_roster"),
        )
        fallback3 = _ensure_athlete_id(fallback3)
        pts_with_id = fallback3

    # finalize id type
    pts_with_id["athlete_id"] = pd.to_numeric(pts_with_id["athlete_id"], errors="coerce").fillna(0).astype(int)

    # --- Join to roster on (week, athlete_id)
    league_points = pts_with_id.merge(
        roster_std[["week", "athlete_id", "fantasy_team_id", "fantasy_team_name", "lineup_slot"]].drop_duplicates(),
        on=["week", "athlete_id"],
        how="left",
        validate="m:1"
    )

    matched = league_points["fantasy_team_id"].notna().sum()
    unmatched = len(league_points) - matched

    # Fill defaults for missing league context
    league_points["fantasy_team_id"] = league_points["fantasy_team_id"].fillna(0).astype(int)
    league_points["fantasy_team_name"] = league_points["fantasy_team_name"].fillna("UNROSTERED")
    if "lineup_slot" not in league_points.columns:
        league_points["lineup_slot"] = ""
    if "status" not in league_points.columns:
        league_points["status"] = ""

    # --- Write player-level league points
    out_player_csv.parent.mkdir(parents=True, exist_ok=True)
    league_points.to_csv(out_player_csv, index=False)

    # --- Aggregate to team-week totals
    def _is_starter(row) -> bool:
        slot = str(row.get("lineup_slot", "")).strip().upper()
        if slot in {"BN", "IR"}:
            return False
        return True

    league_points["__is_starter__"] = league_points.apply(_is_starter, axis=1)

    grp_cols = ["season", "week", "fantasy_team_id", "fantasy_team_name"]
    numeric_pts = ["pts_ppr", "pts_pass", "pts_rush", "pts_rec", "pts_kick", "pts_misc"]

    starters = (
        league_points[league_points["__is_starter__"] == True]  # noqa: E712
        .groupby(grp_cols, as_index=False)[numeric_pts]
        .sum()
        .rename(columns={c: f"{c}_starters" for c in numeric_pts})
    )

    all_roster = (
        league_points.groupby(grp_cols, as_index=False)[numeric_pts]
        .sum()
        .rename(columns={c: f"{c}_all" for c in numeric_pts})
    )

    totals = starters.merge(all_roster, on=grp_cols, how="outer").fillna(0)

    totals_out_cols = grp_cols + [
        "pts_ppr_starters", "pts_pass_starters", "pts_rush_starters", "pts_rec_starters", "pts_kick_starters", "pts_misc_starters",
        "pts_ppr_all", "pts_pass_all", "pts_rush_all", "pts_rec_all", "pts_kick_all", "pts_misc_all",
    ]
    totals = totals[totals_out_cols].sort_values(["week", "fantasy_team_name"])

    totals.to_csv(out_totals_csv, index=False)

    # --- Reporting
    print(f"✅ Wrote {out_player_csv} with {len(league_points):,} player-week rows (matched={matched:,}, unmatched={unmatched:,}).")
    print(f"✅ Wrote {out_totals_csv} with {len(totals):,} team-week rows.")
    print("\nSample player rows:")
    sample_cols = [
        "season","week","fantasy_team_name","team_abbr","athlete_name","position",
        "pts_ppr","pts_pass","pts_rush","pts_rec","pts_kick","pts_misc","status","lineup_slot"
    ]
    print(league_points[sample_cols].head(12).to_string(index=False))

    print("\nSample team totals:")
    print(totals.head(12).to_string(index=False))


if __name__ == "__main__":
    main()