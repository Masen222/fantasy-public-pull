# src/fppull/qc_week_check.py
import os
import sys
from pathlib import Path
import argparse
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
PROC_DIR_TPL = ROOT / "data" / "processed" / "season_{season}"

def _read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def _find_season(proc_root: Path) -> int:
    # Use .env first; else auto-detect the max processed season folder
    load_dotenv(ROOT / ".env")
    s = os.getenv("SEASON", "").strip()
    if s.isdigit():
        return int(s)
    seasons = []
    for d in (proc_root).glob("season_*"):
        try:
            seasons.append(int(d.name.split("_")[-1]))
        except Exception:
            pass
    if not seasons:
        print("No season found in processed dir and SEASON not set.", file=sys.stderr)
        sys.exit(1)
    return max(seasons)

def _load_official(proc_dir: Path) -> pd.DataFrame:
    """
    Prefer a canonical 'team_week_official.csv' if present.
    Fallback to 'scoreboard_week.csv' (typical ESPN export) and infer columns.
    Must return columns: season, week, fantasy_team_id, official_pts
    """
    # 1) Canonical file produced elsewhere in the pipeline
    official_csv = proc_dir / "espn" / "team_week_official.csv"
    if official_csv.exists():
        df = pd.read_csv(official_csv)
        # Flexible column mapping
        colmap = {
            "season": "season",
            "week": "week",
            "fantasy_team_id": "fantasy_team_id",
            "official_pts_starters": "official_pts",   # preferred
            "official_pts": "official_pts",
            "total_points": "official_pts",
        }
        have = {c.lower(): c for c in df.columns}
        need = ["season", "week", "fantasy_team_id"]
        for n in need:
            if n not in have:
                raise SystemExit(f"{official_csv} missing required column '{n}'")
        # find a points column
        pts_col = None
        for k in ["official_pts_starters", "official_pts", "total_points"]:
            if k in have:
                pts_col = have[k]
                break
        if not pts_col:
            raise SystemExit(f"{official_csv} missing any points column")
        return df.rename(columns={
            have["season"]: "season",
            have["week"]: "week",
            have["fantasy_team_id"]: "fantasy_team_id",
            pts_col: "official_pts",
        })[["season","week","fantasy_team_id","official_pts"]]

    # 2) Fallback: scoreboard_week.csv
    sb = proc_dir / "espn" / "scoreboard_week.csv"
    if sb.exists():
        df = pd.read_csv(sb)
        # Common shapes encountered:
        # - 'team_id' | 'week' | 'points' (or 'total_points')
        # - plus optional 'season'
        have = {c.lower(): c for c in df.columns}
        # derive season if not present (take from directory name)
        if "season" not in have:
            # inject a season column based on proc_dir
            try:
                season = int(str(proc_dir).split("season_")[-1])
            except Exception:
                season = 0
            df["season"] = season
            have["season"] = "season"
        # map team id
        team_col = have.get("fantasy_team_id") or have.get("team_id") or have.get("id")
        if not team_col:
            raise SystemExit(f"{sb} missing a team id column")
        # map points
        pts_col = None
        for c in ["official_pts","total_points","points","score"]:
            if c in have:
                pts_col = have[c]
                break
        if not pts_col:
            raise SystemExit(f"{sb} missing a points column (expected one of points/score/total_points)")
        return df.rename(columns={
            have["season"]:"season",
            have["week"]:"week",
            team_col:"fantasy_team_id",
            pts_col:"official_pts",
        })[["season","week","fantasy_team_id","official_pts"]]

    raise SystemExit("No official team totals found. Expected espn/team_week_official.csv or espn/scoreboard_week.csv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", type=int, nargs="*", help="Restrict to these weeks (space-separated).")
    args = ap.parse_args()

    season = _find_season(PROC_DIR_TPL.parent)
    proc_dir = Path(str(PROC_DIR_TPL).format(season=season))

    # Our computed totals (must exist)
    totals_csv = proc_dir / "league_team_week_totals.csv"
    if not totals_csv.exists():
        raise SystemExit(f"Missing {totals_csv}. Run join_league_points.py first.")
    totals = pd.read_csv(totals_csv)

    # Use starters (NOT all)
    need_cols = {"season","week","fantasy_team_id","fantasy_team_name","pts_ppr_starters"}
    if not need_cols.issubset(set(totals.columns)):
        raise SystemExit(f"{totals_csv} missing required columns {sorted(need_cols)}")

    # Drop UNROSTERED
    totals = totals[totals["fantasy_team_id"] != 0].copy()

    # Optional filter weeks
    if args.weeks:
        totals = totals[totals["week"].isin(args.weeks)].copy()

    # Load ESPN official
    official = _load_official(proc_dir)
    official["fantasy_team_id"] = pd.to_numeric(official["fantasy_team_id"], errors="coerce").astype("Int64")
    if args.weeks:
        official = official[official["week"].isin(args.weeks)].copy()

    # Join and compare
    left_cols = ["season","week","fantasy_team_id","fantasy_team_name","pts_ppr_starters"]
    comp = totals[left_cols].merge(
        official,
        on=["season","week","fantasy_team_id"],
        how="left",
        validate="m:1"
    )

    comp["official_pts"] = pd.to_numeric(comp["official_pts"], errors="coerce").fillna(0.0)
    comp["computed_starters"] = comp["pts_ppr_starters"].round(2)
    comp["delta"] = (comp["computed_starters"] - comp["official_pts"]).round(2)

    comp = comp.sort_values(["week","fantasy_team_id"])

    print("\n=== QC: Computed Starters vs Official Team Totals (starters only) ===")
    print(comp[["season","week","fantasy_team_id","fantasy_team_name","computed_starters","official_pts","delta"]].to_string(index=False))

    bad = comp[comp["delta"].abs() > 0.5]
    if len(bad):
        print("\n⚠️  Some teams did not match official starter totals. Inspect join keys & scoring weights.")
        print(bad[["week","fantasy_team_id","fantasy_team_name","delta"]].to_string(index=False))
    else:
        print("\n✅ All teams match official starter totals (within 0.5 pts).")

if __name__ == "__main__":
    main()