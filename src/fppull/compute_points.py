import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# Paths consistent with the rest of the repo
ROOT = Path(__file__).resolve().parents[2]
IN_FILE_TPL  = "data/processed/season_{season}/player_week_stats_wide.csv"
OUT_FILE_TPL = "data/processed/season_{season}/player_week_points.csv"


def _ensure_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a float Series for col; if missing, return zeros of correct length."""
    if col not in df.columns:
        return pd.Series([0.0] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[col].fillna(0), errors="coerce").fillna(0.0).astype("float64")


def main() -> None:
    load_dotenv(ROOT / ".env")
    season = os.getenv("SEASON")
    if not season:
        raise SystemExit("Set SEASON in .env")

    in_path = ROOT / IN_FILE_TPL.format(season=season)
    out_path = ROOT / OUT_FILE_TPL.format(season=season)

    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    df = pd.read_csv(in_path)

    # Base ids/labels we keep
    base_cols = [
        "season", "week", "event_id", "team_abbr",
        "athlete_id", "athlete_name", "position",
    ]
    for c in base_cols:
        if c not in df.columns:
            df[c] = None  # keep shape; better than hard failing for now

    # Pull numeric columns safely
    pass_yds = _ensure_numeric(df, "pass_yds")
    pass_td  = _ensure_numeric(df, "pass_td")
    pass_int = _ensure_numeric(df, "pass_int")

    rush_yds = _ensure_numeric(df, "rush_yds")
    rush_td  = _ensure_numeric(df, "rush_td")

    rec_rec  = _ensure_numeric(df, "rec_rec")
    rec_yds  = _ensure_numeric(df, "rec_yds")
    rec_td   = _ensure_numeric(df, "rec_td")

    fum_lost = _ensure_numeric(df, "fum_lost")

    k_fgm    = _ensure_numeric(df, "k_fgm")
    xp_made  = _ensure_numeric(df, "xp_made")

    # --- Scoring rules (PPR baseline) ---
    # Passing
    pts_pass = 0.04 * pass_yds + 4.0 * pass_td - 2.0 * pass_int
    # Rushing
    pts_rush = 0.10 * rush_yds + 6.0 * rush_td
    # Receiving (PPR = 1.0 per reception)
    pts_rec  = 1.0 * rec_rec + 0.10 * rec_yds + 6.0 * rec_td
    # Misc
    pts_misc = -2.0 * fum_lost
    # Kicking
    pts_kick = 3.0 * k_fgm + 1.0 * xp_made

    pts_total = pts_pass + pts_rush + pts_rec + pts_misc + pts_kick

    out = df[base_cols].copy()
    out["pts_pass"]  = pts_pass.round(2)
    out["pts_rush"]  = pts_rush.round(2)
    out["pts_rec"]   = pts_rec.round(2)
    out["pts_misc"]  = pts_misc.round(2)
    out["pts_kick"]  = pts_kick.round(2)
    out["pts_ppr"]   = pts_total.round(2)

    # Optional: keep a few raw stat columns for transparency/debug
    raw_keep = ["pass_yds", "pass_td", "pass_int",
                "rush_yds", "rush_td",
                "rec_rec", "rec_yds", "rec_td",
                "fum_lost", "k_fgm", "xp_made"]
    for c in raw_keep:
        if c in df.columns:
            out[c] = _ensure_numeric(df, c).astype(int)

    # Sort for readability
    out = out.sort_values(["season", "week", "team_abbr", "athlete_name"], kind="stable").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"✅ Wrote {out_path} with {len(out):,} player-week rows.")
    print("\nSample:")
    sample_cols = [
        "season","week","team_abbr","athlete_name","position",
        "pts_ppr","pts_pass","pts_rush","pts_rec","pts_kick","pts_misc"
    ]
    print(out[sample_cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()