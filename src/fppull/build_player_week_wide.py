import json, re, os
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
IN_FILE_TPL  = "data/processed/season_{season}/player_week_stats_long.csv"
OUT_FILE_TPL = "data/processed/season_{season}/player_week_stats_wide.csv"

def parse_pair(val: str, sep=r"[/-]") -> Tuple[int, int]:
    """Parse things like '23/35' or '3-4' → (23, 35)."""
    if not isinstance(val, str):
        return 0, 0
    m = re.match(rf"^\s*(\d+)\s*{sep}\s*(\d+)\s*$", val)
    if not m:
        return 0, 0
    return int(m.group(1)), int(m.group(2))

def to_int(val: Any) -> int:
    try:
        s = str(val).replace(",", "").strip()
        return int(s) if s != "" else 0
    except Exception:
        return 0

def extract_numeric(stat_group: str, stats: Dict[str, Any]) -> Dict[str, int]:
    """
    Map ESPN label soup → stable numeric keys.
    We handle common variants defensively.
    """
    out = dict(
        pass_cmp=0, pass_att=0, pass_yds=0, pass_td=0, pass_int=0,
        rush_att=0, rush_yds=0, rush_td=0,
        rec_tgt=0, rec_rec=0, rec_yds=0, rec_td=0,
        fum_lost=0,
        k_fgm=0, k_fga=0, xp_made=0, xp_att=0
    )

    # normalize keys (upper + strip)
    norm = {k.upper().strip(): v for k, v in stats.items()}

    sg = stat_group.lower()

    if "passing" in sg:
        # C/ATT, CMP/ATT, or separate C, ATT
        if "C/ATT" in norm:
            c, a = parse_pair(str(norm["C/ATT"]))
            out["pass_cmp"], out["pass_att"] = c, a
        elif "CMP/ATT" in norm:
            c, a = parse_pair(str(norm["CMP/ATT"]))
            out["pass_cmp"], out["pass_att"] = c, a
        else:
            # fallbacks
            out["pass_cmp"] = to_int(norm.get("CMP", norm.get("C")))
            out["pass_att"] = to_int(norm.get("ATT"))

        out["pass_yds"] = to_int(norm.get("YDS"))
        out["pass_td"]  = to_int(norm.get("TD"))
        out["pass_int"] = to_int(norm.get("INT"))

    elif "rushing" in sg:
        out["rush_att"] = to_int(norm.get("CAR", norm.get("ATT")))
        out["rush_yds"] = to_int(norm.get("YDS"))
        out["rush_td"]  = to_int(norm.get("TD"))

    elif "receiving" in sg:
        out["rec_tgt"] = to_int(norm.get("TGT", norm.get("TAR")))
        out["rec_rec"] = to_int(norm.get("REC", norm.get("RECEPTIONS")))
        out["rec_yds"] = to_int(norm.get("YDS"))
        out["rec_td"]  = to_int(norm.get("TD"))

    elif "fumbles" in sg or "misc" in sg:
        # Some summaries list fumbles under "Fumbles" or "Misc"
        # Common labels: FUM, LOST
        out["fum_lost"] = to_int(norm.get("LOST", norm.get("FUM LOST", norm.get("LST", norm.get("FL")))))

    elif "kicking" in sg:
        # Common: FGM-A, XPM-A, sometimes separate FGM/FGA, XPM/XPA
        if "FGM-A" in norm:
            m, a = parse_pair(str(norm["FGM-A"]))
            out["k_fgm"], out["k_fga"] = m, a
        else:
            out["k_fgm"] = to_int(norm.get("FGM", norm.get("FG")))
            out["k_fga"] = to_int(norm.get("FGA"))

        if "XPM-A" in norm:
            m, a = parse_pair(str(norm["XPM-A"]))
            out["xp_made"], out["xp_att"] = m, a
        else:
            out["xp_made"] = to_int(norm.get("XPM"))
            out["xp_att"]  = to_int(norm.get("XPA"))

    # Defensive groups exist too, but for fantasy we typically score them at team DST level,
    # which we’ll handle later from team stats. Per-player IDP could be added similarly.

    return out

def main():
    load_dotenv(ROOT / ".env")
    season = int(os.getenv("SEASON", "0"))
    if not season:
        raise SystemExit("Set SEASON in .env")

    in_path  = ROOT / IN_FILE_TPL.format(season=season)
    out_path = ROOT / OUT_FILE_TPL.format(season=season)

    df = pd.read_csv(in_path)
    # normalize group names and keep only offense/kickers
    df["stat_group"] = df["stat_group"].str.lower()
    KEEP_GROUPS = {"passing", "rushing", "receiving", "fumbles", "kicking"}
    df = df[df["stat_group"].isin(KEEP_GROUPS)]
    print("After filter, stat_group counts:\n", df["stat_group"].value_counts().to_string())
    
    # parse stats_json → dict
    parsed = df["stats_json"].apply(lambda s: json.loads(s) if isinstance(s, str) and s.strip() else {})
    numerics = [
        extract_numeric(sg, d) for sg, d in zip(df["stat_group"], parsed)
    ]
    num_df = pd.DataFrame(numerics)
    base_cols = ["season","week","event_id","team_abbr","athlete_id","athlete_name","position"]
    wide = pd.concat([df[base_cols].reset_index(drop=True), num_df], axis=1)

    # aggregate across stat groups → single row per player/week
    agg_cols = {c: "sum" for c in num_df.columns}
    wide = (
        wide.groupby(base_cols, as_index=False)
            .agg(agg_cols)
            .sort_values(["week","team_abbr","athlete_name"])
            .reset_index(drop=True)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_path, index=False)
    print(f"✅ Wrote {out_path} with {len(wide):,} player-week rows.")

    # Quick peek
    print("\nSample:")
    print(wide.head(12).to_string(index=False))

if __name__ == "__main__":
    main()