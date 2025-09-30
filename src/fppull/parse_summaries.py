import os, json, re
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Iterable

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR_TPL = "data/raw/season_{season}/public/summaries"
OUT_DIR_TPL = "data/processed/season_{season}"
OUT_FILE = "player_week_stats_long.csv"

def newest_per_week(paths: Iterable[Path]) -> Dict[int, Path]:
    # filenames like: summary_y2025_w01_event4017..._20250930T231724Z.json
    by_week: Dict[int, List[Path]] = {}
    for p in paths:
        m = re.search(r"_w(\d{2})_", p.name)
        if not m:
            continue
        wk = int(m.group(1))
        by_week.setdefault(wk, []).append(p)
    # choose newest by timestamp in name
    return {wk: sorted(ps)[-1] for wk, ps in by_week.items()}

def extract_rows(summary: Dict[str, Any], season: int, week: int, event_id: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    box = (summary or {}).get("boxscore") or {}
    players_groups = box.get("players") or []   # list of {team, statistics: [...]}

    for team_block in players_groups:
        team = (team_block.get("team") or {})
        team_id = team.get("id")
        team_abbr = team.get("abbreviation") or team.get("displayName")

        for group in (team_block.get("statistics") or []):
            group_name = group.get("name") or group.get("displayName") or "unknown"
            labels = group.get("labels") or []
            for a in (group.get("athletes") or []):
                ath = a.get("athlete") or {}
                stats = a.get("stats") or []

                # Build stat mapping using labels & stats (if aligned)
                stat_map = {}
                for i, label in enumerate(labels):
                    if i < len(stats):
                        stat_map[label] = stats[i]

                rows.append({
                    "season": season,
                    "week": week,
                    "event_id": event_id,
                    "team_id": team_id,
                    "team_abbr": team_abbr,
                    "athlete_id": ath.get("id"),
                    "athlete_name": ath.get("displayName") or ath.get("fullName"),
                    "position": (ath.get("position") or {}).get("abbreviation"),
                    "stat_group": group_name,          # e.g., passing / rushing / receiving / kicking / defensive...
                    "stats_json": json.dumps(stat_map, separators=(",", ":")),  # keep raw for flexibility
                })
    return rows

def main():
    load_dotenv(ROOT / ".env")
    season = int(os.getenv("SEASON", "0"))
    weeks = [int(w.strip()) for w in os.getenv("WEEKS", "").split(",") if w.strip()]
    if not season or not weeks:
        raise SystemExit("Config error: set SEASON and WEEKS in .env")

    raw_dir = ROOT / RAW_DIR_TPL.format(season=season)
    out_dir = ROOT / OUT_DIR_TPL.format(season=season)
    out_dir.mkdir(parents=True, exist_ok=True)

    # collect the newest summary file per week *per event* (you may have many per week)
    # simpler approach: parse all summaries present; filenames contain week + event id
    all_files = sorted(raw_dir.glob("summary_y*_w*_event*_*.json"))
    if not all_files:
        raise SystemExit(f"No summary files found under {raw_dir}")

    print(f"Found {len(all_files)} summary JSON files.")

    rows: List[Dict[str, Any]] = []
    for p in all_files:
        # parse week & event id from filename
        m_wk = re.search(r"_w(\d{2})_", p.name)
        m_evt = re.search(r"_event(\d+)_", p.name)
        if not (m_wk and m_evt):
            continue
        wk = int(m_wk.group(1))
        evt = m_evt.group(1)

        if wk not in weeks:
            continue

        try:
            data = json.loads(p.read_text())
        except Exception as e:
            print(f"⚠️  skip {p.name}: {e}")
            continue

        rows.extend(extract_rows(data, season, wk, evt))

    if not rows:
        raise SystemExit("No rows extracted (unexpected).")

    df = pd.DataFrame(rows)
    # stable ordering for readability
    df = df.sort_values(["week", "event_id", "team_abbr", "stat_group", "athlete_name"]).reset_index(drop=True)

    out_path = out_dir / OUT_FILE
    df.to_csv(out_path, index=False)
    print(f"✅ Wrote {out_path} with {len(df):,} rows.")

    # small preview
    print("\nSample rows:")
    print(df.head(8).to_string(index=False))

if __name__ == "__main__":
    main()