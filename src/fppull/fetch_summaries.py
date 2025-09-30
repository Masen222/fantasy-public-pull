import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
SCORE_DIR_TPL = "data/raw/season_{season}/public"
SUM_DIR_TPL    = "data/raw/season_{season}/public/summaries"

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())

def save_json(obj: Dict[str, Any], path: Path) -> None:
    text = json.dumps(obj, indent=2, sort_keys=True)
    path.write_text(text)
    rel = path.relative_to(ROOT)
    print(f"💾 wrote {rel} ({len(text)} bytes)")

def list_scoreboard_files(score_dir: Path, season: str, weeks: List[int]) -> List[Path]:
    files: List[Path] = []
    for w in weeks:
        # pick the newest file per week (you may have multiple from earlier tests)
        pattern = f"scoreboard_y{season}_w{w:02d}_*.json"
        candidates = sorted(score_dir.glob(pattern))
        if not candidates:
            print(f"⚠️  no scoreboard JSON found for week {w}; run public_pull first.")
            continue
        files.append(candidates[-1])  # newest by name
    return files

def extract_event_ids(score_json: Dict[str, Any]) -> List[str]:
    evts = score_json.get("events", []) or []
    return [e.get("id") for e in evts if e.get("id")]

def fetch_summary_public(event_id: str) -> Dict[str, Any]:
    """
    Public game summary (includes team + player stats).
    Example:
      https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event=401671973
    """
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"
    r = requests.get(url, params={"event": event_id}, timeout=30)
    r.raise_for_status()
    return r.json()

def main():
    load_dotenv(ROOT / ".env")
    season = os.getenv("SEASON")
    weeks = [int(w.strip()) for w in os.getenv("WEEKS", "").split(",") if w.strip()]

    if not season or not weeks:
        raise SystemExit("Config error: set SEASON and WEEKS in .env")

    score_dir = ROOT / SCORE_DIR_TPL.format(season=season)
    out_dir   = ensure_dir(ROOT / SUM_DIR_TPL.format(season=season))

    print("✅ Config loaded")
    print(f"  SEASON    : {season}")
    print(f"  WEEKS     : {weeks}")
    print(f"  SCORE DIR : {score_dir}")
    print(f"  OUT DIR   : {out_dir}")

    score_files = list_scoreboard_files(score_dir, season, weeks)
    if not score_files:
        raise SystemExit("No scoreboard files found. Run: python src/fppull/public_pull.py")

    for sf in score_files:
        # infer week from filename
        wk = int(sf.name.split("_w")[1][:2])
        print(f"\n🧾 Week {wk}: reading {sf.name}")
        score = load_json(sf)
        event_ids = extract_event_ids(score)
        print(f"   → {len(event_ids)} events found")

        for i, eid in enumerate(event_ids, 1):
            print(f"   [{i}/{len(event_ids)}] Fetching summary for event {eid} …")
            data = fetch_summary_public(eid)
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            out_path = out_dir / f"summary_y{season}_w{wk:02d}_event{eid}_{ts}.json"
            save_json(data, out_path)
            time.sleep(0.2)  # be polite

    print("\n✅ Public summaries fetched for all listed weeks.")

if __name__ == "__main__":
    main()