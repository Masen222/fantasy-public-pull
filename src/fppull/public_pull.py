import os, sys, json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))
    print(f"💾 wrote {path.relative_to(ROOT)} ({len(json.dumps(obj))} bytes)")

def fetch_scoreboard_public(season: str, week: str) -> Dict[str, Any]:
    """
    Public ESPN scoreboard (no auth).
    Example:
      https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?year=2025&week=1
    """
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    params = {"year": season, "week": week}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def main():
    load_dotenv(ROOT / ".env")

    season = os.getenv("SEASON")
    league_id = os.getenv("LEAGUE_ID")  # not used yet, but we keep it flowing
    weeks = os.getenv("WEEKS", "").split(",")
    weeks = [w.strip() for w in weeks if w.strip()]

    if not season or not league_id or not weeks:
        print("Config error: ensure SEASON, LEAGUE_ID, and WEEKS are set in .env", file=sys.stderr)
        sys.exit(1)

    out_dir = ensure_dir(DATA_RAW / f"season_{season}" / "public")

    print("✅ Config loaded")
    print(f"  SEASON   : {season}")
    print(f"  LEAGUE_ID: {league_id}")
    print(f"  WEEKS    : {weeks}")
    print(f"  RAW PATH : {out_dir.as_posix()}")

    # Fetch just the FIRST week listed, as a smoke test.
    week = weeks[0]
    print(f"\n🌐 Fetching public scoreboard for week {week} …")
    data = fetch_scoreboard_public(season, week)

    # Minimal sanity check
    evts = data.get("events", [])
    print(f"   → got {len(evts)} events")

    # Save
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"scoreboard_y{season}_w{int(week):02d}_{ts}.json"
    save_json(data, out_path)

    print("\n✅ Public fetch smoke-test complete.")

if __name__ == "__main__":
    main()