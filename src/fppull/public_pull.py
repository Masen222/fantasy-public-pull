import os, sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"

def main():
    load_dotenv(ROOT / ".env")

    season = os.getenv("SEASON")
    league_id = os.getenv("LEAGUE_ID")
    weeks = os.getenv("WEEKS", "").split(",")
    weeks = [w.strip() for w in weeks if w.strip()]

    if not season or not league_id or not weeks:
        print("Config error: ensure SEASON, LEAGUE_ID, and WEEKS are set in .env", file=sys.stderr)
        sys.exit(1)

    # Ensure folders exist
    (DATA_RAW / f"season_{season}" / "public").mkdir(parents=True, exist_ok=True)

    print("✅ Config loaded")
    print(f"  SEASON   : {season}")
    print(f"  LEAGUE_ID: {league_id}")
    print(f"  WEEKS    : {weeks}")
    print(f"  RAW PATH : {(DATA_RAW / f'season_{season}' / 'public').as_posix()}")

    # No network yet — just prove the plumbing works.
    # Next step we will add the first real public GET call.

if __name__ == "__main__":
    main()
