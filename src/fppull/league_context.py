# src/fppull/league_context.py
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import requests
import pandas as pd
from dotenv import load_dotenv

# ---------------------------
# Helpers
# ---------------------------

def _auth_headers() -> Dict[str, str]:
    """
    Build headers with cookies + realistic UA/Referer + ESPN fantasy headers.
    Requires SWID (with braces) and espn_s2 in .env for private league data.
    """
    s2 = os.getenv("ESPN_S2", "").strip()
    swid = os.getenv("SWID", "").strip()

    cookie_parts = []
    if swid:
        cookie_parts.append(f"SWID={swid}")  # keep braces
    if s2:
        cookie_parts.append(f"espn_s2={s2}")
    cookie_header = "; ".join(cookie_parts)

    headers = {
        # Browser-y headers
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/127.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": "https://fantasy.espn.com",
        "Referer": f"https://fantasy.espn.com/football/league?leagueId={os.getenv('LEAGUE_ID','')}",

        # ESPN fantasy internals (these matter)
        "x-fantasy-source": "kona",
        "x-fantasy-platform": "kona-PROD",
    }
    if cookie_header:
        headers["Cookie"] = cookie_header
    return headers


def _league_api_base() -> str:
    """Return the ESPN Fantasy READS API base for this league."""
    season = os.getenv("SEASON", "").strip()
    league_id = os.getenv("LEAGUE_ID", "").strip()
    if not season or not league_id:
        raise SystemExit("Set SEASON and LEAGUE_ID in .env")
    # reads endpoint is more permissive/stable than the UI API
    return f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/segments/0/leagues/{league_id}"

def _get_json(url: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    GET JSON from ESPN Fantasy API. Detect redirects and non-JSON responses
    to surface cookie / header problems clearly.
    """
    # First request without following redirects so we can detect them
    r = requests.get(url, params=params or {}, headers=_auth_headers(), timeout=30, allow_redirects=False)
    debug_url = r.url
    status = r.status_code
    ctype = r.headers.get("Content-Type", "")

    # Handle redirects explicitly (usually cookie/auth problems)
    if status in (301, 302, 303, 307, 308):
        loc = r.headers.get("Location", "")
        print("❌ ESPN redirected the API request.")
        print(f"   From: {debug_url}")
        print(f"   To  : {loc}")
        print("   This typically means SWID/espn_s2 are missing/expired or headers were rejected.")
        raise requests.HTTPError(f"Redirected ({status}) to {loc}")

    # Normal path: must be JSON
    if status == 403:
        print("❌ 403 Forbidden from ESPN API.")
        print("   • Re-check .env cookies: SWID (with braces) and espn_s2; not expired.")
        print(f"   • URL: {debug_url}")
        r.raise_for_status()

    if "application/json" not in ctype:
        preview = (r.text or "")[:300].replace("\n", " ")
        print("❌ Response was not JSON")
        print(f"   URL      : {debug_url}")
        print(f"   Status   : {status}")
        print(f"   Type     : {ctype}")
        print(f"   Preview  : {preview} …")
        print("   Likely causes: expired cookies, wrong SWID braces, or missing fantasy headers.")
        r.raise_for_status()

    r.raise_for_status()
    return r.json()


# ---------------------------
# Extractors
# ---------------------------

def _extract_team_rows(league_json: Dict[str, Any], season: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for t in league_json.get("teams", []):
        team_id = t.get("id")
        abbrev = t.get("abbrev")  # e.g. 'LAMBO'
        loc = t.get("location", "") or ""
        nick = t.get("nickname", "") or ""
        name = (f"{loc} {nick}").strip() or abbrev or f"Team {team_id}"
        rows.append(
            dict(
                season=season,
                team_id=team_id,
                team_abbr=abbrev,
                team_name=name,
            )
        )
    return rows


def _extract_roster_rows(league_json: Dict[str, Any], week: int, season: int, team_map: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for t in league_json.get("teams", []):
        team_id = t.get("id")
        entries = (t.get("roster") or {}).get("entries", []) or []
        for e in entries:
            player = (e.get("playerPoolEntry") or {}).get("player", {}) or {}
            player_id = player.get("id")
            player_name = player.get("fullName") or (
                (player.get("firstName", "") + " " + player.get("lastName", "")).strip()
            )
            lineup_slot = e.get("lineupSlotId")
            status = (e.get("status") or "")

            tm = team_map.get(int(team_id), {})
            rows.append(
                dict(
                    season=season,
                    week=int(week),
                    fantasy_team_id=team_id,
                    fantasy_team_name=tm.get("team_name", f"Team {team_id}"),
                    athlete_id=player_id,             # <-- standardized name
                    athlete_name=player_name,         # <-- standardized name
                    lineup_slot=float(lineup_slot) if lineup_slot is not None else None,
                    status=status,
                )
            )
    return rows

def _detect_current_week(base_url: str) -> int:
    """
    Ask ESPN for the league settings to detect the 'current' week.
    Prefers mSettings.status.currentMatchupPeriod, with fallbacks.
    """
    try:
        league = _get_json(base_url, params={"view": "mSettings"})
        status = league.get("status", {}) if isinstance(league, dict) else {}
        # Common fields seen in FFL APIs:
        candidates = [
            status.get("currentMatchupPeriod"),
            status.get("latestScoringPeriod"),
            league.get("scoringPeriodId"),
        ]
        for v in candidates:
            if isinstance(v, int) and v >= 1:
                return v
            # sometimes they are strings
            try:
                iv = int(v)
                if iv >= 1:
                    return iv
            except Exception:
                pass
    except Exception:
        pass
    # Final fallback (rare): try env override or default to 1
    try:
        return max(1, int(os.getenv("CURRENT_WEEK", "1")))
    except Exception:
        return 1


def _resolve_weeks(weeks_raw: str, base_url: str) -> list[int]:
    """
    Turn WEEKS env into a concrete list of ints.
    - If blank or 'ALL' (case-insensitive): use weeks 1..current_week (auto-detected).
    - Else: parse comma-separated list.
    """
    weeks_raw = (weeks_raw or "").strip()
    if not weeks_raw or weeks_raw.upper() == "ALL":
        current_week = _detect_current_week(base_url)
        if current_week < 1:
            raise SystemExit("Could not auto-detect CURRENT_WEEK from ESPN.")
        return list(range(1, current_week + 1))

    try:
        weeks = [int(w) for w in weeks_raw.split(",") if w.strip()]
    except ValueError:
        raise SystemExit("WEEKS must be 'ALL' (or blank) or a comma list of integers (e.g., '1,2,3').")
    if not weeks:
        raise SystemExit("WEEKS list parsed empty — check your .env value.")
    return weeks


# ---------------------------
# Main
# ---------------------------

def main():
    # 1) Load env first
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")

    season_s = os.getenv("SEASON")
    league_id = os.getenv("LEAGUE_ID")

    if not season_s or not league_id:
        print("Config error: ensure SEASON and LEAGUE_ID are set in .env", file=sys.stderr)
        sys.exit(1)

    season = int(season_s)
    out_dir = root / "data" / "processed" / f"season_{season}" / "espn"
    out_dir.mkdir(parents=True, exist_ok=True)

    # API base (reads)
    base = _league_api_base()
    print(f"API base: {base}")

    # Resolve weeks (auto-detect when WEEKS is blank or 'ALL')
    weeks = _resolve_weeks(os.getenv("WEEKS", ""), base)
    print(f"Weeks resolved from env/auto: {weeks}")

    # Teams (mTeam)
    league = _get_json(base, params={"view": "mTeam"})
    team_rows = _extract_team_rows(league, season=season)
    teams_df = pd.DataFrame(team_rows).sort_values("team_id").reset_index(drop=True)
    teams_df.to_csv(out_dir / "teams.csv", index=False)
    print(f"💾 wrote {out_dir/'teams.csv'} ({len(teams_df)} teams)")

    team_map = {
        int(r["team_id"]): {"team_name": r["team_name"], "team_abbr": r["team_abbr"]}
        for _, r in teams_df.iterrows()
    }

    # Weekly rosters (mRoster)
    all_rows: List[Dict[str, Any]] = []
    for wk in weeks:
        print(f"🌐 Fetching ESPN weekly rosters for week {wk} …")
        league_w = _get_json(base, params={"scoringPeriodId": wk, "view": "mRoster"})
        week_rows = _extract_roster_rows(league_w, week=wk, season=season, team_map=team_map)
        print(f"   → {len(week_rows)} roster rows")
        all_rows.extend(week_rows)

    roster_df = (
        pd.DataFrame(all_rows)
        .sort_values(["week", "fantasy_team_id", "lineup_slot", "athlete_name"])
        .reset_index(drop=True)
    )
    roster_df.to_csv(out_dir / "roster_week.csv", index=False)
    print(f"💾 wrote {out_dir/'roster_week.csv'} ({len(roster_df)} rows)")
    print("✅ ESPN league context fetch complete (teams + weekly rosters).")