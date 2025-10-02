# src/fppull/fetch_espn_team_scores.py
import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

# -----------------------------------------------------------------------------
# ESPN READS API HELPERS
# -----------------------------------------------------------------------------
READS_BASE_TMPL = (
    "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/"
    "seasons/{season}/segments/0/leagues/{league_id}"
)

def _league_api_base() -> str:
    """Base URL for ESPN Fantasy 'reads' API."""
    season = os.getenv("SEASON", "").strip()
    league_id = os.getenv("LEAGUE_ID", "").strip()
    if not season or not league_id:
        raise SystemExit("Set SEASON and LEAGUE_ID in .env")
    return READS_BASE_TMPL.format(season=season, league_id=league_id)

def _cookie_from_file(path: str) -> Optional[str]:
    """Read a one-line cookie file that contains 'SWID=...; espn_s2=...'."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        txt = p.read_text(encoding="utf-8").strip()
        if "SWID=" in txt or "espn_s2=" in txt:
            return txt
    except Exception:
        pass
    return None

def _auth_headers() -> Dict[str, str]:
    """
    Build headers/cookies so ESPN accepts requests.
    Priority:
      1) COOKIE_FILE (if provided and readable)
      2) SWID + espn_s2 from .env
    """
    cookie_header = None

    cookie_file = os.getenv("COOKIE_FILE", "").strip()
    if cookie_file:
        cookie_header = _cookie_from_file(cookie_file)

    if not cookie_header:
        swid = os.getenv("SWID", "").strip()
        s2 = os.getenv("ESPN_S2", "").strip()
        parts = []
        if swid:
            parts.append(f"SWID={swid}")   # include braces
        if s2:
            parts.append(f"espn_s2={s2}")
        cookie_header = "; ".join(parts) if parts else None

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/127.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Referer": f"https://fantasy.espn.com/football/league?leagueId={os.getenv('LEAGUE_ID','')}",
    }
    if cookie_header:
        headers["Cookie"] = cookie_header
    return headers

def _get_json(url: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    GET JSON from ESPN 'reads' API. Fail early if we get HTML or a redirect.
    """
    r = requests.get(
        url,
        params=params or {},
        headers=_auth_headers(),
        timeout=30,
        allow_redirects=False,
    )
    status = r.status_code
    ctype = r.headers.get("Content-Type", "")

    if status in (301, 302, 303, 307, 308):
        loc = r.headers.get("Location", "")
        raise requests.HTTPError(f"Redirected ({status}) to {loc}")
    if status == 403:
        raise requests.HTTPError("403 Forbidden — check SWID/espn_s2 cookies (not expired)")
    if "application/json" not in ctype:
        preview = (r.text or "")[:300].replace("\n", " ")
        raise requests.HTTPError(f"Non-JSON response ({ctype}). Preview: {preview}")

    r.raise_for_status()
    return r.json()

def _detect_current_week(base_url: str) -> int:
    """
    Ask ESPN for the league settings to detect the 'current' week.
    Prefers mSettings.status.currentMatchupPeriod, with fallbacks.
    """
    try:
        league = _get_json(base_url, params={"view": "mSettings"})
        status = league.get("status", {}) if isinstance(league, dict) else {}
        for v in (
            status.get("currentMatchupPeriod"),
            status.get("latestScoringPeriod"),
            league.get("scoringPeriodId"),
        ):
            if isinstance(v, int) and v >= 1:
                return v
            try:
                iv = int(v)
                if iv >= 1:
                    return iv
            except Exception:
                pass
    except Exception:
        pass
    # Final fallback
    try:
        return max(1, int(os.getenv("CURRENT_WEEK", "1")))
    except Exception:
        return 1

def _resolve_weeks(weeks_raw: str, base_url: str) -> List[int]:
    """
    Turn WEEKS env into a concrete list of ints.
    - If blank or 'ALL': use weeks 1..current_week (auto-detected).
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

# -----------------------------------------------------------------------------
# Extraction helpers
# -----------------------------------------------------------------------------
def _coerce_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def _extract_points_from_team_obj(team_obj: Dict[str, Any]) -> Optional[float]:
    """
    ESPN has varied where the numeric ‘official’ points live.
    Try a series of common locations and return the first numeric found.
    """
    # Direct fields
    for k in ("totalPoints", "appliedStatTotal", "points", "score"):
        v = _coerce_float(team_obj.get(k))
        if v is not None:
            return v

    # Nested candidates
    nested_candidates = [
        ("cumulativeScore", "score"),
        ("rosterForCurrentScoringPeriod", "appliedStatTotal"),
        ("totalPointsLive", None),
        ("adjustment", "points"),
    ]
    for a, b in nested_candidates:
        inner = team_obj.get(a)
        if isinstance(inner, dict):
            v = inner.get(b) if b else inner
            fv = _coerce_float(v)
            if fv is not None:
                return fv

    # Some payloads store totals per period keyed by IDs; sum if found
    for key in ("appliedStatTotalByScoringPeriod", "pointsByScoringPeriod"):
        by_period = team_obj.get(key)
        if isinstance(by_period, dict):
            try:
                return float(sum(_coerce_float(v) or 0.0 for v in by_period.values()))
            except Exception:
                pass

    return None

def _rows_from_schedule_obj(obj: Dict[str, Any]) -> List[Tuple[int, float]]:
    """
    Return list of (teamId, officialPoints) for this matchup object.
    Handles both legacy 'teams' array and new 'home'/'away' objects.
    """
    rows: List[Tuple[int, float]] = []

    # Legacy shape
    if "teams" in obj and isinstance(obj["teams"], list) and obj["teams"]:
        for t in obj["teams"]:
            tid = t.get("teamId")
            pts = _extract_points_from_team_obj(t)
            if tid is not None and pts is not None:
                rows.append((int(tid), float(pts)))
        return rows

    # Newer shape: 'home' and 'away'
    for side in ("home", "away"):
        side_obj = obj.get(side)
        if isinstance(side_obj, dict):
            tid = side_obj.get("teamId")
            pts = _extract_points_from_team_obj(side_obj)
            if tid is not None and pts is not None:
                rows.append((int(tid), float(pts)))

    return rows

# Add this helper near your other helpers (module level).
def _is_for_week(obj: Dict[str, Any], wk: int) -> bool:
    """
    True if this schedule object corresponds to the requested fantasy week.
    We accept either matchupPeriodId or scoringPeriodId matching wk.
    """
    try:
        if int(obj.get("matchupPeriodId", -1)) == int(wk):
            return True
    except Exception:
        pass
    try:
        if int(obj.get("scoringPeriodId", -1)) == int(wk):
            return True
    except Exception:
        pass
    return False

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    load_dotenv(ROOT / ".env")

    season = os.getenv("SEASON")
    league_id = os.getenv("LEAGUE_ID")
    if not season or not league_id:
        print("Config error: ensure SEASON and LEAGUE_ID are set in .env", file=sys.stderr)
        sys.exit(1)

    base = _league_api_base()
    weeks = _resolve_weeks(os.getenv("WEEKS", ""), base)
    print(f"Weeks resolved from env/auto: {weeks}")

    # Load canonical team names (from league_context)
    teams_csv = PROCESSED / f"season_{season}" / "espn" / "teams.csv"
    team_names: Dict[int, str] = {}
    if teams_csv.exists():
        tdf = pd.read_csv(teams_csv)
        # accept "team_name" if present; else try to reconstruct
        name_col = "team_name" if "team_name" in tdf.columns else None
        if not name_col:
            for c in ("teamLocation", "teamNickname"):
                if c not in tdf.columns:
                    tdf[c] = ""
            tdf["team_name"] = (tdf.get("teamLocation", "") + " " + tdf.get("teamNickname", "")).str.strip()
            name_col = "team_name"
        for _, r in tdf.iterrows():
            try:
                team_names[int(r["team_id"])] = str(r[name_col]) if pd.notna(r[name_col]) else f"Team {int(r['team_id'])}"
            except Exception:
                pass

    out_rows: List[Dict[str, Any]] = []
    raw_dir = RAW / f"season_{season}" / "espn"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for wk in weeks:
        try:
            # First try scoringPeriodId
            data = _get_json(base, params={"view": "mMatchupScore", "scoringPeriodId": wk})
            sched = data.get("schedule", [])
            if not isinstance(sched, list):
                sched = []
            # Filter to this week only
            sched = [m for m in sched if _is_for_week(m, wk)]

            # If empty, try matchupPeriodId and filter again
            if not sched:
                data = _get_json(base, params={"view": "mMatchupScore", "matchupPeriodId": wk})
                sched = data.get("schedule", [])
                if not isinstance(sched, list):
                    sched = []
                sched = [m for m in sched if _is_for_week(m, wk)]

            # Save raw for audit/debug (after final fetch)
            (raw_dir / f"matchupscore_w{wk:02d}.json").write_text(json.dumps(data, indent=2))

            wk_rows: List[Tuple[int, float]] = []
            for m in sched:
                wk_rows.extend(_rows_from_schedule_obj(m))

            if not wk_rows:
                print(f"⚠️  Week {wk}: no matchup rows found — check cookies or view params.")
            else:
                print(f"🌐 Week {wk}: collected {len(wk_rows)} team rows")

            for tid, pts in wk_rows:
                out_rows.append({
                    "season": int(season),
                    "week": int(wk),
                    "fantasy_team_id": tid,
                    "fantasy_team_name": team_names.get(tid, f"Team {tid}"),
                    "official_pts": round(float(pts), 2),
                })

        except Exception as e:
            print(f"❌ Week {wk} fetch error: {e}")

    if not out_rows:
        print("No matchup rows found — double-check cookies and league/week config.")
        return

    out_df = pd.DataFrame(out_rows).sort_values(["week", "fantasy_team_id"]).reset_index(drop=True)
    out_path = PROCESSED / f"season_{season}" / "espn" / "team_week_official.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"✅ Wrote {out_path} with {len(out_df)} rows.")

if __name__ == "__main__":
    main()