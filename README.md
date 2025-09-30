# fantasy-public-pull

Goal: Use ESPN **private** league endpoints only to discover league/team/roster, and use **public** endpoints to fetch week-by-week player stats for *all players* (regardless of roster), then join them when building reports.

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data/raw data/processed
# place raw pulls in data/raw ; clean outputs go to data/processed
```
