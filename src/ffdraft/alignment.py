"""True snap alignment (X / Z / Slot / Y) for the 2023-2025 rookie classes.

Source: NFL Big Data Bowl 2027 tracking, which covers rookies only. Nothing comparable
is open for veterans, so this is a *partial* layer: it answers "where does this rookie
line up and what does he do against man vs zone from there", and is silent otherwise.

The Big Data Bowl data is licensed CC BY-NC 4.0 and its competition rules govern use, so
per-player output is NOT shipped with this repo. `scripts/build_alignment.py` builds it
locally from your own copy of the competition files into `$FFDRAFT_DATA`. Without that
file every function here returns an empty result and the rest of the server is unaffected.

Roles (per pass play, non-motion):
  Slot = lined up SLOT_WR     Y = tight end
  X    = outside WR on the line of scrimmage
  Z    = outside WR off the line (>= 2.0 yd behind the spotted line; offensive linemen,
         who are always on the line, sit at a median of ~1.1 yd and 98% under 2.0)
Rates are per *route run*, so they don't depend on who else is on the field.
"""
from __future__ import annotations

import json

from .config import DATA_DIR

FILE = DATA_DIR / "alignment_players.json"
MIN_ROUTES_SHOWN = 50
SORTS = {"yards_per_route": "yprr", "epa_per_target": "epa_t", "tprr": "tprr",
         "catch": "catch", "yards_per_target": "ypt"}

_STATE: dict = {"mtime": None, "data": None}


def load() -> dict:
    """{gsis_id: {name, draft_year, position, team, total_routes, roles: {role: {ALL|MAN|ZONE: {...}}}}}"""
    if not FILE.exists():
        return {}
    m = FILE.stat().st_mtime
    if _STATE["mtime"] != m:
        _STATE["data"] = json.loads(FILE.read_text())
        _STATE["mtime"] = m
    return _STATE["data"]


def available() -> bool:
    return bool(load())


def profile(player_id: str) -> dict | None:
    """Role mix plus man/zone rates per role for one rookie-class player, or None."""
    p = load().get(player_id)
    if not p:
        return None
    roles = {}
    for r, d in p["roles"].items():
        if d["ALL"]["routes"] < MIN_ROUTES_SHOWN:
            continue
        roles[r] = {"route_share_pct": d["route_share_pct"], "overall": d["ALL"],
                    "vs_man": d.get("MAN"), "vs_zone": d.get("ZONE")}
    if not roles:
        return None
    return {"primary_role": p["primary_role"], "total_routes": p["total_routes"], "roles": roles,
            "note": "Big Data Bowl snap tracking, 2023-2025 rookie classes only. Roles with <50 routes omitted; "
                    "man/zone splits under ~100 routes are noisy."}


def leaders(role: str, coverage: str = "MAN", sort_by: str = "yards_per_route",
            min_routes: int = 100) -> list[dict]:
    key = SORTS[sort_by]
    rows = []
    for pid, p in load().items():
        c = p["roles"].get(role, {}).get(coverage)
        if c and c["routes"] >= min_routes:
            rows.append({"player_id": pid, "name": p["name"], "team": p["team"],
                         "draft_year": p["draft_year"], **c})
    return sorted(rows, key=lambda r: r[key], reverse=True)
