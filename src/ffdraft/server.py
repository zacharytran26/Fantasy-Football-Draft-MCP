"""MCP server: a live fantasy football draft analyst.

Run with:  python -m ffdraft.server
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # mcp SDK >= 2.0
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp SDK 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from . import adp as adp_mod
from . import board as bd
from . import features, model, sources
from . import kickers as kk_mod
from .config import (
    CURRENT_SEASON,
    DATA_DIR,
    STATE_DIR,
    LeagueSettings,
    ModelWeights,
    Scoring,
    delete_league,
    load_settings,
    save_settings,
    set_active,
)
from .config import list_leagues as cfg_list_leagues

mcp = _Server("fantasy-draft-analyst")

_CACHE: dict[str, Any] = {"league": None, "weights": None, "adp_csv": {}}
# Boards are keyed by the settings that actually change them, so a 10-team full-PPR
# league and a 13-team half-PPR league each keep their own and switching between
# them is instant rather than an eight-second rebuild.
_BOARDS: dict[str, pd.DataFrame] = {}


def _scoring_label(league: LeagueSettings) -> str:
    """ppr / half_ppr / standard. Anything unusual is treated as standard, which is
    the conservative choice: it assumes no reception credit rather than inventing one."""
    r = float(league.scoring.rec)
    if r >= 0.9:
        return "ppr"
    if r >= 0.35:
        return "half_ppr"
    return "standard"


def _board_path(league: LeagueSettings) -> Path:
    return DATA_DIR / f"board_{league.cache_key()}.parquet"


# ---------------------------------------------------------------- internals

def _settings() -> tuple[LeagueSettings, ModelWeights]:
    if _CACHE["league"] is None:
        _CACHE["league"], _CACHE["weights"] = load_settings()
    return _CACHE["league"], _CACHE["weights"]


def _build_board(force: bool = False) -> pd.DataFrame:
    league, weights = _settings()
    key = league.cache_key()
    path = _board_path(league)

    if not force and key in _BOARDS:
        return _BOARDS[key]
    if not force and path.exists():
        b = pd.read_parquet(path)
        _BOARDS[key] = b
        return b

    tbl = model.build_player_table(league, weights)
    proj = model.project(tbl, league, weights)
    try:
        adp = bd.load_adp(
            csv_path=(_CACHE["adp_csv"] or {}).get(league.name),
            superflex=bool(getattr(league, "superflex", 0)),
        )
    except Exception as exc:
        print(f"ADP unavailable ({type(exc).__name__}); using model rank as proxy")
        adp = None
    proj = bd.attach_adp(proj, adp)
    proj = bd.convert_adp_format(proj, _scoring_label(league))
    proj.to_parquet(path, index=False)
    _BOARDS[key] = proj
    return proj


_KICKER_BOARDS: dict[str, pd.DataFrame] = {}


def _build_kicker_board(force: bool = False) -> pd.DataFrame:
    """Kickers get their own small cache, keyed the same way the skill board is --
    not run through _build_board/model.project at all, see kickers.py's docstring
    for why. No ADP/draft-state involved, so this is much cheaper than the skill
    board and doesn't need a parquet round trip.
    """
    league, _ = _settings()
    key = league.cache_key()
    if not force and key in _KICKER_BOARDS:
        return _KICKER_BOARDS[key]
    b = kk_mod.kicker_board(league)
    _KICKER_BOARDS[key] = b
    return b


def _state() -> bd.DraftState:
    league, _ = _settings()
    return bd.DraftState(league)


def _mark_drafted(b: pd.DataFrame, state: bd.DraftState) -> pd.DataFrame:
    b = b.copy()
    b["drafted"] = b["_key"].isin(state.taken_keys())
    return b


def _rows(df: pd.DataFrame, cols: list[str], n: int) -> list[dict]:
    out = []
    for _, r in df.head(n).iterrows():
        d = {}
        for c in cols:
            v = r.get(c)
            if isinstance(v, (np.floating, float)):
                v = None if not np.isfinite(v) else round(float(v), 3)
            elif isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.bool_,)):
                v = bool(v)
            d[c] = v
        out.append(d)
    return out


# ---------------------------------------------------------------- tools

@mcp.tool()
def configure_league(name: str = "default", teams: int = 12, draft_slot: int = 6,
                     rounds: int = 16, scoring: str = "half_ppr", snake: bool = True,
                     qb: int = 1, rb: int = 2, wr: int = 2, te: int = 1, flex: int = 1,
                     superflex: int = 0, te_premium_bonus: float = 0.0,
                     consistency_weight: float = 0.35,
                     adp_csv_path: str | None = None) -> str:
    """Create or update a league, and make it the active one.

    Give each league a name and you can keep as many as you like side by side —
    a 10-team full PPR and a 13-team half PPR hold separate boards, separate
    replacement levels and separate in-progress drafts.

    scoring: ppr, half_ppr, or standard. Use superflex=1 for a second QB-eligible
    slot, and te_premium_bonus for extra points per tight end reception.
    consistency_weight trades expected points against week-to-week reliability
    (0 = pure upside, 1 = pure floor).
    """
    if not 1 <= draft_slot <= teams:
        return json.dumps({"error": f"draft_slot {draft_slot} is outside a {teams}-team league"})

    starters = {"QB": qb, "RB": rb, "WR": wr, "TE": te, "FLEX": flex, "K": 1, "DST": 1}
    league = LeagueSettings(
        name=name, teams=teams, rounds=rounds, draft_slot=draft_slot, snake=snake,
        scoring=Scoring.preset(scoring), starters=starters,
        superflex=superflex, te_premium_bonus=te_premium_bonus,
    )
    weights = ModelWeights(consistency_weight=consistency_weight)
    save_settings(league, weights)

    csvs = dict(_CACHE.get("adp_csv") or {})
    if adp_csv_path:
        csvs[name] = adp_csv_path
    _CACHE.update({"league": league, "weights": weights, "adp_csv": csvs})

    known, _ = cfg_list_leagues()
    reused = _board_path(league).exists()
    return json.dumps({
        "league": name, "active": True, "teams": teams, "your_slot": draft_slot,
        "scoring": scoring, "superflex": superflex,
        "your_picks": league.picks_for_slot()[:rounds],
        "replacement_levels": league.replacement_ranks(),
        "all_leagues": known,
        "board": "already cached for these settings" if reused
                 else "will build on your next query",
    }, indent=2)


@mcp.tool()
def list_leagues() -> str:
    """Every league you've set up, and which one is active."""
    known, active = cfg_list_leagues()
    out = []
    for nm in known:
        lg, _ = load_settings(nm)
        state = bd.DraftState(lg)
        out.append({
            "name": nm, "active": nm == active, "teams": lg.teams,
            "scoring": ("ppr" if lg.scoring.rec >= 1 else
                        "standard" if lg.scoring.rec == 0 else "half_ppr"),
            "your_slot": lg.draft_slot, "superflex": lg.superflex,
            "picks_recorded": len(state.picks),
        })
    return json.dumps({"active": active, "leagues": out}, indent=2)


@mcp.tool()
def switch_league(name: str) -> str:
    """Make a different league active. Its board and draft resume where you left them."""
    if not set_active(name):
        known, _ = cfg_list_leagues()
        return json.dumps({"error": f"no league named '{name}'", "available": known})
    league, weights = load_settings(name)
    _CACHE.update({"league": league, "weights": weights})
    state = bd.DraftState(league)
    return json.dumps({
        "active": name, "teams": league.teams, "your_slot": league.draft_slot,
        "scoring": ("ppr" if league.scoring.rec >= 1 else
                    "standard" if league.scoring.rec == 0 else "half_ppr"),
        "board": "cached" if _board_path(league).exists() else "will build on next query",
        **state.summary(),
    }, indent=2)


@mcp.tool()
def remove_league(name: str) -> str:
    """Delete a league and its draft history. The board cache is left alone, since
    other leagues with the same format may share it."""
    if not delete_league(name):
        known, _ = cfg_list_leagues()
        return json.dumps({"error": f"no league named '{name}'", "available": known})
    p = STATE_DIR / f"draft_{re.sub(r'[^A-Za-z0-9_-]', '_', name)}.json"
    if p.exists():
        p.unlink()
    if (_CACHE.get("league") or LeagueSettings()).name == name:
        _CACHE.update({"league": None, "weights": None})
    known, active = cfg_list_leagues()
    return json.dumps({"removed": name, "remaining": known, "active": active}, indent=2)


@mcp.tool()
def refresh_data(force_download: bool = False) -> str:
    """Rebuild the player board from source data. Run once before draft day."""
    if force_download:
        for p in sources.CACHE_DIR.glob("*.parquet"):
            p.unlink()
    sources.clear_memory_cache()
    features.clear_derived_cache()
    _BOARDS.clear()
    _KICKER_BOARDS.clear()
    b = _build_board(force=True)
    kb = _build_kicker_board(force=True)
    league, _ = _settings()
    return json.dumps({
        "players_modelled": len(b),
        "by_position": b["position"].value_counts().to_dict(),
        "kickers_modelled": len(kb),
        "seasons": sorted(int(s) for s in sources.weekly_stats()["season"].unique()),
        "datasets_cached": sources.cache_status(),
        "top_10": _rows(b, ["name", "position", "team", "proj_points", "consistency", "adp"], 10),
    }, indent=2, default=str)


@mcp.tool()
def best_available(position: str | None = None, limit: int = 15,
                   sort_by: str = "draft_score") -> str:
    """The next best players still on the board.

    sort_by: draft_score (balanced), vor (raw value), consistency (floor),
    proj_points, or value (biggest gap between ADP and model rank).

    position="K" returns the kicker ranking instead -- kickers aren't run through
    the same model (see kickers.py), so sort_by there is fixed to kicker_ppg
    (recency-weighted fantasy points per game under the league's distance-banded
    scoring) rather than any of the skill-position sort keys above. Not
    draft-state aware like the rest of this tool -- a kicker doesn't disappear
    from the list once drafted, since sync_draft/record_pick don't track K picks.
    """
    if position and position.upper() == "K":
        kb = _build_kicker_board()
        cols = ["name", "team", "kicker_rank", "kicker_ppg", "fg_pct", "fg_made", "fg_att",
                "long_range_share_pct", "fg_long", "context_tier", "pct_fg", "pct_td", "off_epa"]
        return json.dumps({"sorted_by": "kicker_ppg",
                           "kickers": _rows(kb, [c for c in cols if c in kb.columns], limit)}, indent=2)

    b = _mark_drafted(_build_board(), _state())
    avail = b[~b["drafted"]]
    if position:
        avail = avail[avail["position"] == position.upper()]
    key = {"value": "adp_delta"}.get(sort_by, sort_by)
    if key not in avail.columns:
        key = "draft_score"
    avail = avail.sort_values(key, ascending=False)
    cols = ["name", "position", "team", "overall_rank", "pos_rank", "adp", "adp_delta",
            "proj_points", "adj_ppg", "consistency", "startable_rate", "injury_risk", "vor"]
    return json.dumps({"sorted_by": key, "players": _rows(avail, cols, limit)}, indent=2)


@mcp.tool()
def who_should_i_pick(limit: int = 6) -> str:
    """The live draft-analyst call: who to take right now, and why.

    Weighs projected value, week-to-week consistency, your roster's open starting
    slots, and the odds each player survives to your next pick.
    """
    league, _ = _settings()
    state = _state()
    b = _mark_drafted(_build_board(), state)
    nxt = state.next_pick_for_me()
    on_clock = state.on_the_clock
    if nxt is not None and nxt > on_clock:
        current = nxt  # you're not up yet; evaluate for your actual next pick
    else:
        current = on_clock
    after = state.pick_after_next() if nxt == current else nxt

    roster = state.my_roster(b)
    recs = model.recommend(b, league, current_pick=current, next_pick=after,
                           roster=roster, top_n=limit)

    picks = []
    for _, r in recs.iterrows():
        picks.append({
            "player": r["name"], "position": r["position"], "team": r.get("team"),
            "adp": round(float(r["adp"]), 1),
            "proj_points": round(float(r["proj_points"]), 1),
            "consistency": round(float(r["consistency"]), 3),
            "survives_to_next_pick": round(float(r["p_available_next"]), 2),
            "why": model.explain(r),
        })
    return json.dumps({
        "evaluating_pick": current,
        "round": (current - 1) // league.teams + 1,
        "your_next_pick_after_this": after,
        "picks_you_wait": (after - current) if after else None,
        "your_roster": roster,
        "recommendations": picks,
        "headline": (f"Take {picks[0]['player']} — {picks[0]['why']}" if picks else "Board empty"),
    }, indent=2)


@mcp.tool()
def record_pick(player_name: str, overall_pick: int | None = None,
                team_slot: int | None = None) -> str:
    """Log a pick that just happened. Use after every pick if you aren't auto-syncing."""
    state = _state()
    b = _build_board()
    row = bd.match_player(player_name, b)
    resolved = row["name"] if row is not None else player_name
    pick = state.record(resolved, overall_pick, team_slot)
    return json.dumps({
        "recorded": pick,
        "matched_to": resolved if row is not None else "no model match (logged as typed)",
        "position": (row["position"] if row is not None else None),
        **state.summary(),
    }, indent=2)


@mcp.tool()
def sync_draft(platform: str, league_id: str | None = None, draft_id: str | None = None,
               pasted_board: str | None = None, season: int = CURRENT_SEASON) -> str:
    """Pull the current draft board from your platform.

    platform="sleeper" with draft_id -- fully automatic, public API.
    platform="espn" with league_id -- works for public leagues; private ones need
      ESPN_SWID and ESPN_S2 environment variables from a logged-in browser session.
    platform="paste" with pasted_board -- paste the drafted list from any site.
    """
    state = _state()
    b = _build_board()
    platform = platform.lower()

    if platform == "sleeper":
        if not draft_id:
            return json.dumps({"error": "draft_id required for Sleeper"})
        picks = bd.sync_sleeper(draft_id)
    elif platform == "espn":
        if not league_id:
            return json.dumps({"error": "league_id required for ESPN"})
        picks = bd.sync_espn(league_id, season)
    elif platform == "paste":
        if not pasted_board:
            return json.dumps({"error": "pasted_board text required"})
        names = bd.parse_pasted_board(pasted_board)
        picks = [{"overall": i + 1, "slot": None, "name": n} for i, n in enumerate(names)]
    else:
        return json.dumps({"error": f"unknown platform '{platform}'"})

    state.reset()
    unmatched = []
    for p in picks:
        row = bd.match_player(p["name"], b)
        if row is None:
            unmatched.append(p["name"])
        state.record(row["name"] if row is not None else p["name"],
                     p.get("overall"), p.get("slot"))
    return json.dumps({
        "platform": platform, "picks_synced": len(picks),
        "unmatched_names": unmatched[:20],
        **state.summary(),
    }, indent=2)


@mcp.tool()
def draft_status() -> str:
    """Where the draft stands and what your roster looks like."""
    state = _state()
    b = _mark_drafted(_build_board(), state)
    mine = [p for p in state.picks if p["slot"] == state.my_slot]
    idx = b.set_index("_key")
    detail = []
    for p in mine:
        k = bd.norm_name(p["name"])
        r = idx.loc[k] if k in idx.index else None
        detail.append({
            "pick": p["overall"], "player": p["name"],
            "position": (r["position"] if r is not None else None),
            "proj_points": (round(float(r["proj_points"]), 1) if r is not None else None),
        })
    return json.dumps({**state.summary(), "my_team": detail,
                       "roster_counts": state.my_roster(b)}, indent=2)


@mcp.tool()
def undo_pick() -> str:
    """Remove the most recent pick — for when someone mis-enters the board."""
    state = _state()
    removed = state.undo()
    return json.dumps({"removed": removed, **state.summary()}, indent=2)


@mcp.tool()
def reset_draft() -> str:
    """Clear all recorded picks and start fresh."""
    state = _state()
    state.reset()
    return json.dumps({"reset": True, **state.summary()}, indent=2)


@mcp.tool()
def separation_report(position: str = "WR", player_name: str | None = None,
                      limit: int = 20) -> str:
    """Separation and route efficiency, plus the season-long matchup each player draws.

    avg_separation is NFL Next Gen Stats tracking data: yards of daylight between
    receiver and nearest defender when the ball arrives. YPRR and TPRR use routes
    estimated from snap share times team dropbacks. Only players who cleared 250
    routes and 50 targets in a season are included, so these are real workloads
    rather than flattering part-time rates.

    `matchup_z` is the receiver's own team's schedule difficulty for the upcoming
    season, from the same opponent-defense data that drives the model's schedule
    adjustment: positive means an easier slate (opponents allow more fantasy points
    to the position), negative means a tougher one. This is the open-data stand-in
    for a WR/CB matchup chart -- team-level and season-long rather than man-coverage
    and week-to-week, since which specific corner covers which receiver on a given
    snap isn't in any open dataset (that needs per-play charting only commercial
    providers do).

    matchup_z is informational only -- players are ranked by sep_score (talent), not
    by a blended score. A backtest (see matchup_backtest) found that folding schedule
    difficulty into a combined ranking made it a worse predictor of actual finish
    than talent alone for WR, so it isn't blended into the sort here.

    Man-versus-zone splits are not reproducible from open data — that needs
    per-play coverage charting.
    """
    from . import separation as sep_mod

    prof = sep_mod.separation_profile()
    prof = prof[prof["qualified"]]
    if prof.empty:
        return json.dumps({"error": "no qualified players"})

    league, _ = _settings()
    dfn = features.defense_ratings(sc=league.scoring)
    sos = features.strength_of_schedule(CURRENT_SEASON, dfn)
    pos = position.upper()
    sos_col = f"sos_{pos}_z"
    sos_cols = ["team"] + ([sos_col] if sos_col in sos.columns else [])

    if player_name:
        row = bd.match_player(player_name, _build_board())
        target = bd.norm_name(row["name"]) if row is not None else bd.norm_name(player_name)
        hist = prof[prof["_key"] == target].sort_values("season")
        if sos_col in sos.columns:
            hist = hist.merge(sos[sos_cols], on="team", how="left") \
                       .rename(columns={sos_col: "matchup_z"})
        cols = ["season", "team", "avg_separation", "avg_cushion", "yprr", "tprr",
                "rec_targets", "rec_yards", "routes_est", "sep_score"]
        if "matchup_z" in hist.columns:
            cols.append("matchup_z")
        return json.dumps({
            "player": player_name,
            "by_season": _rows(hist, cols, 6),
        }, indent=2, default=str)

    recent = int(prof["season"].max())
    cur = prof[(prof["season"] == recent) & (prof["position"] == pos)].copy()
    cols = ["name", "team", "avg_separation", "avg_cushion", "yprr",
            "tprr", "rec_targets", "routes_est", "sep_score"]
    if sos_col in sos.columns and not cur.empty:
        cur = cur.merge(sos[sos_cols], on="team", how="left").rename(columns={sos_col: "matchup_z"})
        cols.append("matchup_z")
    cur = cur.sort_values("sep_score", ascending=False)
    return json.dumps({
        "season": recent, "position": pos, "schedule_season": CURRENT_SEASON,
        "note": "sep_score is a within-season z-score blending separation, YPRR, TPRR "
                "and YAC over expected -- players are ranked by this. matchup_z is "
                "informational only (positive = easier upcoming schedule for the "
                "position, negative = tougher): a backtest (matchup_backtest) found "
                "blending it into the ranking made predictions worse for WR, not "
                "better, so it's shown for reference but not part of the sort.",
        "players": _rows(cur, cols, limit),
    }, indent=2, default=str)


@mcp.tool()
def value_picks(limit: int = 20, direction: str = "undervalued") -> str:
    """Where the model disagrees with the draft market, on draftable players only.

    Positive gap means the model ranks a player higher than the room does — the
    players you can wait on and still get. Negative means the market is paying more
    than the model thinks they're worth.
    """
    league, _ = _settings()
    b = _mark_drafted(_build_board(), _state())
    avail = b[~b["drafted"]].copy()
    # Only players the market actually ranks. A synthetic fallback ADP means nobody
    # is drafting him, so calling him "undervalued" is meaningless — that is how a
    # fringe receiver kept surfacing next to real draft picks.
    if "adp_source" in avail.columns:
        consensus = avail["adp_source"].astype(str).str.startswith("consensus") | \
                    avail["adp_source"].astype(str).str.contains("ecr|csv", case=False)
        if consensus.any():
            avail = avail[consensus]
    avail = avail[avail["adp"] <= 220]
    avail["market_gap"] = avail["adp"] - avail["overall_rank"]
    asc = direction.lower().startswith("over")
    out = avail.sort_values("market_gap", ascending=asc)
    cols = ["name", "position", "team", "adp", "overall_rank", "pos_rank", "market_gap",
            "proj_points", "consistency", "injury_risk", "sep_score"]
    return json.dumps({
        "direction": direction,
        "adp_source": str(avail["adp_source"].mode().iloc[0]) if "adp_source" in avail else "n/a",
        "note": "market_gap > 0 means the model likes him more than his draft cost",
        "players": _rows(out, cols, limit),
    }, indent=2, default=str)


@mcp.tool()
def on_the_clock(platform: str, league_id: str | None = None, draft_id: str | None = None,
                 pasted_board: str | None = None, season: int = CURRENT_SEASON,
                 limit: int = 6) -> str:
    """The full on-the-clock workflow in one call: sync, status, pick, value, matchup.

    Runs, in order:
    1. sync_draft — a fresh pull from your platform, no cached state.
    2. draft_status — round, on-the-clock, and your roster, confirmed against the sync.
    3. who_should_i_pick — the recommendation, reasoning, and survival odds.
    4. value_picks — market-value context, scoped to this round and next.
    5. separation_report — only when the top recommendation is a WR or TE, that
       player's route efficiency and schedule context.

    Use this instead of calling each tool separately when you're on the clock and
    want the full picture in one shot. platform/league_id/draft_id/pasted_board/season
    are exactly sync_draft's arguments.
    """
    sync = json.loads(sync_draft(platform, league_id, draft_id, pasted_board, season))
    if "error" in sync:
        return json.dumps({"step": "sync_draft", **sync}, indent=2)

    status = json.loads(draft_status())
    rec = json.loads(who_should_i_pick(limit=limit))

    league, _ = _settings()
    rnd = rec.get("round", 1)
    lo, hi = (rnd - 1) * league.teams + 1, (rnd + 1) * league.teams
    pool = json.loads(value_picks(limit=100, direction="undervalued"))
    in_window = [p for p in pool.get("players", [])
                if p.get("adp") is not None and lo <= p["adp"] <= hi]

    result = {
        "sync": sync,
        "draft_status": status,
        "recommendation": rec,
        "value_picks_this_round": {
            "round_window": f"picks {lo}-{hi}",
            "direction": pool.get("direction"),
            "players": in_window[:8],
        },
    }

    picks = rec.get("recommendations") or []
    if picks and picks[0].get("position") in ("WR", "TE"):
        result["separation_report"] = json.loads(
            separation_report(position=picks[0]["position"], player_name=picks[0]["player"]))

    return json.dumps(result, indent=2)


@mcp.tool()
def draft_value_history(seasons: str = "2021,2022,2023,2024", group_by: str = "draft_round") -> str:
    """Backtest: how preseason consensus rank compared to where players actually finished.

    Value is measured in points against what that draft slot actually returned, so
    "did RB5 capital buy RB5 production?" Rank movement would be unfair to early
    picks and would label whole rounds as busts, because undrafted breakouts push
    every drafted player down the final standings.
    """
    league, _ = _settings()
    yrs = [int(s) for s in seasons.split(",") if s.strip()]
    hist = adp_mod.value_history(yrs, league.scoring)
    if hist.empty:
        return json.dumps({"error": "no ECR history available"})
    rates = adp_mod.hit_rates(hist, group_by)
    return json.dumps({
        "seasons": yrs,
        "players_analysed": int((hist["ecr"] <= adp_mod.DRAFTABLE_ECR_CUTOFF).sum()),
        "definitions": {"hit": "scored >=115% of the points that draft slot returned",
                        "bust": "scored <=70%"},
        "by_" + group_by: _rows(rates, list(rates.columns), 30),
    }, indent=2, default=str)


@mcp.tool()
def matchup_backtest(seasons: str = "2021,2022,2023,2024", position: str = "WR",
                     top_n: int = 24) -> str:
    """Backtest: does talent + schedule difficulty predict finish better than talent alone?

    This checks, against real seasons, whether blending schedule difficulty into a
    receiver's talent score improves how well it predicts actual finish. For each
    season, talent (`talent_z`) is that player's separation score from the *prior*
    season only, and matchup difficulty (`matchup_z`) is the same leakage-free
    strength_of_schedule the live recommender uses, both compared to actual fantasy
    points scored.

    A positive `improvement_corr` / `improvement_precision` means the schedule
    adjustment earns its keep. Near zero or negative means talent alone predicts
    just as well or better -- which is what a 2021-2024 WR backtest found, so
    separation_report ranks by talent (sep_score) alone and shows matchup_z as
    reference only. Re-run this if the underlying model changes.
    """
    league, _ = _settings()
    yrs = [int(s) for s in seasons.split(",") if s.strip()]
    hist = adp_mod.matchup_value_backtest(yrs, position.upper(), league.scoring)
    if hist.empty:
        return json.dumps({"error": "no matchup backtest data available for those seasons"})
    summary = adp_mod.matchup_backtest_summary(hist, top_n)

    swing = hist.copy()
    swing["swing"] = swing["matchup_z"].abs()
    swing_cols = ["name", "season", "team", "talent_z", "matchup_z",
                 "matchup_adjusted_score", "points", "finish_pos_rank"]
    biggest_swings = swing.sort_values("swing", ascending=False)

    return json.dumps({
        "position": position.upper(),
        "summary": summary,
        "interpretation": (
            "corr is Spearman rank correlation against actual fantasy points; "
            "top_n_precision is, of each metric's predicted top-N players, what "
            "share actually finished top-N that season, averaged across seasons"
        ),
        "biggest_schedule_swings": _rows(biggest_swings, swing_cols, 15),
    }, indent=2, default=str)


@mcp.tool()
def redzone_shift_backtest(seasons: str = "2022,2023,2024,2025", position: str = "WR",
                          top_n: int = 24) -> str:
    """Backtest: does a team's red zone play-calling identity improve on the
    touchdown-luck signal alone at predicting next season's fantasy points?

    Same idea and same discipline as matchup_backtest, applied to the
    `redzone_identity_shift` feature surfaced (informational only) through
    `team_context`. For each season, `talent_z` is the existing touchdown-luck
    signal `m_td_luck` already uses (a player's own prior-season red zone role vs.
    his position's baseline, z-scored), and `matchup_z` here is that player's
    team's red zone identity shift from that same prior season, z-scored across
    teams -- both leakage-free, both compared to real fantasy points scored.

    A positive `improvement_corr`/`improvement_precision` would mean the shift
    adjustment earns its keep and is worth wiring into `m_td_luck`. A 2022-2025
    run found the opposite for both WR (improvement_corr -0.006 across 300
    player-seasons) and TE (-0.053 across 117): red zone identity shift makes the
    prediction *worse*, not better -- the same conclusion matchup_backtest reached
    for schedule difficulty. This is why the shift stays informational-only in
    `team_context` rather than feeding `draft_score`. Re-run this if the underlying
    model or feature changes; only WR/TE are supported, since a pass-rate shift has
    no defensible sign for a running back.
    """
    league, _ = _settings()
    yrs = [int(s) for s in seasons.split(",") if s.strip()]
    hist = adp_mod.redzone_shift_backtest(yrs, position.upper(), league.scoring)
    if hist.empty:
        return json.dumps({"error": "no red zone shift backtest data available for those seasons"})
    summary = adp_mod.matchup_backtest_summary(hist, top_n)

    swing = hist.copy()
    swing["swing"] = swing["matchup_z"].abs()
    swing_cols = ["name", "season", "team", "talent_z", "matchup_z",
                 "matchup_adjusted_score", "points", "finish_pos_rank"]
    biggest_swings = swing.sort_values("swing", ascending=False)

    return json.dumps({
        "position": position.upper(),
        "summary": summary,
        "interpretation": (
            "corr is Spearman rank correlation against actual fantasy points; "
            "top_n_precision is, of each metric's predicted top-N players, what "
            "share actually finished top-N that season, averaged across seasons; "
            "matchup_z here is the team's red zone identity shift, not schedule"
        ),
        "biggest_shift_swings": _rows(biggest_swings, swing_cols, 15),
    }, indent=2, default=str)


@mcp.tool()
def draft_backtest(league_id: str, season: int, top_n: int = 3) -> str:
    """Replay a real past ESPN draft: the algorithm's pick, the true hindsight-best
    pick, and what you actually took, round by round.

    Give it a past season and your ESPN league id (auto-detects your team and
    draft slot from ESPN_SWID/ESPN_S2) and it rebuilds the board leak-free for
    that season -- only data from strictly before it, the same discipline
    matchup_backtest uses -- then replays the real draft in order. At each of
    your picks it reports three things: what who_should_i_pick's algorithm would
    have recommended given the real board at that exact moment, the true
    hindsight-optimal pick by value over replacement (QB capped at 1 -- a second
    quarterback can't start, so it isn't ranked against real RB/WR/TE need), and
    what you actually took. All three are scored on real points from that season.

    Each of the three picks also carries a value verdict (preseason ECR against
    actual finish -- the value_picks steal/bust framing, against real outcomes
    instead of projections) and team context (that player's team's O-line,
    pace, and schedule difficulty for the season being tested, leak-free --
    what team_context reports, but for a past season instead of always today).

    K/DST aren't modelled anywhere in this tool, so those rounds report your
    actual pick only, same as everywhere else. Only ESPN is supported.
    """
    out = adp_mod.draft_backtest(league_id, season, top_n=top_n)
    return json.dumps(out, indent=2, default=str)


@mcp.tool()
def mock_draft(season: int, n_trials: int = 30, top_n: int = 5) -> str:
    """Monte Carlo mock draft: the live algorithm against many simulated
    opponents, averaged, using your active league's settings.

    No real draft needed -- the other teams are bots that pick by that season's
    real preseason ADP with realistic reach/fall noise (bigger swings plausible
    late, tight consensus at the top) rather than following it exactly, so
    who's actually on the board at your turn varies draw to draw. Your slot
    (from your active league's draft_slot) runs the exact same recommend()
    logic who_should_i_pick uses live. The board is leak-free -- only data from
    strictly before `season` feeds the projections, same discipline
    draft_backtest uses -- so passing the current season runs this against the
    real live board (this year's projections, history through last season)
    instead of a past, already-decided one.

    Scored on real points from `season` when they exist; for a season that
    hasn't been played yet, falls back to the model's own proj_points instead
    (check `scored_on` in the result) -- a forecast of the algorithm's typical
    outcome, not a validated backtest.

    One draw can make the algorithm look better or worse than its true average
    just from bot luck, which is why this runs n_trials and reports the mean,
    not a single result. For each round it also reports the most common picks
    and how often each showed up -- rounds with no real consensus (usually
    round 6+, once enough upstream bot randomness has compounded) should be
    read as "plausible outcomes," not "the pick."

    K/DST aren't modelled, so only skill-position rounds are simulated
    (your league's total rounds minus its K and DST starting slots).
    """
    league, weights = _settings()
    out = adp_mod.mock_draft(league, weights, season, n_trials=n_trials, top_n=top_n)
    return json.dumps(out, indent=2, default=str)


@mcp.tool()
def champion_strategies(league_id: str, seasons: str = "2020,2021,2022,2023,2024,2025") -> str:
    """What actually won your ESPN league, season by season, and which specific
    pick made the difference.

    For each season, finds whichever team finished 1st and pulls their real
    draft. Every pick gets a value verdict -- preseason ECR against actual
    finish, the same steal/bust framing value_picks and draft_backtest use --
    so this answers "what draft-cost bet actually paid off for the winner,"
    not just "what did the champion draft." Reports each champion's opening two
    picks, first QB/TE round, RB/WR volume, and biggest steal, plus
    cross-season patterns: how often champions opened RB-RB, and the median
    round of their first QB.

    biggest_steal also explains *why* it was a steal: usage_trend is that
    player's real early- vs. late-season carries/targets/target share (a role
    expansion actually visible in the box scores), and team_environment is his
    team's O-line ranks, pace, and pass/rush split that season. Most value
    picks turn out to be a volume or role story, not raw talent beating a
    forecast -- this is what shows it concretely.

    ECR history only goes back to 2020 -- seasons before that get position and
    timing data but no value verdicts or steal context. ESPN only.
    """
    yrs = [int(s) for s in seasons.split(",") if s.strip()]
    out = adp_mod.champion_strategies(league_id, yrs)
    return json.dumps(out, indent=2, default=str)


@mcp.tool()
def persistent_value_players(seasons: str = "2021,2022,2023,2024",
                             min_seasons: int = 3, limit: int = 20) -> str:
    """Players who beat their draft cost repeatedly, not once.

    One outperformance is a season; three is a trait. This is the closest the data
    comes to naming players the market persistently misprices.
    """
    league, _ = _settings()
    yrs = [int(s) for s in seasons.split(",") if s.strip()]
    hist = adp_mod.value_history(yrs, league.scoring)
    if hist.empty:
        return json.dumps({"error": "no ECR history available"})
    rep = adp_mod.repeat_value_players(hist, min_seasons)
    cols = ["name", "position", "seasons", "hits", "busts", "hit_rate",
            "avg_value_ratio", "avg_ecr", "avg_games"]
    return json.dumps({
        "min_seasons": min_seasons,
        "best_value": _rows(rep, cols, limit),
        "worst_value": _rows(rep.tail(limit).iloc[::-1], cols, limit),
    }, indent=2, default=str)


@mcp.tool()
def rookie_report(limit: int = 20, position: str | None = None) -> str:
    """Projected rookies for this season, from draft capital and landing spot.

    Rookies have no NFL history, so they're projected off a curve fitted to how draft
    pick converted to first-year production across the last ten classes, then adjusted
    for the offence they landed in. Consistency is deliberately low for all of them:
    rookie roles move mid-season and the floor is a healthy scratch.

    Treat these as the widest error bars on the board.
    """
    b = _mark_drafted(_build_board(), _state())
    r = b[b.get("is_rookie", False) == True]  # noqa: E712
    if position:
        r = r[r["position"] == position.upper()]
    if r.empty:
        return json.dumps({"error": "no rookies on the board — draft class may not be published yet"})
    r = r.sort_values("draft_score", ascending=False)
    cols = ["name", "position", "team", "pick", "draft_round", "college", "adp",
            "overall_rank", "proj_points", "adj_ppg", "exp_games", "consistency",
            "drafted"]
    return json.dumps({
        "rookies": len(r),
        "note": "pick is NFL draft position; adp is fantasy market cost",
        "players": _rows(r, [c for c in cols if c in r.columns], limit),
    }, indent=2, default=str)


@mcp.tool()
def resolve_names(names_csv: str) -> str:
    """Check how names resolve against the board — useful before trusting a paste sync.

    Reports the match type for each name so silent mismatches surface. A name that
    fails to resolve looks like a player who scored zero, which is the single most
    damaging failure mode in this whole pipeline.
    """
    b = _build_board()
    queries = [q.strip() for q in names_csv.split(",") if q.strip()]
    out = []
    for q in queries:
        row, how = bd.match_player_verbose(q, b)
        out.append({
            "query": q,
            "resolved_to": (row["name"] if row is not None else None),
            "position": (row["position"] if row is not None else None),
            "team": (str(row["team"]) if row is not None else None),
            "match_type": how,
        })
    return json.dumps({
        "resolved": sum(1 for o in out if o["resolved_to"]),
        "of": len(out),
        "results": out,
    }, indent=2, default=str)


@mcp.tool()
def prewarm(verbose: bool = True) -> str:
    """Build every cache before draft day so nothing computes while you're on the clock.

    The first query of a session pays for downloading and modelling five seasons.
    Every query after it is served from memory. Run this an hour before your draft,
    not during it.
    """
    import time as _time

    timings, t0 = {}, _time.time()
    steps = [
        ("play_by_play", lambda: sources.play_by_play()),
        ("weekly_stats", lambda: sources.weekly_stats()),
        ("snap_counts", lambda: sources.snap_counts()),
        ("injuries", lambda: sources.injuries()),
        ("rosters", lambda: sources.weekly_rosters()),
        ("schedules", lambda: sources.schedules()),
        ("board", lambda: _build_board()),
        ("oline", lambda: features.oline_ratings()),
        ("pace", lambda: features.team_pace_and_split()),
    ]
    for name, fn in steps:
        s = _time.time()
        try:
            fn()
            timings[name] = round(_time.time() - s, 2)
        except Exception as exc:
            timings[name] = f"failed: {type(exc).__name__}"

    b = _build_board()
    out = {
        "total_seconds": round(_time.time() - t0, 1),
        "players": len(b),
        "rookies": int(b.get("is_rookie", pd.Series(dtype=bool)).sum()),
        "ready": True,
        "note": "All subsequent tool calls are served from memory.",
    }
    if verbose:
        out["step_seconds"] = timings
        out["disk_cache"] = sources.cache_status()
    return json.dumps(out, indent=2, default=str)


@mcp.tool()
def player_report(player_name: str) -> str:
    """Full breakdown of one player: production, role, environment, injury, consistency.

    Also matches kickers (position K), returned with a different, smaller shape --
    distance-banded field goal splits and team offense context instead of the
    skill-position environment multipliers, since kickers aren't run through the
    same model (see kickers.py).
    """
    b = _build_board()
    r = bd.match_player(player_name, b)
    if r is None:
        kb = _build_kicker_board()
        kr = bd.match_player(player_name, kb) if not kb.empty else None
        if kr is None:
            return json.dumps({"error": f"no match for '{player_name}'"})
        fields = ["name", "position", "team", "kicker_rank", "kicker_ppg",
                  "fg_att", "fg_made", "fg_pct", "fg_missed", "fg_blocked",
                  "fg_made_u40", "fg_made_40_49", "fg_made_50p", "fg_long",
                  "long_range_share_pct", "pat_made", "pat_att",
                  "context_tier", "pct_fg", "pct_td", "pct_punt",
                  "plays_per_game", "off_epa", "last_season", "games_last"]
        return json.dumps(_rows(pd.DataFrame([kr]), [f for f in fields if f in kr.index], 1)[0],
                          indent=2)
    fields = ["name", "position", "team", "age", "overall_rank", "pos_rank", "adp", "adp_delta",
              "proj_points", "adj_ppg", "baseline_ppg", "exp_games",
              "consistency", "startable_rate", "spike_rate", "floor", "ceiling", "fp_cv",
              "snap_share", "target_share", "touches",
              "injury_risk", "games_missed_rate", "report_rate", "heavy_seasons", "recent_burden",
              "run_block_rank", "pass_block_rank", "plays_per_game", "neutral_pass_rate",
              "rush_rate", "divisional_games",
              "sep_score", "avg_separation", "avg_cushion", "yprr", "tprr", "yac_oe", "adot",
              "is_rookie", "pick", "draft_round", "college",
              "rz_touches", "rz_td", "rz_td_rate", "rz_baseline_rate",
              "m_oline", "m_volume", "m_schedule", "m_divisional", "m_injury", "m_age",
              "m_separation", "m_td_luck", "m_coverage_trend", "vor"]
    out = _rows(pd.DataFrame([r]), [f for f in fields if f in r.index], 1)[0]
    out["summary"] = model.explain(r)
    return json.dumps(out, indent=2)


@mcp.tool()
def compare_players(names: str) -> str:
    """Compare 2-4 players head to head. Pass a comma-separated list.

    Also matches kickers by name, reported separately from skill-position players
    (different shape, different scale -- kicker_ppg isn't comparable to
    draft_score) since kickers aren't run through the same model. Comparing a
    kicker against a skill player is of limited use for that reason; this is
    mainly for comparing two kickers, or a mixed list where you want both groups'
    numbers in one call.
    """
    b = _build_board()
    kb = _build_kicker_board()
    rows, kicker_rows = [], []
    for n in [x.strip() for x in names.split(",") if x.strip()][:4]:
        r = bd.match_player(n, b)
        if r is not None:
            rows.append(r)
            continue
        kr = bd.match_player(n, kb) if not kb.empty else None
        if kr is not None:
            kicker_rows.append(kr)
    if not rows and not kicker_rows:
        return json.dumps({"error": "no matches"})
    out: dict = {}
    if rows:
        df = pd.DataFrame(rows)
        cols = ["name", "position", "team", "adp", "proj_points", "adj_ppg", "consistency",
                "startable_rate", "spike_rate", "injury_risk", "exp_games", "vor", "draft_score"]
        best = df.sort_values("draft_score", ascending=False).iloc[0]
        out["players"] = _rows(df.sort_values("draft_score", ascending=False), cols, 4)
        out["verdict"] = f"{best['name']} — {model.explain(best)}"
    if kicker_rows:
        kdf = pd.DataFrame(kicker_rows)
        kcols = ["name", "team", "kicker_ppg", "fg_pct", "long_range_share_pct", "context_tier"]
        out["kickers"] = _rows(kdf.sort_values("kicker_ppg", ascending=False), kcols, 4)
    return json.dumps(out, indent=2)


@mcp.tool()
def team_context(team: str) -> str:
    """Offensive environment for an NFL team: O-line, pace, run/pass split, schedule,
    drive efficiency, and red zone play-calling identity.

    `drive_efficiency` and `redzone_identity` are informational context, not folded into
    any player's projection or draft_score -- same convention as `matchup_z` in
    separation_report. `drive_efficiency.pct_td` is the share of that team's drives
    ending in a touchdown (a multiplier on how many scoring chances its players get,
    already reflected in their raw points, so treat this as a confidence check on a
    role rather than an extra adjustment). `redzone_identity.shift` is that team's
    neutral-field pass rate minus its red zone pass rate: a large positive shift means
    the offense gets meaningfully more run-heavy inside the 20 (receiving volume there,
    and the touchdown equity that comes with it, is less trustworthy for that team's
    pass catchers); near zero or negative means the passing game keeps its role even in
    the scoring area.

    `drive_efficiency.pct_fg` doubles as the main kicker-opportunity signal
    `best_available(position="K")`'s `context_tier` is banded from -- how often
    this team's drives stall into a field goal try at all, independent of who's
    kicking.
    """
    league, _ = _settings()
    # No pbp argument: that routes through the memoised builders instead of
    # recomputing a full pass over play-by-play on every call.
    ol = features.oline_ratings()
    pace = features.team_pace_and_split()
    dfn = features.defense_ratings(sc=league.scoring)
    sos = features.strength_of_schedule(CURRENT_SEASON, dfn)
    drive_eff = features.team_drive_efficiency()
    rz_shift = features.redzone_identity_shift()
    t = team.upper()
    recent = int(pace["season"].max())
    out = {
        "team": t,
        "oline": _rows(ol[(ol["team"] == t) & (ol["season"] == recent)],
                       ["season", "run_block_rank", "pass_block_rank", "adj_line_yards",
                        "stuff_rate", "sack_rate"], 1),
        "oline_history": _rows(ol[ol["team"] == t].sort_values("season"),
                               ["season", "run_block_rank", "pass_block_rank"], 6),
        "pace_and_split": _rows(pace[(pace["team"] == t) & (pace["season"] == recent)],
                                ["plays_per_game", "pass_rate", "rush_rate",
                                 "neutral_pass_rate", "off_epa"], 1),
        "schedule": _rows(sos[sos["team"] == t],
                          ["divisional_games"] + [c for c in sos.columns if c.endswith("_z")], 1),
        "drive_efficiency": _rows(
            drive_eff[(drive_eff.get("team") == t) & (drive_eff.get("season", pd.Series(dtype=int)) == recent)]
            if not drive_eff.empty else drive_eff,
            ["season", "drives", "pct_td", "pct_fg", "pct_punt"], 1),
        "redzone_identity": _rows(
            rz_shift[(rz_shift.get("team") == t) & (rz_shift.get("season", pd.Series(dtype=int)) == recent)]
            if not rz_shift.empty else rz_shift,
            ["season", "neutral_pass_rate", "rz_pass_rate", "shift"], 1),
    }
    return json.dumps(out, indent=2, default=str)


@mcp.tool()
def defense_report(position: str = "RB", limit: int = 32) -> str:
    """Defensive rankings against a position — fantasy points allowed, 5-year view.

    Rank 1 = toughest matchup. This is what drives the schedule adjustment.
    """
    league, _ = _settings()
    dfn = features.defense_ratings(sc=league.scoring)
    pos = position.upper()
    col = f"fpa_{pos}"
    if col not in dfn.columns:
        return json.dumps({"error": f"no data for position {pos}"})
    recent = int(dfn["season"].max())
    cur = dfn[dfn["season"] == recent][["team", col, f"{col}_rank", "def_epa_play", "def_rank"]]
    multi = dfn.groupby("team")[col].mean().rename(f"{col}_5yr_avg").reset_index()
    multi[f"{col}_5yr_rank"] = multi[f"{col}_5yr_avg"].rank(method="min").astype(int)
    out = cur.merge(multi, on="team").sort_values(f"{col}_5yr_rank")
    return json.dumps({
        "position": pos, "recent_season": recent,
        "note": "rank 1 = allows fewest fantasy points = toughest matchup",
        "defenses": _rows(out, list(out.columns), limit),
    }, indent=2)


@mcp.tool()
def plan_my_draft(strategy: str = "balanced") -> str:
    """Simulate your whole draft from your slot and return the projected lineup.

    Runs the board forward pick by pick, using ADP to model who realistically falls
    to you at each turn, and applies the same recommendation logic at every stop.
    strategy: balanced, zero_rb, hero_rb, or robust_rb.
    """
    league, weights = _settings()
    state = _state()
    b = _mark_drafted(_build_board(), state).copy()
    b = b[~b["drafted"]]

    tilt = {
        "zero_rb": {"RB": 0.72, "WR": 1.12, "TE": 1.05, "QB": 0.95},
        "hero_rb": {"RB": 1.0, "WR": 1.06, "TE": 1.0, "QB": 0.95},
        "robust_rb": {"RB": 1.16, "WR": 0.94, "TE": 0.98, "QB": 0.92},
        "balanced": {},
    }.get(strategy, {})

    my_picks = [p for p in state.my_picks() if p >= state.on_the_clock]
    roster: dict[str, int] = dict(state.my_roster(b))
    taken: set[str] = set()
    plan = []

    for i, pick in enumerate(my_picks):
        nxt = my_picks[i + 1] if i + 1 < len(my_picks) else None
        pool = b[~b["_key"].isin(taken)].copy()
        # Model who's realistically gone by this pick.
        pool = pool[pool["adp"] > pick - 0.55 * pick ** 0.5 * 2]
        if pool.empty:
            break
        if strategy == "hero_rb" and i > 0 and roster.get("RB", 0) >= 1:
            pool = pool.copy()
            pool.loc[pool["position"] == "RB", "draft_score"] *= 0.7
        for pos, mult in tilt.items():
            pool.loc[pool["position"] == pos, "draft_score"] *= mult

        recs = model.recommend(pool, league, current_pick=pick, next_pick=nxt,
                               roster=roster, top_n=3)
        if recs.empty:
            break
        top = recs.iloc[0]
        taken.add(top["_key"])
        roster[top["position"]] = roster.get(top["position"], 0) + 1
        plan.append({
            "round": (pick - 1) // league.teams + 1, "pick": pick,
            "player": top["name"], "position": top["position"], "team": top.get("team"),
            "adp": round(float(top["adp"]), 1),
            "proj_points": round(float(top["proj_points"]), 1),
            "consistency": round(float(top["consistency"]), 3),
            "alternates": [r["name"] for _, r in recs.iloc[1:].iterrows()],
        })

    total = sum(p["proj_points"] for p in plan)
    return json.dumps({
        "strategy": strategy, "your_slot": state.my_slot,
        "projected_starters_points": round(total, 1),
        "final_roster": roster, "plan": plan,
        "caveat": "ADP-driven simulation of an average draft room. Your league will "
                  "deviate — use who_should_i_pick live rather than following this script.",
    }, indent=2)


@mcp.tool()
def model_settings(consistency_weight: float | None = None, injury_weight: float | None = None,
                   oline_weight: float | None = None, schedule_weight: float | None = None,
                   pace_weight: float | None = None, td_luck_weight: float | None = None,
                   qb_boost: float | None = None, coverage_trend_weight: float | None = None) -> str:
    """Tune how much each factor moves a player. Rebuilds the board.

    td_luck_weight controls how hard a player's red zone touchdown rate gets
    regressed toward what his position converts on average (player_report shows
    rz_touches/rz_td/rz_td_rate/rz_baseline_rate so you can see the raw numbers
    behind the adjustment). Set it to 0 to score players on raw history with no
    touchdown-luck correction at all.

    qb_boost is different from the others: they all adjust the projection from a
    real per-player signal (O-line, pace, etc.); qb_boost is a direct fractional
    lift on QB draft_score you supply because you believe the position is worth
    more than the projection says, not because of any single player's own inputs.
    Comes from champion_strategies/draft_backtest analysis: check whether QB has
    actually beaten its draft cost across your league's real history (not just
    hit rate in general -- that alone doesn't justify this) before setting it
    above 0. It stacks with, and doesn't replace, the roster-need discount that
    already stops the model from wanting a second QB once you have one.

    coverage_trend_weight is the same kind of supplied belief as qb_boost, and
    also defaults to 0. It rewards WR/TE with a short-area profile (high TPRR,
    low aDOT) over boundary/vertical receivers, and RBs with real receiving role
    (target_share), on the theory that 2025's shift to zone coverage and, later,
    back to base personnel (linebackers instead of nickel corners covering the
    slot/backfield) favors those archetypes. Unlike separation or td_luck, this
    isn't backed by a per-player real signal this project can backtest -- open
    data has no man/zone or personnel-package split (see separation.py) -- so
    treat any nonzero value as an opinion you're choosing to weight in, not a
    validated adjustment, and re-check the underlying rates each season.
    """
    league, weights = _settings()
    for name, val in [("consistency_weight", consistency_weight), ("injury", injury_weight),
                      ("oline", oline_weight), ("schedule", schedule_weight),
                      ("pace_volume", pace_weight), ("td_luck", td_luck_weight),
                      ("qb_boost", qb_boost), ("coverage_trend", coverage_trend_weight)]:
        if val is not None:
            setattr(weights, name, float(val))
    save_settings(league, weights)
    _CACHE.update({"weights": weights})
    _BOARDS.pop(league.cache_key(), None)
    p = _board_path(league)
    if p.exists():
        p.unlink()
    return json.dumps({"league": league.name, "weights": weights.__dict__,
                       "board": "will rebuild on next query"}, indent=2)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
