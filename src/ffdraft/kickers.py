"""Kicker fantasy scoring and draft context.

Kickers aren't run through the skill-position model in model.py -- see
docs/methodology.md's "Known limitations". The O-line/pace/schedule/separation/
td-luck machinery in model.project() is built for players whose value comes from
touches and touchdowns; none of it has a meaningful kicker analogue, and forcing a
kicker board through that FANTASY_POSITIONS-shaped pipeline risks quietly breaking
VOR and roster-need logic that already works for QB/RB/WR/TE.

This module is a smaller, self-contained analytical layer instead: a kicker's own
distance-banded scoring (recency-weighted across the same 5-season lookback the
rest of the model uses), plus the team-offense signals that explain how many -- and
how long -- his attempts are likely to be. Those signals (`team_drive_efficiency`,
`team_pace_and_split`) are reused from features.py rather than reinvented: the same
frames `team_context` already surfaces for skill positions answer exactly what
matters for a kicker too -- where a team's drives tend to end (`pct_td`/`pct_fg`/
`pct_punt`, the "average end position" signal) and how much offense it runs
(`plays_per_game`, `off_epa`, the "offensive prowess/efficiency" signal).

Why this is worth a real ranking rather than "take one last": under most leagues'
actual distance-banded scoring (see Scoring.fg_made_* in config.py), a kicker on a
good offense who's trusted from 50+ scores like a low-end WR2/high-end WR3, not the
flat "3 points a field goal" replacement-level afterthought a standard scoring
mental model assumes. The gap between kickers is real and comes from two separable
things: his own leg (accuracy, especially 40+) and how often his offense puts him
in position to kick at all -- `context_tier` exists to tell those apart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import features, sources
from .config import CURRENT_SEASON, LeagueSettings, Scoring


def kicker_fantasy_points(df: pd.DataFrame, sc: Scoring) -> pd.Series:
    """Apply league kicker scoring to weekly box-score rows.

    Distance bands are ESPN's own (see Scoring docstring): under 40 / 40-49 / 50+,
    a flat penalty for any miss or block, PAT made/missed-or-blocked. `fg_made_60_`
    (nflverse's 60+ column) counts toward the 50+ band and additionally toward the
    optional `fg_made_60_bonus`, which is 0 unless a league opts in.
    """
    g = lambda c: df[c].fillna(0) if c in df.columns else 0.0  # noqa: E731
    made_under_40 = g("fg_made_0_19") + g("fg_made_20_29") + g("fg_made_30_39")
    made_40_49 = g("fg_made_40_49")
    made_50_59 = g("fg_made_50_59")
    made_60_plus = g("fg_made_60_")
    missed_or_blocked = g("fg_missed") + g("fg_blocked")
    pat_made = g("pat_made")
    pat_bad = g("pat_missed") + g("pat_blocked")
    return (
        made_under_40 * sc.fg_made_0_39
        + made_40_49 * sc.fg_made_40_49
        + (made_50_59 + made_60_plus) * sc.fg_made_50_plus
        + made_60_plus * sc.fg_made_60_bonus
        + missed_or_blocked * sc.fg_missed
        + pat_made * sc.pat_made
        + pat_bad * sc.pat_missed
    )


_BAND_COLS = [
    "fg_att", "fg_made", "fg_missed", "fg_blocked", "fg_long",
    "fg_made_0_19", "fg_made_20_29", "fg_made_30_39", "fg_made_40_49",
    "fg_made_50_59", "fg_made_60_", "pat_made", "pat_att",
]


def _kicker_weekly(sc: Scoring, seasons=None) -> pd.DataFrame:
    w = sources.weekly_stats(seasons)
    w = w[(w["position"] == "K") & (w["season_type"] == "REG")].copy()
    for c in _BAND_COLS:
        if c not in w.columns:
            w[c] = 0.0
    w["fp"] = kicker_fantasy_points(w, sc)
    w["fg_made_u40"] = w["fg_made_0_19"] + w["fg_made_20_29"] + w["fg_made_30_39"]
    w["fg_made_50p"] = w["fg_made_50_59"] + w["fg_made_60_"]
    return w


def kicker_season_profiles(sc: Scoring, seasons=None) -> pd.DataFrame:
    """Per-kicker, per-season production and field goal distance profile."""
    w = _kicker_weekly(sc, seasons)
    if w.empty:
        return pd.DataFrame(columns=[
            "player_id", "player_display_name", "season", "team", "games",
            "fp_total", "fp_mean", "fg_att", "fg_made", "fg_missed", "fg_blocked",
            "fg_made_u40", "fg_made_40_49", "fg_made_50p", "fg_long", "pat_made", "pat_att",
        ])
    return w.groupby(["player_id", "player_display_name", "season"]).agg(
        team=("recent_team", "last"),
        games=("week", "nunique"),
        fp_total=("fp", "sum"),
        fp_mean=("fp", "mean"),
        fg_att=("fg_att", "sum"),
        fg_made=("fg_made", "sum"),
        fg_missed=("fg_missed", "sum"),
        fg_blocked=("fg_blocked", "sum"),
        fg_made_u40=("fg_made_u40", "sum"),
        fg_made_40_49=("fg_made_40_49", "sum"),
        fg_made_50p=("fg_made_50p", "sum"),
        fg_long=("fg_long", "max"),
        pat_made=("pat_made", "sum"),
        pat_att=("pat_att", "sum"),
    ).reset_index()


def kicker_board(league: LeagueSettings, season: int | None = None) -> pd.DataFrame:
    """Rank kickers for the given season.

    Recency-weighted fantasy points per game (`kicker_ppg`, same 5-season lookback
    and weighting `_season_weighted` uses for skill positions) under the league's
    own distance-banded scoring, plus team-offense context reused from
    `team_drive_efficiency`/`team_pace_and_split` -- informational, like
    `drive_efficiency` in `team_context`, not blended into `kicker_ppg`, since a
    kicker's own points already reflect how often his offense fed him chances.
    Blending it in a second time would double-count exactly what team_context's
    docstring already warns about for skill positions.

    Team is each kicker's most-recently-played team from box scores, not
    reconciled against the current depth chart the way build_player_table does for
    skill positions -- kickers change teams rarely enough in-season that this
    wasn't worth the added complexity, but a just-signed free-agent kicker may
    still show his old team until he's played a game for the new one.
    """
    season = season or CURRENT_SEASON
    sc = league.scoring
    lookback = list(range(season - 5, season))
    profiles = kicker_season_profiles(sc, seasons=lookback)
    if profiles.empty:
        return profiles.assign(position="K", kicker_ppg=pd.Series(dtype=float))

    wmap = features._season_weights(sorted(profiles["season"].unique()))
    p = profiles.copy()
    p["w"] = p["season"].map(wmap).fillna(0)
    p["wv"] = p["fp_mean"].fillna(0) * p["w"]
    agg = p.groupby("player_id").agg(num=("wv", "sum"), den=("w", "sum"))
    ppg = (agg["num"] / agg["den"].replace(0, np.nan)).rename("kicker_ppg")

    latest = profiles.sort_values("season").groupby("player_id").agg(
        name=("player_display_name", "last"),
        team=("team", "last"),
        last_season=("season", "max"),
        games_last=("games", "last"),
    ).reset_index()
    tbl = latest.merge(ppg.reset_index(), on="player_id", how="left")
    tbl["position"] = "K"

    # Recent enough to matter (same trailing-year bound build_player_table uses)
    # and a real sample, not a one-game injury fill-in.
    tbl = tbl[tbl["last_season"] >= profiles["season"].max() - 1]
    tbl = tbl[tbl["games_last"].fillna(0) >= 4].reset_index(drop=True)

    totals = profiles.groupby("player_id").agg(
        fg_att=("fg_att", "sum"), fg_made=("fg_made", "sum"),
        fg_missed=("fg_missed", "sum"), fg_blocked=("fg_blocked", "sum"),
        fg_made_u40=("fg_made_u40", "sum"), fg_made_40_49=("fg_made_40_49", "sum"),
        fg_made_50p=("fg_made_50p", "sum"), fg_long=("fg_long", "max"),
        pat_made=("pat_made", "sum"), pat_att=("pat_att", "sum"),
    ).reset_index()
    tbl = tbl.merge(totals, on="player_id", how="left")
    tbl["fg_pct"] = (100 * tbl["fg_made"] / tbl["fg_att"].replace(0, np.nan)).round(1)
    tbl["long_range_share_pct"] = (
        100 * (tbl["fg_made_40_49"] + tbl["fg_made_50p"]) / tbl["fg_att"].replace(0, np.nan)
    ).round(1)

    # --- team offense context, reused rather than re-derived (see docstring)
    drive_eff = features.team_drive_efficiency()
    pace = features.team_pace_and_split()
    if not drive_eff.empty:
        recent_de = int(drive_eff["season"].max())
        de = drive_eff[drive_eff["season"] == recent_de][["team", "pct_td", "pct_fg", "pct_punt"]]
        tbl = tbl.merge(de, on="team", how="left")
    if not pace.empty:
        recent_pc = int(pace["season"].max())
        pc = pace[pace["season"] == recent_pc][["team", "plays_per_game", "off_epa"]]
        tbl = tbl.merge(pc, on="team", how="left")

    # context_tier: how often this offense's drives stall into a field-goal try at
    # all (pct_fg), quintile-banded across the kickers on the board (one per team,
    # so this is effectively banded across the league) -- independent of the
    # kicker's own accuracy. Same "not folded into the score, confidence check
    # only" convention as drive_efficiency in team_context.
    if "pct_fg" in tbl.columns and tbl["pct_fg"].notna().any():
        ranks = tbl["pct_fg"].rank(pct=True)
        tbl["context_tier"] = np.select(
            [ranks >= 0.8, ranks >= 0.6, ranks >= 0.4, ranks >= 0.2],
            ["Elite", "Above avg", "Average", "Below avg"],
            default="Bottom",
        )
    else:
        tbl["context_tier"] = "Average"

    tbl["kicker_rank"] = tbl["kicker_ppg"].rank(ascending=False, method="min").astype("Int64")
    return tbl.sort_values("kicker_ppg", ascending=False).reset_index(drop=True)
