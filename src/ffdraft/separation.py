"""Separation and route efficiency.

The PFF chart you shared (SEP score, YPRR/TPRR split by man and zone) is licensed
manual charting. But most of what it measures is recoverable from open data:

  * **Separation** — NFL Next Gen Stats publishes `avg_separation`, the tracking-measured
    distance in yards between receiver and nearest defender at the moment the pass
    arrives. That is the same underlying quantity as PFF's SEP, from player tracking
    chips rather than human charting.
  * **Cushion** — how far off the defender lines up pre-snap. Reads coverage respect.
  * **YPRR / TPRR** — need routes run, which no free source publishes directly. Routes
    are estimated as snap share times team dropbacks, which tracks true route counts
    closely for perimeter receivers.

The man-versus-zone split *can* be recovered, from nflverse per-play coverage tags
rather than manual charting -- see `coverage.py`. Those tags come from NGS tracking, cover
roughly half of dropbacks, and measure exposure as snaps on the field instead of routes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import sources
from .config import SEASONS
from .sources import _cached

NGS_BASE = "https://github.com/nflverse/nflverse-data/releases/download/nextgen_stats/"


def ngs_receiving() -> pd.DataFrame:
    """Next Gen Stats receiving: separation, cushion, YAC over expected."""
    def build():
        return pd.read_parquet(NGS_BASE + "ngs_receiving.parquet")
    return _cached("ngs_receiving", build, max_age_days=3.0)


def ngs_rushing() -> pd.DataFrame:
    """Next Gen Stats rushing: yards over expected, time to line of scrimmage,
    and how often a back faces a stacked box."""
    def build():
        return pd.read_parquet(NGS_BASE + "ngs_rushing.parquet")
    return _cached("ngs_rushing", build, max_age_days=3.0)


def estimated_routes(seasons=None) -> pd.DataFrame:
    """Routes run per player-season, estimated from snap share and team dropbacks.

    A receiver runs a route on roughly every dropback he's on the field for, so
    snap share times team pass plays is a close approximation. It slightly
    overstates backs and blocking tight ends, who stay in to protect — hence the
    positional damping below.
    """
    seasons = seasons or SEASONS
    snaps = sources.snap_counts(seasons)
    snaps = snaps[snaps["game_type"] == "REG"]
    pbp = sources.play_by_play(seasons)

    # snap_counts and pbp use different game_id formats, so join on team-week.
    week_map = (
        pbp[pbp["posteam"].notna()].groupby(["season", "posteam", "week"])["pass"].sum()
        .reset_index().rename(columns={"posteam": "team", "pass": "team_dropbacks"})
    )
    s = snaps.merge(week_map, on=["season", "team", "week"], how="left")

    # Backs and in-line tight ends stay in to block on a share of dropbacks.
    damp = s["position"].map({"RB": 0.62, "FB": 0.35, "TE": 0.86}).fillna(0.97)
    s["routes_est"] = s["offense_pct"].fillna(0) * s["team_dropbacks"].fillna(0) * damp

    return s.groupby(["player", "season", "position"]).agg(
        routes_est=("routes_est", "sum"),
        games=("week", "nunique"),
        snap_pct=("offense_pct", "mean"),
    ).reset_index()


def separation_profile(seasons=None) -> pd.DataFrame:
    from .features import _DERIVED
    key = f"sepprof_{tuple(seasons) if seasons else 'default'}"
    if key not in _DERIVED:
        _DERIVED[key] = _separation_profile(seasons)
    return _DERIVED[key]


def _separation_profile(seasons=None) -> pd.DataFrame:
    """Per-player, per-season separation and route efficiency.

    Returns the open-data analogue of the PFF table: separation, cushion, targets
    per route run, yards per route run, plus a composite separation score expressed
    as a z-score against other receivers that season.
    """
    from . import features
    from .names import normalize as norm_name

    seasons = seasons or SEASONS
    ngs = ngs_receiving()
    ngs = ngs[(ngs["season"].isin(seasons)) & (ngs["season_type"] == "REG")]

    # NGS publishes both weekly rows (week > 0) and a season summary (week == 0).
    season_rows = ngs[ngs["week"] == 0]
    if season_rows.empty:
        season_rows = ngs.groupby(["player_gsis_id", "player_display_name",
                                   "player_position", "season", "team_abbr"]).agg(
            avg_separation=("avg_separation", "mean"),
            avg_cushion=("avg_cushion", "mean"),
            avg_yac_above_expectation=("avg_yac_above_expectation", "mean"),
            avg_intended_air_yards=("avg_intended_air_yards", "mean"),
            percent_share_of_intended_air_yards=("percent_share_of_intended_air_yards", "mean"),
            catch_percentage=("catch_percentage", "mean"),
            targets=("targets", "sum"), receptions=("receptions", "sum"),
            yards=("yards", "sum"),
        ).reset_index()

    sep = season_rows.rename(columns={
        "player_display_name": "name", "player_position": "position",
        "team_abbr": "team", "player_gsis_id": "player_id",
    })

    # Real production, for YPRR and TPRR.
    w = sources.weekly_stats(seasons)
    w = w[w["season_type"] == "REG"]
    prod = w.groupby(["player_id", "season"]).agg(
        rec_yards=("receiving_yards", "sum"),
        rec_targets=("targets", "sum"),
        receptions_actual=("receptions", "sum"),
        air_yards=("receiving_air_yards", "sum"),
        rec_epa=("receiving_epa", "sum"),
        target_share=("target_share", "mean"),
        wopr=("wopr", "mean"),
        racr=("racr", "mean"),
    ).reset_index()
    sep = sep.merge(prod, on=["player_id", "season"], how="left")

    routes = estimated_routes(seasons)
    routes["_key"] = routes["player"].map(norm_name)
    sep["_key"] = sep["name"].map(norm_name)
    sep = sep.merge(routes[["_key", "season", "routes_est", "snap_pct"]],
                    on=["_key", "season"], how="left")

    r = sep["routes_est"].replace(0, np.nan)
    sep["yprr"] = sep["rec_yards"] / r
    sep["tprr"] = sep["rec_targets"] / r
    sep["yptarget"] = sep["rec_yards"] / sep["rec_targets"].replace(0, np.nan)

    # Composite separation score, z-scored within season against qualified receivers.
    # Qualification is deliberately strict: these metrics are rate stats, and a
    # part-time receiver with 150 routes will post a flattering YPRR that says
    # nothing about how he'd perform in a real workload.
    sep["qualified"] = (sep["routes_est"].fillna(0) >= 250) & (sep["rec_targets"].fillna(0) >= 50)
    sep["sep_score"] = np.nan
    for _season, chunk in sep.groupby("season"):
        q = chunk[chunk["qualified"]]
        if len(q) < 10:
            continue
        z = (
            1.0 * features._zscore(q["avg_separation"].fillna(q["avg_separation"].mean()))
            + 0.6 * features._zscore(q["yprr"].fillna(q["yprr"].mean()))
            + 0.4 * features._zscore(q["tprr"].fillna(q["tprr"].mean()))
            + 0.3 * features._zscore(
                q["avg_yac_above_expectation"].fillna(0))
        ) / 2.3
        sep.loc[q.index, "sep_score"] = z
    return sep


def separation_summary(seasons=None) -> pd.DataFrame:
    """Recency-weighted separation profile, one row per player, for the model."""
    from . import features

    prof = separation_profile(seasons)
    prof = prof[prof["qualified"]]
    if prof.empty:
        return pd.DataFrame(columns=["player_id", "sep_score", "yprr", "tprr"])
    wmap = features._season_weights(sorted(prof["season"].unique()))
    prof = prof.copy()
    prof["w"] = prof["season"].map(wmap).fillna(0)

    # Weighted averages as vectorised sum(value*w)/sum(w) per column, skipping NaN
    # by zeroing both numerator and denominator. A groupby.apply building a Series
    # per player does the same maths at Python speed.
    cols = {
        "sep_score": "sep_score", "avg_separation": "avg_separation",
        "avg_cushion": "avg_cushion", "yprr": "yprr", "tprr": "tprr",
        "yac_oe": "avg_yac_above_expectation", "adot": "avg_intended_air_yards",
    }
    keys = ["player_id", "name", "position"]
    agg_spec, frame = {}, prof[keys + ["w", "routes_est"]].copy()
    for out_col, src in cols.items():
        v = prof[src]
        ok = v.notna()
        frame[f"_n_{out_col}"] = (v.fillna(0) * prof["w"]).where(ok, 0.0)
        frame[f"_d_{out_col}"] = prof["w"].where(ok, 0.0)
        agg_spec[f"_n_{out_col}"] = "sum"
        agg_spec[f"_d_{out_col}"] = "sum"
    agg_spec["routes_est"] = "mean"
    g = frame.groupby(keys, observed=True).agg(agg_spec)
    g["seasons_qualified"] = frame.groupby(keys, observed=True).size()
    for out_col in cols:
        g[out_col] = g[f"_n_{out_col}"] / g[f"_d_{out_col}"].replace(0, np.nan)
    g = g.drop(columns=[c for c in g.columns if c.startswith(("_n_", "_d_"))])
    return g.reset_index()
