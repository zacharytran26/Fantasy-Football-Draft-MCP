"""Open-source data loaders.

Everything here comes from nflverse (https://github.com/nflverse), which publishes
play-by-play, weekly stats, snap counts, injury reports, rosters and schedules under
a permissive license. No scraping of paywalled sites, no brittle HTML parsing.

Each loader caches to local parquet so a draft never waits on the network.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CACHE_DIR, SEASONS

NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"
NFLDATA = "https://raw.githubusercontent.com/nflverse/nfldata/master/data"

# Only the PBP columns we actually model on. Reading all ~380 is needlessly slow.
PBP_COLS = [
    "season", "week", "game_id", "posteam", "defteam", "play_type",
    "pass", "rush", "epa", "success", "wp", "down", "ydstogo",
    "yards_gained", "sack", "qb_hit", "score_differential",
    "rush_attempt", "pass_attempt", "penalty", "interception",
    "pass_touchdown", "rush_touchdown", "air_yards", "series",
    "yardline_100", "rusher_player_id", "receiver_player_id",
    "drive", "fixed_drive_result",
]


_MEM: dict[str, pd.DataFrame] = {}


def _cached(name: str, builder, max_age_days: float = 7.0) -> pd.DataFrame:
    """Read from memory, then local parquet, rebuilding if missing or stale.

    The in-memory layer matters more than it looks. Several tools each pull
    play-by-play, and re-reading a quarter-million rows off disk on every call
    added most of a second to tools that should feel instant during a live draft.
    Frames are treated as read-only by callers, so one shared copy is safe.
    """
    if name in _MEM:
        return _MEM[name]
    path = CACHE_DIR / f"{name}.parquet"
    if path.exists():
        age_days = (time.time() - path.stat().st_mtime) / 86400
        if age_days < max_age_days:
            df = pd.read_parquet(path)
            _MEM[name] = df
            return df
    df = builder()
    df.to_parquet(path, index=False)
    _MEM[name] = df
    return df


def clear_memory_cache() -> None:
    """Drop in-memory frames, keeping the parquet cache. Used after a forced refresh."""
    _MEM.clear()


def _shrink(df: pd.DataFrame, cat_cols=()) -> pd.DataFrame:
    """Downcast numerics and categorise repeated strings.

    Play-by-play is the memory hog: team codes and play types repeat across a
    quarter-million rows, and every float defaults to 64-bit for values that never
    need it.
    """
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


def _concat_seasons(url_tmpl: str, seasons, columns=None) -> pd.DataFrame:
    frames = []
    for s in seasons:
        try:
            frames.append(pd.read_parquet(url_tmpl.format(season=s), columns=columns))
        except Exception as exc:  # a season may not be published yet
            print(f"  ! skipped {url_tmpl.format(season=s)}: {type(exc).__name__}")
    if not frames:
        raise RuntimeError(f"no seasons loaded for {url_tmpl}")
    return pd.concat(frames, ignore_index=True)


def weekly_stats(seasons=None) -> pd.DataFrame:
    """Per-player, per-week offensive box score for the lookback window.

    nflverse renamed this release from `player_stats` to `stats_player_week` starting
    with 2025, and dropped a few columns along the way. Both layouts are handled and
    normalised so the rest of the codebase sees one consistent schema.
    """
    seasons = seasons or SEASONS
    key = f"weekly_stats_{min(seasons)}_{max(seasons)}"

    def build():
        frames = []
        for s in seasons:
            df = None
            for tmpl in (NFLVERSE + "/stats_player/stats_player_week_{season}.parquet",
                         NFLVERSE + "/player_stats/player_stats_{season}.parquet"):
                try:
                    df = pd.read_parquet(tmpl.format(season=s))
                    break
                except Exception:
                    continue
            if df is None:
                print(f"  ! no weekly stats published for {s}")
                continue
            frames.append(_normalise_weekly(df))
        if not frames:
            raise RuntimeError("no weekly stats loaded")
        return pd.concat(frames, ignore_index=True)

    return _cached(key, build)


def _normalise_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Reconcile the pre-2025 and 2025+ weekly stats layouts."""
    df = df.copy()
    if "recent_team" not in df.columns and "team" in df.columns:
        df["recent_team"] = df["team"]
    # The new release renamed passing interceptions and split sack columns.
    if "interceptions" not in df.columns:
        for alt in ("passing_interceptions", "pass_interceptions"):
            if alt in df.columns:
                df["interceptions"] = df[alt]
                break
        else:
            df["interceptions"] = 0.0
    for col in ("sacks", "sack_yards", "dakota"):
        if col not in df.columns:
            df[col] = np.nan
    return df


def snap_counts(seasons=None) -> pd.DataFrame:
    """Per-player, per-game snap share. The truest signal of role."""
    seasons = seasons or SEASONS
    key = f"snaps_{min(seasons)}_{max(seasons)}"
    return _cached(key, lambda: _concat_seasons(
        NFLVERSE + "/snap_counts/snap_counts_{season}.parquet", seasons
    ))


def injuries(seasons=None) -> pd.DataFrame:
    """Official weekly injury reports (practice status + game designation)."""
    seasons = seasons or SEASONS
    key = f"injuries_{min(seasons)}_{max(seasons)}"
    return _cached(key, lambda: _concat_seasons(
        NFLVERSE + "/injuries/injuries_{season}.parquet", seasons
    ))


def weekly_rosters(seasons=None) -> pd.DataFrame:
    """Week-by-week rosters. Carries birth_date plus the espn_id / sleeper_id crosswalk."""
    seasons = seasons or SEASONS
    key = f"rosters_{min(seasons)}_{max(seasons)}"
    return _cached(key, lambda: _concat_seasons(
        NFLVERSE + "/weekly_rosters/roster_weekly_{season}.parquet", seasons
    ))


def players() -> pd.DataFrame:
    """Master player table: IDs across platforms, birth date, draft capital."""
    return _cached("players", lambda: pd.read_parquet(NFLVERSE + "/players/players.parquet"),
                   max_age_days=3.0)


# nflverse has used both PFR-style and standard team codes across releases; keep
# this in sync with rookies.PFR_TEAM_FIX rather than importing it, since rookies
# imports from this module and a reverse import would be circular.
_TEAM_CODE_FIX = {
    "LVR": "LV", "NOR": "NO", "SFO": "SF", "LAR": "LA", "TAM": "TB",
    "KAN": "KC", "GNB": "GB", "NWE": "NE", "SDG": "LAC", "STL": "LA",
    "OAK": "LV", "RAI": "LV", "PHO": "ARI", "ARZ": "ARI", "CLT": "IND",
    "RAV": "BAL", "HTX": "HOU", "OTI": "TEN", "CRD": "ARI", "BLT": "BAL",
    "JAC": "JAX", "WSH": "WAS",
}


# The feed also carries defensive and special-teams rows (formations like "Base
# 4-3 D", return units, etc.), and ~7% of players show up under more than one of
# these on a given snapshot -- a WR who's also the kick returner, for instance.
# Restricting to skill positions before dedup avoids picking up his return-unit
# slot instead of his real offensive one; this app only models these four anyway.
_DEPTH_CHART_POSITIONS = ("QB", "RB", "WR", "TE")


def depth_charts(season: int | None = None) -> pd.DataFrame:
    """Each player's current NFL team and depth-chart slot, from the official
    team-filed depth charts.

    Weekly box-score stats only update a player's team once he's actually played a
    game for it, so trades, cuts and free-agent signings are invisible until Week 1
    snaps get recorded -- which is well after most drafts happen. Depth charts are
    submitted by the teams themselves and update through the offseason, so they
    catch a move as soon as a team reports it. Refreshed daily (max_age_days=1),
    since August roster churn is fast.

    Returns columns [player_id, team, depth_rank] (1 = starter, per the official
    chart); empty DataFrame if the season's depth chart isn't published yet (very
    early offseason) rather than raising, since this is a "nice if available"
    override, not a hard dependency. depth_rank is NaN wherever the feed doesn't
    carry a rank column at all.
    """
    from .config import CURRENT_SEASON
    season = season or CURRENT_SEASON
    key = f"depth_charts_{season}"

    def build():
        empty = pd.DataFrame(columns=["player_id", "team", "depth_rank"])
        try:
            df = pd.read_parquet(NFLVERSE + f"/depth_charts/depth_charts_{season}.parquet")
        except Exception as exc:
            print(f"  ! no depth chart published yet for {season}: {type(exc).__name__}")
            return empty
        # Column names have varied across nflverse depth_chart schema versions.
        if "team" not in df.columns and "club_code" in df.columns:
            df["team"] = df["club_code"]
        if "player_id" not in df.columns and "gsis_id" in df.columns:
            df["player_id"] = df["gsis_id"]
        if "player_id" not in df.columns or "team" not in df.columns:
            print("  ! depth chart missing expected columns, skipping team override")
            return empty
        if "pos_abb" in df.columns:
            df = df[df["pos_abb"].isin(_DEPTH_CHART_POSITIONS)]
        df = df.dropna(subset=["player_id", "team"]).copy()
        df["team"] = df["team"].astype("string").str.upper().replace(_TEAM_CODE_FIX)
        df["depth_rank"] = df["pos_rank"] if "pos_rank" in df.columns else pd.NA
        # Keep each player's most recent report. Schema has varied between a
        # season-long feed with a "week" column and a rolling feed stamped with a
        # "dt" scrape timestamp instead -- sort by whichever exists so keep='last'
        # actually keeps the latest report rather than an arbitrary row order.
        sort_cols = [c for c in ("dt", "week") if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols)
        return df.drop_duplicates("player_id", keep="last")[["player_id", "team", "depth_rank"]]

    return _cached(key, build, max_age_days=1.0)


def schedules() -> pd.DataFrame:
    """All games incl. future season when released. `div_game` flags divisional matchups."""
    return _cached("schedules", lambda: pd.read_csv(f"{NFLDATA}/games.csv"), max_age_days=1.0)


def play_by_play(seasons=None, columns=None) -> pd.DataFrame:
    """Play-by-play. Heavy (~1 min/season on first pull), then cached.

    This is what powers the O-line, pace, run/pass split and defensive rankings —
    computing them from plays is more reliable than scraping somebody's ranking table.
    """
    seasons = seasons or SEASONS
    columns = columns or PBP_COLS
    key = f"pbp_{min(seasons)}_{max(seasons)}"

    def build():
        print(f"Downloading play-by-play for {min(seasons)}-{max(seasons)} (one-time, a few minutes)...")
        df = _concat_seasons(NFLVERSE + "/pbp/play_by_play_{season}.parquet", seasons, columns)
        return _shrink(df, cat_cols=("posteam", "defteam", "play_type", "game_id"))

    return _cached(key, build, max_age_days=30.0)


def cache_status() -> list[dict]:
    out = []
    for p in sorted(Path(CACHE_DIR).glob("*.parquet")):
        out.append({
            "dataset": p.stem,
            "size_mb": round(p.stat().st_size / 1e6, 1),
            "age_days": round((time.time() - p.stat().st_mtime) / 86400, 2),
        })
    return out
