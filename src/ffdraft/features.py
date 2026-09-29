"""Derived features. Everything is computed from raw plays and box scores rather than
scraped from someone's published ranking table — the numbers stay reproducible and
don't break when a website changes its HTML.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import sources
from .config import (
    AGE_CLIFF,
    AGE_DECAY,
    FANTASY_POSITIONS,
    RECENCY_WEIGHTS,
    SPIKE_THRESHOLD,
    STARTABLE_THRESHOLD,
    WORKLOAD_BURDEN,
    Scoring,
)

# ---------------------------------------------------------------- scoring

def fantasy_points(df: pd.DataFrame, sc: Scoring, te_bonus: float = 0.0) -> pd.Series:
    """Apply league scoring to weekly box-score rows."""
    g = lambda c: df[c].fillna(0) if c in df.columns else 0.0  # noqa: E731
    pts = (
        g("passing_yards") * sc.pass_yd
        + g("passing_tds") * sc.pass_td
        + g("interceptions") * sc.interception
        + g("rushing_yards") * sc.rush_yd
        + g("rushing_tds") * sc.rush_td
        + g("receptions") * sc.rec
        + g("receiving_yards") * sc.rec_yd
        + g("receiving_tds") * sc.rec_td
        + (g("rushing_fumbles_lost") + g("receiving_fumbles_lost") + g("sack_fumbles_lost"))
        * sc.fumble_lost
        + (g("passing_2pt_conversions") + g("rushing_2pt_conversions")
           + g("receiving_2pt_conversions")) * sc.two_pt
    )
    if te_bonus and "position" in df.columns:
        pts = pts + np.where(df["position"].eq("TE"), g("receptions") * te_bonus, 0.0)
    return pts


# Derived team frames are pure functions of the cached play-by-play, but each one
# costs a full pass over ~250k rows. Several tools ask for the same frame, so the
# results are memoised for the process lifetime.
_DERIVED: dict[str, pd.DataFrame] = {}


def clear_derived_cache() -> None:
    _DERIVED.clear()


def _memo(key: str, builder):
    if key not in _DERIVED:
        _DERIVED[key] = builder()
    return _DERIVED[key]


def _zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and sd > 0 else pd.Series(0.0, index=s.index)


def _season_weights(seasons: list[int]) -> dict[int, float]:
    """Recency weights aligned to however many seasons actually loaded."""
    seasons = sorted(seasons)
    w = RECENCY_WEIGHTS[-len(seasons):]
    total = sum(w)
    return {s: wi / total for s, wi in zip(seasons, w)}


# ---------------------------------------------------------------- team environment

def team_pace_and_split(pbp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Plays per game, run/pass split, and neutral-script pass rate by team-season.

    Neutral script = win probability between 20% and 80%, which strips out
    garbage-time pass volume and kneel-down run volume. That matters: a team's
    raw pass rate mostly tells you whether they were losing.
    """
    if pbp is None:
        return _memo("pace", lambda: _team_pace_and_split(sources.play_by_play()))
    return _team_pace_and_split(pbp)


def _team_pace_and_split(pbp: pd.DataFrame) -> pd.DataFrame:
    off = pbp[pbp["posteam"].notna() & pbp["play_type"].isin(["pass", "run"])].copy()

    grp = off.groupby(["season", "posteam"])
    base = grp.agg(
        plays=("play_type", "size"),
        games=("game_id", "nunique"),
        pass_plays=("pass", "sum"),
        rush_plays=("rush", "sum"),
        off_epa=("epa", "mean"),
    ).reset_index()

    neutral = off[off["wp"].between(0.2, 0.8)]
    neu = neutral.groupby(["season", "posteam"]).agg(
        neutral_plays=("play_type", "size"),
        neutral_pass=("pass", "sum"),
    ).reset_index()

    out = base.merge(neu, on=["season", "posteam"], how="left")
    out["plays_per_game"] = out["plays"] / out["games"].clip(lower=1)
    out["pass_rate"] = out["pass_plays"] / out["plays"].clip(lower=1)
    out["rush_rate"] = 1 - out["pass_rate"]
    out["neutral_pass_rate"] = out["neutral_pass"] / out["neutral_plays"].clip(lower=1)
    return out.rename(columns={"posteam": "team"})


def oline_ratings(pbp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Offensive line quality, split into run block and pass block.

    Run block uses Adjusted Line Yards: rushing yards are credited to the line on a
    sliding scale, because the line owns the first few yards and the back owns the
    breakaway. Pass block uses pressure allowed (sacks + QB hits) per dropback.
    """
    if pbp is None:
        return _memo("oline", lambda: _oline_ratings(sources.play_by_play()))
    return _oline_ratings(pbp)


def _oline_ratings(pbp: pd.DataFrame) -> pd.DataFrame:

    # --- run block: adjusted line yards
    runs = pbp[(pbp["play_type"] == "run") & pbp["posteam"].notna()].copy()
    y = runs["yards_gained"].fillna(0)
    aly = np.select(
        [y < 0, y <= 4, y <= 10],
        [y * 1.2, y * 1.0, 4 + (y - 4) * 0.5],
        default=4 + 6 * 0.5,  # yards past 10 are the back, not the line
    )
    runs["aly"] = aly
    runs["stuffed"] = (y <= 0).astype(float)
    run_g = runs.groupby(["season", "posteam"]).agg(
        adj_line_yards=("aly", "mean"),
        stuff_rate=("stuffed", "mean"),
        rush_epa=("epa", "mean"),
    ).reset_index()

    # --- pass block: pressure allowed per dropback
    drops = pbp[(pbp["play_type"].isin(["pass", "run"])) & (pbp["pass"] == 1)
                & pbp["posteam"].notna()].copy()
    pass_g = drops.groupby(["season", "posteam"]).agg(
        sack_rate=("sack", "mean"),
        hit_rate=("qb_hit", "mean"),
        pass_epa=("epa", "mean"),
    ).reset_index()

    ol = run_g.merge(pass_g, on=["season", "posteam"], how="outer").rename(
        columns={"posteam": "team"}
    )

    # Per-season z-scores so a rating means "relative to that year's league".
    for _season, chunk in ol.groupby("season"):
        idx = chunk.index
        run_score = _zscore(chunk["adj_line_yards"]) - _zscore(chunk["stuff_rate"])
        pass_score = -_zscore(chunk["sack_rate"]) - _zscore(chunk["hit_rate"])
        ol.loc[idx, "run_block_z"] = (run_score / 2).values
        ol.loc[idx, "pass_block_z"] = (pass_score / 2).values
    ol["run_block_rank"] = ol.groupby("season")["run_block_z"].rank(ascending=False).astype(int)
    ol["pass_block_rank"] = ol.groupby("season")["pass_block_z"].rank(ascending=False).astype(int)
    return ol


def defense_ratings(pbp: pd.DataFrame | None = None,
                    weekly: pd.DataFrame | None = None,
                    sc: Scoring | None = None) -> pd.DataFrame:
    """Defensive strength by team-season, overall and by position defended.

    Two views: EPA allowed (how good the defense actually is) and fantasy points
    allowed per game to each position (what it costs the players you'd draft).
    The second is what 'RB rankings against opposing teams' really means.
    """
    if pbp is None and weekly is None:
        _sc = sc or Scoring()
        # Key on the scoring values, not id(): CPython reuses ids after garbage
        # collection, so an id-keyed cache can silently serve another league's numbers.
        key = "defense_" + ",".join(f"{k}={v}" for k, v in sorted(vars(_sc).items()))
        return _memo(key, lambda: _defense_ratings(
            sources.play_by_play(), sources.weekly_stats(), _sc))
    return _defense_ratings(
        sources.play_by_play() if pbp is None else pbp,
        sources.weekly_stats() if weekly is None else weekly,
        sc or Scoring(),
    )


def _defense_ratings(pbp: pd.DataFrame, weekly: pd.DataFrame, sc: Scoring) -> pd.DataFrame:
    d = pbp[pbp["defteam"].notna() & pbp["play_type"].isin(["pass", "run"])]
    # Split pass and rush before grouping. The obvious version passes a lambda that
    # indexes back into the parent frame with .loc for every group, which is an O(n)
    # lookup repeated once per team-season.
    base = d.groupby(["season", "defteam"], observed=True)["epa"].mean()
    ep = d[d["pass"] == 1].groupby(["season", "defteam"], observed=True)["epa"].mean()
    er = d[d["rush"] == 1].groupby(["season", "defteam"], observed=True)["epa"].mean()
    epa = pd.concat([
        base.rename("def_epa_play"), ep.rename("def_epa_pass"), er.rename("def_epa_rush"),
    ], axis=1).reset_index().rename(columns={"defteam": "team"})

    # Fantasy points allowed per game by position.
    w = weekly[weekly["position"].isin(FANTASY_POSITIONS)].copy()
    w["fp"] = fantasy_points(w, sc)
    fpa = (
        w.groupby(["season", "opponent_team", "position", "week"])["fp"].sum()
        .groupby(level=["season", "opponent_team", "position"]).mean()
        .reset_index().rename(columns={"opponent_team": "team", "fp": "fpa_per_game"})
    )
    fpa_wide = fpa.pivot_table(index=["season", "team"], columns="position",
                               values="fpa_per_game").reset_index()
    fpa_wide.columns.name = None
    fpa_wide = fpa_wide.rename(columns={p: f"fpa_{p}" for p in FANTASY_POSITIONS})

    out = epa.merge(fpa_wide, on=["season", "team"], how="outer")
    for pos in FANTASY_POSITIONS:
        col = f"fpa_{pos}"
        if col in out.columns:
            # Rank 1 = toughest defense against that position.
            out[f"{col}_rank"] = out.groupby("season")[col].rank(method="min").astype("Int64")
    out["def_rank"] = out.groupby("season")["def_epa_play"].rank(method="min").astype("Int64")
    return out


def team_drive_efficiency(pbp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Share of each team's offensive drives that end in a touchdown, field goal, or punt.

    Where a team *starts* drives is mostly special teams and field position -- where it
    *ends* them, and how often that's a touchdown, is the offense itself. This is the
    context that separates "same role, different opportunity": a mid-tier receiver on a
    team that finishes 30%+ of drives with a touchdown gets meaningfully more scoring
    chances than the same role on a team that punts on four of every nine possessions,
    even though that difference is invisible in a season-long target share.

    Requires the `fixed_drive_result` column, added to PBP_COLS after this function was
    written -- a play_by_play parquet cached before that change won't have it. Run
    `refresh_data(force_download=true)` if this returns empty for a season that should
    have data.
    """
    if pbp is None:
        return _memo("drive_efficiency", lambda: _team_drive_efficiency(sources.play_by_play()))
    return _team_drive_efficiency(pbp)


def _team_drive_efficiency(pbp: pd.DataFrame) -> pd.DataFrame:
    if "fixed_drive_result" not in pbp.columns:
        return pd.DataFrame(columns=["season", "team", "drives", "pct_td", "pct_fg", "pct_punt"])

    drives = (
        pbp[pbp["posteam"].notna() & pbp["drive"].notna()]
        .groupby(["season", "posteam", "drive"], observed=True)["fixed_drive_result"]
        .first()
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    out = drives.groupby(["season", "team"]).agg(
        drives=("fixed_drive_result", "size"),
        tds=("fixed_drive_result", lambda s: s.astype("string").eq("Touchdown").sum()),
        fgs=("fixed_drive_result", lambda s: s.astype("string").eq("Field goal").sum()),
        punts=("fixed_drive_result", lambda s: s.astype("string").eq("Punt").sum()),
    ).reset_index()
    out["pct_td"] = 100 * out["tds"] / out["drives"].clip(lower=1)
    out["pct_fg"] = 100 * out["fgs"] / out["drives"].clip(lower=1)
    out["pct_punt"] = 100 * out["punts"] / out["drives"].clip(lower=1)
    return out.drop(columns=["tds", "fgs", "punts"])


def redzone_identity_shift(pbp: pd.DataFrame | None = None) -> pd.DataFrame:
    """How much a team's pass rate drops once it crosses into the red zone.

    Compares red zone (yardline_100 <= 20) pass rate against that same team's pass rate
    everywhere else on the field. A team can throw the ball at a normal rate for 80 yards
    and still hand the ball to its running back (or a rushing QB) the moment it's inside
    the 20 -- a receiver's season-long target share says nothing about whether he keeps
    that role in exactly the situations where touchdowns happen. `shift` is neutral minus
    red-zone pass rate: positive means the offense gets meaningfully more run-heavy near
    the goal line (receiving volume there is less trustworthy), near zero or negative means
    the passing game keeps its role even in the scoring area.

    Informational, like `matchup_z` in separation_report -- not folded into any player's
    projection. A per-team tendency isn't the same as a per-player role, and blending an
    unvalidated new signal into draft_score is exactly the mistake matchup_backtest exists
    to catch (it already found doing this with schedule difficulty made WR predictions
    worse, not better). Use it as read-before-you-draft context, not a score adjustment.
    """
    if pbp is None:
        return _memo("redzone_identity_shift", lambda: _redzone_identity_shift(sources.play_by_play()))
    return _redzone_identity_shift(pbp)


def _redzone_identity_shift(pbp: pd.DataFrame) -> pd.DataFrame:
    off = pbp[pbp["posteam"].notna() & pbp["play_type"].isin(["pass", "run"])]
    rz = off[off["yardline_100"].notna() & (off["yardline_100"] <= 20)]
    neutral = off[off["yardline_100"].notna() & (off["yardline_100"] > 20)]

    def _pass_rate(df: pd.DataFrame) -> pd.DataFrame:
        g = df.groupby(["season", "posteam"], observed=True)
        return (g["pass"].sum() / g["pass"].size().clip(lower=1)).rename("pass_rate").reset_index()

    rz_rate = _pass_rate(rz).rename(columns={"pass_rate": "rz_pass_rate"})
    neu_rate = _pass_rate(neutral).rename(columns={"pass_rate": "neutral_pass_rate"})
    out = neu_rate.merge(rz_rate, on=["season", "posteam"], how="inner").rename(columns={"posteam": "team"})
    out["rz_pass_rate"] *= 100
    out["neutral_pass_rate"] *= 100
    out["shift"] = out["neutral_pass_rate"] - out["rz_pass_rate"]
    return out


def player_redzone_role(pbp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Red zone touches and touchdowns, per player-season, from raw plays.

    Box scores give season touchdown totals but nothing about *where* they came
    from — a running back who scored on 40% of his red zone carries and one who
    scored on 15% can post an identical touchdown line if the second one just got
    more chances near the goal line. That distinction is what the touchdown-luck
    regression in model.project needs: 'touches' near the end zone and how many of
    them actually became scores, so a player's rate can be compared to what his
    position converts on average.

    A touch is a carry (rusher_player_id on a run play) or a target (receiver_player_id
    on a pass play), matching how touches are already counted for workload burden in
    injury_risk. Red zone = yardline_100 <= 20. Receiving touchdowns require the pass
    to be complete — an incomplete red zone target is a real look that just didn't
    convert, not a non-event, so it still counts as a touch with rz_td=0.
    """
    if pbp is None:
        return _memo("redzone_role", lambda: _player_redzone_role(sources.play_by_play()))
    return _player_redzone_role(pbp)


def _player_redzone_role(pbp: pd.DataFrame) -> pd.DataFrame:
    rz = pbp[pbp["yardline_100"].notna() & (pbp["yardline_100"] <= 20)]

    runs = rz[(rz["play_type"] == "run") & rz["rusher_player_id"].notna()]
    rush_g = runs.groupby(["season", "rusher_player_id"], observed=True).agg(
        rz_rush_att=("rush_touchdown", "size"),
        rz_rush_td=("rush_touchdown", "sum"),
    ).reset_index().rename(columns={"rusher_player_id": "player_id"})

    passes = rz[(rz["play_type"] == "pass") & rz["receiver_player_id"].notna()]
    rec_g = passes.groupby(["season", "receiver_player_id"], observed=True).agg(
        rz_targets=("pass_touchdown", "size"),
        rz_rec_td=("pass_touchdown", "sum"),
    ).reset_index().rename(columns={"receiver_player_id": "player_id"})

    out = rush_g.merge(rec_g, on=["season", "player_id"], how="outer")
    for c in ("rz_rush_att", "rz_rush_td", "rz_targets", "rz_rec_td"):
        out[c] = out[c].fillna(0.0)
    out["rz_touches"] = out["rz_rush_att"] + out["rz_targets"]
    out["rz_td"] = out["rz_rush_td"] + out["rz_rec_td"]
    return out[["season", "player_id", "rz_touches", "rz_td"]]


def strength_of_schedule(target_season: int, defense: pd.DataFrame) -> pd.DataFrame:
    key = f"sos_{target_season}_{len(defense)}"
    if key in _DERIVED:
        return _DERIVED[key]
    out = _strength_of_schedule(target_season, defense)
    _DERIVED[key] = out
    return out


def _strength_of_schedule(target_season: int, defense: pd.DataFrame) -> pd.DataFrame:
    """Per-team schedule difficulty for the upcoming season, by position defended.

    Opponent quality is a recency-weighted blend of that defense's last few years,
    which is far more stable than last season alone. Divisional games are counted
    separately because you face those six defenses twice — they carry double weight
    and they're the matchups you can least escape.
    """
    sched = sources.schedules()
    games = sched[sched["season"] == target_season]
    if games.empty:  # schedule not published yet — fall back to last known season
        target_season = int(sched["season"].max())
        games = sched[sched["season"] == target_season]

    # Recency-weighted defensive profile per team.
    hist = defense[defense["season"] < target_season].copy()
    if hist.empty:
        hist = defense.copy()
    wmap = _season_weights(sorted(hist["season"].unique()))
    hist["w"] = hist["season"].map(wmap).fillna(0)
    val_cols = [c for c in hist.columns if c.startswith("fpa_") and not c.endswith("_rank")]
    val_cols += ["def_epa_play"]
    h = hist.copy()
    num = {}
    for c in val_cols:
        h[f"_n_{c}"] = h[c].fillna(h[c].mean()) * h["w"]
        num[f"_n_{c}"] = "sum"
    h["_w"] = h["w"]
    num["_w"] = "sum"
    agg = h.groupby("team", observed=True).agg(num)
    prof = pd.DataFrame({"team": agg.index})
    for c in val_cols:
        prof[c] = (agg[f"_n_{c}"] / agg["_w"].replace(0, np.nan)).to_numpy()

    rows = []
    for _, gm in games.iterrows():
        rows.append({"team": gm["home_team"], "opp": gm["away_team"], "div": gm.get("div_game", 0)})
        rows.append({"team": gm["away_team"], "opp": gm["home_team"], "div": gm.get("div_game", 0)})
    sos = pd.DataFrame(rows).merge(prof, left_on="opp", right_on="team",
                                   how="left", suffixes=("", "_o")).drop(columns=["team_o"])

    agg = {c: "mean" for c in val_cols}
    agg["div"] = "sum"
    out = sos.groupby("team").agg(agg).reset_index()
    out = out.rename(columns={c: f"sos_{c}" for c in val_cols})
    out = out.rename(columns={"div": "divisional_games"})
    out["season"] = target_season
    # Positive = easier schedule (opponents allow more fantasy points).
    for pos in FANTASY_POSITIONS:
        c = f"sos_fpa_{pos}"
        if c in out.columns:
            out[f"sos_{pos}_z"] = _zscore(out[c])
    return out


def bye_weeks(season: int | None = None) -> dict[str, int]:
    """Each team's bye week for `season`, derived from the schedule rather than a
    separate feed -- a team's bye is simply the week in that season's regular-season
    span where it shows up in neither `home_team` nor `away_team`.

    Empty dict if that season's schedule isn't published yet (byes aren't set until
    the full slate drops) or doesn't parse into exactly one missing week per team --
    a "nice if available" signal, same posture as depth_charts, not a hard
    dependency, so recommend()'s bye-collision check just sees "no bye" and skips it.
    """
    from .config import CURRENT_SEASON

    season = season or CURRENT_SEASON
    sched = sources.schedules()
    reg = sched[(sched["season"] == season) & (sched["game_type"] == "REG")]
    if reg.empty:
        return {}

    teams = pd.unique(pd.concat([reg["home_team"], reg["away_team"]]))
    weeks = sorted(reg["week"].unique())
    out = {}
    for team in teams:
        played = set(reg.loc[(reg["home_team"] == team) | (reg["away_team"] == team), "week"])
        missing = [w for w in weeks if w not in played]
        if len(missing) == 1:
            out[str(team)] = int(missing[0])
    return out


def head_coaches(season: int | None = None) -> dict[str, str]:
    """Each team's head coach entering `season`, from the schedule feed's
    home_coach/away_coach columns (the same feed strength_of_schedule already
    reads) -- specifically the Week 1 coach, since that's who a drafter actually
    knows about before the season starts. A team that fires its coach mid-season
    still reads as having entered the season with the Week 1 hire; that in-season
    firing is exactly the kind of event a preseason draft decision couldn't have
    used anyway.

    Empty dict if that season's schedule isn't published yet.
    """
    from .config import CURRENT_SEASON

    season = season or CURRENT_SEASON
    sched = sources.schedules()
    reg = sched[(sched["season"] == season) & (sched["game_type"] == "REG")]
    if reg.empty:
        return {}
    week1 = reg[reg["week"] == reg["week"].min()]
    out: dict[str, str] = {}
    for _, g in week1.iterrows():
        if pd.notna(g.get("home_coach")):
            out[g["home_team"]] = g["home_coach"]
        if pd.notna(g.get("away_coach")):
            out[g["away_team"]] = g["away_coach"]
    return out


def head_coach_changes(season: int) -> dict[str, bool]:
    """Which teams enter `season` with a different Week 1 head coach than they had
    entering `season - 1` -- a real discontinuity that breaks the "recency-weighted
    blend of past seasons" assumption every other team-level feature in this module
    relies on (oline_ratings, team_pace_and_split, redzone_identity_shift, schedule
    difficulty): a new coach can install a different scheme entirely, making last
    year's tendencies actively misleading rather than just noisy.

    Only flags head coach turnover -- offensive coordinator changes matter at
    least as much for offensive scheme, and adp.hc_change_qb_efficiency_volatility
    found this specifically doesn't predict QB efficiency volatility the way a
    real OC change does (most HC hires leave the play-caller untouched). See
    offensive_coordinator_changes for the OC-specific version, sourced from a
    static compiled dataset since there's no live structured feed for OC hires
    the way there is for head coaches.

    A team with no known prior-season coach (first year in the data, or a
    relocated/renamed franchise) reads as no change -- there's nothing to compare
    against, so guessing "changed" would be as unfounded as guessing "same."
    """
    prior = head_coaches(season - 1)
    current = head_coaches(season)
    return {team: (prior.get(team) is not None and coach != prior.get(team))
           for team, coach in current.items()}


_OC_CHANGE_HISTORY_PATH = Path(__file__).resolve().parent.parent.parent / "docs" / "oc_change_history.csv"


def offensive_coordinator_changes(season: int) -> dict[str, bool]:
    """Which teams enter `season` with a different offensive coordinator (real
    play-caller change, not just head-coach turnover) than the year before.

    Deliberate exception to this module's own rule, stated in its docstring:
    everything else here is computed from raw plays/box scores precisely so it
    can't go stale or need re-scraping. This can't be -- there is no structured,
    dynamically-fetchable feed for OC hires/fires the way nflverse's schedule
    feed carries head coaches (see head_coach_changes). docs/oc_change_history.csv
    is a static, versioned, manually-compiled artifact instead: 2021-2025 sourced
    from each team's own Wikipedia season article (compiled and cross-checked
    team by team), 2026 sourced from 4for4's published coaching-changes hub
    (its real underlying JSON API, not a scraped/summarized page). Silently
    returns {} for any season not in that file, and needs a manual refresh every
    offseason -- there's no way around that without a real structured public
    dataset this module could poll the way it polls nflverse.

    adp.oc_change_qb_efficiency_volatility backtests this against real QB
    efficiency and finds a genuine effect -- unlike head_coach_changes, which
    comes back null for QB efficiency specifically. See that function's
    docstring for the numbers, and the important caveat that the effect is real
    but *undirected* (a new OC can raise or lower a QB's efficiency), so it
    doesn't license a specific score adjustment on its own.

    Also flags whoever holds the OC *title* changing, not necessarily whether
    play-calling itself changed hands (an internal promotion can move the title
    without moving the play-caller) -- see model.attach_oc_change_flag's
    docstring for a real 2026 example this surfaced.
    """
    if not _OC_CHANGE_HISTORY_PATH.exists():
        return {}
    df = pd.read_csv(_OC_CHANGE_HISTORY_PATH)
    df = df[df["season"] == season]
    if df.empty:
        return {}
    return dict(zip(df["team"], df["oc_changed"] == "yes"))


def vacated_role_events(weekly: pd.DataFrame, byes: dict[int, dict[str, int]],
                        positions: tuple[str, ...] = ("WR", "TE"),
                        min_games: int = 6) -> pd.DataFrame:
    """One row per (a teammate's target-share performance in a week his same-team,
    same-position teammate didn't play) -- the generalized version of "if Chase is
    out, how much does Higgins' role actually grow," across every real absence in
    `weekly` instead of one hand-picked pair.

    For each such absence (team T, position group P, player `player_out` missing a
    week that isn't T's bye -- byes have every teammate out too, not a real
    role-vacancy signal), and each teammate who did play that week:
      - `trailing_share`: the teammate's own average target_share in weeks *before*
        this one where `player_out` was active -- the predictor, using only
        information available before the absence, the way a live "who benefits"
        read would have to.
      - `baseline_share`: the teammate's average target_share across every *other*
        week (before or after) `player_out` was active -- the ground truth for
        "what's normal for this teammate," allowed to use future weeks since it's
        establishing a comparison baseline, not predicting anything.
      - `actual_share`: the teammate's target_share in the absence week itself.
      - `share_lift` = actual_share - baseline_share: the real redistribution.

    `min_games` filters to players with a real sample at that position for that
    team-season (a one-game cameo isn't a "teammate" in any usable sense). Needs a
    trailing sample *and* an "other weeks" sample to compute both columns, so a
    player's very first eligible absence of the season (no prior in-weeks yet) is
    dropped rather than left half-populated.
    """
    w = weekly[weekly["position"].isin(positions)].dropna(subset=["target_share"]).copy()
    rows = []
    for season, s in w.groupby("season"):
        bye_map = byes.get(season, {})
        for team, t in s.groupby("recent_team"):
            team_weeks = sorted(t["week"].unique())
            bye = bye_map.get(team)
            for pos, p in t.groupby("position"):
                counts = p.groupby("player_display_name")["week"].nunique()
                roster = counts[counts >= min_games].index.tolist()
                if len(roster) < 2:
                    continue
                for player_out in roster:
                    out_weeks = set(p.loc[p["player_display_name"] == player_out, "week"])
                    missed = [wk for wk in team_weeks if wk not in out_weeks and wk != bye]
                    for wk in missed:
                        for mate in (m for m in roster if m != player_out):
                            mate_rows = p[p["player_display_name"] == mate]
                            prior_in = mate_rows[(mate_rows["week"] < wk) &
                                                 (mate_rows["week"].isin(out_weeks))]
                            this_wk = mate_rows[mate_rows["week"] == wk]
                            other_in = mate_rows[(mate_rows["week"] != wk) &
                                                 (mate_rows["week"].isin(out_weeks))]
                            if prior_in.empty or this_wk.empty or other_in.empty:
                                continue
                            rows.append({
                                "season": season, "team": team, "position": pos,
                                "week": int(wk), "player_out": player_out, "teammate": mate,
                                "trailing_share": float(prior_in["target_share"].mean()),
                                "actual_share": float(this_wk["target_share"].iloc[0]),
                                "baseline_share": float(other_in["target_share"].mean()),
                            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["share_lift"] = out["actual_share"] - out["baseline_share"]
    return out


# ---------------------------------------------------------------- player level

def player_season_profiles(sc: Scoring, te_bonus: float = 0.0, seasons=None) -> pd.DataFrame:
    """Per-player, per-season production, role and week-to-week consistency.

    seasons, when given, bounds this to exactly those seasons instead of every
    season in the cache -- what build_player_table uses to build a leak-free
    board for a past draft, so a backtest can't see the season it's predicting.
    """
    key = ("profiles_" + ",".join(f"{k}={v}" for k, v in sorted(vars(sc).items())) +
           f"|{te_bonus}|{tuple(seasons) if seasons else 'all'}")
    if key in _DERIVED:
        return _DERIVED[key]
    out = _player_season_profiles(sc, te_bonus, seasons)
    _DERIVED[key] = out
    return out


def _player_season_profiles(sc: Scoring, te_bonus: float = 0.0, seasons=None) -> pd.DataFrame:
    w = sources.weekly_stats(seasons)
    w = w[w["position"].isin(FANTASY_POSITIONS) & (w["season_type"] == "REG")].copy()
    w["fp"] = fantasy_points(w, sc, te_bonus)

    thresh = w["position"].map(STARTABLE_THRESHOLD)
    spike = w["position"].map(SPIKE_THRESHOLD)
    w["startable"] = (w["fp"] >= thresh).astype(float)
    w["spike"] = (w["fp"] >= spike).astype(float)
    w["touches"] = w.get("carries", 0).fillna(0) + w.get("targets", 0).fillna(0)

    prof = w.groupby(["player_id", "player_display_name", "position", "season"]).agg(
        games=("week", "nunique"),
        fp_total=("fp", "sum"),
        fp_mean=("fp", "mean"),
        fp_sd=("fp", "std"),
        fp_median=("fp", "median"),
        startable_rate=("startable", "mean"),
        spike_rate=("spike", "mean"),
        touches=("touches", "sum"),
        targets=("targets", "sum"),
        rec_per_game=("receptions", "mean"),
        carries=("carries", "sum"),
        target_share=("target_share", "mean"),
        team=("recent_team", "last"),
    ).reset_index()

    # Floor: mean of the worst 40% of weeks. Ceiling: mean of the best 20%.
    # Computed with ranks rather than sorting each player-season separately, which
    # meant thousands of small sorts for a result one grouped pass can produce.
    ws = w[["player_id", "season", "fp"]].copy()
    grp = ws.groupby(["player_id", "season"], observed=True)["fp"]
    n = grp.transform("size")
    pct = grp.rank(method="first", pct=True)
    ws["is_floor"] = pct <= 0.4
    ws["is_ceiling"] = pct > 0.8
    # Single-game seasons have no meaningful tail, so keep the one value in both.
    ws.loc[n <= 1, ["is_floor", "is_ceiling"]] = True
    # Mask into separate columns and take grouped means. An apply over the groups
    # does the same arithmetic but pays Python-level indexing on every one of them,
    # which dominated the whole build.
    ws["fp_floor"] = ws["fp"].where(ws["is_floor"])
    ws["fp_ceiling"] = ws["fp"].where(ws["is_ceiling"])
    tails = ws.groupby(["player_id", "season"], observed=True).agg(
        floor=("fp_floor", "mean"),
        ceiling=("fp_ceiling", "mean"),
    ).reset_index()
    prof = prof.merge(tails, on=["player_id", "season"], how="left")

    # Snap share — the cleanest read on role, independent of touchdown luck.
    try:
        snaps = sources.snap_counts()
        snaps = snaps[snaps["game_type"] == "REG"]
        sn = snaps.groupby(["player", "season"])["offense_pct"].mean().reset_index()
        prof = prof.merge(sn, left_on=["player_display_name", "season"],
                          right_on=["player", "season"], how="left").drop(columns=["player"])
        prof = prof.rename(columns={"offense_pct": "snap_share"})
    except Exception:
        prof["snap_share"] = np.nan

    prof["fp_cv"] = prof["fp_sd"] / prof["fp_mean"].replace(0, np.nan)
    return prof


def injury_risk(profiles: pd.DataFrame) -> pd.DataFrame:
    """Likelihood of missing time, from availability history plus workload burden.

    Three inputs, because each catches something the others miss:
      1. games missed relative to a 17-game season, recency-weighted
      2. how often they showed up on injury reports even when they played
      3. workload burden — seasons of heavy touch volume, which is a
         well-documented leading indicator for backs especially
    """
    inj = sources.injuries()
    rosters = sources.weekly_rosters()

    wmap = _season_weights(sorted(profiles["season"].unique()))
    p = profiles.copy()
    p["w"] = p["season"].map(wmap).fillna(0)

    # 1. Availability, recency-weighted. Done as vectorised sums rather than a
    # per-player apply: the apply recomputed the season-weight table once for every
    # player on the board, which was the single slowest step in the whole build.
    p["avail"] = (p["games"] / 17).clip(0, 1)
    p["avail_w"] = p["avail"] * p["w"]
    g = p.groupby(["player_id", "position"], observed=True).agg(
        seasons=("season", "nunique"),
        _num=("avail_w", "sum"),
        _den=("w", "sum"),
    ).reset_index()
    g["games_missed_rate"] = 1 - (g["_num"] / g["_den"].replace(0, np.nan))
    avail = g.drop(columns=["_num", "_den"])

    # 2. Injury-report burden: weeks listed with any game-status designation.
    if not inj.empty:
        inj = inj.copy()
        inj["flagged"] = inj["report_status"].notna().astype(float)
        rep = inj.groupby("gsis_id").agg(
            report_weeks=("flagged", "sum"),
            report_seasons=("season", "nunique"),
        ).reset_index()
        rep["report_rate"] = (rep["report_weeks"] / (rep["report_seasons"] * 17)).clip(0, 1)
        avail = avail.merge(rep[["gsis_id", "report_rate"]], left_on="player_id",
                            right_on="gsis_id", how="left").drop(columns=["gsis_id"])
    avail["report_rate"] = avail.get("report_rate", pd.Series(dtype=float)).fillna(0.15)

    # 3. Workload burden: seasons above the positional touch threshold, recency-weighted.
    p["burden_line"] = p["position"].map(WORKLOAD_BURDEN)
    p["over_burden"] = (p["touches"] > p["burden_line"]).astype(float)
    p["burden_ratio"] = (p["touches"] / p["burden_line"]).clip(0, 2)
    p["burden_w"] = p["burden_ratio"] * p["w"]
    bg = p.groupby("player_id", observed=True).agg(
        heavy_seasons=("over_burden", "sum"),
        _bnum=("burden_w", "sum"),
        _bden=("w", "sum"),
    ).reset_index()
    bg["recent_burden"] = bg["_bnum"] / bg["_bden"].replace(0, np.nan)
    burden = bg.drop(columns=["_bnum", "_bden"])
    avail = avail.merge(burden, on="player_id", how="left")

    # Age, needed for the RB-specific burden interaction.
    ages = rosters.sort_values("week").groupby("gsis_id").agg(
        birth_date=("birth_date", "last"), season=("season", "max"),
    ).reset_index()
    ages["age"] = (pd.to_datetime(f"{int(ages['season'].max())}-09-01")
                   - pd.to_datetime(ages["birth_date"], errors="coerce")).dt.days / 365.25
    avail = avail.merge(ages[["gsis_id", "age"]], left_on="player_id",
                        right_on="gsis_id", how="left").drop(columns=["gsis_id"])

    # Blend into a single 0-1 risk score. Backs carrying heavy volume past the age
    # cliff get an extra penalty; that combination is where seasons go to die.
    base = {"RB": 0.30, "WR": 0.20, "TE": 0.22, "QB": 0.16}
    avail["pos_base"] = avail["position"].map(base).fillna(0.20)
    cliff = avail["position"].map(AGE_CLIFF).fillna(30)
    age_excess = (avail["age"].fillna(26) - cliff).clip(lower=0)

    avail["injury_risk"] = (
        0.35 * avail["pos_base"]
        + 0.30 * avail["games_missed_rate"].fillna(0.1).clip(0, 1)
        + 0.15 * avail["report_rate"].clip(0, 1)
        + 0.12 * (avail["recent_burden"].fillna(0.5) / 2).clip(0, 1)
        + 0.08 * (age_excess * avail["recent_burden"].fillna(0.5) / 4).clip(0, 1)
    ).clip(0.02, 0.85)

    return avail[["player_id", "position", "age", "games_missed_rate", "report_rate",
                  "heavy_seasons", "recent_burden", "injury_risk"]]


def age_adjustment(position: str, age: float | None) -> float:
    """Multiplier from the positional aging curve."""
    if age is None or not np.isfinite(age):
        return 1.0
    cliff = AGE_CLIFF.get(position, 30)
    decay = AGE_DECAY.get(position, 0.05)
    if age <= cliff:
        # Young players still ascending, mostly a WR/TE effect.
        return 1.0 + min(0.06, max(0.0, (cliff - age) * 0.012))
    return float(max(0.55, 1.0 - decay * (age - cliff)))
