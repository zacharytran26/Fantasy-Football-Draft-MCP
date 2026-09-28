"""The draft model.

Pipeline: baseline projection -> environment adjustments -> consistency blend ->
value over replacement -> pick-aware urgency.

The last step is the one that actually wins drafts. Knowing a player is good is easy;
knowing whether he'll still be there when you pick again is what decides who to take now.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import features, sources
from .config import CURRENT_SEASON, FANTASY_POSITIONS, LeagueSettings, ModelWeights


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def touchdown_luck_multiplier(rz_touches: pd.Series, rz_td: pd.Series,
                              baseline_rate: pd.Series, weight: float,
                              min_touches: int = 8) -> pd.Series:
    """How far a player's red zone conversion rate sits from his position's baseline,
    expressed as a bounded multiplier on projected production.

    A player who scored on far more of his red zone touches than his position
    converts on average gets pulled down; one who scored on far fewer gets pulled up.
    Below min_touches a rate is mostly noise, so those players sit neutral (1.0)
    rather than swinging on a two- or three-touch sample.

    weight is the maximum fractional move in either direction, same convention as
    every other environment multiplier in project() -- z-scored, clipped to +/-2.5,
    then scaled by weight so this factor can't dominate the projection on its own.
    """
    touches = rz_touches.fillna(0.0)
    td = rz_td.fillna(0.0)
    qualifies = touches >= min_touches
    expected = touches * baseline_rate.fillna(0.0)
    surplus = td - expected

    # Z-score over the qualifying population only, so the non-qualifying players
    # pinned to neutral below don't drag the scale everyone else is measured on.
    pool = surplus.where(qualifies)
    sd = pool.std(ddof=0)
    if sd and sd > 0:
        z = (surplus - pool.mean()) / sd
    else:
        z = pd.Series(0.0, index=surplus.index)
    # Sign-flipped: a positive surplus (overperformed) should push the multiplier
    # below 1, a negative surplus (underperformed) should push it above 1.
    z = (-z).clip(-2.5, 2.5) / 2.5
    mult = 1 + z * weight
    return mult.where(qualifies, 1.0)


def apply_current_team(tbl: pd.DataFrame, depth_chart: pd.DataFrame) -> pd.DataFrame:
    """Override each player's team with the current depth chart, when available.

    `tbl["team"]` (built from weekly box scores) reflects whichever team a player
    last actually played a game for, which can be a full offseason stale by draft
    day -- trades, cuts and free-agent signings don't show up until Week 1 snaps
    get recorded. Depth charts are filed by the teams themselves, so they catch a
    move as soon as it's reported. A player missing from the depth chart (e.g. a
    rookie before training camp, or if this season's chart isn't out yet) just
    keeps his last known team -- this is a correction, not a hard requirement.

    Also flags `off_roster`: True for a player absent from a *populated* depth
    chart, meaning every team has filed one and this player is on none of them --
    released, retired mid-cycle, or otherwise not currently rosterable, as
    opposed to a rookie or a player whose team simply hasn't reported yet (which
    an empty depth_chart already covers by skipping the flag entirely below). A
    player like this can still carry a strong projection from last season's real
    production even though he has no path to touches this season; project()
    discounts him for it the same way it discounts a stale season.

    Must run before the O-line / pace / schedule merges below, which key off
    `team`: otherwise a traded player would get graded on his old team's offense.

    Also carries over `depth_rank` (1 = starter, per the official chart) when the
    feed has it -- roster_construction_mult's handcuff check prefers this over
    inferring "backup" from a lower draft_score, since two similarly-projected
    players can be a real committee (neither is really the other's insurance)
    while a clearly-labelled RB2 is a real handcuff even on the rare occasion his
    own draft_score isn't the lower of the two.
    """
    if depth_chart is None or depth_chart.empty or "player_id" not in tbl.columns:
        tbl = tbl.copy()
        tbl["off_roster"] = False
        if "depth_rank" not in tbl.columns:
            tbl["depth_rank"] = np.nan
        return tbl
    cols = ["player_id", "team"] + (["depth_rank"] if "depth_rank" in depth_chart.columns else [])
    out = tbl.merge(
        depth_chart[cols].rename(columns={"team": "_current_team"}),
        on="player_id", how="left",
    )
    has_current = out["_current_team"].notna()
    out.loc[has_current, "team"] = out.loc[has_current, "_current_team"]
    out["off_roster"] = ~has_current
    if "depth_rank" not in out.columns:
        out["depth_rank"] = np.nan
    return out.drop(columns=["_current_team"])


def _season_weighted(profiles: pd.DataFrame, col: str) -> pd.Series:
    wmap = features._season_weights(sorted(profiles["season"].unique()))
    p = profiles.copy()
    p["w"] = p["season"].map(wmap).fillna(0)
    p["wv"] = p[col].fillna(0) * p["w"]
    agg = p.groupby("player_id").agg(num=("wv", "sum"), den=("w", "sum"))
    return (agg["num"] / agg["den"].replace(0, np.nan)).rename(col)


def build_player_table(league: LeagueSettings, weights: ModelWeights,
                       season: int = CURRENT_SEASON) -> pd.DataFrame:
    """Assemble every modelled feature into one row per player.

    Every history-derived input (production, O-line, pace, defense, separation) is
    bounded to the five seasons strictly before `season` -- the same lookback the
    live board uses by construction, since weekly_stats/play_by_play don't yet have
    a season that hasn't happened. Passing a past `season` (e.g. 2025 to backtest a
    2025 draft) makes that bound explicit: nothing from `season` or later leaks in,
    so this is what draft_backtest uses to see only what a drafter could have known
    on that draft day. The live depth-chart team override is skipped for a past
    season too -- it reflects today's rosters, not that season's.
    """
    sc = league.scoring
    lookback = list(range(season - 5, season))
    profiles = features.player_season_profiles(sc, league.te_premium_bonus, seasons=lookback)

    # --- recency-weighted production and reliability
    agg = pd.concat([
        _season_weighted(profiles, "fp_mean"),
        _season_weighted(profiles, "startable_rate"),
        _season_weighted(profiles, "spike_rate"),
        _season_weighted(profiles, "floor"),
        _season_weighted(profiles, "ceiling"),
        _season_weighted(profiles, "fp_cv"),
        _season_weighted(profiles, "snap_share"),
        _season_weighted(profiles, "touches"),
        _season_weighted(profiles, "target_share"),
        _season_weighted(profiles, "rec_per_game"),
    ], axis=1).reset_index()

    latest = profiles.sort_values("season").groupby("player_id").agg(
        name=("player_display_name", "last"),
        position=("position", "last"),
        team=("team", "last"),
        last_season=("season", "max"),
        seasons_played=("season", "nunique"),
        games_last=("games", "last"),
    ).reset_index()
    tbl = latest.merge(agg, on="player_id", how="left")

    # Keep players who were active recently AND held a real offensive role. Without
    # the role filter the board fills with special-teamers and emergency starters
    # whose tiny samples produce noisy, flattering rates.
    tbl = tbl[tbl["last_season"] >= profiles["season"].max() - 1]
    role = (
        (tbl["games_last"].fillna(0) >= 6)
        | (tbl["snap_share"].fillna(0) >= 0.40)
        | (tbl["touches"].fillna(0) >= 45)
    )
    tbl = tbl[role].reset_index(drop=True)

    # --- injury & age
    risk = features.injury_risk(profiles)
    tbl = tbl.merge(risk.drop(columns=["position"]), on="player_id", how="left")
    tbl["injury_risk"] = tbl["injury_risk"].fillna(0.25)
    tbl["age"] = tbl["age"].fillna(26.0) + (season - profiles["season"].max())

    # --- team environment
    if season >= CURRENT_SEASON:
        tbl = apply_current_team(tbl, sources.depth_charts())
    pbp = sources.play_by_play(seasons=lookback)
    ol = features.oline_ratings(pbp)
    pace = features.team_pace_and_split(pbp)
    dfn = features.defense_ratings(pbp, sources.weekly_stats(lookback), sc=sc)
    sos = features.strength_of_schedule(season, dfn)

    recent = int(pace["season"].max())
    ol_r = ol[ol["season"] == recent][["team", "run_block_z", "pass_block_z",
                                       "run_block_rank", "pass_block_rank"]]
    pace_r = pace[pace["season"] == recent][["team", "plays_per_game", "pass_rate",
                                             "rush_rate", "neutral_pass_rate"]]
    tbl = tbl.merge(ol_r, on="team", how="left").merge(pace_r, on="team", how="left")
    sos_cols = ["team", "divisional_games"] + [c for c in sos.columns if c.endswith("_z")]
    tbl = tbl.merge(sos[sos_cols], on="team", how="left")

    # --- red zone role, for the touchdown-luck regression in project()
    # Recency-weighted the same way as touches/target_share: every (player, season)
    # the player had *any* offensive role gets a row, filling zero red zone touches
    # rather than dropping the season, so a player's weighting window matches his
    # other features exactly instead of narrowing to just his red-zone seasons.
    rz = profiles[["player_id", "season"]].drop_duplicates().merge(
        features.player_redzone_role(pbp), on=["player_id", "season"], how="left")
    rz["rz_touches"] = rz["rz_touches"].fillna(0.0)
    rz["rz_td"] = rz["rz_td"].fillna(0.0)
    rz_w = pd.concat([
        _season_weighted(rz, "rz_touches"), _season_weighted(rz, "rz_td"),
    ], axis=1).reset_index()
    tbl = tbl.merge(rz_w, on="player_id", how="left")

    # --- separation and route efficiency (pass catchers only)
    try:
        from . import separation as sep_mod
        sep = sep_mod.separation_summary(seasons=lookback)
        if not sep.empty:
            tbl = tbl.merge(
                sep[["player_id", "sep_score", "avg_separation", "avg_cushion",
                     "yprr", "tprr", "yac_oe", "adot", "seasons_qualified"]],
                on="player_id", how="left",
            )
    except Exception as exc:
        print(f"separation data unavailable ({type(exc).__name__})")
    for c in ("sep_score", "yprr", "tprr", "avg_separation", "adot"):
        if c not in tbl.columns:
            tbl[c] = np.nan
    tbl["is_rookie"] = False

    # --- rookies, projected from draft capital rather than history
    try:
        from . import rookies as rk
        curves = rk.fit_draft_curves(sc, seasons=list(range(season - rk.FIT_SEASONS, season)))
        rb = rk.rookie_board(season, sc, curves)
        if not rb.empty:
            rb["rookie_consistency"] = [
                rk.rookie_consistency_prior(p, k, curves)
                for p, k in zip(rb["position"], rb["pick"])
            ]
            # Rookies get the same landing-spot context veterans do.
            rb = rb.merge(ol_r, on="team", how="left").merge(pace_r, on="team", how="left")
            rb = rb.merge(sos[sos_cols], on="team", how="left")
            # Drop anyone already modelled from real production (returning players
            # occasionally appear in a later draft class through supplemental entry).
            rb = rb[~rb["player_id"].isin(tbl["player_id"])]
            tbl = pd.concat([tbl, rb], ignore_index=True)
            tbl["is_rookie"] = tbl["is_rookie"].fillna(True)
    except Exception as exc:
        print(f"rookie projections unavailable ({type(exc).__name__}: {exc})")

    # Rookies join after the veteran fills, so backfill anything the scorer needs.
    # A single NaN here propagates through every multiplier and voids the whole rank.
    defaults = {
        "injury_risk": 0.22, "games_missed_rate": 0.10, "report_rate": 0.12,
        "heavy_seasons": 0.0, "recent_burden": 0.5, "seasons_played": 0,
        "games_last": 0, "age": 22.0, "startable_rate": np.nan, "fp_mean": np.nan,
        "divisional_games": 6.0, "rz_touches": 0.0, "rz_td": 0.0,
        "off_roster": False,
    }
    for col, val in defaults.items():
        if col in tbl.columns:
            tbl[col] = tbl[col].fillna(val)
        else:
            tbl[col] = val
    return tbl


def project(tbl: pd.DataFrame, league: LeagueSettings, weights: ModelWeights) -> pd.DataFrame:
    """Turn features into a projection, a reliability score, and a draft value."""
    t = tbl.copy()
    w = weights

    # ---- baseline: recency-weighted PPG, regressed toward the mean for small
    # samples so a two-game hot streak doesn't outrank a proven starter.
    #
    # The regression target has to be the mean of *starter-caliber* players at the
    # position, not the mean of everyone who logged a snap. Including third-string
    # backs drags the target down so far that regressing toward it would cut a
    # genuine RB1 by a third. We take the top N by weighted PPG as the comparison set.
    starter_n = {"QB": 24, "RB": 36, "WR": 48, "TE": 20}
    pos_target = {}
    for pos, chunk in t.groupby("position"):
        ranked = chunk.sort_values("fp_mean", ascending=False)
        n = min(starter_n.get(pos, 30), max(3, len(ranked)))
        pos_target[pos] = float(ranked["fp_mean"].head(n).mean())
    t["pos_target"] = t["position"].map(pos_target)

    # Shrink on games played, not seasons: 17 healthy games is a real sample,
    # three cameo appearances across two years is not.
    # Rookies arrive with a baseline already fitted from draft capital, so stash it
    # before the veteran regression runs. Letting that regression touch them would
    # replace the fitted value with a positional average and erase the whole
    # distinction between a top-five pick and a sixth-rounder.
    is_rook = t.get("is_rookie", pd.Series(False, index=t.index)).fillna(False).astype(bool)
    rookie_baseline = t["baseline_ppg"].copy() if "baseline_ppg" in t.columns else None

    games = t["games_last"].fillna(0) + 8 * (t["seasons_played"].fillna(1) - 1).clip(lower=0)
    shrink = (games / (games + 10)).clip(0.15, 0.93)
    t["baseline_ppg"] = t["fp_mean"].fillna(t["pos_target"]) * shrink + t["pos_target"] * (1 - shrink)

    if rookie_baseline is not None and is_rook.any():
        t.loc[is_rook, "baseline_ppg"] = rookie_baseline[is_rook]
        # Rookie availability is already capital-scaled; don't shrink them like vets.
        shrink = shrink.mask(is_rook, 0.60)

    # A veteran who hasn't played as recently as the board's freshest players is a
    # real unknown -- retired, hurt long-term, or just out of the league -- and last
    # season's box score can't tell us which. Discounted hard (60% per season stale,
    # compounding) rather than left at face value: without this, a two-seasons-
    # retired running back's still-strong 2020 form outprojected most of a real
    # board in a 2022 backtest and became the model's runaway top recommendation.
    # Rookies are unaffected (NaN last_season -> stale 0, and they're already
    # capital-scaled above, not derived from fp_mean).
    stale = (t["last_season"].max() - t["last_season"]).clip(lower=0).fillna(0)
    t["baseline_ppg"] = t["baseline_ppg"] * (0.4 ** stale)

    # A player absent from every team's current depth chart produced last season's
    # numbers for a roster he's no longer on -- cut, retired mid-cycle, or simply
    # without a team right now. last_season alone won't catch this (he may well
    # have played a full slate as recently as anyone else on the board), so it's a
    # separate check from the staleness above. Same 0.4x the model already uses for
    # one stale season: severe enough that a genuine value signal (real ADP already
    # crashed for the same reason) doesn't get doubled by an inflated projection on
    # top of it, without fully zeroing a talent who could still land a role.
    off_roster = t.get("off_roster", pd.Series(False, index=t.index)).fillna(False).astype(bool)
    t.loc[off_roster & ~is_rook, "baseline_ppg"] *= 0.4

    # ---- environment multipliers, each bounded by its configured weight
    def bounded(series: pd.Series, weight: float) -> pd.Series:
        z = series.fillna(0).clip(-2.5, 2.5) / 2.5
        return 1 + z * weight

    is_rb = t["position"].eq("RB")
    block_z = np.where(is_rb, t["run_block_z"].fillna(0), t["pass_block_z"].fillna(0))
    t["m_oline"] = bounded(pd.Series(block_z, index=t.index), w.oline)

    # Volume: pace helps everyone; run/pass split helps the side of the ball you're on.
    pace_z = features._zscore(t["plays_per_game"].fillna(t["plays_per_game"].mean()))
    split = np.where(is_rb, t["rush_rate"].fillna(0.43), t["neutral_pass_rate"].fillna(0.57))
    split_z = features._zscore(pd.Series(split, index=t.index))
    t["m_volume"] = bounded((pace_z + split_z) / 2, w.pace_volume)

    # Schedule: positive z = opponents allow more points to this position.
    sos_z = pd.Series(0.0, index=t.index)
    for pos in FANTASY_POSITIONS:
        col = f"sos_{pos}_z"
        if col in t.columns:
            sos_z = sos_z.where(t["position"] != pos, t[col].fillna(0))
    t["m_schedule"] = bounded(sos_z, w.schedule)

    # Divisional games: six games against defenses that game-plan for you specifically.
    # Modelled as a small drag that scales with how tough those division defenses are.
    div = t["divisional_games"].fillna(6)
    t["m_divisional"] = 1 - w.divisional * ((div - 6) / 6 + 0.5) * (-sos_z.fillna(0).clip(-2, 2) / 2)

    # Injury: expected games available out of 17. The risk score is a relative
    # ranking, not a literal miss probability, so it's scaled before converting.
    t["exp_games"] = (17 * (1 - t["injury_risk"] * 0.62)).clip(7, 17)
    if is_rook.any() and "exp_games" in tbl.columns:
        # Rookie availability comes from draft capital, not injury history they
        # don't have yet.
        t.loc[is_rook, "exp_games"] = tbl["exp_games"][is_rook].values
        t.loc[is_rook, "m_injury"] = 1.0
    t["m_injury"] = 1 - (t["injury_risk"] - 0.22).clip(-0.2, 0.6) * w.injury

    t["m_age"] = [features.age_adjustment(p, a) for p, a in zip(t["position"], t["age"])]

    # Separation and route efficiency, for pass catchers only. This is the signal
    # that distinguishes a receiver whose production came from genuine route-winning
    # from one riding target volume he may not keep. It only applies to players who
    # qualified on route count — everyone else sits at neutral.
    sep_z = pd.Series(0.0, index=t.index)
    if "sep_score" in t.columns:
        catchers = t["position"].isin(["WR", "TE"]) & t["sep_score"].notna()
        if catchers.any():
            sep_z.loc[catchers] = t.loc[catchers, "sep_score"].clip(-2.5, 2.5)
    t["m_separation"] = bounded(sep_z, w.separation)

    # Coverage-scheme trend, opt-in via w.coverage_trend (defaults to 0 -- see
    # ModelWeights.coverage_trend docstring for why this stays off by default).
    # For WR/TE: reward a short-area profile (high TPRR, low aDOT) over a
    # boundary/vertical one, since that's the archetype that beats zone coverage
    # and draws linebackers rather than nickel corners. For RB: reward receiving
    # role (target_share) directly, since backs already schemed into the passing
    # game are the ones positioned to exploit the same trend.
    ct_z = pd.Series(0.0, index=t.index)
    if w.coverage_trend:
        catchers = t["position"].isin(["WR", "TE"]) & t.get(
            "tprr", pd.Series(np.nan, index=t.index)).notna()
        if catchers.any():
            tprr_z = features._zscore(t.loc[catchers, "tprr"])
            adot = t.loc[catchers, "adot"]
            adot_z = features._zscore(adot.fillna(adot.mean()))
            ct_z.loc[catchers] = ((tprr_z - adot_z) / 2).clip(-2.5, 2.5)
        is_rb = t["position"].eq("RB") & t.get(
            "target_share", pd.Series(np.nan, index=t.index)).notna()
        if is_rb.any():
            ts = t.loc[is_rb, "target_share"]
            ct_z.loc[is_rb] = features._zscore(ts).clip(-2.5, 2.5)
    t["m_coverage_trend"] = bounded(ct_z, w.coverage_trend)

    # Touchdown luck: a player's own red zone conversion rate, regressed toward what
    # his position converts on average. Touchdowns dominate fantasy scoring and are
    # much noisier than yardage — a back who scored on 40% of his red zone carries
    # one season is not a repeatable event, he's a running back who is about to score
    # on a lot fewer of them next season. Uses the same "starter-caliber cohort" the
    # baseline regression above uses for pos_target, not the league as a whole,
    # because a bench-caliber player's red zone rate is exactly the kind of small,
    # unrepresentative sample that shouldn't set the bar a real starter is judged against.
    rz_baseline = {}
    for pos, chunk in t.groupby("position"):
        ranked = chunk.sort_values("fp_mean", ascending=False)
        n = min(starter_n.get(pos, 30), max(3, len(ranked)))
        top = ranked.head(n)
        touch_sum = top["rz_touches"].sum()
        rz_baseline[pos] = float(top["rz_td"].sum() / touch_sum) if touch_sum > 0 else 0.18
    t["rz_baseline_rate"] = t["position"].map(rz_baseline)
    t["rz_td_rate"] = t["rz_td"] / t["rz_touches"].replace(0, np.nan)
    t["m_td_luck"] = touchdown_luck_multiplier(
        t["rz_touches"], t["rz_td"], t["rz_baseline_rate"], w.td_luck)

    t["adj_ppg"] = (t["baseline_ppg"] * t["m_oline"] * t["m_volume"] * t["m_schedule"]
                    * t["m_divisional"] * t["m_injury"] * t["m_age"] * t["m_separation"]
                    * t["m_td_luck"] * t["m_coverage_trend"])
    t["proj_points"] = t["adj_ppg"] * t["exp_games"]

    # Full-PPR equivalent of the same projection. Published consensus rankings are
    # PPR, so converting the market's opinion into this league's format needs each
    # player's reception volume. The conversion itself is exact arithmetic — half
    # PPR is simply PPR minus half a point per catch — so the only estimate involved
    # is the reception count, which the projection already implies.
    rec_pg = t["rec_per_game"] if "rec_per_game" in t.columns else pd.Series(
        np.nan, index=t.index)
    # Rookies and anyone without a reception history fall back to positional norms.
    pos_rec = {"WR": 3.4, "TE": 2.8, "RB": 2.2, "QB": 0.0}
    rec_pg = rec_pg.fillna(t["position"].map(pos_rec)).fillna(0.0).clip(lower=0)
    t["proj_receptions"] = rec_pg * t["exp_games"]
    ppr_gap = 1.0 - float(league.scoring.rec)
    t["proj_points_ppr"] = t["proj_points"] + ppr_gap * t["proj_receptions"]

    # ---- consistency: the "will he give me a usable week" score.
    # Startable rate is the backbone; low variance and a high floor reinforce it;
    # injury risk cuts it, because an absent player is a zero.
    cv = t["fp_cv"].fillna(t["fp_cv"].median())
    cv_score = 1 - features._zscore(cv).clip(-2, 2) / 4
    floor_ratio = (t["floor"] / t["fp_mean"].replace(0, np.nan)).fillna(0.4).clip(0, 1)
    raw_consistency = (
        0.45 * t["startable_rate"].fillna(0.35)
        + 0.25 * floor_ratio
        + 0.15 * cv_score.clip(0, 1.5) / 1.5
        + 0.15 * (1 - t["injury_risk"])
    ).clip(0, 1)

    # Consistency needs the same small-sample regression the projection gets.
    # A backup who caught two touchdowns in his only three appearances will show a
    # perfect startable rate and near-zero variance; without this he outranks
    # established starters on "reliability" he has never actually demonstrated.
    cons_target = t["position"].map(
        t.assign(_c=raw_consistency).groupby("position").apply(
            lambda g: g.nlargest(40, "fp_mean")["_c"].mean(), include_groups=False
        ).to_dict()
    ).fillna(raw_consistency.mean())
    t["consistency"] = (raw_consistency * shrink + cons_target * (1 - shrink)).clip(0, 1)
    if is_rook.any() and "rookie_consistency" in t.columns:
        # Rookies get an explicit prior instead. They are the most volatile group in
        # fantasy: roles move mid-season and the floor is a healthy scratch.
        t.loc[is_rook, "consistency"] = t.loc[is_rook, "rookie_consistency"].fillna(0.35)
    t["consistency_sample_games"] = games

    # ---- value over replacement, then a consistency-weighted blend.
    repl = league.replacement_ranks()
    t["pos_rank"] = t.groupby("position")["proj_points"].rank(ascending=False, method="min")
    baselines = {}
    for pos in FANTASY_POSITIONS:
        chunk = t[t["position"] == pos].sort_values("proj_points", ascending=False)
        n = min(repl.get(pos, 24), max(1, len(chunk)))
        baselines[pos] = float(chunk["proj_points"].iloc[n - 1]) if len(chunk) else 0.0
    t["replacement_points"] = t["position"].map(baselines)
    t["vor"] = t["proj_points"] - t["replacement_points"]

    # Scale consistency onto the VOR distribution so the blend is apples to apples.
    vor_sd = t["vor"].std(ddof=0) or 1.0
    consistency_pts = (t["consistency"] - t["consistency"].mean()) / (
        t["consistency"].std(ddof=0) or 1.0) * vor_sd
    cw = weights.consistency_weight
    t["draft_score"] = (1 - cw) * t["vor"] + cw * consistency_pts
    if weights.qb_boost:
        t.loc[t["position"] == "QB", "draft_score"] *= (1 + weights.qb_boost)
    # Any player whose score is undefined would silently sort to the bottom rather
    # than announcing itself, so fail loudly instead.
    if t["draft_score"].isna().any():
        bad = t.loc[t["draft_score"].isna(), ["name", "position", "is_rookie"]]
        raise ValueError(f"{len(bad)} players scored NaN, e.g. "
                         f"{bad.head(3).to_dict('records')}")
    t["overall_rank"] = t["draft_score"].rank(ascending=False, method="min").astype(int)
    return t.sort_values("draft_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- pick-aware layer

def survival_probability(adp: float, current_pick: int, next_pick: int,
                         sd_floor: float = 5.0) -> float:
    """Chance a player is still on the board at your next pick.

    ADP is treated as the centre of a normal distribution whose spread widens later
    in the draft, which matches how real draft variance behaves: pick 3 goes where
    pick 3 goes, pick 90 is a coin flip across twenty names.
    """
    if not np.isfinite(adp):
        return 0.5
    sd = max(sd_floor, 0.22 * adp)
    # P(this player's realised draft slot lands after our next pick)
    p_survive = 1 - _norm_cdf((next_pick - adp) / sd)
    p_gone_now = _norm_cdf((current_pick - adp) / sd)
    if p_gone_now >= 0.999:
        return 0.0
    return float(np.clip(p_survive / max(1e-6, 1 - p_gone_now), 0.0, 1.0))


def survival_probability_vec(adp: np.ndarray, current_pick: int, next_pick: int,
                             sd_floor: float = 5.0) -> np.ndarray:
    """Vectorised form of survival_probability. plan_my_draft evaluates the whole
    board once per round, so the scalar version ran thousands of times per call."""
    adp = np.asarray(adp, dtype=float)
    sd = np.maximum(sd_floor, 0.22 * adp)
    # erf is available elementwise via numpy's own implementation path.
    _erf = np.vectorize(math.erf, otypes=[float])
    ncdf = lambda z: 0.5 * (1 + _erf(z / math.sqrt(2)))  # noqa: E731
    p_survive = 1 - ncdf((next_pick - adp) / sd)
    p_gone_now = ncdf((current_pick - adp) / sd)
    out = np.where(p_gone_now >= 0.999, 0.0,
                   p_survive / np.maximum(1e-6, 1 - p_gone_now))
    return np.clip(np.nan_to_num(out, nan=0.5), 0.0, 1.0)


def recommend(board: pd.DataFrame, league: LeagueSettings, current_pick: int,
              next_pick: int | None, roster: dict[str, int] | None = None,
              top_n: int = 8, roster_players: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rank available players for the pick that's on the clock.

    Ideas driving the ordering beyond raw value:
      * Opportunity cost — a player you're confident survives to your next pick is
        worth less right now than an equally good player who certainly won't.
      * Roster need — value is discounted once a position is full and the player
        would only be a bench body, and boosted when a starting slot is still open.
      * Roster construction — `roster_players` (full rows for who you already
        own, from DraftState.my_roster_players, not just position counts) drives
        bye-week collision and real-team exposure discounts, and a handcuff bonus.
        Unlike the above, this isn't a forecast, just bookkeeping over players you
        already have -- see roster_construction_mult's docstring.

    A two-turn lookahead (an optional third pick horizon feeding fallback_value)
    was tried and deliberately left out: survival_probability is provably
    non-increasing as the pick horizon extends (checked against 20k random draws,
    min difference 0.0), so expected_best_at_next_pick at a *later* absolute pick
    can never exceed its value at an earlier one -- max(value_at_next,
    value_at_next_next) always just equals value_at_next. It would have been dead
    code, not a real improvement, so it isn't here.
    """
    avail = board[~board["drafted"]].copy() if "drafted" in board.columns else board.copy()
    if avail.empty:
        return avail

    roster = roster or {}
    if next_pick:
        avail["p_available_next"] = survival_probability_vec(
            avail["adp"].to_numpy(), current_pick, next_pick)
    else:
        avail["p_available_next"] = 0.0

    # The heart of it: what a position is expected to still offer at your next turn.
    # Raw value says take the best player; that's wrong in a snake draft, because
    # passing on an elite QB costs you almost nothing (the QB you get two rounds
    # later is nearly as good) while passing on an elite RB costs a great deal.
    # Only the *marginal* gain over the wait matters.
    fallback = expected_best_at_next_pick(avail)
    avail["fallback_value"] = avail["position"].map(fallback).fillna(0.0)
    avail["marginal_value"] = avail["draft_score"] - avail["fallback_value"]
    avail["urgency"] = 1 - avail["p_available_next"]

    need = _positional_need(league, roster)
    avail["need_mult"] = avail["position"].map(need).fillna(0.7)

    rc = roster_construction_mult(avail, roster_players, league)
    avail["bye_mult"] = rc["bye_mult"].to_numpy()
    avail["exposure_mult"] = rc["exposure_mult"].to_numpy()
    avail["handcuff_mult"] = rc["handcuff_mult"].to_numpy()

    # A small share of raw value is retained so a truly generational player still
    # rises even when his position is deep behind him.
    avail["pick_value"] = (
        (0.80 * avail["marginal_value"] + 0.20 * avail["draft_score"])
        * avail["need_mult"] * avail["bye_mult"]
        * avail["exposure_mult"] * avail["handcuff_mult"]
    )
    return avail.sort_values("pick_value", ascending=False).head(top_n)


def expected_best_at_next_pick(avail: pd.DataFrame) -> dict[str, float]:
    """Expected draft_score of the best player at each position who survives to your
    next pick.

    Walks each position from the top down, accumulating the chance every better
    player is already gone. That product is the probability this player is the best
    one left, and the sum over players is the expected value of waiting.
    """
    out: dict[str, float] = {}
    for pos, chunk in avail.groupby("position"):
        chunk = chunk.sort_values("draft_score", ascending=False)
        expected, p_all_gone = 0.0, 1.0
        for _, r in chunk.iterrows():
            p = float(r.get("p_available_next", 0.0))
            expected += float(r["draft_score"]) * p * p_all_gone
            p_all_gone *= (1 - p)
            if p_all_gone < 0.005:
                break
        # If the position empties out entirely, waiting is worth the worst on the board.
        out[pos] = expected + p_all_gone * float(chunk["draft_score"].min() if len(chunk) else 0)
    return out


def likely_alternative_by_position(avail: pd.DataFrame, current_pick: int,
                                   next_pick: int | None) -> dict[str, dict]:
    """Per position, a concrete name for "if you wait, here's who you'd probably
    still get" -- the best remaining player with at least even odds
    (p_available_next >= 0.5) of lasting to `next_pick`, or the single best
    survival chance on the board if nobody clears that bar.

    Uses the exact same survival_probability_vec numbers expected_best_at_next_pick
    already relies on -- the mechanism position_scarcity_entropy_backtest found to
    be, by a wide margin, the strongest real predictor of the actual cost of waiting
    (implied_cost_corr 0.606, against entropy's -0.153). This adds no new signal,
    just a readable identity instead of an aggregate expected value.
    """
    if avail.empty:
        return {}
    avail = avail.copy()
    if next_pick:
        avail["p_available_next"] = survival_probability_vec(
            avail["adp"].to_numpy(), current_pick, next_pick)
    else:
        avail["p_available_next"] = 0.0

    out: dict[str, dict] = {}
    for pos, chunk in avail.groupby("position"):
        chunk = chunk.sort_values("draft_score", ascending=False)
        likely = chunk[chunk["p_available_next"] >= 0.5]
        pick = likely.iloc[0] if not likely.empty else chunk.iloc[0]
        out[pos] = {
            "name": pick["name"],
            "p_available_next": round(float(pick["p_available_next"]), 2),
            "likely": bool(not likely.empty),
        }
    return out


def position_scarcity_entropy(avail: pd.DataFrame, top_k: int = 12) -> dict[str, float]:
    """Shannon entropy of each position's remaining talent distribution, normalized
    to 0-1 -- a shape-of-the-pool signal distinct from expected_best_at_next_pick's
    timing-based urgency, and from anything ADP-based, since it only looks at how
    draft_score is distributed among what's left.

    For each position, takes the top `top_k` remaining players by draft_score,
    treats their scores as an (unnormalized) probability distribution over "who's
    the best one left", and computes Shannon entropy. A real talent cliff --
    one or two players clearly ahead of a bunched-up rest of the pool -- concentrates
    the distribution and pushes entropy toward 0: missing the top name costs real
    value because nothing close remains. A flat pool where the next several players
    are all roughly interchangeable pushes entropy toward 1 (its max, log2(top_k)):
    there's no rush, whoever's left after this pick is about as good.

    This is close to identical in spirit to expected_best_at_next_pick, deliberately:
    both walk the same sorted per-position pool. The difference is what each does
    with player values -- expected_best_at_next_pick weights by p_available_next to
    ask "how much value do I get if I wait", while this ignores pick timing entirely
    and asks "how lumpy is the pool itself" -- the two are meant to be compared
    against each other in a backtest, not assumed complementary.

    Sketch only -- not wired into recommend() or draft_score.
    adp.position_scarcity_entropy_backtest checked exactly that question: a 2021-2025
    run (1000 samples) found entropy actually *anti*-correlates with the real,
    per-position-normalized cost of waiting (entropy_corr -0.153) -- worse than doing
    nothing, and far worse than expected_best_at_next_pick's own survival-probability
    estimate of the same drop (implied_cost_corr 0.606, the mechanism recommend()
    already runs live). This isn't just a scale artifact either -- the same check
    against the raw, unnormalized drop looked modestly positive within every
    individual position, but that reading flipped negative once the target was
    normalized, because value_now and entropy both drift with pick depth and were
    only ever moving together through that lurking variable. Same conclusion
    redzone_shift_backtest and position_run_backtest each reached for their own
    factor: this stays a sketch, not wired into recommend() or draft_score.
    """
    out: dict[str, float] = {}
    for pos, chunk in avail.groupby("position"):
        chunk = chunk.sort_values("draft_score", ascending=False).head(top_k)
        scores = chunk["draft_score"].to_numpy(dtype=float)
        scores = scores - min(0.0, scores.min())  # shift so a negative score can't
                                                   # flip the sign of the distribution
        total = scores.sum()
        if total <= 0 or len(scores) < 2:
            out[pos] = 1.0  # nothing left to be scarce about -- treat as "flat"
            continue
        p = scores / total
        # np.where(p > 0, p * log2(p), 0) still evaluates log2(0) for the masked-out
        # side before discarding it -- real boards have plenty of exactly-zero
        # draft_score bench fodder, so that's not a hypothetical, it's a guaranteed
        # RuntimeWarning on every live board. Masking before the log2 call avoids
        # computing it on zeros at all instead of computing-then-discarding.
        nz = p[p > 0]
        h = -np.sum(nz * np.log2(nz))
        out[pos] = float(h / np.log2(len(scores)))
    return out


# How much a bench player at each position is worth relative to the one ahead of him.
# RB and WR depth holds real value because injuries and byes force them into lineups
# constantly. A second QB, in a non-superflex league, never starts at all outside
# this league's own FLEX/superflex rules -- QB isn't flex-eligible like RB/WR/TE
# are, so 0.20 wasn't steep enough: a mock_draft check (30 trials, 1-QB league)
# had the model rostering a real backup QB (Mahomes-plus tier, not a late dart
# throw) in 27 of 30 trials, because QB draft_score is structurally so much
# larger than every other position's at that draft slot -- passing yards/TDs
# outscore the field even for a QB1-caliber name nobody would actually roster
# twice -- that even an 80% discount left him ahead of real bench RB/WR value.
# 0.04 pushes a true backup below that bench value in the same test.
BACKUP_DECAY = {"QB": 0.04, "TE": 0.28, "RB": 0.72, "WR": 0.70}
# Past these counts a player cannot realistically help you, whatever his projection.
ROSTER_CAP = {"QB": 2, "TE": 2, "RB": 6, "WR": 7}

# Discount keyed on how many *starting slots* a bye week would leave unfilled at
# that position group (see _bye_shortfall) -- not a raw "how many teammates share
# this bye" count. A raw count flags harmless bench redundancy (two backup WRs on
# bye while five others are healthy) exactly the same as a real problem (your only
# three RBs all sharing one), which is what the first version of this check did.
BYE_SHORTFALL_DECAY = {1: 0.85, 2: 0.60, 3: 0.40}
# Discount keyed on how many rostered players already come from this NFL team --
# a scheme change, coordinator firing or O-line injury can tank several of your
# players in the same week, a correlated-bust risk raw draft_score can't see.
TEAM_EXPOSURE_DECAY = {2: 0.95, 3: 0.85, 4: 0.70}
# Bonus for a player who reads as the backup (same team, same position) to someone
# already on your roster -- insurance value: losing your starter hands this player
# the touches on the same offense. Real handcuffing is a running-back-specific
# idea (a timeshare/committee shift concentrates onto one backup); it doesn't have
# a clean analogue at the other positions.
HANDCUFF_BONUS = 1.15
HANDCUFF_ELIGIBLE = {"RB"}


def _tiered_mult(count: int, table: dict[int, float]) -> float:
    """The decay for the highest threshold in `table` that `count` meets or beats,
    else 1.0 (no effect) -- shared by the bye-shortfall and team-exposure checks."""
    mult = 1.0
    for threshold in sorted(table):
        if count >= threshold:
            mult = table[threshold]
    return mult


def _bye_shortfall(position: str, bye, roster_players: pd.DataFrame,
                   league: LeagueSettings) -> int:
    """How many starting slots would go unfilled the week `bye` falls, if a player
    at `position` with that bye joined the roster -- 0 if the roster (plus this
    candidate) still covers every starter at that position group that week.

    Checked against the combined FLEX-eligible group (RB/WR/TE plus the FLEX slot)
    rather than the exact position alone, since a thin RB week can be covered by an
    extra rostered WR/TE in the FLEX slot and vice versa. QB folds in the
    superflex slot count the same way _positional_need does, since a second
    required QB only exists in superflex.
    """
    if bye is None or pd.isna(bye) or not {"position", "bye"}.issubset(roster_players.columns):
        return 0
    if position in league.flex_eligible:
        group = league.flex_eligible
        required = sum(league.starters.get(p, 0) for p in group) + league.starters.get("FLEX", 0)
    elif position == "QB":
        group = ("QB",)
        required = league.starters.get("QB", 0) + (getattr(league, "superflex", 0) or 0)
    else:
        group = (position,)
        required = league.starters.get(position, 0)
    pool = roster_players[roster_players["position"].isin(group)]
    total_owned = len(pool) + 1  # + this candidate
    on_bye = int((pool["bye"] == bye).sum()) + 1  # + this candidate
    return max(0, required - (total_owned - on_bye))


def roster_construction_mult(avail: pd.DataFrame, roster_players: pd.DataFrame | None,
                             league: LeagueSettings | None = None) -> pd.DataFrame:
    """Per-candidate multipliers driven by who's already on your roster: bye-week
    lineup shortfalls, real-team exposure, and handcuff insurance value.

    Unlike PositionMarkov or position_scarcity_entropy, this doesn't need its own
    backtest before being trusted -- it isn't forecasting anything about the draft
    or the season, it's deterministic bookkeeping over players you already own, the
    same category _positional_need's need_mult already falls into.

    Returns a DataFrame of `bye_mult` / `exposure_mult` / `handcuff_mult` columns
    aligned to avail's index, all 1.0 when roster_players is empty or missing the
    columns (`team`, `bye`, `position`, `draft_score`, `depth_rank`) a check needs --
    a league or board without bye-week data (schedule not published yet), team
    data, or a published depth chart just skips that one check rather than raising.
    `league` defaults to a generic LeagueSettings() when omitted, since the bye
    check needs starter counts to know what a "shortfall" even means.
    """
    league = league or LeagueSettings()
    n = len(avail)
    out = pd.DataFrame({"bye_mult": np.ones(n), "exposure_mult": np.ones(n),
                        "handcuff_mult": np.ones(n)}, index=avail.index)
    if roster_players is None or roster_players.empty:
        return out

    if "bye" in avail.columns and {"position", "bye"}.issubset(roster_players.columns):
        out["bye_mult"] = avail.apply(
            lambda r: _tiered_mult(
                _bye_shortfall(r["position"], r.get("bye"), roster_players, league),
                BYE_SHORTFALL_DECAY),
            axis=1)

    if "team" in roster_players.columns and "team" in avail.columns:
        team_counts = roster_players["team"].dropna().value_counts().to_dict()
        out["exposure_mult"] = avail["team"].map(
            lambda t: _tiered_mult(team_counts.get(t, 0), TEAM_EXPOSURE_DECAY)
            if pd.notna(t) else 1.0)

        if {"position", "draft_score"}.issubset(roster_players.columns) and \
                {"position", "draft_score"}.issubset(avail.columns):
            has_depth_rank = ("depth_rank" in roster_players.columns
                             and roster_players["depth_rank"].notna().any())
            best_owned = {}
            for key, chunk in roster_players.groupby(["team", "position"]):
                if has_depth_rank and chunk["depth_rank"].notna().any():
                    row = chunk.loc[chunk["depth_rank"].idxmin()]
                    best_owned[key] = {"depth_rank": row["depth_rank"],
                                       "draft_score": row.get("draft_score")}
                else:
                    row = chunk.loc[chunk["draft_score"].idxmax()]
                    best_owned[key] = {"depth_rank": np.nan, "draft_score": row["draft_score"]}

            def handcuff(row) -> float:
                if row["position"] not in HANDCUFF_ELIGIBLE:
                    return 1.0
                owned = best_owned.get((row.get("team"), row["position"]))
                if owned is None:
                    return 1.0
                cand_rank = row.get("depth_rank")
                # Real depth-chart order first: a clearly-labelled RB2 is a real
                # handcuff even on the rare occasion his own draft_score isn't the
                # lower of the two, and two similarly-projected backs who are both
                # rank 1/1A (a real committee) correctly get no bonus either way --
                # value alone can't tell a committee from a true backup.
                if pd.notna(cand_rank) and pd.notna(owned["depth_rank"]):
                    return HANDCUFF_BONUS if cand_rank > owned["depth_rank"] else 1.0
                owned_score = owned.get("draft_score")
                if owned_score is not None and pd.notna(owned_score):
                    return HANDCUFF_BONUS if row["draft_score"] < owned_score else 1.0
                return 1.0

            out["handcuff_mult"] = avail.apply(handcuff, axis=1)

    return out


def _positional_need(league: LeagueSettings, roster: dict[str, int]) -> dict[str, float]:
    """Multiplier per position, driven by the chance the player ever starts for you.

    An empty starting slot is worth a premium. A bench spot is worth only as much as
    the odds it gets pressed into the lineup, which decays fast at positions with one
    starting slot — the reason a great backup quarterback wins you nothing.
    """
    need = {}
    superflex = getattr(league, "superflex", 0) or 0
    flex_slots = league.starters.get("FLEX", 0)
    flex_filled = sum(max(0, roster.get(p, 0) - league.starters.get(p, 0))
                      for p in league.flex_eligible)

    # In superflex the second quarterback is a starter, not a bench body, so every
    # rule that assumes one QB slot has to shift: more of them are required, they
    # stay valuable deeper, and the early-round dampener must not apply.
    required_qb = league.starters.get("QB", 0) + superflex
    caps = dict(ROSTER_CAP)
    decay = dict(BACKUP_DECAY)
    if superflex:
        caps["QB"] = league.starters.get("QB", 0) + superflex + 1
        decay["QB"] = 0.55

    for pos in FANTASY_POSITIONS:
        required = required_qb if pos == "QB" else league.starters.get(pos, 0)
        have = roster.get(pos, 0)
        cap = caps.get(pos, 6)
        if have >= cap:
            need[pos] = 0.02
        elif have < required:
            need[pos] = 1.18
        elif pos in league.flex_eligible and flex_filled < flex_slots:
            need[pos] = 1.05
        else:
            depth = have - required + (1 if pos in league.flex_eligible
                                       and flex_filled >= flex_slots else 0)
            need[pos] = max(0.03, decay.get(pos, 0.5) ** max(1, depth))

    # Don't let a lone QB/TE slot pull you into reaching in the early rounds, where
    # the elite RB and WR you'd be passing on are the scarcer resource. This is a
    # 1-QB argument only — in superflex, quarterbacks genuinely are the scarce
    # resource and reaching for one early is correct.
    filled = sum(roster.values())
    if not superflex and roster.get("QB", 0) == 0 and filled < 5:
        need["QB"] = min(need["QB"], 0.80)
    if roster.get("TE", 0) == 0 and filled < 3:
        need["TE"] = min(need["TE"], 0.88)
    return need


def explain(row: pd.Series) -> str:
    """Plain-language reasoning for a recommendation."""
    bits = []
    if row.get("pos_rank"):
        bits.append(f"{row['position']}{int(row['pos_rank'])} by projection "
                    f"({row['proj_points']:.0f} pts, {row['adj_ppg']:.1f}/gm)")
    c = row.get("consistency")
    if c is not None and np.isfinite(c):
        sr = row.get("startable_rate")
        bits.append(f"consistency {c:.2f}" + (f", startable in {sr:.0%} of weeks" if np.isfinite(sr or np.nan) else ""))
    for label, key in [
        ("O-line", "m_oline"), ("volume/pace", "m_volume"),
        ("schedule", "m_schedule"), ("age curve", "m_age"),
        ("separation", "m_separation"), ("touchdown regression", "m_td_luck"),
        ("coverage trend", "m_coverage_trend"),
    ]:
        v = row.get(key)
        if v is not None and np.isfinite(v) and abs(v - 1) > 0.02:
            bits.append(f"{label} {'+' if v > 1 else ''}{(v - 1) * 100:.1f}%")
    ir = row.get("injury_risk")
    if ir is not None and np.isfinite(ir):
        bits.append(f"injury risk {ir:.0%} (~{row.get('exp_games', 17):.0f} games)")
    p = row.get("p_available_next")
    if p is not None and np.isfinite(p):
        bits.append(f"{p:.0%} chance he lasts to your next pick")
    bm = row.get("bye_mult")
    if bm is not None and np.isfinite(bm) and bm < 0.97:
        bits.append(f"bye-week clash with your roster ({bm:.2f}x)")
    em = row.get("exposure_mult")
    if em is not None and np.isfinite(em) and em < 0.97:
        bits.append(f"heavy {row.get('team', 'that team')} exposure already ({em:.2f}x)")
    hm = row.get("handcuff_mult")
    if hm is not None and np.isfinite(hm) and hm > 1.02:
        bits.append(f"handcuff value behind your own {row.get('team', '')} RB".rstrip())
    return "; ".join(bits)
