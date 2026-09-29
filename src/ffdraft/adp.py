"""Preseason rankings vs. actual finish — the 'value pick' engine.

Draft position is a market price. What matters is which players systematically beat
that price. This module pairs FantasyPros preseason expert consensus rank (a very
close stand-in for ADP, published back to 2020) with the fantasy points each player
actually finished with, so hit rates can be measured rather than assumed.

Source: dynastyprocess/data, which mirrors FantasyPros ECR history.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import sources
from .config import FANTASY_POSITIONS, Scoring
from .sources import _cached

ECR_URL = "https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_fpecr.parquet"

# A player has to be startable-relevant for a hit/bust label to mean anything.
DRAFTABLE_ECR_CUTOFF = 200


def _ecr_raw(page_type: str = "redraft-overall") -> pd.DataFrame:
    def build():
        d = pd.read_parquet(ECR_URL)
        d = d[d["page_type"].isin(["redraft-overall", "redraft-op"])]
        d = d[d["pos"].isin(FANTASY_POSITIONS)]
        keep = ["player", "pos", "tm", "ecr", "sd", "best", "worst",
                "scrape_date", "page_type"]
        return d[[c for c in keep if c in d.columns]].copy()

    d = _cached("fp_ecr_redraft", build, max_age_days=1.0)
    d = d[d["page_type"] == page_type]
    d = d.copy()
    d["scrape_date"] = pd.to_datetime(d["scrape_date"])
    return d


def preseason_ecr(season: int, superflex: bool = False) -> pd.DataFrame:
    """Consensus rank as of the last August scrape before the given season.

    August is the right snapshot: rankings before then still price in players who
    won't make a roster, and September rankings have already absorbed preseason
    injuries, which would leak information into a backtest.

    Superflex leagues get their own consensus, because the two markets barely
    resemble each other — in 2026 Josh Allen is the 26th pick in a 1-QB league and
    the 1st overall pick in superflex. Pricing a superflex draft off 1-QB rankings
    would make every quarterback look like a bargain and wreck the whole
    opportunity-cost calculation.
    """
    from .names import normalize as norm_name

    d = _ecr_raw("redraft-op" if superflex else "redraft-overall")
    aug = d[(d["scrape_date"].dt.year == season) & (d["scrape_date"].dt.month.isin([7, 8]))]
    if aug.empty:  # fall back to the earliest snapshot in that season
        aug = d[(d["scrape_date"].dt.year == season) & (d["scrape_date"].dt.month <= 9)]
    if aug.empty:
        return pd.DataFrame(columns=["name", "position", "ecr", "_key"])
    latest = aug["scrape_date"].max()
    snap = aug[aug["scrape_date"] == latest].copy()
    snap = snap.rename(columns={"player": "name", "pos": "position", "tm": "team"})
    snap["ecr"] = pd.to_numeric(snap["ecr"], errors="coerce")
    snap = snap.dropna(subset=["ecr"]).sort_values("ecr")
    snap["adp"] = snap["ecr"]
    snap["pos_ecr"] = snap.groupby("position")["ecr"].rank(method="min").astype(int)
    snap["_key"] = snap["name"].map(norm_name)
    snap["snapshot"] = latest.date().isoformat()
    return snap.drop_duplicates("_key").reset_index(drop=True)


def season_finish(season: int, sc: Scoring | None = None,
                  te_bonus: float = 0.0) -> pd.DataFrame:
    """Where each player actually finished: total points, overall and positional rank."""
    from . import features
    from .names import normalize as norm_name

    sc = sc or Scoring()
    w = sources.weekly_stats([season])
    w = w[w["position"].isin(FANTASY_POSITIONS) & (w["season_type"] == "REG")].copy()
    w["fp"] = features.fantasy_points(w, sc, te_bonus)

    thresh = w["position"].map({"QB": 18.0, "RB": 12.0, "WR": 12.0, "TE": 9.0})
    w["startable"] = (w["fp"] >= thresh).astype(float)

    fin = w.groupby(["player_id", "player_display_name", "position"]).agg(
        games=("week", "nunique"),
        points=("fp", "sum"),
        ppg=("fp", "mean"),
        startable_rate=("startable", "mean"),
    ).reset_index().rename(columns={"player_display_name": "name"})

    fin["finish_pos_rank"] = fin.groupby("position")["points"].rank(
        ascending=False, method="min").astype(int)
    fin["finish_overall"] = fin["points"].rank(ascending=False, method="min").astype(int)
    fin["_key"] = fin["name"].map(norm_name)
    fin["season"] = season
    return fin


def _resolve_unmatched(merged: pd.DataFrame, fin: pd.DataFrame) -> pd.DataFrame:
    """Second pass for players whose ECR name doesn't match the stats name exactly.

    Ranking sites and nflverse disagree constantly on given names — "Josh Palmer"
    versus "Joshua Palmer", "Cam" versus "Cameron", "Marquise" versus "Hollywood".
    Left unresolved these look like players who scored zero all season, which would
    plant fabricated busts right in the middle of the results.
    """
    from difflib import get_close_matches

    miss = merged[merged["points"].isna() | (merged["points"] == 0)]
    if miss.empty:
        return merged

    for idx, row in miss.iterrows():
        pool = fin[fin["position"] == row["position"]]
        if pool.empty:
            continue
        keys = pool["_key"].tolist()
        # Same last name plus a compatible first initial is the reliable signal.
        parts = str(row["_key"]).split()
        if len(parts) >= 2:
            last, first_i = parts[-1], parts[0][:1]
            cand = pool[pool["_key"].str.endswith(" " + last)
                        & pool["_key"].str.startswith(first_i)]
            if len(cand) == 1:
                hit = cand.iloc[0]
                for c in ("points", "ppg", "games", "startable_rate",
                          "finish_pos_rank", "finish_overall"):
                    merged.loc[idx, c] = hit[c]
                merged.loc[idx, "match"] = "alias"
                continue
        close = get_close_matches(str(row["_key"]), keys, n=1, cutoff=0.88)
        if close:
            hit = pool[pool["_key"] == close[0]].iloc[0]
            for c in ("points", "ppg", "games", "startable_rate",
                      "finish_pos_rank", "finish_overall"):
                merged.loc[idx, c] = hit[c]
            merged.loc[idx, "match"] = "fuzzy"
    return merged


def _format_shift_ecr(pre: pd.DataFrame, season: int, sc: Scoring) -> pd.DataFrame:
    """Convert PPR consensus ranks to the league's format for a historical season.

    Published consensus is PPR, but the finishes it's being scored against use this
    league's scoring. Comparing the two directly would systematically flag every
    reception-heavy receiver as a bust in a standard league and every early-down
    back as a hit, which is an artefact of the mismatch rather than a real finding.

    The reception estimate uses the *prior* season's catches — what a drafter
    actually knew in August. Using the season's own receptions would leak the
    result being measured back into the prediction.
    """
    gap = 1.0 - float(sc.rec)
    if abs(gap) < 1e-9:
        pre = pre.copy()
        pre["ecr_format"] = "ppr"
        return pre

    from .names import normalize as norm_name

    try:
        prior = sources.weekly_stats([season - 1])
    except Exception:
        pre = pre.copy()
        pre["ecr_format"] = "ppr (unconverted: no prior season)"
        return pre

    from . import features

    prior = prior[prior["season_type"] == "REG"].copy()
    prior["pts_ppr"] = features.fantasy_points(prior, Scoring.preset("ppr"))
    prior["pts_fmt"] = features.fantasy_points(prior, sc)
    tot = prior.groupby("player_display_name")[["pts_ppr", "pts_fmt"]].sum().reset_index()
    tot["_key"] = tot["player_display_name"].map(norm_name)

    p = pre.merge(tot[["_key", "pts_ppr", "pts_fmt"]], on="_key", how="left")
    # Players with no prior season sit at the median and simply don't move.
    have = p["pts_ppr"].notna() & p["pts_fmt"].notna()
    rank_ppr = p.loc[have, "pts_ppr"].rank(ascending=False, method="min")
    rank_fmt = p.loc[have, "pts_fmt"].rank(ascending=False, method="min")
    p["_shift"] = 0.0
    p.loc[have, "_shift"] = (rank_fmt - rank_ppr) * 0.6

    p["ecr_ppr"] = p["ecr"]
    p["ecr"] = (p["ecr"] + p["_shift"]).clip(lower=1.0)
    p = p.sort_values("ecr")
    p["pos_ecr"] = p.groupby("position")["ecr"].rank(method="min").astype(int)
    p["ecr_format"] = "half_ppr" if gap <= 0.6 else "standard"
    return p.reset_index(drop=True)


def adp_vs_finish(season: int, sc: Scoring | None = None) -> pd.DataFrame:
    """Join preseason rank to final finish for one season.

    Comparison is positional rank against positional rank. Overall ranks aren't
    comparable across positions — QB1 scoring 400 points and RB1 scoring 300 are
    both wild successes, but their overall finishes look nothing alike.
    """
    sc = sc or Scoring()
    pre = preseason_ecr(season)
    if pre.empty:
        return pd.DataFrame()
    pre = _format_shift_ecr(pre, season, sc)
    fin = season_finish(season, sc)

    keep = ["_key", "name", "position", "team", "ecr", "pos_ecr", "sd", "snapshot"]
    keep += [c for c in ("ecr_ppr", "ecr_format") if c in pre.columns]
    m = pre[keep].merge(
        fin[["_key", "points", "ppg", "games", "startable_rate",
             "finish_pos_rank", "finish_overall"]],
        on="_key", how="left",
    )
    m["match"] = np.where(m["points"].notna(), "exact", "none")
    m = _resolve_unmatched(m, fin)

    m["season"] = season
    # A player still unmatched after alias and fuzzy passes either genuinely never
    # took a snap, or the name is unresolvable. Either way he's excluded from hit
    # and bust rates rather than counted as a zero, which would bias them.
    m["unresolved"] = m["points"].isna()
    m["finish_pos_rank"] = m["finish_pos_rank"].fillna(999)
    m["points"] = m["points"].fillna(0.0)
    m["games"] = m["games"].fillna(0)
    m["pos_rank_delta"] = m["pos_ecr"] - m["finish_pos_rank"]
    m["draft_round"] = np.ceil(m["ecr"] / 12).astype(int)

    # Value is measured in points against what that draft slot actually returned,
    # not in rank movement. Rank movement is unfair to early picks — an RB drafted
    # RB2 cannot rise a tier, and every drafted player gets pushed down the final
    # standings by undrafted breakouts, which would label whole rounds as busts.
    # "If you spent RB5 capital, did you get RB5 production?" is the honest test.
    m["expected_points"] = np.nan
    for pos, chunk in m.groupby("position"):
        curve = fin[fin["position"] == pos].sort_values("points", ascending=False)
        pts = curve["points"].to_numpy()
        if len(pts) == 0:
            continue
        slots = chunk["pos_ecr"].clip(1, len(pts)).astype(int).to_numpy() - 1
        m.loc[chunk.index, "expected_points"] = pts[slots]

    m["value_points"] = m["points"] - m["expected_points"]
    m["value_ratio"] = m["points"] / m["expected_points"].replace(0, np.nan)
    m["hit"] = m["value_ratio"] >= 1.15
    m["bust"] = m["value_ratio"] <= 0.70
    m["value_score"] = m["value_ratio"] - 1
    return m.sort_values("ecr").reset_index(drop=True)


def value_history(seasons: list[int], sc: Scoring | None = None) -> pd.DataFrame:
    """Stack multiple seasons of preseason-rank vs finish."""
    frames = []
    for s in seasons:
        try:
            df = adp_vs_finish(s, sc)
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            print(f"  ! {s}: {type(exc).__name__}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def hit_rates(hist: pd.DataFrame, by: str = "draft_round") -> pd.DataFrame:
    """Hit and bust rates by draft round or position — where value actually lives."""
    h = hist[(hist["ecr"] <= DRAFTABLE_ECR_CUTOFF) & (~hist.get("unresolved", False))]
    g = h.groupby(by).agg(
        n=("hit", "size"),
        hit_rate=("hit", "mean"),
        bust_rate=("bust", "mean"),
        median_value_ratio=("value_ratio", "median"),
        mean_games=("games", "mean"),
    ).reset_index()
    return g.sort_values(by)


def matchup_value_backtest(seasons: list[int], position: str = "WR",
                           sc: Scoring | None = None) -> pd.DataFrame:
    """Did talent + schedule-adjusted matchup predict finish better than talent alone?

    Same idea as separation_report's schedule adjustment (talent z-score plus
    schedule-difficulty z-score), tested against real outcomes the same way
    adp_vs_finish backtests consensus rank: nothing here has seen the season it's
    scoring. A 2021-2024 WR run found talent alone predicts better, so
    separation_report ranks by talent and shows matchup_z for reference only --
    this function is what proved that, and it's here to re-check if the model
    or the data underlying it changes.

    Talent (`talent_z`) is that player's separation score from the *prior* season
    only -- what a drafter actually knew in August, not a mid-season update. Matchup
    difficulty (`matchup_z`) comes from strength_of_schedule() computed exactly the
    way the live model computes it: opponent defensive strength is a recency-weighted
    blend of seasons strictly before the one being predicted, so the defense side
    can't leak either. The schedule itself (who plays whom) is legitimately known
    in advance -- the NFL publishes it -- so using it isn't leakage.

    Actual finish is real fantasy points from that season's box scores.
    """
    from . import features
    from . import separation as sep_mod

    sc = sc or Scoring()
    position = position.upper()
    frames = []
    for season in seasons:
        try:
            prior = sep_mod.separation_profile([season - 1])
            prior = prior[prior["qualified"] & (prior["position"] == position)]
            if prior.empty:
                print(f"  ! {season}: no qualified {position}s in {season - 1} to score talent from")
                continue
            talent = prior[["player_id", "sep_score"]].rename(
                columns={"sep_score": "talent_z"})

            fin = season_finish(season, sc)
            fin = fin[fin["position"] == position]
            if fin.empty:
                continue

            dfn = features.defense_ratings(sc=sc)
            sos = features.strength_of_schedule(season, dfn)
            sos_col = f"sos_{position}_z"
            if sos_col not in sos.columns:
                print(f"  ! {season}: no schedule data for {position}")
                continue

            w = sources.weekly_stats([season])
            w = w[w["position"] == position]
            team_of = (w.sort_values("week").groupby("player_id")["recent_team"]
                       .last().rename("team").reset_index())

            m = fin.merge(talent, on="player_id", how="inner")
            if m.empty:
                continue
            m = m.merge(team_of, on="player_id", how="left")
            m = m.merge(sos[["team", sos_col]], on="team", how="left")
            m = m.rename(columns={sos_col: "matchup_z"})
            m["matchup_z"] = m["matchup_z"].fillna(0.0)
            m["matchup_adjusted_score"] = m["talent_z"] + m["matchup_z"]
            m["season"] = season
            frames.append(m)
        except Exception as exc:
            print(f"  ! {season}: {type(exc).__name__}: {exc}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def redzone_shift_backtest(seasons: list[int], position: str = "WR",
                          sc: Scoring | None = None) -> pd.DataFrame:
    """Does a team's red zone play-calling identity improve on the touchdown-luck
    signal alone at predicting next season's fantasy points?

    Same discipline as matchup_value_backtest, and the output uses the exact same
    column names (`talent_z`, `matchup_adjusted_score`, `points`, `season`) on
    purpose, so it can be scored by the same matchup_backtest_summary rather than
    duplicating that logic.

    `talent_z` is the existing touchdown-luck signal the live model already uses
    (positive = scored fewer red zone touchdowns than his role predicted -- the
    buy-low direction `m_td_luck` regresses toward), z-scored across that position's
    qualifying cohort, computed from strictly the season *before* the one being
    predicted -- what a drafter actually knew in August. `matchup_adjusted_score`
    subtracts that player's team's red zone identity shift (z-scored across teams,
    same season) on the theory that a team going noticeably run-heavy inside the 20
    undercuts a receiver's or tight end's apparent red zone role even when his own
    prior-season rate looked like a buy-low.

    Only meaningful for pass catchers -- red zone identity shift is a *pass rate*
    signal with no defensible sign for a running back (more run-heavy could mean
    more goal-line carries, not fewer), so this raises for anything else.
    """
    from . import features

    sc = sc or Scoring()
    position = position.upper()
    if position not in ("WR", "TE"):
        raise ValueError("redzone_shift_backtest only supports WR/TE -- "
                         "red zone identity shift has no defensible sign for RB/QB")

    min_touches = 8
    starter_n = 48 if position == "WR" else 20
    frames = []
    for season in seasons:
        prior = season - 1
        try:
            pbp_prior = sources.play_by_play(seasons=[prior])
            rz_role = features.player_redzone_role(pbp_prior)
            rz_role = rz_role[rz_role["season"] == prior]

            w = sources.weekly_stats([prior])
            w = w[w["position"] == position]
            if w.empty:
                continue
            names = (w[["player_id", "player_display_name"]]
                    .drop_duplicates("player_id"))
            team_of = (w.sort_values("week").groupby("player_id")["recent_team"]
                      .last().rename("team").reset_index())

            role = rz_role.merge(names, on="player_id", how="inner").merge(
                team_of, on="player_id", how="left")
            role = role[role["rz_touches"] >= min_touches].copy()
            if role.empty:
                print(f"  ! {season}: no qualifying {position}s with "
                     f"{min_touches}+ RZ touches in {prior}")
                continue

            # Position baseline conversion rate for that prior season, from the
            # highest-red-zone-volume cohort -- a proxy for "starter caliber" here
            # since this lightweight backtest doesn't recompute full fp_mean.
            top = role.sort_values("rz_touches", ascending=False).head(starter_n)
            touch_sum = top["rz_touches"].sum()
            baseline_rate = float(top["rz_td"].sum() / touch_sum) if touch_sum else 0.2

            role["expected_td"] = role["rz_touches"] * baseline_rate
            role["surplus"] = role["rz_td"] - role["expected_td"]
            sd = role["surplus"].std(ddof=0) or 1.0
            # Sign-flipped like touchdown_luck_multiplier: underperformed (positive
            # surplus is negative here) -> positive talent_z -> buy-low.
            role["talent_z"] = -(role["surplus"] - role["surplus"].mean()) / sd

            rz_shift = features.redzone_identity_shift(pbp_prior)
            rz_shift = rz_shift[rz_shift["season"] == prior]
            if rz_shift.empty:
                shift_map = pd.DataFrame(columns=["team", "shift_z"])
            else:
                shift_sd = rz_shift["shift"].std(ddof=0) or 1.0
                shift_map = rz_shift.assign(
                    shift_z=(rz_shift["shift"] - rz_shift["shift"].mean()) / shift_sd
                )[["team", "shift_z"]]

            role = role.merge(shift_map, on="team", how="left")
            role["shift_z"] = role["shift_z"].fillna(0.0)
            role["matchup_adjusted_score"] = role["talent_z"] - role["shift_z"]

            fin = season_finish(season, sc)
            fin = fin[fin["position"] == position][
                ["player_id", "points", "finish_pos_rank"]]
            m = role.merge(fin, on="player_id", how="inner")
            if m.empty:
                continue
            m["season"] = season
            m = m.rename(columns={"player_display_name": "name",
                                  "shift_z": "matchup_z"})
            frames.append(m[["player_id", "name", "team", "season", "talent_z",
                            "matchup_z", "matchup_adjusted_score", "points",
                            "finish_pos_rank"]])
        except Exception as exc:
            print(f"  ! {season}: {type(exc).__name__}: {exc}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def bootstrap_ci(hist: pd.DataFrame, summary_fn: Callable[[pd.DataFrame], dict],
                 metrics: list[str], n_boot: int = 1000, seed: int = 0,
                 group_col: str = "season") -> dict[str, dict]:
    """95% confidence intervals for backtest summary metrics, shared by every
    backtest in this module.

    Every backtest here has been reporting bare point estimates -- "entropy_corr
    -0.153," "top1_accuracy 0.371 vs. 0.345" -- with no way to tell whether a gap
    like that is real or just noise from having only a handful of seasons/absences
    to test on. This puts an actual interval around each headline number.

    Resamples the distinct groups in `hist` (seasons, by default -- every backtest
    in this module already carries a `season` column) WITH replacement, keeping
    the same count as the original, rebuilds a hist from those groups' rows, and
    recomputes `summary_fn` on it -- `n_boot` times. This is a *block* bootstrap
    over whole seasons rather than individual rows, since transitions/events within
    the same season aren't independent of each other; resampling rows directly
    would understate the real uncertainty. It's cheap because none of these
    summary_fns refit anything -- they aggregate numbers the backtest already
    computed per row, so this is pure resampling and arithmetic, safe to run on
    every call rather than as a separate opt-in step.

    Returns, per metric, {"point": <value on the real data>, "lo": <2.5th
    percentile>, "hi": <97.5th percentile>} -- lo/hi are None when there are
    fewer than 2 groups to resample, or fewer than 10 usable bootstrap draws for
    that metric (e.g. it's undefined whenever a resample happens to be missing a
    denominator).

    Caveat worth keeping in mind when reading the interval: with as few as 4-5
    seasons (typical here), this measures how much the reported number could
    plausibly swing given the seasons actually observed -- necessarily a wide
    interval -- not whether the effect would replicate in some future, unseen
    room or season. A CI that straddles zero means "this could easily be noise,"
    not "there is definitely no effect."
    """
    groups = hist[group_col].unique() if group_col in hist.columns else []
    point = summary_fn(hist)
    if len(groups) < 2:
        return {m: {"point": point.get(m), "lo": None, "hi": None} for m in metrics}

    rng = np.random.default_rng(seed)
    by_group = {g: hist[hist[group_col] == g] for g in groups}
    draws: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_boot):
        sample_groups = rng.choice(groups, size=len(groups), replace=True)
        resampled = pd.concat([by_group[g] for g in sample_groups], ignore_index=True)
        s = summary_fn(resampled)
        for m in metrics:
            v = s.get(m)
            if v is not None and isinstance(v, (int, float)) and np.isfinite(v):
                draws[m].append(float(v))

    out = {}
    for m in metrics:
        vals = draws[m]
        if len(vals) < 10:
            out[m] = {"point": point.get(m), "lo": None, "hi": None}
        else:
            out[m] = {"point": point.get(m),
                      "lo": float(np.percentile(vals, 2.5)),
                      "hi": float(np.percentile(vals, 97.5))}
    return out


def position_run_backtest(seasons: list[int], smoothing: float = 1.0) -> pd.DataFrame:
    """Does board.PositionMarkov predict the next pick's position better than
    baselines that ignore the current one?

    IMPORTANT caveat this backtest can't get around: there's no stored archive of
    real snake-draft pick order across many rooms and seasons, only each season's
    preseason ECR snapshot (preseason_ecr) -- the same market-order proxy
    PositionMarkov.from_adp_order itself is built from. So this measures something
    real but narrower than "does it predict human draft-room behavior": whether the
    position-transition structure implied by consensus rank is stable enough
    season-to-season to generalize at all. That's a necessary condition for the live
    model to be useful, not a sufficient one -- an actual positional run is a
    room-psychology phenomenon (scarcity panic feeding on itself) that this proxy,
    being a single fixed ranking rather than reactive human picks, can't fully
    capture. Validating that needs real completed pick order (DraftState.picks,
    sync_espn, sync_sleeper) fed through PositionMarkov.from_sequences instead.

    Leave-one-season-out: for each test season, fit a model on every *other*
    season's ECR-order sequence (board.PositionMarkov.from_sequences), then, for
    every adjacent pair in the test season's own sequence, record the trained
    model's P(actual next position | current position) alongside two baselines that
    have no notion of "current position" to compare against: marginal frequency
    (that position's overall share of picks in the training seasons) and uniform
    (1/4 flat). A `persistence` column (always predict "same as last") is also
    recorded for the accuracy comparison, since "assume the run continues" is the
    naive baseline a room-behavior model most needs to beat.
    """
    from . import board as bd

    per_season_seq: dict[int, list[str]] = {}
    for season in seasons:
        ecr = preseason_ecr(season)
        if ecr.empty:
            print(f"  ! {season}: no preseason ECR snapshot")
            continue
        seq = ecr.sort_values("ecr")["position"]
        seq = seq[seq.isin(bd.POSITIONS)].tolist()
        if len(seq) < 20:
            print(f"  ! {season}: too few ranked skill-position players ({len(seq)})")
            continue
        per_season_seq[season] = seq

    return _position_run_leave_one_out(per_season_seq, smoothing)


def _position_run_leave_one_out(per_season_seq: dict[int, list[str]],
                                smoothing: float = 1.0) -> pd.DataFrame:
    """Shared leave-one-season-out evaluation for position_run_backtest (ECR-order
    sequences) and real_draft_position_run_backtest (real completed draft order) --
    identical scoring either way, so position_run_backtest_summary works on both.
    """
    from . import board as bd

    if len(per_season_seq) < 2:
        return pd.DataFrame()

    rows = []
    for test_season, test_seq in per_season_seq.items():
        train_seqs = [seq for s, seq in per_season_seq.items() if s != test_season]
        model = bd.PositionMarkov.from_sequences(train_seqs, smoothing=smoothing)

        marginal_counts = {p: 0 for p in bd.POSITIONS}
        for seq in train_seqs:
            for p in seq:
                marginal_counts[p] += 1
        total = sum(marginal_counts.values()) or 1
        marginal_probs = {p: c / total for p, c in marginal_counts.items()}
        uniform_probs = {p: 1.0 / len(bd.POSITIONS) for p in bd.POSITIONS}

        for i in range(len(test_seq) - 1):
            current, actual_next = test_seq[i], test_seq[i + 1]
            markov_probs = model.transition_probs(current)
            rows.append({
                "season": test_season, "pick_idx": i,
                "current": current, "actual_next": actual_next,
                "markov_prob_actual": markov_probs.get(actual_next, 0.0),
                "marginal_prob_actual": marginal_probs.get(actual_next, 0.0),
                "uniform_prob_actual": uniform_probs.get(actual_next, 0.0),
                "markov_top1": max(markov_probs, key=markov_probs.get),
                "marginal_top1": max(marginal_probs, key=marginal_probs.get),
                "persistence_top1": current,
            })
    return pd.DataFrame(rows)


def real_draft_position_run_backtest(sequences: dict[int, list[str]],
                                     smoothing: float = 1.0) -> pd.DataFrame:
    """The real-data version of position_run_backtest: does PositionMarkov predict
    the next pick's position better than the same baselines, fit and evaluated on
    actual completed draft order instead of the ECR-order proxy?

    `sequences` maps season -> the real position-of-pick sequence for that draft
    (e.g. one real league's picks each year, resolved to positions and filtered to
    QB/RB/WR/TE via board.POSITIONS). Unlike position_run_backtest, this is
    reactive human behavior -- actual room psychology, not a fixed market ranking
    -- so it's the real test of whether PositionMarkov captures positional runs,
    not just whether consensus rank order is stable season to season. The
    trade-off is sample size: one league's five drafts is ~800-1000 real
    transitions total, split four ways for leave-one-season-out training, far
    thinner than what a market-order proxy across many more "seasons" could offer,
    so treat any result here as suggestive for *this room specifically*, not
    proof the mechanism generalizes to other draft rooms.

    Same leave-one-season-out evaluation as position_run_backtest, scored by the
    same position_run_backtest_summary. A real run against one league's 2021-2025
    drafts (694 real transitions, resolved via board.resolve_pick_positions) found
    the same answer the ECR-order proxy did, not a hidden signal the proxy was
    masking: Markov beat uniform (+0.098 logloss) and persistence (+0.086
    accuracy), but was a wash against marginal frequency on logloss (+0.0005,
    essentially zero) and slightly *worse* than it on accuracy (-0.010). Positional
    runs, at least in this room, genuinely aren't more predictable than base rates
    -- it isn't that the ADP-order proxy failed to capture something real.
    bootstrap_ci on the logloss gap actually straddles zero here (-0.0084 to
    +0.0097) -- genuine noise on real draft order, distinct from the ECR-proxy
    version's CI, which stayed entirely negative (see position_run_backtest).
    """
    return _position_run_leave_one_out(sequences, smoothing)


def position_run_backtest_summary(hist: pd.DataFrame) -> dict:
    """Score position_run_backtest's output: log-loss (lower is better, so
    improvement = baseline - markov) and top-1 accuracy for the Markov model
    against the marginal-frequency, uniform, and persistence baselines.

    A positive improvement_* means the Markov transition structure is earning its
    added complexity over that baseline. Near zero or negative means the simpler
    baseline predicts the next position just as well, and -- following the same
    rule redzone_shift_backtest's negative result set for `redzone_identity_shift`
    -- PositionMarkov shouldn't be wired into anything live until this comes back
    positive.
    """
    if hist.empty:
        return {"n_transitions": 0}

    def logloss(prob_col: str) -> float:
        p = hist[prob_col].clip(lower=1e-6)
        return float(-np.log(p).mean())

    def accuracy(top1_col: str) -> float:
        return float((hist[top1_col] == hist["actual_next"]).mean())

    markov_ll = logloss("markov_prob_actual")
    marginal_ll = logloss("marginal_prob_actual")
    uniform_ll = logloss("uniform_prob_actual")
    markov_acc = accuracy("markov_top1")
    marginal_acc = accuracy("marginal_top1")
    persistence_acc = accuracy("persistence_top1")

    return {
        "n_transitions": int(len(hist)),
        "seasons": sorted(int(s) for s in hist["season"].unique()),
        "markov_logloss": markov_ll,
        "marginal_logloss": marginal_ll,
        "uniform_logloss": uniform_ll,
        "improvement_logloss_vs_marginal": marginal_ll - markov_ll,
        "improvement_logloss_vs_uniform": uniform_ll - markov_ll,
        "markov_top1_accuracy": markov_acc,
        "marginal_top1_accuracy": marginal_acc,
        "persistence_top1_accuracy": persistence_acc,
        "improvement_accuracy_vs_marginal": markov_acc - marginal_acc,
        "improvement_accuracy_vs_persistence": markov_acc - persistence_acc,
    }


def position_scarcity_entropy_backtest(seasons: list[int], league=None, weights=None,
                                       window: int | None = None, stride: int = 4,
                                       cutoff: int = DRAFTABLE_ECR_CUTOFF) -> pd.DataFrame:
    """Does model.position_scarcity_entropy predict the real cost of waiting on a
    position, better than baselines that ignore the pool's shape -- including the
    survival-probability mechanism already live in recommend()?

    Same ground-truth limitation as position_run_backtest: there's no bulk archive
    of real snake-draft order, so "who's picked when" comes from that season's ADP
    order instead -- a leak-free board (build_player_table + project, bounded to
    seasons strictly before the one being tested, then ADP-joined, exactly like
    draft_backtest/mock_draft build it) sorted by ADP into a pseudo draft order.

    At sampled pick indices within the first `cutoff` picks, for each position with
    at least 2 players left in that pseudo order:
      - `entropy`: position_scarcity_entropy's reading of the *current* remaining pool
      - `cost` (the real target): how much the best remaining player at that position
        actually drops in draft_score once `window` more picks are removed in ADP
        order (window defaults to league.teams -- a proxy for "until your next turn")
      - `pool_size` and `raw_gap` (best minus second-best, unnormalized): two
        baselines that read the same pool but ignore its shape
      - `implied_cost`: what expected_best_at_next_pick -- the mechanism recommend()
        already runs live, using each player's own ADP-implied survival odds --
        predicts for that same horizon. Entropy needs to beat this, or at least not
        be redundant with it, to be worth adding rather than just recomputing what
        urgency/marginal_value already capture.

    `cost`, `raw_gap`, and `implied_cost` are all also recorded as a `_frac` version
    (divided by `value_now`, the current best remaining player's draft_score) --
    position_scarcity_entropy_backtest_summary scores against those, not the raw
    ones. Positions don't share a draft_score scale (QB's runs structurally larger
    than every other position's, per BACKUP_DECAY's own comment in model.py; WR's
    pool is far deeper than QB's per-position too), so pooling raw, unnormalized
    drops across positions before rank-correlating them measures "which position
    scores bigger" as much as it measures any real relationship -- confirmed by a
    first run of this backtest where entropy correlated *negatively* with raw `cost`
    pooled across positions (-0.195) despite a modestly *positive* correlation within
    every individual position (QB +0.23, RB +0.37, WR +0.22, TE -0.05), a Simpson's
    paradox from exactly this scale confound.

    Normalizing didn't rescue the metric, though -- it exposed a second, unrelated
    problem. Against `cost_frac`, the pooled correlation stayed negative (-0.153),
    and the within-position correlations that looked positive against raw `cost`
    flipped mostly negative too (QB -0.20, RB -0.35, WR -0.15, TE +0.10). That
    reversal means the original within-position positive reading wasn't real
    signal either: `value_now` and `entropy` both drift with pick depth (bigger
    early, smaller/flatter late), so raw `cost` and entropy trended together purely
    through depth as a lurking variable, not through any real relationship to each
    other. Once the target is normalized to remove that drift, entropy doesn't help
    -- if anything it points the wrong way. bootstrap_ci confirms this isn't noise:
    entropy_corr's 95% CI is entirely negative (-0.223 to -0.075).
    """
    from . import board as bd
    from . import model
    from .config import LeagueSettings, ModelWeights

    league = league or LeagueSettings()
    weights = weights or ModelWeights()
    sc_label = "ppr" if float(league.scoring.rec) >= 0.9 else \
               "half_ppr" if float(league.scoring.rec) >= 0.35 else "standard"
    window = window or league.teams

    rows = []
    for season in seasons:
        try:
            tbl = model.build_player_table(league, weights, season=season)
            proj = model.project(tbl, league, weights)
            adp = bd.load_adp(season=season)
            proj = bd.attach_adp(proj, adp)
            board = bd.convert_adp_format(proj, sc_label)
        except Exception as exc:
            print(f"  ! {season}: {type(exc).__name__}: {exc}")
            continue

        ordered = board.sort_values("adp").reset_index(drop=True)
        last_i = min(cutoff, len(ordered) - window) - 1
        if last_i < 1:
            print(f"  ! {season}: board too small for window={window}, cutoff={cutoff}")
            continue

        for i in range(0, last_i, stride):
            current_pick, next_pick = i + 1, i + 1 + window
            remaining = ordered.iloc[i:].copy()
            future = ordered.iloc[i + window:]

            remaining["p_available_next"] = model.survival_probability_vec(
                remaining["adp"].to_numpy(), current_pick, next_pick)
            expected_map = model.expected_best_at_next_pick(remaining)
            entropy_map = model.position_scarcity_entropy(remaining)

            for pos, chunk in remaining.groupby("position"):
                if pos not in bd.POSITIONS or len(chunk) < 2:
                    continue
                top2 = chunk.nlargest(2, "draft_score")["draft_score"].tolist()
                value_now, gap = top2[0], top2[0] - top2[1]

                fut_pos = future[future["position"] == pos]
                value_future = float(fut_pos["draft_score"].max()) if not fut_pos.empty else 0.0

                implied_cost = max(0.0, value_now - expected_map.get(pos, 0.0))
                cost = max(0.0, value_now - value_future)
                frac = (lambda x: x / value_now) if value_now > 0 else (lambda x: 0.0)

                rows.append({
                    "season": season, "pick_idx": current_pick, "position": pos,
                    "entropy": entropy_map.get(pos, 1.0), "pool_size": len(chunk),
                    "raw_gap": gap, "raw_gap_frac": frac(gap),
                    "implied_cost": implied_cost, "implied_cost_frac": frac(implied_cost),
                    "cost": cost, "cost_frac": frac(cost),
                })
    return pd.DataFrame(rows)


def position_scarcity_entropy_backtest_summary(hist: pd.DataFrame) -> dict:
    """Score position_scarcity_entropy_backtest's output.

    Scores against `cost_frac` (the drop as a fraction of the current best remaining
    player's own draft_score), not raw `cost` -- positions don't share a draft_score
    scale, so correlating a normalized target is what keeps this an honest test of
    "does entropy predict the shape of the drop" instead of "which position happens
    to score bigger" (see position_scarcity_entropy_backtest's docstring for the
    Simpson's-paradox result that motivated this). raw_gap_frac and implied_cost_frac
    are scored the same way for a fair comparison; pool_size is a plain count with no
    natural scale to normalize by, so it's left as-is.

    Every predictor is oriented so higher means "predicts a bigger drop": entropy is
    reported as `1 - entropy` (a lumpier, more concentrated pool reads as higher
    risk), pool_size is negated (fewer players left = more risk), and raw_gap_frac /
    implied_cost_frac are already oriented that way.

    A positive improvement_vs_* means entropy is catching real drop-off signal that
    baseline misses. Near zero or negative means the simpler baseline -- or, for
    improvement_vs_implied_cost, the survival-probability mechanism already live in
    recommend() -- predicts the actual cost of waiting just as well, and -- same rule
    position_run_backtest's result applied to PositionMarkov -- position_scarcity_entropy
    shouldn't get wired into recommend()/draft_score.
    """
    if hist.empty:
        return {"n_samples": 0}

    def spearman(a, b):
        return float(pd.Series(a).rank().corr(pd.Series(b).rank()))

    entropy_risk = 1.0 - hist["entropy"]
    pool_risk = -hist["pool_size"]
    target = hist["cost_frac"]

    entropy_corr = spearman(entropy_risk, target)
    gap_corr = spearman(hist["raw_gap_frac"], target)
    pool_corr = spearman(pool_risk, target)
    implied_corr = spearman(hist["implied_cost_frac"], target)

    return {
        "n_samples": int(len(hist)),
        "seasons": sorted(int(s) for s in hist["season"].unique()),
        "entropy_corr": entropy_corr,
        "raw_gap_corr": gap_corr,
        "pool_size_corr": pool_corr,
        "implied_cost_corr": implied_corr,
        "improvement_vs_raw_gap": entropy_corr - gap_corr,
        "improvement_vs_pool_size": entropy_corr - pool_corr,
        "improvement_vs_implied_cost": entropy_corr - implied_corr,
    }


def matchup_backtest_summary(hist: pd.DataFrame, top_n: int = 24) -> dict:
    """Compare talent-only vs matchup-adjusted score against actual finish.

    Spearman (rank) correlation against actual fantasy points, since raw fantasy
    points are heavily right-skewed and a rank-based measure is what actually
    matters for draft decisions. Also a top-N precision check computed per season
    then averaged: of the players each metric would have ranked in the top N, what
    share actually finished top N that season. N=24 is roughly the WR2-or-better
    cutoff in a 12-team league.

    A positive `improvement_corr` / `improvement_precision` means the matchup
    adjustment helped. Near zero or negative means talent alone did just as well,
    and the adjustment isn't earning its added complexity.
    """
    h = hist.dropna(subset=["talent_z", "matchup_adjusted_score", "points"]).copy()
    if h.empty:
        return {"n_player_seasons": 0}

    def spearman(a, b):
        return float(pd.Series(a).rank().corr(pd.Series(b).rank()))

    talent_precisions, matchup_precisions = [], []
    for _season, chunk in h.groupby("season"):
        chunk = chunk.copy()
        chunk["actual_top"] = chunk["points"].rank(ascending=False, method="min") <= top_n
        talent_top = chunk.sort_values("talent_z", ascending=False).head(top_n)
        matchup_top = chunk.sort_values("matchup_adjusted_score", ascending=False).head(top_n)
        if len(talent_top):
            talent_precisions.append(talent_top["actual_top"].mean())
        if len(matchup_top):
            matchup_precisions.append(matchup_top["actual_top"].mean())

    talent_corr = spearman(h["talent_z"], h["points"])
    matchup_corr = spearman(h["matchup_adjusted_score"], h["points"])
    talent_prec = float(np.mean(talent_precisions)) if talent_precisions else float("nan")
    matchup_prec = float(np.mean(matchup_precisions)) if matchup_precisions else float("nan")

    return {
        "n_player_seasons": int(len(h)),
        "seasons": sorted(int(s) for s in h["season"].unique()),
        "top_n": top_n,
        "talent_only_corr": talent_corr,
        "matchup_adjusted_corr": matchup_corr,
        "improvement_corr": matchup_corr - talent_corr,
        "talent_only_top_n_precision": talent_prec,
        "matchup_adjusted_top_n_precision": matchup_prec,
        "improvement_precision": matchup_prec - talent_prec,
    }


def vacated_role_backtest(seasons: list[int], positions: tuple[str, ...] = ("WR", "TE"),
                          min_games: int = 6) -> pd.DataFrame:
    """Does knowing who already has volume tell you who benefits when a same-team,
    same-position starter misses a game? The general version of the Chase-out/
    Higgins-in question: not one hand-picked pair, but every real qualifying
    absence in `seasons`.

    Thin wrapper around features.vacated_role_events -- real box scores only, no
    leak-free bound needed the way projection-based backtests need one, since this
    is scoring a predictor (trailing target_share, using only weeks before the
    absence) against a real historical outcome (the target_share teammates actually
    posted that week), not projecting anything about a season that hasn't happened.
    """
    from . import features

    w = sources.weekly_stats(seasons)
    w = w[w["season_type"] == "REG"]
    byes = {s: features.bye_weeks(s) for s in seasons}
    return features.vacated_role_events(w, byes, positions=positions, min_games=min_games)


def vacated_role_backtest_summary(hist: pd.DataFrame) -> dict:
    """Score vacated_role_backtest's output two ways:

    `trailing_share_vs_lift_corr`: Spearman correlation, pooled across every
    teammate-event row, between trailing_share (the predictor) and share_lift (the
    real redistribution that week). No cross-position scale confound to worry about
    here the way position_scarcity_entropy_backtest had -- target_share is already
    a common 0-1 scale, not an absolute draft_score that differs by position.

    `top1_accuracy`: for each distinct absence (one team, one week, one player out),
    does the teammate with the *highest* trailing_share also turn out to be the one
    with the *biggest* actual share_lift that week? Compared against
    `random_baseline_accuracy` (1/n_candidates, averaged per event) -- the chance
    accuracy this many candidate teammates would produce by construction, since a
    3-candidate race has an easier baseline to beat than a 5-candidate one.

    A positive improvement_vs_random and a clearly positive corr together would mean
    this is worth building into a live "who benefits if X sits" read. Same rule as
    every other backtest this session: if it comes back weak, it stays a documented
    finding, not a live feature. A 2021-2025 run (13,507 teammate-rows, 3,950
    absences) found exactly that ambiguity: top1_accuracy barely beat the random
    baseline (0.371 vs. 0.345, +0.025), while trailing_share_vs_lift_corr was
    actually negative (-0.118) -- existing volume gives a slight edge at naming the
    single biggest gainer, but doesn't track the size of anyone's bump, plausibly a
    ceiling effect (a player already getting a lot of targets has less room left to
    grow proportionally). bootstrap_ci turns that "barely" into two distinct, both
    real findings: trailing_share_vs_lift_corr's CI is entirely negative (-0.139 to
    -0.095, a reliable failure at bump *size*), while top1_accuracy and
    improvement_vs_random's CIs never cross zero (0.371-0.488, 0.025-0.293) -- a
    small but real edge at *who*, just not *how much*. Not clean enough to wire in
    either way.
    """
    if hist.empty:
        return {"n_events": 0}

    def spearman(a, b):
        return float(pd.Series(a).rank().corr(pd.Series(b).rank()))

    corr = spearman(hist["trailing_share"], hist["share_lift"])

    hits, n_candidates = [], []
    for _, ev in hist.groupby(["season", "team", "week", "player_out"]):
        if len(ev) < 2:
            continue
        predicted = ev.loc[ev["trailing_share"].idxmax(), "teammate"]
        actual_best = ev.loc[ev["share_lift"].idxmax(), "teammate"]
        hits.append(predicted == actual_best)
        n_candidates.append(len(ev))

    accuracy = float(np.mean(hits)) if hits else float("nan")
    random_baseline = float(np.mean([1.0 / n for n in n_candidates])) if n_candidates else float("nan")

    return {
        "n_teammate_rows": int(len(hist)),
        "n_absence_events": len(hits),
        "seasons": sorted(int(s) for s in hist["season"].unique()),
        "trailing_share_vs_lift_corr": corr,
        "top1_accuracy": accuracy,
        "random_baseline_accuracy": random_baseline,
        "improvement_vs_random": accuracy - random_baseline,
    }


def hc_change_role_volatility(seasons: list[int], positions: tuple[str, ...] = ("WR", "TE"),
                              min_games: int = 6) -> pd.DataFrame:
    """One row per player who played real snaps for the *same* team in both
    `season - 1` and `season`: does his team's head coach changing entering
    `season` predict a bigger year-over-year shift in his role than a team with
    coaching continuity?

    Restricted to players who stayed on the same team in both seasons (the merge
    on `player_id` + `recent_team` enforces this) specifically to isolate the
    coaching-change question from the separate, already-modelled question of what
    happens when a *player* changes teams (apply_current_team already re-grades a
    traded player on his new team's O-line/pace/schedule).

    `role_shift` is the absolute year-over-year change in target_share -- the
    magnitude, not the direction, since a coaching change could plausibly raise or
    lower a given player's specific share; what matters for "is last year's
    history still trustworthy" is how much it moved, either way.
    """
    from . import features

    rows = []
    for season in seasons:
        try:
            hc_changed = features.head_coach_changes(season)
            if not hc_changed:
                print(f"  ! {season}: no head coach data (schedule not published?)")
                continue

            def role(szn):
                w = sources.weekly_stats([szn])
                w = w[(w["season_type"] == "REG") & (w["position"].isin(positions))]
                return (w.groupby(["player_id", "player_display_name", "recent_team"])
                       .agg(games=("week", "nunique"), target_share=("target_share", "mean"))
                       .reset_index())

            prior = role(season - 1)
            cur = role(season)
            prior_q = prior[prior["games"] >= min_games]
            cur_q = cur[cur["games"] >= min_games]

            m = prior_q.merge(cur_q, on=["player_id", "recent_team"], suffixes=("_prior", "_cur"))
            if m.empty:
                continue
            m["role_shift"] = (m["target_share_cur"] - m["target_share_prior"]).abs()
            m["hc_changed"] = m["recent_team"].map(hc_changed).fillna(False)
            m["season"] = season
            rows.append(m.rename(columns={"player_display_name_cur": "name",
                                         "recent_team": "team"})
                       [["season", "player_id", "name", "team", "target_share_prior",
                        "target_share_cur", "role_shift", "hc_changed"]])
        except Exception as exc:
            print(f"  ! {season}: {type(exc).__name__}: {exc}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def hc_change_role_volatility_summary(hist: pd.DataFrame) -> dict:
    """Score hc_change_role_volatility's output: do players on a team with a new
    head coach actually show a bigger year-over-year role shift than players on a
    team with coaching continuity?

    `difference_in_means` = mean role_shift (HC changed) - mean role_shift (HC
    same). Positive means a coaching change really does predict more role
    volatility -- worth folding into how much a team's O-line/pace/redzone-identity
    history should be trusted for players there. Near zero or negative means
    coaching turnover isn't actually more disruptive than normal year-to-year
    noise, and -- same rule as every other backtest this session -- it stays a
    documented finding, not a live discount, until it clears that bar.

    A 2021-2025 run (715 same-team player-seasons, 141 with a coaching change)
    cleared that bar, unlike every new signal tested earlier this session:
    difference_in_means +0.0088 target_share points (mean role_shift 0.0428 on a
    changed team vs. 0.0340 with continuity, roughly 26% bigger), and
    bootstrap_ci's 95% interval is entirely positive (+0.0047 to +0.0139) -- not
    noise. This is the first sketch this session found that actually validates.
    Still not wired into any live feature (that would mean deciding exactly how
    much to discount team-history trust and where -- a design choice, not
    something this backtest alone settles), but unlike PositionMarkov,
    position_scarcity_entropy, and vacated-role insurance, this one earned the
    right to be considered for it.
    """
    if hist.empty:
        return {"n_players": 0}
    changed = hist[hist["hc_changed"]]
    same = hist[~hist["hc_changed"]]
    return {
        "n_players": int(len(hist)),
        "n_hc_changed": int(len(changed)),
        "n_hc_same": int(len(same)),
        "seasons": sorted(int(s) for s in hist["season"].unique()),
        "mean_role_shift_hc_changed": float(changed["role_shift"].mean()) if not changed.empty else float("nan"),
        "mean_role_shift_hc_same": float(same["role_shift"].mean()) if not same.empty else float("nan"),
        "median_role_shift_hc_changed": float(changed["role_shift"].median()) if not changed.empty else float("nan"),
        "median_role_shift_hc_same": float(same["role_shift"].median()) if not same.empty else float("nan"),
        "difference_in_means": (float(changed["role_shift"].mean() - same["role_shift"].mean())
                                if not changed.empty and not same.empty else float("nan")),
    }


_HC_SHRINKAGE_LEVELS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)


def hc_change_shrinkage_backtest(seasons: list[int], positions: tuple[str, ...] = ("WR", "TE"),
                                 min_games: int = 6,
                                 shrinkage_levels: tuple[float, ...] = _HC_SHRINKAGE_LEVELS
                                 ) -> pd.DataFrame:
    """hc_change_role_volatility_backtest showed *that* a coaching change makes a
    player's own recent-season role less trustworthy as a predictor. This tests
    the actual fix: does shrinking his trailing target_share toward the position
    average -- trusting his own history less, a positional baseline more --
    predict his *real* target_share that season more accurately than trusting his
    own history at full weight (shrinkage=0.0, today's implicit behavior, since
    nothing currently discounts a coaching-change player's role inputs at all)?

    Restricted to the same-team, coaching-change cohort hc_change_role_volatility
    identified -- shrinkage is a claim about players whose team's continuity broke,
    not about players generally, so testing it on everyone would dilute the
    question this is actually asking. `pos_baseline` (the shrink target) is the
    *prior* season's average target_share among other qualifying players at that
    position, leak-free -- what a drafter could have known before the season
    being predicted, the same standard every other backtest here uses. shrinkage=
    1.0 means ignore the player's own history entirely and just guess the position
    average; shrinkage=0.0 means trust it completely, ignoring the coaching change.
    """
    from . import features

    rows = []
    for season in seasons:
        try:
            hc_changed = features.head_coach_changes(season)
            if not hc_changed:
                print(f"  ! {season}: no head coach data (schedule not published?)")
                continue

            w_prior = sources.weekly_stats([season - 1])
            w_prior = w_prior[(w_prior["season_type"] == "REG")
                              & (w_prior["position"].isin(positions))]
            w_cur = sources.weekly_stats([season])
            w_cur = w_cur[(w_cur["season_type"] == "REG") & (w_cur["position"].isin(positions))]

            prior_agg = (w_prior.groupby(["player_id", "recent_team", "position"])
                        .agg(games=("week", "nunique"), target_share=("target_share", "mean"))
                        .reset_index())
            cur_agg = (w_cur.groupby(["player_id", "recent_team"])
                      .agg(games=("week", "nunique"), target_share=("target_share", "mean"))
                      .reset_index())

            prior_q = prior_agg[prior_agg["games"] >= min_games]
            cur_q = cur_agg[cur_agg["games"] >= min_games]
            if prior_q.empty:
                continue
            # Leak-free position baseline: only ever the prior season's own
            # cohort, never anything from the season being predicted.
            pos_baseline = prior_q.groupby("position")["target_share"].mean().to_dict()

            m = prior_q.merge(cur_q, on=["player_id", "recent_team"], suffixes=("_prior", "_cur"))
            if m.empty:
                continue
            m["hc_changed"] = m["recent_team"].map(hc_changed).fillna(False)
            m = m[m["hc_changed"]].copy()
            if m.empty:
                continue
            m["pos_baseline"] = m["position"].map(pos_baseline)
            m["season"] = season
            rows.append(m[["season", "player_id", "position", "target_share_prior",
                          "target_share_cur", "pos_baseline"]])
        except Exception as exc:
            print(f"  ! {season}: {type(exc).__name__}: {exc}")

    hist = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if hist.empty:
        return hist
    for lv in shrinkage_levels:
        pred = (1 - lv) * hist["target_share_prior"] + lv * hist["pos_baseline"]
        hist[f"abs_err_shrink_{lv}"] = (pred - hist["target_share_cur"]).abs()
    return hist


def hc_change_shrinkage_summary(hist: pd.DataFrame,
                                shrinkage_levels: tuple[float, ...] = _HC_SHRINKAGE_LEVELS) -> dict:
    """Score hc_change_shrinkage_backtest's output: which shrinkage level (if any)
    predicts the coaching-change cohort's real target_share best?

    `improvement_vs_no_shrinkage` = mean absolute error at shrinkage=0.0 (trust the
    player's own history fully, today's behavior) minus the best level found --
    positive means some real shrinkage toward the position baseline predicts
    better than fully trusting a coaching-change player's own recent history, and
    is worth wiring into the live projection. Near zero or negative means shrinkage
    doesn't actually help even for this specific, already-validated-as-volatile
    cohort, and the discount shouldn't be built despite hc_change_role_volatility's
    positive result -- knowing a group is *more volatile* doesn't automatically
    mean a specific correction *reduces error* for them.

    A 2021-2025 run (141 coaching-change players) found a real but modest, only
    borderline-significant effect -- a genuinely different, weaker verdict than
    hc_change_role_volatility's clean positive: light shrinkage (0.2, i.e.
    80% own history / 20% position baseline) minimized error at 0.0407 vs.
    no-shrinkage's 0.0428 (+0.0021, ~5% relative). Error rises past that --
    aggressive shrinkage (1.0, ignore his own history entirely) is worse than no
    shrinkage at all (0.0606), meaning a coaching-change player's own talent/role
    still carries real signal that full reversion throws away. The 95% CI is
    [0.0, +0.0039] -- the lower bound touching exactly zero, and that's before
    accounting for the fact that this CI re-picks whichever level looks best in
    each resample rather than testing one level fixed in advance (a "best of
    several tries" comparison, which reads more confident than a single
    pre-committed one would). Real, but not the clean win hc_change_role_volatility
    was -- a judgment call whether a ~5% error reduction this marginal is worth
    wiring into the live projection.
    """
    if hist.empty:
        return {"n_players": 0}
    errs = {lv: float(hist[f"abs_err_shrink_{lv}"].mean())
           for lv in shrinkage_levels if f"abs_err_shrink_{lv}" in hist.columns}
    best = min(errs, key=errs.get) if errs else None
    baseline_err = errs.get(0.0)
    return {
        "n_players": int(len(hist)),
        "seasons": sorted(int(s) for s in hist["season"].unique()),
        "mean_abs_error_by_shrinkage": errs,
        "best_shrinkage": best,
        "best_mean_abs_error": errs.get(best) if best is not None else None,
        "no_shrinkage_mean_abs_error": baseline_err,
        "improvement_vs_no_shrinkage": (baseline_err - errs[best])
                                       if best is not None and baseline_err is not None else None,
    }


_QB_MIN_DROPBACKS = 150


def hc_change_qb_efficiency_volatility(seasons: list[int],
                                       min_dropbacks: int = _QB_MIN_DROPBACKS) -> pd.DataFrame:
    """The Caleb Williams / Jared Goff observation, generalized: one row per QB who
    started for the *same* team in both `season - 1` and `season` -- does his
    team's head coach changing entering `season` predict a bigger year-over-year
    shift in passing efficiency (EPA per dropback) than coaching continuity, the
    same question hc_change_role_volatility asked about WR/TE target_share?

    Restricted the same way: the merge on player_id + recent_team enforces "same
    team both seasons," isolating a coaching-change effect from a player simply
    changing teams. `min_dropbacks` (attempts + sacks_suffered), not min_games --
    a QB's efficiency reading needs real pass-game volume to mean anything, and
    games played alone doesn't guarantee that (a QB pulled early in a few blowouts
    could clear a games threshold on very few actual dropbacks).

    role_shift is the absolute year-over-year change in EPA/dropback -- magnitude,
    not direction, since a coaching change can plausibly help (Ben Johnson arriving
    in Chicago raised Caleb Williams' efficiency) or hurt (Ben Johnson leaving
    Detroit lowered Jared Goff's) -- what matters here is how much a coaching
    change moves the number, either way.

    IMPORTANT limitation, sharper here than anywhere else this session's used
    features.head_coach_changes: that Detroit/Goff case is itself invisible to
    this backtest's cohort. Ben Johnson left as *offensive coordinator*; Dan
    Campbell stayed head coach throughout. head_coach_changes only sees head
    coach turnover (there's no clean public OC dataset -- see its docstring), so
    the exact real-world example that motivated this function can never appear
    in the "hc_changed" group it builds. Every case below is a real head-coach
    change instead (e.g. the Jets hiring Aaron Glenn for 2025) -- a related but
    distinct question: does *head-coach* turnover specifically, most of which
    leaves the offensive coordinator and scheme untouched, still show up as
    bigger QB volatility?

    A 2021-2025 run (114 same-team QB-seasons clearing 150 dropbacks in both
    years, 23 with a head-coach change) found no: mean role_shift was actually
    about the same either way (0.112 EPA/dropback with a coaching change vs.
    0.105 with continuity, +0.0071), and bootstrap_ci's 95% interval straddles
    zero (-0.0287 to +0.0605) -- not distinguishable from noise, unlike
    hc_change_role_volatility's clean positive result for WR/TE target_share.
    The most plausible reason, given the limitation above: most head-coach
    hires don't change who calls plays on offense, so lumping every HC change
    together dilutes whatever real OC-driven effect exists (visible in single
    cases like Williams/Johnson) into noise at the aggregate level. A cohort of
    confirmed *playcaller* changes, not just HC changes, is the test this
    backtest can't run without that missing OC dataset.
    """
    from . import features

    rows = []
    for season in seasons:
        try:
            hc_changed = features.head_coach_changes(season)
            if not hc_changed:
                print(f"  ! {season}: no head coach data (schedule not published?)")
                continue

            def qb_efficiency(szn):
                w = sources.weekly_stats([szn])
                w = w[(w["season_type"] == "REG") & (w["position"] == "QB")]
                w = w.assign(dropbacks=w["attempts"].fillna(0) + w["sacks_suffered"].fillna(0))
                g = (w.groupby(["player_id", "player_display_name", "recent_team"])
                    .agg(games=("week", "nunique"), dropbacks=("dropbacks", "sum"),
                         passing_epa=("passing_epa", "sum"))
                    .reset_index())
                g["epa_per_db"] = g["passing_epa"] / g["dropbacks"].replace(0, np.nan)
                return g

            prior = qb_efficiency(season - 1)
            cur = qb_efficiency(season)
            prior_q = prior[prior["dropbacks"] >= min_dropbacks].dropna(subset=["epa_per_db"])
            cur_q = cur[cur["dropbacks"] >= min_dropbacks].dropna(subset=["epa_per_db"])
            if prior_q.empty:
                continue

            m = prior_q.merge(cur_q, on=["player_id", "recent_team"], suffixes=("_prior", "_cur"))
            if m.empty:
                continue
            m["role_shift"] = (m["epa_per_db_cur"] - m["epa_per_db_prior"]).abs()
            m["hc_changed"] = m["recent_team"].map(hc_changed).fillna(False)
            m["season"] = season
            rows.append(m.rename(columns={"player_display_name_cur": "name",
                                          "recent_team": "team"})
                       [["season", "player_id", "name", "team", "epa_per_db_prior",
                        "epa_per_db_cur", "role_shift", "hc_changed"]])
        except Exception as exc:
            print(f"  ! {season}: {type(exc).__name__}: {exc}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def hc_change_qb_efficiency_volatility_summary(hist: pd.DataFrame) -> dict:
    """Score hc_change_qb_efficiency_volatility's output: the same
    difference-in-means test hc_change_role_volatility_summary runs for WR/TE
    target_share, applied here to QB EPA/dropback.

    Near zero or negative difference_in_means -- what the 2021-2025 real run
    found -- means head-coach turnover doesn't reliably predict bigger QB
    efficiency swings than continuity, and per this session's standing rule,
    this stays a documented (negative) finding rather than a live discount.
    See hc_change_qb_efficiency_volatility's docstring for why: this cohort is
    built from head-coach changes, most of which don't touch the play-caller,
    so it can't see the specific offensive-coordinator effect the Williams/Goff
    cases actually demonstrate.
    """
    if hist.empty:
        return {"n_players": 0}
    changed = hist[hist["hc_changed"]]
    same = hist[~hist["hc_changed"]]
    return {
        "n_players": int(len(hist)),
        "n_hc_changed": int(len(changed)),
        "n_hc_same": int(len(same)),
        "seasons": sorted(int(s) for s in hist["season"].unique()),
        "mean_role_shift_hc_changed": float(changed["role_shift"].mean()) if not changed.empty else float("nan"),
        "mean_role_shift_hc_same": float(same["role_shift"].mean()) if not same.empty else float("nan"),
        "median_role_shift_hc_changed": float(changed["role_shift"].median()) if not changed.empty else float("nan"),
        "median_role_shift_hc_same": float(same["role_shift"].median()) if not same.empty else float("nan"),
        "difference_in_means": (float(changed["role_shift"].mean() - same["role_shift"].mean())
                                if not changed.empty and not same.empty else float("nan")),
    }


def hc_change_qb_efficiency_shrinkage_backtest(seasons: list[int],
                                               min_dropbacks: int = _QB_MIN_DROPBACKS,
                                               shrinkage_levels: tuple[float, ...] = _HC_SHRINKAGE_LEVELS
                                               ) -> pd.DataFrame:
    """Unlike hc_change_shrinkage_backtest, this keeps BOTH cohorts -- head-coach
    changed and continuity -- in the same table, not just the changed one. The
    real question worth asking here isn't "does shrinking a QB's EPA/dropback
    toward the position baseline predict his real efficiency better than trusting
    his own history" (year-to-year QB efficiency is famously volatile for
    everyone, coaching change or not, so shrinkage was always likely to help some
    cohort); it's "does it help the head-coach-change cohort by *more* than it
    helps a coaching-continuity cohort" -- the only version of this test that
    would actually validate a coaching-specific discount rather than rediscovering
    generic QB mean-reversion. `pos_baseline` (the shrink target) is the *prior*
    season's average EPA/dropback among all qualifying starters, leak-free --
    computed once per season from every starter, not separately for each cohort,
    so both groups are being pulled toward the same target.

    A 2021-2025 run found real error reduction from shrinkage in *both* cohorts:
    best level 0.7 for the head-coach-change group cut mean absolute error from
    0.1117 (no shrinkage) to 0.0850 (-24%), and best level 0.4 for the continuity
    group cut it from 0.1047 to 0.0912 (-13%). The gap between those two
    improvements (+0.0133, coaching-change benefiting more) is exactly the
    "coaching-specific" effect hc_change_shrinkage_backtest validated for WR/TE
    role -- but here bootstrap_ci's 95% interval on that gap straddles zero
    (-0.0073 to +0.0381). Shrinkage is a good idea for any QB's trailing
    efficiency reading, coaching change or not; this backtest can't show it's
    *especially* good for the coaching-change cohort specifically. Left as a
    documented negative finding, same as hc_change_qb_efficiency_volatility.
    """
    from . import features

    rows = []
    for season in seasons:
        try:
            hc_changed = features.head_coach_changes(season)
            if not hc_changed:
                print(f"  ! {season}: no head coach data (schedule not published?)")
                continue

            w_prior = sources.weekly_stats([season - 1])
            w_prior = w_prior[(w_prior["season_type"] == "REG") & (w_prior["position"] == "QB")]
            w_prior = w_prior.assign(
                dropbacks=w_prior["attempts"].fillna(0) + w_prior["sacks_suffered"].fillna(0))
            w_cur = sources.weekly_stats([season])
            w_cur = w_cur[(w_cur["season_type"] == "REG") & (w_cur["position"] == "QB")]
            w_cur = w_cur.assign(
                dropbacks=w_cur["attempts"].fillna(0) + w_cur["sacks_suffered"].fillna(0))

            prior_agg = (w_prior.groupby(["player_id", "recent_team"])
                        .agg(dropbacks=("dropbacks", "sum"), passing_epa=("passing_epa", "sum"))
                        .reset_index())
            prior_agg["epa_per_db"] = prior_agg["passing_epa"] / prior_agg["dropbacks"].replace(0, np.nan)
            cur_agg = (w_cur.groupby(["player_id", "recent_team"])
                      .agg(dropbacks=("dropbacks", "sum"), passing_epa=("passing_epa", "sum"))
                      .reset_index())
            cur_agg["epa_per_db"] = cur_agg["passing_epa"] / cur_agg["dropbacks"].replace(0, np.nan)

            prior_q = prior_agg[prior_agg["dropbacks"] >= min_dropbacks].dropna(subset=["epa_per_db"])
            cur_q = cur_agg[cur_agg["dropbacks"] >= min_dropbacks].dropna(subset=["epa_per_db"])
            if prior_q.empty:
                continue
            # Leak-free position baseline: only the prior season's own cohort,
            # shared by both the changed and continuity groups below.
            pos_baseline = float(prior_q["epa_per_db"].mean())

            m = prior_q.merge(cur_q, on=["player_id", "recent_team"], suffixes=("_prior", "_cur"))
            if m.empty:
                continue
            m["hc_changed"] = m["recent_team"].map(hc_changed).fillna(False)
            m["pos_baseline"] = pos_baseline
            m["season"] = season
            rows.append(m[["season", "player_id", "epa_per_db_prior", "epa_per_db_cur",
                          "pos_baseline", "hc_changed"]])
        except Exception as exc:
            print(f"  ! {season}: {type(exc).__name__}: {exc}")

    hist = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if hist.empty:
        return hist
    for lv in shrinkage_levels:
        pred = (1 - lv) * hist["epa_per_db_prior"] + lv * hist["pos_baseline"]
        hist[f"abs_err_shrink_{lv}"] = (pred - hist["epa_per_db_cur"]).abs()
    return hist


def hc_change_qb_efficiency_shrinkage_summary(hist: pd.DataFrame,
                                              shrinkage_levels: tuple[float, ...] = _HC_SHRINKAGE_LEVELS
                                              ) -> dict:
    """Score hc_change_qb_efficiency_shrinkage_backtest's output for both cohorts,
    and report the gap between them -- `improvement_is_coaching_specific` -- which
    is the number that actually matters. Near zero or negative means shrinkage
    helps any QB's trailing efficiency reading about equally, coaching change or
    not, and per this session's rule this stays a documented finding rather than
    a live discount. See the backtest's docstring for the real 2021-2025 numbers
    (gap +0.0133, 95% CI -0.0073 to +0.0381 -- not distinguishable from zero).
    """
    if hist.empty:
        return {"n_players": 0}

    def errs_for(g: pd.DataFrame) -> tuple[dict, float | None]:
        e = {lv: float(g[f"abs_err_shrink_{lv}"].mean())
            for lv in shrinkage_levels if f"abs_err_shrink_{lv}" in g.columns}
        best = min(e, key=e.get) if e else None
        return e, best

    changed = hist[hist["hc_changed"]]
    same = hist[~hist["hc_changed"]]
    errs_c, best_c = errs_for(changed) if not changed.empty else ({}, None)
    errs_s, best_s = errs_for(same) if not same.empty else ({}, None)
    improve_c = (errs_c.get(0.0) - errs_c[best_c]) if best_c is not None else None
    improve_s = (errs_s.get(0.0) - errs_s[best_s]) if best_s is not None else None

    return {
        "n_hc_changed": int(len(changed)),
        "n_hc_same": int(len(same)),
        "seasons": sorted(int(s) for s in hist["season"].unique()),
        "mean_abs_error_by_shrinkage_hc_changed": errs_c,
        "mean_abs_error_by_shrinkage_hc_same": errs_s,
        "best_shrinkage_hc_changed": best_c,
        "best_shrinkage_hc_same": best_s,
        "improvement_vs_no_shrinkage_hc_changed": improve_c,
        "improvement_vs_no_shrinkage_hc_same": improve_s,
        "improvement_is_coaching_specific": (improve_c - improve_s)
                                           if improve_c is not None and improve_s is not None else None,
    }


def oc_change_qb_efficiency_volatility(seasons: list[int],
                                       min_dropbacks: int = _QB_MIN_DROPBACKS) -> pd.DataFrame:
    """The real-offensive-coordinator version of hc_change_qb_efficiency_volatility,
    now that features.offensive_coordinator_changes exists -- does a team's actual
    play-caller changing entering `season` predict a bigger year-over-year shift in
    its QB's passing efficiency (EPA/dropback) than continuity, when "coaching
    change" means the play-caller specifically rather than head-coach turnover in
    general?

    Same structure as hc_change_qb_efficiency_volatility (same-team merge on
    player_id + recent_team, min_dropbacks over games as the qualifying bar,
    role_shift as the unsigned year-over-year change), swapping in
    features.offensive_coordinator_changes for features.head_coach_changes.

    A 2021-2025 run (110 same-team QB-seasons clearing 150 dropbacks both years,
    53 with a real OC change) found what hc_change_qb_efficiency_volatility
    couldn't: mean role_shift was 0.1206 EPA/dropback with an OC change vs. 0.0942
    with continuity (+0.0263, ~28% bigger), and bootstrap_ci's 95% interval is
    entirely positive (+0.0013 to +0.0497) -- not noise, and a real validation of
    the pattern the Williams/Goff case studies suggested. Confirms the diagnosis
    in hc_change_qb_efficiency_volatility's docstring: the null result there was
    a data-availability artifact (most head-coach hires don't touch the
    play-caller), not evidence the underlying effect isn't real.

    IMPORTANT caveat this doesn't resolve: role_shift is unsigned. This says a new
    play-caller reliably makes a QB's efficiency *less predictable*, not which way
    it moves -- Ben Johnson arriving in Chicago raised Caleb Williams' efficiency;
    Ben Johnson leaving Detroit lowered Jared Goff's. See
    oc_change_qb_efficiency_shrinkage_backtest for why that distinction matters:
    an undirected volatility finding doesn't automatically license a specific
    score correction.
    """
    from . import features

    rows = []
    for season in seasons:
        try:
            oc_changed = features.offensive_coordinator_changes(season)
            if not oc_changed:
                print(f"  ! {season}: no OC-change data available")
                continue

            def qb_efficiency(szn):
                w = sources.weekly_stats([szn])
                w = w[(w["season_type"] == "REG") & (w["position"] == "QB")]
                w = w.assign(dropbacks=w["attempts"].fillna(0) + w["sacks_suffered"].fillna(0))
                g = (w.groupby(["player_id", "player_display_name", "recent_team"])
                    .agg(games=("week", "nunique"), dropbacks=("dropbacks", "sum"),
                         passing_epa=("passing_epa", "sum"))
                    .reset_index())
                g["epa_per_db"] = g["passing_epa"] / g["dropbacks"].replace(0, np.nan)
                return g

            prior = qb_efficiency(season - 1)
            cur = qb_efficiency(season)
            prior_q = prior[prior["dropbacks"] >= min_dropbacks].dropna(subset=["epa_per_db"])
            cur_q = cur[cur["dropbacks"] >= min_dropbacks].dropna(subset=["epa_per_db"])
            if prior_q.empty:
                continue

            m = prior_q.merge(cur_q, on=["player_id", "recent_team"], suffixes=("_prior", "_cur"))
            if m.empty:
                continue
            m["role_shift"] = (m["epa_per_db_cur"] - m["epa_per_db_prior"]).abs()
            m["oc_changed"] = m["recent_team"].map(oc_changed)
            m = m.dropna(subset=["oc_changed"])
            if m.empty:
                continue
            m["oc_changed"] = m["oc_changed"].astype(bool)
            m["season"] = season
            rows.append(m.rename(columns={"player_display_name_cur": "name",
                                          "recent_team": "team"})
                       [["season", "player_id", "name", "team", "epa_per_db_prior",
                        "epa_per_db_cur", "role_shift", "oc_changed"]])
        except Exception as exc:
            print(f"  ! {season}: {type(exc).__name__}: {exc}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def oc_change_qb_efficiency_volatility_summary(hist: pd.DataFrame) -> dict:
    """Score oc_change_qb_efficiency_volatility's output: the same
    difference-in-means test hc_change_role_volatility_summary runs for WR/TE
    target_share and hc_change_qb_efficiency_volatility_summary runs (and fails
    to validate) for head-coach-only QB changes.

    A positive difference_in_means -- what the 2021-2025 real run found
    (+0.0263, 95% CI +0.0013 to +0.0497, entirely positive) -- means real OC
    turnover does predict more QB efficiency volatility than continuity. Unlike
    every prior null result this session found for QB coaching effects, this one
    clears the bar. See oc_change_qb_efficiency_shrinkage_backtest before
    treating that as license to adjust a QB's projected score, though --
    role_shift being unsigned means this validates "less predictable," not
    "predictably worse" or "predictably better."
    """
    if hist.empty:
        return {"n_players": 0}
    changed = hist[hist["oc_changed"]]
    same = hist[~hist["oc_changed"]]
    return {
        "n_players": int(len(hist)),
        "n_oc_changed": int(len(changed)),
        "n_oc_same": int(len(same)),
        "seasons": sorted(int(s) for s in hist["season"].unique()),
        "mean_role_shift_oc_changed": float(changed["role_shift"].mean()) if not changed.empty else float("nan"),
        "mean_role_shift_oc_same": float(same["role_shift"].mean()) if not same.empty else float("nan"),
        "median_role_shift_oc_changed": float(changed["role_shift"].median()) if not changed.empty else float("nan"),
        "median_role_shift_oc_same": float(same["role_shift"].median()) if not same.empty else float("nan"),
        "difference_in_means": (float(changed["role_shift"].mean() - same["role_shift"].mean())
                                if not changed.empty and not same.empty else float("nan")),
    }


def oc_change_qb_efficiency_shrinkage_backtest(seasons: list[int],
                                               min_dropbacks: int = _QB_MIN_DROPBACKS,
                                               shrinkage_levels: tuple[float, ...] = _HC_SHRINKAGE_LEVELS
                                               ) -> pd.DataFrame:
    """The real-OC version of hc_change_qb_efficiency_shrinkage_backtest -- now
    that oc_change_qb_efficiency_volatility has validated the underlying effect,
    does shrinking a QB's trailing EPA/dropback toward the position baseline
    actually predict his real efficiency better for the OC-changed cohort
    *specifically*, more than it helps a coaching-continuity cohort (the only
    version of this test that would license an actual score correction, per the
    same logic hc_change_qb_efficiency_shrinkage_backtest used)?

    Same structure: both cohorts kept in the same table, shared leak-free
    `pos_baseline` from the prior season's full qualifying cohort.

    A 2021-2025 run found the answer is no, despite the volatility finding being
    real: best shrinkage (0.7) cut the OC-change cohort's error 11% (0.1206 ->
    0.1068); best shrinkage (0.4) cut the continuity cohort's error 18% (0.0942
    -> 0.0777) -- shrinkage helped the *continuity* cohort more, not the
    OC-change one. The gap (-0.0028, negative) has a 95% CI of -0.0137 to
    +0.0124 -- indistinguishable from zero either way. This is the expected
    consequence of role_shift being unsigned: shrinkage only reduces error when
    the deviation trends back toward the baseline, but a new play-caller can push
    a QB's efficiency up (Williams) or down (Goff) roughly evenly, so pulling
    every OC-change QB's number toward the average helps some and hurts others
    in a way that washes out on net. The volatility finding stands -- a
    new-OC QB's trailing efficiency really is less trustworthy -- but there's no
    validated *directional* correction to apply from this test alone.
    """
    from . import features

    rows = []
    for season in seasons:
        try:
            oc_changed = features.offensive_coordinator_changes(season)
            if not oc_changed:
                print(f"  ! {season}: no OC-change data available")
                continue

            w_prior = sources.weekly_stats([season - 1])
            w_prior = w_prior[(w_prior["season_type"] == "REG") & (w_prior["position"] == "QB")]
            w_prior = w_prior.assign(
                dropbacks=w_prior["attempts"].fillna(0) + w_prior["sacks_suffered"].fillna(0))
            w_cur = sources.weekly_stats([season])
            w_cur = w_cur[(w_cur["season_type"] == "REG") & (w_cur["position"] == "QB")]
            w_cur = w_cur.assign(
                dropbacks=w_cur["attempts"].fillna(0) + w_cur["sacks_suffered"].fillna(0))

            prior_agg = (w_prior.groupby(["player_id", "recent_team"])
                        .agg(dropbacks=("dropbacks", "sum"), passing_epa=("passing_epa", "sum"))
                        .reset_index())
            prior_agg["epa_per_db"] = prior_agg["passing_epa"] / prior_agg["dropbacks"].replace(0, np.nan)
            cur_agg = (w_cur.groupby(["player_id", "recent_team"])
                      .agg(dropbacks=("dropbacks", "sum"), passing_epa=("passing_epa", "sum"))
                      .reset_index())
            cur_agg["epa_per_db"] = cur_agg["passing_epa"] / cur_agg["dropbacks"].replace(0, np.nan)

            prior_q = prior_agg[prior_agg["dropbacks"] >= min_dropbacks].dropna(subset=["epa_per_db"])
            cur_q = cur_agg[cur_agg["dropbacks"] >= min_dropbacks].dropna(subset=["epa_per_db"])
            if prior_q.empty:
                continue
            pos_baseline = float(prior_q["epa_per_db"].mean())

            m = prior_q.merge(cur_q, on=["player_id", "recent_team"], suffixes=("_prior", "_cur"))
            if m.empty:
                continue
            m["oc_changed"] = m["recent_team"].map(oc_changed)
            m = m.dropna(subset=["oc_changed"])
            if m.empty:
                continue
            m["oc_changed"] = m["oc_changed"].astype(bool)
            m["pos_baseline"] = pos_baseline
            m["season"] = season
            rows.append(m[["season", "player_id", "epa_per_db_prior", "epa_per_db_cur",
                          "pos_baseline", "oc_changed"]])
        except Exception as exc:
            print(f"  ! {season}: {type(exc).__name__}: {exc}")

    hist = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if hist.empty:
        return hist
    for lv in shrinkage_levels:
        pred = (1 - lv) * hist["epa_per_db_prior"] + lv * hist["pos_baseline"]
        hist[f"abs_err_shrink_{lv}"] = (pred - hist["epa_per_db_cur"]).abs()
    return hist


def oc_change_qb_efficiency_shrinkage_summary(hist: pd.DataFrame,
                                              shrinkage_levels: tuple[float, ...] = _HC_SHRINKAGE_LEVELS
                                              ) -> dict:
    """Score oc_change_qb_efficiency_shrinkage_backtest's output for both
    cohorts, and report the gap -- `improvement_is_coaching_specific` -- which is
    the number that actually matters, same logic as
    hc_change_qb_efficiency_shrinkage_summary. Near zero or negative (what the
    2021-2025 run found: -0.0028, 95% CI -0.0137 to +0.0124) means shrinkage
    doesn't specifically help the OC-change cohort more than continuity, despite
    oc_change_qb_efficiency_volatility validating that the underlying volatility
    is real -- see that backtest's docstring for why (role_shift is unsigned).
    """
    if hist.empty:
        return {"n_players": 0}

    def errs_for(g: pd.DataFrame) -> tuple[dict, float | None]:
        e = {lv: float(g[f"abs_err_shrink_{lv}"].mean())
            for lv in shrinkage_levels if f"abs_err_shrink_{lv}" in g.columns}
        best = min(e, key=e.get) if e else None
        return e, best

    changed = hist[hist["oc_changed"]]
    same = hist[~hist["oc_changed"]]
    errs_c, best_c = errs_for(changed) if not changed.empty else ({}, None)
    errs_s, best_s = errs_for(same) if not same.empty else ({}, None)
    improve_c = (errs_c.get(0.0) - errs_c[best_c]) if best_c is not None else None
    improve_s = (errs_s.get(0.0) - errs_s[best_s]) if best_s is not None else None

    return {
        "n_oc_changed": int(len(changed)),
        "n_oc_same": int(len(same)),
        "seasons": sorted(int(s) for s in hist["season"].unique()),
        "mean_abs_error_by_shrinkage_oc_changed": errs_c,
        "mean_abs_error_by_shrinkage_oc_same": errs_s,
        "best_shrinkage_oc_changed": best_c,
        "best_shrinkage_oc_same": best_s,
        "improvement_vs_no_shrinkage_oc_changed": improve_c,
        "improvement_vs_no_shrinkage_oc_same": improve_s,
        "improvement_is_coaching_specific": (improve_c - improve_s)
                                           if improve_c is not None and improve_s is not None else None,
    }


_OC_HISTORY_CSV_PATH = Path(__file__).resolve().parent.parent.parent / "docs" / "oc_history_2020_2025.csv"


def _lateral_oc_moves() -> list[tuple[str, int, str, int, str]]:
    """Real offensive-coordinator-to-offensive-coordinator moves: the same named
    person holding the OC title at team A one season and a *different* team B the
    next, mined from docs/oc_history_2020_2025.csv's already-cleaned
    `offensive_coordinator` column.

    Deliberately excludes the much more common case of a coordinator getting
    promoted to head coach elsewhere (e.g. Ben Johnson: Detroit OC through 2024,
    Chicago HC from 2025) -- that's a real playcaller move too, but the OC-title
    dataset can't see it (see offensive_coordinator_changes's docstring), so
    including it would silently undercount rather than help. Only lateral
    OC-to-OC hires, where the title itself tracks the person continuously, are
    usable here.

    Returns (name, from_season, from_team, to_season, to_team) tuples. A 2020-2025
    run of the underlying dataset finds 12 such moves -- a small, real sample, not
    a large one; oc_scheme_transfer_backtest's docstring is explicit about what
    that does and doesn't support concluding.
    """
    if not _OC_HISTORY_CSV_PATH.exists():
        return []
    df = pd.read_csv(_OC_HISTORY_CSV_PATH)
    df = df[df["offensive_coordinator"].notna() & (df["offensive_coordinator"] != "")]
    by_name: dict[str, dict[int, str]] = {}
    for _, r in df.iterrows():
        by_name.setdefault(r["offensive_coordinator"], {})[int(r["season"])] = r["team"]

    moves = []
    for name, by_season in by_name.items():
        for y in sorted(by_season):
            if (y - 1) in by_season and by_season[y - 1] != by_season[y]:
                moves.append((name, y - 1, by_season[y - 1], y, by_season[y]))
    return moves


_SCHEME_TEAM_REMAP = {"LAR": "LA"}


def oc_scheme_transfer_backtest(seasons: list[int]) -> pd.DataFrame:
    """Does an incoming offensive coordinator's own pass/run tendency at his prior
    team predict his new team's actual pass rate better than simply assuming the
    new team keeps doing what it did the year before he arrived (continuity)?

    Motivated directly by the coaching-change work this session already
    validated for QB efficiency (oc_change_qb_efficiency_volatility): that found
    a real OC change makes a QB's efficiency *less predictable*, but couldn't say
    which way it would move. If a coach's own scheme identity travels with him,
    that would be a *directional*, actionable signal instead -- bullish for a
    team's RBs when a run-heavy playcaller arrives, bullish for its WR/TE volume
    when a pass-heavy one does.

    For every real lateral OC-to-OC move `_lateral_oc_moves` finds (coach X: team
    A in season Y-1 -> team B in season Y) landing in `seasons`, using
    features.neutral_script_pass_rate (score-script-neutral pass rate, real play
    data, no leak-free bound needed since every input is real box scores from
    seasons already played):
      - `prior_source`: team A's pass rate in season Y-1 -- the coach's own,
        most recent tendency, the new signal being tested
      - `new_before`: team B's pass rate in season Y-1 -- what team B did the
        year *before* he arrived, the continuity baseline
      - `new_after`: team B's actual pass rate in season Y -- the real outcome

    IMPORTANT sample-size caveat, sharper here than anywhere else in this
    codebase: only 12 such moves exist across 2021-2025 (most coordinator
    turnover is either a first-time hire or a promotion to head coach elsewhere,
    neither of which this dataset can trace -- see _lateral_oc_moves). Every
    other backtest here has at least 100+ observations; this has 12. Treat
    anything found here as a plausible first look, not a settled answer the way
    the QB efficiency volatility finding is.

    A 2021-2025 run found no signal to act on: the coach's own prior tendency has
    essentially zero rank correlation with the new team's actual pass rate
    (Spearman 0.00), while pure continuity is *negatively* correlated (-0.34,
    plausibly just mean-reversion noise at this sample size, not a real
    "fade the prior year" signal). By mean absolute error, continuity actually
    beats the coach-identity predictor (0.051 vs 0.057), and the coach predictor
    only wins the individual-event comparison 4 of 12 times. The scheme-transfer
    hypothesis is well-motivated but doesn't show up in the data available to
    test it -- team personnel and context appear to dominate an incoming
    playcaller's known tendency, at least on this simple metric.
    """
    from . import features

    moves = [m for m in _lateral_oc_moves() if m[3] in seasons]
    if not moves:
        return pd.DataFrame()

    needed = sorted({y for m in moves for y in (m[1], m[3])})
    try:
        pbp = sources.play_by_play(seasons=needed)
    except Exception as exc:
        print(f"  ! play-by-play unavailable for {needed}: {type(exc).__name__}: {exc}")
        return pd.DataFrame()
    rates = features.neutral_script_pass_rate(pbp)
    rate_map = {(int(r["season"]), r["team"]): float(r["pass_rate"]) for _, r in rates.iterrows()}

    def rate(season: int, team: str) -> float | None:
        team = _SCHEME_TEAM_REMAP.get(team, team)
        return rate_map.get((season, team))

    rows = []
    for name, ya, ta, yb, tb in moves:
        prior_source, new_before, new_after = rate(ya, ta), rate(ya, tb), rate(yb, tb)
        if prior_source is None or new_before is None or new_after is None:
            continue
        rows.append({
            "coach": name, "from_team": ta, "to_team": tb, "season": yb,
            "prior_source": prior_source, "new_before": new_before, "new_after": new_after,
            "err_coach": abs(prior_source - new_after),
            "err_continuity": abs(new_before - new_after),
        })
    return pd.DataFrame(rows)


def oc_scheme_transfer_backtest_summary(hist: pd.DataFrame) -> dict:
    """Score oc_scheme_transfer_backtest's output: does the incoming coach's own
    pass-rate tendency (`prior_source`) predict the new team's actual pass rate
    (`new_after`) better than continuity (`new_before`)?

    `coach_win_rate`: share of individual moves where the coach-identity
    predictor was closer to the real outcome than continuity was -- compared
    against 0.5 (coin flip) rather than any bootstrap CI, since n=12 is too thin
    for a stable resampled interval to mean much (see the backtest's docstring).
    A positive `mae_improvement_vs_continuity` (continuity's error minus the
    coach predictor's error) would mean the scheme-transfer idea is worth
    building into projections; near zero or negative -- what the real run found
    -- means it isn't, at least not on the evidence 12 real moves can supply.
    """
    if hist.empty:
        return {"n_moves": 0}
    def spearman(a, b):
        return float(pd.Series(a).rank().corr(pd.Series(b).rank()))

    return {
        "n_moves": int(len(hist)),
        "seasons": sorted(int(s) for s in hist["season"].unique()),
        "coach_win_rate": float(hist["err_coach"].lt(hist["err_continuity"]).mean()),
        "mean_abs_error_coach": float(hist["err_coach"].mean()),
        "mean_abs_error_continuity": float(hist["err_continuity"].mean()),
        "mae_improvement_vs_continuity": float(hist["err_continuity"].mean() - hist["err_coach"].mean()),
        "corr_prior_source_vs_actual": spearman(hist["prior_source"], hist["new_after"]),
        "corr_continuity_vs_actual": spearman(hist["new_before"], hist["new_after"]),
    }


def repeat_value_players(hist: pd.DataFrame, min_seasons: int = 2) -> pd.DataFrame:
    """Players who beat their draft slot repeatedly rather than once.

    One outperformance is a season; two or three is a trait. This is the closest
    thing in the data to a list of players the market persistently underrates.
    """
    h = hist[(hist["ecr"] <= DRAFTABLE_ECR_CUTOFF) & (~hist.get("unresolved", False))]
    g = h.groupby(["_key", "name", "position"]).agg(
        seasons=("season", "nunique"),
        hits=("hit", "sum"),
        busts=("bust", "sum"),
        avg_value_ratio=("value_ratio", "mean"),
        avg_ecr=("ecr", "mean"),
        avg_games=("games", "mean"),
    ).reset_index()
    g = g[g["seasons"] >= min_seasons]
    g["hit_rate"] = g["hits"] / g["seasons"]
    return g.sort_values("avg_value_ratio", ascending=False).reset_index(drop=True)


# QB is capped at 1 for the hindsight-optimal side of draft_backtest regardless of
# league roster caps elsewhere: a second QB scores zero in a 1-QB starting lineup
# except bye weeks, so ranking it against real RB/WR/TE need by raw value or even
# league-wide VOR overstates it enormously. This is the same conclusion
# who_should_i_pick already reaches live via _positional_need's BACKUP_DECAY, just
# enforced as a hard cap here since the optimal side isn't running that pipeline.
_OPTIMAL_POSITION_CAPS = {"QB": 1, "RB": 5, "WR": 6, "TE": 2}


def draft_backtest(league_id: str, season: int, platform: str = "espn",
                   top_n: int = 3) -> dict:
    """Replay a real past draft leak-free: what the live algorithm would have
    recommended at each of your picks, and the true hindsight-optimal pick by
    value over replacement, against what you actually took.

    "Leak-free" means the board is built the same way draft_backtest's caller
    (matchup_backtest's sibling) always should be for a past season: production
    stats, O-line/pace/defense, and rookie draft curves are bounded to seasons
    strictly before `season`, and ADP is that season's real preseason snapshot --
    see model.build_player_table's docstring. Nothing from the season being
    predicted leaks into the prediction.

    Each of the three picks in a round (yours, the algorithm's, the true
    optimal's) carries two more things, both computed for the season being
    tested rather than today:
      - a value verdict: preseason ECR against actual finish, the same
        steal/bust framing value_picks uses live, just against real outcomes
        instead of projections
      - team context: that player's team's O-line ranks, pace, and schedule
        difficulty that season -- what team_context reports, but leak-free for
        a past season instead of always reading the current one

    Only ESPN is supported (auto-detects your team and draft slot from
    ESPN_SWID/ESPN_S2). K/DST aren't modelled anywhere in this tool, so those
    rounds report your actual pick with no comparison, same as everywhere else.
    """
    from . import board as bd
    from . import model
    from .config import LeagueSettings, ModelWeights

    if platform != "espn":
        return {"error": "draft_backtest only supports platform='espn' for now"}

    ctx = bd.espn_league_context(league_id, season)
    if ctx["my_team_id"] is None:
        return {"error": "couldn't find your team -- check ESPN_SWID/ESPN_S2 and that "
                         "you're a member of this league"}
    if ctx["draft_slot"] is None:
        return {"error": f"no draft found for league {league_id} in {season}"}

    league = LeagueSettings(
        name=f"_backtest_{league_id}_{season}", teams=ctx["teams"],
        rounds=ctx["rounds"], draft_slot=ctx["draft_slot"], snake=True,
        scoring=Scoring.preset(ctx["scoring"]), starters=ctx["starters"],
    )
    weights = ModelWeights()

    tbl = model.build_player_table(league, weights, season=season)
    proj = model.project(tbl, league, weights)
    adp = bd.load_adp(season=season, superflex=False)
    proj = bd.attach_adp(proj, adp)
    board = bd.convert_adp_format(proj, ctx["scoring"])
    board["drafted"] = False

    fin = season_finish(season, league.scoring)
    fin_idx = fin.set_index("_key")

    from .names import normalize as norm_name

    def actual_points(name: str) -> float | None:
        # A period in initials ("D.J. Moore") normalizes differently from the
        # no-period form ("DJ Moore") that season_finish's player_display_name
        # sometimes uses -- try both so a real season isn't reported as missing.
        for candidate in (name, name.replace(".", "")):
            key = norm_name(candidate)
            if key in fin_idx.index:
                r = fin_idx.loc[key]
                if isinstance(r, pd.DataFrame):
                    r = r.iloc[0]
                return float(r["points"])
        return None

    board = board.assign(actual_points=board["name"].map(actual_points))

    replacement_rank = league.replacement_ranks()
    replacement_pts = {}
    for pos, rank in replacement_rank.items():
        sub = fin[fin["position"] == pos].sort_values("finish_pos_rank")
        at_or_after = sub[sub["finish_pos_rank"] >= rank]
        replacement_pts[pos] = float(at_or_after["points"].iloc[0]) if not at_or_after.empty else 0.0
    board = board.assign(
        actual_vor=board.apply(
            lambda r: (r["actual_points"] - replacement_pts.get(r["position"], 0.0))
            if pd.notna(r["actual_points"]) else None, axis=1))

    ecr = preseason_ecr(season, superflex=False)
    ecr_idx = ecr.set_index("_key")

    def value_info(name: str | None) -> dict | None:
        """Preseason draft cost vs. actual finish -- a market-value verdict,
        distinct from actual_vor (which measures against positional replacement,
        not against what the room paid for the player)."""
        if name is None:
            return None
        for candidate in (name, name.replace(".", "")):
            key = norm_name(candidate)
            if key not in ecr_idx.index:
                continue
            e = ecr_idx.loc[key]
            if isinstance(e, pd.DataFrame):
                e = e.iloc[0]
            f = fin_idx.loc[key] if key in fin_idx.index else None
            if isinstance(f, pd.DataFrame):
                f = f.iloc[0]
            if f is None:
                return {"preseason_ecr": int(e["ecr"]), "preseason_pos_rank": int(e["pos_ecr"]),
                        "actual_finish_overall": None, "verdict": f"no {season} result"}
            delta = int(e["ecr"]) - int(f["finish_overall"])
            if delta >= 30:
                verdict = f"STEAL ({delta:+d} spots)"
            elif delta >= 5:
                verdict = f"good value ({delta:+d})"
            elif delta > -15:
                verdict = f"fair, at cost ({delta:+d})"
            elif delta > -40:
                verdict = f"underperformed ({delta:+d})"
            else:
                verdict = f"BUST ({delta:+d})"
            return {
                "preseason_ecr": int(e["ecr"]), "preseason_pos_rank": int(e["pos_ecr"]),
                "actual_finish_overall": int(f["finish_overall"]),
                "actual_finish_pos_rank": int(f["finish_pos_rank"]), "verdict": verdict,
            }
        return None

    def team_ctx(name: str | None) -> dict | None:
        """O-line, pace and schedule for this player's team that season -- the
        same leak-free numbers team_context reports live, pulled from this
        backtest's own board instead of today's data, since team_context itself
        always reads the current season."""
        if name is None:
            return None
        row = board[board["name"] == name]
        if row.empty:
            return None
        r = row.iloc[0]
        sos_col = f"sos_{r.get('position')}_z"
        return {
            "team": r.get("team"),
            "oline_run_block_rank": (int(r["run_block_rank"])
                                     if pd.notna(r.get("run_block_rank")) else None),
            "oline_pass_block_rank": (int(r["pass_block_rank"])
                                      if pd.notna(r.get("pass_block_rank")) else None),
            "plays_per_game": (round(float(r["plays_per_game"]), 1)
                               if pd.notna(r.get("plays_per_game")) else None),
            "pass_rate": round(float(r["pass_rate"]), 3) if pd.notna(r.get("pass_rate")) else None,
            "schedule_z": (round(float(r[sos_col]), 2)
                           if sos_col in r.index and pd.notna(r.get(sos_col)) else None),
        }

    picks = bd.sync_espn(league_id, season=season)
    my_overalls = set(league.picks_for_slot(league.draft_slot)[:league.rounds])

    state = bd.DraftState(league, name=f"_backtest_scratch_{league_id}_{season}")
    state.reset()
    optimal_taken_keys: set[str] = set()
    my_taken_pos: dict[str, int] = {}
    # The algo, like optimal, plays out its own hypothetical draft rather than just
    # reacting fresh at each of your real turns -- otherwise a player it recommended
    # in an earlier round, but that nobody in the real draft actually took before
    # your next turn, is still sitting on the board and gets recommended again. Roster
    # need is tracked the same way, off the algo's own picks, not your real ones --
    # its need discount has to reflect what it would actually have rostered by then.
    algo_taken_keys: set[str] = set()
    algo_roster: dict[str, int] = {}

    rows = []
    your_total, algo_total, optimal_total = 0.0, 0.0, 0.0
    for p in picks:
        overall = p["overall"]
        if overall in my_overalls:
            b = board.copy()
            b.loc[b["_key"].isin(state.taken_keys() | algo_taken_keys), "drafted"] = True

            nxt = state.next_pick_for_me()
            on_clock = state.on_the_clock
            current = nxt if (nxt is not None and nxt > on_clock) else on_clock
            after = state.pick_after_next() if nxt == current else nxt
            recs = model.recommend(b, league, current_pick=current, next_pick=after,
                                   roster=algo_roster, top_n=top_n)
            algo_top = recs.iloc[0] if not recs.empty else None
            if algo_top is not None:
                algo_taken_keys.add(algo_top["_key"])
                algo_roster[algo_top["position"]] = algo_roster.get(algo_top["position"], 0) + 1

            opt_avail = b[~b["_key"].isin(optimal_taken_keys) & ~b["drafted"]]
            opt_avail = opt_avail[opt_avail["position"].map(
                lambda pos: my_taken_pos.get(pos, 0) < _OPTIMAL_POSITION_CAPS.get(pos, 99))]
            opt_avail = opt_avail.dropna(subset=["actual_vor"]).sort_values(
                "actual_vor", ascending=False)
            optimal = opt_avail.iloc[0] if not opt_avail.empty else None
            if optimal is not None:
                my_taken_pos[optimal["position"]] = my_taken_pos.get(optimal["position"], 0) + 1
                optimal_taken_keys.add(optimal["_key"])

            algo_name = algo_top["name"] if algo_top is not None else None
            optimal_name = optimal["name"] if optimal is not None else None

            yp = actual_points(p["name"])
            ap = float(algo_top["actual_points"]) if algo_top is not None and pd.notna(algo_top["actual_points"]) else None
            op = float(optimal["actual_points"]) if optimal is not None and pd.notna(optimal["actual_points"]) else None
            # Only count rounds where your own pick was a modelled position --
            # otherwise a K/DST round would compare your None against algo/optimal's
            # real skill-position alternative, inflating their totals unfairly.
            if yp is not None:
                your_total += yp
                algo_total += ap or 0.0
                optimal_total += op or 0.0

            rows.append({
                "round": (overall - 1) // league.teams + 1, "overall": overall,
                "your_pick": p["name"], "your_points": round(yp, 1) if yp is not None else None,
                "your_pick_value": value_info(p["name"]), "your_pick_team_context": team_ctx(p["name"]),
                "algo_pick": algo_name, "algo_points": round(ap, 1) if ap is not None else None,
                "algo_pick_value": value_info(algo_name), "algo_pick_team_context": team_ctx(algo_name),
                "optimal_pick": optimal_name, "optimal_points": round(op, 1) if op is not None else None,
                "optimal_vor": (round(float(optimal["actual_vor"]), 1)
                               if optimal is not None and pd.notna(optimal["actual_vor"]) else None),
                "optimal_pick_value": value_info(optimal_name),
                "optimal_pick_team_context": team_ctx(optimal_name),
            })

        row = board[board["name"] == p["name"]]
        resolved = row.iloc[0]["name"] if not row.empty else p["name"]
        state.record(resolved, overall)

    state.path.unlink(missing_ok=True)

    return {
        "league": ctx["league_name"], "season": season, "teams": ctx["teams"],
        "scoring": ctx["scoring"], "your_draft_slot": league.draft_slot,
        "note": "K/DST aren't modelled -- those rounds show your actual pick only. "
                "algo_pick is what who_should_i_pick would say live; optimal_pick is "
                "the true hindsight-best value-over-replacement pick, QB capped at 1 "
                "since a second quarterback can't start. *_value compares preseason "
                "draft cost (ECR) to actual finish -- a market verdict, distinct from "
                "actual_vor which measures against positional replacement, not cost. "
                "*_team_context is that player's team's O-line/pace/schedule for the "
                "season being tested (leak-free, not today's), the same numbers "
                "team_context reports live for the current season.",
        "totals": {
            "your_points": round(your_total, 1),
            "algo_points": round(algo_total, 1),
            "optimal_points": round(optimal_total, 1),
        },
        "rounds": rows,
    }


def multi_season_draft_backtest(league_id: str, seasons: list[int],
                                top_n: int = 3) -> pd.DataFrame:
    """Runs draft_backtest across many real seasons and compiles a row per real
    pick you made: your actual points, what the live algorithm (recommend(), the
    exact mechanism who_should_i_pick uses) would have taken instead and its real
    points, and the true hindsight-optimal pick's real points.

    This is the concrete "is the tool actually ready" check the other backtests
    this session weren't: those tested whether a *new, unvalidated* signal
    (PositionMarkov, position_scarcity_entropy, vacated-role insurance) should be
    added on top of the live model. This instead asks whether the live model
    itself, as it already stands, would have beaten what you actually drafted,
    using real outcomes from real past drafts in your own league.

    K/DST rounds are excluded (draft_backtest reports `your_points: None` for
    them, since neither position is modelled). A season draft_backtest can't
    replay (no ESPN draft found, credentials missing, league didn't exist that
    year) is skipped with a printed note rather than failing the whole run.
    """
    rows = []
    for season in seasons:
        try:
            out = draft_backtest(league_id, season, top_n=top_n)
        except Exception as exc:
            print(f"  ! {season}: {type(exc).__name__}: {exc}")
            continue
        if "error" in out:
            print(f"  ! {season}: {out['error']}")
            continue
        for r in out["rounds"]:
            if r["your_points"] is None:
                continue
            rows.append({
                "season": season, "round": r["round"], "overall": r["overall"],
                "your_pick": r["your_pick"], "your_points": r["your_points"],
                "algo_pick": r["algo_pick"], "algo_points": r["algo_points"] or 0.0,
                "optimal_pick": r["optimal_pick"], "optimal_points": r["optimal_points"] or 0.0,
            })
    return pd.DataFrame(rows)


def multi_season_draft_backtest_summary(hist: pd.DataFrame, early_late_cutoff: int = 6) -> dict:
    """Score multi_season_draft_backtest's output: does the live algorithm
    actually outscore what you drafted, and how close does either get to the true
    hindsight-optimal?

    `algo_beats_your_pick_rate` is the share of real picks where the algorithm's
    recommendation would have scored more than what you actually took that round
    -- the most direct "would this have helped" number. `algo_pct_of_optimal` /
    `your_pct_of_optimal` put both totals on the same scale (100% would mean
    matching the true hindsight-best every single pick, impossible in practice
    but useful as a ceiling to measure the gap against).

    Also splits `algo_improvement_over_you_per_pick` into rounds <= `early_late_cutoff`
    and rounds after it. A real run (one league, 2021-2025, 53 picks) found the
    *overall* gap statistically inconclusive from only 5 seasons (bootstrap 95% CI
    -19.1 to +1.6, straddling zero) but a stark, clearly-signed structure hiding
    underneath the noisy overall average: the algorithm clearly outperformed in
    rounds 1-6 (+36.6 pts/pick) and clearly underperformed in rounds 7+ (-67.8
    pts/pick) -- a real weakness in bench/late-round value-finding masked by
    strong early-round performance when only the flat total is reported. Worth
    checking this split specifically before trusting an aggregate number that
    looks fine (or alarming) on its own.
    """
    if hist.empty:
        return {"n_picks": 0}
    optimal_sum = float(hist["optimal_points"].sum())
    diff = hist["algo_points"] - hist["your_points"]
    early = hist[hist["round"] <= early_late_cutoff]
    late = hist[hist["round"] > early_late_cutoff]
    return {
        "n_picks": int(len(hist)),
        "seasons": sorted(int(s) for s in hist["season"].unique()),
        "total_your_points": float(hist["your_points"].sum()),
        "total_algo_points": float(hist["algo_points"].sum()),
        "total_optimal_points": optimal_sum,
        "algo_beats_your_pick_rate": float((hist["algo_points"] > hist["your_points"]).mean()),
        "algo_pct_of_optimal": (float(hist["algo_points"].sum() / optimal_sum)
                                if optimal_sum else float("nan")),
        "your_pct_of_optimal": (float(hist["your_points"].sum() / optimal_sum)
                                if optimal_sum else float("nan")),
        "algo_improvement_over_you_per_pick": float(diff.mean()),
        "early_late_cutoff": early_late_cutoff,
        "early_rounds_improvement_per_pick": float(early["algo_points"].sub(early["your_points"]).mean())
                                            if not early.empty else float("nan"),
        "late_rounds_improvement_per_pick": float(late["algo_points"].sub(late["your_points"]).mean())
                                           if not late.empty else float("nan"),
    }


_MOCK_BOT_CAPS = {"QB": 3, "RB": 6, "WR": 7, "TE": 3}  # loose -- realism comes from
                                                        # ADP + noise, this just stops
                                                        # a degenerate all-one-position bot


def mock_draft(league, weights, season: int, n_trials: int = 30,
               top_n: int = 5, seed: int = 0) -> dict:
    """Monte Carlo mock draft: the live algorithm at your slot against n_trials
    independent drafts of ADP-driven bots, scored on real points from `season`
    when they exist, or the model's own proj_points when they don't yet (the
    current/future season -- see scored_on in the result).

    Unlike draft_backtest, this doesn't need (or use) a real draft -- the other
    teams are bots that pick by that season's real preseason ADP with realistic
    reach/fall noise (bigger swings plausible late, tight consensus at the very
    top) rather than following it exactly, so who's actually on the board at
    your turn varies draw to draw. Your slot runs the same model.recommend()
    who_should_i_pick uses live. The board is leak-free, same bound as
    draft_backtest: nothing from `season` or later feeds the projections -- so
    passing the current season runs this against the exact live board (this
    year's projections, built from history through last season) rather than a
    past, already-decided one.

    A single trial can make the algorithm look better or worse than its true
    average just from bot luck -- that's the reason for averaging many. Returns
    the mean/median/std/range of total points across trials, plus per round the
    most-common picks and how often each one showed up, so low-consistency
    rounds (usually round 6+) are visible rather than hidden behind one draw.

    K/DST aren't modelled, so only the league's skill-position rounds are
    simulated (total rounds minus K and DST starting slots).
    """
    from . import board as bd
    from . import model

    sc_label = "ppr" if float(league.scoring.rec) >= 0.9 else \
               "half_ppr" if float(league.scoring.rec) >= 0.35 else "standard"

    tbl = model.build_player_table(league, weights, season=season)
    proj = model.project(tbl, league, weights)
    adp = bd.load_adp(season=season, superflex=bool(getattr(league, "superflex", 0)))
    proj = bd.attach_adp(proj, adp)
    board = bd.convert_adp_format(proj, sc_label)
    board["drafted"] = False

    # A season that hasn't been played yet (the live/current one, most commonly)
    # has no real box scores to score picks against -- season_finish raises rather
    # than returning empty, since "no games played" and "scraping gap" shouldn't
    # look the same. Fall back to the board's own proj_points in that case: still a
    # real Monte Carlo stress test of who the algorithm lands on given bot-driven
    # board variance, just evaluated on the model's forecast instead of hindsight.
    from .names import normalize as norm_name
    try:
        fin = season_finish(season, league.scoring).set_index("_key")
        scored_on = "actual"
    except RuntimeError:
        fin = None
        scored_on = "projected"
        proj_lookup = {norm_name(n): p for n, p in zip(board["name"], board["proj_points"])}

    def actual_points(name: str) -> float:
        if fin is None:
            return float(proj_lookup.get(norm_name(name), 0.0))
        for candidate in (name, name.replace(".", "")):
            key = norm_name(candidate)
            if key in fin.index:
                r = fin.loc[key]
                if isinstance(r, pd.DataFrame):
                    r = r.iloc[0]
                return float(r["points"])
        return 0.0

    teams = league.teams
    my_slot = league.draft_slot
    sim_rounds = max(1, league.rounds - league.starters.get("K", 0)
                     - league.starters.get("DST", 0))
    total_picks = sim_rounds * teams

    def slot_for_pick(overall: int) -> int:
        rnd = (overall - 1) // teams + 1
        idx = (overall - 1) % teams + 1
        return (teams - idx + 1) if (league.snake and rnd % 2 == 0) else idx

    trial_totals = []
    round_picks: dict[int, list[tuple[str, float]]] = {r: [] for r in range(1, sim_rounds + 1)}

    for trial in range(n_trials):
        rng = np.random.default_rng(seed + trial)
        state = bd.DraftState(league, name=f"_mockdraft_scratch_{id(league)}")
        state.reset()
        rosters: dict[int, dict[str, int]] = {s: {} for s in range(1, teams + 1)}
        my_picks_this_trial = []

        for overall in range(1, total_picks + 1):
            slot = slot_for_pick(overall)
            b = board.copy()
            b.loc[b["_key"].isin(state.taken_keys()), "drafted"] = True
            pool = b[~b["drafted"]]
            if pool.empty:
                break

            if slot == my_slot:
                roster = state.my_roster(b)
                nxt = state.next_pick_for_me()
                on_clock = state.on_the_clock
                current = nxt if (nxt is not None and nxt > on_clock) else on_clock
                after = state.pick_after_next() if nxt == current else nxt
                recs = model.recommend(pool, league, current_pick=current, next_pick=after,
                                       roster=roster, top_n=top_n)
                if recs.empty:
                    break
                chosen = recs.iloc[0]["name"]
                rnd = (overall - 1) // teams + 1
                my_picks_this_trial.append((rnd, chosen))
            else:
                r = rosters[slot]
                avail = pool[pool["position"].map(
                    lambda p: r.get(p, 0) < _MOCK_BOT_CAPS.get(p, 99))]
                if avail.empty:
                    avail = pool
                sigma = np.maximum(3.0, 0.25 * avail["adp"].to_numpy())
                noisy = avail["adp"].to_numpy() + rng.normal(0, sigma)
                best = avail.iloc[int(np.argmin(noisy))]
                chosen = best["name"]
                rosters[slot][best["position"]] = rosters[slot].get(best["position"], 0) + 1

            state.record(chosen, overall)

        state.path.unlink(missing_ok=True)

        trial_total = 0.0
        for rnd, name in my_picks_this_trial:
            pts = actual_points(name)
            trial_total += pts
            round_picks[rnd].append((name, pts))
        trial_totals.append(trial_total)

    totals = np.array(trial_totals) if trial_totals else np.array([0.0])

    rounds_out = []
    for rnd in range(1, sim_rounds + 1):
        picks = round_picks[rnd]
        if not picks:
            continue
        by_name: dict[str, list[float]] = {}
        for name, pts in picks:
            by_name.setdefault(name, []).append(pts)
        ranked = sorted(by_name.items(), key=lambda kv: -len(kv[1]))
        rounds_out.append({
            "round": rnd,
            "trials_reaching_round": len(picks),
            "avg_points": round(float(np.mean([p[1] for p in picks])), 1),
            "picks": [
                {"player": name, "trials": len(pts_list),
                 "frequency": round(len(pts_list) / len(picks), 2),
                 "avg_points": round(float(np.mean(pts_list)), 1)}
                for name, pts_list in ranked[:5]
            ],
        })

    return {
        "season": season, "n_trials": n_trials, "your_draft_slot": my_slot,
        "simulated_rounds": sim_rounds, "scored_on": scored_on,
        "note": ("Other teams are ADP bots with reach/fall noise, not real opponents "
                 "-- this is a stress test of the algorithm's typical behavior, not a "
                 "replay of any specific draft. K/DST aren't modelled, so only "
                 "skill-position rounds are simulated. "
                 + (f"{season} hasn't been played -- picks are scored on the model's "
                    "own proj_points (leak-free through the prior season), not real "
                    "outcomes, so this reads as a forecast, not a validated backtest."
                    if scored_on == "projected" else
                    f"Picks are scored on real {season} points.")),
        "totals": {
            "mean_points": round(float(totals.mean()), 1),
            "median_points": round(float(np.median(totals)), 1),
            "std_dev": round(float(totals.std()), 1),
            "min_points": round(float(totals.min()), 1),
            "max_points": round(float(totals.max()), 1),
        },
        "rounds": rounds_out,
    }


def _pick_context(name: str, season: int) -> dict | None:
    """The concrete "why" behind a value pick: how his usage changed over the
    season (early vs. late-season carries/targets/target share -- a real role
    expansion, not just a good month) and his team's offensive environment
    (O-line ranks, pace, pass/rush split) for that specific season.

    Unlike the leak-free construction elsewhere in this module, this is pure
    retrospective explanation -- it uses that season's own play-by-play, not
    data bounded before it, since the point is understanding what happened,
    not predicting it in advance.
    """
    from . import features
    from .names import normalize as norm_name

    w = sources.weekly_stats([season])
    w = w[w["season_type"] == "REG"]
    rows = pd.DataFrame()
    for candidate in (name, name.replace(".", "")):
        key = norm_name(candidate)
        rows = w[w["player_display_name"].map(norm_name) == key]
        if not rows.empty:
            break
    if rows.empty:
        return None
    rows = rows.sort_values("week")
    team_mode = rows["recent_team"].mode()
    team = team_mode.iloc[0] if not team_mode.empty else None

    def summarize(chunk: pd.DataFrame) -> dict | None:
        if chunk.empty:
            return None
        out = {"games": int(len(chunk)), "avg_carries": round(float(chunk["carries"].mean()), 1),
              "avg_targets": round(float(chunk["targets"].mean()), 1)}
        if chunk["target_share"].notna().any():
            out["avg_target_share"] = round(float(chunk["target_share"].mean()), 3)
        # carries/targets describe a skill-position role change; a QB breakout is
        # a passing-volume story instead, so track that too rather than showing
        # near-zero carries/targets as if that were the whole picture.
        if chunk["attempts"].notna().any() and chunk["attempts"].mean() > 5:
            out["avg_pass_attempts"] = round(float(chunk["attempts"].mean()), 1)
            out["avg_pass_yards"] = round(float(chunk["passing_yards"].mean()), 1)
            out["avg_rush_yards"] = round(float(chunk["rushing_yards"].mean()), 1)
        return out

    usage_trend = {"weeks_1_9": summarize(rows[rows["week"] <= 9]),
                   "weeks_10_plus": summarize(rows[rows["week"] > 9])}

    team_environment = {"team": team}
    if team:
        try:
            pbp = sources.play_by_play([season])
            ol = features.oline_ratings(pbp)
            pace = features.team_pace_and_split(pbp)
            ol_row = ol[(ol["team"] == team) & (ol["season"] == season)]
            pace_row = pace[(pace["team"] == team) & (pace["season"] == season)]
            if not ol_row.empty:
                r = ol_row.iloc[0]
                team_environment["run_block_rank"] = int(r["run_block_rank"])
                team_environment["pass_block_rank"] = int(r["pass_block_rank"])
            if not pace_row.empty:
                r = pace_row.iloc[0]
                team_environment["plays_per_game"] = round(float(r["plays_per_game"]), 1)
                team_environment["pass_rate"] = round(float(r["pass_rate"]), 3)
                team_environment["rush_rate"] = round(float(r["rush_rate"]), 3)
        except Exception as exc:
            team_environment["note"] = f"play-by-play unavailable for {season} ({type(exc).__name__})"

    return {"team": team, "usage_trend": usage_trend, "team_environment": team_environment}


def champion_strategies(league_id: str, seasons: list[int]) -> dict:
    """What actually won this ESPN league, season by season: each champion's
    real draft, and which specific pick was the difference-maker.

    For each season, finds the team that finished 1st (ESPN's rankCalculatedFinal)
    and pulls their real draft. Every pick gets a value verdict -- preseason ECR
    against actual finish, the same steal/bust framing draft_backtest uses --
    so "what did the champion draft" becomes "what draft-cost bet actually paid
    off," not just a list of names. Reports each season's opening two picks,
    first QB/TE round, RB/WR volume, and biggest steal, plus cross-season
    aggregates (how often champions opened RB-RB, the median first-QB round).

    biggest_steal also carries a context block explaining *why* it was a
    steal, not just that it was: usage_trend is that player's real early- vs.
    late-season carries/targets/target share (a role expansion actually
    visible in the box scores, not assumed), and team_environment is his
    team's O-line ranks, pace, and pass/rush split that season. Most
    value picks turn out to be a volume or role story, not raw talent
    outperforming a forecast -- this is what shows that concretely.

    ECR history only goes back to 2020 -- earlier seasons get position/timing
    data but no value verdicts or steal context. ESPN only.
    """
    import os
    import requests
    from . import board as bd
    from . import sources
    from .board import norm_name

    r = sources.weekly_rosters()
    pos_map = (r.dropna(subset=["full_name", "position"])
                .assign(_key=lambda d: d["full_name"].map(norm_name))
                .drop_duplicates("_key").set_index("_key")["position"].to_dict())

    def resolve_position(name: str) -> str:
        if "D/ST" in name:
            return "DST"
        return pos_map.get(norm_name(name), "UNK")

    swid = os.environ.get("ESPN_SWID")
    espn_s2 = os.environ.get("ESPN_S2")
    cookies = {}
    if swid and espn_s2:
        cookies = {"SWID": swid if swid.startswith("{") else f"{{{swid}}}", "espn_s2": espn_s2}

    out_seasons = []
    for season in seasons:
        url = (f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}"
              f"/segments/0/leagues/{league_id}")
        resp = requests.get(url, params={"view": ["mTeam", "mDraftDetail"]},
                           cookies=cookies, timeout=20,
                           headers={"User-Agent": "ffdraft-mcp/1.0"})
        if resp.status_code != 200:
            out_seasons.append({"season": season, "error": f"fetch failed ({resp.status_code})"})
            continue
        data = resp.json()
        teams = data.get("teams") or []
        champ = next((t for t in teams if t.get("rankCalculatedFinal") == 1), None)
        if champ is None:
            out_seasons.append({"season": season, "error": "no final standings yet"})
            continue
        n_teams = len(teams)

        # sync_espn already resolves names (crosswalk + team-defense mapping) for
        # every pick in the league; just pick out the champion's by team id from
        # the raw draft order this same response carries.
        resolved = {p["overall"]: p["name"] for p in bd.sync_espn(league_id, season=season)}
        raw_picks = (data.get("draftDetail") or {}).get("picks") or []
        champ_raw = sorted([p for p in raw_picks if p.get("teamId") == champ["id"]
                           and p.get("playerId", -1) != -1],
                          key=lambda p: p.get("overallPickNumber", 0))
        picks = []
        for p in champ_raw:
            overall = p["overallPickNumber"]
            name = resolved.get(overall, f"ESPN#{p.get('playerId')}")
            picks.append({"round": (overall - 1) // n_teams + 1,
                         "overall": overall, "name": name,
                         "position": resolve_position(name)})

        ecr_idx = preseason_ecr(season, superflex=False).set_index("_key")
        fin_idx = season_finish(season, sc=Scoring.preset("half_ppr")).set_index("_key")
        for p in picks:
            e = f = None
            for cand in (p["name"], p["name"].replace(".", "")):
                key = norm_name(cand)
                if key in ecr_idx.index and e is None:
                    e = ecr_idx.loc[key]
                    if isinstance(e, pd.DataFrame):
                        e = e.iloc[0]
                if key in fin_idx.index and f is None:
                    f = fin_idx.loc[key]
                    if isinstance(f, pd.DataFrame):
                        f = f.iloc[0]
            p["preseason_ecr"] = int(e["ecr"]) if e is not None else None
            p["actual_finish"] = int(f["finish_overall"]) if f is not None else None
            p["actual_points"] = round(float(f["points"]), 1) if f is not None else None
            p["value_delta"] = (p["preseason_ecr"] - p["actual_finish"]
                                if p["preseason_ecr"] is not None and p["actual_finish"] is not None
                                else None)

        r1 = picks[0] if picks else None
        r2 = picks[1] if len(picks) > 1 else None
        first_qb = next((p for p in picks if p["position"] == "QB"), None)
        first_te = next((p for p in picks if p["position"] == "TE"), None)
        rb_ct = sum(1 for p in picks if p["position"] == "RB")
        wr_ct = sum(1 for p in picks if p["position"] == "WR")
        steal = max((p for p in picks if p["value_delta"] is not None),
                   key=lambda p: p["value_delta"], default=None)
        if steal is not None:
            steal = dict(steal, context=_pick_context(steal["name"], season))

        champ_name = f"{champ.get('location', '')} {champ.get('nickname', '')}".strip() or champ.get("name")
        out_seasons.append({
            "season": season, "teams": n_teams, "champion": champ_name,
            "champion_record": champ.get("record", {}).get("overall"),
            "opened": {"round_1": r1, "round_2": r2},
            "first_qb_round": first_qb["round"] if first_qb else None,
            "first_te_round": first_te["round"] if first_te else None,
            "rb_drafted": rb_ct, "wr_drafted": wr_ct,
            "biggest_steal": steal,
            "full_draft": picks,
        })

    valid = [s for s in out_seasons if "error" not in s]
    rb_rb_open = sum(1 for s in valid
                     if s["opened"]["round_1"] and s["opened"]["round_2"]
                     and s["opened"]["round_1"]["position"] == "RB"
                     and s["opened"]["round_2"]["position"] == "RB")
    qb_rounds = [s["first_qb_round"] for s in valid if s["first_qb_round"]]
    te_rounds = [s["first_te_round"] for s in valid if s["first_te_round"]]

    return {
        "league_id": league_id,
        "note": "ECR history only goes back to 2020 -- earlier seasons have position/"
                "timing data but no value_delta/biggest_steal.",
        "cross_season_patterns": {
            "seasons_analyzed": len(valid),
            "opened_rb_rb": f"{rb_rb_open}/{len(valid)}",
            "first_qb_round_median": (sorted(qb_rounds)[len(qb_rounds) // 2]
                                      if qb_rounds else None),
            "first_qb_round_range": ([min(qb_rounds), max(qb_rounds)] if qb_rounds else None),
            "first_te_round_range": ([min(te_rounds), max(te_rounds)] if te_rounds else None),
            "avg_rb_drafted": (round(sum(s["rb_drafted"] for s in valid) / len(valid), 1)
                              if valid else None),
            "avg_wr_drafted": (round(sum(s["wr_drafted"] for s in valid) / len(valid), 1)
                              if valid else None),
        },
        "seasons": out_seasons,
    }
