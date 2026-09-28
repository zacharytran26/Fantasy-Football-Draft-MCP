"""Preseason rankings vs. actual finish — the 'value pick' engine.

Draft position is a market price. What matters is which players systematically beat
that price. This module pairs FantasyPros preseason expert consensus rank (a very
close stand-in for ADP, published back to 2020) with the fantasy points each player
actually finished with, so hit rates can be measured rather than assumed.

Source: dynastyprocess/data, which mirrors FantasyPros ECR history.
"""
from __future__ import annotations

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
    -- if anything it points the wrong way.
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
    grow proportionally). Not clean enough to wire in.
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
