"""Man-versus-zone production, from per-play coverage tags.

nflverse publishes `pbp_participation` with `defense_coverage_type` (Cover 0/1/2/3/4/6,
2-Man) and the 11 offensive players on the field for every play. Joined to play-by-play
that recovers most of what `separation.py` used to call out of reach: how each skill
player produces against man versus zone coverage.

Limits worth knowing before trusting a number:

  * Roughly half of dropbacks carry a coverage tag, and the tag is a model output from
    NGS tracking, not hand charting. Cover 9 is folded into Cover 3.
  * Exposure is *snaps on the field*, not routes run. It includes snaps spent blocking,
    so `tpps` (targets per pass snap) understates true TPRR, most for TEs and RBs.
    Compare a player's man split to his own zone split, or to players at his position.
  * The tags are not consistent across seasons: league-wide man share reads 30%, 29%, 44%,
    50%, 31% for 2021-2025 (Cover 1 alone swings 23% -> 37% -> 22%), far more than real scheme
    change. Pooled-window comparisons are fine (every player shares the same tags), and the
    backtests use within-season ranks, but do not read a raw man rate across seasons as a trend.
  * Receiver *alignment* (X / Z / slot) is not in any open dataset. `role_table` assigns
    roles by target rank within a team-season as a stand-in -- see its docstring.
    True alignment exists only for the 2023-2025 rookie classes (alignment.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import sources
from .config import SEASONS
from .sources import NFLVERSE, _cached

PART_URL = NFLVERSE + "/pbp_participation/pbp_participation_{season}.parquet"

COV_LABEL = {
    "COVER_0": "Cover 0", "COVER_1": "Cover 1", "2_MAN": "2-Man",
    "COVER_2": "Cover 2", "COVER_3": "Cover 3", "COVER_9": "Cover 3",
    "COVER_4": "Cover 4", "COVER_6": "Cover 6",
}
COV_ORDER = ["Cover 0", "Cover 1", "2-Man", "Cover 2", "Cover 3", "Cover 4", "Cover 6"]
MAN = ("Cover 0", "Cover 1", "2-Man")
ZONE = ("Cover 2", "Cover 3", "Cover 4", "Cover 6")
SKILL = ("WR", "TE", "RB")
ROLES = ["X", "Z", "Slot", "Y", "RB"]

# Seasons used when a caller doesn't say: the last three of the lookback window.
RECENT_SEASONS = 3

_PBP_COLS = [
    "season", "season_type", "game_id", "play_id", "posteam", "defteam", "qb_dropback",
    "pass_attempt", "sack", "complete_pass", "yards_gained", "yards_after_catch",
    "air_yards", "epa", "interception", "pass_touchdown", "receiver_player_id",
]


def _default_seasons(seasons=None) -> list[int]:
    return list(seasons) if seasons else list(SEASONS)[-RECENT_SEASONS:]


def _read_strict(url_tmpl: str, seasons, columns) -> pd.DataFrame:
    """Every requested season or an error -- never a silently partial frame that then
    gets cached (the failure mode sources.weekly_stats guards against)."""
    frames, missing = [], []
    for s in seasons:
        try:
            frames.append(pd.read_parquet(url_tmpl.format(season=s), columns=columns))
        except Exception:
            missing.append(s)
    if missing:
        raise RuntimeError(f"coverage data unavailable for season(s) {missing} "
                           f"({url_tmpl}); refusing to cache a partial result")
    return pd.concat(frames, ignore_index=True)


def build_raw(pbp: pd.DataFrame, part: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Join dropbacks to coverage tags and aggregate. Pure function of its inputs so it
    can be tested on a handful of synthetic plays."""
    pbp = pbp[(pbp["season_type"] == "REG") & (pbp["qb_dropback"] == 1)].copy()
    pbp["play_id"] = pbp["play_id"].astype(int)
    part = part.rename(columns={"nflverse_game_id": "game_id"}).copy()
    part["play_id"] = part["play_id"].astype(int)
    part["cov"] = part["defense_coverage_type"].map(COV_LABEL)
    part = part.dropna(subset=["cov", "offense_players"])
    plays = pbp.merge(part[["game_id", "play_id", "cov", "offense_players"]],
                      on=["game_id", "play_id"], how="inner")

    ex = plays[["season", "cov", "offense_players"]].copy()
    ex["player_id"] = ex.pop("offense_players").str.split(";")
    snaps = (ex.explode("player_id").dropna(subset=["player_id"])
             .groupby(["season", "player_id", "cov"]).size().rename("snaps").reset_index())

    t = plays[(plays["pass_attempt"] == 1) & (plays["sack"] != 1)
              & plays["receiver_player_id"].notna()].copy()
    t["comp"] = (t["complete_pass"] == 1).astype(int)
    t["yds"] = np.where(t["comp"] == 1, t["yards_gained"].fillna(0), 0.0)
    t["yac"] = np.where(t["comp"] == 1, t["yards_after_catch"].fillna(0), 0.0)
    t["air"] = t["air_yards"].fillna(0)
    t["epa"] = t["epa"].fillna(0)
    t["int"] = (t["interception"] == 1).astype(int)
    t["td"] = (t["pass_touchdown"] == 1).astype(int)
    tg = (t.rename(columns={"receiver_player_id": "player_id"})
          .groupby(["season", "player_id", "cov"])
          .agg(targets=("comp", "size"), rec=("comp", "sum"), rec_yds=("yds", "sum"),
               yac=("yac", "sum"), air_yards=("air", "sum"), epa=("epa", "sum"),
               ints=("int", "sum"), tds=("td", "sum")).reset_index())
    player = snaps.merge(tg, on=["season", "player_id", "cov"], how="outer")
    for c in player.columns.difference(["season", "player_id", "cov"]):
        player[c] = player[c].fillna(0)

    team_def = plays.groupby(["season", "defteam", "cov"]).size().rename("plays").reset_index()
    targets = t.rename(columns={"receiver_player_id": "player_id"})[
        ["season", "posteam", "defteam", "player_id", "cov", "comp", "yds", "yac", "air", "epa", "int"]]
    return {"player": player, "team_def": team_def, "targets": targets}


def _raw(seasons=None) -> dict[str, pd.DataFrame]:
    """Aggregates for the full lookback window, shared by every caller. Callers filter by
    season themselves, so a 3-season view and a backtest reuse one download and one cache."""
    seasons = list(SEASONS)
    key = f"{min(seasons)}_{max(seasons)}"
    names = ("player", "team_def", "targets")
    built: dict = {}

    def make(name):
        def build():
            if not built:
                built.update(build_raw(_read_strict(NFLVERSE + "/pbp/play_by_play_{season}.parquet",
                                                    seasons, _PBP_COLS),
                                       _read_strict(PART_URL, seasons,
                                                    ["nflverse_game_id", "play_id", "offense_players",
                                                     "defense_coverage_type"])))
            return built[name]
        return build

    return {n: _cached(f"coverage_{n}_{key}", make(n), max_age_days=30.0) for n in names}


def _stats(g: pd.DataFrame) -> pd.DataFrame:
    """Rate stats from summed counts (never average per-row rates)."""
    t1 = g["targets"].clip(lower=1)
    r1 = g["rec"].clip(lower=1)
    out = pd.DataFrame(index=g.index)
    out["snaps"] = g["snaps"].astype(int)
    out["targets"] = g["targets"].astype(int)
    out["tpps"] = (100 * g["targets"] / g["snaps"].clip(lower=1)).round(1)
    out["catch"] = (100 * g["rec"] / t1).round(1)
    out["adot"] = (g["air_yards"] / t1).round(1)
    out["ypt"] = (g["rec_yds"] / t1).round(2)
    out["yac"] = (g["yac"] / r1).round(1)
    out["yps"] = (g["rec_yds"] / g["snaps"].clip(lower=1)).round(2)
    out["epa_t"] = (g["epa"] / t1).round(2)
    return out


_SUMS = ["snaps", "targets", "rec", "rec_yds", "yac", "air_yards", "epa", "ints", "tds"]


def _positions() -> pd.DataFrame:
    p = sources.players()[["gsis_id", "display_name", "position", "latest_team"]]
    return p.rename(columns={"gsis_id": "player_id", "display_name": "name", "latest_team": "team"})


def player_splits(seasons=None, min_snaps: int = 0) -> pd.DataFrame:
    """One row per WR/TE/RB: stats against man (`man_*`) and zone (`zone_*`), plus the
    man-minus-zone differences. Counts are pooled across `seasons` before rates are taken."""
    seasons = _default_seasons(seasons)
    p = _raw(seasons)["player"]
    p = p[p["season"].isin(seasons)].copy()
    p["mz"] = np.where(p["cov"].isin(MAN), "man", "zone")
    g = p.groupby(["player_id", "mz"])[_SUMS].sum()
    st = _stats(g).unstack("mz")
    st.columns = [f"{mz}_{c}" for c, mz in st.columns]
    st = st.dropna(subset=["man_snaps", "zone_snaps"]).reset_index()
    st = st.merge(_positions(), on="player_id", how="left")
    st["position"] = st["position"].replace({"FB": "RB"})
    st = st[st["position"].isin(SKILL)]
    st = st[(st["man_snaps"] >= min_snaps) & (st["zone_snaps"] >= min_snaps)].copy()
    for c in ("tpps", "ypt", "yps", "epa_t", "catch"):
        st[f"diff_{c}"] = (st[f"man_{c}"] - st[f"zone_{c}"]).round(2)
    return st.reset_index(drop=True)


def position_baselines(seasons=None) -> dict:
    """Pooled man/zone rates per position (counts summed, then rates), so a player
    reads against his own position rather than against the whole league."""
    seasons = _default_seasons(seasons)
    p = _raw(seasons)["player"]
    p = p[p["season"].isin(seasons)].copy()
    pos = _positions().set_index("player_id")["position"].replace({"FB": "RB"})
    p["position"] = p["player_id"].map(pos)
    p = p[p["position"].isin(SKILL)]
    p["mz"] = np.where(p["cov"].isin(MAN), "man", "zone")
    st = _stats(p.groupby(["position", "mz"])[_SUMS].sum())
    return {pos_: {mz: st.loc[(pos_, mz), ["tpps", "catch", "ypt", "yps", "epa_t"]].to_dict()
                   for mz in ("man", "zone")} for pos_ in st.index.get_level_values(0).unique()}


def assign_roles(targets: pd.DataFrame) -> pd.DataFrame:
    """Stand-in for alignment: X = a team's most-targeted WR that season, Z = second,
    Slot = third, Y = most-targeted TE, RB = any back. It labels *role*, not where the
    player lined up -- the WR3 is not necessarily a slot receiver."""
    pos = _positions().set_index("player_id")["position"].replace({"FB": "RB"})
    t = targets.copy()
    t["position"] = t["player_id"].map(pos)
    cnt = (t.groupby(["season", "posteam", "position", "player_id"]).size()
           .rename("n").reset_index())
    cnt["rank"] = cnt.groupby(["season", "posteam", "position"])["n"].rank(method="first", ascending=False)
    wr = {1: "X", 2: "Z", 3: "Slot"}
    cnt["role"] = np.select(
        [cnt["position"].eq("WR"), cnt["position"].eq("TE") & cnt["rank"].eq(1), cnt["position"].eq("RB")],
        [cnt["rank"].map(wr), "Y", "RB"], default=None)
    return t.merge(cnt[["season", "posteam", "player_id", "role"]], on=["season", "posteam", "player_id"], how="left")


def role_table(seasons=None) -> dict:
    """League-wide share of targets by role against each coverage, with efficiency."""
    seasons = _default_seasons(seasons)
    t = _raw(seasons)["targets"]
    t = assign_roles(t[t["season"].isin(seasons)])
    out = {"n_targets": int(len(t)), "ALL": {}, "coverages": {}}
    def block(d):
        n = max(1, len(d))
        res = {"n": int(len(d))}
        for r in ROLES:
            x = d[d["role"] == r]
            m = max(1, len(x))
            res[r] = {"share": round(100 * len(x) / n, 1), "ypt": round(float(x["yds"].sum()) / m, 2),
                      "epa_t": round(float(x["epa"].sum()) / m, 2), "catch": round(100 * float(x["comp"].sum()) / m, 1),
                      "adot": round(float(x["air"].sum()) / m, 1)}
        return res
    out["ALL"] = block(t)
    for c in COV_ORDER:
        out["coverages"][c] = block(t[t["cov"] == c])
    out["note"] = ("Roles are target-rank stand-ins (X=WR1, Z=WR2, Slot=WR3, Y=TE1), not snap alignment. "
                   "A coverage whose share is above ALL is one that feeds that role.")
    return out


def team_profile(team: str, seasons=None) -> dict:
    """Coverage mix a team's defense runs, and how its offense's targets split by role."""
    seasons = _default_seasons(seasons)
    raw = _raw(seasons)
    team = sources._TEAM_CODE_FIX.get(team.upper(), team.upper())
    td = raw["team_def"]
    td = td[(td["defteam"] == team) & td["season"].isin(seasons)]
    mix = (td.groupby("cov")["plays"].sum())
    mix = (100 * mix / mix.sum()).round(1).reindex(COV_ORDER).fillna(0.0) if len(mix) else mix
    tg = assign_roles(raw["targets"])
    tg = tg[(tg["posteam"] == team) & tg["season"].isin(seasons)]
    by = {}
    for c in COV_ORDER:
        d = tg[tg["cov"] == c]
        n = max(1, len(d))
        by[c] = {"n": int(len(d)), **{r: round(100 * float((d["role"] == r).sum()) / n, 1) for r in ROLES}}
    n = max(1, len(tg))
    return {"team": team, "seasons": seasons,
            "defense_coverage_mix_pct": mix.to_dict() if len(mix) else {},
            "offense_target_share_all": {r: round(100 * float((tg["role"] == r).sum()) / n, 1) for r in ROLES},
            "offense_target_share_by_coverage": by}


def split_persistence(position: str = "WR", seasons=None, min_snaps: int = 120) -> dict:
    """Does a player's man-minus-zone gap repeat? Spearman correlation of each
    metric's gap in one season against the next, among players with `min_snaps`
    against both coverages in both seasons. A gap that doesn't repeat is luck."""
    seasons = _default_seasons(seasons)
    p = _raw(seasons)["player"]
    pos = _positions().set_index("player_id")["position"].replace({"FB": "RB"})
    p = p.assign(position=p["player_id"].map(pos))
    p = p[(p["position"] == position.upper()) & p["season"].isin(seasons)].copy()
    p["mz"] = np.where(p["cov"].isin(MAN), "man", "zone")
    g = p.groupby(["season", "player_id", "mz"])[_SUMS].sum()
    st = _stats(g).unstack("mz")
    st.columns = [f"{mz}_{c}" for c, mz in st.columns]
    st = st.dropna(subset=["man_snaps", "zone_snaps"])
    st = st[(st["man_snaps"] >= min_snaps) & (st["zone_snaps"] >= min_snaps)]
    res = {}
    for metric in ("tpps", "ypt", "yps", "epa_t", "catch"):
        gap = (st[f"man_{metric}"] - st[f"zone_{metric}"]).unstack("season")
        pairs = []
        for a, b in zip(sorted(seasons)[:-1], sorted(seasons)[1:]):
            if a in gap.columns and b in gap.columns:
                d = gap[[a, b]].dropna()
                if len(d) >= 8:
                    rho = float(d[a].rank().corr(d[b].rank()))
                    pairs.append({"from": int(a), "to": int(b), "n": int(len(d)),
                                  "spearman": round(rho, 2) if np.isfinite(rho) else None})
        res[metric] = pairs
    return {"position": position.upper(), "min_snaps_each_coverage": min_snaps, "year_to_year_gap_persistence": res}


# ----------------------------------------------------------------------------- backtests

def defense_man_rate() -> pd.DataFrame:
    """Share of each defense's tagged dropbacks that were man coverage, per team-season."""
    td = _raw()["team_def"].copy()
    td["man"] = td["cov"].isin(MAN)
    g = td.groupby(["season", "defteam"]).apply(
        lambda d: pd.Series({"man_rate": 100 * d.loc[d["man"], "plays"].sum() / d["plays"].sum(),
                             "plays": d["plays"].sum()}), include_groups=False).reset_index()
    return g.rename(columns={"defteam": "team"})


def schedule_man_exposure(season: int, prior_window: int = 2) -> pd.DataFrame:
    """How man-heavy a team's schedule is, from opponents' coverage in PRIOR seasons only.

    The schedule is public in May, so who a team plays is legitimately known; each
    opponent's man rate is pooled over the `prior_window` seasons before `season`, so
    nothing from the season being predicted leaks in. `sched_man_z` is standardised
    across teams that season."""
    rates = defense_man_rate()
    prior = rates[rates["season"].isin(range(season - prior_window, season))]
    opp = (prior.assign(m=prior["man_rate"] * prior["plays"]).groupby("team")
           .apply(lambda d: d["m"].sum() / d["plays"].sum(), include_groups=False).rename("opp_man"))
    sch = sources.schedules()
    sch = sch[(sch["season"] == season) & (sch["game_type"] == "REG")]
    games = pd.concat([sch[["home_team", "away_team"]].set_axis(["team", "opp"], axis=1),
                       sch[["away_team", "home_team"]].set_axis(["team", "opp"], axis=1)])
    games["opp_man"] = games["opp"].map(opp)
    out = games.groupby("team")["opp_man"].mean().rename("sched_man").reset_index()
    out["sched_man_z"] = (out["sched_man"] - out["sched_man"].mean()) / out["sched_man"].std(ddof=0)
    return out


def coverage_value_backtest(seasons, position: str = "WR", sc=None, prior_window: int = 3,
                            min_snaps: int = 100, min_games: int = 6) -> pd.DataFrame:
    """Would knowing a player's man/zone profile in August have helped predict his
    next-season fantasy output beyond what his own prior production already says?

    Same discipline as adp.matchup_value_backtest: nothing here has seen the season it
    scores. For each `season`: talent = that player's points per game the year before;
    coverage features come from the `prior_window` seasons before it (man vs zone yards
    per snap, and their gap `lean`); `sched_man_z` is how man-heavy the player's team
    schedule is, from opponents' prior-season coverage. The target is points per game
    in `season` for players with at least `min_games` games (coverage should not be
    credited or blamed for injuries)."""
    from .adp import season_finish
    from .config import Scoring
    sc = sc or Scoring()
    position = position.upper()
    frames = []
    for season in seasons:
        prior = [y for y in range(season - prior_window, season) if y in SEASONS]
        if not prior:
            continue
        sp = player_splits(prior)
        sp = sp[(sp["position"] == position) & (sp["man_snaps"] >= min_snaps) & (sp["zone_snaps"] >= min_snaps)]
        sp = sp[["player_id", "man_yps", "zone_yps", "man_tpps", "zone_tpps"]].copy()
        sp["lean"] = sp["man_yps"] - sp["zone_yps"]
        pf = season_finish(season - 1, sc)
        pf = pf[(pf["position"] == position) & (pf["games"] >= min_games)][["player_id", "ppg"]]
        pf = pf.rename(columns={"ppg": "prior_ppg"})
        fin = season_finish(season, sc)
        fin = fin[(fin["position"] == position) & (fin["games"] >= min_games)][["player_id", "name", "points", "ppg"]]
        w = sources.weekly_stats([season])
        team_of = (w[w["position"] == position].sort_values("week").groupby("player_id")["recent_team"]
                   .last().rename("team").reset_index())
        team_of["team"] = team_of["team"].replace(sources._TEAM_CODE_FIX)
        m = (fin.merge(pf, on="player_id").merge(sp, on="player_id").merge(team_of, on="player_id", how="left")
             .merge(schedule_man_exposure(season)[["team", "sched_man_z"]], on="team", how="left"))
        m["sched_man_z"] = m["sched_man_z"].fillna(0.0)
        m["season"] = season
        frames.append(m)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)[0]


def _predict(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X]) @ beta


def _spearman(a, b) -> float:
    return float(pd.Series(np.asarray(a)).rank().corr(pd.Series(np.asarray(b)).rank()))


def summarize_value_backtest(h: pd.DataFrame, top_n: int = 24, n_boot: int = 1000, seed: int = 0) -> dict:
    """Does coverage add anything to prior production? Two views, both on out-of-sample data.

    1. Coefficients: ppg ~ prior_ppg + lean_z + sched_z + lean_z*sched_z, with a bootstrap
       over *players* (a player appears in up to three seasons, so rows aren't independent).
       Effects are in fantasy points per game per 1 SD of the feature.
    2. Out-of-sample ranking: for each season, fit on the OTHER seasons and predict it.
       Compare Spearman and top-N precision for prior_ppg alone vs. adding coverage.
    """
    h = h.dropna(subset=["ppg", "prior_ppg", "lean"]).copy()
    if len(h) < 30 or h["season"].nunique() < 2:
        return {"n_player_seasons": int(len(h)), "error": "not enough player-seasons"}
    h["lean_z"] = (h["lean"] - h["lean"].mean()) / h["lean"].std(ddof=0)
    h["inter"] = h["lean_z"] * h["sched_man_z"]
    cols = {"M0 prior_ppg": ["prior_ppg"], "M1 +lean": ["prior_ppg", "lean_z"],
            "M2 +lean +schedule x lean": ["prior_ppg", "lean_z", "sched_man_z", "inter"]}

    def fit(df):
        return _ols(df[cols["M2 +lean +schedule x lean"]].to_numpy(), df["ppg"].to_numpy())

    beta = fit(h)
    rng = np.random.default_rng(seed)
    pids = h["player_id"].unique()
    groups = {k: g for k, g in h.groupby("player_id")}
    boots = []
    for _ in range(n_boot):
        samp = pd.concat([groups[k] for k in rng.choice(pids, len(pids))])
        boots.append(fit(samp))
    boots = np.array(boots)

    def coef(i):
        lo, hi = np.percentile(boots[:, i], [2.5, 97.5])
        return {"ppg_per_sd": round(float(beta[i]), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)],
                "excludes_zero": bool(lo > 0 or hi < 0)}

    oos = {k: [] for k in cols}
    prec = {k: [] for k in cols}
    pred_all = {k: pd.Series(index=h.index, dtype=float) for k in cols}
    for s in sorted(h["season"].unique()):
        tr, te = h[h["season"] != s], h[h["season"] == s]
        for k, c in cols.items():
            b = _ols(tr[c].to_numpy(), tr["ppg"].to_numpy())
            pred_all[k].loc[te.index] = _predict(b, te[c].to_numpy())
        actual_top = te["ppg"].rank(ascending=False, method="min") <= top_n
        for k in cols:
            top = pred_all[k].loc[te.index].sort_values(ascending=False).head(top_n).index
            prec[k].append(float(actual_top.loc[top].mean()))
    for k in cols:
        oos[k] = _spearman(pred_all[k], h["ppg"])
    base = "M0 prior_ppg"
    return {
        "n_player_seasons": int(len(h)), "n_players": int(len(pids)), "seasons": sorted(int(x) for x in h["season"].unique()),
        "top_n": top_n,
        "effect_of_lean": coef(2), "effect_of_schedule_x_lean": coef(4),
        "oos_spearman": {k: round(v, 4) for k, v in oos.items()},
        "oos_top_n_precision": {k: round(float(np.mean(v)), 4) for k, v in prec.items()},
        "improvement_spearman_M1": round(oos["M1 +lean"] - oos[base], 4),
        "improvement_spearman_M2": round(oos["M2 +lean +schedule x lean"] - oos[base], 4),
        "improvement_precision_M2": round(float(np.mean(prec["M2 +lean +schedule x lean"]) - np.mean(prec[base])), 4),
    }
