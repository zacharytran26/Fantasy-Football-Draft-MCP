"""Man-vs-zone coverage splits: joining, exposure, rates, roles, persistence."""
import pandas as pd
import pytest

from ffdraft import coverage as cv


def _pbp(rows):
    base = dict(season=2025, season_type="REG", qb_dropback=1, pass_attempt=1, sack=0, complete_pass=0,
                yards_gained=0.0, yards_after_catch=0.0, air_yards=0.0, epa=0.0, interception=0,
                pass_touchdown=0, receiver_player_id=None, posteam="AAA", defteam="BBB")
    return pd.DataFrame([{**base, "game_id": "g1", **r} for r in rows])


def _part(rows):
    return pd.DataFrame([{"nflverse_game_id": "g1", "offense_players": "A;B", **r} for r in rows])


def _raw():
    pbp = _pbp([
        dict(play_id=1, receiver_player_id="A", complete_pass=1, yards_gained=10.0, yards_after_catch=4.0, air_yards=6.0, epa=0.5),
        dict(play_id=2, receiver_player_id="A", air_yards=12.0, epa=-0.4),
        dict(play_id=3, receiver_player_id="B", complete_pass=1, yards_gained=20.0, yards_after_catch=2.0, air_yards=18.0, epa=1.0),
        dict(play_id=4, sack=1, pass_attempt=0),                                   # sack: snap, no target
        dict(play_id=5, receiver_player_id="A", complete_pass=1, yards_gained=7.0),  # no coverage tag
        dict(play_id=6, receiver_player_id="A", season_type="POST", complete_pass=1, yards_gained=50.0),
    ])
    part = _part([
        dict(play_id=1, defense_coverage_type="COVER_1"), dict(play_id=2, defense_coverage_type="COVER_3"),
        dict(play_id=3, defense_coverage_type="COVER_9"), dict(play_id=4, defense_coverage_type="COVER_1"),
        dict(play_id=6, defense_coverage_type="COVER_1"),
    ])
    return cv.build_raw(pbp, part)


class TestBuildRaw:
    def test_exposure_counts_every_tagged_dropback_for_everyone_on_the_field(self):
        p = _raw()["player"].set_index(["player_id", "cov"])
        assert p.loc[("A", "Cover 1"), "snaps"] == 2          # play 1 + the sack
        assert p.loc[("B", "Cover 1"), "snaps"] == 2

    def test_targets_and_yardage_only_from_completions(self):
        p = _raw()["player"].set_index(["player_id", "cov"])
        a1 = p.loc[("A", "Cover 1")]
        assert (a1["targets"], a1["rec"], a1["rec_yds"], a1["yac"]) == (1, 1, 10.0, 4.0)
        a3 = p.loc[("A", "Cover 3")]
        assert (a3["targets"], a3["rec"], a3["rec_yds"], a3["air_yards"]) == (1, 0, 0.0, 12.0)

    def test_cover_9_folds_into_cover_3(self):
        p = _raw()["player"].set_index(["player_id", "cov"])
        assert p.loc[("B", "Cover 3"), "targets"] == 1
        assert "Cover 9" not in set(_raw()["player"]["cov"])

    def test_untagged_plays_and_postseason_are_excluded(self):
        p = _raw()["player"]
        assert p[p["player_id"] == "A"]["rec_yds"].sum() == 10.0   # not +7 (untagged) or +50 (POST)

    def test_sack_is_not_a_target(self):
        assert _raw()["targets"]["comp"].size == 3


class TestStats:
    def test_rates_come_from_summed_counts(self):
        g = pd.DataFrame([dict(snaps=20, targets=5, rec=4, rec_yds=60.0, yac=12.0, air_yards=50.0, epa=2.5)])
        s = cv._stats(g).iloc[0]
        assert (s["tpps"], s["catch"], s["adot"], s["ypt"], s["yac"], s["yps"], s["epa_t"]) == (25.0, 80.0, 10.0, 12.0, 3.0, 3.0, 0.5)

    def test_zero_targets_and_snaps_do_not_divide_by_zero(self):
        g = pd.DataFrame([dict(snaps=0, targets=0, rec=0, rec_yds=0.0, yac=0.0, air_yards=0.0, epa=0.0)])
        assert cv._stats(g).iloc[0]["ypt"] == 0.0


class TestRoles:
    def test_roles_by_target_rank_within_team_season(self, monkeypatch):
        monkeypatch.setattr(cv, "_positions", lambda: pd.DataFrame({
            "player_id": ["w1", "w2", "w3", "w4", "t1", "t2", "r1"],
            "position": ["WR", "WR", "WR", "WR", "TE", "TE", "RB"]}))
        ids = ["w1"] * 5 + ["w2"] * 4 + ["w3"] * 3 + ["w4"] * 2 + ["t1"] * 3 + ["t2"] + ["r1"] * 2
        t = pd.DataFrame({"season": 2025, "posteam": "AAA", "player_id": ids})
        out = cv.assign_roles(t).drop_duplicates("player_id").set_index("player_id")["role"]
        assert out["w1"] == "X" and out["w2"] == "Z" and out["w3"] == "Slot"
        assert out["w4"] is None or pd.isna(out["w4"])
        assert out["t1"] == "Y" and (out["t2"] is None or pd.isna(out["t2"]))
        assert out["r1"] == "RB"

    def test_ranking_does_not_leak_across_teams(self, monkeypatch):
        monkeypatch.setattr(cv, "_positions", lambda: pd.DataFrame({"player_id": ["a", "b"], "position": ["WR", "WR"]}))
        t = pd.DataFrame({"season": 2025, "posteam": ["AAA", "BBB"], "player_id": ["a", "b"]})
        assert set(cv.assign_roles(t)["role"]) == {"X"}


class TestPersistence:
    def test_perfectly_persistent_gap_scores_one(self, monkeypatch):
        rows = []
        for i in range(12):
            for season in (2024, 2025):
                for cov, tg in (("Cover 1", 10 + i), ("Cover 3", 10)):
                    rows.append(dict(season=season, player_id=f"p{i}", cov=cov, snaps=200, targets=tg, rec=tg // 2,
                                     rec_yds=float(tg * 8), yac=0.0, air_yards=0.0, epa=0.0, ints=0, tds=0))
        monkeypatch.setattr(cv, "_raw", lambda seasons=None: {"player": pd.DataFrame(rows)})
        monkeypatch.setattr(cv, "_positions", lambda: pd.DataFrame({"player_id": [f"p{i}" for i in range(12)], "position": "WR"}))
        r = cv.split_persistence("WR", [2024, 2025], min_snaps=100)
        assert r["year_to_year_gap_persistence"]["tpps"][0]["spearman"] == pytest.approx(1.0)


class TestDraftFlowSummary:
    """The inline coverage read shown inside who_should_i_pick / best_available."""

    def _ctx(self, man_yps, zone_yps):
        row = {"position": "WR", "man_yps": man_yps, "zone_yps": zone_yps, "man_tpps": 25.0, "zone_tpps": 24.0,
               "man_catch": 60.0, "zone_catch": 70.0, "man_snaps": 300.0, "zone_snaps": 400.0}
        base = {"WR": {"man": {"yps": 1.5}, "zone": {"yps": 1.4}}}
        return {"by_id": {"p1": row}, "base": base}

    def test_no_verdict_is_attached_because_the_backtest_found_no_signal(self):
        from ffdraft.server import _coverage_summary
        assert "lean" not in _coverage_summary("p1", self._ctx(3.5, 1.0))

    def test_reports_both_sides_against_the_position_average(self):
        from ffdraft.server import _coverage_summary
        s = _coverage_summary("p1", self._ctx(2.5, 2.0))
        assert s["vs_man"]["yds_per_snap"] == 2.5 and s["vs_zone"]["snaps"] == 400
        assert s["position_avg_yds_per_snap"] == {"man": 1.5, "zone": 1.4}

    def test_missing_player_or_failed_load_is_none_not_an_error(self):
        from ffdraft.server import _coverage_summary
        assert _coverage_summary("zzz", self._ctx(2, 2)) is None
        assert _coverage_summary("p1", None) is None


class TestValueBacktest:
    def _frame(self, effect, n=240, seed=3):
        import numpy as np
        rng = np.random.default_rng(seed)
        df = pd.DataFrame({"player_id": [f"p{i % 120}" for i in range(n)], "season": [2023 + (i % 3) for i in range(n)],
                           "prior_ppg": rng.normal(12, 4, n), "lean": rng.normal(0, 0.5, n), "sched_man_z": rng.normal(0, 1, n)})
        lean_z = (df["lean"] - df["lean"].mean()) / df["lean"].std(ddof=0)
        df["ppg"] = 2 + 0.8 * df["prior_ppg"] + effect * lean_z + rng.normal(0, 2, n)
        return df

    def test_a_planted_effect_is_detected_and_improves_ranking(self):
        r = cv.summarize_value_backtest(self._frame(effect=2.0), top_n=20, n_boot=300)
        assert r["effect_of_lean"]["excludes_zero"] and r["improvement_spearman_M1"] > 0

    def test_pure_noise_is_not_reported_as_signal(self):
        r = cv.summarize_value_backtest(self._frame(effect=0.0), top_n=20, n_boot=300)
        assert not r["effect_of_lean"]["excludes_zero"] and r["improvement_spearman_M1"] < 0.02

    def test_too_few_rows_say_so_instead_of_fitting(self):
        assert "error" in cv.summarize_value_backtest(self._frame(1.0, n=12), n_boot=10)
