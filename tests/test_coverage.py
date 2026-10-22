"""Man-vs-zone coverage splits: joining, exposure, rates, roles, persistence."""
import numpy as np
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


class TestPocket:
    def _raw_with_pressure(self):
        pbp = _pbp([
            dict(play_id=1, receiver_player_id="A", complete_pass=1, yards_gained=12.0, epa=0.6),
            dict(play_id=2, receiver_player_id="A", complete_pass=1, yards_gained=2.0, epa=-0.1),
            dict(play_id=3, sack=1, pass_attempt=0, epa=-1.2, yards_gained=-7.0),
            dict(play_id=4, receiver_player_id="B", interception=1, epa=-4.0),
            dict(play_id=5, receiver_player_id="B", epa=-0.5),
        ])
        pbp["qb_scramble"] = 0
        part = _part([
            dict(play_id=1, defense_coverage_type="COVER_1", was_pressure=False, time_to_throw=2.5, number_of_pass_rushers=4),
            dict(play_id=2, defense_coverage_type="COVER_1", was_pressure=True, time_to_throw=2.0, number_of_pass_rushers=5),
            dict(play_id=3, defense_coverage_type="COVER_1", was_pressure=True, time_to_throw=3.5, number_of_pass_rushers=5),
            dict(play_id=4, defense_coverage_type="COVER_3", was_pressure=True, time_to_throw=2.4, number_of_pass_rushers=4),
            dict(play_id=5, defense_coverage_type="COVER_3", was_pressure=False, time_to_throw=2.9, number_of_pass_rushers=4),
        ])
        return cv.build_raw(pbp, part)

    def test_pressure_splits_a_players_snaps_and_targets(self):
        p = self._raw_with_pressure()["player"].set_index(["player_id", "cov", "pocket"])
        assert p.loc[("A", "Cover 1", "clean"), "targets"] == 1 and p.loc[("A", "Cover 1", "pressured"), "targets"] == 1
        assert p.loc[("A", "Cover 1", "pressured"), "snaps"] == 2     # the pressured catch and the sack

    def test_outcomes_partition_every_tagged_dropback(self):
        db = self._raw_with_pressure()["dropbacks"]
        assert db["n"].sum() == 5
        by = db.groupby("outcome")["n"].sum().to_dict()
        assert by == {"good_catch": 1, "marginal_catch": 1, "sack": 1, "interception": 1, "incomplete": 1}

    def test_time_to_throw_buckets_and_missing_columns(self):
        db = self._raw_with_pressure()["dropbacks"]
        assert set(db["ttt"]) == {"<2.2s", "2.2-2.7s", "2.7-3.2s", ">3.2s"}
        assert "pocket" in _raw()["player"].columns          # synthetic data with no pressure columns -> all clean
        assert set(_raw()["player"]["pocket"]) == {"clean"}

    def test_unknown_pocket_is_rejected(self):
        import pytest
        with pytest.raises(ValueError):
            cv._pocket(_raw()["player"], "sideways")
        assert len(cv._pocket(self._raw_with_pressure()["player"], "pressured")) > 0

    def test_pocket_outcomes_and_protection_from_the_aggregate(self, monkeypatch):
        raw = self._raw_with_pressure()
        monkeypatch.setattr(cv, "_raw", lambda seasons=None: raw)
        o = cv.pocket_outcomes([2025])
        assert o["by_pocket"]["pressured"]["sack"] == pytest.approx(100 / 3, abs=0.1)   # 1 of 3 pressured dropbacks
        assert o["by_coverage"]["Cover 1"]["pressure_rate_pct"] == pytest.approx(66.7, abs=0.1)
        prot = cv.team_protection([2025]).iloc[0]
        assert prot["team"] == "AAA" and prot["pressure_rate"] == pytest.approx(60.0)


class TestPressureRebound:
    def _frame(self, rebound, n_teams=32, per_team=6, seed=5):
        rng = np.random.default_rng(seed)
        rows = []
        for season in (2023, 2024):
            for i in range(n_teams):
                pct = (i + 1) / n_teams
                for j in range(per_team):
                    prior = rng.normal(12, 3)
                    rows.append(dict(season=season, team=f"T{i}", position="WR", player_id=f"{season}-{i}-{j}", stayed=True,
                                     prior_ppg=prior, prior_pct=pct, prior_pressure=20 + 10 * pct,
                                     same_pressure=25 + rng.normal(0, 2), delta_z=rng.normal(),
                                     ppg=prior + rebound * (pct - 0.5) * 2 + rng.normal(0, 1.5)))
        h = pd.DataFrame(rows)
        h["resid"] = h["ppg"] - np.polyval(np.polyfit(h["prior_ppg"], h["ppg"], 1), h["prior_ppg"])
        return h

    def test_a_planted_rebound_is_detected(self):
        r = cv.summarize_pressure_rebound(self._frame(rebound=2.0), n_boot=300)["WR"]["stayed_on_team"]
        assert r["rebound_ppg"] > 1.0 and r["excludes_zero"]

    def test_no_rebound_is_not_reported_as_one(self):
        r = cv.summarize_pressure_rebound(self._frame(rebound=0.0), n_boot=300)["WR"]["stayed_on_team"]
        assert not r["excludes_zero"] and abs(r["rebound_ppg"]) < 0.5

    def test_mean_reversion_stat_is_reported(self):
        out = cv.summarize_pressure_rebound(self._frame(rebound=0.0), n_boot=50)
        assert set(out["team_mean_reversion"]) == {"corr_prior_pressure_vs_change", "corr_prior_vs_same_pressure"}


class TestPressureUsage:
    def _frame(self, effect, n_teams=32, per_team=4, seed=9):
        rng = np.random.default_rng(seed)
        rows = []
        for season in (2023, 2024, 2025):
            for i in range(n_teams):
                pct = (i + 1) / n_teams
                for _ in range(per_team):
                    prior = rng.normal(5, 1.5)
                    rows.append(dict(season=season, team=f"T{i}", position="TE", tgt_g_p=prior, yds_g_p=prior * 8,
                                     prior_pct=pct, press_share_p=rng.uniform(0.1, 0.4), short_share_p=rng.uniform(0.1, 0.5),
                                     tgt_g=prior + effect * (pct - 0.5) * 2 + rng.normal(0, 0.8),
                                     yds_g=prior * 8 + rng.normal(0, 8)))
        return pd.DataFrame(rows)

    def test_planted_regression_in_te_volume_is_detected(self):
        r = cv.summarize_pressure_usage(self._frame(effect=-1.0), "TE", n_boot=200)
        e = r["outcomes"]["tgt_g"]["team_pressure_allowed_prior"]
        assert e["effect_per_sd"] < -0.2 and e["excludes_zero"] and e["oos_mse_change_pct"] < 0

    def test_noise_is_not_reported_as_an_effect(self):
        r = cv.summarize_pressure_usage(self._frame(effect=0.0), "TE", n_boot=200)
        assert not r["outcomes"]["tgt_g"]["team_pressure_allowed_prior"]["excludes_zero"]

    def test_too_few_rows_say_so(self):
        assert "error" in cv.summarize_pressure_usage(self._frame(0.0, n_teams=2, per_team=2), "TE", n_boot=10)
