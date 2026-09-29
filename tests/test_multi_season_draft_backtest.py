"""multi_season_draft_backtest: aggregates draft_backtest across real seasons into
a scorecard for whether the live model actually beats what you drafted. Mocks
draft_backtest itself (heavy: full leak-free board construction) rather than the
underlying data sources -- this module is about the aggregation math, not
draft_backtest's own per-pick replay logic."""
import pandas as pd

from ffdraft import adp


def _round(overall, round_, your_pts, algo_pts, optimal_pts):
    return {
        "round": round_, "overall": overall,
        "your_pick": "You", "your_points": your_pts,
        "algo_pick": "Algo", "algo_points": algo_pts,
        "optimal_pick": "Optimal", "optimal_points": optimal_pts,
    }


def _draft_backtest_result(rounds):
    return {"totals": {}, "rounds": rounds}


class TestMultiSeasonDraftBacktest:
    def test_aggregates_real_picks_across_seasons(self, monkeypatch):
        def fake_draft_backtest(league_id, season, top_n=3):
            if season == 2021:
                return _draft_backtest_result([
                    _round(3, 1, 100.0, 120.0, 150.0),
                    _round(23, 2, 80.0, 70.0, 100.0),
                ])
            return _draft_backtest_result([
                _round(3, 1, 90.0, 130.0, 140.0),
            ])

        monkeypatch.setattr(adp, "draft_backtest", fake_draft_backtest)

        hist = adp.multi_season_draft_backtest("12856", [2021, 2022])
        assert len(hist) == 3
        assert set(hist["season"]) == {2021, 2022}

        summary = adp.multi_season_draft_backtest_summary(hist)
        assert summary["n_picks"] == 3
        assert summary["total_your_points"] == 270.0
        assert summary["total_algo_points"] == 320.0
        assert summary["total_optimal_points"] == 390.0
        # Algo beat "you" in 2 of 3 picks (120>100, 130>90), not in the third (70<80).
        assert summary["algo_beats_your_pick_rate"] == pytest_approx(2 / 3)
        assert summary["algo_pct_of_optimal"] == pytest_approx(320.0 / 390.0)
        assert summary["your_pct_of_optimal"] == pytest_approx(270.0 / 390.0)

    def test_kd_st_rounds_are_excluded(self, monkeypatch):
        def fake_draft_backtest(league_id, season, top_n=3):
            rounds = [_round(3, 1, 100.0, 110.0, 120.0)]
            kdst = _round(150, 15, None, None, None)
            kdst["your_points"] = None  # draft_backtest's real "not modelled" marker
            rounds.append(kdst)
            return _draft_backtest_result(rounds)

        monkeypatch.setattr(adp, "draft_backtest", fake_draft_backtest)

        hist = adp.multi_season_draft_backtest("12856", [2021])
        assert len(hist) == 1  # the K/DST round is dropped

    def test_a_season_draft_backtest_cant_replay_is_skipped(self, monkeypatch):
        def fake_draft_backtest(league_id, season, top_n=3):
            if season == 2020:
                return {"error": "no draft found for league 12856 in 2020"}
            return _draft_backtest_result([_round(3, 1, 100.0, 110.0, 120.0)])

        monkeypatch.setattr(adp, "draft_backtest", fake_draft_backtest)

        hist = adp.multi_season_draft_backtest("12856", [2020, 2021])
        assert set(hist["season"]) == {2021}

    def test_empty_history_returns_empty_summary(self):
        assert adp.multi_season_draft_backtest_summary(pd.DataFrame()) == {"n_picks": 0}

    def test_early_late_round_split_can_diverge_from_the_flat_average(self):
        # Algo dominates round 1, loses badly in round 10 -- a flat average would
        # net these out and hide that the split is actually lopsided.
        hist = pd.DataFrame([
            {"season": 2021, "round": 1, "your_points": 100.0, "algo_points": 150.0,
             "optimal_points": 200.0},
            {"season": 2021, "round": 10, "your_points": 100.0, "algo_points": 50.0,
             "optimal_points": 150.0},
        ])
        summary = adp.multi_season_draft_backtest_summary(hist, early_late_cutoff=6)
        assert summary["algo_improvement_over_you_per_pick"] == 0.0  # (+50, -50) averages to 0
        assert summary["early_rounds_improvement_per_pick"] == 50.0
        assert summary["late_rounds_improvement_per_pick"] == -50.0


def pytest_approx(x):
    import pytest
    return pytest.approx(x, abs=1e-9)
