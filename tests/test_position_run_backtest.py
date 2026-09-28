"""position_run_backtest: does board.PositionMarkov predict the next pick's
position better than baselines that ignore the current one? Data assembly tested
here; the scoring math (position_run_backtest_summary) is exercised directly too,
since it doesn't share matchup_backtest_summary's harness."""
import pandas as pd

from ffdraft import adp


def _ecr_rows(season: int, order: list[tuple[str, str]]) -> pd.DataFrame:
    """order: (name, position) pairs already in ECR (draft-cost) rank order."""
    rows = []
    for rank, (name, pos) in enumerate(order, start=1):
        rows.append({
            "player": name, "pos": pos, "tm": "NA", "ecr": float(rank),
            "sd": 1.0, "best": rank, "worst": rank,
            "scrape_date": pd.Timestamp(f"{season}-08-15"), "page_type": "redraft-overall",
        })
    return pd.DataFrame(rows)


def _alternating_order(n: int = 40) -> list[tuple[str, str]]:
    return [(f"P{i}", "RB" if i % 2 == 0 else "WR") for i in range(n)]


class TestPositionRunBacktest:
    def test_markov_beats_baselines_on_a_learnable_alternating_pattern(self, monkeypatch):
        # Every season strictly alternates RB/WR -- a first-order Markov chain
        # should learn P(WR | RB) = 1 and P(RB | WR) = 1, which the
        # marginal-frequency baseline (50/50 regardless of current pick) and the
        # persistence baseline ("assume the run continues") both get wrong on
        # every single transition.
        seasons = [2021, 2022, 2023, 2024]

        def fake_ecr_raw(page_type="redraft-overall"):
            return pd.concat([_ecr_rows(s, _alternating_order()) for s in seasons],
                            ignore_index=True)

        monkeypatch.setattr(adp, "_ecr_raw", fake_ecr_raw)

        hist = adp.position_run_backtest(seasons)
        assert not hist.empty
        assert set(hist["season"]) == set(seasons)

        summary = adp.position_run_backtest_summary(hist)
        assert summary["markov_top1_accuracy"] == 1.0
        assert summary["persistence_top1_accuracy"] == 0.0
        assert summary["marginal_top1_accuracy"] < summary["markov_top1_accuracy"]
        assert summary["improvement_logloss_vs_marginal"] > 0
        assert summary["improvement_logloss_vs_uniform"] > 0
        assert summary["improvement_accuracy_vs_marginal"] > 0
        assert summary["improvement_accuracy_vs_persistence"] > 0

    def test_no_signal_when_position_order_is_independent_noise(self, monkeypatch):
        # Each season is an independently shuffled RB/WR coin flip -- no transition
        # structure carries over from training seasons to the held-out one, so the
        # Markov model shouldn't beat marginal frequency by any real margin (both
        # converge to ~50/50 either way). A fixed seed keeps this reproducible.
        import random

        seasons = [2021, 2022, 2023, 2024]

        def fake_ecr_raw(page_type="redraft-overall"):
            rng = random.Random(0)
            frames = []
            for s in seasons:
                positions = ["RB"] * 20 + ["WR"] * 20
                rng.shuffle(positions)
                order = [(f"P{s}_{i}", pos) for i, pos in enumerate(positions)]
                frames.append(_ecr_rows(s, order))
            return pd.concat(frames, ignore_index=True)

        monkeypatch.setattr(adp, "_ecr_raw", fake_ecr_raw)

        hist = adp.position_run_backtest(seasons)
        summary = adp.position_run_backtest_summary(hist)
        # No real signal to learn -- improvement should stay small in magnitude
        # rather than look confidently better (or confidently worse) than chance.
        assert abs(summary["improvement_logloss_vs_marginal"]) < 0.1
        assert abs(summary["improvement_accuracy_vs_marginal"]) < 0.2

    def test_too_few_seasons_returns_empty(self, monkeypatch):
        def fake_ecr_raw(page_type="redraft-overall"):
            return _ecr_rows(2024, _alternating_order())

        monkeypatch.setattr(adp, "_ecr_raw", fake_ecr_raw)

        hist = adp.position_run_backtest([2024])
        assert hist.empty
        assert adp.position_run_backtest_summary(hist) == {"n_transitions": 0}
