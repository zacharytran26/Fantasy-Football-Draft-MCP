"""bootstrap_ci: block-bootstrap confidence intervals over whole seasons, shared
by every backtest in adp.py."""
import pandas as pd

from ffdraft import adp


def _mean_summary(hist: pd.DataFrame) -> dict:
    return {"mean": float(hist["value"].mean())}


class TestBootstrapCI:
    def test_no_variance_when_every_group_is_identical(self):
        hist = pd.DataFrame({
            "season": [2021, 2021, 2022, 2022, 2023, 2023],
            "value": [10, 10, 10, 10, 10, 10],
        })
        out = adp.bootstrap_ci(hist, _mean_summary, ["mean"], n_boot=50)
        assert out["mean"]["point"] == 10.0
        assert out["mean"]["lo"] == 10.0
        assert out["mean"]["hi"] == 10.0

    def test_single_group_returns_no_interval(self):
        hist = pd.DataFrame({"season": [2021] * 4, "value": [1, 2, 3, 4]})
        out = adp.bootstrap_ci(hist, _mean_summary, ["mean"])
        assert out["mean"]["point"] == 2.5
        assert out["mean"]["lo"] is None
        assert out["mean"]["hi"] is None

    def test_interval_brackets_the_point_estimate_for_varying_seasons(self):
        hist = pd.DataFrame({
            "season": [2021] * 3 + [2022] * 3 + [2023] * 3 + [2024] * 3,
            "value": [1, 1, 1, 5, 5, 5, 9, 9, 9, 5, 5, 5],
        })
        out = adp.bootstrap_ci(hist, _mean_summary, ["mean"], n_boot=500, seed=1)
        assert out["mean"]["lo"] <= out["mean"]["point"] <= out["mean"]["hi"]
        assert out["mean"]["lo"] < out["mean"]["hi"]  # real season-to-season spread

    def test_missing_metric_in_some_resamples_is_skipped_not_crashed(self):
        hist = pd.DataFrame({"season": [2021, 2022, 2023], "value": [1, 2, 3]})

        def summary_fn(h):
            big = h[h["value"] > 2]
            out = {"mean": float(h["value"].mean())}
            if not big.empty:
                out["ratio"] = float(len(big) / len(h))
            return out

        out = adp.bootstrap_ci(hist, summary_fn, ["mean", "ratio"], n_boot=50)
        assert out["mean"]["lo"] is not None
        assert "ratio" in out  # present even if too few draws had it defined
