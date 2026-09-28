"""position_scarcity_entropy_backtest: does model.position_scarcity_entropy predict
the real cost of waiting on a position, and does it add anything over the
survival-probability mechanism already live in recommend()? Data assembly and the
correlation direction are tested here with a hand-built board; running it for real
against several seasons (see server.position_scarcity_entropy_backtest's docstring)
is what actually decides whether the feature earns its keep."""
import pandas as pd

from ffdraft import adp, board as bd, model
from ffdraft.config import LeagueSettings
from ffdraft.names import normalize as norm_name


def _synthetic_board() -> pd.DataFrame:
    """20 players, RB and WR interleaved 1:1 in intended ADP-rank order. RB's scores
    decay exponentially (100, 60, 36, 22, ...) so the pool stays genuinely lumpy at
    every prefix -- there's always a real gap between the best remaining RB and the
    rest. WR's scores decay by 1 each rank (40, 39, 38, ...) so the pool stays flat
    throughout -- any remaining WR is about as good as the next. Real cost of
    waiting should read consistently higher for RB than WR at every pick sampled,
    the same direction position_scarcity_entropy (low entropy = high risk) predicts.
    """
    rb_scores = [100, 60, 36, 22, 14, 9, 6, 4, 3, 2]
    wr_scores = [40, 39, 38, 37, 36, 35, 34, 33, 32, 31]
    rows = []
    for i in range(10):
        rows.append((f"RB_{i}", "RB", rb_scores[i]))
        rows.append((f"WR_{i}", "WR", wr_scores[i]))
    df = pd.DataFrame(rows, columns=["name", "position", "draft_score"])
    df["overall_rank"] = range(1, len(df) + 1)
    df["pos_rank"] = df.groupby("position")["draft_score"].rank(ascending=False).astype(int)
    return df


def _fake_load_adp(board: pd.DataFrame):
    def load(season=None, **kwargs):
        return pd.DataFrame({
            "_key": [norm_name(n) for n in board["name"]],
            "adp": board["overall_rank"].astype(float),
        })
    return load


class TestPositionScarcityEntropyBacktest:
    def test_cliff_position_reads_higher_risk_and_higher_cost_than_flat_position(
            self, monkeypatch):
        synth = _synthetic_board()

        monkeypatch.setattr(model, "build_player_table", lambda league, weights, season: pd.DataFrame())
        monkeypatch.setattr(model, "project", lambda tbl, league, weights: synth.copy())
        monkeypatch.setattr(bd, "load_adp", _fake_load_adp(synth))

        league = LeagueSettings(teams=4)
        hist = adp.position_scarcity_entropy_backtest(
            [2024], league=league, window=3, stride=1, cutoff=20)

        assert not hist.empty
        assert set(hist["position"]) == {"RB", "WR"}

        # At every pick sampled, RB's remaining pool stays lumpier (lower entropy)
        # than WR's flat one, and on average costs more (as a fraction of its own
        # current best value) to wait on -- the near-empty tail end of the window is
        # noisy pick to pick, so this checks the trend rather than every single row.
        assert (hist.loc[hist["position"] == "RB", "entropy"].to_numpy()
               < hist.loc[hist["position"] == "WR", "entropy"].to_numpy()).all()
        assert (hist.loc[hist["position"] == "RB", "cost_frac"].mean()
               > hist.loc[hist["position"] == "WR", "cost_frac"].mean())

        summary = adp.position_scarcity_entropy_backtest_summary(hist)
        assert summary["n_samples"] == len(hist)
        # Entropy should track the actual drop-off in the expected direction on this
        # deliberately lopsided board.
        assert summary["entropy_corr"] > 0.5
        for key in ("improvement_vs_raw_gap", "improvement_vs_pool_size",
                   "improvement_vs_implied_cost"):
            assert key in summary

    def test_board_too_small_for_window_returns_empty(self, monkeypatch):
        synth = _synthetic_board()
        monkeypatch.setattr(model, "build_player_table", lambda league, weights, season: pd.DataFrame())
        monkeypatch.setattr(model, "project", lambda tbl, league, weights: synth.copy())
        monkeypatch.setattr(bd, "load_adp", _fake_load_adp(synth))

        league = LeagueSettings(teams=4)
        hist = adp.position_scarcity_entropy_backtest(
            [2024], league=league, window=30, stride=1, cutoff=20)

        assert hist.empty
        assert adp.position_scarcity_entropy_backtest_summary(hist) == {"n_samples": 0}
