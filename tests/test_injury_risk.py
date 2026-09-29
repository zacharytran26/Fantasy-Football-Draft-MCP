"""features.injury_risk: recalibrated after adp.injury_risk_backtest found the
original hand-weighted blend miscalibrated against real games played -- these
lock in the two real findings that drove the fix (workload burden should raise,
not lower, expected games; the blend should beat its raw games_missed_rate
ingredient) so neither regresses silently again."""
import pandas as pd
import pytest

from ffdraft import features, sources
from ffdraft.features import _INJURY_MODEL_COEF, _INJURY_MODEL_INTERCEPT


def _profile_row(player_id, position, games, touches, season=2024):
    return {"season": season, "player_id": player_id, "position": position,
           "games": games, "touches": touches}


def _mock_sources(monkeypatch):
    monkeypatch.setattr(sources, "injuries", lambda: pd.DataFrame(
        columns=["gsis_id", "season", "report_status"]))
    # One unrelated roster row -- enough for the age lookup's season.max() to
    # resolve without crashing on an empty frame; no test here depends on age.
    monkeypatch.setattr(sources, "weekly_rosters", lambda: pd.DataFrame(
        [{"gsis_id": "irrelevant", "week": 1, "birth_date": "1995-01-01", "season": 2024}]))


class TestInjuryRisk:
    def test_durable_heavy_workload_player_gets_lower_risk_than_injury_prone_one(
            self, monkeypatch):
        _mock_sources(monkeypatch)
        profiles = pd.DataFrame([
            _profile_row("durable", "RB", games=17, touches=340),
            _profile_row("fragile", "RB", games=8, touches=120),
        ])
        out = features.injury_risk(profiles)
        durable = out[out["player_id"] == "durable"].iloc[0]
        fragile = out[out["player_id"] == "fragile"].iloc[0]
        assert durable["injury_risk"] < fragile["injury_risk"]

    def test_heavier_workload_lowers_risk_not_raises_it(self, monkeypatch):
        # The bug this regression-tests: the retired formula treated workload
        # burden as risk-additive, but adp.injury_risk_backtest found it real but
        # backwards -- heavy touches mostly identifies a valued, healthy starter
        # getting real opportunity (+0.396 Spearman with next-season games
        # played), not a wear-and-tear signal. Two players with identical
        # availability history but different workload should NOT have the
        # heavier-workload one come out riskier.
        _mock_sources(monkeypatch)
        profiles = pd.DataFrame([
            _profile_row("heavy", "RB", games=16, touches=340),
            _profile_row("light", "RB", games=16, touches=150),
        ])
        out = features.injury_risk(profiles)
        heavy = out[out["player_id"] == "heavy"].iloc[0]
        light = out[out["player_id"] == "light"].iloc[0]
        assert heavy["recent_burden"] > light["recent_burden"]
        assert heavy["injury_risk"] < light["injury_risk"]

    def test_formula_matches_the_documented_coefficients(self, monkeypatch):
        _mock_sources(monkeypatch)
        profiles = pd.DataFrame([_profile_row("p1", "WR", games=17, touches=145)])
        out = features.injury_risk(profiles)
        row = out.iloc[0]

        c = _INJURY_MODEL_COEF
        expected_exp_games = (
            _INJURY_MODEL_INTERCEPT
            + c["games_missed_rate"] * row["games_missed_rate"]
            + c["report_rate"] * row["report_rate"]
            + c["recent_burden"] * row["recent_burden"]
            + c["pos_base"] * 0.20  # WR base rate
        )
        expected_exp_games = min(17.0, max(7.0, expected_exp_games))
        expected_risk = min(0.85, max(0.02, (17 - expected_exp_games) / 17))
        assert row["injury_risk"] == pytest.approx(expected_risk, abs=1e-6)

    def test_output_is_bounded_and_has_expected_columns(self, monkeypatch):
        _mock_sources(monkeypatch)
        profiles = pd.DataFrame([
            _profile_row("never_plays", "RB", games=0, touches=0),
            _profile_row("iron_man", "QB", games=17, touches=600),
        ])
        out = features.injury_risk(profiles)
        assert set(out.columns) == {"player_id", "position", "age", "games_missed_rate",
                                    "report_rate", "heavy_seasons", "recent_burden",
                                    "injury_risk"}
        assert out["injury_risk"].between(0.02, 0.85).all()
