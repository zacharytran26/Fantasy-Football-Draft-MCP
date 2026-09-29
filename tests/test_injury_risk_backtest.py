"""injury_risk_backtest: does features.injury_risk (already live in exp_games
and the projection discount) actually predict real games played, and does the
full blend beat its own raw games_missed_rate ingredient?"""
import pandas as pd

from ffdraft import adp, features, sources


def _profile_row(season, player_id, name, position, games, touches=100):
    return {"season": season, "player_id": player_id, "player_display_name": name,
           "position": position, "games": games, "touches": touches,
           "target_share": 0.15, "snap_share": 0.6, "team": "SF"}


def _weekly_row(season, week, player_id):
    return {"season": season, "week": week, "player_id": player_id, "season_type": "REG"}


def _roster_row(season, week, gsis_id, status="ACT"):
    return {"season": season, "week": week, "gsis_id": gsis_id, "status": status,
           "game_type": "REG"}


class TestInjuryRiskBacktest:
    def test_excludes_players_who_left_the_league(self, monkeypatch):
        # p1 stays in the league all of 2025 (rostered every week) but only plays a
        # handful of games -- a real in-season injury case, should be KEPT.
        # p2 played a full 2024 but is rostered nowhere in 2025 -- retired/cut,
        # should be EXCLUDED even though his "17 minus games played" looks like a
        # total wipeout.
        profiles = pd.DataFrame([
            _profile_row(2024, "p1", "Hurt Guy", "RB", games=17),
            _profile_row(2024, "p2", "Retired Guy", "RB", games=17),
        ])
        monkeypatch.setattr(features, "player_season_profiles", lambda sc, bonus, seasons: profiles)

        def fake_injury_risk(profiles):
            return pd.DataFrame([
                {"player_id": "p1", "position": "RB", "age": 26.0,
                 "games_missed_rate": 0.1, "report_rate": 0.1, "heavy_seasons": 1,
                 "recent_burden": 1.0, "injury_risk": 0.3},
                {"player_id": "p2", "position": "RB", "age": 26.0,
                 "games_missed_rate": 0.1, "report_rate": 0.1, "heavy_seasons": 1,
                 "recent_burden": 1.0, "injury_risk": 0.3},
            ])
        monkeypatch.setattr(features, "injury_risk", fake_injury_risk)

        weekly = pd.DataFrame([_weekly_row(2025, wk, "p1") for wk in range(1, 5)])
        monkeypatch.setattr(sources, "weekly_stats", lambda seasons: weekly)

        rosters = pd.DataFrame(
            [_roster_row(2025, wk, "p1") for wk in range(1, 18)]
            # p2 has zero rows in the 2025 roster feed at all.
        )
        monkeypatch.setattr(sources, "weekly_rosters", lambda seasons: rosters)

        hist = adp.injury_risk_backtest([2025], min_prior_games=6)
        assert list(hist["player_id"]) == ["p1"]
        assert hist.iloc[0]["games_actual"] == 4

    def test_low_prior_games_is_excluded(self, monkeypatch):
        profiles = pd.DataFrame([_profile_row(2024, "p1", "Bench Guy", "WR", games=3)])
        monkeypatch.setattr(features, "player_season_profiles", lambda sc, bonus, seasons: profiles)
        monkeypatch.setattr(features, "injury_risk", lambda profiles: pd.DataFrame([
            {"player_id": "p1", "position": "WR", "age": 26.0, "games_missed_rate": 0.1,
             "report_rate": 0.1, "heavy_seasons": 0, "recent_burden": 0.5, "injury_risk": 0.2},
        ]))
        monkeypatch.setattr(sources, "weekly_stats", lambda seasons: pd.DataFrame(
            columns=["season", "week", "player_id", "season_type"]))
        monkeypatch.setattr(sources, "weekly_rosters", lambda seasons: pd.DataFrame(
            columns=["season", "week", "gsis_id", "status", "game_type"]))

        hist = adp.injury_risk_backtest([2025], min_prior_games=6)
        assert hist.empty

    def test_empty_profiles_is_skipped(self, monkeypatch):
        monkeypatch.setattr(features, "player_season_profiles",
                           lambda sc, bonus, seasons: pd.DataFrame())
        hist = adp.injury_risk_backtest([2099])
        assert hist.empty

    def test_empty_history_returns_empty_summary(self):
        assert adp.injury_risk_backtest_summary(pd.DataFrame()) == {"n_players": 0}

    def test_summary_scores_correlation_and_mae(self):
        # A clean, small case: injury_risk correctly orders both players, and
        # exp_games_pred is closer to the actual than games_last for one of them.
        hist = pd.DataFrame([
            {"season": 2025, "player_id": "p1", "position": "RB", "injury_risk": 0.2,
             "games_missed_rate": 0.1, "games_last": 17, "exp_games_pred": 15.0,
             "games_actual": 16},
            {"season": 2025, "player_id": "p2", "position": "RB", "injury_risk": 0.6,
             "games_missed_rate": 0.5, "games_last": 10, "exp_games_pred": 10.0,
             "games_actual": 8},
        ])
        summary = adp.injury_risk_backtest_summary(hist)
        assert summary["n_players"] == 2
        assert summary["mean_abs_error_model"] >= 0
        assert "injury_risk_is_worth_it" in summary
