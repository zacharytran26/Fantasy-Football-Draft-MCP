"""Team-level context features: drive efficiency, red zone play-calling identity,
and bye weeks."""
import pandas as pd

from ffdraft import features, sources
from ffdraft.features import _redzone_identity_shift, _team_drive_efficiency


def _game(season, week, home, away, game_type="REG"):
    return {"season": season, "week": week, "home_team": home, "away_team": away,
           "game_type": game_type}


def _play(season, team, play_type, yardline_100, drive, fixed_drive_result, is_pass):
    return {
        "season": season, "posteam": team, "play_type": play_type,
        "pass": 1 if is_pass else 0, "rush": 0 if is_pass else 1,
        "yardline_100": yardline_100, "drive": drive,
        "fixed_drive_result": fixed_drive_result,
    }


class TestTeamDriveEfficiency:
    def test_counts_drive_outcomes_once_per_drive_not_per_play(self):
        pbp = pd.DataFrame([
            _play(2025, "BUF", "pass", 50, 1, "Touchdown", True),
            _play(2025, "BUF", "run", 5, 1, "Touchdown", False),   # same drive, same result
            _play(2025, "BUF", "run", 60, 2, "Punt", False),
            _play(2025, "BUF", "pass", 40, 3, "Field goal", True),
        ])
        out = _team_drive_efficiency(pbp)
        row = out[(out["team"] == "BUF") & (out["season"] == 2025)].iloc[0]
        assert row["drives"] == 3
        assert row["pct_td"] == pytest_approx(100 / 3)
        assert row["pct_punt"] == pytest_approx(100 / 3)
        assert row["pct_fg"] == pytest_approx(100 / 3)

    def test_missing_column_returns_empty_frame_not_a_crash(self):
        pbp = pd.DataFrame([{"season": 2025, "posteam": "BUF", "drive": 1}])
        out = _team_drive_efficiency(pbp)
        assert out.empty
        assert "pct_td" in out.columns


class TestRedzoneIdentityShift:
    def test_run_heavy_redzone_team_shows_positive_shift(self):
        pbp = pd.DataFrame([
            _play(2025, "PHI", "pass", 50, 1, "Touchdown", True),
            _play(2025, "PHI", "pass", 45, 1, "Touchdown", True),
            _play(2025, "PHI", "run", 10, 1, "Touchdown", False),
            _play(2025, "PHI", "run", 5, 1, "Touchdown", False),
        ])
        out = _redzone_identity_shift(pbp)
        row = out[(out["team"] == "PHI") & (out["season"] == 2025)].iloc[0]
        assert row["neutral_pass_rate"] == 100.0
        assert row["rz_pass_rate"] == 0.0
        assert row["shift"] == pytest_approx(100.0)

    def test_flat_shift_team_shows_near_zero(self):
        pbp = pd.DataFrame([
            _play(2025, "ARI", "pass", 50, 1, "Touchdown", True),
            _play(2025, "ARI", "run", 45, 1, "Touchdown", False),
            _play(2025, "ARI", "pass", 10, 2, "Touchdown", True),
            _play(2025, "ARI", "run", 5, 2, "Touchdown", False),
        ])
        out = _redzone_identity_shift(pbp)
        row = out[(out["team"] == "ARI") & (out["season"] == 2025)].iloc[0]
        assert row["shift"] == pytest_approx(0.0)


class TestByeWeeks:
    def test_derives_the_one_week_a_team_has_no_game(self, monkeypatch):
        games = [
            _game(2025, 1, "BUF", "MIA"), _game(2025, 2, "NYJ", "BUF"),
            _game(2025, 3, "BUF", "NE"), _game(2025, 4, "MIA", "BUF"),
            _game(2025, 5, "NYJ", "NE"),   # BUF sits this one out
            _game(2025, 6, "BUF", "MIA"), _game(2025, 7, "NE", "BUF"),
        ]
        monkeypatch.setattr(sources, "schedules", lambda: pd.DataFrame(games))
        assert features.bye_weeks(2025)["BUF"] == 5

    def test_unpublished_schedule_returns_empty(self, monkeypatch):
        monkeypatch.setattr(sources, "schedules", lambda: pd.DataFrame(
            columns=["season", "week", "home_team", "away_team", "game_type"]))
        assert features.bye_weeks(2030) == {}

    def test_postseason_games_are_excluded(self, monkeypatch):
        games = [_game(2025, 1, "BUF", "MIA"), _game(2025, 2, "MIA", "BUF"),
                _game(2025, 19, "BUF", "KC", game_type="POST")]
        monkeypatch.setattr(sources, "schedules", lambda: pd.DataFrame(games))
        # Only two REG weeks exist and BUF played both -- no bye to report.
        assert "BUF" not in features.bye_weeks(2025)


def _weekly_row(season, team, position, name, week, target_share):
    return {"season": season, "recent_team": team, "position": position,
           "player_display_name": name, "week": week, "target_share": target_share}


class TestVacatedRoleEvents:
    def test_backup_gets_credited_for_a_real_absence(self):
        rows = []
        for wk in (1, 2, 3, 4, 6):
            rows.append(_weekly_row(2099, "CIN", "WR", "Star", wk, 0.40))
        # Backup's normal share is 0.10; week 5 (Star's only absence) it spikes to 0.30.
        for wk, share in zip((1, 2, 3, 4, 5, 6), (0.10, 0.10, 0.10, 0.10, 0.30, 0.10)):
            rows.append(_weekly_row(2099, "CIN", "WR", "Backup", wk, share))
        weekly = pd.DataFrame(rows)

        out = features.vacated_role_events(weekly, byes={}, positions=("WR",), min_games=3)
        assert len(out) == 1
        row = out.iloc[0]
        assert row["player_out"] == "Star" and row["teammate"] == "Backup"
        assert row["week"] == 5
        assert abs(row["trailing_share"] - 0.10) < 1e-9
        assert abs(row["baseline_share"] - 0.10) < 1e-9
        assert abs(row["actual_share"] - 0.30) < 1e-9
        assert abs(row["share_lift"] - 0.20) < 1e-9

    def test_bye_week_is_not_treated_as_an_absence(self):
        rows = []
        for wk in (1, 2, 3, 4):  # both sit out week 5 -- team's bye, not an injury
            rows.append(_weekly_row(2099, "CIN", "WR", "Star", wk, 0.40))
            rows.append(_weekly_row(2099, "CIN", "WR", "Backup", wk, 0.10))
        weekly = pd.DataFrame(rows)

        out = features.vacated_role_events(weekly, byes={2099: {"CIN": 5}},
                                           positions=("WR",), min_games=3)
        assert out.empty

    def test_cameo_player_does_not_count_as_a_teammate(self):
        rows = []
        for wk in (1, 2, 3, 4, 5):
            rows.append(_weekly_row(2099, "CIN", "WR", "Star", wk, 0.40))
        rows.append(_weekly_row(2099, "CIN", "WR", "OneGameGuy", 5, 0.30))
        weekly = pd.DataFrame(rows)

        # OneGameGuy played only week 5 -- below min_games -- so Star's other missed
        # weeks (there are none here, but even a real absence) can't credit him.
        out = features.vacated_role_events(weekly, byes={}, positions=("WR",), min_games=3)
        assert out.empty


def _week1_game(season, home, away, home_coach, away_coach):
    return {"season": season, "week": 1, "game_type": "REG",
           "home_team": home, "away_team": away,
           "home_coach": home_coach, "away_coach": away_coach}


class TestHeadCoaches:
    def test_reads_week1_coach_per_team(self, monkeypatch):
        games = pd.DataFrame([
            _week1_game(2025, "BUF", "MIA", "Sean McDermott", "Mike McDaniel"),
            _week1_game(2025, "SF", "NYJ", "Kyle Shanahan", "Robert Saleh"),
        ])
        monkeypatch.setattr(sources, "schedules", lambda: games)
        out = features.head_coaches(2025)
        assert out == {"BUF": "Sean McDermott", "MIA": "Mike McDaniel",
                       "SF": "Kyle Shanahan", "NYJ": "Robert Saleh"}

    def test_ignores_later_weeks(self, monkeypatch):
        games = pd.DataFrame([
            _week1_game(2025, "BUF", "MIA", "Sean McDermott", "Mike McDaniel"),
            {**_week1_game(2025, "BUF", "NYJ", "Interim Coach", "Robert Saleh"), "week": 8},
        ])
        monkeypatch.setattr(sources, "schedules", lambda: games)
        # A midseason firing (week 8) shouldn't override the Week 1 coach a
        # drafter actually knew about before the season started.
        assert features.head_coaches(2025)["BUF"] == "Sean McDermott"

    def test_unpublished_schedule_returns_empty(self, monkeypatch):
        monkeypatch.setattr(sources, "schedules", lambda: pd.DataFrame(
            columns=["season", "week", "game_type", "home_team", "away_team",
                    "home_coach", "away_coach"]))
        assert features.head_coaches(2099) == {}


class TestHeadCoachChanges:
    def test_flags_a_real_coaching_change(self, monkeypatch):
        games = pd.DataFrame([
            _week1_game(2024, "NYJ", "BUF", "Robert Saleh", "Sean McDermott"),
            _week1_game(2025, "NYJ", "MIA", "Aaron Glenn", "Mike McDaniel"),
        ])
        monkeypatch.setattr(sources, "schedules", lambda: games)
        out = features.head_coach_changes(2025)
        assert out["NYJ"] is True

    def test_continuity_reads_as_no_change(self, monkeypatch):
        games = pd.DataFrame([
            _week1_game(2024, "BUF", "NYJ", "Sean McDermott", "Robert Saleh"),
            _week1_game(2025, "BUF", "MIA", "Sean McDermott", "Mike McDaniel"),
        ])
        monkeypatch.setattr(sources, "schedules", lambda: games)
        out = features.head_coach_changes(2025)
        assert out["BUF"] is False

    def test_no_prior_season_data_reads_as_no_change(self, monkeypatch):
        games = pd.DataFrame([_week1_game(2025, "BUF", "MIA", "Sean McDermott", "Mike McDaniel")])

        def fake_schedules():
            return games

        monkeypatch.setattr(sources, "schedules", fake_schedules)
        out = features.head_coach_changes(2025)
        assert out["BUF"] is False


def pytest_approx(x):
    import pytest
    return pytest.approx(x, abs=0.5)
