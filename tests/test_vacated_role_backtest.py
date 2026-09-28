"""vacated_role_backtest: does a teammate's trailing target_share predict who
benefits when a same-position starter misses a game? Data assembly (real weekly
stats, mocked) tested here; the events math itself is covered directly in
test_features.py's TestVacatedRoleEvents."""
import pandas as pd

from ffdraft import adp, features, sources


def _row(season, team, position, name, week, target_share, season_type="REG"):
    return {"season": season, "recent_team": team, "position": position,
           "player_display_name": name, "week": week, "target_share": target_share,
           "season_type": season_type}


class TestVacatedRoleBacktest:
    def test_predictable_backup_beats_the_random_baseline(self, monkeypatch):
        # CIN: Star (WR) sits week 5. Of two teammates, HighVolume already runs a
        # bigger normal target share (0.20) than LowVolume (0.05), and HighVolume is
        # the one who actually absorbs the vacated targets (0.20 -> 0.45). A
        # trailing-share predictor should call this correctly every time it's this
        # clean, comfortably beating a 1-in-2 random guess.
        rows = []
        for wk in (1, 2, 3, 4, 6, 7, 8):
            rows.append(_row(2099, "CIN", "WR", "Star", wk, 0.40))
            rows.append(_row(2099, "CIN", "WR", "HighVolume", wk, 0.20))
            rows.append(_row(2099, "CIN", "WR", "LowVolume", wk, 0.05))
        rows.append(_row(2099, "CIN", "WR", "HighVolume", 5, 0.45))
        rows.append(_row(2099, "CIN", "WR", "LowVolume", 5, 0.06))
        weekly = pd.DataFrame(rows)

        monkeypatch.setattr(sources, "weekly_stats", lambda seasons: weekly)
        monkeypatch.setattr(features, "bye_weeks", lambda season: {})

        hist = adp.vacated_role_backtest([2099], positions=("WR",), min_games=3)
        assert not hist.empty

        summary = adp.vacated_role_backtest_summary(hist)
        assert summary["n_absence_events"] == 1
        assert summary["top1_accuracy"] == 1.0
        assert summary["random_baseline_accuracy"] == 0.5
        assert summary["improvement_vs_random"] == 0.5
        assert summary["trailing_share_vs_lift_corr"] > 0

    def test_no_qualifying_absences_returns_empty(self, monkeypatch):
        # Everyone plays every week -- no real absence to learn from.
        rows = [_row(2099, "CIN", "WR", name, wk, 0.2)
               for name in ("A", "B") for wk in range(1, 6)]
        weekly = pd.DataFrame(rows)
        monkeypatch.setattr(sources, "weekly_stats", lambda seasons: weekly)
        monkeypatch.setattr(features, "bye_weeks", lambda season: {})

        hist = adp.vacated_role_backtest([2099], positions=("WR",), min_games=3)
        assert hist.empty
        assert adp.vacated_role_backtest_summary(hist) == {"n_events": 0}
