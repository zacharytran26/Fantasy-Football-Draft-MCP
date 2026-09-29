"""hc_change_role_volatility: does a head-coach change predict a bigger
year-over-year shift in a player's role than coaching continuity?"""
import pandas as pd

from ffdraft import adp, features, sources


def _weekly_row(season, team, name, week, target_share, position="WR"):
    return {"season": season, "recent_team": team, "position": position,
           "player_id": name, "player_display_name": name, "week": week,
           "target_share": target_share, "season_type": "REG"}


def _qb_row(season, team, name, week, attempts, sacks_suffered, passing_epa):
    return {"season": season, "recent_team": team, "position": "QB",
           "player_id": name, "player_display_name": name, "week": week,
           "attempts": attempts, "sacks_suffered": sacks_suffered,
           "passing_epa": passing_epa, "season_type": "REG"}


class TestHCChangeRoleVolatility:
    def test_hc_change_team_shows_a_bigger_role_shift(self, monkeypatch):
        # NE has a coaching change entering 2025; MIA has continuity. Each has one
        # player whose target_share moves a lot on the changed team, a little on
        # the stable one.
        prior_rows = []
        cur_rows = []
        for wk in range(1, 7):
            prior_rows.append(_weekly_row(2024, "NE", "NE_Player", wk, 0.20))
            cur_rows.append(_weekly_row(2025, "NE", "NE_Player", wk, 0.45))
            prior_rows.append(_weekly_row(2024, "MIA", "MIA_Player", wk, 0.20))
            cur_rows.append(_weekly_row(2025, "MIA", "MIA_Player", wk, 0.22))

        def fake_weekly_stats(seasons):
            return pd.DataFrame(prior_rows if seasons == [2024] else cur_rows)

        monkeypatch.setattr(sources, "weekly_stats", fake_weekly_stats)
        monkeypatch.setattr(features, "head_coach_changes",
                           lambda season: {"NE": True, "MIA": False})

        hist = adp.hc_change_role_volatility([2025], positions=("WR",), min_games=3)
        assert not hist.empty
        assert set(hist["team"]) == {"NE", "MIA"}

        ne_row = hist[hist["team"] == "NE"].iloc[0]
        mia_row = hist[hist["team"] == "MIA"].iloc[0]
        assert bool(ne_row["hc_changed"])
        assert not bool(mia_row["hc_changed"])
        assert ne_row["role_shift"] > mia_row["role_shift"]

        summary = adp.hc_change_role_volatility_summary(hist)
        assert summary["n_hc_changed"] == 1
        assert summary["n_hc_same"] == 1
        assert summary["difference_in_means"] > 0

    def test_player_who_changed_teams_is_excluded(self, monkeypatch):
        # Same player_id, different team in each season -- shouldn't merge, since
        # this backtest is specifically about coaching continuity, not team changes.
        prior_rows = [_weekly_row(2024, "OldTeam", "Mover", wk, 0.20) for wk in range(1, 7)]
        cur_rows = [_weekly_row(2025, "NewTeam", "Mover", wk, 0.40) for wk in range(1, 7)]

        def fake_weekly_stats(seasons):
            return pd.DataFrame(prior_rows if seasons == [2024] else cur_rows)

        monkeypatch.setattr(sources, "weekly_stats", fake_weekly_stats)
        monkeypatch.setattr(features, "head_coach_changes",
                           lambda season: {"NewTeam": True, "OldTeam": False})

        hist = adp.hc_change_role_volatility([2025], positions=("WR",), min_games=3)
        assert hist.empty

    def test_no_head_coach_data_is_skipped(self, monkeypatch):
        monkeypatch.setattr(features, "head_coach_changes", lambda season: {})
        hist = adp.hc_change_role_volatility([2099])
        assert hist.empty

    def test_empty_history_returns_empty_summary(self):
        assert adp.hc_change_role_volatility_summary(pd.DataFrame()) == {"n_players": 0}


class TestHCChangeShrinkageBacktest:
    def test_shrinkage_helps_when_role_reverts_toward_baseline(self, monkeypatch):
        prior_rows, cur_rows = [], []
        # Two players on coaching-change teams: one high prior share, one low --
        # both actually land near the position baseline (0.20) this season.
        prior_rows += [_weekly_row(2024, "NE", "HighShare", wk, 0.35) for wk in range(1, 7)]
        cur_rows += [_weekly_row(2025, "NE", "HighShare", wk, 0.22) for wk in range(1, 7)]
        prior_rows += [_weekly_row(2024, "CHI", "LowShare", wk, 0.05) for wk in range(1, 7)]
        cur_rows += [_weekly_row(2025, "CHI", "LowShare", wk, 0.18) for wk in range(1, 7)]
        # Two players on stable teams, included only to shape the position baseline.
        prior_rows += [_weekly_row(2024, "MIA", "Stable1", wk, 0.20) for wk in range(1, 7)]
        cur_rows += [_weekly_row(2025, "MIA", "Stable1", wk, 0.20) for wk in range(1, 7)]
        prior_rows += [_weekly_row(2024, "BUF", "Stable2", wk, 0.20) for wk in range(1, 7)]
        cur_rows += [_weekly_row(2025, "BUF", "Stable2", wk, 0.20) for wk in range(1, 7)]

        def fake_weekly_stats(seasons):
            return pd.DataFrame(prior_rows if seasons == [2024] else cur_rows)

        monkeypatch.setattr(sources, "weekly_stats", fake_weekly_stats)
        monkeypatch.setattr(features, "head_coach_changes",
                           lambda season: {"NE": True, "CHI": True, "MIA": False, "BUF": False})

        hist = adp.hc_change_shrinkage_backtest([2025], positions=("WR",), min_games=3,
                                                shrinkage_levels=(0.0, 0.5, 1.0))
        assert len(hist) == 2  # only the two coaching-change players

        summary = adp.hc_change_shrinkage_summary(hist, shrinkage_levels=(0.0, 0.5, 1.0))
        assert summary["n_players"] == 2
        assert summary["best_shrinkage"] == 1.0
        assert summary["improvement_vs_no_shrinkage"] > 0

    def test_no_shrinkage_wins_when_own_history_is_still_the_best_predictor(self, monkeypatch):
        # Coaching-change players whose role stays exactly what their own prior
        # season said -- shrinking toward the (different) baseline only hurts.
        prior_rows = [_weekly_row(2024, "NE", "Steady", wk, 0.35) for wk in range(1, 7)]
        cur_rows = [_weekly_row(2025, "NE", "Steady", wk, 0.35) for wk in range(1, 7)]
        prior_rows += [_weekly_row(2024, "MIA", "Other", wk, 0.10) for wk in range(1, 7)]
        cur_rows += [_weekly_row(2025, "MIA", "Other", wk, 0.10) for wk in range(1, 7)]

        def fake_weekly_stats(seasons):
            return pd.DataFrame(prior_rows if seasons == [2024] else cur_rows)

        monkeypatch.setattr(sources, "weekly_stats", fake_weekly_stats)
        monkeypatch.setattr(features, "head_coach_changes",
                           lambda season: {"NE": True, "MIA": False})

        hist = adp.hc_change_shrinkage_backtest([2025], positions=("WR",), min_games=3,
                                                shrinkage_levels=(0.0, 0.5, 1.0))
        summary = adp.hc_change_shrinkage_summary(hist, shrinkage_levels=(0.0, 0.5, 1.0))
        assert summary["best_shrinkage"] == 0.0
        assert summary["improvement_vs_no_shrinkage"] == 0.0

    def test_no_head_coach_data_is_skipped(self, monkeypatch):
        monkeypatch.setattr(features, "head_coach_changes", lambda season: {})
        hist = adp.hc_change_shrinkage_backtest([2099])
        assert hist.empty

    def test_empty_history_returns_empty_summary(self):
        assert adp.hc_change_shrinkage_summary(pd.DataFrame()) == {"n_players": 0}


class TestHCChangeQBEfficiencyVolatility:
    def test_hc_change_qb_shows_a_bigger_efficiency_shift(self, monkeypatch):
        # NE has a coaching change entering 2025; MIA has continuity. Each QB's
        # EPA/dropback moves a lot on the changed team, a little on the stable one.
        prior_rows, cur_rows = [], []
        for wk in range(1, 7):
            prior_rows.append(_qb_row(2024, "NE", "NE_QB", wk, 30, 2, 2.0))
            cur_rows.append(_qb_row(2025, "NE", "NE_QB", wk, 30, 2, 14.0))
            prior_rows.append(_qb_row(2024, "MIA", "MIA_QB", wk, 30, 2, 2.0))
            cur_rows.append(_qb_row(2025, "MIA", "MIA_QB", wk, 30, 2, 2.5))

        def fake_weekly_stats(seasons):
            return pd.DataFrame(prior_rows if seasons == [2024] else cur_rows)

        monkeypatch.setattr(sources, "weekly_stats", fake_weekly_stats)
        monkeypatch.setattr(features, "head_coach_changes",
                           lambda season: {"NE": True, "MIA": False})

        hist = adp.hc_change_qb_efficiency_volatility([2025], min_dropbacks=100)
        assert not hist.empty
        assert set(hist["team"]) == {"NE", "MIA"}

        ne_row = hist[hist["team"] == "NE"].iloc[0]
        mia_row = hist[hist["team"] == "MIA"].iloc[0]
        assert bool(ne_row["hc_changed"])
        assert not bool(mia_row["hc_changed"])
        assert ne_row["role_shift"] > mia_row["role_shift"]

        summary = adp.hc_change_qb_efficiency_volatility_summary(hist)
        assert summary["n_hc_changed"] == 1
        assert summary["n_hc_same"] == 1
        assert summary["difference_in_means"] > 0

    def test_low_dropback_qb_is_excluded(self, monkeypatch):
        rows = [_qb_row(2024, "NE", "Backup", 1, 10, 1, 1.0)]
        cur_rows = [_qb_row(2025, "NE", "Backup", 1, 10, 1, 1.0)]

        def fake_weekly_stats(seasons):
            return pd.DataFrame(rows if seasons == [2024] else cur_rows)

        monkeypatch.setattr(sources, "weekly_stats", fake_weekly_stats)
        monkeypatch.setattr(features, "head_coach_changes", lambda season: {"NE": True})

        hist = adp.hc_change_qb_efficiency_volatility([2025], min_dropbacks=100)
        assert hist.empty

    def test_no_head_coach_data_is_skipped(self, monkeypatch):
        monkeypatch.setattr(features, "head_coach_changes", lambda season: {})
        hist = adp.hc_change_qb_efficiency_volatility([2099])
        assert hist.empty

    def test_empty_history_returns_empty_summary(self):
        assert adp.hc_change_qb_efficiency_volatility_summary(pd.DataFrame()) == {"n_players": 0}


class TestHCChangeQBEfficiencyShrinkageBacktest:
    def test_both_cohorts_are_kept_and_scored_separately(self, monkeypatch):
        prior_rows, cur_rows = [], []
        # Coaching-change QB: high prior efficiency, reverts hard toward baseline.
        for wk in range(1, 7):
            prior_rows.append(_qb_row(2024, "NE", "ChangedQB", wk, 30, 2, 6.0))
            cur_rows.append(_qb_row(2025, "NE", "ChangedQB", wk, 30, 2, 0.0))
        # Continuity QB, included to shape the position baseline and give a
        # non-empty "hc_same" cohort.
        for wk in range(1, 7):
            prior_rows.append(_qb_row(2024, "MIA", "StableQB", wk, 30, 2, 0.0))
            cur_rows.append(_qb_row(2025, "MIA", "StableQB", wk, 30, 2, 0.0))

        def fake_weekly_stats(seasons):
            return pd.DataFrame(prior_rows if seasons == [2024] else cur_rows)

        monkeypatch.setattr(sources, "weekly_stats", fake_weekly_stats)
        monkeypatch.setattr(features, "head_coach_changes",
                           lambda season: {"NE": True, "MIA": False})

        hist = adp.hc_change_qb_efficiency_shrinkage_backtest(
            [2025], min_dropbacks=100, shrinkage_levels=(0.0, 0.5, 1.0))
        assert len(hist) == 2  # both cohorts kept, unlike hc_change_shrinkage_backtest

        summary = adp.hc_change_qb_efficiency_shrinkage_summary(
            hist, shrinkage_levels=(0.0, 0.5, 1.0))
        assert summary["n_hc_changed"] == 1
        assert summary["n_hc_same"] == 1
        assert summary["improvement_vs_no_shrinkage_hc_changed"] >= 0

    def test_no_head_coach_data_is_skipped(self, monkeypatch):
        monkeypatch.setattr(features, "head_coach_changes", lambda season: {})
        hist = adp.hc_change_qb_efficiency_shrinkage_backtest([2099])
        assert hist.empty

    def test_empty_history_returns_empty_summary(self):
        assert adp.hc_change_qb_efficiency_shrinkage_summary(pd.DataFrame()) == {"n_players": 0}


class TestOCChangeQBEfficiencyVolatility:
    def test_oc_change_qb_shows_a_bigger_efficiency_shift(self, monkeypatch):
        prior_rows, cur_rows = [], []
        for wk in range(1, 7):
            prior_rows.append(_qb_row(2024, "NE", "NE_QB", wk, 30, 2, 2.0))
            cur_rows.append(_qb_row(2025, "NE", "NE_QB", wk, 30, 2, 14.0))
            prior_rows.append(_qb_row(2024, "MIA", "MIA_QB", wk, 30, 2, 2.0))
            cur_rows.append(_qb_row(2025, "MIA", "MIA_QB", wk, 30, 2, 2.5))

        def fake_weekly_stats(seasons):
            return pd.DataFrame(prior_rows if seasons == [2024] else cur_rows)

        monkeypatch.setattr(sources, "weekly_stats", fake_weekly_stats)
        monkeypatch.setattr(features, "offensive_coordinator_changes",
                           lambda season: {"NE": True, "MIA": False})

        hist = adp.oc_change_qb_efficiency_volatility([2025], min_dropbacks=100)
        assert not hist.empty
        assert set(hist["team"]) == {"NE", "MIA"}

        ne_row = hist[hist["team"] == "NE"].iloc[0]
        mia_row = hist[hist["team"] == "MIA"].iloc[0]
        assert bool(ne_row["oc_changed"])
        assert not bool(mia_row["oc_changed"])
        assert ne_row["role_shift"] > mia_row["role_shift"]

        summary = adp.oc_change_qb_efficiency_volatility_summary(hist)
        assert summary["n_oc_changed"] == 1
        assert summary["n_oc_same"] == 1
        assert summary["difference_in_means"] > 0

    def test_no_oc_change_data_is_skipped(self, monkeypatch):
        monkeypatch.setattr(features, "offensive_coordinator_changes", lambda season: {})
        hist = adp.oc_change_qb_efficiency_volatility([2099])
        assert hist.empty

    def test_empty_history_returns_empty_summary(self):
        assert adp.oc_change_qb_efficiency_volatility_summary(pd.DataFrame()) == {"n_players": 0}


class TestOCChangeQBEfficiencyShrinkageBacktest:
    def test_both_cohorts_are_kept_and_scored_separately(self, monkeypatch):
        prior_rows, cur_rows = [], []
        for wk in range(1, 7):
            prior_rows.append(_qb_row(2024, "NE", "ChangedQB", wk, 30, 2, 6.0))
            cur_rows.append(_qb_row(2025, "NE", "ChangedQB", wk, 30, 2, 0.0))
        for wk in range(1, 7):
            prior_rows.append(_qb_row(2024, "MIA", "StableQB", wk, 30, 2, 0.0))
            cur_rows.append(_qb_row(2025, "MIA", "StableQB", wk, 30, 2, 0.0))

        def fake_weekly_stats(seasons):
            return pd.DataFrame(prior_rows if seasons == [2024] else cur_rows)

        monkeypatch.setattr(sources, "weekly_stats", fake_weekly_stats)
        monkeypatch.setattr(features, "offensive_coordinator_changes",
                           lambda season: {"NE": True, "MIA": False})

        hist = adp.oc_change_qb_efficiency_shrinkage_backtest(
            [2025], min_dropbacks=100, shrinkage_levels=(0.0, 0.5, 1.0))
        assert len(hist) == 2

        summary = adp.oc_change_qb_efficiency_shrinkage_summary(
            hist, shrinkage_levels=(0.0, 0.5, 1.0))
        assert summary["n_oc_changed"] == 1
        assert summary["n_oc_same"] == 1
        assert summary["improvement_vs_no_shrinkage_oc_changed"] >= 0

    def test_no_oc_change_data_is_skipped(self, monkeypatch):
        monkeypatch.setattr(features, "offensive_coordinator_changes", lambda season: {})
        hist = adp.oc_change_qb_efficiency_shrinkage_backtest([2099])
        assert hist.empty

    def test_empty_history_returns_empty_summary(self):
        assert adp.oc_change_qb_efficiency_shrinkage_summary(pd.DataFrame()) == {"n_players": 0}
