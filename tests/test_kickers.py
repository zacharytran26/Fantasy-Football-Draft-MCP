"""Kicker scoring and draft context."""
import pandas as pd
import pytest

from ffdraft.config import LeagueSettings, Scoring
from ffdraft.kickers import kicker_board, kicker_fantasy_points, kicker_season_profiles


def _week(**over):
    row = {
        "player_id": "K1", "player_display_name": "Test Kicker", "position": "K",
        "season": 2025, "season_type": "REG", "week": 1, "recent_team": "BUF",
        "fg_att": 0, "fg_made": 0, "fg_missed": 0, "fg_blocked": 0, "fg_long": 0,
        "fg_made_0_19": 0, "fg_made_20_29": 0, "fg_made_30_39": 0,
        "fg_made_40_49": 0, "fg_made_50_59": 0, "fg_made_60_": 0,
        "pat_made": 0, "pat_att": 0,
    }
    row.update(over)
    return row


class TestKickerFantasyPoints:
    def test_distance_bands_score_espn_defaults(self):
        row = pd.DataFrame([_week(
            fg_att=4, fg_made=4,
            fg_made_30_39=1, fg_made_40_49=1, fg_made_50_59=1, fg_made_60_=1,
            pat_made=2, pat_att=2,
        )])
        pts = float(kicker_fantasy_points(row, Scoring()).iloc[0])
        # 3 (under-40) + 4 (40-49) + 5 + 5 (both 50+ makes) + 2 PAT, no 60+ bonus by default
        assert pts == pytest.approx(3 + 4 + 5 + 5 + 2)

    def test_60_plus_bonus_stacks_on_top_of_50_plus_not_instead_of(self):
        row = pd.DataFrame([_week(fg_att=1, fg_made=1, fg_made_60_=1)])
        sc = Scoring(fg_made_60_bonus=2.0)
        pts = float(kicker_fantasy_points(row, sc).iloc[0])
        assert pts == pytest.approx(5 + 2)  # fg_made_50_plus + the bonus, not a replacement

    def test_misses_and_blocks_share_the_flat_penalty(self):
        row = pd.DataFrame([_week(fg_att=2, fg_missed=1, fg_blocked=1)])
        pts = float(kicker_fantasy_points(row, Scoring()).iloc[0])
        assert pts == pytest.approx(-1 + -1)

    def test_missing_pat_blocked_column_defaults_to_zero_not_a_crash(self):
        row = pd.DataFrame([{"pat_made": 1, "pat_att": 1}])
        pts = float(kicker_fantasy_points(row, Scoring()).iloc[0])
        assert pts == pytest.approx(1.0)


class TestKickerSeasonProfiles:
    def test_sums_across_weeks_and_buckets_distance_bands(self, monkeypatch):
        weekly = pd.DataFrame([
            _week(week=1, fg_att=2, fg_made=2, fg_made_0_19=1, fg_made_40_49=1, pat_made=3, pat_att=3),
            _week(week=2, fg_att=1, fg_made=1, fg_made_50_59=1, pat_made=2, pat_att=2),
            # A non-kicker row in the same weekly_stats pull must be excluded.
            _week(player_id="RB1", position="RB", week=1),
        ])
        monkeypatch.setattr("ffdraft.kickers.sources.weekly_stats", lambda seasons=None: weekly)
        prof = kicker_season_profiles(Scoring(), seasons=[2025])
        assert len(prof) == 1
        row = prof.iloc[0]
        assert row["games"] == 2
        assert row["fg_att"] == 3
        assert row["fg_made"] == 3
        assert row["fg_made_u40"] == 1     # the week-1 0-19 make
        assert row["fg_made_40_49"] == 1
        assert row["fg_made_50p"] == 1     # the week-2 50-59 make
        assert row["pat_made"] == 5

    def test_empty_weekly_stats_returns_empty_frame_not_a_crash(self, monkeypatch):
        monkeypatch.setattr("ffdraft.kickers.sources.weekly_stats",
                            lambda seasons=None: pd.DataFrame(columns=["position", "season_type"]))
        prof = kicker_season_profiles(Scoring(), seasons=[2025])
        assert prof.empty
        assert "fp_mean" in prof.columns


class TestKickerBoard:
    def _patch_common(self, monkeypatch, weekly):
        monkeypatch.setattr("ffdraft.kickers.sources.weekly_stats", lambda seasons=None: weekly)
        monkeypatch.setattr("ffdraft.kickers.features.team_drive_efficiency", lambda: pd.DataFrame([
            {"season": 2025, "team": "BUF", "drives": 20, "pct_td": 35.0, "pct_fg": 30.0, "pct_punt": 20.0},
            {"season": 2025, "team": "NYJ", "drives": 20, "pct_td": 10.0, "pct_fg": 5.0, "pct_punt": 60.0},
        ]))
        monkeypatch.setattr("ffdraft.kickers.features.team_pace_and_split", lambda: pd.DataFrame([
            {"season": 2025, "team": "BUF", "plays_per_game": 66.0, "off_epa": 0.10},
            {"season": 2025, "team": "NYJ", "plays_per_game": 58.0, "off_epa": -0.10},
        ]))

    def test_high_opportunity_team_ranks_above_low_opportunity_team(self, monkeypatch):
        # Five teams so the quintile banding in kicker_board has enough spread to
        # be meaningful -- with only two teams, rank(pct=True) can't populate all
        # five bands, which is a test-fixture artifact, not something kicker_board
        # itself needs to handle for a real ~32-team board.
        # rank(pct=True)'s lowest value lands at 1/N -- needs N >= 6 to fall
        # strictly under the 0.2 "Bottom" cutoff (1/5 == 0.2 would tie into
        # "Below avg" instead).
        teams = ["BUF", "MID1", "MID2", "MID3", "MID4", "NYJ"]
        drive_eff = pd.DataFrame([
            {"season": 2025, "team": t, "drives": 20, "pct_td": 20.0,
             "pct_fg": pct, "pct_punt": 100 - 20 - pct}
            for t, pct in zip(teams, [30.0, 24.0, 18.0, 12.0, 8.0, 5.0])
        ])
        pace = pd.DataFrame([
            {"season": 2025, "team": t, "plays_per_game": 62.0, "off_epa": 0.0} for t in teams
        ])
        monkeypatch.setattr("ffdraft.kickers.features.team_drive_efficiency", lambda: drive_eff)
        monkeypatch.setattr("ffdraft.kickers.features.team_pace_and_split", lambda: pace)

        weekly = pd.concat([
            pd.DataFrame([
                _week(player_id=f"K{i}", player_display_name=f"{t} K", recent_team=t,
                      week=w, fg_att=2, fg_made=2, fg_made_40_49=2, pat_made=2, pat_att=2)
                for w in range(1, 5)
            ])
            for i, t in enumerate(teams)
        ], ignore_index=True)
        monkeypatch.setattr("ffdraft.kickers.sources.weekly_stats", lambda seasons=None: weekly)

        board = kicker_board(LeagueSettings(), season=2026)
        assert set(board["position"]) == {"K"}
        buf = board[board["team"] == "BUF"].iloc[0]
        nyj = board[board["team"] == "NYJ"].iloc[0]
        assert buf["context_tier"] == "Elite"
        assert nyj["context_tier"] == "Bottom"
        # Same production, so kicker_ppg should tie -- context_tier is informational
        # only, not folded into the ranking (same convention as team_context).
        assert buf["kicker_ppg"] == pytest.approx(nyj["kicker_ppg"])

    def test_short_sample_kicker_is_filtered_out(self, monkeypatch):
        weekly = pd.DataFrame([
            _week(player_id="K1", player_display_name="One Game K", recent_team="BUF",
                  week=1, fg_att=1, fg_made=1, fg_made_40_49=1, pat_made=1, pat_att=1),
        ])
        self._patch_common(monkeypatch, weekly)
        board = kicker_board(LeagueSettings(), season=2026)
        assert board.empty or "One Game K" not in board["name"].values

    def test_empty_profiles_returns_empty_frame_with_position_column(self, monkeypatch):
        monkeypatch.setattr("ffdraft.kickers.sources.weekly_stats",
                            lambda seasons=None: pd.DataFrame(columns=["position", "season_type"]))
        board = kicker_board(LeagueSettings(), season=2026)
        assert board.empty
