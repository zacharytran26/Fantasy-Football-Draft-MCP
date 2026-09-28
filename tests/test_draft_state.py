"""DraftState: roster-player lookups and multi-pick lookahead, on an isolated state
directory so tests never touch the real ~/.ffdraft/state files."""
import pandas as pd
import pytest

from ffdraft import board
from ffdraft.config import LeagueSettings


@pytest.fixture(autouse=True)
def _isolated_state_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(board, "STATE_DIR", tmp_path)


def _board():
    return pd.DataFrame([
        {"name": "Alpha", "position": "RB", "_key": "alpha"},
        {"name": "Bravo", "position": "WR", "_key": "bravo"},
        {"name": "Charlie", "position": "RB", "_key": "charlie"},
    ])


class TestMyRosterPlayers:
    def test_returns_full_rows_for_my_picks_only(self):
        league = LeagueSettings(teams=4, draft_slot=1, rounds=3)
        state = board.DraftState(league, name="my_roster_players")
        state.reset()
        state.record("Alpha", overall=1, team_slot=1)
        state.record("Bravo", overall=2, team_slot=2)  # someone else's pick
        state.record("Charlie", overall=8, team_slot=1)

        mine = state.my_roster_players(_board())
        assert set(mine["name"]) == {"Alpha", "Charlie"}

    def test_empty_when_no_picks_made(self):
        league = LeagueSettings(teams=4, draft_slot=1, rounds=3)
        state = board.DraftState(league, name="my_roster_players_empty")
        state.reset()
        assert state.my_roster_players(_board()).empty


class TestUpcomingPicks:
    def test_lists_my_next_n_picks_in_order(self):
        # Slot 2 in a 4-team snake draft: picks 2, 7, 10, 15.
        league = LeagueSettings(teams=4, draft_slot=2, rounds=4, snake=True)
        state = board.DraftState(league, name="upcoming_picks")
        state.reset()
        assert state.upcoming_picks(after=1, n=3) == [2, 7, 10]

    def test_after_filters_out_earlier_picks(self):
        league = LeagueSettings(teams=4, draft_slot=2, rounds=4, snake=True)
        state = board.DraftState(league, name="upcoming_picks_after")
        state.reset()
        assert state.upcoming_picks(after=8, n=2) == [10, 15]
