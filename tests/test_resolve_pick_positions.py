"""resolve_pick_positions: mapping sync_espn/sync_sleeper's raw name-only picks to
real positions from that season's box scores, dropping D/ST and kickers."""
import pandas as pd

from ffdraft import board, sources


def _weekly_row(name, position):
    return {"player_display_name": name, "position": position, "season_type": "REG"}


class TestResolvePickPositions:
    def test_resolves_names_to_positions_in_pick_order(self, monkeypatch):
        weekly = pd.DataFrame([
            _weekly_row("Christian McCaffrey", "RB"),
            _weekly_row("Justin Jefferson", "WR"),
            _weekly_row("Travis Kelce", "TE"),
        ])
        monkeypatch.setattr(sources, "weekly_stats", lambda seasons: weekly)

        picks = [
            {"overall": 2, "name": "Justin Jefferson"},
            {"overall": 1, "name": "Christian McCaffrey"},
            {"overall": 3, "name": "Travis Kelce"},
        ]
        out = board.resolve_pick_positions(picks, 2024)
        assert out == ["RB", "WR", "TE"]  # sorted by overall, not input order

    def test_drops_dst_picks_without_resolving_them(self, monkeypatch):
        weekly = pd.DataFrame([_weekly_row("Christian McCaffrey", "RB")])
        monkeypatch.setattr(sources, "weekly_stats", lambda seasons: weekly)

        picks = [
            {"overall": 1, "name": "Christian McCaffrey"},
            {"overall": 2, "name": "49ers D/ST"},
        ]
        out = board.resolve_pick_positions(picks, 2024)
        assert out == ["RB"]

    def test_unresolvable_name_is_dropped_not_guessed(self, monkeypatch):
        weekly = pd.DataFrame([_weekly_row("Christian McCaffrey", "RB")])
        monkeypatch.setattr(sources, "weekly_stats", lambda seasons: weekly)

        picks = [
            {"overall": 1, "name": "Christian McCaffrey"},
            {"overall": 2, "name": "Totally Unknown Player"},
        ]
        out = board.resolve_pick_positions(picks, 2024)
        assert out == ["RB"]

    def test_no_weekly_data_returns_empty(self, monkeypatch):
        monkeypatch.setattr(sources, "weekly_stats", lambda seasons: pd.DataFrame(
            columns=["player_display_name", "position", "season_type"]))
        out = board.resolve_pick_positions([{"overall": 1, "name": "Anyone"}], 2024)
        assert out == []
