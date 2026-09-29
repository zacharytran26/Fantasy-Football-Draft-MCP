"""oc_scheme_transfer_backtest: does an incoming OC's own pass-rate tendency at
his prior team predict his new team's actual pass rate better than continuity?"""
import pandas as pd

from ffdraft import adp, sources


def _oc_row(team, season, oc):
    return {"team": team, "season": season, "offensive_coordinator_raw": oc,
           "offensive_coordinator": oc, "oc_changed": ""}


def _play(season, team, down, wp, is_pass):
    return {"season": season, "posteam": team, "play_type": "pass" if is_pass else "run",
           "pass": 1 if is_pass else 0, "down": down, "wp": wp}


class TestLateralOCMoves:
    def test_finds_a_real_lateral_move_and_excludes_a_promotion(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "oc_history.csv"
        pd.DataFrame([
            _oc_row("AAA", 2023, "Coach X"),
            _oc_row("BBB", 2024, "Coach X"),   # lateral move: AAA -> BBB
            _oc_row("CCC", 2023, "Coach Y"),
            _oc_row("CCC", 2024, "Coach Y"),   # stayed put -- not a move
        ]).to_csv(csv_path, index=False)
        monkeypatch.setattr(adp, "_OC_HISTORY_CSV_PATH", csv_path)

        moves = adp._lateral_oc_moves()
        assert ("Coach X", 2023, "AAA", 2024, "BBB") in moves
        assert not any(m[0] == "Coach Y" for m in moves)

    def test_missing_file_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(adp, "_OC_HISTORY_CSV_PATH", tmp_path / "nope.csv")
        assert adp._lateral_oc_moves() == []


class TestOCSchemeTransferBacktest:
    def test_computes_pass_rates_for_a_real_move(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "oc_history.csv"
        pd.DataFrame([
            _oc_row("AAA", 2023, "Coach X"),
            _oc_row("BBB", 2024, "Coach X"),
        ]).to_csv(csv_path, index=False)
        monkeypatch.setattr(adp, "_OC_HISTORY_CSV_PATH", csv_path)

        pbp = pd.DataFrame([
            # AAA, 2023 (coach's prior team): all pass -> prior_source = 1.0
            _play(2023, "AAA", 1, 0.5, True), _play(2023, "AAA", 2, 0.5, True),
            # BBB, 2023 (new team, before he arrives): all run -> new_before = 0.0
            _play(2023, "BBB", 1, 0.5, False), _play(2023, "BBB", 2, 0.5, False),
            # BBB, 2024 (new team, after he arrives): all pass -> new_after = 1.0
            _play(2024, "BBB", 1, 0.5, True), _play(2024, "BBB", 2, 0.5, True),
        ])
        monkeypatch.setattr(sources, "play_by_play", lambda seasons: pbp)

        hist = adp.oc_scheme_transfer_backtest([2024])
        assert len(hist) == 1
        row = hist.iloc[0]
        assert row["prior_source"] == 1.0
        assert row["new_before"] == 0.0
        assert row["new_after"] == 1.0
        assert row["err_coach"] == 0.0       # coach's own tendency predicted perfectly here
        assert row["err_continuity"] == 1.0  # continuity predicted the exact opposite

        summary = adp.oc_scheme_transfer_backtest_summary(hist)
        assert summary["n_moves"] == 1
        assert summary["coach_win_rate"] == 1.0

    def test_no_moves_in_range_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(adp, "_OC_HISTORY_CSV_PATH", tmp_path / "nope.csv")
        hist = adp.oc_scheme_transfer_backtest([2024])
        assert hist.empty

    def test_empty_history_returns_empty_summary(self):
        assert adp.oc_scheme_transfer_backtest_summary(pd.DataFrame()) == {"n_moves": 0}
