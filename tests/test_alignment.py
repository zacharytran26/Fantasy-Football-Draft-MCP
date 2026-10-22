"""Rookie snap-alignment layer: optional local file, honest about thin samples."""
import json

import pytest

from ffdraft import alignment as al


def _cell(routes, yprr, tprr=20.0):
    return {"routes": routes, "targets": 10, "tprr": tprr, "catch": 60.0, "adot": 9.0, "ypt": 8.0,
            "yac": 4.0, "yprr": yprr, "epa_t": 0.2}


@pytest.fixture
def data_file(tmp_path, monkeypatch):
    f = tmp_path / "alignment_players.json"
    f.write_text(json.dumps({
        "p1": {"name": "Slot Guy", "draft_year": 2023, "position": "WR", "team": "AAA", "total_routes": 700,
               "primary_role": "Slot", "roles": {
                   "Slot": {"ALL": _cell(600, 1.8), "MAN": _cell(150, 2.5), "ZONE": _cell(450, 1.6), "route_share_pct": 86},
                   "Z": {"ALL": _cell(30, 1.0), "route_share_pct": 4}}},
        "p2": {"name": "Other Guy", "draft_year": 2024, "position": "WR", "team": "BBB", "total_routes": 400,
               "primary_role": "Slot", "roles": {"Slot": {"ALL": _cell(400, 1.0), "MAN": _cell(80, 3.0), "route_share_pct": 100}}},
    }))
    monkeypatch.setattr(al, "FILE", f)
    monkeypatch.setitem(al._STATE, "mtime", None)
    return f


class TestAlignment:
    def test_missing_file_is_empty_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(al, "FILE", tmp_path / "nope.json")
        monkeypatch.setitem(al._STATE, "mtime", None)
        assert not al.available() and al.profile("p1") is None and al.leaders("Slot") == []

    def test_profile_hides_roles_under_the_route_floor(self, data_file):
        p = al.profile("p1")
        assert set(p["roles"]) == {"Slot"} and p["roles"]["Slot"]["vs_man"]["yprr"] == 2.5

    def test_leaders_respect_min_routes_and_sort(self, data_file):
        names = [r["name"] for r in al.leaders("Slot", "MAN", "yards_per_route", min_routes=100)]
        assert names == ["Slot Guy"]                      # p2 has only 80 man routes
        names = [r["name"] for r in al.leaders("Slot", "MAN", "yards_per_route", min_routes=50)]
        assert names == ["Other Guy", "Slot Guy"]

    def test_unknown_player_is_none(self, data_file):
        assert al.profile("zzz") is None
