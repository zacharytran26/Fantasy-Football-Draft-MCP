"""weekly_stats: raises on any missing season in a multi-season pull rather than
silently caching a partial result -- the bug a transient 2022 nflverse fetch
failure surfaced (a 2020-2025 pull silently came back missing all of 2022, and
that incomplete result got cached for every future caller requesting that range)."""
import pandas as pd
import pytest

from ffdraft import sources


def _row(season):
    return {"season": season, "week": 1, "player_id": "p1", "player_display_name": "P1",
           "position": "WR", "recent_team": "SF", "season_type": "REG"}


class TestWeeklyStatsPartialFetch:
    def test_raises_when_one_season_of_several_fails_to_fetch(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sources, "CACHE_DIR", tmp_path)

        def fake_read_parquet(url):
            if "2022" in url:
                raise FileNotFoundError("transient fetch failure")
            season = int(url.rsplit("_", 1)[-1].split(".")[0])
            return pd.DataFrame([_row(season)])

        monkeypatch.setattr(sources.pd, "read_parquet", fake_read_parquet)

        with pytest.raises(RuntimeError, match="2022"):
            sources.weekly_stats([2021, 2022, 2023])

        # The failed build must not have cached a partial result.
        assert not (tmp_path / "weekly_stats_2021_2023.parquet").exists()

    def test_succeeds_when_every_season_fetches(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sources, "CACHE_DIR", tmp_path)

        def fake_read_parquet(url):
            season = int(url.rsplit("_", 1)[-1].split(".")[0])
            return pd.DataFrame([_row(season)])

        monkeypatch.setattr(sources.pd, "read_parquet", fake_read_parquet)

        out = sources.weekly_stats([2021, 2022, 2023])
        assert sorted(out["season"].unique()) == [2021, 2022, 2023]

    def test_single_unpublished_season_still_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sources, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(sources.pd, "read_parquet",
                           lambda url: (_ for _ in ()).throw(FileNotFoundError("no chart yet")))

        with pytest.raises(RuntimeError, match="2099"):
            sources.weekly_stats([2099])
