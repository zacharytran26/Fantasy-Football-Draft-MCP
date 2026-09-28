"""depth_charts: skill-position filtering, depth_rank extraction, and picking the
most recent report regardless of which scrape-recency column the feed happens to
use this season."""
import pandas as pd

from ffdraft import sources


class TestDepthCharts:
    def test_filters_to_skill_positions_and_keeps_the_latest_dt(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sources, "CACHE_DIR", tmp_path)
        raw = pd.DataFrame([
            # A WR who's also the kick returner on an earlier snapshot: without the
            # pos_abb filter, the KR row could win the dedup and clobber his real
            # WR depth_rank.
            {"gsis_id": "p1", "team": "SF", "pos_abb": "WR", "pos_rank": 2, "dt": "2025-08-01"},
            {"gsis_id": "p1", "team": "SF", "pos_abb": "KR", "pos_rank": 1, "dt": "2025-08-01"},
            # A later snapshot promotes him to WR1.
            {"gsis_id": "p1", "team": "SF", "pos_abb": "WR", "pos_rank": 1, "dt": "2025-08-15"},
        ])
        monkeypatch.setattr(sources.pd, "read_parquet", lambda url: raw)

        out = sources.depth_charts(2099)
        assert list(out.columns) == ["player_id", "team", "depth_rank"]
        row = out[out["player_id"] == "p1"].iloc[0]
        assert row["depth_rank"] == 1  # the latest (Aug 15) WR report, not the KR row

    def test_missing_pos_rank_column_gives_nan_depth_rank(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sources, "CACHE_DIR", tmp_path)
        raw = pd.DataFrame([{"gsis_id": "p1", "team": "SF", "pos_abb": "WR", "week": 1}])
        monkeypatch.setattr(sources.pd, "read_parquet", lambda url: raw)

        out = sources.depth_charts(2098)
        assert pd.isna(out.loc[0, "depth_rank"])

    def test_unpublished_chart_returns_empty_with_expected_columns(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sources, "CACHE_DIR", tmp_path)

        def boom(url):
            raise FileNotFoundError("no chart yet")

        monkeypatch.setattr(sources.pd, "read_parquet", boom)

        out = sources.depth_charts(2097)
        assert list(out.columns) == ["player_id", "team", "depth_rank"]
        assert out.empty
