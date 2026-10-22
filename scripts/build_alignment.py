#!/usr/bin/env python3
"""Build rookie snap-alignment profiles from your local copy of the NFL Big Data Bowl 2027 files.

    python scripts/build_alignment.py /path/to/bdb/parquet

Expects parquet versions of player_play, players, games and game_tracking_{2023,2024,2025}
(the competition CSVs, converted with 'NA' as null). Writes $FFDRAFT_DATA/alignment_players.json
keyed by nflverse gsis_id, joined on birth date + surname (no fuzzy name matching), and prints
anything that failed to join. The output is derived from CC BY-NC data and stays on your machine.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ffdraft import sources
from ffdraft.alignment import FILE

OFF_LINE_YD = 2.0
MAN_TYPES = {"COVER_0_MAN", "COVER_1_MAN", "COVER_2_MAN"}
ZONE_TYPES = {"COVER_2_ZONE", "COVER_3_ZONE", "COVER_4_ZONE", "COVER_6_ZONE"}


def _surname(s: str) -> str:
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", str(s).lower().replace(".", ""))
    return re.sub(r"[^a-z]", "", s.split()[-1]) if s.split() else ""


def build(bdb: Path) -> dict:
    snaps = pd.concat([
        pd.read_parquet(bdb / f"game_tracking_{y}.parquet", columns=["game_id", "play_id", "nfl_id", "x", "event"],
                        filters=[("event", "==", "ball_snap")]).drop(columns="event")
        for y in (2023, 2024, 2025)], ignore_index=True).rename(columns={"x": "snap_x"})
    pp = pd.read_parquet(bdb / "player_play.parquet")
    games = pd.read_parquet(bdb / "games.parquet")[["game_id", "season", "season_type"]]
    players = pd.read_parquet(bdb / "players.parquet")

    pp = pp[(pp["team_abbr"] == pp["possession_team"]) & pp["route_ran"].notna()
            & (pp["play_nullified_by_penalty"] != "Y")]
    pp = (pp.merge(games, on="game_id").query("season_type == 'REG'")
          .merge(players[["nfl_id", "nfl_position"]], on="nfl_id").query("nfl_position in ['WR','TE']")
          .merge(snaps, on=["game_id", "play_id", "nfl_id"]))
    pp = pp[~pp["in_motion_at_ball_snap"].fillna(False).astype(bool)]

    # Distance behind the spotted line of scrimmage, in tracking x coordinates (endzones are 0-10 and 110-120).
    yfo = np.where(pp["yardline_side"] == pp["possession_team"], pp["yardline_number"], 100 - pp["yardline_number"])
    los_x = np.where(pp["play_direction"] == "right", 10 + yfo, 110 - yfo)
    off_los = (pp["snap_x"] - los_x).abs()
    pp["role"] = np.select(
        [pp["nfl_position"] == "TE", pp["lined_up_position"] == "SLOT_WR",
         (pp["lined_up_position"] == "WR") & (off_los < OFF_LINE_YD), pp["lined_up_position"] == "WR"],
        ["Y", "Slot", "X", "Z"], default=None)
    pp = pp[pp["role"].notna()].copy()
    pp["mz"] = np.select([pp["team_coverage_type"].isin(MAN_TYPES), pp["team_coverage_type"].isin(ZONE_TYPES)],
                         ["MAN", "ZONE"], default=None)

    tgt = pp["target"].fillna(False).astype(bool)
    comp = tgt & (pp["pass_result"] == "C")
    pp["t"], pp["rec"] = tgt.astype(int), comp.astype(int)
    pp["yds"] = np.where(tgt, pp["rec_yards"].fillna(0), 0.0)
    pp["yac"] = np.where(comp, pp["yards_after_catch"].fillna(0), 0.0)
    pp["air"] = np.where(tgt, pp["pass_length"].fillna(0), 0.0)
    pp["epa"] = np.where(tgt, pp["expected_points_added"].fillna(0), 0.0)

    def stats(g):
        n, t, r = len(g), int(g["t"].sum()), int(g["rec"].sum())
        t1, r1 = max(1, t), max(1, r)
        return {"routes": n, "targets": t, "tprr": round(100 * t / n, 1), "catch": round(100 * r / t1, 1),
                "adot": round(g["air"].sum() / t1, 1), "ypt": round(g["yds"].sum() / t1, 2),
                "yac": round(g["yac"].sum() / r1, 1), "yprr": round(g["yds"].sum() / n, 2),
                "epa_t": round(g["epa"].sum() / t1, 2)}

    # nflverse crosswalk: birth date + surname, unique matches only.
    nv = sources.players()[["gsis_id", "birth_date", "display_name", "latest_team", "position"]].dropna(subset=["gsis_id"])
    nv["bd"] = pd.to_datetime(nv["birth_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    nv["sn"] = nv["display_name"].map(_surname)
    players = players.assign(bd=pd.to_datetime(players["birth_date"], errors="coerce").dt.strftime("%Y-%m-%d"),
                             sn=players["display_name"].map(_surname))
    xw = players.merge(nv, on=["bd", "sn"], how="left", suffixes=("", "_nv"))
    xw = xw.groupby("nfl_id").filter(lambda d: d["gsis_id"].notna().sum() == 1).dropna(subset=["gsis_id"])
    gsis = xw.set_index("nfl_id")["gsis_id"].to_dict()

    out, unmatched = {}, []
    for nid, g in pp.groupby("nfl_id"):
        if nid not in gsis:
            unmatched.append(str(players.set_index("nfl_id").at[nid, "display_name"]))
            continue
        total = len(g)
        roles = {}
        for role, rg in g.groupby("role"):
            d = {"ALL": stats(rg), "route_share_pct": round(100 * len(rg) / total)}
            for mz, mg in rg.groupby("mz"):
                d[mz] = stats(mg)
            roles[role] = d
        info = players.set_index("nfl_id").loc[nid]
        out[gsis[nid]] = {"name": str(info["display_name"]), "draft_year": int(info["draft_year"]),
                          "position": str(info["nfl_position"]), "team": str(nv.set_index("gsis_id").at[gsis[nid], "latest_team"]),
                          "total_routes": total, "primary_role": max(roles, key=lambda r: roles[r]["ALL"]["routes"]),
                          "roles": roles}
    print(f"{len(out)} players written; {len(unmatched)} without a unique nflverse match: {unmatched[:15]}")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    FILE.parent.mkdir(parents=True, exist_ok=True)
    FILE.write_text(json.dumps(build(Path(sys.argv[1]).expanduser())))
    print("wrote", FILE)
