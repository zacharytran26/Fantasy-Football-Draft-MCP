"""League settings, scoring rules, and model weights."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

CACHE_DIR = Path(os.environ.get("FFDRAFT_CACHE", Path.home() / ".ffdraft" / "cache"))
DATA_DIR = Path(os.environ.get("FFDRAFT_DATA", Path.home() / ".ffdraft" / "data"))
STATE_DIR = Path(os.environ.get("FFDRAFT_STATE", Path.home() / ".ffdraft" / "state"))
for _d in (CACHE_DIR, DATA_DIR, STATE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Seasons pulled for the 5-year lookback. Override with FFDRAFT_SEASONS=2021,2022,...
CURRENT_SEASON = int(os.environ.get("FFDRAFT_SEASON", "2026"))
SEASONS = [int(s) for s in os.environ["FFDRAFT_SEASONS"].split(",")] if os.environ.get(
    "FFDRAFT_SEASONS"
) else list(range(CURRENT_SEASON - 5, CURRENT_SEASON))

# Recency weights applied oldest -> newest across SEASONS.
RECENCY_WEIGHTS = [0.05, 0.10, 0.17, 0.28, 0.40]

FANTASY_POSITIONS = ("QB", "RB", "WR", "TE")

# Weekly point thresholds that count as a "startable" week, by position.
# Used for the consistency / floor score.
STARTABLE_THRESHOLD = {"QB": 18.0, "RB": 12.0, "WR": 12.0, "TE": 9.0}
# Weekly point thresholds that count as a "league-winning" week.
SPIKE_THRESHOLD = {"QB": 25.0, "RB": 20.0, "WR": 20.0, "TE": 16.0}

# Age at which each position's production curve turns down, and the annual decay after.
AGE_CLIFF = {"QB": 34, "RB": 27, "WR": 30, "TE": 31}
AGE_DECAY = {"QB": 0.03, "RB": 0.09, "WR": 0.04, "TE": 0.04}

# Season touch counts above which the following-year injury/decline risk rises.
WORKLOAD_BURDEN = {"RB": 300, "WR": 145, "TE": 110, "QB": 620}


@dataclass
class Scoring:
    """Per-event scoring. Defaults are half-PPR."""

    pass_yd: float = 0.04
    pass_td: float = 4.0
    interception: float = -2.0
    rush_yd: float = 0.1
    rush_td: float = 6.0
    rec: float = 0.5
    rec_yd: float = 0.1
    rec_td: float = 6.0
    fumble_lost: float = -2.0
    two_pt: float = 2.0

    # Kicker scoring. Defaults match ESPN's own public default scoring template
    # (games.espn.com/s/ffllm scoring master list): three distance bands, a flat
    # penalty for any miss regardless of distance, PAT made/missed. Like every
    # other field here, these are defaults, not a universal constant -- a specific
    # league's real settings should override them the same way a custom rec value
    # would, this project just doesn't have an ESPN-sync path for it yet (see
    # espn_league_context, which only derives the PPR/half/standard preset from
    # statId 53, not a full custom scoring set).
    fg_made_0_39: float = 3.0
    fg_made_40_49: float = 4.0
    fg_made_50_plus: float = 5.0
    fg_missed: float = -1.0       # any distance, made blocks fall under this too
    pat_made: float = 1.0
    pat_missed: float = -1.0      # made blocks fall under this too
    # Extra points for a 50+ make that's specifically 60+, ON TOP OF fg_made_50_plus
    # (not a replacement -- ESPN's default template has no 60+ tier at all). Off by
    # default, same convention as qb_boost/coverage_trend in ModelWeights: some
    # custom leagues add this bonus given how much more common 60+ attempts have
    # become, but it isn't a default ESPN category, so it doesn't ship enabled.
    fg_made_60_bonus: float = 0.0

    @classmethod
    def preset(cls, name: str) -> Scoring:
        name = name.lower().replace("-", "_")
        if name in ("ppr", "full_ppr"):
            return cls(rec=1.0)
        if name in ("standard", "non_ppr"):
            return cls(rec=0.0)
        if name in ("te_premium", "tep"):
            return cls(rec=1.0)  # TE bonus handled in projection layer
        return cls()


@dataclass
class LeagueSettings:
    name: str = "default"
    teams: int = 12
    rounds: int = 16
    draft_slot: int = 6           # your 1-indexed pick in round 1
    snake: bool = True
    scoring: Scoring = field(default_factory=Scoring)
    starters: dict = field(default_factory=lambda: {
        "QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DST": 1,
    })
    flex_eligible: tuple = ("RB", "WR", "TE")
    te_premium_bonus: float = 0.0  # extra points per TE reception
    superflex: int = 0             # slots where a QB may also start

    def picks_for_slot(self, slot: int | None = None) -> list[int]:
        """Overall pick numbers belonging to a draft slot."""
        slot = slot or self.draft_slot
        picks = []
        for rnd in range(1, self.rounds + 1):
            if self.snake and rnd % 2 == 0:
                pick_in_round = self.teams - slot + 1
            else:
                pick_in_round = slot
            picks.append((rnd - 1) * self.teams + pick_in_round)
        return picks

    def replacement_ranks(self) -> dict[str, int]:
        """Positional rank of the 'replacement level' player for VOR.

        Scales with league size, which is the whole point: the last startable running
        back in a 10-team league is a much better player than in a 13-team league, so
        the same player is worth less value over replacement in the smaller league.
        """
        t = self.teams
        base = {p: self.starters.get(p, 0) * t for p in FANTASY_POSITIONS}
        # Distribute FLEX slots across eligible positions by historical draft share.
        flex_total = self.starters.get("FLEX", 0) * t
        share = {"RB": 0.45, "WR": 0.45, "TE": 0.10}
        for pos, frac in share.items():
            if pos in base:
                base[pos] += round(flex_total * frac)
        # Superflex makes quarterbacks scarce: nearly every team starts two, so the
        # replacement quarterback is far deeper down the list than in a 1-QB league.
        if self.superflex:
            base["QB"] += round(self.superflex * t * 0.9)
        # Bench depth teams actually roster before a position stops mattering,
        # scaled to league size.
        scale = t / 12
        bench_pad = {"QB": 4, "RB": 12, "WR": 14, "TE": 4}
        return {p: max(1, base.get(p, 1) + round(bench_pad.get(p, 0) * scale))
                for p in FANTASY_POSITIONS}

    def roster_slots(self) -> int:
        return sum(self.starters.values()) + self.superflex

    def cache_key(self) -> str:
        """Identifies a board. Two leagues sharing these settings share a board;
        anything that changes projections or replacement level must appear here."""
        sc = ",".join(f"{k}={v}" for k, v in sorted(asdict(self.scoring).items()))
        st = ",".join(f"{k}={v}" for k, v in sorted(self.starters.items()))
        raw = f"t{self.teams}|sf{self.superflex}|tep{self.te_premium_bonus}|{sc}|{st}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]


@dataclass
class ModelWeights:
    """How much each factor moves a player off their baseline projection.

    Each value is the maximum fractional swing (+/-) that factor can apply.
    Tune these; they are the opinionated part of the model.
    """

    oline: float = 0.06          # O-line quality (run block for RB, pass block for QB/pass catchers)
    pace_volume: float = 0.07    # team plays/game + run-pass split vs. player's role
    schedule: float = 0.05       # 5-yr opponent defensive strength, recency-weighted
    divisional: float = 0.015    # 6 games vs. known division defenses
    injury: float = 0.10         # injury history + workload burden
    separation: float = 0.05     # route-win quality for WR/TE (NGS separation, YPRR, TPRR)
    age: float = 0.08            # position age curve
    # Red zone touchdown rate regressed toward the position's baseline conversion
    # rate (from real per-player red zone touches/TDs, not a belief you supply --
    # unlike qb_boost below). A player who scored on far more or fewer of his red
    # zone touches than his position converts on average gets pulled toward that
    # baseline, since a single season's conversion rate is one of the noisiest
    # signals in the model and touchdowns are worth more to the score than
    # anything else a player does.
    td_luck: float = 0.06
    # How the final ranking trades off ceiling vs. floor. 0 = pure expected points,
    # 1 = pure week-to-week reliability. The user asked for consistency, so this leans high.
    consistency_weight: float = 0.35
    # Direct positional tilt on draft_score, separate from every weight above -- those
    # all adjust the *projection* from real per-player signals (O-line, pace, etc.);
    # this instead says "this position is worth more than the projection alone says,"
    # a belief you supply rather than something derived from the player's own inputs.
    # Added after a real-draft-history check (draft_backtest/champion_strategies) found
    # quarterbacks beating their actual draft cost by 20-55 spots in every one of 5
    # measurable seasons in one league, not just a hot year or two -- consistent with
    # the broader, well-known dynamic that QB scoring has outpaced ADP since only one
    # roster slot is required regardless of how much the position scores. Defaults to
    # 0 (no change from prior behavior); a fractional draft_score multiplier for QB
    # only, e.g. 0.12 for +12%. Re-run that check before trusting a value here for a
    # league you haven't verified it in -- this isn't a universal constant.
    qb_boost: float = 0.0
    # Same "supplied belief, not derived from this player's own history" convention
    # as qb_boost, defaulting to off for the same reason: unlike separation/td_luck,
    # this isn't backed by a per-player real signal this codebase can backtest --
    # man-vs-zone and personnel-package rates aren't in any open dataset (see
    # separation.py's docstring), so there's no equivalent of matchup_backtest or
    # redzone_shift_backtest to validate it against. The belief, from 2025-season
    # external analysis (PFF, MatchQuarters, Sharp Football): defenses shifted hard
    # to zone (man coverage 22.6% of snaps, down from 33%+) and, later in the
    # season, from nickel back to base personnel (nickel ~68%->61%, base ~23%->29%)
    # after split-safety shells stopped working. Base personnel means a linebacker,
    # not a nickel corner, increasingly covers the slot/backfield -- a mismatch that
    # rewards short-area quickness over boundary/vertical separation, and backs
    # already schemed into the passing game. Re-derive these rates for the season
    # you're drafting before trusting a nonzero value here; they will decay.
    coverage_trend: float = 0.0


LEAGUES_PATH = STATE_DIR / "leagues.json"


def _blank_store() -> dict:
    return {"active": None, "leagues": {}}


def _read_store() -> dict:
    if not LEAGUES_PATH.exists():
        return _blank_store()
    try:
        raw = json.loads(LEAGUES_PATH.read_text())
        if "leagues" not in raw:
            return _blank_store()
        return raw
    except (json.JSONDecodeError, OSError):
        return _blank_store()


def _known_fields(cls, raw: dict) -> dict:
    """Drop keys that no longer exist on the dataclass, e.g. a field retired
    since this entry was saved -- old leagues should still load, not crash."""
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in valid}


def _deserialize(entry: dict) -> tuple[LeagueSettings, ModelWeights]:
    sc_raw = entry.get("scoring", {})
    sc = (Scoring(**_known_fields(Scoring, sc_raw)) if isinstance(sc_raw, dict)
          else Scoring.preset(str(sc_raw)))
    lg_kwargs = _known_fields(LeagueSettings,
                               {k: v for k, v in entry.get("league", {}).items() if k != "scoring"})
    if "flex_eligible" in lg_kwargs:
        lg_kwargs["flex_eligible"] = tuple(lg_kwargs["flex_eligible"])
    league = LeagueSettings(scoring=sc, **lg_kwargs)
    weights = ModelWeights(**_known_fields(ModelWeights, entry.get("weights", {})))
    return league, weights


def _serialize(league: LeagueSettings, weights: ModelWeights) -> dict:
    return {
        "league": {k: v for k, v in asdict(league).items() if k != "scoring"},
        "scoring": asdict(league.scoring),
        "weights": asdict(weights),
    }


def list_leagues() -> tuple[list[str], str | None]:
    store = _read_store()
    return sorted(store["leagues"]), store.get("active")


def load_settings(name: str | None = None) -> tuple[LeagueSettings, ModelWeights]:
    """Load a named league, or the active one. Falls back to defaults if none exist."""
    store = _read_store()
    name = name or store.get("active")
    entry = store["leagues"].get(name) if name else None
    if not entry:
        return LeagueSettings(), ModelWeights()
    return _deserialize(entry)


def save_settings(league: LeagueSettings, weights: ModelWeights,
                  make_active: bool = True) -> Path:
    """Persist a league under its own name, leaving other leagues untouched.

    Keeping leagues separate matters: a 10-team full-PPR league and a 13-team
    half-PPR league produce different projections, different replacement levels
    and different draft boards, and their in-progress drafts must not mix.
    """
    store = _read_store()
    store["leagues"][league.name] = _serialize(league, weights)
    if make_active or not store.get("active"):
        store["active"] = league.name
    LEAGUES_PATH.write_text(json.dumps(store, indent=2))
    return LEAGUES_PATH


def set_active(name: str) -> bool:
    store = _read_store()
    if name not in store["leagues"]:
        return False
    store["active"] = name
    LEAGUES_PATH.write_text(json.dumps(store, indent=2))
    return True


def delete_league(name: str) -> bool:
    store = _read_store()
    if name not in store["leagues"]:
        return False
    del store["leagues"][name]
    if store.get("active") == name:
        store["active"] = next(iter(sorted(store["leagues"])), None)
    LEAGUES_PATH.write_text(json.dumps(store, indent=2))
    return True
