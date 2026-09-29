"""Draft logic: survival probability, roster need, and scoring-format conversion."""
import numpy as np
import pandas as pd

from ffdraft.board import FORMAT_SHIFT_DAMPING, attach_bye_weeks, convert_adp_format, synthetic_adp
from ffdraft.config import LeagueSettings
from ffdraft.model import (
    HANDCUFF_BONUS,
    _positional_need,
    apply_current_team,
    expected_best_at_next_pick,
    likely_alternative_by_position,
    roster_construction_mult,
    survival_probability,
    survival_probability_vec,
    touchdown_luck_multiplier,
)


class TestSurvival:
    def test_a_player_going_before_your_pick_is_gone(self):
        assert survival_probability(adp=5, current_pick=20, next_pick=33) < 0.05

    def test_a_late_adp_player_survives(self):
        assert survival_probability(adp=120, current_pick=20, next_pick=33) > 0.9

    def test_probability_falls_as_the_wait_lengthens(self):
        short = survival_probability(adp=40, current_pick=20, next_pick=25)
        long = survival_probability(adp=40, current_pick=20, next_pick=60)
        assert short > long

    def test_always_a_probability(self):
        for adp in (1, 10, 50, 100, 250):
            for nxt in (12, 40, 90):
                p = survival_probability(adp, 10, nxt)
                assert 0.0 <= p <= 1.0

    def test_vectorised_matches_scalar(self):
        adps = np.array([3.0, 25.0, 60.0, 140.0])
        vec = survival_probability_vec(adps, 20, 33)
        scal = [survival_probability(a, 20, 33) for a in adps]
        assert np.allclose(vec, scal, atol=1e-9)

    def test_missing_adp_does_not_produce_nan(self):
        out = survival_probability_vec(np.array([np.nan, 30.0]), 10, 20)
        assert not np.isnan(out).any()


class TestPositionalNeed:
    def test_empty_starting_slot_is_a_premium(self):
        need = _positional_need(LeagueSettings(teams=12), {})
        assert need["RB"] > 1.0 and need["WR"] > 1.0

    def test_backup_quarterback_is_nearly_worthless_in_one_qb(self):
        need = _positional_need(LeagueSettings(teams=12), {"QB": 1, "RB": 2, "WR": 2, "TE": 1})
        assert need["QB"] < 0.3

    def test_third_quarterback_is_worthless(self):
        need = _positional_need(LeagueSettings(teams=12), {"QB": 2, "RB": 2, "WR": 2, "TE": 1})
        assert need["QB"] < 0.05

    def test_superflex_keeps_the_second_quarterback_valuable(self):
        roster = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
        one_qb = _positional_need(LeagueSettings(teams=12), roster)["QB"]
        sflex = _positional_need(LeagueSettings(teams=12, superflex=1), roster)["QB"]
        assert sflex > one_qb * 3
        assert sflex > 1.0  # it's still a starting slot

    def test_running_back_depth_holds_value(self):
        """Backs get hurt constantly, so bench backs actually enter lineups."""
        roster = {"QB": 1, "RB": 3, "WR": 2, "TE": 1}
        need = _positional_need(LeagueSettings(teams=12), roster)
        assert need["RB"] > need["QB"]

    def test_roster_cap_shuts_a_position_off(self):
        need = _positional_need(LeagueSettings(teams=12), {"WR": 9})
        assert need["WR"] < 0.05


class TestOpportunityCost:
    def test_value_of_waiting_reflects_who_survives(self):
        board = pd.DataFrame([
            {"position": "QB", "draft_score": 100.0, "p_available_next": 0.9},
            {"position": "QB", "draft_score": 95.0, "p_available_next": 0.95},
            {"position": "RB", "draft_score": 100.0, "p_available_next": 0.01},
            {"position": "RB", "draft_score": 40.0, "p_available_next": 0.99},
        ])
        fallback = expected_best_at_next_pick(board)
        # Quarterbacks survive, so waiting costs almost nothing.
        assert fallback["QB"] > 90
        # The elite back will be gone; waiting drops you to a much worse player.
        assert fallback["RB"] < 60

    def test_empty_position_is_handled(self):
        board = pd.DataFrame([{"position": "TE", "draft_score": 10.0,
                               "p_available_next": 0.0}])
        out = expected_best_at_next_pick(board)
        assert np.isfinite(out["TE"])


class TestCurrentTeam:
    def test_depth_chart_overrides_a_stale_team(self):
        """A player traded since he last played a game should show the new team."""
        tbl = pd.DataFrame([{"player_id": "p1", "name": "Trade Guy", "team": "OLD"}])
        dc = pd.DataFrame([{"player_id": "p1", "team": "NEW"}])
        out = apply_current_team(tbl, dc)
        assert out.loc[0, "team"] == "NEW"

    def test_player_missing_from_depth_chart_keeps_last_known_team(self):
        tbl = pd.DataFrame([{"player_id": "p1", "name": "Rookie", "team": "OLD"}])
        dc = pd.DataFrame([{"player_id": "p2", "team": "NEW"}])
        out = apply_current_team(tbl, dc)
        assert out.loc[0, "team"] == "OLD"

    def test_empty_depth_chart_is_a_no_op(self):
        tbl = pd.DataFrame([{"player_id": "p1", "name": "X", "team": "OLD"}])
        out = apply_current_team(tbl, pd.DataFrame(columns=["player_id", "team"]))
        assert out.loc[0, "team"] == "OLD"

    def test_none_depth_chart_is_a_no_op(self):
        tbl = pd.DataFrame([{"player_id": "p1", "name": "X", "team": "OLD"}])
        out = apply_current_team(tbl, None)
        assert out.loc[0, "team"] == "OLD"

    def test_multiple_players_only_matched_ones_move(self):
        tbl = pd.DataFrame([
            {"player_id": "p1", "name": "Traded", "team": "OLD"},
            {"player_id": "p2", "name": "Stayed", "team": "SAME"},
        ])
        dc = pd.DataFrame([
            {"player_id": "p1", "team": "NEW"},
            {"player_id": "p2", "team": "SAME"},
        ])
        out = apply_current_team(tbl, dc).set_index("player_id")
        assert out.loc["p1", "team"] == "NEW"
        assert out.loc["p2", "team"] == "SAME"

    def test_depth_rank_carries_over_when_the_feed_has_it(self):
        tbl = pd.DataFrame([{"player_id": "p1", "name": "Backup", "team": "OLD"}])
        dc = pd.DataFrame([{"player_id": "p1", "team": "SF", "depth_rank": 2}])
        out = apply_current_team(tbl, dc)
        assert out.loc[0, "depth_rank"] == 2

    def test_depth_rank_is_nan_when_the_feed_lacks_it(self):
        tbl = pd.DataFrame([{"player_id": "p1", "name": "X", "team": "OLD"}])
        dc = pd.DataFrame([{"player_id": "p1", "team": "NEW"}])
        out = apply_current_team(tbl, dc)
        assert out.loc[0, "depth_rank"] is None or pd.isna(out.loc[0, "depth_rank"])


class TestTouchdownLuck:
    """touchdown_luck_multiplier is a cross-sectional z-score, like every other
    environment multiplier in project() -- it needs a real spread of players to
    compare against, so single-player cases are exercised as one row in a small
    board rather than in isolation.
    """

    def test_overperformer_gets_discounted_relative_to_the_field(self):
        # Player 0 converted way more red zone touches than baseline predicts;
        # players 1-2 landed close to it.
        touches = pd.Series([20.0, 20.0, 20.0])
        td = pd.Series([10.0, 4.0, 5.0])         # baseline expects 4 on 20 touches
        baseline = pd.Series([0.20, 0.20, 0.20])
        m = touchdown_luck_multiplier(touches, td, baseline, weight=0.06)
        assert m.iloc[0] < 1.0
        assert m.iloc[0] < m.iloc[1]

    def test_underperformer_gets_boosted_relative_to_the_field(self):
        touches = pd.Series([20.0, 20.0, 20.0])
        td = pd.Series([1.0, 4.0, 5.0])          # baseline expects 4 on 20 touches
        baseline = pd.Series([0.20, 0.20, 0.20])
        m = touchdown_luck_multiplier(touches, td, baseline, weight=0.06)
        assert m.iloc[0] > 1.0
        assert m.iloc[0] > m.iloc[1]

    def test_small_sample_is_pinned_neutral_even_in_a_skewed_field(self):
        """A two-touch, two-score '100%' sample sits at exactly 1.0, regardless of
        how much variance the qualifying players around it carry."""
        touches = pd.Series([2.0, 20.0, 20.0])
        td = pd.Series([2.0, 10.0, 1.0])
        baseline = pd.Series([0.20, 0.20, 0.20])
        m = touchdown_luck_multiplier(touches, td, baseline, weight=0.06, min_touches=8)
        assert m.iloc[0] == 1.0

    def test_weight_zero_disables_the_adjustment(self):
        touches = pd.Series([20.0, 20.0])
        td = pd.Series([10.0, 1.0])
        baseline = pd.Series([0.20, 0.20])
        m = touchdown_luck_multiplier(touches, td, baseline, weight=0.0)
        assert (m == 1.0).all()

    def test_never_exceeds_the_configured_weight(self):
        touches = pd.Series([50.0, 50.0, 50.0])
        td = pd.Series([49.0, 0.0, 10.0])   # one huge overperformer, one huge underperformer
        baseline = pd.Series([0.20, 0.20, 0.20])
        m = touchdown_luck_multiplier(touches, td, baseline, weight=0.06)
        assert ((m - 1.0).abs() <= 0.06 + 1e-9).all()

    def test_missing_baseline_does_not_produce_nan(self):
        touches = pd.Series([20.0, 20.0])
        td = pd.Series([5.0, 8.0])
        baseline = pd.Series([np.nan, 0.20])
        m = touchdown_luck_multiplier(touches, td, baseline, weight=0.06)
        assert np.isfinite(m).all()

    def test_uniform_field_is_neutral(self):
        """Everyone matches the baseline exactly -- no spread, no adjustment."""
        touches = pd.Series([20.0, 30.0, 40.0])
        td = pd.Series([4.0, 6.0, 8.0])     # each exactly 20%
        baseline = pd.Series([0.20, 0.20, 0.20])
        m = touchdown_luck_multiplier(touches, td, baseline, weight=0.06)
        assert (m == 1.0).all()


class TestSyntheticAdp:
    def test_quarterbacks_and_tight_ends_slide_past_their_value(self):
        """A room does not draft in value order: the QB1 goes far later than RB1."""
        assert synthetic_adp("QB", 1) > synthetic_adp("RB", 1) * 5
        assert synthetic_adp("TE", 1) > synthetic_adp("WR", 1) * 3

    def test_monotonic_within_a_position(self):
        for pos in ("QB", "RB", "WR", "TE"):
            vals = [synthetic_adp(pos, r) for r in range(1, 30)]
            assert vals == sorted(vals)


class TestFormatConversion:
    @staticmethod
    def _board():
        """A board with realistic depth. Format conversion works on rank shifts, so
        a three-player board can't move anyone — the ordering has nowhere to go."""
        rows = []
        for i in range(60):   # receivers: high reception volume
            rows.append({"name": f"WR{i}", "position": "WR",
                         "proj_points": 240 - i * 2.5,
                         "receptions": 95 - i})
        for i in range(50):   # backs: mixed, some pass-catching some not
            rows.append({"name": f"RB{i}", "position": "RB",
                         "proj_points": 250 - i * 3.0,
                         "receptions": (70 - i) if i % 2 == 0 else 15})
        for i in range(24):   # quarterbacks: zero receptions
            rows.append({"name": f"QB{i}", "position": "QB",
                         "proj_points": 380 - i * 6.0, "receptions": 0})
        b = pd.DataFrame(rows)
        b["receptions"] = b["receptions"].clip(lower=0)
        b = b.sort_values("proj_points", ascending=False).reset_index(drop=True)
        b["overall_rank"] = np.arange(1, len(b) + 1)
        b["adp"] = b["overall_rank"].astype(float)
        # proj_points here are league-format points; PPR adds the missing credit.
        return b

    def _converted(self, label, gap):
        b = self._board()
        b["proj_points_ppr"] = b["proj_points"] + gap * b["receptions"]
        return convert_adp_format(b, label).set_index("name")

    def test_ppr_league_leaves_rankings_untouched(self):
        b = self._board()
        b["proj_points_ppr"] = b["proj_points"]
        out = convert_adp_format(b, "ppr")
        assert out["adp"].equals(b["adp"])
        assert out["adp_format"].iloc[0] == "ppr"

    def test_reception_heavy_players_fall_in_standard(self):
        out = self._converted("standard", 1.0)
        # WR0 catches 95 passes; RB1 catches 15.
        assert out.loc["WR0", "adp"] > out.loc["WR0", "adp_ppr"]
        assert out.loc["RB1", "adp"] < out.loc["RB1", "adp_ppr"]

    def test_quarterbacks_move_less_than_receivers(self):
        out = self._converted("standard", 1.0)
        qb_move = float(abs(out.loc["QB0", "adp"] - out.loc["QB0", "adp_ppr"]))
        wr_move = float(abs(out.loc["WR0", "adp"] - out.loc["WR0", "adp_ppr"]))
        assert qb_move < wr_move

    def test_half_ppr_shift_is_smaller_than_standard(self):
        half = self._converted("half_ppr", 0.5)
        std = self._converted("standard", 1.0)
        assert abs(half.loc["WR0", "adp_shift"]) < abs(std.loc["WR0", "adp_shift"])

    def test_shift_is_damped_not_applied_whole(self):
        assert 0 < FORMAT_SHIFT_DAMPING < 1

    def test_adp_never_goes_below_one(self):
        for label, gap in [("half_ppr", 0.5), ("standard", 1.0)]:
            out = self._converted(label, gap)
            assert (out["adp"] >= 1.0).all()

    def test_missing_ppr_column_is_a_no_op(self):
        b = self._board().drop(columns=[])
        out = convert_adp_format(b, "standard")
        assert out["adp"].equals(b["adp"])


class TestByeWeekAttachment:
    def test_maps_team_to_bye_via_features(self, monkeypatch):
        import ffdraft.features as features_mod

        monkeypatch.setattr(features_mod, "bye_weeks", lambda season=None: {"BUF": 7})
        out = attach_bye_weeks(pd.DataFrame([{"name": "X", "team": "BUF"}]))
        assert out.loc[0, "bye"] == 7

    def test_missing_team_column_does_not_raise(self):
        out = attach_bye_weeks(pd.DataFrame([{"name": "X"}]))
        assert out["bye"].isna().all()


class TestRosterConstructionMult:
    # A small league (one starter per flex-eligible position, one FLEX) so the
    # combined flex-eligible requirement is an easy-to-reason-about 4: RB(1) +
    # WR(1) + TE(1) + FLEX(1).
    _SMALL_LEAGUE = LeagueSettings(starters={"QB": 1, "RB": 1, "WR": 1, "TE": 1,
                                             "FLEX": 1, "K": 0, "DST": 0})

    def test_healthy_depth_absorbs_a_lone_bye_with_no_discount(self):
        # 4 flex-eligible players already owned (meets the requirement on its own,
        # with real depth), none sharing the candidate's bye -- adding a 5th on a
        # bye nobody else has still leaves 4 healthy for 4 required slots.
        avail = pd.DataFrame([{"position": "WR", "team": "MIA", "bye": 6, "draft_score": 50.0}])
        owned = pd.DataFrame([
            {"position": "RB", "team": "BUF", "bye": 1, "draft_score": 80.0},
            {"position": "RB", "team": "KC", "bye": 1, "draft_score": 40.0},
            {"position": "WR", "team": "SF", "bye": 1, "draft_score": 60.0},
            {"position": "TE", "team": "DAL", "bye": 1, "draft_score": 30.0},
        ])
        out = roster_construction_mult(avail, owned, self._SMALL_LEAGUE)
        assert out["bye_mult"].iloc[0] == 1.0

    def test_zero_cushion_roster_shows_a_modest_shortfall_even_alone(self):
        # Exactly at the required depth (4) with zero bench cushion -- even a bye
        # week nobody else shares still leaves the roster one body short that week.
        avail = pd.DataFrame([{"position": "WR", "team": "MIA", "bye": 6, "draft_score": 50.0}])
        owned = pd.DataFrame([
            {"position": "RB", "team": "BUF", "bye": 1, "draft_score": 80.0},
            {"position": "WR", "team": "SF", "bye": 1, "draft_score": 60.0},
            {"position": "TE", "team": "DAL", "bye": 1, "draft_score": 30.0},
        ])
        out = roster_construction_mult(avail, owned, self._SMALL_LEAGUE)
        assert out["bye_mult"].iloc[0] == 0.85

    def test_a_real_bye_week_collision_is_a_severe_discount(self):
        # 3 of the (exactly-4-required) flex-eligible players already share the
        # candidate's bye week -- a real, severe shortfall, not just thin depth.
        avail = pd.DataFrame([{"position": "WR", "team": "MIA", "bye": 6, "draft_score": 50.0}])
        owned = pd.DataFrame([
            {"position": "RB", "team": "BUF", "bye": 6, "draft_score": 80.0},
            {"position": "WR", "team": "SF", "bye": 6, "draft_score": 60.0},
            {"position": "TE", "team": "DAL", "bye": 6, "draft_score": 30.0},
        ])
        out = roster_construction_mult(avail, owned, self._SMALL_LEAGUE)
        assert out["bye_mult"].iloc[0] == 0.40

    def test_healthy_depth_shows_no_injury_shortfall(self):
        # 4 durable flex-eligible players (exp_games near the 17-game max) already
        # meet the requirement on their own -- adding a durable 5th shows no
        # fragility discount.
        avail = pd.DataFrame([{"position": "WR", "exp_games": 17.0, "draft_score": 50.0}])
        owned = pd.DataFrame([
            {"position": "RB", "exp_games": 17.0, "draft_score": 80.0},
            {"position": "RB", "exp_games": 16.0, "draft_score": 40.0},
            {"position": "WR", "exp_games": 17.0, "draft_score": 60.0},
            {"position": "TE", "exp_games": 16.0, "draft_score": 30.0},
        ])
        out = roster_construction_mult(avail, owned, self._SMALL_LEAGUE)
        assert out["injury_mult"].iloc[0] == 1.0

    def test_fragile_roster_shows_a_moderate_injury_discount(self):
        # 3 flex-eligible players each averaging ~10 of 17 games -- a real, if
        # probabilistic, depth risk at a thin position.
        avail = pd.DataFrame([{"position": "WR", "exp_games": 10.0, "draft_score": 50.0}])
        owned = pd.DataFrame([
            {"position": "RB", "exp_games": 10.0, "draft_score": 80.0},
            {"position": "WR", "exp_games": 10.0, "draft_score": 60.0},
            {"position": "TE", "exp_games": 10.0, "draft_score": 30.0},
        ])
        out = roster_construction_mult(avail, owned, self._SMALL_LEAGUE)
        assert out["injury_mult"].iloc[0] == 0.83

    def test_severely_fragile_roster_is_a_steep_injury_discount(self):
        avail = pd.DataFrame([{"position": "WR", "exp_games": 5.0, "draft_score": 50.0}])
        owned = pd.DataFrame([
            {"position": "RB", "exp_games": 5.0, "draft_score": 80.0},
            {"position": "WR", "exp_games": 5.0, "draft_score": 60.0},
            {"position": "TE", "exp_games": 5.0, "draft_score": 30.0},
        ])
        out = roster_construction_mult(avail, owned, self._SMALL_LEAGUE)
        assert out["injury_mult"].iloc[0] == 0.65

    def test_differentiates_a_durable_from_a_fragile_candidate_past_saturation(self):
        # Regression check for the bug the smoke test caught: rounding the
        # continuous shortfall to a whole slot collapsed two genuinely different
        # risk levels (2.9 vs 3.4) into the same bucket, scoring a durable and a
        # fragile candidate identically. A bigger, more realistic league (RB2/WR2/
        # TE1/FLEX1 = required 6) exercises exactly that gap.
        league = LeagueSettings(teams=10)
        owned = pd.DataFrame([
            {"position": "RB", "exp_games": 9.0, "draft_score": 70.0},
            {"position": "RB", "exp_games": 10.0, "draft_score": 50.0},
            {"position": "WR", "exp_games": 17.0, "draft_score": 60.0},
        ])
        avail = pd.DataFrame([
            {"position": "RB", "exp_games": 17.0, "draft_score": 54.0},  # durable
            {"position": "RB", "exp_games": 8.0, "draft_score": 55.0},   # fragile
        ])
        out = roster_construction_mult(avail, owned, league)
        assert out["injury_mult"].iloc[0] > out["injury_mult"].iloc[1]

    def test_missing_exp_games_column_is_a_no_op(self):
        avail = pd.DataFrame([{"position": "WR", "draft_score": 50.0}])
        owned = pd.DataFrame([{"position": "RB", "draft_score": 80.0}])
        out = roster_construction_mult(avail, owned, self._SMALL_LEAGUE)
        assert out["injury_mult"].iloc[0] == 1.0

    def test_team_exposure_discounts_heavy_stacking(self):
        avail = pd.DataFrame([{"position": "WR", "team": "BUF", "bye": 12, "draft_score": 50.0}])
        owned = pd.DataFrame([
            {"position": "RB", "team": "BUF", "bye": 12, "draft_score": 80.0},
            {"position": "QB", "team": "BUF", "bye": 12, "draft_score": 70.0},
        ])
        out = roster_construction_mult(avail, owned)
        assert out["exposure_mult"].iloc[0] == 0.95

    def test_handcuff_bonus_for_a_clear_backup_running_back(self):
        """No depth_rank column at all -- the value-proxy fallback, same behavior
        as before real depth-chart data was wired in."""
        avail = pd.DataFrame([{"position": "RB", "team": "SF", "bye": 9, "draft_score": 20.0}])
        owned = pd.DataFrame([{"position": "RB", "team": "SF", "bye": 9, "draft_score": 90.0}])
        out = roster_construction_mult(avail, owned)
        assert out["handcuff_mult"].iloc[0] == HANDCUFF_BONUS

    def test_no_handcuff_bonus_when_candidate_outscores_your_own_starter(self):
        """A same-team, same-position player who beats your own starter is a
        competing option, not insurance -- no bonus (value-proxy fallback)."""
        avail = pd.DataFrame([{"position": "RB", "team": "SF", "bye": 9, "draft_score": 95.0}])
        owned = pd.DataFrame([{"position": "RB", "team": "SF", "bye": 9, "draft_score": 90.0}])
        out = roster_construction_mult(avail, owned)
        assert out["handcuff_mult"].iloc[0] == 1.0

    def test_depth_chart_calls_a_real_handcuff_even_when_value_disagrees(self):
        # Candidate actually projects HIGHER than the rostered starter -- the
        # value-proxy fallback would have called this a lateral, not insurance --
        # but the real depth chart clearly lists him RB2 behind your own RB1.
        avail = pd.DataFrame([{"position": "RB", "team": "SF", "bye": 9,
                              "draft_score": 95.0, "depth_rank": 2}])
        owned = pd.DataFrame([{"position": "RB", "team": "SF", "bye": 9,
                              "draft_score": 90.0, "depth_rank": 1}])
        out = roster_construction_mult(avail, owned)
        assert out["handcuff_mult"].iloc[0] == HANDCUFF_BONUS

    def test_depth_chart_gives_no_bonus_to_a_co_starter(self):
        # Both backs are listed as rank 1 on the real depth chart (a true
        # committee) -- no bonus, whatever the value gap between them, since the
        # depth chart itself doesn't call this one a backup to the other.
        avail = pd.DataFrame([{"position": "RB", "team": "SF", "bye": 9,
                              "draft_score": 40.0, "depth_rank": 1}])
        owned = pd.DataFrame([{"position": "RB", "team": "SF", "bye": 9,
                              "draft_score": 45.0, "depth_rank": 1}])
        out = roster_construction_mult(avail, owned)
        assert out["handcuff_mult"].iloc[0] == 1.0

    def test_handcuff_does_not_apply_outside_running_back(self):
        avail = pd.DataFrame([{"position": "WR", "team": "SF", "bye": 9, "draft_score": 20.0}])
        owned = pd.DataFrame([{"position": "WR", "team": "SF", "bye": 9, "draft_score": 90.0}])
        out = roster_construction_mult(avail, owned)
        assert out["handcuff_mult"].iloc[0] == 1.0

    def test_empty_roster_is_a_no_op(self):
        avail = pd.DataFrame([{"position": "WR", "team": "SF", "bye": 9, "draft_score": 20.0}])
        out = roster_construction_mult(avail, None)
        assert (out == 1.0).all().all()


class TestLikelyAlternative:
    def test_names_the_best_survivor_at_even_odds(self):
        avail = pd.DataFrame([
            {"position": "RB", "name": "Elite", "draft_score": 100.0, "adp": 3.0},
            {"position": "RB", "name": "Solid", "draft_score": 60.0, "adp": 40.0},
        ])
        out = likely_alternative_by_position(avail, current_pick=10, next_pick=20)
        assert out["RB"]["name"] == "Solid"
        assert out["RB"]["likely"] is True

    def test_falls_back_to_best_chance_when_nobody_clears_even_odds(self):
        # adp=5 is long gone by pick 10, and waiting the full 20 picks to pick 30
        # only makes that more certain -- no survivor clears even odds.
        avail = pd.DataFrame([{"position": "QB", "name": "Only", "draft_score": 50.0, "adp": 5.0}])
        out = likely_alternative_by_position(avail, current_pick=10, next_pick=30)
        assert out["QB"]["name"] == "Only"
        assert out["QB"]["likely"] is False
