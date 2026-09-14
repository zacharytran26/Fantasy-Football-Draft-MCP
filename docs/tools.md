# Tool reference

Every tool the MCP server exposes. You won't call these by name in practice — ask in
plain language and the model picks — but this is what's available and what each returns.

## Setup

### `configure_league`
Create or update a named league and make it active.

| Argument | Default | Notes |
|---|---|---|
| `name` | `"default"` | Any label. Reusing a name updates that league. |
| `teams` | 12 | Any size. |
| `draft_slot` | 6 | Your first-round pick, 1-indexed. Validated against `teams`. |
| `rounds` | 16 | |
| `scoring` | `"half_ppr"` | `ppr`, `half_ppr`, `standard`. |
| `snake` | `true` | `false` for linear drafts. |
| `qb` `rb` `wr` `te` `flex` | 1/2/2/1/1 | Starting slots. |
| `superflex` | 0 | Slots where a QB may also start. |
| `te_premium_bonus` | 0.0 | Extra points per TE reception. |
| `consistency_weight` | 0.35 | 0 = pure upside, 1 = pure floor. |
| `adp_csv_path` | — | Your platform's ADP export. Beats consensus. |

### `list_leagues` / `switch_league` / `remove_league`
Manage multiple leagues. Switching is instant — boards are cached per format.

### `prewarm`
Build every cache. **Run before draft day.** The first query otherwise pays ~8 seconds,
or minutes on a genuinely cold cache.

### `refresh_data`
Rebuild from source. `force_download=true` re-downloads everything; use when nflverse
publishes new data mid-season.

### `model_settings`
Retune factor weights: `consistency_weight`, `injury_weight`, `oline_weight`,
`schedule_weight`, `pace_weight`, `td_luck_weight`. Rebuilds the board.

`td_luck_weight` (default 0.06) controls how hard a player's red zone touchdown
rate gets pulled toward what his position converts on average — see Touchdown luck
in [methodology.md](methodology.md). Set it to 0 for a board scored on raw history
with no touchdown-luck correction.

`qb_boost` is different from the rest — those all scale a real per-player signal
(O-line, pace, etc.); `qb_boost` is a direct fractional lift on QB `draft_score`
you supply because you believe the position is worth more than the projection
says, not because of any one player's inputs. It exists because
`champion_strategies`/`draft_backtest` can show whether QB has actually beaten
its draft cost across a specific league's real history — verify that before
setting it above 0, since it isn't a universal constant. Stacks with, doesn't
replace, the roster-need discount that already stops the model wanting a second
QB once you have one — a boost makes QB1 more competitive, it doesn't undo the
one-starting-slot logic.

## During the draft

### `on_the_clock`
The whole on-the-clock workflow in one call: `sync_draft` (fresh pull, no cached
state) → `draft_status` (round, on-the-clock, roster, confirmed against the sync)
→ `who_should_i_pick` (recommendation, reasoning, survival odds) → `value_picks`
(scoped to your current round and next) → `separation_report`, appended only when
the top recommendation is a WR or TE, for that player's route efficiency and
schedule context. Takes `sync_draft`'s arguments (`platform`, `league_id`,
`draft_id`, `pasted_board`, `season`) plus `limit` for how many recommendations
`who_should_i_pick` returns. Use this instead of the five calls separately when
you're on the clock and want the full picture at once.

### `who_should_i_pick`
The main one. Returns ranked recommendations with reasoning, the pick being evaluated,
your roster, and each player's odds of surviving to your next pick.

### `best_available`
Next best on the board. `sort_by`: `draft_score` (balanced), `vor`, `consistency`,
`proj_points`, or `value` (biggest ADP-to-model gap). Filter with `position`.

`position="K"` returns the kicker ranking instead of the skill-position board —
kickers aren't run through the same model (see
[methodology.md#kickers](methodology.md#kickers)), so this ignores `sort_by` and
always orders by `kicker_ppg` (recency-weighted fantasy points per game under the
league's distance-banded scoring). Also not draft-state aware: a kicker doesn't
drop off the list once picked, since `sync_draft`/`record_pick` don't track K.

### `sync_draft`
Pull the live board.

- `platform="sleeper"`, `draft_id` — automatic, public API, no credentials.
- `platform="espn"`, `league_id` — public leagues work as-is; private need
  `ESPN_SWID` and `ESPN_S2` environment variables.
- `platform="paste"`, `pasted_board` — any platform. Handles numbered lists,
  "Round 3, Pick 7 — Name", comma-separated runs, trailing team and position tags.

### `record_pick` / `undo_pick` / `reset_draft` / `draft_status`
Manual board management. `record_pick` accepts shorthand.

### `plan_my_draft`
Simulate every remaining pick from your slot. `strategy`: `balanced`, `zero_rb`,
`hero_rb`, `robust_rb`. ADP-driven, so treat it as preparation rather than a script.

## Research

### `player_report`
Every modelled factor for one player: production, role, environment multipliers, injury
components, separation, draft capital for rookies. Includes red zone role
(`rz_touches`, `rz_td`, `rz_td_rate`) against the position's baseline conversion rate
(`rz_baseline_rate`) and the resulting `m_td_luck` multiplier — surfaced in the plain-
language `summary` as "touchdown regression" whenever it moves the projection.

Also matches kickers by name, falling back to a differently-shaped report — FG
splits by distance, PAT, and team offense context instead of the skill-position
multipliers above — since kickers aren't run through the same model. See
[methodology.md#kickers](methodology.md#kickers).

### `compare_players`
Two to four players head to head, with a verdict. Also matches kickers by name,
reported separately from skill-position players (`kicker_ppg` isn't comparable to
`draft_score`) — mainly useful for comparing two kickers head to head.

### `rookie_report`
This year's class, projected from draft capital and landing spot. Widest error bars on
the board.

### `separation_report`
Separation, cushion, YPRR and TPRR. Pass `player_name` for one player's history, or
`position` for a leaderboard. Only players clearing 250 estimated routes and 50 targets.
Ranked by `sep_score` (talent). Also returns `matchup_z` (the player's team's upcoming
schedule difficulty at that position -- the season-long, team-level stand-in for a
WR/CB matchup chart), shown for reference only: a backtest (`matchup_backtest`) found
blending it into the ranking made WR predictions worse than talent alone, not better.

### `value_picks`
Where the model disagrees with the market. `direction`: `undervalued` or `overvalued`.
Restricted to players the market actually ranks.

### `team_context`
An NFL team's offensive environment: O-line ranks with history, pace, run/pass split,
schedule difficulty, divisional games, drive efficiency, and red zone play-calling
identity.

`drive_efficiency` (`pct_td`/`pct_fg`/`pct_punt`) is the share of that team's drives
ending in each outcome -- a multiplier on how many scoring chances its players get,
already reflected in their raw points, so read it as a confidence check on a role
rather than an extra score adjustment. `redzone_identity.shift` is neutral-field pass
rate minus red zone pass rate: a large positive shift means the offense goes notably
run-heavy inside the 20 (that team's receivers keep season-long volume but may lose
target share exactly where touchdowns happen); near zero or negative means the passing
game keeps its role in the scoring area too. Both are informational, like `matchup_z`
in `separation_report` -- not folded into `draft_score`, since blending an
unvalidated new signal into the score is exactly the mistake `matchup_backtest` exists
to catch.

Needs the `fixed_drive_result`/`drive` play-by-play columns, added after earlier cached
`play_by_play` parquets were built. If `drive_efficiency`/`redzone_identity` come back
empty for a season that should have data, run `refresh_data(force_download=true)`.

### `defense_report`
Fantasy points allowed by position, current season and five-year. Rank 1 = toughest.

### `draft_value_history`
Backtest consensus rank against actual finish. `group_by`: `draft_round` or `position`.
Converted to your scoring format.

### `persistent_value_players`
Players who beat their draft cost repeatedly rather than once.

### `matchup_backtest`
Validates whether blending schedule difficulty into talent predicts actual finish
better than talent alone. Talent comes from the *prior* season's separation score
only, schedule difficulty from the same leakage-free `strength_of_schedule` the live
recommender uses — nothing here has seen the season it's scoring. Reports Spearman
correlation and top-N precision for both metrics side by side, plus the players
where schedule swung the pick most. 2021-2024 result for WR: talent alone wins —
`separation_report` ranks by `sep_score` accordingly. Re-run this if the model
changes to see whether that still holds.

### `redzone_shift_backtest`
Backtest: does blending a team's red zone identity shift (from `team_context`) into the
existing touchdown-luck signal predict next season's points better than touchdown-luck
alone? Same leak-free discipline and same output shape as `matchup_backtest`, scored by
the same summary. A 2022-2025 run found the shift makes predictions *worse* for both WR
(`improvement_corr` -0.006, 300 player-seasons) and TE (-0.053, 117) — which is why
`redzone_identity_shift` stays informational-only in `team_context` rather than feeding
`m_td_luck`/`draft_score`. WR/TE only, since a pass-rate shift has no defensible sign for
a running back.

### `draft_backtest`
Replays a real past ESPN draft round by round: what `who_should_i_pick`'s algorithm
would have recommended given the real board at that exact moment, the true
hindsight-optimal pick by value over replacement (QB capped at 1 — a second
quarterback can't start, so it isn't ranked against real RB/WR/TE need), and what
you actually took, all scored on real points from that season. `league_id` and
`season` are all it needs — your team and draft slot are auto-detected from
`ESPN_SWID`/`ESPN_S2`, and league settings (teams, scoring, roster) are read
straight from ESPN. The board is leak-free: every history-derived input is bounded
to seasons strictly before the one being predicted, same as `matchup_backtest`.

Each of the three picks in a round also carries a value verdict (preseason ECR
against actual finish — the `value_picks` steal/bust framing, against real
outcomes instead of projections) and team context (that player's team's O-line
ranks, pace, and schedule difficulty for the season being tested — the same
numbers `team_context` reports, but leak-free for a past season instead of
always reading today's). K/DST aren't modelled anywhere in this tool, so those
rounds report your actual pick only, with no value or team context. ESPN only,
for now.

### `mock_draft`
Monte Carlo mock draft: the live algorithm against many simulated opponents,
averaged. Unlike `draft_backtest`, no real draft is needed or used — the other
teams are bots that pick by that season's real preseason ADP with realistic
reach/fall noise (bigger swings plausible late, tight consensus at the top)
rather than following it exactly, so who's actually on the board at your turn
varies draw to draw. Your slot (from your *active* configured league — run
`configure_league` first) runs the same `recommend()` logic `who_should_i_pick`
uses live, and everything is scored on real points from `season` against the
same leak-free board `draft_backtest` builds.

One draw can make the algorithm look better or worse than its true average just
from bot luck, which is why this runs `n_trials` (default 30) and reports the
mean/median/range, not a single result. For each round it also reports the most
common picks and how often each showed up — rounds with no real consensus
(usually round 6+) should be read as "plausible outcomes," not "the pick." K/DST
aren't modelled, so only skill-position rounds run (your league's total rounds
minus its K and DST starting slots).

### `champion_strategies`
What actually won your ESPN league, season by season, and which specific pick
made the difference — not just what the champion drafted, but which draft-cost
bet paid off. For each season it finds whichever team finished 1st and pulls
their real draft, running every pick through the same preseason-ECR-vs-actual-
finish value verdict `value_picks`/`draft_backtest` use. Reports each
champion's opening two picks, first QB/TE round, RB/WR volume, and biggest
steal, plus cross-season patterns: how often champions opened RB-RB, and the
median round of their first QB. ECR history only goes back to 2020 — earlier
seasons get position/timing data but no value verdicts. ESPN only.

`biggest_steal` also explains *why* it was a steal, not just that it was:
`usage_trend` is that player's real early- vs. late-season carries/targets/
target share (or pass attempts/yards for a QB) — an actual role or volume
change visible in the box scores, not assumed — and `team_environment` is his
team's O-line ranks, pace, and pass/rush split that season. Most value picks
turn out to be a volume or situation story, not raw talent beating a forecast;
this is what shows it concretely instead of leaving it to speculation.

### `resolve_names`
Check how names resolve before trusting a paste sync. Reports match type per name.

## Environment variables

| Variable | Purpose |
|---|---|
| `FFDRAFT_SEASON` | Season being drafted (default 2026) |
| `FFDRAFT_SEASONS` | Override lookback, e.g. `2021,2022,2023,2024,2025` |
| `FFDRAFT_CACHE` / `FFDRAFT_DATA` / `FFDRAFT_STATE` | Storage paths |
| `ESPN_SWID` / `ESPN_S2` | Private ESPN leagues — see [SECURITY.md](../SECURITY.md) |
