# Changelog

All notable changes to this project. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
[semantic](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

**Man-versus-zone coverage splits**
- New `coverage.py`: joins nflverse `pbp_participation` coverage tags (Cover 0/1/2/3/4/6,
  2-Man; about half of dropbacks are tagged) to play-by-play and the offensive players on
  the field. Corrects an earlier claim in `separation.py`, the docs and `CONTRIBUTING.md`
  that man/zone splits were not reproducible from open data.
- New tools: `coverage_report` (per-player or leaderboard, WR/TE/RB, veterans included,
  with an `available_only` filter), `coverage_roles` (league- or team-level target share by
  role against each coverage), `alignment_report`, and `coverage_split_backtest`.
  `player_report` gains a `coverage` section.
- Exposure is snaps on the field, not routes run, so `tpps` understates TPRR; comparisons
  are against the player's position baseline. Roles in `coverage_roles` are target-rank
  stand-ins (X=WR1, Z=WR2, Slot=WR3, Y=TE1), not alignment.
- Available on request, not in the recommendation output: `prewarm` builds the coverage data
  so `coverage_report` / `player_report` answer instantly, but `who_should_i_pick`,
  `on_the_clock`, `best_available` and `compare_players` do not show it (see the backtest
  below).
- Pocket-aware: coverage results are now split by pocket (`clean` / `pressured`, from nflverse's
  `was_pressure`) and time to throw, because a sped-up QB turns routes into sacks, incompletions,
  interceptions and marginal checkdowns. New `pocket_report` (outcome mix and EPA by coverage, pocket and
  time-to-throw bucket, plus every team's pressure rate allowed, sack rate and time to throw);
  `pocket` argument on `coverage_report`, `coverage_roles` and both backtests. 2023-2025: pressured
  dropbacks average -0.44 EPA vs +0.24 clean, and Cover 0 pressures the QB on ~40% of dropbacks. The
  coverage pattern survives holding the pocket clean, and pressure adds 3-5 points of RB target share on
  top. The value backtest now also tests the QB/protection side (prior-season team pressure rate, and
  its interaction with the man/zone gap): no improvement for WR, TE or RB. Sacks have no coverage tag
  before 2023, so pocket tables warn on earlier windows and the team pressure feature uses the ex-sack rate.
- New `pressure_usage_backtest`, answering "RBs and TEs live on dump-offs while WRs carry chunk plays":
  tests volume (targets and receiving yards per game) instead of points, because pressure shifts targets
  toward checkdowns and a points total can hide opposite effects. At the play level, 74% of clean-pocket RB
  targets are within 2 air yards, and under pressure WR targets go much deeper (aDOT 13.2 vs 9.9). Predictively, TEs on
  teams that allowed heavy pressure last year see fewer targets per game the next (-0.30 per SD, CI
  [-0.49, -0.11], ~4% lower out-of-sample error; own pressured share -0.22), i.e. pressure-inflated TE volume regresses.
  RB same sign, not significant; WR nothing; the dump-off share itself doesn't predict.
- New `pressure_rebound_backtest`: do skill players on a team with bad protection last year bounce back
  (worst vs best quintile of pressure allowed, same-team players, ppg residual after prior ppg,
  team-season bootstrap)? Unconfirmed for WR (+0.75 ppg [-0.25, +1.88], about what the mechanics
  predict), opposite signs for TE and RB, QB underpowered; all intervals include zero.
- New `coverage_value_backtest` (repo-style, prior-season inputs only, out-of-sample): adding
  a player's man-minus-zone gap, alone or interacted with how man-heavy his schedule is, to his
  prior points per game does NOT improve next-season prediction for WR, TE or RB (effects
  straddle zero; out-of-sample Spearman unchanged or slightly worse). Schedule man-exposure
  varies only ~2 points across teams, so it cannot move RB/WR target share (+0.15 pts of
  share per SD, CI [-0.5, +0.7]). Coverage was therefore taken out of the recommendation
  output entirely and kept as on-request context. The source's coverage tags also drift between seasons (league man share
  30/29/44/50/31% for 2021-2025); noted in the docs.
- **Informational only, not in `draft_score`.** `coverage_split_backtest` finds a player's
  man-minus-zone gap barely repeats season to season (WR rank correlation roughly 0.0-0.25
  on 45-58 players per season pair), so it fails the same bar `matchup_z` did. The stable
  signal is role/position-level (e.g. man coverage cuts RB target share ~5-9 points).
- New `alignment.py` + `scripts/build_alignment.py`: true snap alignment (X on the line,
  Z off the line, Slot, Y) for the 2023-2025 rookie classes from Big Data Bowl 2027
  tracking, joined to nflverse ids by birth date + surname. That data is CC BY-NC and the
  competition rules govern it, so nothing per-player is committed: the script builds a
  local file in `$FFDRAFT_DATA`, and every tool degrades gracefully without it.

**Kicker draft ranking**
- New `kickers.py`: a standalone kicker ranking, deliberately not run through
  `model.project()` (see its module docstring for why the O-line/pace/schedule/
  separation/td-luck machinery doesn't have a kicker analogue worth forcing).
  `kicker_ppg` is recency-weighted fantasy points per game across the same
  5-season lookback the skill-position baseline uses, scored under new
  distance-banded `Scoring` fields (`fg_made_0_39`/`fg_made_40_49`/
  `fg_made_50_plus`, `fg_missed`, `pat_made`/`pat_missed`, optional
  `fg_made_60_bonus`) defaulting to ESPN's public default scoring template.
- Team-offense context is reused, not re-derived: `context_tier` quintile-bands
  `features.team_drive_efficiency`'s `pct_fg` (how often a team's drives stall
  into a field goal try — the "which team's situation makes a kicker draftable"
  signal) across the board, alongside `team_pace_and_split`'s `plays_per_game`/
  `off_epa`. Informational only, same "not folded into the score" convention as
  `drive_efficiency` in `team_context`.
- `best_available(position="K")` returns the kicker board (sorted by
  `kicker_ppg`, ignores `sort_by`); `player_report` and `compare_players` now
  fall back to matching a kicker by name when no skill-position player matches.
  Not wired into `who_should_i_pick`, `draft_backtest`, or `mock_draft` — those
  still treat K/DST as outside the model.
- Rationale: under most leagues' real distance-banded scoring, a kicker trusted
  from 50+ on a good offense scores like a low-end WR2/high-end WR3 across a
  season, not the flat "3 points a field goal" replacement-level afterthought a
  standard scoring mental model assumes — worth more than "take one last."
  See [methodology.md#kickers](docs/methodology.md#kickers).

**Team drive efficiency and red zone identity**
- New `features.team_drive_efficiency` (share of a team's drives ending in a
  touchdown/field goal/punt) and `features.redzone_identity_shift` (a team's neutral
  pass rate minus its red zone pass rate), surfaced through `team_context`.
- Both are informational only, like `matchup_z` in `separation_report` — not folded
  into `draft_score`. New `redzone_shift_backtest` tool tested whether blending the
  identity shift into the touchdown-luck signal (`m_td_luck`) improves prediction of
  next-season points; a 2022-2025 run found it makes predictions *worse* for both WR
  (`improvement_corr` -0.006, 300 player-seasons) and TE (-0.053, 117 player-seasons),
  so it stays informational, matching the conclusion `matchup_backtest` already
  reached for schedule difficulty.
- `team_drive_efficiency` needs new play-by-play columns (`drive`,
  `fixed_drive_result`) not present in a `play_by_play` cache built before this
  change — run `refresh_data(force_download=true)` if it comes back empty.

**Touchdown luck**
- New environment multiplier (`m_td_luck`, weight `td_luck` / `td_luck_weight` in
  `model_settings`, default 0.06): a player's red zone touch/touchdown rate, from raw
  play-by-play, regressed toward what his position converts on average — computed
  from the same starter-caliber cohort the baseline projection regresses toward.
  Below 8 red zone touches a rate is treated as noise and sits neutral.
- `player_report` now shows `rz_touches`, `rz_td`, `rz_td_rate`, `rz_baseline_rate`
  alongside the multiplier, and `explain()`'s plain-language summary surfaces it as
  "touchdown regression" whenever it's non-trivial.
- New `features.player_redzone_role` (raw plays → per-player-season red zone
  touches/TDs) and `model.touchdown_luck_multiplier` (the bounded z-score
  adjustment, independently unit-tested).

## [1.0.0] — 2026-08-12

First release. A live fantasy football draft analyst exposed over MCP.

### Added

**Draft recommendations**
- `who_should_i_pick` weighs projected value, week-to-week consistency, open starting
  slots, and the odds each player survives to your next pick.
- Positional opportunity cost rather than raw value: what matters is the marginal gain
  over what a position still offers at your next turn, not who grades highest overall.
- `plan_my_draft` simulates all your picks from your slot under balanced, zero-RB,
  hero-RB or robust-RB strategies.

**Projections**
- Recency-weighted production across five seasons, converted to your exact scoring.
- Offensive line from adjusted line yards and pressure allowed per dropback.
- Neutral-script pace and run/pass split, measured only between 20% and 80% win
  probability so garbage time doesn't distort it.
- Five-year defensive strength by position defended, plus divisional schedule weighting.
- Injury risk from availability history, injury-report frequency and workload burden.
- Positional aging curves.

**Separation and route efficiency**
- NGS tracking separation and cushion, plus estimated YPRR and TPRR, as an open-data
  stand-in for paywalled charting. Man/zone splits are not reproducible.

**Rookies**
- Projected from draft capital, fitted to ten years of first-year outcomes and blended
  with empirical pick bins so the curve can't extrapolate past the data.

**Scoring formats**
- PPR, half PPR, standard, superflex and TE premium. Consensus rankings are converted
  between formats, since only PPR is published upstream.

**Multiple leagues**
- Named leagues with separate boards, replacement levels and in-progress drafts.

**Platform sync**
- Sleeper (public API), ESPN (public leagues, or private with cookies), or paste from
  anywhere.

**Name resolution**
- Aliases, nicknames, suffixes, punctuation, bare surnames, initialisms and typos.
  Ambiguous names name their candidates rather than guessing.

**Analysis**
- `draft_value_history` backtests consensus rank against actual finish across 913
  draftable player-seasons.
- `value_picks`, `separation_report`, `defense_report`, `team_context`, `player_report`,
  `compare_players`, `rookie_report`.

### Notes on accuracy

Several defaults were calibrated against data rather than assumed, and the reasoning is
documented in `docs/methodology.md`:

- Baseline projections regress toward the mean of *starter-caliber* players, not all
  players. Regressing toward a mean that includes third-stringers cut genuine starters
  by roughly a third.
- Consistency is regressed for small samples, so a backup with three good games doesn't
  outrank proven starters on reliability he never demonstrated.
- Rookie curves are capped at each pick bin's observed 75th percentile.
- Format conversion is damped to 0.6, because draft rooms move less than pure points
  arithmetic implies.
