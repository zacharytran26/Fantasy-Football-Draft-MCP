# Fantasy Football Draft Analyst (MCP Server)

A live draft assistant. Connect it to Claude, sync your ESPN or Sleeper draft board, and
ask "who should I pick?" at every turn. It answers with a recommendation, the reasoning,
and the odds each player survives to your next pick.

[![CI](https://github.com/zacharytran26/Fantasy-Football-Draft-MCP/actions/workflows/ci.yml/badge.svg)](https://github.com/zacharytran26/Fantasy-Football-Draft-MCP/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Tested end to end against live data: **631 players** (551 veterans, 80 rookies) modelled
from **247,284 plays** across five seasons, priced against real 2026 preseason consensus.

---

## Quick start

```bash
git clone https://github.com/zacharytran26/Fantasy-Football-Draft-MCP.git
cd Fantasy-Football-Draft-MCP

# Use a virtual environment (required on macOS with Homebrew Python, and
# recommended everywhere else) -- see docs/quickstart.md if `pip`/`python`
# aren't found or you hit "externally-managed-environment".
#
# If this repo lives under an iCloud-synced folder (Desktop or Documents on
# macOS), put the venv somewhere NOT synced instead, e.g. ~/.venvs/ffdraft-mcp
# -- see docs/troubleshooting.md for why.
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .

# One-time data build (~3-6 min; downloads 5 seasons, then caches)
python setup_data.py
```

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "fantasy-draft": {
      "command": "/absolute/path/to/ff-draft-mcp/.venv/bin/python",
      "args": ["-m", "ffdraft.server"],
      "env": {
        "FFDRAFT_SEASON": "2026",
        "ESPN_SWID": "optional-for-private-espn-leagues",
        "ESPN_S2": "optional-for-private-espn-leagues",
        "PYTHONPATH": "/absolute/path/to/ff-draft-mcp/src"
      }
    }
  }
}
```

Use absolute paths for both `command` and `PYTHONPATH` — Claude Desktop launches the server
from an undefined working directory, so plain `python` or relative paths won't resolve.
`PYTHONPATH` is a deliberate belt-and-suspenders: on macOS with Python 3.13, an editable
install's `.pth` file can end up with the OS "hidden" flag set (cause varies), and Python
3.13 silently skips hidden `.pth` files at startup, which breaks the `ffdraft` import even
though `pip install -e .` succeeded. Pointing `PYTHONPATH` straight at `src/` sidesteps that
mechanism entirely. See [troubleshooting](docs/troubleshooting.md) if the server still
doesn't appear.

Then in Claude:

```
Set up my home league: 10 teams, full PPR, I pick 4th.
Set up my work league: 13 teams, half PPR, I pick 11th.
Switch to home. Sync my Sleeper draft 1234567890. Who should I pick?
```

---

## Documentation

| Guide | What's in it |
|---|---|
| [Quickstart](docs/quickstart.md) | Install to first recommendation in ten minutes |
| [Tool reference](docs/tools.md) | Every tool, argument and environment variable |
| [Methodology](docs/methodology.md) | How projections work and why the defaults are what they are |
| [Troubleshooting](docs/troubleshooting.md) | When something looks wrong |
| [Examples](examples/) | A full draft session, prompt snippets, sample ADP file |
| [Security](SECURITY.md) | ESPN cookies, network access, what's stored locally |
| [Contributing](CONTRIBUTING.md) | Tests, lint, and the bar for model changes |
| [Changelog](CHANGELOG.md) | Release history |

---

## What it does

**Recommends picks by opportunity cost, not raw value.** Knowing a player is good is
easy; knowing whether he'll still be there at your next turn is what decides who to take
now. For each position the model walks the board top-down, accumulating the probability
every better player is gone, and values a pick by its marginal gain over waiting.

At pick 6 with the top five off the board, that means recommending Amon-Ra St. Brown over
Josh Allen — Allen grades as the highest-value player left, but has a 73% chance of
lasting to pick 19, and the receiver has 8%.

**Projects players from five seasons of open data.** Offensive line from adjusted line
yards and pressure allowed, neutral-script pace and run/pass split, five-year defensive
strength by position, divisional schedule, injury history crossed with workload burden,
positional aging curves. Computed from raw plays rather than scraped from ranking tables,
so it recomputes under your scoring and doesn't break when a website changes its HTML.

**Optimises for consistency.** The default leans toward week-to-week reliability over
raw ceiling. Tune it with one number.

**Handles your actual leagues.** PPR, half PPR, standard, superflex, TE premium, any
size, any slot. Multiple leagues side by side with separate boards and drafts.

**Separation and route efficiency** from NGS tracking data — the open-data stand-in for
paywalled charting. Validated against a published PFF table: TPRR within a few
hundredths, ordering correct.

**Rookies** projected from draft capital, fitted to ten years of first-year outcomes.

**Live sync** from Sleeper (public API, no credentials), ESPN, or paste from anywhere.

**Name matching** that survives real drafts — `JSN`, `CMC`, `Bijan`, `Hollywood Brown`,
`Joshua Palmer`, and typos. Ambiguous names name their candidates rather than guessing.

---

## Tools

| Tool | What it does |
|---|---|
| `configure_league` | Create or update a named league; makes it active |
| `list_leagues` | All your leagues and which is active |
| `switch_league` | Change active league; board and draft resume instantly |
| `remove_league` | Delete a league and its draft history |
| `refresh_data` | Rebuild the board from source. Run once before draft day |
| `sync_draft` | Pull the live board from Sleeper, ESPN, or pasted text |
| `on_the_clock` | **The main one.** Sync, status, recommendation, round-scoped value, and matchup detail — one call |
| `who_should_i_pick` | Recommendation + reasoning for the pick on the clock |
| `best_available` | Next best players, sortable by value, consistency, or ADP bargain |
| `record_pick` / `undo_pick` / `reset_draft` | Manual board management |
| `draft_status` | Your roster and where the draft stands |
| `prewarm` | Build all caches before draft day so nothing computes on the clock |
| `rookie_report` | This year's rookie class, projected from draft capital |
| `resolve_names` | Check how names resolve before trusting a paste sync |
| `separation_report` | Separation, cushion, YPRR, TPRR, plus a schedule-adjusted matchup score |
| `coverage_report` | (Context only, not in recommendations) Man vs zone splits for any WR/TE/RB (targets per snap, catch %, yards, EPA) against the position baseline |
| `coverage_roles` | Which coverages feed which receiver roles, league-wide or for one team |
| `alignment_report` | True X / Z / Slot / Y leaders vs man or zone (2023-2025 rookie classes; needs a local build) |
| `pocket_report` | How dropbacks end by coverage, pressure and time to throw; team pressure rate allowed (context only) |
| `coverage_split_backtest` | Does a player's man-vs-zone gap repeat year to year? (mostly no) |
| `coverage_value_backtest` | Does coverage improve next-season prediction beyond prior production? (no, for WR/TE/RB) |
| `value_picks` | Where the model disagrees with the draft market |
| `draft_value_history` | Backtest: preseason rank vs actual finish, by round and position |
| `matchup_backtest` | Backtest: does schedule-adjusted matchup score beat talent alone at predicting finish? |
| `redzone_shift_backtest` | Backtest: does a team's red zone identity shift beat touchdown-luck alone at predicting finish? (2022-2025: no — stays informational) |
| `draft_backtest` | Replay a real past ESPN draft: algorithm's pick vs. true optimal vs. what you actually took, round by round, with value verdicts and team context for each |
| `mock_draft` | Monte Carlo mock draft: the algorithm vs. many simulated ADP-bot opponents, averaged over N trials |
| `champion_strategies` | What actually won your ESPN league each season, and which specific pick made the difference |
| `persistent_value_players` | Players who beat their draft cost year after year |
| `player_report` | Every modelled factor for one player |
| `compare_players` | Head to head, up to four |
| `team_context` | O-line ranks, pace, run/pass split, schedule for an NFL team |
| `defense_report` | Fantasy points allowed by position, 5-year view |
| `plan_my_draft` | Simulate all 16 of your picks from your slot |
| `model_settings` | Retune factor weights |

---

---

## Data attribution

This project computes everything from open sources and ships no third-party data:

- [nflverse](https://github.com/nflverse) — play-by-play, weekly stats, snap counts,
  injury reports, rosters, schedules, draft picks, combine.
- NFL Next Gen Stats (mirrored by nflverse) — separation, cushion, YAC over expected.
- [dynastyprocess/data](https://github.com/dynastyprocess/data) — FantasyPros expert
  consensus rank history.

Please respect the terms of those upstream projects. Nothing here scrapes paywalled
sources, and no proprietary data is redistributed.
