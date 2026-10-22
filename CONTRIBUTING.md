# Contributing

## Setup

```bash
git clone https://github.com/zacharytran26/Fantasy-Football-Draft-MCP.git
cd Fantasy-Football-Draft-MCP
pip install -e .
pip install pytest ruff
```

## Before opening a PR

```bash
pytest tests -q          # 71 tests, under a second
ruff check src tests
```

The test suite is deliberately **offline** — no network, no cached NFL data. That keeps
CI fast and stops it breaking when an upstream dataset moves. Keep new tests that way
where you can; if a test genuinely needs data, mark it `@pytest.mark.integration` and
skip when the cache is absent.

## Changing the model

Model changes need **evidence, not reasoning**. Show before-and-after on real players — a
board top-10, or the projections that moved.

This isn't bureaucracy. Several defaults here exist because a perfectly sensible-looking
first attempt produced visibly wrong output:

- Regressing small samples toward the all-player positional mean cut genuine starters by
  a third, because the mean included third-stringers.
- A rookie curve fitted with plain log-linear regression promised 19.4 PPG at pick 3 when
  the top-ten bin has averaged 15.9 across six players in a decade.
- Applying the full scoring-format shift sent Derrick Henry from ADP 38 to 1.0 in
  standard. Right direction, absurd magnitude.
- The draft simulator once took eight quarterbacks, because the roster-need floor of 0.62
  couldn't stop a position whose marginal value recovered as its best players left.

Each of those looked fine in the diff. They were caught by printing a board and reading
it. Please do that.

## Data sources

Open sources only — nflverse, Next Gen Stats, public consensus rankings. No paywalled
data, no scraping around access controls, and no redistributing third-party datasets in
the repo.

Some things genuinely can't be built here. True receiver alignment (X/Z/slot) for
veterans needs tracking or charting that only commercial providers have; the only open
alignment data covers the 2023-2025 rookie classes (see `alignment.py`). That's a real
limit, not an oversight. (Man/zone splits, once thought out of reach, are in `coverage.py`.)

## Name matching

The alias map in `names.py` grows from real cases. If a name failed to resolve, open an
issue with what you typed and who you meant — those reports are useful even when they
look trivial. An unresolved name doesn't raise an error; it silently becomes a player who
scored zero.

When adding aliases, prefer the automatic path. Initialisms generate from name tokens, so
`JSN` works without an entry. Only hand-map what can't be derived, like `CMC`, which comes
from capitals inside a surname.

## Style

Ruff config lives in `ruff.toml`. Comments should explain *why*, especially where a value
was tuned — the number itself is visible in the code, the reason it isn't 1.0 is not.

## Reporting security issues

Not through public issues. See [SECURITY.md](SECURITY.md).
