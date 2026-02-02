# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Project

```bash
source venv/bin/activate
python3 main.py                    # Full recommendations (default)
python3 main.py --auto             # Auto-pilot: execute transfers/captain/chips
python3 main.py --my-team          # Squad analysis with ratings
python3 main.py --league-spy       # Rival squad intel
python3 main.py --backtest 18-22   # Blind backtest over GW range
python3 main.py --evaluate 20      # Evaluate + auto-tune weights for GW
python3 main.py --compare 18,19,20 # Compare model vs baselines
python3 main.py --no-cache         # Bypass 6-hour cache
```

Team ID (`7907269`) is set in `config.py` and used by default for all commands.

```bash
pip install -r requirements.txt    # Install dependencies
python3 -m py_compile <file>       # Syntax check (no test suite exists)
```

## Architecture

Four-layer pipeline: **Data → Analysis → Output**, orchestrated by `main.py`.

### Data Layer (`data/`)
- `fpl_api.py` — Async FPL API wrapper. `FPLDataFetcher.load_all_data()` fetches bootstrap + fixtures, builds `Player`/`Team`/`Fixture` dataclasses. All HTTP via aiohttp.
- `fpl_actions.py` — Authenticated actions (transfers, captain, chips). Uses Selenium for browser login with cookie persistence (`~/.fpl_cookies`, 7-day TTL). FPL blocks programmatic login via DataDome, so a real Chrome window is required on first use.
- `news_scraper.py` — Scrapes Fantasy Football Scout RSS + injury pages via BeautifulSoup.
- `cache.py` — File-based cache with configurable TTL (default 6 hours). Keys like `bootstrap`, `fixtures`, `picks_{team}_{gw}`.

### Analysis Layer (`analysis/`)
- `player_scorer.py` — Weighted scoring across 6 factors (form, xGI, fixtures, value, ICT, minutes). Weights in `config.SCORING_WEIGHTS`, auto-tuned by `evaluator.py`.
- `fixture_analyzer.py` — FDR ratings with home/away split, DGW/BGW detection.
- `league_spy.py` — Scans rival squads in mini-leagues. Produces `LeagueIntel` dataclass with ownership/captain/differential analysis. Integrated into default and `--auto` recommendation flows.
- `differential.py`, `chip_optimizer.py`, `backtester.py`, `evaluator.py`, `comparative_backtest.py` — Specialized analyzers.

### Output Layer (`output/`)
- `recommendations.py` — `RecommendationEngine` generates `FullRecommendation` (injuries, transfers, captains, differentials, chips, fixtures). Accepts optional `LeagueIntel` to adjust captain picks (0.85x fade / 1.15x target multipliers) and transfer scores (1.15x differential bonus / 0.85x template penalty). `format_full_recommendations()` renders the full report.

### Key Data Flow
```
main_async() → FPLDataFetcher.load_all_data()
             → PlayerScorer, FixtureAnalyzer, etc. initialized
             → fetch_team_squad() if team_id set
             → LeagueSpy.analyze_league() for auto/default modes
             → RecommendationEngine(league_intel=...)
             → get_full_recommendations() → format output
```

## Key Patterns

- **All API calls are async** — aiohttp sessions with rate limiting (1s delays, exponential backoff on 429/403).
- **Dataclasses everywhere** — `Player`, `Team`, `ScoredPlayer`, `Differential`, `LeagueIntel`, `TransferRecommendation`, `CaptainPick`, `ChipStrategy`, etc.
- **Blind backtesting** — `PreGWSnapshot` captures only data available before target GW to prevent leakage.
- **FPL rule enforcement** — `_would_violate_rules()` checks 3-per-team, position limits, and squad size on every transfer recommendation.
- **League spy integration** — When `team_id` is set in auto/default modes, `LeagueSpy` runs automatically. If it fails, recommendations proceed without league intel (graceful degradation).

## Auth System (`data/fpl_actions.py`)

FPL uses DataDome bot protection. The auth flow:
1. Try saved cookies from `~/.fpl_cookies`
2. If expired/missing, open Chrome via Selenium for manual login
3. Poll for `csrftoken` + `sessionid` cookies
4. Save cookies (7-day expiry) — subsequent runs need no login

All authenticated API calls require `Cookie`, `X-CSRFToken`, `Referer`, `Origin`, and `X-Requested-With` headers.

## Config (`config.py`)

Scoring weights, differential thresholds, fixture lookahead, cache TTL, expected BGW/DGW gameweeks, and display limits. `TEAM_ID = 7907269` is the default team.
