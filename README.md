# FPL Comeback Assistant

A Python tool to help you climb the Fantasy Premier League rankings with data-driven recommendations.

## Features

- **Transfer Recommendations**: Identifies optimal player swaps based on form, fixtures, and value
- **Captain Picker**: Ranks captain options with expected points calculations
- **Differential Finder**: Discovers low-ownership gems with high upside
- **Chip Strategy Optimizer**: Recommends when to use Wildcard, Free Hit, Triple Captain, and Bench Boost
- **Injury Tracking**: Flags unavailable players from your squad
- **Fixture Analysis**: Rates upcoming fixtures for all teams

## Installation

```bash
# Clone the repository
git clone https://github.com/najmatics-stack/fpl-comeback-assistant.git
cd fpl-comeback-assistant

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage (General Recommendations)

```bash
python main.py
```

### With Your Team ID

Find your team ID in your FPL URL: `https://fantasy.premierleague.com/entry/YOUR_ID/event/1`

```bash
python main.py --team-id 12345678
```

### Command Line Options

```bash
# Full recommendations with team-specific analysis
python main.py --team-id 12345678

# Show top differentials by position
python main.py --differentials

# Show fixture difficulty rankings
python main.py --fixtures

# Show top players by position
python main.py --top-players

# Detailed chip strategy
python main.py --chip-strategy

# Specify available chips
python main.py --chips wildcard bench_boost

# Fresh data (bypass cache)
python main.py --no-cache

# Minimal output
python main.py --quiet
```

## Configuration

Edit `config.py` to customize:

- `TEAM_ID`: Your default FPL team ID
- `SCORING_WEIGHTS`: Adjust player ranking algorithm weights
- `DIFFERENTIAL_MAX_OWNERSHIP`: Ownership threshold for differentials
- `EXPECTED_BGW` / `EXPECTED_DGW`: Update expected blank/double gameweeks

## Output Example

```
============================================================
  FPL COMEBACK ASSISTANT - GW24 RECOMMENDATIONS
============================================================

📋 INJURY/AVAILABILITY ALERTS
----------------------------------------
   ⚠️ Salah (LIV): 75% - minor knock
   ❌ Haaland (MCI): OUT - muscle injury

🔄 TRANSFER RECOMMENDATIONS
----------------------------------------
   1. OUT: Watkins (AVL) → IN: Isak (NEW)
      +£0.5m | +2.3 score | better form, better fixtures

👑 CAPTAIN PICKS
----------------------------------------
   1. Salah (LIV) - 7.8 exp pts ⚠️doubt
      vs BOU(H) | excellent form, DOUBLE GW
   2. Palmer (CHE) - 6.9 exp pts
      vs WHU(H) | good form, easy fixture

🎯 DIFFERENTIAL PICKS
----------------------------------------
   MID:
      • Eze (CRY) £6.2m - 5.1% owned
        hot form, easy fixtures

🃏 CHIP STRATEGY
----------------------------------------
   📅 BENCH BOOST: GW33
      15 doubles in DGW33
   📅 TRIPLE CAPTAIN: GW36
      Premium double in DGW36
```

## Project Structure

```
fpl-comeback-assistant/
├── main.py                 # Entry point
├── config.py               # Settings
├── data/
│   ├── fpl_api.py          # FPL API integration
│   ├── news_scraper.py     # Injury/news scraping
│   └── cache.py            # Local caching
├── analysis/
│   ├── player_scorer.py    # Player ranking algorithm
│   ├── fixture_analyzer.py # FDR analysis
│   ├── differential.py     # Low-ownership finder
│   └── chip_optimizer.py   # Chip timing
├── output/
│   └── recommendations.py  # Formatting
└── requirements.txt
```

## Comeback Strategy Tips

1. **Be aggressive with differentials** - You need to gain ground on rivals
2. **Don't follow the crowd** - Avoid high-ownership template picks
3. **Save chips for DGWs** - Maximum point potential
4. **Take calculated hits** - A -4 for a 10+ point gain is worth it
5. **Captain aggressively** - Pick ceiling over floor

## Data Sources

- **FPL API**: `https://fantasy.premierleague.com/api/bootstrap-static/`
- **Fantasy Football Scout**: Injury updates and team news

## License

MIT License
