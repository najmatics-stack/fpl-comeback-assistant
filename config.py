"""
FPL Comeback Assistant Configuration
"""

# Your FPL Team ID (find it in your team URL: https://fantasy.premierleague.com/entry/YOUR_ID/event/1)
TEAM_ID = None  # Set your team ID here, e.g., 12345678

# Current gameweek (will be auto-detected if None)
CURRENT_GW = None

# Scoring weights for player evaluation
SCORING_WEIGHTS = {
    "form": 0.25,
    "xgi_per_90": 0.20,
    "fixture_ease": 0.20,
    "value_score": 0.15,
    "ict_index": 0.10,
    "minutes_security": 0.10,
}

# Differential thresholds
DIFFERENTIAL_MAX_OWNERSHIP = 10.0  # Players owned by less than 10%
DIFFERENTIAL_MIN_FORM = 4.0  # Minimum form score
DIFFERENTIAL_MIN_MINUTES = 60  # Minimum minutes per game

# Fixture lookahead (number of gameweeks to analyze)
FIXTURE_LOOKAHEAD = 5

# Cache settings
CACHE_EXPIRY_HOURS = 6  # How long to cache API data

# News sources
NEWS_RSS_URL = "https://www.fantasyfootballscout.co.uk/feed/"
INJURIES_URL = "https://www.fantasyfootballscout.co.uk/fantasy-football-injuries/"
TEAM_NEWS_URL = "https://www.fantasyfootballscout.co.uk/team-news/"

# Expected blank and double gameweeks (update as season progresses)
EXPECTED_BGW = [31, 34]  # Blank gameweeks
EXPECTED_DGW = [33, 36]  # Double gameweeks

# Display settings
TOP_TRANSFERS = 5  # Number of transfer suggestions to show
TOP_CAPTAINS = 3  # Number of captain options to show
TOP_DIFFERENTIALS = 5  # Number of differentials per position
