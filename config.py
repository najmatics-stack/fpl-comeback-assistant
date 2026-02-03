"""
FPL Comeback Assistant Configuration
"""

# Your FPL Team ID (find it in your team URL: https://fantasy.premierleague.com/entry/YOUR_ID/event/1)
TEAM_ID = 7907269  # Najm's team

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

# Position-specific weight profiles for the enhanced scoring model
# BACKTEST PROVEN (20 GWs): Pure ownership beats ALL multi-factor models
# Ownership: 0.283 rank corr, 4.9 Top10 avg - the backtest winner
# Strategy: 100% ownership for selection, xGI for captaincy
POSITION_WEIGHTS = {
    "GKP": {
        "ep_next": 0.00,
        "form": 0.00,
        "fixture_ease": 0.00,
        "defensive": 0.00,
        "value_score": 0.00,
        "minutes_security": 0.00,
        "ownership": 1.00,  # PURE OWNERSHIP - 20 GW backtest winner
        "recent_points": 0.00,
    },
    "DEF": {
        "ep_next": 0.00,
        "form": 0.00,
        "xgi_per_90": 0.00,
        "fixture_ease": 0.00,
        "defensive": 0.00,
        "value_score": 0.00,
        "minutes_security": 0.00,
        "ownership": 1.00,  # PURE OWNERSHIP - 20 GW backtest winner
        "recent_points": 0.00,
    },
    "MID": {
        "ep_next": 0.00,
        "form": 0.00,
        "xgi_per_90": 0.00,
        "fixture_ease": 0.00,
        "ict_position": 0.00,
        "value_score": 0.00,
        "minutes_security": 0.00,
        "ownership": 1.00,  # PURE OWNERSHIP - 20 GW backtest winner
        "recent_points": 0.00,
    },
    "FWD": {
        "ep_next": 0.00,
        "form": 0.00,
        "xgi_per_90": 0.00,
        "fixture_ease": 0.00,
        "ict_position": 0.00,
        "value_score": 0.00,
        "minutes_security": 0.00,
        "ownership": 1.00,  # PURE OWNERSHIP - 20 GW backtest winner
        "recent_points": 0.00,
    },
}

# Captain selection strategy
# BACKTEST PROVEN: xGI beats ownership for captaincy (7.3 vs 7.1 pts/week)
CAPTAIN_USE_XGI = True  # Use xGI ranking for captain picks

# Position-specific xGI multipliers (converts xGI/90 to 0-10 scale)
XGI_POSITION_MULTIPLIERS = {"GKP": 25.0, "DEF": 20.0, "MID": 12.5, "FWD": 11.0}

# Fixture recency decay weights: GW+1 through GW+5
# Softened decay for better multi-week planning
FIXTURE_DECAY_WEIGHTS = [1.0, 0.85, 0.70, 0.55, 0.45]

# Interaction term weights (bonuses for multiplicative effects)
# DISABLED: Pure ownership backtest winner had no interactions
TEAM_FORM_INTERACTION_WEIGHT = 0.00  # Team form × fixture ease
OWNERSHIP_FORM_INTERACTION_WEIGHT = 0.00  # Ownership × form (popular + hot = delivers)

# Pure ownership mode: disable ALL bonuses (set_piece, bonus_magnet, transfer_momentum)
# BACKTEST PROVEN: Pure ownership beats all fancy bonus systems
PURE_OWNERSHIP_MODE = True

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

# Auto-mode defaults
AUTO_MAX_HITS = 1  # Max hits allowed (each hit = -4 pts)
AUTO_MIN_SCORE_GAIN_FREE = 0.5  # Min score gain for free transfers
AUTO_MIN_SCORE_GAIN_HIT = 1.5  # Min score gain for hit transfers
AUTO_MAX_TRANSFERS = 2  # Max total transfers per week
AUTO_RISK_LEVEL = "balanced"  # conservative, balanced, aggressive
