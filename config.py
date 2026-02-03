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
# Optimized for PREDICTION ACCURACY (rank correlation with actual points)
# Key insight: ownership (0.390) and value_score (0.369) are strongest signals
# Weak signals fixture_ease (0.098) and xgi_per_90 (0.086) heavily reduced
# Added recent_points for "hot hand" effect
POSITION_WEIGHTS = {
    "GKP": {
        "ep_next": 0.15,
        "form": 0.10,
        "fixture_ease": 0.04,  # Weak signal - minimized
        "defensive": 0.16,
        "value_score": 0.18,  # Strong signal
        "minutes_security": 0.12,
        "ownership": 0.15,  # Strongest signal
        "recent_points": 0.10,  # Hot hand
    },
    "DEF": {
        "ep_next": 0.15,
        "form": 0.10,
        "xgi_per_90": 0.04,  # Weak signal - minimized
        "fixture_ease": 0.04,  # Weak signal - minimized
        "defensive": 0.12,
        "value_score": 0.18,  # Strong signal
        "minutes_security": 0.08,
        "ownership": 0.17,  # Strongest signal
        "recent_points": 0.12,  # Hot hand
    },
    "MID": {
        "ep_next": 0.14,
        "form": 0.10,
        "xgi_per_90": 0.04,  # Weak signal - minimized
        "fixture_ease": 0.04,  # Weak signal - minimized
        "ict_position": 0.08,
        "value_score": 0.18,  # Strong signal
        "minutes_security": 0.08,
        "ownership": 0.18,  # Strongest signal - template mids matter
        "recent_points": 0.16,  # Hot hand - crucial for mids
    },
    "FWD": {
        "ep_next": 0.14,
        "form": 0.10,
        "xgi_per_90": 0.04,  # Weak signal - minimized
        "fixture_ease": 0.04,  # Weak signal - minimized
        "ict_position": 0.08,
        "value_score": 0.18,  # Strong signal
        "minutes_security": 0.08,
        "ownership": 0.18,  # Strongest signal - Haaland effect
        "recent_points": 0.16,  # Hot hand - crucial for fwds
    },
}

# Position-specific xGI multipliers (converts xGI/90 to 0-10 scale)
XGI_POSITION_MULTIPLIERS = {"GKP": 25.0, "DEF": 20.0, "MID": 12.5, "FWD": 11.0}

# Fixture recency decay weights: GW+1 through GW+5
# Softened decay for better multi-week planning
FIXTURE_DECAY_WEIGHTS = [1.0, 0.85, 0.70, 0.55, 0.45]

# Interaction term weights (bonuses for multiplicative effects)
TEAM_FORM_INTERACTION_WEIGHT = 0.15  # Team form × fixture ease
OWNERSHIP_FORM_INTERACTION_WEIGHT = 0.25  # Ownership × form (popular + hot = delivers)

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
