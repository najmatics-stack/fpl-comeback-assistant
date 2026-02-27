"""
Player scoring algorithm for ranking and recommendations
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from data.fpl_api import FPLDataFetcher, Player
from data.news_scraper import NewsScraper, InjuryStatus
from analysis.fixture_analyzer import FixtureAnalyzer

import config

WEIGHTS_FILE = Path("weights_history.json")


def compute_weighted_score(
    factors: Dict[str, float],
    weights: Dict[str, float],
    bonuses: Optional[Dict[str, float]] = None,
) -> float:
    """Single source of truth for combining factor scores into an overall score.

    Args:
        factors: Factor name -> score (0-10 scale each)
        weights: Factor name -> weight (should sum to ~1.0)
        bonuses: Optional additive bonuses (not weighted, added directly)

    Returns:
        Overall weighted score
    """
    score = sum(factors.get(k, 0) * w for k, w in weights.items())
    if bonuses:
        score += sum(bonuses.values())
    return score


@dataclass
class ScoredPlayer:
    """Player with computed scores"""

    player: Player
    overall_score: float
    form_score: float
    xgi_score: float
    fixture_score: float
    value_score: float
    ict_score: float
    minutes_score: float
    availability: str  # "fit", "doubt", "injured", "suspended"
    injury_details: Optional[str]
    # New scoring dimensions (defaults for backward compat)
    ep_next_score: float = 0.0
    defensive_score: float = 0.0
    transfer_momentum: float = 0.0
    availability_multiplier: float = 1.0
    form_fixture_interaction: float = 0.0  # Team form × fixture ease interaction
    ownership_score: float = 0.0  # Global ownership wisdom-of-crowds signal
    recent_points_score: float = 0.0  # Last GW actual points (hot hand)
    ownership_form_interaction: float = 0.0  # Popular + in-form = delivers


def load_tuned_weights() -> Optional[Dict[str, float]]:
    """Load the most recently tuned weights from weights_history.json"""
    if not WEIGHTS_FILE.exists():
        return None

    try:
        with open(WEIGHTS_FILE) as f:
            history = json.load(f)

        if not history:
            return None

        # Find the latest entry - keys can be "23" or "23-7gw"
        def parse_gw(key: str) -> int:
            if "-" in key:
                return int(key.split("-")[0])
            return int(key)

        latest_key = max(history.keys(), key=parse_gw)
        weights = history[latest_key].get("weights")
        if weights and config.DEBUG:
            print(f"   [model] Loaded tuned weights from {latest_key} evaluation")
        return weights
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


class PlayerScorer:
    """Scores and ranks players for FPL recommendations"""

    def __init__(
        self,
        fpl_data: FPLDataFetcher,
        fixture_analyzer: FixtureAnalyzer,
        news_scraper: Optional[NewsScraper] = None,
        use_tuned_weights: bool = True,
    ):
        self.fpl = fpl_data
        self.fixtures = fixture_analyzer
        self.news = news_scraper
        self._injuries: Optional[Dict[str, InjuryStatus]] = None
        self._score_cache: Dict[int, ScoredPlayer] = {}  # player_id -> cached result

        # Try to load tuned weights, fall back to config defaults
        self.tuned_weights = None
        if use_tuned_weights:
            self.tuned_weights = load_tuned_weights()

        self.weights = config.SCORING_WEIGHTS

    def _get_injuries(self) -> Dict[str, InjuryStatus]:
        """Get injury data (cached)"""
        if self._injuries is None and self.news:
            self._injuries = self.news.scrape_injuries()
        return self._injuries or {}

    def _get_player_availability(self, player: Player) -> Tuple[str, Optional[str]]:
        """Determine player availability status"""
        # First check FPL API status
        if player.status == "i":
            return "injured", player.news
        elif player.status == "s":
            return "suspended", player.news
        elif player.status == "u":
            return "injured", player.news  # Unavailable = injured/left club

        # Check chance of playing
        if player.chance_of_playing is not None:
            if player.chance_of_playing == 0:
                return "injured", player.news
            elif player.chance_of_playing <= 50:
                return "doubt", f"{player.chance_of_playing}% - {player.news}"
            elif player.chance_of_playing <= 75:
                return "doubt", f"{player.chance_of_playing}% - {player.news}"

        # Check scraped news
        injuries = self._get_injuries()
        player_key = player.web_name.lower()

        for key, injury in injuries.items():
            if player_key in key or key in player_key:
                return injury.status, injury.details

        return "fit", None

    def _calculate_form_score(self, player: Player) -> float:
        """Calculate form score (0-10)"""
        return min(10, player.form)

    def _calculate_xgi_score(self, player: Player) -> float:
        """Calculate xGI per 90 score (0-10), position-aware"""
        if player.minutes < 90:
            return 0

        xgi_per_90 = (player.expected_goal_involvements / player.minutes) * 90

        # Position-adjusted thresholds: defenders get credit for lower xGI
        mult = config.XGI_POSITION_MULTIPLIERS.get(player.position, 12.5)

        return min(10, xgi_per_90 * mult)

    def _calculate_fixture_score(self, player: Player) -> float:
        """Calculate fixture ease score (0-10) for IMMEDIATE next GW only.

        Uses single-fixture analysis, not multi-week average.
        Our analysis shows +32.7% better performance vs bottom-half teams,
        so this factor DOES matter when measured correctly.
        """
        return self.fixtures.get_next_fixture_ease(player.team_id, player.position)

    def _calculate_fixture_run_score(self, player: Player) -> float:
        """Calculate multi-GW fixture ease score (0-10) for transfer planning.

        Uses 5-GW lookahead with decay weights. Better for evaluating
        transfer targets over multiple weeks, not single-GW prediction.
        """
        return self.fixtures.get_fixture_ease_score(player.team_id, player.position)

    def _calculate_value_score(self, player: Player) -> float:
        """Calculate value score blending FPL's value_form with season value (0-10)"""
        if player.price <= 0:
            return 0

        # FPL-calculated recent value (points per £m over last 4 GWs)
        recent_value = min(10, player.value_form * 3.0)

        # Season-long value
        season_value = min(10, player.total_points / player.price / 1.5)

        # Blend: 60% recent, 40% season
        return recent_value * 0.6 + season_value * 0.4

    def _calculate_ict_score(self, player: Player) -> float:
        """Calculate ICT index score (0-10)"""
        return min(10, player.ict_index / 50)

    def _calculate_minutes_score(self, player: Player) -> float:
        """Calculate minutes security (0-10) using starts ratio"""
        current_gw = self.fpl.get_current_gameweek()
        if current_gw == 0:
            return 5

        avg_minutes = player.minutes / current_gw if current_gw > 0 else 0
        base = min(10, avg_minutes / 9)

        # Bonus for high starts ratio (nailed on)
        if current_gw > 3 and player.starts > 0:
            starts_ratio = player.starts / current_gw
            if starts_ratio >= 0.9:
                base = min(10, base + 1.0)  # Nailed bonus
            elif starts_ratio < 0.5:
                base *= 0.7  # Rotation risk

        return base

    def _calculate_set_piece_bonus(self, player: Player) -> float:
        """Bonus for set piece takers (0-2). Pens/FKs = extra goal routes."""
        bonus = 0.0
        if player.penalties_order is not None and player.penalties_order <= 1:
            bonus += 1.5  # Penalty taker is huge
        elif player.penalties_order is not None and player.penalties_order <= 2:
            bonus += 0.5  # Backup pen taker
        if (
            player.direct_freekicks_order is not None
            and player.direct_freekicks_order <= 1
        ):
            bonus += 0.3
        if (
            player.corners_and_indirect_freekicks_order is not None
            and player.corners_and_indirect_freekicks_order <= 1
        ):
            bonus += 0.2  # Corners = assist potential
        return bonus

    def _calculate_bonus_magnet_score(self, player: Player) -> float:
        """Score for BPS/bonus consistency (0-1). Bonus pts are free points."""
        current_gw = self.fpl.get_current_gameweek()
        if current_gw == 0 or player.minutes < 180:
            return 0

        bonus_per_game = player.bonus / current_gw
        # Top bonus magnets average ~0.8-1.5 bonus/game
        return min(1.0, bonus_per_game / 1.5)

    def _calculate_ep_next_score(self, player: Player) -> float:
        """Calculate score from FPL's own expected points prediction (0-10).
        ep_next typically ranges 0-8; this is the strongest single signal."""
        return min(10, player.ep_next * 1.5)

    def _calculate_defensive_score(self, player: Player) -> float:
        """Calculate defensive contribution score (0-10). GKP/DEF only."""
        if player.position not in ("GKP", "DEF"):
            return 0.0

        current_gw = self.fpl.get_current_gameweek()
        if current_gw == 0 or player.minutes < 180:
            return 0.0

        nineties = player.minutes / 90

        if player.position == "GKP":
            # Clean sheets + saves contribution
            saves_per_90 = player.saves / nineties if nineties > 0 else 0
            score = player.clean_sheets_per_90 * 12.5 + saves_per_90 * 0.6
        else:
            # DEF: clean sheets minus goals conceded penalty
            gc_per_90 = player.goals_conceded / nineties if nineties > 0 else 0
            score = player.clean_sheets_per_90 * 12.5 - gc_per_90 * 0.8

        return max(0, min(10, score))

    def _calculate_transfer_momentum(self, player: Player) -> float:
        """Calculate transfer momentum bonus (0-2). Market confidence signal.
        Position-aware: attackers get bigger boost from momentum."""
        net_transfers = player.transfers_in_event - player.transfers_out_event
        base = max(0, min(2.0, net_transfers / 50000))
        # Position multipliers: attackers benefit more from momentum
        pos_mult = {"FWD": 1.5, "MID": 1.2, "DEF": 0.8, "GKP": 0.5}.get(
            player.position, 1.0
        )
        return base * pos_mult

    def _calculate_ownership_score(self, player: Player) -> float:
        """Calculate ownership score (0-10). Wisdom of crowds signal.

        High ownership = many managers believe in this player.
        This is a strong baseline signal that captures collective intelligence.

        BACKTEST PROVEN: Linear scaling matches the ownership baseline that
        won the 20-GW comparison. Log scaling compresses differences too much.
        """
        # Ownership typically ranges 0-60% for most players
        # Top owned players: 50-60%, template players: 20-40%, differentials: <10%
        ownership = player.selected_by_percent

        # LINEAR scaling to preserve ownership signal strength
        # 50% ownership -> 8.3 score, 20% -> 3.3, 5% -> 0.83
        # This matches the raw ownership baseline that won the backtest
        score = ownership / 6.0  # 60% = 10.0 (max realistic ownership)

        return min(10, max(0, score))

    def _calculate_form_fixture_interaction(self, player: Player) -> float:
        """Calculate team form × fixture ease interaction (0-3).
        Strong teams with easy fixtures should score higher than the sum of parts.
        This captures contextual synergy the individual factors miss."""
        team = self.fpl.get_team(player.team_id)
        if not team or team.form is None:
            return 0.0

        # Normalize team form (typically 0-3 range) to 0-1
        team_form_norm = min(1.0, team.form / 3.0)

        # Get fixture ease (0-10) and normalize to 0-1
        fixture_ease = self.fixtures.get_fixture_ease_score(
            player.team_id, player.position
        )
        fixture_ease_norm = fixture_ease / 10.0

        # Interaction: when both are high, bonus is multiplicative
        # Example: team_form=0.8, fixture=0.8 -> interaction=0.64*3=1.92
        # But team_form=0.3, fixture=0.8 -> interaction=0.24*3=0.72
        interaction = team_form_norm * fixture_ease_norm * 3.0

        return interaction

    def _calculate_ownership_form_interaction(self, player: Player) -> float:
        """Calculate ownership × form interaction (0-3).
        Hot players that many people own tend to deliver consistently.
        This captures the 'template captain' effect."""
        ownership_norm = min(1.0, player.selected_by_percent / 50.0)  # 50% = max
        form_norm = min(1.0, player.form / 8.0)  # 8.0 form = elite

        # Interaction: popular + in-form = reliable points
        interaction = ownership_norm * form_norm * 3.0

        return interaction

    def _calculate_recent_points_score(self, player: Player) -> float:
        """Calculate recent actual points score (0-10).
        Uses event_points (last GW) as a 'hot hand' indicator.
        Players who just scored big often continue form."""
        # event_points is last GW actual points
        last_gw_pts = player.event_points

        # Scale: 15+ pts = 10, 10 pts = 6.7, 5 pts = 3.3, 0 = 0
        score = min(10, last_gw_pts / 1.5)

        # Blend with form for stability (70% recent, 30% form)
        form_component = min(10, player.form * 1.2)
        blended = score * 0.7 + form_component * 0.3

        return min(10, blended)

    def _calculate_ict_position_score(self, player: Player) -> float:
        """Calculate position-decomposed ICT score (0-10).
        FWD: threat-heavy, MID: creativity+threat, DEF: influence-heavy."""
        # Normalize raw ICT components (typically 0-1000+ over season)
        current_gw = self.fpl.get_current_gameweek()
        if current_gw == 0:
            return 0.0

        influence = player.influence / current_gw
        creativity = player.creativity / current_gw
        threat = player.threat / current_gw

        if player.position == "FWD":
            raw = threat * 0.6 + creativity * 0.25 + influence * 0.15
        elif player.position == "MID":
            raw = creativity * 0.4 + threat * 0.4 + influence * 0.2
        elif player.position == "DEF":
            raw = influence * 0.5 + creativity * 0.3 + threat * 0.2
        else:  # GKP
            raw = influence * 0.8 + creativity * 0.1 + threat * 0.1

        return min(10, raw / 5)

    def _calculate_availability_multiplier(self, player: Player) -> float:
        """Calculate availability as a probability multiplier (0.0-1.0).
        Uses chance_of_playing directly instead of crude binary thresholds."""
        if player.status in ("i", "s", "u"):
            return 0.0

        if player.chance_of_playing is not None:
            if player.chance_of_playing == 0:
                return 0.0
            return player.chance_of_playing / 100.0

        # Available with no flag
        return 1.0

    def score_player(self, player: Player) -> ScoredPlayer:
        """Calculate all scores for a player with position-aware weighting"""
        # Return cached result if available (same player scored multiple times per run)
        if player.id in self._score_cache:
            return self._score_cache[player.id]

        form_score = self._calculate_form_score(player)
        xgi_score = self._calculate_xgi_score(player)
        fixture_score = self._calculate_fixture_score(player)
        value_score = self._calculate_value_score(player)
        ict_score = self._calculate_ict_score(player)
        minutes_score = self._calculate_minutes_score(player)
        ep_next_score = self._calculate_ep_next_score(player)
        defensive_score = self._calculate_defensive_score(player)
        transfer_momentum = self._calculate_transfer_momentum(player)
        ict_position_score = self._calculate_ict_position_score(player)
        availability_mult = self._calculate_availability_multiplier(player)
        form_fixture_interaction = self._calculate_form_fixture_interaction(player)
        ownership_score = self._calculate_ownership_score(player)
        recent_points_score = self._calculate_recent_points_score(player)
        ownership_form_interaction = self._calculate_ownership_form_interaction(player)

        # Weight priority: 1) tuned weights from evaluator, 2) position-specific, 3) legacy
        pos_weights = getattr(config, "POSITION_WEIGHTS", {}).get(player.position)

        # Build factor dict (same structure for all weight systems)
        factors = {
            "ep_next": ep_next_score,
            "form": form_score,
            "xgi_per_90": xgi_score,
            "fixture_ease": fixture_score,
            "defensive": defensive_score,
            "ict_position": ict_position_score,
            "value_score": value_score,
            "minutes_security": minutes_score,
            "ownership": ownership_score,  # Wisdom of crowds signal
            "recent_points": recent_points_score,  # Hot hand indicator
        }

        # Select weights: tuned > position-specific > legacy
        if self.tuned_weights:
            # Use tuned weights (from evaluator), fill missing with position weights
            weights = {}
            for k in factors:
                if k in self.tuned_weights:
                    weights[k] = self.tuned_weights[k]
                elif pos_weights and k in pos_weights:
                    weights[k] = pos_weights[k]
                else:
                    weights[k] = 0.1  # Default fallback
            # Normalize to sum to 1.0
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
        elif pos_weights:
            weights = pos_weights
        else:
            # Fall back to legacy weights with legacy factors
            factors = {
                "form": form_score,
                "xgi_per_90": xgi_score,
                "fixture_ease": fixture_score,
                "value_score": value_score,
                "ict_index": ict_score,
                "minutes_security": minutes_score,
            }
            weights = self.weights

        # Calculate interaction weights from config
        interaction_weight = getattr(config, "TEAM_FORM_INTERACTION_WEIGHT", 0.15)
        ownership_form_weight = getattr(
            config, "OWNERSHIP_FORM_INTERACTION_WEIGHT", 0.20
        )
        pure_ownership_mode = getattr(config, "PURE_OWNERSHIP_MODE", False)

        # BACKTEST PROVEN: Pure ownership beats all bonus systems
        if pure_ownership_mode:
            bonuses = {}  # No bonuses in pure ownership mode
        else:
            bonuses = {
                "set_piece": self._calculate_set_piece_bonus(player),
                "bonus_magnet": self._calculate_bonus_magnet_score(player),
                "transfer_momentum": transfer_momentum,
                "form_fixture_interaction": form_fixture_interaction
                * interaction_weight,
                "ownership_form_interaction": ownership_form_interaction
                * ownership_form_weight,
            }

        overall_score = compute_weighted_score(factors, weights, bonuses)

        availability, injury_details = self._get_player_availability(player)

        # Availability penalty: injured/suspended = 0, doubt = continuous probability
        # No artificial floor - injured players should not compete for recommendations
        if availability in ("injured", "suspended"):
            overall_score = 0.0  # Completely unavailable
        elif availability == "doubt":
            # Use continuous probability from chance_of_playing
            # 75% chance -> 0.75x, 50% chance -> 0.5x, 25% chance -> 0.25x
            overall_score *= availability_mult

        result = ScoredPlayer(
            player=player,
            overall_score=overall_score,
            form_score=form_score,
            xgi_score=xgi_score,
            fixture_score=fixture_score,
            value_score=value_score,
            ict_score=ict_score,
            minutes_score=minutes_score,
            availability=availability,
            injury_details=injury_details,
            ep_next_score=ep_next_score,
            defensive_score=defensive_score,
            transfer_momentum=transfer_momentum,
            availability_multiplier=availability_mult,
            form_fixture_interaction=form_fixture_interaction,
            ownership_score=ownership_score,
            recent_points_score=recent_points_score,
            ownership_form_interaction=ownership_form_interaction,
        )
        self._score_cache[player.id] = result
        return result

    def get_top_players(
        self,
        position: Optional[str] = None,
        max_price: Optional[float] = None,
        limit: int = 20,
        exclude_unavailable: bool = True,
    ) -> List[ScoredPlayer]:
        """Get top-ranked players"""
        if position:
            players = self.fpl.get_players_by_position(position)
        else:
            players = self.fpl.get_all_players()

        # Filter by price
        if max_price:
            players = [p for p in players if p.price <= max_price]

        # Score all players
        scored = [self.score_player(p) for p in players]

        # Filter unavailable
        if exclude_unavailable:
            scored = [s for s in scored if s.availability in ("fit", "doubt")]

        # Sort by score
        scored.sort(key=lambda x: x.overall_score, reverse=True)

        return scored[:limit]

    def get_top_by_position(
        self, limit: int = 10, max_price: Optional[float] = None
    ) -> Dict[str, List[ScoredPlayer]]:
        """Get top players for each position"""
        positions = ["GKP", "DEF", "MID", "FWD"]
        result = {}

        for pos in positions:
            result[pos] = self.get_top_players(
                position=pos, max_price=max_price, limit=limit
            )

        return result

    def compare_players(
        self, player_a_id: int, player_b_id: int
    ) -> Tuple[ScoredPlayer, ScoredPlayer]:
        """Compare two players side by side"""
        player_a = self.fpl.get_player(player_a_id)
        player_b = self.fpl.get_player(player_b_id)

        if not player_a or not player_b:
            raise ValueError("Player not found")

        return self.score_player(player_a), self.score_player(player_b)

    def format_scored_player(self, sp: ScoredPlayer, detailed: bool = False) -> str:
        """Format a scored player for display"""
        avail_icon = {
            "fit": "✓",
            "doubt": "⚠️",
            "injured": "❌",
            "suspended": "🚫",
        }.get(sp.availability, "?")

        base = (
            f"{avail_icon} {sp.player.web_name} ({sp.player.team}) "
            f"£{sp.player.price}m - Score: {sp.overall_score:.2f}"
        )

        if detailed:
            base += (
                f"\n   Form: {sp.form_score:.1f} | xGI: {sp.xgi_score:.1f} | "
                f"Fix: {sp.fixture_score:.1f} | Val: {sp.value_score:.1f}"
            )
            if sp.injury_details:
                base += f"\n   ⚠️ {sp.injury_details}"

        return base
