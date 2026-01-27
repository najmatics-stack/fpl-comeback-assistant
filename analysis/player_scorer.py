"""
Player scoring algorithm for ranking and recommendations
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from data.fpl_api import FPLDataFetcher, Player
from data.news_scraper import NewsScraper, InjuryStatus
from analysis.fixture_analyzer import FixtureAnalyzer

import config


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


class PlayerScorer:
    """Scores and ranks players for FPL recommendations"""

    def __init__(
        self,
        fpl_data: FPLDataFetcher,
        fixture_analyzer: FixtureAnalyzer,
        news_scraper: Optional[NewsScraper] = None,
    ):
        self.fpl = fpl_data
        self.fixtures = fixture_analyzer
        self.news = news_scraper
        self.weights = config.SCORING_WEIGHTS
        self._injuries: Optional[Dict[str, InjuryStatus]] = None

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
        # Form is typically 0-10 in FPL
        return min(10, player.form)

    def _calculate_xgi_score(self, player: Player) -> float:
        """Calculate xGI per 90 score (0-10)"""
        if player.minutes < 90:
            return 0

        xgi_per_90 = (player.expected_goal_involvements / player.minutes) * 90

        # Normalize: 0.8+ xGI/90 = 10, scale down from there
        # Top attackers average ~0.6-0.8 xGI/90
        return min(10, xgi_per_90 * 12.5)

    def _calculate_fixture_score(self, player: Player) -> float:
        """Calculate fixture ease score (0-10)"""
        return self.fixtures.get_fixture_ease_score(player.team_id)

    def _calculate_value_score(self, player: Player) -> float:
        """Calculate value score (points per million) (0-10)"""
        if player.price <= 0:
            return 0

        points_per_million = player.total_points / player.price

        # Good value is ~15+ points per million, scale to 0-10
        return min(10, points_per_million / 1.5)

    def _calculate_ict_score(self, player: Player) -> float:
        """Calculate ICT index score (0-10)"""
        # ICT index typically ranges 0-500+ for top players
        # Normalize to 0-10
        return min(10, player.ict_index / 50)

    def _calculate_minutes_score(self, player: Player) -> float:
        """Calculate minutes security score (0-10)"""
        current_gw = self.fpl.get_current_gameweek()
        if current_gw == 0:
            return 5  # Unknown

        # Calculate average minutes per game
        games_played = current_gw  # Approximate
        avg_minutes = player.minutes / games_played if games_played > 0 else 0

        # 90 minutes = 10, scale down
        return min(10, avg_minutes / 9)

    def score_player(self, player: Player) -> ScoredPlayer:
        """Calculate all scores for a player"""
        form_score = self._calculate_form_score(player)
        xgi_score = self._calculate_xgi_score(player)
        fixture_score = self._calculate_fixture_score(player)
        value_score = self._calculate_value_score(player)
        ict_score = self._calculate_ict_score(player)
        minutes_score = self._calculate_minutes_score(player)

        # Weighted overall score
        overall_score = (
            form_score * self.weights["form"]
            + xgi_score * self.weights["xgi_per_90"]
            + fixture_score * self.weights["fixture_ease"]
            + value_score * self.weights["value_score"]
            + ict_score * self.weights["ict_index"]
            + minutes_score * self.weights["minutes_security"]
        )

        availability, injury_details = self._get_player_availability(player)

        # Penalize unavailable players
        if availability == "injured" or availability == "suspended":
            overall_score *= 0.1
        elif availability == "doubt":
            overall_score *= 0.7

        return ScoredPlayer(
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
        )

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
