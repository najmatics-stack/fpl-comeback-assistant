"""
Differential finder - identifies low-ownership high-potential players
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from data.fpl_api import FPLDataFetcher
from analysis.fixture_analyzer import FixtureAnalyzer
from analysis.player_scorer import PlayerScorer, ScoredPlayer

import config


@dataclass
class Differential:
    """A differential player recommendation"""

    scored_player: ScoredPlayer
    differential_score: float  # Expected points / ownership
    ownership: float
    upside_reason: str


class DifferentialFinder:
    """Finds low-ownership players with high expected returns"""

    def __init__(
        self,
        fpl_data: FPLDataFetcher,
        player_scorer: PlayerScorer,
        fixture_analyzer: FixtureAnalyzer,
    ):
        self.fpl = fpl_data
        self.scorer = player_scorer
        self.fixtures = fixture_analyzer

        # Thresholds from config
        self.max_ownership = config.DIFFERENTIAL_MAX_OWNERSHIP
        self.min_form = config.DIFFERENTIAL_MIN_FORM
        self.min_minutes = config.DIFFERENTIAL_MIN_MINUTES

    def _calculate_differential_score(self, sp: ScoredPlayer) -> float:
        """
        Calculate differential score: expected value / ownership
        Higher score = better differential
        """
        # Avoid division by zero, add 1 to ownership
        ownership_factor = sp.player.selected_by_percent + 1
        return sp.overall_score / ownership_factor * 10

    def _get_upside_reason(self, sp: ScoredPlayer) -> str:
        """Generate explanation for why player is a good differential"""
        reasons = []

        # Check form
        if sp.player.form >= 5.0:
            reasons.append(f"hot form ({sp.player.form})")
        elif sp.player.form >= 4.0:
            reasons.append(f"good form ({sp.player.form})")

        # Check fixtures
        fixture_run = self.fixtures.get_fixture_run(sp.player.team_id)
        if fixture_run.avg_difficulty <= 2.5:
            reasons.append("easy fixtures")
        if fixture_run.has_double:
            reasons.append("DGW incoming")

        # Check value
        ppg = sp.player.points_per_game
        if ppg > 5:
            reasons.append(f"high PPG ({ppg})")

        # Check xG metrics
        if sp.player.minutes > 0:
            xgi_per_90 = (sp.player.expected_goal_involvements / sp.player.minutes) * 90
            if xgi_per_90 > 0.5:
                reasons.append(f"good xGI ({xgi_per_90:.2f}/90)")

        # Price bracket analysis
        if sp.player.position in ("MID", "FWD"):
            if sp.player.price < 7.0 and sp.player.form > 4:
                reasons.append("budget enabler")
            elif sp.player.price >= 10.0 and sp.overall_score > 6:
                reasons.append("premium pick")

        return ", ".join(reasons) if reasons else "under the radar"

    def find_differentials(
        self,
        position: Optional[str] = None,
        max_price: Optional[float] = None,
        limit: int = 5,
    ) -> List[Differential]:
        """Find top differential picks"""
        # Get all players
        if position:
            players = self.fpl.get_players_by_position(position)
        else:
            players = self.fpl.get_all_players()

        # Apply filters
        filtered = []
        for p in players:
            # Ownership filter
            if p.selected_by_percent > self.max_ownership:
                continue

            # Form filter
            if p.form < self.min_form:
                continue

            # Minutes filter (needs regular game time)
            current_gw = self.fpl.get_current_gameweek()
            if current_gw > 0:
                avg_minutes = p.minutes / current_gw
                if avg_minutes < self.min_minutes:
                    continue

            # Price filter
            if max_price and p.price > max_price:
                continue

            # Must be available
            if p.status not in ("a", "d"):  # available or doubtful only
                continue

            filtered.append(p)

        # Score players
        differentials = []
        for p in filtered:
            sp = self.scorer.score_player(p)

            # Skip injured/suspended
            if sp.availability in ("injured", "suspended"):
                continue

            diff_score = self._calculate_differential_score(sp)
            reason = self._get_upside_reason(sp)

            differentials.append(
                Differential(
                    scored_player=sp,
                    differential_score=diff_score,
                    ownership=p.selected_by_percent,
                    upside_reason=reason,
                )
            )

        # Sort by differential score
        differentials.sort(key=lambda x: x.differential_score, reverse=True)

        return differentials[:limit]

    def find_differentials_by_position(
        self, limit_per_position: int = 5
    ) -> Dict[str, List[Differential]]:
        """Find top differentials for each position"""
        positions = ["GKP", "DEF", "MID", "FWD"]
        result = {}

        for pos in positions:
            result[pos] = self.find_differentials(
                position=pos, limit=limit_per_position
            )

        return result

    def find_punt_picks(
        self, max_price: float = 5.5, limit: int = 10
    ) -> List[Differential]:
        """Find cheap punt options (budget enablers)"""
        # Lower form requirement for punts
        original_min_form = self.min_form
        self.min_form = 3.0

        # Increase ownership ceiling slightly
        original_max_ownership = self.max_ownership
        self.max_ownership = 15.0

        punts = self.find_differentials(max_price=max_price, limit=limit)

        # Restore original thresholds
        self.min_form = original_min_form
        self.max_ownership = original_max_ownership

        return punts

    def format_differential(self, diff: Differential) -> str:
        """Format differential for display"""
        sp = diff.scored_player
        p = sp.player

        return (
            f"{p.web_name} ({p.team} {p.position}) - £{p.price}m\n"
            f"   Ownership: {diff.ownership:.1f}% | Form: {p.form} | "
            f"Score: {sp.overall_score:.2f}\n"
            f"   Why: {diff.upside_reason}"
        )

    def get_differential_summary(self) -> str:
        """Get formatted summary of top differentials"""
        diffs_by_pos = self.find_differentials_by_position(limit_per_position=3)

        lines = ["TOP DIFFERENTIALS BY POSITION", "=" * 40]

        for pos, diffs in diffs_by_pos.items():
            lines.append(f"\n{pos}:")
            if not diffs:
                lines.append("   No qualifying differentials found")
            else:
                for diff in diffs:
                    p = diff.scored_player.player
                    lines.append(
                        f"   {p.web_name} ({p.team}) £{p.price}m - "
                        f"{diff.ownership:.1f}% owned - {diff.upside_reason}"
                    )

        return "\n".join(lines)
