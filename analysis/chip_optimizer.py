"""
Chip strategy optimizer - recommends when to use FPL chips
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from data.fpl_api import FPLDataFetcher
from analysis.fixture_analyzer import FixtureAnalyzer

import config


class Chip(Enum):
    """FPL Chips"""

    WILDCARD = "wildcard"
    FREE_HIT = "free_hit"
    TRIPLE_CAPTAIN = "triple_captain"
    BENCH_BOOST = "bench_boost"


@dataclass
class ChipRecommendation:
    """A chip usage recommendation"""

    chip: Chip
    recommended_gw: Optional[int]
    reason: str
    priority: int  # 1 = use now, 2 = plan for soon, 3 = hold
    details: str


@dataclass
class ChipStrategy:
    """Overall chip strategy"""

    current_gw: int
    remaining_gws: int
    recommendations: List[ChipRecommendation]
    dgw_info: Dict[int, List[str]]  # GW -> team names
    bgw_info: Dict[int, List[str]]  # GW -> team names
    summary: str


class ChipOptimizer:
    """Optimizes chip usage strategy"""

    def __init__(self, fpl_data: FPLDataFetcher, fixture_analyzer: FixtureAnalyzer):
        self.fpl = fpl_data
        self.fixtures = fixture_analyzer
        self.expected_bgw = config.EXPECTED_BGW
        self.expected_dgw = config.EXPECTED_DGW

    def _get_upcoming_dgw_bgw(
        self,
    ) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
        """Get confirmed and expected DGW/BGW info"""
        current_gw = self.fpl.get_current_gameweek()

        # Get confirmed from API
        dgw_teams = self.fixtures.get_double_gameweeks()
        bgw_teams = self.fixtures.get_blank_gameweeks()

        # Convert team IDs to names
        dgw_info = {}
        for gw, team_ids in dgw_teams.items():
            if gw >= current_gw:
                teams = [
                    self.fpl.get_team(tid).short_name
                    for tid in team_ids
                    if self.fpl.get_team(tid)
                ]
                if teams:
                    dgw_info[gw] = teams

        bgw_info = {}
        for gw, team_ids in bgw_teams.items():
            if gw >= current_gw:
                teams = [
                    self.fpl.get_team(tid).short_name
                    for tid in team_ids
                    if self.fpl.get_team(tid)
                ]
                if teams:
                    bgw_info[gw] = teams

        # Add expected DGW/BGW if not yet confirmed
        for gw in self.expected_dgw:
            if gw >= current_gw and gw not in dgw_info:
                dgw_info[gw] = ["TBC"]  # To be confirmed

        for gw in self.expected_bgw:
            if gw >= current_gw and gw not in bgw_info:
                bgw_info[gw] = ["TBC"]

        return dgw_info, bgw_info

    def _recommend_wildcard(
        self,
        available: bool,
        current_gw: int,
        dgw_info: Dict,
        bgw_info: Dict,
    ) -> ChipRecommendation:
        """Generate wildcard recommendation"""
        if not available:
            return ChipRecommendation(
                chip=Chip.WILDCARD,
                recommended_gw=None,
                reason="Already used",
                priority=3,
                details="Wildcard has been played",
            )

        # Find the best time - usually before a DGW swing
        upcoming_dgws = sorted([gw for gw in dgw_info.keys() if gw > current_gw])

        if upcoming_dgws:
            # Recommend using 1-2 weeks before DGW to prepare
            target_gw = upcoming_dgws[0] - 1
            if target_gw <= current_gw:
                target_gw = upcoming_dgws[0]

            return ChipRecommendation(
                chip=Chip.WILDCARD,
                recommended_gw=target_gw,
                reason=f"Prepare for DGW{upcoming_dgws[0]}",
                priority=2 if target_gw > current_gw + 2 else 1,
                details=(
                    f"Use wildcard in GW{target_gw} to load up on DGW players. "
                    f"Target teams: {', '.join(dgw_info.get(upcoming_dgws[0], ['TBC']))}"
                ),
            )

        # No DGW insight - recommend based on squad needs
        return ChipRecommendation(
            chip=Chip.WILDCARD,
            recommended_gw=None,
            reason="Save for fixture swing or squad crisis",
            priority=3,
            details=(
                "Hold wildcard until DGWs are confirmed or squad needs major overhaul. "
                "Don't waste on small fixes - use free transfers."
            ),
        )

    def _recommend_free_hit(
        self,
        available: bool,
        current_gw: int,
        dgw_info: Dict,
        bgw_info: Dict,
    ) -> ChipRecommendation:
        """Generate free hit recommendation"""
        if not available:
            return ChipRecommendation(
                chip=Chip.FREE_HIT,
                recommended_gw=None,
                reason="Already used",
                priority=3,
                details="Free Hit has been played",
            )

        # Free Hit is best for BGWs
        upcoming_bgws = sorted([gw for gw in bgw_info.keys() if gw > current_gw])

        if upcoming_bgws:
            target_gw = upcoming_bgws[0]
            blanking_teams = bgw_info.get(target_gw, ["TBC"])

            return ChipRecommendation(
                chip=Chip.FREE_HIT,
                recommended_gw=target_gw,
                reason=f"Navigate BGW{target_gw}",
                priority=2 if target_gw > current_gw + 3 else 1,
                details=(
                    f"Use Free Hit in GW{target_gw} to field 11 playing players. "
                    f"Teams blanking: {', '.join(blanking_teams)}"
                ),
            )

        # No BGW - could use for big DGW
        upcoming_dgws = sorted([gw for gw in dgw_info.keys() if gw > current_gw])
        if upcoming_dgws:
            target_gw = upcoming_dgws[-1]  # Later DGW
            return ChipRecommendation(
                chip=Chip.FREE_HIT,
                recommended_gw=target_gw,
                reason=f"Maximize DGW{target_gw}",
                priority=3,
                details=(
                    f"Could use Free Hit in DGW{target_gw} to build optimal one-week team. "
                    "But BGW usage is typically better value."
                ),
            )

        return ChipRecommendation(
            chip=Chip.FREE_HIT,
            recommended_gw=None,
            reason="Save for BGW",
            priority=3,
            details="Hold Free Hit for blank gameweeks when many teams don't play.",
        )

    def _recommend_triple_captain(
        self,
        available: bool,
        current_gw: int,
        dgw_info: Dict,
    ) -> ChipRecommendation:
        """Generate triple captain recommendation"""
        if not available:
            return ChipRecommendation(
                chip=Chip.TRIPLE_CAPTAIN,
                recommended_gw=None,
                reason="Already used",
                priority=3,
                details="Triple Captain has been played",
            )

        # TC is best on premium player in DGW
        upcoming_dgws = sorted([gw for gw in dgw_info.keys() if gw > current_gw])

        if upcoming_dgws:
            # Find DGW with best premium options
            for gw in upcoming_dgws:
                teams = dgw_info.get(gw, [])
                # Check if premium teams have DGW (e.g., Liverpool, Man City, Arsenal)
                premium_teams = {"LIV", "MCI", "ARS", "CHE", "MUN", "TOT"}
                has_premium = any(t in premium_teams for t in teams) or "TBC" in teams

                if has_premium:
                    return ChipRecommendation(
                        chip=Chip.TRIPLE_CAPTAIN,
                        recommended_gw=gw,
                        reason=f"Premium double in DGW{gw}",
                        priority=2 if gw > current_gw + 3 else 1,
                        details=(
                            f"Use TC in DGW{gw} on Salah/Haaland/top premium with two good fixtures. "
                            f"DGW teams: {', '.join(teams)}"
                        ),
                    )

            # No clear premium DGW yet
            return ChipRecommendation(
                chip=Chip.TRIPLE_CAPTAIN,
                recommended_gw=upcoming_dgws[-1],
                reason="Save for best DGW",
                priority=3,
                details=(
                    f"Hold TC for a DGW where premiums have two favorable fixtures. "
                    f"Upcoming DGWs: {', '.join(map(str, upcoming_dgws))}"
                ),
            )

        return ChipRecommendation(
            chip=Chip.TRIPLE_CAPTAIN,
            recommended_gw=None,
            reason="Save for DGW",
            priority=3,
            details="Hold Triple Captain for double gameweeks with premium player options.",
        )

    def _recommend_bench_boost(
        self,
        available: bool,
        current_gw: int,
        dgw_info: Dict,
    ) -> ChipRecommendation:
        """Generate bench boost recommendation"""
        if not available:
            return ChipRecommendation(
                chip=Chip.BENCH_BOOST,
                recommended_gw=None,
                reason="Already used",
                priority=3,
                details="Bench Boost has been played",
            )

        # BB is best in DGW with strong bench (often after WC)
        upcoming_dgws = sorted([gw for gw in dgw_info.keys() if gw > current_gw])

        if upcoming_dgws:
            # Recommend first big DGW
            target_gw = upcoming_dgws[0]
            teams = dgw_info.get(target_gw, ["TBC"])

            return ChipRecommendation(
                chip=Chip.BENCH_BOOST,
                recommended_gw=target_gw,
                reason=f"15 doubles in DGW{target_gw}",
                priority=2 if target_gw > current_gw + 3 else 1,
                details=(
                    f"Use BB in DGW{target_gw} with 15 players who all double. "
                    f"Best paired with Wildcard the week before. "
                    f"DGW teams: {', '.join(teams)}"
                ),
            )

        return ChipRecommendation(
            chip=Chip.BENCH_BOOST,
            recommended_gw=None,
            reason="Save for DGW",
            priority=3,
            details=(
                "Hold Bench Boost for double gameweek. "
                "Maximize by using Wildcard the week before to build a full 15-player DGW squad."
            ),
        )

    def get_chip_strategy(
        self,
        available_chips: Optional[List[str]] = None,
    ) -> ChipStrategy:
        """Generate full chip strategy"""
        current_gw = self.fpl.get_current_gameweek()
        remaining_gws = 38 - current_gw

        # Default to all chips available
        if available_chips is None:
            available_chips = ["wildcard", "free_hit", "triple_captain", "bench_boost"]

        available_set = set(c.lower() for c in available_chips)

        # Get DGW/BGW info
        dgw_info, bgw_info = self._get_upcoming_dgw_bgw()

        # Generate recommendations
        recommendations = [
            self._recommend_wildcard(
                "wildcard" in available_set, current_gw, dgw_info, bgw_info
            ),
            self._recommend_free_hit(
                "free_hit" in available_set, current_gw, dgw_info, bgw_info
            ),
            self._recommend_triple_captain(
                "triple_captain" in available_set, current_gw, dgw_info
            ),
            self._recommend_bench_boost(
                "bench_boost" in available_set, current_gw, dgw_info
            ),
        ]

        # Sort by priority
        recommendations.sort(key=lambda x: x.priority)

        # Generate summary
        summary = self._generate_summary(
            current_gw, remaining_gws, recommendations, dgw_info, bgw_info
        )

        return ChipStrategy(
            current_gw=current_gw,
            remaining_gws=remaining_gws,
            recommendations=recommendations,
            dgw_info=dgw_info,
            bgw_info=bgw_info,
            summary=summary,
        )

    def _generate_summary(
        self,
        current_gw: int,
        remaining_gws: int,
        recommendations: List[ChipRecommendation],
        dgw_info: Dict,
        bgw_info: Dict,
    ) -> str:
        """Generate human-readable strategy summary"""
        lines = [
            f"CHIP STRATEGY (GW{current_gw} - {remaining_gws} GWs remaining)",
            "=" * 50,
        ]

        # DGW/BGW overview
        if dgw_info:
            lines.append("\nDouble Gameweeks:")
            for gw, teams in sorted(dgw_info.items()):
                lines.append(f"   GW{gw}: {', '.join(teams)}")

        if bgw_info:
            lines.append("\nBlank Gameweeks:")
            for gw, teams in sorted(bgw_info.items()):
                lines.append(f"   GW{gw}: {', '.join(teams)} blanking")

        # Recommendations
        lines.append("\nCHIP RECOMMENDATIONS:")
        for rec in recommendations:
            priority_label = {1: "USE NOW", 2: "PLAN FOR", 3: "HOLD"}.get(
                rec.priority, "?"
            )
            gw_str = f"GW{rec.recommended_gw}" if rec.recommended_gw else "TBD"
            lines.append(f"\n{rec.chip.value.upper()} [{priority_label}] - {gw_str}")
            lines.append(f"   {rec.reason}")
            lines.append(f"   → {rec.details}")

        # Comeback strategy note
        lines.append("\n" + "-" * 50)
        lines.append("COMEBACK TIP: With all chips available, plan a chip sequence:")
        lines.append("   1. WC before first big DGW to prepare")
        lines.append("   2. BB in that DGW with 15 doublers")
        lines.append("   3. FH in the main BGW to avoid blanks")
        lines.append("   4. TC on premium in later DGW")

        return "\n".join(lines)

    def format_recommendation(self, rec: ChipRecommendation) -> str:
        """Format a single recommendation for display"""
        priority_icon = {1: "🔥", 2: "📅", 3: "💤"}.get(rec.priority, "?")
        gw_str = f"GW{rec.recommended_gw}" if rec.recommended_gw else "Hold"

        return (
            f"{priority_icon} {rec.chip.value.upper()} → {gw_str}\n"
            f"   {rec.reason}\n"
            f"   {rec.details}"
        )
