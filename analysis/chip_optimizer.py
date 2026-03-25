"""
Chip strategy optimizer - recommends when to use FPL chips
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from data.fpl_api import FPLDataFetcher
from analysis.fixture_analyzer import FixtureAnalyzer

import config

if TYPE_CHECKING:
    from analysis.player_scorer import PlayerScorer, ScoredPlayer
    from analysis.league_spy import LeagueIntel


class Chip(Enum):
    """FPL Chips"""

    WILDCARD = "wildcard"
    FREE_HIT = "free_hit"
    TRIPLE_CAPTAIN = "triple_captain"
    BENCH_BOOST = "bench_boost"


@dataclass
class ChipPlan:
    """Forward-looking chip deployment plan for the season run-in"""

    assignments: List[Tuple[int, str]]  # (GW, chip_name) ordered pairs
    total_expected_gain: float
    per_chip_reasoning: Dict[str, str]  # chip_name -> reasoning
    constraints: List[str]  # constraints that were applied


@dataclass
class ChipRecommendation:
    """A chip usage recommendation"""

    chip: Chip
    recommended_gw: Optional[int]
    reason: str
    priority: int  # 1 = use now, 2 = plan for soon, 3 = hold
    details: str


@dataclass
class ContextualChipRec:
    """A chip recommendation with this-week scoring context"""

    chip: Chip
    recommended_gw: Optional[int]
    reason: str
    priority: int  # 1=USE NOW, 2=PLAN FOR, 3=HOLD
    details: str
    this_week_score: float  # 0.0–10.0: how good is THIS GW for this chip
    this_week_factors: List[str] = field(default_factory=list)


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
        """Get confirmed and expected DGW/BGW info.

        Uses confirmed fixture data from the API as the primary source.
        Falls back to config.EXPECTED_BGW/DGW only for GWs where fixtures
        haven't been scheduled yet. Unscheduled fixtures (event=None) indicate
        postponed matches that will likely create future DGWs.
        """
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

        # Add expected DGW/BGW from config only if not yet confirmed by API
        # AND the GW doesn't already have a full 10-fixture slate (which
        # contradicts the expectation).
        unscheduled = self.fpl.get_unscheduled_fixture_count()
        tbc_label = "TBC" if unscheduled == 0 else f"TBC ({unscheduled} unscheduled)"

        # Count confirmed fixtures per GW to detect full slates
        gw_fixture_counts = self.fpl.get_fixture_counts_by_gw()

        for gw in self.expected_dgw:
            if gw >= current_gw and gw not in dgw_info:
                # Don't add TBC if GW already has 10 fixtures (full slate)
                if gw_fixture_counts.get(gw, 0) >= 10:
                    continue
                dgw_info[gw] = [tbc_label]

        for gw in self.expected_bgw:
            if gw >= current_gw and gw not in bgw_info:
                # Don't add TBC if GW already has 10 fixtures (no blanks)
                if gw_fixture_counts.get(gw, 0) >= 10:
                    continue
                bgw_info[gw] = [tbc_label]

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

        # No DGW insight - recommend based on squad needs and season stage
        remaining = 38 - current_gw
        unscheduled = self.fpl.get_unscheduled_fixture_count()

        if remaining <= 8 and unscheduled > 0:
            return ChipRecommendation(
                chip=Chip.WILDCARD,
                recommended_gw=None,
                reason=f"Use before DGW if announced ({unscheduled} match TBD)",
                priority=2,
                details=(
                    f"{unscheduled} unscheduled fixture(s) may create a DGW. "
                    "If a DGW is announced, use WC the week before to load doublers, "
                    "then BB in the DGW itself. Otherwise use for a fixture swing."
                ),
            )
        elif remaining <= 5:
            return ChipRecommendation(
                chip=Chip.WILDCARD,
                recommended_gw=None,
                reason="Use soon — season ending",
                priority=2,
                details=(
                    f"Only {remaining} GWs left. Use WC for a major squad overhaul "
                    "targeting the best remaining fixtures. Don't let it expire."
                ),
            )
        else:
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

        # No DGWs on the horizon — adjust advice based on how many GWs remain
        remaining = 38 - current_gw
        unscheduled = self.fpl.get_unscheduled_fixture_count()

        if remaining <= 8 and unscheduled == 0:
            # Season is ending and no DGW is coming — use on best single-fixture week
            return ChipRecommendation(
                chip=Chip.TRIPLE_CAPTAIN,
                recommended_gw=None,
                reason="No DGW left — use on best captain fixture",
                priority=2,
                details=(
                    f"No DGWs remaining and only {remaining} GWs left. "
                    "Use TC on a week where your best premium has an easy home fixture. "
                    "Don't let it expire unused."
                ),
            )
        elif remaining <= 8 and unscheduled > 0:
            return ChipRecommendation(
                chip=Chip.TRIPLE_CAPTAIN,
                recommended_gw=None,
                reason=f"Wait for DGW ({unscheduled} match TBD)",
                priority=3,
                details=(
                    f"{unscheduled} unscheduled fixture(s) may create a small DGW. "
                    "If a DGW is announced, use TC there. Otherwise use on best single fixture."
                ),
            )
        else:
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

        # No DGWs on the horizon — adjust advice based on remaining season
        remaining = 38 - current_gw
        unscheduled = self.fpl.get_unscheduled_fixture_count()

        if remaining <= 8 and unscheduled == 0:
            return ChipRecommendation(
                chip=Chip.BENCH_BOOST,
                recommended_gw=None,
                reason="No DGW left — use when bench is strong",
                priority=2,
                details=(
                    f"No DGWs remaining and only {remaining} GWs left. "
                    "Use BB on a week when your bench has good fixtures. "
                    "Pair with WC if your bench needs upgrading first."
                ),
            )
        elif remaining <= 8 and unscheduled > 0:
            return ChipRecommendation(
                chip=Chip.BENCH_BOOST,
                recommended_gw=None,
                reason=f"Wait for DGW ({unscheduled} match TBD)",
                priority=3,
                details=(
                    f"{unscheduled} unscheduled fixture(s) may create a small DGW. "
                    "If announced, use BB there (ideally after WC to maximize bench). "
                    "Otherwise use on a week with strong bench fixtures."
                ),
            )
        else:
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

        # Short-circuit when all chips have been used
        if not available_chips:
            dgw_info, bgw_info = self._get_upcoming_dgw_bgw()
            return ChipStrategy(
                current_gw=current_gw,
                remaining_gws=remaining_gws,
                recommendations=[],
                dgw_info=dgw_info,
                bgw_info=bgw_info,
                summary="All chips have been used.",
            )

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

    # ------------------------------------------------------------------
    # Contextual this-week scoring (for interactive auto mode)
    # ------------------------------------------------------------------

    def _score_wc_this_week(
        self,
        squad_health: Dict[str, int],
        scored_players: List[ScoredPlayer],
        dgw_info: Dict[int, List[str]],
        current_gw: int,
        remaining_gws: int,
        league_gap: Optional[int],
    ) -> Tuple[float, List[str]]:
        """Score how good THIS week is for Wildcard (0-10)."""
        score = 0.0
        factors: List[str] = []

        # Injury pressure (0-4): injured * 1.5 + doubt * 0.5, capped at 4
        injury_pts = min(
            squad_health["injured"] * 1.5 + squad_health["doubt"] * 0.5, 4.0
        )
        if injury_pts > 0:
            score += injury_pts
            factors.append(
                f"{squad_health['injured']} injured + {squad_health['doubt']} doubtful"
            )

        # DGW prep (0-3): 3 if DGW 1-2 weeks away, 1.5 if 3-4 weeks
        upcoming_dgws = sorted(gw for gw in dgw_info if gw > current_gw)
        if upcoming_dgws:
            gap = upcoming_dgws[0] - current_gw
            if gap <= 2:
                score += 3.0
                factors.append(f"DGW{upcoming_dgws[0]} in {gap} week(s) — rebuild now")
            elif gap <= 4:
                score += 1.5
                factors.append(f"DGW{upcoming_dgws[0]} in {gap} weeks — prep soon")

        # Squad weakness (0-2): average score below threshold
        if scored_players:
            avg_score = sum(sp.overall_score for sp in scored_players) / len(
                scored_players
            )
            if avg_score < 5.0:
                weakness_pts = min((5.0 - avg_score) * 1.0, 2.0)
                score += weakness_pts
                factors.append(f"Squad avg score {avg_score:.1f} (below par)")

        # Urgency (0-1): <8 GWs left or large points gap
        if remaining_gws < 8:
            score += 0.5
            factors.append(f"Only {remaining_gws} GWs left")
        if league_gap and league_gap > 150:
            score += 0.5
            factors.append(f"{league_gap} pts behind — need overhaul")

        return min(score, 10.0), factors

    def _score_fh_this_week(
        self,
        squad_ids: List[int],
        scored_players: List[ScoredPlayer],
        current_gw: int,
        dgw_raw: Dict[int, List[int]],
        bgw_raw: Dict[int, List[int]],
    ) -> Tuple[float, List[str]]:
        """Score how good THIS week is for Free Hit (0-10)."""
        score = 0.0
        factors: List[str] = []

        # Blank exposure (0-5): squad players whose team blanks THIS GW
        blanking_teams = set(bgw_raw.get(current_gw, []))
        if blanking_teams:
            squad_blanks = sum(
                1
                for sp in scored_players
                if sp.player.team_id in blanking_teams
            )
            blank_pts = min(squad_blanks / 15.0 * 10.0, 5.0)
            if blank_pts > 0:
                score += blank_pts
                factors.append(
                    f"{squad_blanks}/15 squad players blank this GW"
                )

        # DGW opportunity (0-3): if THIS GW is a DGW, how many slots squad is missing
        dgw_teams_this_gw = set(dgw_raw.get(current_gw, []))
        if dgw_teams_this_gw:
            squad_dgw = sum(
                1
                for sp in scored_players
                if sp.player.team_id in dgw_teams_this_gw
            )
            missing = max(0, 11 - squad_dgw)  # ideally 11 DGW starters
            dgw_pts = min(missing / 11.0 * 3.0, 3.0)
            if dgw_pts > 0:
                score += dgw_pts
                factors.append(
                    f"DGW this week — only {squad_dgw}/15 players double"
                )

        # Squad pain (0-2): injured/doubtful boost (FH swaps whole squad)
        unfit = sum(
            1 for sp in scored_players if sp.availability in ("injured", "doubt")
        )
        if unfit >= 2:
            score += min(unfit * 0.5, 2.0)
            factors.append(f"{unfit} players unfit — FH swaps entire squad")

        return min(score, 10.0), factors

    def _score_tc_this_week(
        self,
        squad_ids: List[int],
        scored_players: List[ScoredPlayer],
        current_gw: int,
        dgw_raw: Dict[int, List[int]],
    ) -> Tuple[float, List[str]]:
        """Score how good THIS week is for Triple Captain (0-10)."""
        score = 0.0
        factors: List[str] = []

        if not scored_players:
            return 0.0, ["No squad data"]

        # Best captain available (0-4): highest ep_next in squad
        best = max(scored_players, key=lambda sp: sp.player.ep_next)
        ep = best.player.ep_next
        if ep >= 8:
            score += 4.0
        elif ep >= 6:
            score += 3.0
        elif ep >= 4:
            score += 2.0
        else:
            score += 1.0
        factors.append(f"Best captain: {best.player.web_name} ({ep:.1f} xPts)")

        # DGW bonus (0-4): if best captain's team has DGW THIS week
        dgw_teams_this_gw = set(dgw_raw.get(current_gw, []))
        if best.player.team_id in dgw_teams_this_gw:
            score += 4.0
            factors.append(f"{best.player.web_name} has DGW — 3x two games!")
        elif dgw_teams_this_gw:
            # DGW exists but best captain doesn't have it
            dgw_names = [
                sp.player.web_name
                for sp in scored_players
                if sp.player.team_id in dgw_teams_this_gw
            ][:3]
            if dgw_names:
                factors.append(
                    f"DGW players in squad: {', '.join(dgw_names)} (not top pick)"
                )

        # Fixture ease (0-2): best captain fixture quality
        if best.fixture_score >= 8.0:
            score += 2.0
            factors.append("Easy fixture")
        elif best.fixture_score >= 6.0:
            score += 1.0
            factors.append("Decent fixture")

        return min(score, 10.0), factors

    def _score_bb_this_week(
        self,
        squad_ids: List[int],
        scored_players: List[ScoredPlayer],
        current_gw: int,
        dgw_raw: Dict[int, List[int]],
    ) -> Tuple[float, List[str]]:
        """Score how good THIS week is for Bench Boost (0-10)."""
        score = 0.0
        factors: List[str] = []

        if len(scored_players) < 15:
            return 0.0, ["Incomplete squad data"]

        dgw_teams_this_gw = set(dgw_raw.get(current_gw, []))

        # DGW coverage (0-4): how many of 15 squad players have DGW this week
        dgw_count = sum(
            1 for sp in scored_players if sp.player.team_id in dgw_teams_this_gw
        )
        if dgw_teams_this_gw:
            dgw_pts = dgw_count / 15.0 * 4.0
            score += dgw_pts
            factors.append(f"{dgw_count}/15 players have DGW this week")

        # Bench fitness (0-3): bench players (lowest 4 scorers) that are fit
        sorted_by_score = sorted(scored_players, key=lambda sp: sp.overall_score)
        bench = sorted_by_score[:4]
        bench_fit = sum(1 for sp in bench if sp.availability == "fit")
        bench_fit_pts = bench_fit / 4.0 * 3.0
        score += bench_fit_pts
        if bench_fit < 4:
            factors.append(f"Only {bench_fit}/4 bench players fit")
        else:
            factors.append("Full bench fitness")

        # Bench quality (0-3): average score of bottom 4
        bench_avg = sum(sp.overall_score for sp in bench) / 4.0
        quality_pts = min(bench_avg / 7.0 * 3.0, 3.0)
        score += quality_pts
        factors.append(f"Bench avg score: {bench_avg:.1f}")

        return min(score, 10.0), factors

    def get_contextual_chip_strategy(
        self,
        available_chips: List[str],
        squad_ids: List[int],
        scorer: PlayerScorer,
        league_intel: Optional[LeagueIntel] = None,
    ) -> dict:
        """Generate squad-aware chip recommendations with this-week scores.

        Returns dict with:
          - recommendations: List[ContextualChipRec] sorted by this_week_score desc
          - squad_health: {fit, doubt, injured}
          - league_gap: Optional[int]
          - remaining_gws: int
          - current_gw: int
          - dgw_info / bgw_info: from _get_upcoming_dgw_bgw()
        """
        current_gw = self.fpl.get_current_gameweek()
        remaining_gws = 38 - current_gw

        # DGW/BGW data (named, for display)
        dgw_info, bgw_info = self._get_upcoming_dgw_bgw()

        # Short-circuit when all chips have been used
        if not available_chips:
            return {
                "recommendations": [],
                "squad_health": {"fit": 0, "doubt": 0, "injured": 0},
                "league_gap": league_intel.points_to_leader if league_intel else None,
                "remaining_gws": remaining_gws,
                "current_gw": current_gw,
                "dgw_info": dgw_info,
                "bgw_info": bgw_info,
            }

        # Raw DGW/BGW (team_id lists, for scoring)
        dgw_raw = self.fixtures.get_double_gameweeks()
        bgw_raw = self.fixtures.get_blank_gameweeks()

        # Score each squad player
        scored_players: List[ScoredPlayer] = []
        squad_health = {"fit": 0, "doubt": 0, "injured": 0}
        for pid in squad_ids:
            player = self.fpl.get_player(pid)
            if player:
                sp = scorer.score_player(player)
                scored_players.append(sp)
                if sp.availability == "fit":
                    squad_health["fit"] += 1
                elif sp.availability == "doubt":
                    squad_health["doubt"] += 1
                else:  # injured / suspended
                    squad_health["injured"] += 1

        # League gap
        league_gap = league_intel.points_to_leader if league_intel else None

        available_set = set(c.lower() for c in available_chips)

        # Build one ContextualChipRec per chip
        chip_configs = [
            (Chip.WILDCARD, "wildcard"),
            (Chip.FREE_HIT, "free_hit"),
            (Chip.TRIPLE_CAPTAIN, "triple_captain"),
            (Chip.BENCH_BOOST, "bench_boost"),
        ]

        recs: List[ContextualChipRec] = []
        for chip_enum, chip_key in chip_configs:
            if chip_key not in available_set:
                continue

            # Get base recommendation from existing methods
            if chip_enum == Chip.WILDCARD:
                base = self._recommend_wildcard(True, current_gw, dgw_info, bgw_info)
                tw_score, tw_factors = self._score_wc_this_week(
                    squad_health, scored_players, dgw_info,
                    current_gw, remaining_gws, league_gap,
                )
            elif chip_enum == Chip.FREE_HIT:
                base = self._recommend_free_hit(True, current_gw, dgw_info, bgw_info)
                tw_score, tw_factors = self._score_fh_this_week(
                    squad_ids, scored_players, current_gw, dgw_raw, bgw_raw,
                )
            elif chip_enum == Chip.TRIPLE_CAPTAIN:
                base = self._recommend_triple_captain(True, current_gw, dgw_info)
                tw_score, tw_factors = self._score_tc_this_week(
                    squad_ids, scored_players, current_gw, dgw_raw,
                )
            else:  # BENCH_BOOST
                base = self._recommend_bench_boost(True, current_gw, dgw_info)
                tw_score, tw_factors = self._score_bb_this_week(
                    squad_ids, scored_players, current_gw, dgw_raw,
                )

            recs.append(
                ContextualChipRec(
                    chip=chip_enum,
                    recommended_gw=base.recommended_gw,
                    reason=base.reason,
                    priority=base.priority,
                    details=base.details,
                    this_week_score=round(tw_score, 1),
                    this_week_factors=tw_factors,
                )
            )

        # Sort by this_week_score descending
        recs.sort(key=lambda r: r.this_week_score, reverse=True)

        # Use the chip planner to assign specific GWs with one-chip-per-GW
        # enforcement and strategic constraints (WC before BB, etc.)
        chip_plan = self.plan_remaining_chips(available_chips, current_gw)

        # Map plan assignments back onto recommendations
        plan_map = {chip_name: gw for gw, chip_name in chip_plan.assignments}
        for rec in recs:
            chip_key = rec.chip.value  # e.g. "wildcard", "bench_boost"
            planned_gw = plan_map.get(chip_key)
            if planned_gw is not None:
                rec.recommended_gw = planned_gw
                plan_reason = chip_plan.per_chip_reasoning.get(chip_key, "")
                # Keep the base reason but update the GW target
                if plan_reason:
                    rec.reason = plan_reason

        return {
            "recommendations": recs,
            "squad_health": squad_health,
            "league_gap": league_gap,
            "remaining_gws": remaining_gws,
            "current_gw": current_gw,
            "next_gw": self.fpl.get_next_gameweek(),
            "dgw_info": dgw_info,
            "bgw_info": bgw_info,
        }

    # ------------------------------------------------------------------
    # Forward-looking chip plan for the season run-in
    # ------------------------------------------------------------------

    def plan_remaining_chips(
        self,
        available_chips: List[str],
        current_gw: int,
    ) -> ChipPlan:
        """Plan optimal chip deployment across remaining GWs.

        Uses BGW/DGW schedules, fixture difficulty, and strategic
        constraints to assign each available chip to a target GW.

        Args:
            available_chips: e.g. ["wildcard", "bench_boost", "triple_captain", "free_hit"]
            current_gw: the current (or next) gameweek number

        Returns:
            ChipPlan with ordered (GW, chip) assignments and reasoning.
        """
        # Use next_gw so we don't assign chips to a finished GW
        next_gw = self.fpl.get_next_gameweek()
        start = max(current_gw, next_gw)
        remaining_gws = list(range(start, 39))  # GW next..38
        if not remaining_gws or not available_chips:
            return ChipPlan(
                assignments=[],
                total_expected_gain=0.0,
                per_chip_reasoning={},
                constraints=["No chips or GWs remaining"],
            )

        available_set = set(c.lower() for c in available_chips)

        # Gather DGW/BGW data (named + raw)
        dgw_info, bgw_info = self._get_upcoming_dgw_bgw()
        dgw_raw = self.fixtures.get_double_gameweeks()
        bgw_raw = self.fixtures.get_blank_gameweeks()

        # ---- Compute per-GW chip value for each chip type ----
        # chip_values[chip_name][gw] = (score, reasoning_snippet)
        chip_values: Dict[str, Dict[int, Tuple[float, str]]] = {}

        for chip_name in available_set:
            chip_values[chip_name] = {}
            for gw in remaining_gws:
                score, reason = self._compute_chip_gw_value(
                    chip_name, gw, current_gw,
                    dgw_info, bgw_info, dgw_raw, bgw_raw,
                )
                chip_values[chip_name][gw] = (score, reason)

        # ---- Apply strategic constraints ----
        constraints_applied: List[str] = []

        # Constraint: WC should ideally be used before BB
        wc_before_bb = "wildcard" in available_set and "bench_boost" in available_set
        if wc_before_bb:
            constraints_applied.append("WC scheduled before BB to optimize squad for Bench Boost")

        # Constraint: TC and BB should target different DGWs
        spread_tc_bb = "triple_captain" in available_set and "bench_boost" in available_set
        if spread_tc_bb:
            constraints_applied.append("TC and BB target different DGWs to spread value")

        # Constraint: FH best on BGW
        if "free_hit" in available_set:
            constraints_applied.append("Free Hit prioritized for BGWs")

        # ---- Greedy assignment with constraint enforcement ----
        assignments: List[Tuple[int, str]] = []
        per_chip_reasoning: Dict[str, str] = {}
        total_gain = 0.0
        used_chips: set = set()
        used_gws: set = set()

        # Pre-sort: assign FH first (BGW-specific), then WC (must precede BB),
        # then BB, then TC. This ordering naturally respects constraints.
        chip_order = []
        for c in ["free_hit", "wildcard", "bench_boost", "triple_captain"]:
            if c in available_set:
                chip_order.append(c)
        # Add any unexpected chip names
        for c in available_set:
            if c not in chip_order:
                chip_order.append(c)

        # Track assigned BB GW for TC constraint
        bb_gw: Optional[int] = None

        for chip_name in chip_order:
            if chip_name in used_chips:
                continue

            # Get scored GWs for this chip, sorted descending by value
            scored_gws = sorted(
                chip_values[chip_name].items(),
                key=lambda x: x[1][0],
                reverse=True,
            )

            assigned = False
            for gw, (score, reason) in scored_gws:
                # Skip GWs already used by another chip
                if gw in used_gws:
                    continue

                # Constraint: WC must come before BB
                if chip_name == "wildcard" and wc_before_bb and "bench_boost" not in used_chips:
                    # Find best BB GW to ensure WC is before it
                    best_bb_gws = sorted(
                        chip_values["bench_boost"].items(),
                        key=lambda x: x[1][0],
                        reverse=True,
                    )
                    best_bb_gw = next(
                        (g for g, _ in best_bb_gws if g not in used_gws and g != gw),
                        None,
                    )
                    if best_bb_gw is not None and gw >= best_bb_gw:
                        # WC must be before BB; skip this GW and try an earlier one
                        continue

                # Constraint: TC and BB target different GWs (already handled by
                # used_gws, but also avoid same DGW cluster if both are DGW chips)
                if chip_name == "triple_captain" and bb_gw is not None and gw == bb_gw:
                    continue

                # Accept this assignment
                assignments.append((gw, chip_name))
                per_chip_reasoning[chip_name] = reason
                total_gain += score
                used_chips.add(chip_name)
                used_gws.add(gw)
                if chip_name == "bench_boost":
                    bb_gw = gw
                assigned = True
                break

            if not assigned:
                # Fallback: assign to best remaining GW ignoring constraints
                for gw, (score, reason) in scored_gws:
                    if gw not in used_gws:
                        assignments.append((gw, chip_name))
                        per_chip_reasoning[chip_name] = reason
                        total_gain += score
                        used_chips.add(chip_name)
                        used_gws.add(gw)
                        if chip_name == "bench_boost":
                            bb_gw = gw
                        break

        # Sort assignments by GW
        assignments.sort(key=lambda x: x[0])

        return ChipPlan(
            assignments=assignments,
            total_expected_gain=round(total_gain, 1),
            per_chip_reasoning=per_chip_reasoning,
            constraints=constraints_applied,
        )

    def _compute_chip_gw_value(
        self,
        chip_name: str,
        gw: int,
        current_gw: int,
        dgw_info: Dict[int, List[str]],
        bgw_info: Dict[int, List[str]],
        dgw_raw: Dict[int, List[int]],
        bgw_raw: Dict[int, List[int]],
    ) -> Tuple[float, str]:
        """Compute how valuable using a chip in a specific GW would be.

        Returns (score 0-10, reasoning string).
        """
        if chip_name == "free_hit":
            return self._value_free_hit_gw(gw, current_gw, dgw_info, bgw_info, bgw_raw)
        elif chip_name == "bench_boost":
            return self._value_bench_boost_gw(gw, current_gw, dgw_info, dgw_raw)
        elif chip_name == "triple_captain":
            return self._value_triple_captain_gw(gw, current_gw, dgw_info, dgw_raw)
        elif chip_name == "wildcard":
            return self._value_wildcard_gw(gw, current_gw, dgw_info, bgw_info)
        else:
            return 1.0, f"Unknown chip '{chip_name}'"

    def _value_free_hit_gw(
        self,
        gw: int,
        current_gw: int,
        dgw_info: Dict[int, List[str]],
        bgw_info: Dict[int, List[str]],
        bgw_raw: Dict[int, List[int]],
    ) -> Tuple[float, str]:
        """Free Hit value: high on BGWs, moderate on large DGWs."""
        score = 1.0  # baseline for any GW
        reasons = []

        if gw in bgw_info:
            blanking_teams = bgw_raw.get(gw, [])
            blank_count = len(blanking_teams)
            # More teams blanking = higher FH value
            bgw_bonus = min(blank_count / 20.0 * 8.0, 8.0)
            score += bgw_bonus
            reasons.append(f"BGW{gw}: {blank_count} teams blank — build full XI from playing teams")

        if gw in dgw_info:
            dgw_teams = dgw_info.get(gw, [])
            dgw_bonus = min(len(dgw_teams) / 20.0 * 4.0, 4.0)
            score += dgw_bonus
            reasons.append(f"DGW{gw}: {len(dgw_teams)} teams double — FH to stack doublers")

        if not reasons:
            reasons.append(f"GW{gw}: Normal week — low FH value, save for BGW")

        return min(score, 10.0), "; ".join(reasons)

    def _value_bench_boost_gw(
        self,
        gw: int,
        current_gw: int,
        dgw_info: Dict[int, List[str]],
        dgw_raw: Dict[int, List[int]],
    ) -> Tuple[float, str]:
        """Bench Boost value: high on DGWs with many doublers."""
        score = 1.0
        reasons = []

        if gw in dgw_info:
            dgw_team_ids = dgw_raw.get(gw, [])
            dgw_names = dgw_info.get(gw, [])
            # Use confirmed team count if available, otherwise estimate from expected DGW
            dgw_count = len(dgw_team_ids) if dgw_team_ids else len(dgw_names)
            # If TBC (expected but unconfirmed), assume ~10 teams will double
            if dgw_count == 0 or (len(dgw_names) == 1 and dgw_names[0] == "TBC"):
                dgw_count = 10  # Conservative estimate for expected DGW

            # More teams doubling = easier to fill 15 with doublers
            dgw_bonus = min(dgw_count / 20.0 * 8.0, 8.0)
            score += dgw_bonus

            # Check if premium teams are doubling
            premium_teams = {"LIV", "MCI", "ARS", "CHE", "MUN", "TOT"}
            dgw_names_set = set(dgw_names)
            premium_doubles = dgw_names_set & premium_teams
            if premium_doubles:
                score += 1.0
                reasons.append(
                    f"DGW{gw}: {dgw_count} teams double incl. "
                    f"{', '.join(sorted(premium_doubles))} — 15 doublers possible"
                )
            else:
                reasons.append(f"DGW{gw}: ~{dgw_count} teams expected to double — stack squad with doublers")
        else:
            # No DGW — BB value depends on how many teams have easy fixtures
            # (more easy fixtures = more bench players likely to return points).
            # Count teams with ease >= 7 (strong matchups).
            teams = self.fpl.get_all_teams()
            easy_teams = sum(
                1 for t in teams
                if self.fixtures.get_fixture_ease_for_gw(t.id, gw) >= 7.0
            )
            if easy_teams >= 12:
                score += 2.0
                reasons.append(f"{easy_teams} teams with easy fixtures — best BB week")
            elif easy_teams >= 8:
                score += 1.0
                reasons.append(f"{easy_teams} teams with easy fixtures")
            elif easy_teams >= 5:
                score += 0.5
                reasons.append(f"Mixed fixtures ({easy_teams} easy)")
            else:
                reasons.append(f"Few easy fixtures — weak BB week")

        return min(score, 10.0), "; ".join(reasons)

    def _value_triple_captain_gw(
        self,
        gw: int,
        current_gw: int,
        dgw_info: Dict[int, List[str]],
        dgw_raw: Dict[int, List[int]],
    ) -> Tuple[float, str]:
        """Triple Captain value: high on DGWs with premium captain options."""
        score = 1.0
        reasons = []

        if gw in dgw_info:
            dgw_names = dgw_info.get(gw, [])
            dgw_names_set = set(dgw_names)
            premium_teams = {"LIV", "MCI", "ARS", "CHE", "MUN", "TOT"}
            premium_doubles = dgw_names_set & premium_teams

            # If TBC, assume premiums likely to double (conservative optimism)
            is_tbc = len(dgw_names) == 1 and dgw_names[0] == "TBC"

            if premium_doubles:
                # High value: Haaland/Salah etc. with two games
                score += 7.0
                reasons.append(
                    f"DGW{gw}: {', '.join(sorted(premium_doubles))} double — "
                    f"3x premium captain across two fixtures"
                )
            elif is_tbc:
                # Expected DGW but teams TBC — assume moderate value
                score += 5.5
                reasons.append(
                    f"DGW{gw}: Teams TBC — likely premium doubles, "
                    f"TC high-ceiling play"
                )
            else:
                score += 4.0
                reasons.append(
                    f"DGW{gw}: No premium teams doubling — TC still decent "
                    f"but captain floor is lower"
                )

            # Assess fixture quality for doubling teams
            dgw_team_ids = dgw_raw.get(gw, [])
            if dgw_team_ids:
                ease_scores = []
                for tid in dgw_team_ids:
                    ease = self.fixtures.get_fixture_ease_for_gw(tid, gw)
                    ease_scores.append(ease)
                avg_ease = sum(ease_scores) / len(ease_scores)
                if avg_ease >= 6.0:
                    score += 1.5
                    reasons.append(f"Avg fixture ease {avg_ease:.1f}/10 — favorable matchups")
                elif avg_ease <= 3.5:
                    score -= 1.0
                    reasons.append(f"Avg fixture ease {avg_ease:.1f}/10 — tough matchups reduce TC value")
        else:
            # No DGW — TC value depends on premium captain fixture ease.
            # Score based on best premium fixture AND how many premiums are easy
            # (more options = more likely your captain benefits).
            premium_team_names = {"LIV", "MCI", "ARS", "CHE", "MUN", "TOT"}
            teams = self.fpl.get_all_teams()
            premium_ids = {t.id: t.short_name for t in teams if t.short_name in premium_team_names}

            premium_eases = []
            for tid, name in premium_ids.items():
                ease = self.fixtures.get_fixture_ease_for_gw(tid, gw)
                premium_eases.append((name, ease))
            premium_eases.sort(key=lambda x: -x[1])

            best_team, best_ease = premium_eases[0] if premium_eases else ("?", 5.0)
            great_count = sum(1 for _, e in premium_eases if e >= 8.0)
            good_count = sum(1 for _, e in premium_eases if e >= 6.0)

            if best_ease >= 8.0:
                # Base bonus for having a great premium fixture
                score += 2.0
                # Extra for multiple premiums with great fixtures (more captain options)
                score += min(great_count * 0.5, 1.5)
                great_names = [n for n, e in premium_eases if e >= 8.0]
                reasons.append(f"{', '.join(great_names)} easy — strong TC week")
            elif best_ease >= 6.0:
                score += 1.0 + min(good_count * 0.3, 0.9)
                reasons.append(f"{best_team} has decent fixture")
            elif best_ease >= 4.0:
                score += 0.5
                reasons.append(f"No premium with easy fixture")
            else:
                reasons.append(f"Premium captains face tough fixtures")

        return max(0.0, min(score, 10.0)), "; ".join(reasons)

    def _value_wildcard_gw(
        self,
        gw: int,
        current_gw: int,
        dgw_info: Dict[int, List[str]],
        bgw_info: Dict[int, List[str]],
    ) -> Tuple[float, str]:
        """Wildcard value: high 1-2 GWs before DGW, good at fixture swings."""
        score = 1.0
        reasons = []

        upcoming_dgws = sorted(g for g in dgw_info if g > current_gw)
        upcoming_bgws = sorted(g for g in bgw_info if g > current_gw)

        # Best scenario: WC 1-2 weeks before a DGW to prepare squad
        for dgw in upcoming_dgws:
            gap = dgw - gw
            if gap == 1:
                score += 7.0
                reasons.append(
                    f"GW{gw}: 1 week before DGW{dgw} — rebuild squad to maximize doublers"
                )
                break
            elif gap == 2:
                score += 5.5
                reasons.append(
                    f"GW{gw}: 2 weeks before DGW{dgw} — good time to restructure"
                )
                break
            elif gap == 0:
                # WC on the DGW itself is okay but less ideal (no prep time)
                score += 4.0
                reasons.append(
                    f"GW{gw}: DGW week itself — can still restructure but less prep"
                )
                break

        # Bonus: WC before a BGW helps navigate without FH
        for bgw in upcoming_bgws:
            gap = bgw - gw
            if 1 <= gap <= 2:
                score += 2.0
                reasons.append(
                    f"Also near BGW{bgw} — WC helps avoid blank players"
                )
                break

        # Penalty: too late in season reduces WC value (fewer GWs to benefit)
        remaining_after = 38 - gw
        if remaining_after <= 2:
            score -= 2.0
            reasons.append(f"Only {remaining_after} GWs left after — limited WC benefit")
        elif remaining_after <= 4:
            score -= 0.5
            reasons.append(f"{remaining_after} GWs left — use soon or lose it")

        # Bonus: WC has more value on weeks with big fixture swings
        # (many teams changing from hard to easy or vice versa)
        avg_ease = self._avg_fixture_ease_for_gw(gw)
        if avg_ease >= 6.0 and not reasons:
            score += 1.0
            reasons.append(f"Easy fixtures — good week to restructure")
        elif not reasons:
            reasons.append(f"No nearby DGW/BGW — WC for fixture swing")

        return max(0.0, min(score, 10.0)), "; ".join(reasons)

    def _avg_fixture_ease_for_gw(self, gw: int) -> float:
        """Average fixture ease across all teams for a given GW (0-10 scale).

        Higher = easier fixtures overall. Used to differentiate normal GWs
        when no DGW/BGW exists.
        """
        teams = self.fpl.get_all_teams()
        if not teams:
            return 5.0
        ease_scores = [
            self.fixtures.get_fixture_ease_for_gw(t.id, gw) for t in teams
        ]
        return sum(ease_scores) / len(ease_scores) if ease_scores else 5.0

    @staticmethod
    def format_chip_plan(plan: ChipPlan) -> str:
        """Format a ChipPlan into a clean roadmap display."""
        lines = []
        lines.append("CHIP DEPLOYMENT ROADMAP")
        lines.append("=" * 50)

        if not plan.assignments:
            lines.append("No chips to plan.")
            return "\n".join(lines)

        # Chip assignments timeline
        lines.append("")
        chip_labels = {
            "wildcard": "WILDCARD",
            "free_hit": "FREE HIT",
            "triple_captain": "TRIPLE CAPTAIN",
            "bench_boost": "BENCH BOOST",
        }

        for gw, chip_name in plan.assignments:
            label = chip_labels.get(chip_name, chip_name.upper())
            reasoning = plan.per_chip_reasoning.get(chip_name, "")
            lines.append(f"  GW{gw:>2}  ->  {label}")
            if reasoning:
                # Wrap long reasoning lines
                for part in reasoning.split("; "):
                    lines.append(f"          {part}")

        # Total expected gain
        lines.append("")
        lines.append(f"Combined chip value score: {plan.total_expected_gain}")

        # Constraints applied
        if plan.constraints:
            lines.append("")
            lines.append("Constraints applied:")
            for c in plan.constraints:
                lines.append(f"  - {c}")

        lines.append("=" * 50)
        return "\n".join(lines)
