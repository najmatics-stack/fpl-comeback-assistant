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

        return {
            "recommendations": recs,
            "squad_health": squad_health,
            "league_gap": league_gap,
            "remaining_gws": remaining_gws,
            "current_gw": current_gw,
            "dgw_info": dgw_info,
            "bgw_info": bgw_info,
        }
