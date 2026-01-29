"""
Recommendation engine - generates transfer, captain, and strategic advice
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from tabulate import tabulate

from data.fpl_api import FPLDataFetcher, Player
from data.news_scraper import NewsScraper, InjuryStatus
from analysis.player_scorer import PlayerScorer, ScoredPlayer
from analysis.fixture_analyzer import FixtureAnalyzer
from analysis.differential import DifferentialFinder, Differential
from analysis.chip_optimizer import ChipOptimizer, ChipStrategy

import config


@dataclass
class TransferRecommendation:
    """A recommended transfer"""

    player_out: ScoredPlayer
    player_in: ScoredPlayer
    score_gain: float
    price_diff: float
    reason: str


@dataclass
class CaptainPick:
    """A captain recommendation"""

    player: ScoredPlayer
    expected_points: float
    reason: str
    fixture_info: str


@dataclass
class FullRecommendation:
    """Complete set of recommendations"""

    gameweek: int
    injury_alerts: List[Tuple[str, str, str]]  # (name, status, details)
    transfers: List[TransferRecommendation]
    captain_picks: List[CaptainPick]
    differentials: Dict[str, List[Differential]]
    chip_strategy: ChipStrategy
    fixture_summary: str


class RecommendationEngine:
    """Generates comprehensive FPL recommendations"""

    def __init__(
        self,
        fpl_data: FPLDataFetcher,
        player_scorer: PlayerScorer,
        fixture_analyzer: FixtureAnalyzer,
        differential_finder: DifferentialFinder,
        chip_optimizer: ChipOptimizer,
        news_scraper: Optional[NewsScraper] = None,
    ):
        self.fpl = fpl_data
        self.scorer = player_scorer
        self.fixtures = fixture_analyzer
        self.differentials = differential_finder
        self.chips = chip_optimizer
        self.news = news_scraper

    def get_injury_alerts(
        self, squad_player_ids: Optional[List[int]] = None
    ) -> List[Tuple[str, str, str]]:
        """Get injury/availability alerts"""
        alerts = []

        if squad_player_ids:
            # Check specific squad
            for pid in squad_player_ids:
                player = self.fpl.get_player(pid)
                if player:
                    sp = self.scorer.score_player(player)
                    if sp.availability != "fit":
                        icon = {"doubt": "⚠️", "injured": "❌", "suspended": "🚫"}.get(
                            sp.availability, "?"
                        )
                        alerts.append(
                            (
                                f"{icon} {player.web_name}",
                                sp.availability.upper(),
                                sp.injury_details or player.news or "No details",
                            )
                        )
        else:
            # Check all players with issues
            for player in self.fpl.get_all_players():
                if player.status != "a" or (
                    player.chance_of_playing is not None
                    and player.chance_of_playing < 100
                ):
                    # Only include notable players (high ownership or points)
                    if player.selected_by_percent > 5 or player.total_points > 50:
                        sp = self.scorer.score_player(player)
                        if sp.availability != "fit":
                            icon = {
                                "doubt": "⚠️",
                                "injured": "❌",
                                "suspended": "🚫",
                            }.get(sp.availability, "?")
                            alerts.append(
                                (
                                    f"{icon} {player.web_name} ({player.team})",
                                    sp.availability.upper(),
                                    sp.injury_details or player.news or "No details",
                                )
                            )

        # Sort: injured/suspended first, then doubtful
        status_order = {"INJURED": 0, "SUSPENDED": 1, "DOUBT": 2}
        alerts.sort(key=lambda x: status_order.get(x[1], 3))

        return alerts[:15]  # Limit to top 15

    def _get_squad_team_counts(self, squad_ids: List[int]) -> Dict[int, int]:
        """Count players per team in current squad"""
        counts: Dict[int, int] = {}
        for pid in squad_ids:
            player = self.fpl.get_player(pid)
            if player:
                counts[player.team_id] = counts.get(player.team_id, 0) + 1
        return counts

    def _get_squad_position_counts(self, squad_ids: List[int]) -> Dict[str, int]:
        """Count players per position in current squad"""
        counts: Dict[str, int] = {}
        for pid in squad_ids:
            player = self.fpl.get_player(pid)
            if player:
                counts[player.position] = counts.get(player.position, 0) + 1
        return counts

    def _would_violate_rules(
        self,
        player_in_id: int,
        player_out_id: int,
        squad_ids: List[int],
        pending_transfers: List[TransferRecommendation],
    ) -> bool:
        """Check if a transfer would violate FPL rules"""
        player_in = self.fpl.get_player(player_in_id)
        player_out = self.fpl.get_player(player_out_id)
        if not player_in or not player_out:
            return True

        # Build effective squad state after pending transfers
        effective_ids = set(squad_ids)
        for tr in pending_transfers:
            effective_ids.discard(tr.player_out.player.id)
            effective_ids.add(tr.player_in.player.id)
        effective_ids.discard(player_out_id)
        effective_ids.add(player_in_id)

        # Rule 1: Max 3 players per team
        team_counts: Dict[int, int] = {}
        for pid in effective_ids:
            p = self.fpl.get_player(pid)
            if p:
                team_counts[p.team_id] = team_counts.get(p.team_id, 0) + 1

        if team_counts.get(player_in.team_id, 0) > 3:
            return True

        # Rule 2: Position limits (2 GKP, 5 DEF, 5 MID, 3 FWD)
        pos_limits = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
        pos_counts: Dict[str, int] = {}
        for pid in effective_ids:
            p = self.fpl.get_player(pid)
            if p:
                pos_counts[p.position] = pos_counts.get(p.position, 0) + 1

        if pos_counts.get(player_in.position, 0) > pos_limits.get(player_in.position, 5):
            return True

        # Rule 3: Squad size must stay at 15
        if len(effective_ids) != 15:
            return True

        return False

    def get_transfer_recommendations(
        self,
        current_squad_ids: List[int],
        budget: float = 0.0,
        free_transfers: int = 1,
        limit: int = 5,
    ) -> List[TransferRecommendation]:
        """Generate transfer recommendations respecting all FPL rules"""
        recommendations: List[TransferRecommendation] = []

        # Score current squad
        current_squad = []
        for pid in current_squad_ids:
            player = self.fpl.get_player(pid)
            if player:
                current_squad.append(self.scorer.score_player(player))

        # Sort weakest first
        current_squad.sort(key=lambda x: x.overall_score)

        # Consider more weak players but only recommend up to limit
        candidates = current_squad[: limit + 3]
        remaining_budget = budget

        for weak in candidates:
            if len(recommendations) >= limit:
                break

            position = weak.player.position
            max_price = weak.player.price + remaining_budget

            # Get top players in that position
            top_players = self.scorer.get_top_players(
                position=position, max_price=max_price, limit=30
            )

            for top in top_players:
                # Skip players already in squad
                if top.player.id in current_squad_ids:
                    continue

                # Skip if already recommended as an incoming player
                if any(r.player_in.player.id == top.player.id for r in recommendations):
                    continue

                # Skip if outgoing player already used
                if any(r.player_out.player.id == weak.player.id for r in recommendations):
                    break

                # Enforce FPL rules
                if self._would_violate_rules(
                    top.player.id, weak.player.id, current_squad_ids, recommendations
                ):
                    continue

                # Check budget
                price_diff = top.player.price - weak.player.price
                if price_diff > remaining_budget:
                    continue

                score_gain = top.overall_score - weak.overall_score

                # For hits (-4), require higher threshold
                transfer_number = len(recommendations) + 1
                is_hit = transfer_number > free_transfers
                min_gain = 1.5 if is_hit else 0.5

                if score_gain <= min_gain:
                    continue

                # Generate reason
                reasons = []
                if top.fixture_score > weak.fixture_score + 1:
                    reasons.append("better fixtures")
                if top.form_score > weak.form_score + 1:
                    reasons.append("better form")
                if top.xgi_score > weak.xgi_score + 1:
                    reasons.append("higher xGI")
                if weak.availability in ("injured", "suspended"):
                    reasons.append(f"{weak.player.web_name} {weak.availability}")
                if is_hit:
                    reasons.append(f"-4 hit (worth +{score_gain:.1f} gain)")

                reason = ", ".join(reasons) if reasons else "overall upgrade"

                recommendations.append(
                    TransferRecommendation(
                        player_out=weak,
                        player_in=top,
                        score_gain=score_gain,
                        price_diff=price_diff,
                        reason=reason,
                    )
                )
                remaining_budget -= price_diff
                break

        recommendations.sort(key=lambda x: x.score_gain, reverse=True)
        return recommendations[:limit]

    def get_general_transfer_suggestions(self, limit: int = 5) -> List[TransferRecommendation]:
        """Get general transfer suggestions without knowing current squad"""
        recommendations = []

        positions = ["GKP", "DEF", "MID", "FWD"]

        for pos in positions:
            # Get top 5 and bottom 5 by ownership in that position
            players = self.fpl.get_players_by_position(pos)

            # High ownership (likely to have)
            high_ownership = sorted(
                players, key=lambda x: x.selected_by_percent, reverse=True
            )[:10]

            for ho_player in high_ownership:
                ho_scored = self.scorer.score_player(ho_player)

                # Skip if player is still good
                if ho_scored.overall_score > 5.5:
                    continue

                # Find better option
                top_players = self.scorer.get_top_players(
                    position=pos, max_price=ho_player.price + 1.0, limit=10
                )

                for top in top_players:
                    if top.player.id == ho_player.id:
                        continue

                    score_gain = top.overall_score - ho_scored.overall_score
                    if score_gain > 1.0:  # Significant upgrade
                        reasons = []
                        if top.fixture_score > ho_scored.fixture_score + 1:
                            reasons.append("better fixtures")
                        if top.form_score > ho_scored.form_score + 1:
                            reasons.append("better form")

                        recommendations.append(
                            TransferRecommendation(
                                player_out=ho_scored,
                                player_in=top,
                                score_gain=score_gain,
                                price_diff=top.player.price - ho_player.price,
                                reason=", ".join(reasons) or "overall upgrade",
                            )
                        )
                        break

        recommendations.sort(key=lambda x: x.score_gain, reverse=True)
        return recommendations[:limit]

    def get_captain_picks(
        self, squad_ids: Optional[List[int]] = None, limit: int = 3
    ) -> List[CaptainPick]:
        """Get captain recommendations"""
        if squad_ids:
            # Captain from squad
            players = [self.fpl.get_player(pid) for pid in squad_ids]
            players = [p for p in players if p]
        else:
            # Get top overall players
            top = self.scorer.get_top_players(limit=20)
            players = [sp.player for sp in top]

        captain_picks = []

        for player in players:
            sp = self.scorer.score_player(player)

            # Skip unavailable
            if sp.availability in ("injured", "suspended"):
                continue

            # Calculate expected points (rough estimate)
            # Base on form, fixture, and position multipliers
            fixture_run = self.fixtures.get_fixture_run(player.team_id)

            base_exp = player.form * 1.5  # Form-based expectation
            fixture_mult = (5 - fixture_run.avg_difficulty) / 4 + 0.5  # 0.5-1.25x

            # Position bonus (attackers score more when they score)
            pos_mult = {"GKP": 0.7, "DEF": 0.9, "MID": 1.0, "FWD": 1.1}.get(
                player.position, 1.0
            )

            # DGW bonus
            if fixture_run.has_double:
                base_exp *= 1.8

            expected_pts = base_exp * fixture_mult * pos_mult

            # Doubt penalty
            if sp.availability == "doubt":
                expected_pts *= 0.75

            # Generate fixture info
            fixtures_str = " + ".join(
                f"{opp}" for _, opp, diff, _ in fixture_run.fixtures[:2]
            )

            # Generate reason
            reasons = []
            if player.form >= 6:
                reasons.append("excellent form")
            elif player.form >= 4:
                reasons.append("good form")

            if fixture_run.avg_difficulty <= 2.5:
                reasons.append("easy fixture")
            if fixture_run.has_double:
                reasons.append("DOUBLE GW")

            if player.selected_by_percent > 30:
                reasons.append("safe pick")
            elif player.selected_by_percent < 10:
                reasons.append("differential")

            captain_picks.append(
                CaptainPick(
                    player=sp,
                    expected_points=expected_pts,
                    reason=", ".join(reasons) or "solid option",
                    fixture_info=fixtures_str,
                )
            )

        # Sort by expected points
        captain_picks.sort(key=lambda x: x.expected_points, reverse=True)
        return captain_picks[:limit]

    def get_full_recommendations(
        self,
        squad_ids: Optional[List[int]] = None,
        available_chips: Optional[List[str]] = None,
    ) -> FullRecommendation:
        """Generate comprehensive recommendations"""
        current_gw = self.fpl.get_current_gameweek()

        # Injury alerts
        injury_alerts = self.get_injury_alerts(squad_ids)

        # Transfer recommendations
        if squad_ids:
            transfers = self.get_transfer_recommendations(squad_ids)
        else:
            transfers = self.get_general_transfer_suggestions()

        # Captain picks
        captain_picks = self.get_captain_picks(squad_ids)

        # Differentials
        differentials = self.differentials.find_differentials_by_position(
            limit_per_position=config.TOP_DIFFERENTIALS
        )

        # Chip strategy
        chip_strategy = self.chips.get_chip_strategy(available_chips)

        # Fixture summary
        best_fixtures = self.fixtures.get_best_fixtures(5)
        fixture_lines = ["BEST FIXTURES NEXT 5 GWs:"]
        for run in best_fixtures:
            fixture_lines.append(f"   {self.fixtures.format_fixture_run(run)}")
        fixture_summary = "\n".join(fixture_lines)

        return FullRecommendation(
            gameweek=current_gw,
            injury_alerts=injury_alerts,
            transfers=transfers,
            captain_picks=captain_picks,
            differentials=differentials,
            chip_strategy=chip_strategy,
            fixture_summary=fixture_summary,
        )

    def format_full_recommendations(self, rec: FullRecommendation) -> str:
        """Format full recommendations for display"""
        lines = [
            "",
            "=" * 60,
            f"  FPL COMEBACK ASSISTANT - GW{rec.gameweek + 1} RECOMMENDATIONS",
            "=" * 60,
            "",
        ]

        # Injury Alerts
        lines.append("📋 INJURY/AVAILABILITY ALERTS")
        lines.append("-" * 40)
        if rec.injury_alerts:
            for name, status, details in rec.injury_alerts[:10]:
                lines.append(f"   {name}: {details[:50]}")
        else:
            lines.append("   ✓ No major injury concerns")
        lines.append("")

        # Transfer Recommendations
        lines.append("🔄 TRANSFER RECOMMENDATIONS")
        lines.append("-" * 40)
        if rec.transfers:
            for i, tr in enumerate(rec.transfers, 1):
                out_p = tr.player_out.player
                in_p = tr.player_in.player
                price_str = (
                    f"+£{tr.price_diff:.1f}m"
                    if tr.price_diff > 0
                    else f"£{tr.price_diff:.1f}m"
                )
                lines.append(
                    f"   {i}. OUT: {out_p.web_name} ({out_p.team}) → "
                    f"IN: {in_p.web_name} ({in_p.team})"
                )
                lines.append(
                    f"      {price_str} | +{tr.score_gain:.1f} score | {tr.reason}"
                )
        else:
            lines.append("   No clear transfer targets")
        lines.append("")

        # Captain Picks
        lines.append("👑 CAPTAIN PICKS")
        lines.append("-" * 40)
        for i, cp in enumerate(rec.captain_picks, 1):
            p = cp.player.player
            avail = "" if cp.player.availability == "fit" else f" ⚠️{cp.player.availability}"
            lines.append(
                f"   {i}. {p.web_name} ({p.team}) - {cp.expected_points:.1f} exp pts{avail}"
            )
            lines.append(f"      vs {cp.fixture_info} | {cp.reason}")
        lines.append("")

        # Differentials
        lines.append("🎯 DIFFERENTIAL PICKS (Low Ownership, High Potential)")
        lines.append("-" * 40)
        for pos in ["MID", "FWD", "DEF", "GKP"]:
            diffs = rec.differentials.get(pos, [])[:2]
            if diffs:
                lines.append(f"   {pos}:")
                for diff in diffs:
                    p = diff.scored_player.player
                    lines.append(
                        f"      • {p.web_name} ({p.team}) £{p.price}m - "
                        f"{diff.ownership:.1f}% owned"
                    )
                    lines.append(f"        {diff.upside_reason}")
        lines.append("")

        # Chip Strategy (condensed)
        lines.append("🃏 CHIP STRATEGY")
        lines.append("-" * 40)
        for chip_rec in rec.chip_strategy.recommendations:
            priority_icon = {1: "🔥", 2: "📅", 3: "💤"}.get(chip_rec.priority, "?")
            gw_str = f"GW{chip_rec.recommended_gw}" if chip_rec.recommended_gw else "Hold"
            lines.append(f"   {priority_icon} {chip_rec.chip.value.upper()}: {gw_str}")
            lines.append(f"      {chip_rec.reason}")
        lines.append("")

        # Fixture Summary
        lines.append("📅 FIXTURE TICKER")
        lines.append("-" * 40)
        best = self.fixtures.get_best_fixtures(5)
        for run in best:
            fixtures_str = " ".join(f"{opp}" for _, opp, _, _ in run.fixtures[:5])
            lines.append(f"   {run.team_name[:12]:12} {fixtures_str}")
        lines.append("")

        # Comeback Tips
        lines.append("💡 COMEBACK TIPS")
        lines.append("-" * 40)
        lines.append("   • Target differentials - you need to gain ground!")
        lines.append("   • Don't follow the template - be bold with picks")
        lines.append("   • Save chips for DGWs - maximum point potential")
        lines.append("   • Captain ceiling > floor - pick explosive options")
        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)
