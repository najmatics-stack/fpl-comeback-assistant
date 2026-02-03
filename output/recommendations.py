"""
Recommendation engine - generates transfer, captain, and strategic advice
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from tabulate import tabulate

from data.fpl_api import FPLDataFetcher, Player
from data.news_scraper import NewsScraper, InjuryStatus
from analysis.player_scorer import PlayerScorer, ScoredPlayer
from analysis.fixture_analyzer import FixtureAnalyzer
from analysis.differential import DifferentialFinder, Differential
from analysis.chip_optimizer import ChipOptimizer, ChipStrategy
from analysis.league_spy import LeagueIntel

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


@dataclass
class TransferPlan:
    """A transfer plan option for interactive auto mode"""

    label: str  # "conservative", "balanced", "aggressive"
    display_label: str  # "A: CONSERVATIVE", etc.
    transfers: List[TransferRecommendation]
    total_score_gain: float
    hit_cost: int  # 0, -4, -8
    is_recommended: bool


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
        league_intel: Optional[LeagueIntel] = None,
    ):
        self.fpl = fpl_data
        self.scorer = player_scorer
        self.fixtures = fixture_analyzer
        self.differentials = differential_finder
        self.chips = chip_optimizer
        self.news = news_scraper
        self.league_intel = league_intel

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

        # Rule 1: Max 3 players per team (check ALL teams, not just incoming)
        team_counts: Dict[int, int] = {}
        for pid in effective_ids:
            p = self.fpl.get_player(pid)
            if p:
                team_counts[p.team_id] = team_counts.get(p.team_id, 0) + 1

        for tid, count in team_counts.items():
            if count > 3:
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

    def _get_over_limit_teams(self, squad_ids: List[int]) -> Dict[int, int]:
        """Find teams with more than 3 players in the squad"""
        team_counts: Dict[int, int] = {}
        for pid in squad_ids:
            p = self.fpl.get_player(pid)
            if p:
                team_counts[p.team_id] = team_counts.get(p.team_id, 0) + 1
        return {tid: count for tid, count in team_counts.items() if count > 3}

    def get_transfer_recommendations(
        self,
        current_squad_ids: List[int],
        budget: float = 0.0,
        free_transfers: int = 1,
        limit: int = 5,
        min_gain_free: float = 0.5,
        min_gain_hit: float = 1.5,
        locked_ids: Optional[Set[int]] = None,
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

        # Force transfers from over-limit teams (e.g., 4 players from one team
        # due to mid-season transfers). FPL API rejects ALL transfers until
        # the squad is brought back to 3-per-team.
        over_limit = self._get_over_limit_teams(current_squad_ids)
        forced_out = []
        if over_limit:
            for tid, count in over_limit.items():
                excess = count - 3
                # Pick the weakest players from the over-limit team
                team_players = [sp for sp in current_squad if sp.player.team_id == tid]
                team_players.sort(key=lambda x: x.overall_score)
                forced_out.extend(team_players[:excess])

        # Build candidate list: forced transfers first, then weakest players
        forced_ids = {sp.player.id for sp in forced_out}
        remaining = [sp for sp in current_squad if sp.player.id not in forced_ids]
        # Exclude locked players from candidates (but forced transfers take precedence)
        if locked_ids:
            remaining = [sp for sp in remaining if sp.player.id not in locked_ids]
        candidates = forced_out + remaining[: limit + 3 - len(forced_out)]
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

                # League spy adjustments
                if self.league_intel:
                    in_ownership = self.league_intel.league_ownership.get(top.player.id, 0)
                    if in_ownership < 20 and top.player.form >= 4.0:
                        score_gain *= 1.15  # League differential bonus
                    out_ownership = self.league_intel.league_ownership.get(weak.player.id, 0)
                    if out_ownership >= 70:
                        score_gain *= 0.85  # Penalty for selling must-haves

                    # Global vs league: heavily boost safe differentials
                    # World validates the pick, rivals don't own — maximum exploitation
                    safe_diff_ids = {e.player_id for e in self.league_intel.safe_differentials}
                    if top.player.id in safe_diff_ids:
                        score_gain *= 1.50  # Heavy bonus: world-proven league differential

                    # Trending hidden gems get a strong boost too
                    trending_ids = {e.player_id for e in self.league_intel.trending_hidden}
                    if top.player.id in trending_ids:
                        score_gain *= 1.35  # World is moving here, league hasn't caught on

                    # Selling a league-overweight player your rivals cling to = exploitable
                    overweight_ids = {e.player_id for e in self.league_intel.league_overweight}
                    if weak.player.id in overweight_ids:
                        score_gain *= 1.25  # Rivals over-index, selling exploits the gap

                # For hits (-4), require higher threshold
                transfer_number = len(recommendations) + 1
                is_hit = transfer_number > free_transfers
                min_gain = min_gain_hit if is_hit else min_gain_free

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
                if self.league_intel:
                    in_ownership = self.league_intel.league_ownership.get(top.player.id, 0)
                    out_ownership = self.league_intel.league_ownership.get(weak.player.id, 0)
                    if in_ownership < 20 and top.player.form >= 4.0:
                        reasons.append(f"league differential ({in_ownership:.0f}% rivals)")
                    if out_ownership >= 70:
                        reasons.append(f"rival template ({out_ownership:.0f}% own)")

                    # Global vs league reasons
                    safe_diff_map = {e.player_id: e for e in self.league_intel.safe_differentials}
                    if top.player.id in safe_diff_map:
                        e = safe_diff_map[top.player.id]
                        reasons.append(
                            f"EXPLOIT: world owns {e.global_ownership:.0f}%, league only {e.league_ownership:.0f}%"
                        )
                    trending_map = {e.player_id: e for e in self.league_intel.trending_hidden}
                    if top.player.id in trending_map:
                        e = trending_map[top.player.id]
                        xfers = e.transfers_in_event
                        xfers_str = f"{xfers/1000:.0f}k" if xfers >= 1000 else str(xfers)
                        reasons.append(
                            f"TRENDING: +{xfers_str} transfers, league asleep"
                        )
                    overweight_map = {e.player_id: e for e in self.league_intel.league_overweight}
                    if weak.player.id in overweight_map:
                        e = overweight_map[weak.player.id]
                        reasons.append(
                            f"EXPLOIT: league over-indexes ({e.league_ownership:.0f}% vs {e.global_ownership:.0f}% global)"
                        )

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
        """Get captain recommendations using ep_next anchor + ceiling analysis"""
        if squad_ids:
            players = [self.fpl.get_player(pid) for pid in squad_ids]
            players = [p for p in players if p]
        else:
            top = self.scorer.get_top_players(limit=20)
            players = [sp.player for sp in top]

        current_gw = self.fpl.get_current_gameweek()
        captain_picks = []

        for player in players:
            sp = self.scorer.score_player(player)

            if sp.availability in ("injured", "suspended"):
                continue

            fixture_run = self.fixtures.get_fixture_run(player.team_id)

            # ep_next anchor: FPL's own prediction is the strongest single signal
            if player.ep_next > 0:
                base_exp = player.ep_next
            else:
                base_exp = player.form * 1.5  # Fallback

            # Fixture quality multiplier
            fixture_mult = (5 - fixture_run.avg_difficulty) / 4 + 0.5  # 0.5-1.25x

            # Position bonus
            pos_mult = {"GKP": 0.7, "DEF": 0.9, "MID": 1.0, "FWD": 1.1}.get(
                player.position, 1.0
            )

            # Ceiling analysis: dreamteam frequency indicates explosive potential
            if current_gw > 0 and player.dreamteam_count > 0:
                explosive_rate = player.dreamteam_count / current_gw
                # 0-30% bonus for explosive players
                ceiling_bonus = 1.0 + min(0.3, explosive_rate * 3.0)
            else:
                ceiling_bonus = 1.0

            # Quality-weighted DGW: scale by fixture quality instead of flat 1.8x
            if fixture_run.has_double:
                dgw_fixtures = [
                    diff for gw, _, diff, _ in fixture_run.fixtures
                    if sum(1 for g, _, _, _ in fixture_run.fixtures if g == gw) > 1
                    or fixture_run.has_double
                ]
                if dgw_fixtures:
                    avg_dgw_diff = sum(dgw_fixtures) / len(dgw_fixtures)
                    # Easy double (avg diff ~2) → 1.9x, hard double (avg diff ~4) → 1.5x
                    dgw_mult = 2.1 - avg_dgw_diff * 0.15
                    dgw_mult = max(1.5, min(1.9, dgw_mult))
                else:
                    dgw_mult = 1.8
                base_exp *= dgw_mult

            expected_pts = base_exp * fixture_mult * pos_mult * ceiling_bonus

            # Probability-based doubt penalty using availability multiplier
            if sp.availability == "doubt":
                expected_pts *= sp.availability_multiplier

            # League spy adjustments
            if self.league_intel:
                if player.id in self.league_intel.captain_fades:
                    expected_pts *= 0.85
                elif player.id in self.league_intel.captain_targets:
                    expected_pts *= 1.15

                # Global vs league captain boost: if the world backs them
                # but your league doesn't captain them, that's maximum exploitation
                safe_diff_ids = {e.player_id for e in self.league_intel.safe_differentials}
                if player.id in safe_diff_ids:
                    captain_count = self.league_intel.league_captains.get(player.id, 0)
                    if captain_count <= 1:
                        expected_pts *= 1.30  # World-backed, league-ignored = massive edge
                    else:
                        expected_pts *= 1.15  # World-backed even if a few rivals captain

                # Trending hidden as captain = bold but world-validated move
                trending_ids = {e.player_id for e in self.league_intel.trending_hidden}
                if player.id in trending_ids:
                    captain_count = self.league_intel.league_captains.get(player.id, 0)
                    if captain_count == 0:
                        expected_pts *= 1.25  # Nobody in league captaining, world is moving here

            # Generate fixture info
            fixtures_str = " + ".join(
                f"{opp}" for _, opp, diff, _ in fixture_run.fixtures[:2]
            )

            # Generate reason with richer detail
            reasons = []
            if player.ep_next > 0:
                reasons.append(f"FPL predicts {player.ep_next:.1f} pts")
            if player.form >= 6:
                reasons.append("excellent form")
            elif player.form >= 4:
                reasons.append("good form")

            if current_gw > 0 and player.dreamteam_count >= 3:
                reasons.append(f"explosive ({player.dreamteam_count} dreamteams)")

            if fixture_run.avg_difficulty <= 2.5:
                reasons.append("easy fixture")
            if fixture_run.has_double:
                reasons.append("DOUBLE GW")

            if self.league_intel:
                num_rivals = len(self.league_intel.rivals)
                cap_count = self.league_intel.league_captains.get(player.id, 0)
                if player.id in self.league_intel.captain_fades:
                    reasons.append(f"fade: {cap_count}/{num_rivals} rivals captaining")
                elif player.id in self.league_intel.captain_targets:
                    reasons.append(f"differential captain ({cap_count}/{num_rivals} rivals)")

                safe_diff_map = {e.player_id: e for e in self.league_intel.safe_differentials}
                if player.id in safe_diff_map:
                    e = safe_diff_map[player.id]
                    reasons.append(
                        f"EXPLOIT: world owns {e.global_ownership:.0f}%, league only {e.league_ownership:.0f}%"
                    )
                trending_map = {e.player_id: e for e in self.league_intel.trending_hidden}
                if player.id in trending_map:
                    e = trending_map[player.id]
                    xfers = e.transfers_in_event
                    xfers_str = f"{xfers/1000:.0f}k" if xfers >= 1000 else str(xfers)
                    reasons.append(f"TRENDING: +{xfers_str} transfers, league asleep")
            elif player.selected_by_percent > 30:
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

    def get_transfer_plans(
        self,
        current_squad_ids: List[int],
        budget: float = 0.0,
        free_transfers: int = 1,
        max_hits: int = 1,
        min_gain_free: float = 0.5,
        min_gain_hit: float = 1.5,
        risk_level: str = "balanced",
        locked_ids: Optional[Set[int]] = None,
    ) -> List[TransferPlan]:
        """Generate 3 transfer plans: conservative, balanced, aggressive"""
        # Risk multipliers for thresholds
        risk_mult = {"conservative": 1.5, "balanced": 1.0, "aggressive": 0.5}.get(
            risk_level, 1.0
        )

        plans = []

        # Plan A: Conservative — only free transfers, no hits
        conservative = self.get_transfer_recommendations(
            current_squad_ids,
            budget=budget,
            free_transfers=free_transfers,
            limit=free_transfers,
            min_gain_free=min_gain_free * risk_mult,
            min_gain_hit=999,  # effectively disallow hits
            locked_ids=locked_ids,
        )
        cons_gain = sum(t.score_gain for t in conservative)
        plans.append(TransferPlan(
            label="conservative",
            display_label="A: CONSERVATIVE",
            transfers=conservative,
            total_score_gain=cons_gain,
            hit_cost=0,
            is_recommended=False,
        ))

        # Plan B: Balanced — free + up to max_hits
        balanced_limit = free_transfers + max_hits
        balanced = self.get_transfer_recommendations(
            current_squad_ids,
            budget=budget,
            free_transfers=free_transfers,
            limit=balanced_limit,
            min_gain_free=min_gain_free * risk_mult,
            min_gain_hit=min_gain_hit * risk_mult,
            locked_ids=locked_ids,
        )
        bal_hits = max(0, len(balanced) - free_transfers)
        bal_gain = sum(t.score_gain for t in balanced)
        plans.append(TransferPlan(
            label="balanced",
            display_label="B: BALANCED",
            transfers=balanced,
            total_score_gain=bal_gain,
            hit_cost=bal_hits * -4,
            is_recommended=True,
        ))

        # Plan C: Aggressive — more transfers, lower thresholds
        agg_limit = free_transfers + max(max_hits, 2)
        agg_mult = risk_mult * 0.5  # even lower thresholds
        aggressive = self.get_transfer_recommendations(
            current_squad_ids,
            budget=budget,
            free_transfers=free_transfers,
            limit=agg_limit,
            min_gain_free=min_gain_free * agg_mult,
            min_gain_hit=min_gain_hit * agg_mult,
            locked_ids=locked_ids,
        )
        agg_hits = max(0, len(aggressive) - free_transfers)
        agg_gain = sum(t.score_gain for t in aggressive)
        plans.append(TransferPlan(
            label="aggressive",
            display_label="C: AGGRESSIVE",
            transfers=aggressive,
            total_score_gain=agg_gain,
            hit_cost=agg_hits * -4,
            is_recommended=False,
        ))

        return plans

    def find_player_by_name(self, query: str) -> Optional[Player]:
        """Find a player by case-insensitive name substring match.
        Prefers exact match on web_name, then shortest substring match."""
        query_lower = query.lower().strip()
        if not query_lower:
            return None

        # Exact match first
        for p in self.fpl.get_all_players():
            if p.web_name.lower() == query_lower:
                return p

        # Substring matches — prefer shortest web_name (most specific)
        matches = []
        for p in self.fpl.get_all_players():
            if query_lower in p.web_name.lower():
                matches.append(p)

        if matches:
            matches.sort(key=lambda p: len(p.web_name))
            return matches[0]

        return None

    def get_free_hit_squad(
        self,
        current_squad_ids: List[int],
        total_budget: float = 100.0,
    ) -> Tuple[List[ScoredPlayer], List[TransferRecommendation]]:
        """Build an optimal 15-player Free Hit squad from scratch.

        Uses a greedy algorithm: score all players, sort by overall_score,
        and greedily pick the best available respecting position limits,
        3-per-team rule, and total budget.

        Returns (new_squad, transfers_needed) where transfers_needed only
        includes players that differ from current_squad_ids.
        """
        pos_limits = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}

        # Gather top candidates per position
        candidates_by_pos: Dict[str, List[ScoredPlayer]] = {}
        all_candidates: List[ScoredPlayer] = []
        for pos in pos_limits:
            candidates = self.scorer.get_top_players(
                position=pos, limit=50, exclude_unavailable=True
            )
            candidates_by_pos[pos] = candidates
            all_candidates.extend(candidates)

        # Sort by overall score descending
        all_candidates.sort(key=lambda sp: sp.overall_score, reverse=True)

        # Greedy selection
        selected: List[ScoredPlayer] = []
        selected_ids: set = set()
        pos_counts: Dict[str, int] = {pos: 0 for pos in pos_limits}
        team_counts: Dict[int, int] = {}
        remaining_budget = total_budget
        total_slots = 15

        def _calc_reserved() -> float:
            """Calculate minimum budget to fill remaining unfilled slots,
            using the cheapest non-selected candidates per position."""
            reserved = 0.0
            for p, limit in pos_limits.items():
                need = limit - pos_counts[p]
                if need <= 0:
                    continue
                # Get prices of non-selected candidates in this position
                avail_prices = sorted(
                    sp.player.price for sp in candidates_by_pos[p]
                    if sp.player.id not in selected_ids
                )
                # Sum the cheapest `need` prices
                for i in range(min(need, len(avail_prices))):
                    reserved += avail_prices[i]
                # If not enough candidates, use a high fallback
                if len(avail_prices) < need:
                    reserved += (need - len(avail_prices)) * 5.0
            return reserved

        for sp in all_candidates:
            if len(selected) >= total_slots:
                break

            pos = sp.player.position
            tid = sp.player.team_id
            price = sp.player.price

            # Skip if position full
            if pos_counts[pos] >= pos_limits[pos]:
                continue

            # Skip if 3-per-team limit reached
            if team_counts.get(tid, 0) >= 3:
                continue

            # Check: after picking this player, can we still fill remaining slots?
            # Temporarily add to selected to get accurate reservation
            selected_ids.add(sp.player.id)
            pos_counts[pos] += 1
            reserved = _calc_reserved()
            pos_counts[pos] -= 1
            selected_ids.discard(sp.player.id)

            if price + reserved > remaining_budget:
                continue

            # Select this player
            selected.append(sp)
            selected_ids.add(sp.player.id)
            pos_counts[pos] += 1
            team_counts[tid] = team_counts.get(tid, 0) + 1
            remaining_budget -= price

        # Fallback: fill any remaining positions with cheapest available players
        if len(selected) < total_slots:
            for pos in pos_limits:
                if pos_counts[pos] >= pos_limits[pos]:
                    continue
                cheap = sorted(candidates_by_pos[pos], key=lambda sp: sp.player.price)
                for sp in cheap:
                    if pos_counts[pos] >= pos_limits[pos]:
                        break
                    if sp.player.id in selected_ids:
                        continue
                    if team_counts.get(sp.player.team_id, 0) >= 3:
                        continue
                    if sp.player.price > remaining_budget:
                        continue
                    selected.append(sp)
                    selected_ids.add(sp.player.id)
                    pos_counts[pos] += 1
                    team_counts[sp.player.team_id] = team_counts.get(sp.player.team_id, 0) + 1
                    remaining_budget -= sp.player.price

        # Build transfer list: pair current squad out → new squad in by position
        current_set = set(current_squad_ids)
        new_set = {sp.player.id for sp in selected}

        # Players going out (in current but not in new)
        outs = []
        for pid in current_squad_ids:
            if pid not in new_set:
                player = self.fpl.get_player(pid)
                if player:
                    outs.append(self.scorer.score_player(player))

        # Players coming in (in new but not in current)
        ins = []
        for sp in selected:
            if sp.player.id not in current_set:
                ins.append(sp)

        # Pair by position
        transfers: List[TransferRecommendation] = []
        ins_by_pos: Dict[str, List[ScoredPlayer]] = {}
        for sp in ins:
            ins_by_pos.setdefault(sp.player.position, []).append(sp)

        outs_by_pos: Dict[str, List[ScoredPlayer]] = {}
        for sp in outs:
            outs_by_pos.setdefault(sp.player.position, []).append(sp)

        for pos in pos_limits:
            pos_ins = ins_by_pos.get(pos, [])
            pos_outs = outs_by_pos.get(pos, [])
            # Sort outs worst-first, ins best-first
            pos_outs.sort(key=lambda sp: sp.overall_score)
            pos_ins.sort(key=lambda sp: sp.overall_score, reverse=True)

            for out_sp, in_sp in zip(pos_outs, pos_ins):
                gain = in_sp.overall_score - out_sp.overall_score
                price_diff = in_sp.player.price - out_sp.player.price
                transfers.append(TransferRecommendation(
                    player_out=out_sp,
                    player_in=in_sp,
                    score_gain=gain,
                    price_diff=price_diff,
                    reason="free hit rebuild",
                ))

        transfers.sort(key=lambda t: t.score_gain, reverse=True)
        return selected, transfers

    def format_free_hit_squad(
        self,
        squad: List[ScoredPlayer],
        transfers: List[TransferRecommendation],
        total_budget: float,
    ) -> str:
        """Format a Free Hit squad for terminal display."""
        pos_order = ["GKP", "DEF", "MID", "FWD"]
        by_pos: Dict[str, List[ScoredPlayer]] = {p: [] for p in pos_order}
        for sp in squad:
            by_pos[sp.player.position].append(sp)

        squad_cost = sum(sp.player.price for sp in squad)
        remaining = total_budget - squad_cost

        lines = ["\n   FREE HIT SQUAD"]
        lines.append("   " + "-" * 50)

        for pos in pos_order:
            players = sorted(by_pos[pos], key=lambda sp: sp.overall_score, reverse=True)
            lines.append(f"\n   {pos}:")
            for sp in players:
                p = sp.player
                fixture_run = self.fixtures.get_fixture_run(p.team_id)
                next_fix = fixture_run.fixtures[0][1] if fixture_run.fixtures else "?"
                lines.append(
                    f"      {p.web_name:15} ({p.team:3}) "
                    f"£{p.price}m | Score: {sp.overall_score:.1f} | "
                    f"Form: {p.form} | Next: {next_fix}"
                )

        lines.append(f"\n   Squad cost: £{squad_cost:.1f}m | "
                      f"Remaining: £{remaining:.1f}m")

        if transfers:
            lines.append(f"\n   Transfers needed ({len(transfers)}):")
            for tr in transfers:
                out_p = tr.player_out.player
                in_p = tr.player_in.player
                lines.append(
                    f"      {out_p.web_name} ({out_p.team}) -> "
                    f"{in_p.web_name} ({in_p.team}) [{tr.score_gain:+.1f}]"
                )

        return "\n".join(lines)

    def format_transfer_plans(self, plans: List[TransferPlan]) -> str:
        """Format transfer plans for terminal display"""
        lines = []
        for plan in plans:
            rec_tag = " (RECOMMENDED)" if plan.is_recommended else ""
            lines.append(f"\n   --- {plan.display_label}{rec_tag} ---")
            if not plan.transfers:
                lines.append("   No transfers")
            else:
                for i, tr in enumerate(plan.transfers, 1):
                    out_p = tr.player_out.player
                    in_p = tr.player_in.player
                    price_str = (
                        f"+{tr.price_diff:.1f}m"
                        if tr.price_diff > 0
                        else f"{tr.price_diff:.1f}m"
                    )
                    lines.append(
                        f"   {i}. {out_p.web_name} ({out_p.team}) -> "
                        f"{in_p.web_name} ({in_p.team})  "
                        f"[{price_str} | +{tr.score_gain:.1f}]"
                    )
                    lines.append(f"      {tr.reason}")
            hit_str = f" ({plan.hit_cost} pts)" if plan.hit_cost < 0 else " (free)"
            lines.append(
                f"   Total: +{plan.total_score_gain:.1f} score gain{hit_str}"
            )
        return "\n".join(lines)

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

        # League Intel
        if self.league_intel:
            lines.append("🕵️ LEAGUE INTEL")
            lines.append("-" * 40)
            lines.append(
                f"   {self.league_intel.league_name} | "
                f"Rank: {self.league_intel.your_rank}/{self.league_intel.total_managers} | "
                f"{self.league_intel.points_to_leader} pts behind leader"
            )

            # Captain advice
            num_rivals = len(self.league_intel.rivals)
            if self.league_intel.captain_fades:
                fade_names = []
                for pid in self.league_intel.captain_fades:
                    p = self.fpl.get_player(pid)
                    if p:
                        count = self.league_intel.league_captains.get(pid, 0)
                        fade_names.append(f"{p.web_name} ({count}/{num_rivals})")
                if fade_names:
                    lines.append(f"   Captain FADE: {', '.join(fade_names)}")

            if self.league_intel.captain_targets:
                target_names = []
                for pid in self.league_intel.captain_targets[:3]:
                    p = self.fpl.get_player(pid)
                    if p:
                        count = self.league_intel.league_captains.get(pid, 0)
                        target_names.append(f"{p.web_name} ({count}/{num_rivals})")
                if target_names:
                    lines.append(f"   Captain TARGET: {', '.join(target_names)}")

            # Top league differentials
            if self.league_intel.differential_vs_league:
                diff_entries = []
                for pid in self.league_intel.differential_vs_league[:5]:
                    p = self.fpl.get_player(pid)
                    if p:
                        pct = self.league_intel.league_ownership.get(pid, 0)
                        diff_entries.append((p, pct))
                diff_entries.sort(key=lambda x: x[0].form, reverse=True)
                if diff_entries:
                    lines.append("   League differentials:")
                    for p, pct in diff_entries[:3]:
                        lines.append(
                            f"      {p.web_name} ({p.team}) form:{p.form} - {pct:.0f}% of rivals"
                        )
            lines.append("")

        # Global vs League Exploitation
        if self.league_intel and (
            self.league_intel.safe_differentials
            or self.league_intel.league_overweight
            or self.league_intel.trending_hidden
        ):
            lines.append("🌍 GLOBAL vs LEAGUE EXPLOITATION")
            lines.append("-" * 40)

            if self.league_intel.safe_differentials:
                lines.append("   SAFE DIFFERENTIALS (world owns, your league doesn't):")
                for entry in self.league_intel.safe_differentials[:4]:
                    lines.append(
                        f"      {entry.web_name:15} ({entry.team}) "
                        f"Global: {entry.global_ownership:.0f}% | League: {entry.league_ownership:.0f}% | "
                        f"Form: {entry.form}"
                    )

            if self.league_intel.league_overweight:
                lines.append("   RIVALS OVER-INDEXING (league owns, world doesn't):")
                for entry in self.league_intel.league_overweight[:4]:
                    lines.append(
                        f"      {entry.web_name:15} ({entry.team}) "
                        f"League: {entry.league_ownership:.0f}% | Global: {entry.global_ownership:.0f}% | "
                        f"Form: {entry.form}"
                    )

            if self.league_intel.trending_hidden:
                lines.append("   TRENDING HIDDEN GEMS (world moving to, nobody owns yet):")
                for entry in self.league_intel.trending_hidden[:4]:
                    xfers = entry.transfers_in_event
                    xfers_str = f"{xfers/1000:.0f}k" if xfers >= 1000 else str(xfers)
                    lines.append(
                        f"      {entry.web_name:15} ({entry.team}) "
                        f"+{xfers_str} transfers | {entry.global_ownership:.0f}% owned | "
                        f"Form: {entry.form}"
                    )

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
