"""
Fixture difficulty analysis
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from data.fpl_api import FPLDataFetcher, Fixture, Team

import config


@dataclass
class FixtureRun:
    """Analysis of upcoming fixtures for a team"""

    team_id: int
    team_name: str
    fixtures: List[Tuple[int, str, int, bool]]  # (gw, opponent, difficulty, is_home)
    avg_difficulty: float
    easy_count: int  # FDR 1-2
    hard_count: int  # FDR 4-5
    has_double: bool
    has_blank: bool


class FixtureAnalyzer:
    """Analyzes fixture difficulty for teams and players"""

    def __init__(self, fpl_data: FPLDataFetcher):
        self.fpl = fpl_data
        self.lookahead = config.FIXTURE_LOOKAHEAD
        self._dgw_data: Optional[Dict[int, List[int]]] = None
        self._bgw_data: Optional[Dict[int, List[int]]] = None

    def get_fixture_run(self, team_id: int) -> FixtureRun:
        """Get detailed fixture analysis for a team"""
        team = self.fpl.get_team(team_id)
        fixtures = self.fpl.get_fixtures_for_team(team_id, self.lookahead)

        fixture_list = []
        difficulties = []

        for f in fixtures:
            is_home = f.home_team_id == team_id
            opponent_id = f.away_team_id if is_home else f.home_team_id
            opponent = self.fpl.get_team(opponent_id)
            difficulty = f.home_team_difficulty if is_home else f.away_team_difficulty

            fixture_list.append(
                (
                    f.gameweek,
                    f"{opponent.short_name}({'H' if is_home else 'A'})",
                    difficulty,
                    is_home,
                )
            )
            difficulties.append(difficulty)

        avg_diff = sum(difficulties) / len(difficulties) if difficulties else 3.0
        easy_count = sum(1 for d in difficulties if d <= 2)
        hard_count = sum(1 for d in difficulties if d >= 4)

        # Check for DGW/BGW
        dgw = self.get_double_gameweeks()
        bgw = self.get_blank_gameweeks()
        current_gw = self.fpl.get_current_gameweek()

        has_double = any(
            team_id in dgw.get(gw, [])
            for gw in range(current_gw, current_gw + self.lookahead)
        )
        has_blank = any(
            team_id in bgw.get(gw, [])
            for gw in range(current_gw, current_gw + self.lookahead)
        )

        return FixtureRun(
            team_id=team_id,
            team_name=team.name if team else "Unknown",
            fixtures=fixture_list,
            avg_difficulty=avg_diff,
            easy_count=easy_count,
            hard_count=hard_count,
            has_double=has_double,
            has_blank=has_blank,
        )

    def get_all_fixture_runs(self) -> List[FixtureRun]:
        """Get fixture runs for all teams, sorted by easiest"""
        runs = []
        for team in self.fpl.get_all_teams():
            runs.append(self.get_fixture_run(team.id))

        return sorted(runs, key=lambda x: x.avg_difficulty)

    def get_double_gameweeks(self) -> Dict[int, List[int]]:
        """Get double gameweeks (cached)"""
        if self._dgw_data is None:
            self._dgw_data = self.fpl.get_double_gameweeks()
        return self._dgw_data

    def get_blank_gameweeks(self) -> Dict[int, List[int]]:
        """Get blank gameweeks (cached)"""
        if self._bgw_data is None:
            self._bgw_data = self.fpl.get_blank_gameweeks()
        return self._bgw_data

    def get_fixture_ease_score(self, team_id: int) -> float:
        """
        Calculate fixture ease score (0-10, higher = easier)
        Used in player scoring algorithm
        """
        run = self.get_fixture_run(team_id)

        # Invert difficulty (5 = hardest becomes 0, 1 = easiest becomes 10)
        base_score = (5 - run.avg_difficulty) * 2

        # Bonus for easy fixtures
        base_score += run.easy_count * 0.5

        # Penalty for hard fixtures
        base_score -= run.hard_count * 0.5

        # Bonus for double gameweek
        if run.has_double:
            base_score += 1.0

        # Penalty for blank gameweek
        if run.has_blank:
            base_score -= 1.5

        return max(0, min(10, base_score))

    def format_fixture_run(self, run: FixtureRun) -> str:
        """Format fixture run for display"""
        fixtures_str = " ".join(
            f"{opp}({diff})" for _, opp, diff, _ in run.fixtures
        )
        extras = []
        if run.has_double:
            extras.append("DGW")
        if run.has_blank:
            extras.append("BGW")

        extras_str = f" [{', '.join(extras)}]" if extras else ""

        return f"{run.team_name}: {fixtures_str} (avg {run.avg_difficulty:.1f}){extras_str}"

    def get_best_fixtures(self, top_n: int = 10) -> List[FixtureRun]:
        """Get teams with the best upcoming fixtures"""
        return self.get_all_fixture_runs()[:top_n]

    def get_worst_fixtures(self, top_n: int = 10) -> List[FixtureRun]:
        """Get teams with the worst upcoming fixtures"""
        return self.get_all_fixture_runs()[-top_n:][::-1]

    def get_dgw_teams(self, gameweek: int) -> List[Team]:
        """Get teams with a double in a specific gameweek"""
        dgw = self.get_double_gameweeks()
        team_ids = dgw.get(gameweek, [])
        return [self.fpl.get_team(tid) for tid in team_ids if self.fpl.get_team(tid)]

    def get_bgw_teams(self, gameweek: int) -> List[Team]:
        """Get teams with a blank in a specific gameweek"""
        bgw = self.get_blank_gameweeks()
        team_ids = bgw.get(gameweek, [])
        return [self.fpl.get_team(tid) for tid in team_ids if self.fpl.get_team(tid)]
