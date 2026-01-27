"""
FPL API data fetching using the fpl library
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd

from .cache import Cache


@dataclass
class Player:
    """Player data container"""

    id: int
    name: str
    web_name: str
    team: str
    team_id: int
    position: str
    price: float
    total_points: int
    form: float
    points_per_game: float
    selected_by_percent: float
    minutes: int
    goals_scored: int
    assists: int
    clean_sheets: int
    expected_goals: float
    expected_assists: float
    expected_goal_involvements: float
    influence: float
    creativity: float
    threat: float
    ict_index: float
    starts: int
    chance_of_playing: Optional[int]
    news: str
    status: str  # 'a' = available, 'd' = doubtful, 'i' = injured, 's' = suspended, 'u' = unavailable


@dataclass
class Team:
    """Team data container"""

    id: int
    name: str
    short_name: str
    strength: int
    strength_attack_home: int
    strength_attack_away: int
    strength_defence_home: int
    strength_defence_away: int


@dataclass
class Fixture:
    """Fixture data container"""

    id: int
    gameweek: int
    home_team_id: int
    away_team_id: int
    home_team_difficulty: int
    away_team_difficulty: int
    finished: bool
    home_score: Optional[int]
    away_score: Optional[int]


class FPLDataFetcher:
    """Fetches and processes FPL API data"""

    BASE_URL = "https://fantasy.premierleague.com/api"

    POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

    def __init__(self):
        self.cache = Cache()
        self._bootstrap_data: Optional[Dict] = None
        self._fixtures_data: Optional[List] = None
        self._teams: Dict[int, Team] = {}
        self._players: Dict[int, Player] = {}

    async def fetch_bootstrap(self) -> Dict[str, Any]:
        """Fetch main bootstrap-static data"""
        cached = self.cache.get("bootstrap")
        if cached:
            return cached

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.BASE_URL}/bootstrap-static/") as resp:
                data = await resp.json()
                self.cache.set("bootstrap", data)
                return data

    async def fetch_fixtures(self) -> List[Dict]:
        """Fetch all fixtures"""
        cached = self.cache.get("fixtures")
        if cached:
            return cached

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.BASE_URL}/fixtures/") as resp:
                data = await resp.json()
                self.cache.set("fixtures", data)
                return data

    async def fetch_team_data(self, team_id: int) -> Dict[str, Any]:
        """Fetch a specific manager's team data"""
        cache_key = f"team_{team_id}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.BASE_URL}/entry/{team_id}/") as resp:
                data = await resp.json()
                self.cache.set(cache_key, data)
                return data

    async def fetch_team_picks(self, team_id: int, gameweek: int) -> Dict[str, Any]:
        """Fetch a manager's picks for a specific gameweek"""
        cache_key = f"picks_{team_id}_{gameweek}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        async with aiohttp.ClientSession() as session:
            url = f"{self.BASE_URL}/entry/{team_id}/event/{gameweek}/picks/"
            async with session.get(url) as resp:
                if resp.status == 404:
                    return {}
                data = await resp.json()
                self.cache.set(cache_key, data)
                return data

    async def fetch_player_history(self, player_id: int) -> Dict[str, Any]:
        """Fetch a player's detailed history"""
        cache_key = f"player_{player_id}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.BASE_URL}/element-summary/{player_id}/"
            ) as resp:
                data = await resp.json()
                self.cache.set(cache_key, data)
                return data

    async def load_all_data(self) -> None:
        """Load and process all required data"""
        self._bootstrap_data = await self.fetch_bootstrap()
        self._fixtures_data = await self.fetch_fixtures()
        self._process_teams()
        self._process_players()

    def _process_teams(self) -> None:
        """Process team data from bootstrap"""
        if not self._bootstrap_data:
            return

        for t in self._bootstrap_data["teams"]:
            team = Team(
                id=t["id"],
                name=t["name"],
                short_name=t["short_name"],
                strength=t["strength"],
                strength_attack_home=t["strength_attack_home"],
                strength_attack_away=t["strength_attack_away"],
                strength_defence_home=t["strength_defence_home"],
                strength_defence_away=t["strength_defence_away"],
            )
            self._teams[team.id] = team

    def _process_players(self) -> None:
        """Process player data from bootstrap"""
        if not self._bootstrap_data:
            return

        for p in self._bootstrap_data["elements"]:
            team = self._teams.get(p["team"])
            player = Player(
                id=p["id"],
                name=f"{p['first_name']} {p['second_name']}",
                web_name=p["web_name"],
                team=team.short_name if team else "???",
                team_id=p["team"],
                position=self.POSITIONS.get(p["element_type"], "???"),
                price=p["now_cost"] / 10,
                total_points=p["total_points"],
                form=float(p["form"]),
                points_per_game=float(p["points_per_game"]),
                selected_by_percent=float(p["selected_by_percent"]),
                minutes=p["minutes"],
                goals_scored=p["goals_scored"],
                assists=p["assists"],
                clean_sheets=p["clean_sheets"],
                expected_goals=float(p["expected_goals"] or 0),
                expected_assists=float(p["expected_assists"] or 0),
                expected_goal_involvements=float(
                    p["expected_goal_involvements"] or 0
                ),
                influence=float(p["influence"]),
                creativity=float(p["creativity"]),
                threat=float(p["threat"]),
                ict_index=float(p["ict_index"]),
                starts=p["starts"],
                chance_of_playing=p["chance_of_playing_next_round"],
                news=p["news"],
                status=p["status"],
            )
            self._players[player.id] = player

    def get_current_gameweek(self) -> int:
        """Get the current gameweek number"""
        if not self._bootstrap_data:
            return 1

        for event in self._bootstrap_data["events"]:
            if event["is_current"]:
                return event["id"]

        # If no current, find next
        for event in self._bootstrap_data["events"]:
            if event["is_next"]:
                return event["id"]

        return 1

    def get_all_players(self) -> List[Player]:
        """Get all players"""
        return list(self._players.values())

    def get_player(self, player_id: int) -> Optional[Player]:
        """Get a specific player by ID"""
        return self._players.get(player_id)

    def get_players_by_position(self, position: str) -> List[Player]:
        """Get all players of a specific position"""
        return [p for p in self._players.values() if p.position == position]

    def get_team(self, team_id: int) -> Optional[Team]:
        """Get a specific team by ID"""
        return self._teams.get(team_id)

    def get_all_teams(self) -> List[Team]:
        """Get all teams"""
        return list(self._teams.values())

    def get_fixtures_for_team(
        self, team_id: int, gameweeks: int = 5
    ) -> List[Fixture]:
        """Get upcoming fixtures for a team"""
        if not self._fixtures_data:
            return []

        current_gw = self.get_current_gameweek()
        fixtures = []

        for f in self._fixtures_data:
            gw = f.get("event")
            if gw is None:
                continue
            if gw < current_gw or gw >= current_gw + gameweeks:
                continue
            if f["team_h"] == team_id or f["team_a"] == team_id:
                fixture = Fixture(
                    id=f["id"],
                    gameweek=gw,
                    home_team_id=f["team_h"],
                    away_team_id=f["team_a"],
                    home_team_difficulty=f["team_h_difficulty"],
                    away_team_difficulty=f["team_a_difficulty"],
                    finished=f["finished"],
                    home_score=f.get("team_h_score"),
                    away_score=f.get("team_a_score"),
                )
                fixtures.append(fixture)

        return sorted(fixtures, key=lambda x: x.gameweek)

    def get_fixture_difficulty(self, team_id: int, gameweeks: int = 5) -> float:
        """Calculate average fixture difficulty for upcoming games"""
        fixtures = self.get_fixtures_for_team(team_id, gameweeks)
        if not fixtures:
            return 3.0  # Neutral if no fixtures

        difficulties = []
        for f in fixtures:
            if f.home_team_id == team_id:
                difficulties.append(f.home_team_difficulty)
            else:
                difficulties.append(f.away_team_difficulty)

        return sum(difficulties) / len(difficulties)

    def get_double_gameweeks(self) -> Dict[int, List[int]]:
        """Find gameweeks where teams play twice"""
        if not self._fixtures_data:
            return {}

        gw_counts: Dict[int, Dict[int, int]] = {}  # {gw: {team_id: count}}

        for f in self._fixtures_data:
            gw = f.get("event")
            if gw is None:
                continue

            if gw not in gw_counts:
                gw_counts[gw] = {}

            for team_id in [f["team_h"], f["team_a"]]:
                gw_counts[gw][team_id] = gw_counts[gw].get(team_id, 0) + 1

        # Find DGWs
        dgw_teams: Dict[int, List[int]] = {}
        for gw, teams in gw_counts.items():
            double_teams = [t for t, count in teams.items() if count > 1]
            if double_teams:
                dgw_teams[gw] = double_teams

        return dgw_teams

    def get_blank_gameweeks(self) -> Dict[int, List[int]]:
        """Find gameweeks where teams don't play"""
        if not self._fixtures_data:
            return {}

        current_gw = self.get_current_gameweek()
        all_team_ids = set(self._teams.keys())

        gw_teams: Dict[int, set] = {}
        for f in self._fixtures_data:
            gw = f.get("event")
            if gw is None or gw < current_gw:
                continue

            if gw not in gw_teams:
                gw_teams[gw] = set()

            gw_teams[gw].add(f["team_h"])
            gw_teams[gw].add(f["team_a"])

        # Find BGWs
        bgw_teams: Dict[int, List[int]] = {}
        for gw, playing_teams in gw_teams.items():
            blank_teams = list(all_team_ids - playing_teams)
            if blank_teams:
                bgw_teams[gw] = blank_teams

        return bgw_teams

    def to_dataframe(self) -> pd.DataFrame:
        """Convert player data to pandas DataFrame"""
        players = self.get_all_players()
        return pd.DataFrame([vars(p) for p in players])


# Convenience function to run async code
def get_fpl_data() -> FPLDataFetcher:
    """Fetch all FPL data synchronously"""
    fetcher = FPLDataFetcher()
    asyncio.run(fetcher.load_all_data())
    return fetcher
