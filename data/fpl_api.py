"""
FPL API data fetching using the fpl library
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd

import config
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
    bonus: int
    bps: int  # Bonus points system score
    penalties_order: Optional[int]  # None = not on pens, 1 = first choice
    corners_and_indirect_freekicks_order: Optional[int]
    direct_freekicks_order: Optional[int]
    chance_of_playing: Optional[int]
    news: str
    status: str  # 'a' = available, 'd' = doubtful, 'i' = injured, 's' = suspended, 'u' = unavailable
    # New fields for model intelligence
    ep_next: float = 0.0  # FPL's expected points for next GW
    ep_this: float = 0.0  # FPL's expected points this GW
    value_form: float = 0.0  # Points per £m over last 4 GWs (FPL-calculated)
    value_season: float = 0.0  # Points per £m over season
    xg_per_90: float = 0.0  # Expected goals per 90
    xa_per_90: float = 0.0  # Expected assists per 90
    xgi_per_90: float = 0.0  # Expected goal involvements per 90
    transfers_in_event: int = 0  # Transfers in this GW
    transfers_out_event: int = 0  # Transfers out this GW
    cost_change_event: int = 0  # Price change this GW (tenths)
    goals_conceded: int = 0  # Goals conceded (DEF/GKP)
    saves: int = 0  # Saves (GKP)
    clean_sheets_per_90: float = 0.0  # Normalized clean sheet rate
    event_points: int = 0  # Last GW actual points
    dreamteam_count: int = 0  # Elite performance frequency


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
    # New fields for model intelligence
    form: Optional[float] = None  # Team form (null early season)
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    points: int = 0  # League points
    position: int = 0  # League table position


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
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create a shared aiohttp session (reuses TCP connections)"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the shared session"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def fetch_bootstrap(self) -> Dict[str, Any]:
        """Fetch main bootstrap-static data"""
        cached = self.cache.get("bootstrap")
        if cached:
            return cached

        session = await self._get_session()
        async with session.get(f"{self.BASE_URL}/bootstrap-static/") as resp:
            data = await resp.json()
            self.cache.set("bootstrap", data)
            return data

    async def fetch_fixtures(self) -> List[Dict]:
        """Fetch all fixtures"""
        cached = self.cache.get("fixtures")
        if cached:
            return cached

        session = await self._get_session()
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

        session = await self._get_session()
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

        session = await self._get_session()
        url = f"{self.BASE_URL}/entry/{team_id}/event/{gameweek}/picks/"
        async with session.get(url) as resp:
            if resp.status == 404:
                return {}
            data = await resp.json()
            self.cache.set(cache_key, data)
            return data

    async def fetch_team_transfers(self, team_id: int) -> List[Dict[str, Any]]:
        """Fetch all transfers made by a manager (public endpoint, no auth needed)"""
        session = await self._get_session()
        url = f"{self.BASE_URL}/entry/{team_id}/transfers/"
        async with session.get(url) as resp:
            if resp.status != 200:
                return []
            return await resp.json()

    async def fetch_current_squad(self, team_id: int) -> Optional[List[dict]]:
        """Fetch the ACTUAL current squad, accounting for pending transfers.

        The picks endpoint returns the squad at GW deadline. If transfers have
        already been made for the next GW, those are applied on top to get the
        real current squad.
        """
        current_gw = self.get_current_gameweek()
        if config.DEBUG:
            print(f"   [debug] fetch_current_squad: current_gw={current_gw}")

        picks_data = await self.fetch_team_picks(team_id, current_gw)

        if not picks_data or "picks" not in picks_data:
            if config.DEBUG:
                print(f"   [debug] No picks for GW{current_gw}, trying GW{current_gw - 1}")
            picks_data = await self.fetch_team_picks(team_id, current_gw - 1)

        if not picks_data or "picks" not in picks_data:
            if config.DEBUG:
                print("   [debug] No picks data found at all")
            return None

        picks = list(picks_data["picks"])
        base_ids = [p["element"] for p in picks]
        if config.DEBUG:
            print(f"   [debug] Base squad from picks endpoint: {base_ids}")

        # Fetch transfers and apply any made for the next GW
        transfers = await self.fetch_team_transfers(team_id)
        if config.DEBUG:
            print(f"   [debug] Total transfers returned by API: {len(transfers)}")

            # Show the most recent transfers for debugging
            if transfers:
                recent = transfers[:6]  # API returns newest first
                for t in recent:
                    print(
                        f"   [debug]   transfer: event={t.get('event')} "
                        f"out={t.get('element_out')} in={t.get('element_in')} "
                        f"time={t.get('time', '?')}"
                    )

        # Apply transfers for BOTH current GW and next GW
        # (current GW transfers may not be reflected in picks if GW is in progress)
        next_gw = current_gw + 1
        pending = [t for t in transfers if t.get("event") in (current_gw, next_gw)]
        if config.DEBUG:
            print(
                f"   [debug] Pending transfers for GW{current_gw} or GW{next_gw}: {len(pending)}"
            )

        if pending:
            for t in pending:
                out_id = t["element_out"]
                in_id = t["element_in"]
                # Only apply if out_id is still in picks and in_id is not
                current_ids = [p["element"] for p in picks]
                if out_id in current_ids and in_id not in current_ids:
                    for i, p in enumerate(picks):
                        if p["element"] == out_id:
                            picks[i] = dict(p, element=in_id)
                            break
                    if config.DEBUG:
                        print(f"   [debug]   Applied: {out_id} -> {in_id}")
                else:
                    if config.DEBUG:
                        print(
                            f"   [debug]   Skipped (already applied): {out_id} -> {in_id}"
                        )

            if config.DEBUG:
                applied_ids = [p["element"] for p in picks]
                print(f"   [debug] Adjusted squad: {applied_ids}")
        else:
            if config.DEBUG:
                print("   [debug] No pending transfers found — squad unchanged")

        return picks

    async def fetch_player_history(self, player_id: int) -> Dict[str, Any]:
        """Fetch a player's detailed history"""
        cache_key = f"player_{player_id}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        session = await self._get_session()
        async with session.get(
            f"{self.BASE_URL}/element-summary/{player_id}/"
        ) as resp:
            data = await resp.json()
            self.cache.set(cache_key, data)
            return data

    async def load_all_data(self) -> None:
        """Load and process all required data"""
        self._bootstrap_data, self._fixtures_data = await asyncio.gather(
            self.fetch_bootstrap(), self.fetch_fixtures()
        )
        self._process_teams()
        self._process_players()

    def _process_teams(self) -> None:
        """Process team data from bootstrap"""
        if not self._bootstrap_data:
            return

        for t in self._bootstrap_data["teams"]:
            # Parse form safely — API returns null early season
            raw_form = t.get("form")
            team_form = float(raw_form) if raw_form is not None else None

            team = Team(
                id=t["id"],
                name=t["name"],
                short_name=t["short_name"],
                strength=t["strength"],
                strength_attack_home=t["strength_attack_home"],
                strength_attack_away=t["strength_attack_away"],
                strength_defence_home=t["strength_defence_home"],
                strength_defence_away=t["strength_defence_away"],
                form=team_form,
                played=t.get("played", 0),
                wins=t.get("win", 0),
                draws=t.get("draw", 0),
                losses=t.get("loss", 0),
                points=t.get("points", 0),
                position=t.get("position", 0),
            )
            self._teams[team.id] = team

    def _process_players(self) -> None:
        """Process player data from bootstrap"""
        if not self._bootstrap_data:
            return

        for p in self._bootstrap_data["elements"]:
            team = self._teams.get(p["team"])
            minutes = p["minutes"]

            # Calculate per-90 stats safely
            if minutes >= 90:
                nineties = minutes / 90
                xg_per_90 = float(p.get("expected_goals") or 0) / nineties
                xa_per_90 = float(p.get("expected_assists") or 0) / nineties
                xgi_per_90 = float(p.get("expected_goal_involvements") or 0) / nineties
                cs_per_90 = p.get("clean_sheets", 0) / nineties
            else:
                xg_per_90 = 0.0
                xa_per_90 = 0.0
                xgi_per_90 = 0.0
                cs_per_90 = 0.0

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
                minutes=minutes,
                goals_scored=p["goals_scored"],
                assists=p["assists"],
                clean_sheets=p["clean_sheets"],
                expected_goals=float(p["expected_goals"] or 0),
                expected_assists=float(p["expected_assists"] or 0),
                expected_goal_involvements=float(p["expected_goal_involvements"] or 0),
                influence=float(p["influence"]),
                creativity=float(p["creativity"]),
                threat=float(p["threat"]),
                ict_index=float(p["ict_index"]),
                starts=p["starts"],
                bonus=p["bonus"],
                bps=p["bps"],
                penalties_order=p.get("penalties_order"),
                corners_and_indirect_freekicks_order=p.get(
                    "corners_and_indirect_freekicks_order"
                ),
                direct_freekicks_order=p.get("direct_freekicks_order"),
                chance_of_playing=p["chance_of_playing_next_round"],
                news=p["news"],
                status=p["status"],
                ep_next=float(p.get("ep_next") or 0),
                ep_this=float(p.get("ep_this") or 0),
                value_form=float(p.get("value_form") or 0),
                value_season=float(p.get("value_season") or 0),
                xg_per_90=xg_per_90,
                xa_per_90=xa_per_90,
                xgi_per_90=xgi_per_90,
                transfers_in_event=p.get("transfers_in_event", 0),
                transfers_out_event=p.get("transfers_out_event", 0),
                cost_change_event=p.get("cost_change_event", 0),
                goals_conceded=p.get("goals_conceded", 0),
                saves=p.get("saves", 0),
                clean_sheets_per_90=cs_per_90,
                event_points=p.get("event_points", 0),
                dreamteam_count=p.get("dreamteam_count", 0),
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

    def get_fixtures_for_team(self, team_id: int, gameweeks: int = 5) -> List[Fixture]:
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
