"""
League spy - analyzes rival squads in your mini-league
to find strategic advantages.
"""

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import aiohttp

from data.cache import Cache
from data.fpl_api import FPLDataFetcher, Player


@dataclass
class Rival:
    """A mini-league rival"""

    team_id: int
    manager_name: str
    team_name: str
    total_points: int
    rank: int
    squad_ids: List[int] = field(default_factory=list)
    captain_id: Optional[int] = None


@dataclass
class GlobalVsLeague:
    """A player with a discrepancy between global and league ownership"""

    player_id: int
    web_name: str
    team: str
    position: str
    global_ownership: float  # % of all FPL managers
    league_ownership: float  # % of rivals in mini-league
    discrepancy: float  # global - league (positive = world owns more)
    form: float
    transfers_in_event: int
    category: str  # "safe_differential", "league_overweight", "trending_hidden"


@dataclass
class LeagueIntel:
    """Intelligence on your mini-league"""

    league_name: str
    your_rank: int
    total_managers: int
    points_to_leader: int
    rivals: List[Rival]

    # Player ownership within the league
    league_ownership: Dict[int, float]  # player_id -> % of rivals who own them
    league_captains: Dict[int, int]  # player_id -> number of rivals captaining

    # Strategic picks
    differential_vs_league: List[int]  # player_ids nobody/few rivals have
    must_have: List[int]  # player_ids almost everyone has (risky to not own)
    captain_fades: List[int]  # popular captain picks to avoid for differentiation
    captain_targets: List[int]  # good picks that few rivals are captaining

    # Global vs league discrepancy analysis
    safe_differentials: List[GlobalVsLeague] = field(
        default_factory=list
    )  # World owns, league doesn't
    league_overweight: List[GlobalVsLeague] = field(
        default_factory=list
    )  # League over-indexes vs world
    trending_hidden: List[GlobalVsLeague] = field(
        default_factory=list
    )  # High transfer momentum, low ownership


class LeagueSpy:
    """Fetches and analyzes mini-league rival data"""

    BASE_URL = "https://fantasy.premierleague.com/api"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    MAX_RETRIES = 3

    def __init__(self, fpl_data: FPLDataFetcher, your_team_id: int):
        self.fpl = fpl_data
        self.your_team_id = your_team_id
        self.cache = Cache()

    async def _fetch_json(
        self, session: aiohttp.ClientSession, url: str, cache_key: Optional[str] = None
    ) -> Dict:
        """Fetch JSON with caching and exponential backoff for 429/403"""
        if cache_key:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        for attempt in range(self.MAX_RETRIES):
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if cache_key:
                        self.cache.set(cache_key, data)
                    return data
                if resp.status in (429, 403):
                    wait = 2 ** (attempt + 1)
                    print(f"   Rate limited ({resp.status}), retrying in {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                return {}
        return {}

    async def _fetch_league_standings(
        self, session: aiohttp.ClientSession, league_id: int
    ) -> Dict:
        url = f"{self.BASE_URL}/leagues-classic/{league_id}/standings/"
        return await self._fetch_json(session, url, cache_key=f"league_{league_id}")

    async def _fetch_picks(
        self, session: aiohttp.ClientSession, team_id: int, gw: int
    ) -> Dict:
        url = f"{self.BASE_URL}/entry/{team_id}/event/{gw}/picks/"
        return await self._fetch_json(
            session, url, cache_key=f"spy_picks_{team_id}_{gw}"
        )

    async def _fetch_entry(self, session: aiohttp.ClientSession, team_id: int) -> Dict:
        url = f"{self.BASE_URL}/entry/{team_id}/"
        return await self._fetch_json(session, url, cache_key=f"spy_entry_{team_id}")

    async def _find_league_id(self, session: aiohttp.ClientSession) -> Optional[int]:
        """Find the user's classic mini-league (smallest private league)"""
        entry = await self._fetch_entry(session, self.your_team_id)
        leagues = entry.get("leagues", {}).get("classic", [])

        # Filter to private leagues (not overall/country), prefer smallest
        private_leagues = [
            lg
            for lg in leagues
            if lg.get("league_type") == "x"  # 'x' = private classic league
        ]

        if not private_leagues:
            # Fallback: any non-global league
            private_leagues = [
                lg
                for lg in leagues
                if lg.get("entry_rank", 0) > 0 and lg.get("entry_rank", 999999) < 50
            ]

        if not private_leagues:
            return None

        # Pick the smallest league (most likely the competitive one)
        private_leagues.sort(key=lambda lg: lg.get("entry_rank", 999999))
        return private_leagues[0]["id"]

    async def analyze_league(
        self, league_id: Optional[int] = None
    ) -> Optional[LeagueIntel]:
        """Analyze a mini-league and build intelligence"""
        current_gw = self.fpl.get_current_gameweek()

        async with aiohttp.ClientSession(headers=self.HEADERS) as session:
            # Auto-detect league if not provided
            if league_id is None:
                league_id = await self._find_league_id(session)
                if league_id is None:
                    print("   Could not find a private mini-league")
                    return None

            # Fetch standings
            standings = await self._fetch_league_standings(session, league_id)
            if not standings:
                print(f"   Could not fetch league {league_id}")
                return None

            league_name = standings.get("league", {}).get("name", "Unknown League")
            results = standings.get("standings", {}).get("results", [])

            if not results:
                print("   No standings found")
                return None

            # Build rival list
            rivals = []
            your_rank = 0
            your_points = 0

            for entry in results:
                team_id = entry["entry"]
                rival = Rival(
                    team_id=team_id,
                    manager_name=entry.get("player_name", "Unknown"),
                    team_name=entry.get("entry_name", "Unknown"),
                    total_points=entry.get("total", 0),
                    rank=entry.get("rank", 0),
                )

                if team_id == self.your_team_id:
                    your_rank = rival.rank
                    your_points = rival.total_points
                else:
                    rivals.append(rival)

            leader_points = results[0]["total"] if results else 0
            points_to_leader = leader_points - your_points

            # Fetch each rival's squad (batched)
            print(f"   Scanning {len(rivals)} rival squads...")
            for rival in rivals:
                picks_data = await self._fetch_picks(session, rival.team_id, current_gw)
                if picks_data and "picks" in picks_data:
                    rival.squad_ids = [p["element"] for p in picks_data["picks"]]
                    for p in picks_data["picks"]:
                        if p.get("is_captain"):
                            rival.captain_id = p["element"]
                await asyncio.sleep(1.0)  # Rate limit

        # Analyze ownership within league
        num_rivals = len(rivals)
        if num_rivals == 0:
            return None

        player_counts: Counter = Counter()
        captain_counts: Counter = Counter()

        for rival in rivals:
            for pid in rival.squad_ids:
                player_counts[pid] += 1
            if rival.captain_id:
                captain_counts[rival.captain_id] += 1

        league_ownership = {
            pid: (count / num_rivals) * 100 for pid, count in player_counts.items()
        }

        league_captains = dict(captain_counts)

        # Strategic analysis
        # Differentials: good players that <20% of rivals own
        all_players = self.fpl.get_all_players()
        scored_available = [
            p
            for p in all_players
            if p.status == "a" and p.form >= 3.0 and p.minutes > 0
        ]

        differential_vs_league = [
            p.id
            for p in scored_available
            if league_ownership.get(p.id, 0) < 20 and p.form >= 4.0
        ]

        # Must-haves: players >70% of rivals own (dangerous to not have)
        must_have = [pid for pid, pct in league_ownership.items() if pct >= 70]

        # Captain fades: popular rival captains (captaining same = no advantage)
        captain_fades = [
            pid
            for pid, count in captain_counts.items()
            if count >= num_rivals * 0.4  # >40% of rivals captaining
        ]

        # Captain targets: strong players that few rivals captain
        captain_targets = [
            p.id
            for p in scored_available
            if p.form >= 5.0
            and captain_counts.get(p.id, 0) <= 1
            and p.total_points > 50
        ]

        # Global vs league discrepancy analysis
        safe_diffs, league_over, trending = self._analyze_global_vs_league(
            league_ownership, all_players
        )

        return LeagueIntel(
            league_name=league_name,
            your_rank=your_rank,
            total_managers=num_rivals + 1,
            points_to_leader=points_to_leader,
            rivals=rivals,
            league_ownership=league_ownership,
            league_captains=league_captains,
            differential_vs_league=differential_vs_league,
            must_have=must_have,
            captain_fades=captain_fades,
            captain_targets=captain_targets,
            safe_differentials=safe_diffs,
            league_overweight=league_over,
            trending_hidden=trending,
        )

    def _analyze_global_vs_league(
        self,
        league_ownership: Dict[int, float],
        all_players: List[Player],
    ) -> Tuple[List[GlobalVsLeague], List[GlobalVsLeague], List[GlobalVsLeague]]:
        """Compare global FPL ownership against mini-league ownership to find exploitable gaps.

        Returns:
            safe_differentials: Players the world owns but your league doesn't (reliable edge)
            league_overweight: Players your league over-indexes on vs the world (potential fade targets)
            trending_hidden: Players with high transfer momentum that neither league nor world fully owns
        """
        safe_differentials = []
        league_overweight = []
        trending_hidden = []

        for player in all_players:
            if player.status != "a" or player.minutes == 0:
                continue

            global_pct = player.selected_by_percent
            league_pct = league_ownership.get(player.id, 0)
            discrepancy = global_pct - league_pct

            def _make_entry(category: str) -> GlobalVsLeague:
                return GlobalVsLeague(
                    player_id=player.id,
                    web_name=player.web_name,
                    team=player.team,
                    position=player.position,
                    global_ownership=global_pct,
                    league_ownership=league_pct,
                    discrepancy=discrepancy,
                    form=player.form,
                    transfers_in_event=player.transfers_in_event,
                    category=category,
                )

            # Safe differentials: world owns significantly more than league
            # These are proven picks that give you an edge over rivals
            if (
                discrepancy > 15
                and global_pct > 15
                and league_pct < 30
                and player.form >= 3.0
            ):
                safe_differentials.append(_make_entry("safe_differential"))

            # League overweight: league owns significantly more than the world
            # Your rivals are over-indexing — if these players fail, rivals lose ground
            if discrepancy < -20 and league_pct > 30 and player.form < 5.0:
                league_overweight.append(_make_entry("league_overweight"))

            # Trending hidden: high transfer-in momentum, still under-owned everywhere
            # The world is moving to these players — early mover advantage
            if (
                player.transfers_in_event > 50000
                and global_pct < 15
                and league_pct < 20
                and player.form >= 4.0
            ):
                trending_hidden.append(_make_entry("trending_hidden"))

        # Sort each list by most exploitable
        safe_differentials.sort(key=lambda x: x.discrepancy, reverse=True)
        league_overweight.sort(key=lambda x: x.discrepancy)  # Most negative first
        trending_hidden.sort(key=lambda x: x.transfers_in_event, reverse=True)

        return safe_differentials[:10], league_overweight[:10], trending_hidden[:10]

    def format_intel(self, intel: LeagueIntel) -> str:
        """Format league intelligence for display"""
        lines = [
            f"\n{'=' * 60}",
            f"  LEAGUE SPY: {intel.league_name}",
            f"{'=' * 60}",
            f"\n  Your Rank: {intel.your_rank}/{intel.total_managers}",
            f"  Points Behind Leader: {intel.points_to_leader}",
        ]

        # Standings
        lines.append("\n  STANDINGS:")
        for rival in sorted(intel.rivals, key=lambda r: r.rank):
            lines.append(
                f"    #{rival.rank} {rival.manager_name:20} {rival.total_points} pts - {rival.team_name}"
            )

        # Most owned in league
        lines.append("\n  MOST OWNED IN YOUR LEAGUE:")
        top_owned = sorted(
            intel.league_ownership.items(), key=lambda x: x[1], reverse=True
        )[:10]
        for pid, pct in top_owned:
            p = self.fpl.get_player(pid)
            if p:
                lines.append(f"    {p.web_name:15} ({p.team}) - {pct:.0f}% of rivals")

        # Popular captains
        if intel.league_captains:
            lines.append("\n  RIVAL CAPTAINS THIS GW:")
            for pid, count in sorted(
                intel.league_captains.items(), key=lambda x: x[1], reverse=True
            ):
                p = self.fpl.get_player(pid)
                if p:
                    lines.append(
                        f"    {p.web_name:15} - {count}/{len(intel.rivals)} rivals"
                    )

        # Must haves (risky to not own)
        if intel.must_have:
            lines.append("\n  MUST-HAVES (>70% rival ownership):")
            for pid in intel.must_have[:5]:
                p = self.fpl.get_player(pid)
                if p:
                    pct = intel.league_ownership.get(pid, 0)
                    lines.append(f"    {p.web_name:15} ({p.team}) - {pct:.0f}% own")

        # Captain strategy
        lines.append("\n  CAPTAIN STRATEGY:")
        if intel.captain_fades:
            fade_names = []
            for pid in intel.captain_fades:
                p = self.fpl.get_player(pid)
                if p:
                    fade_names.append(p.web_name)
            if fade_names:
                lines.append(f"    FADE (no advantage): {', '.join(fade_names)}")

        if intel.captain_targets:
            target_players = []
            for pid in intel.captain_targets[:5]:
                p = self.fpl.get_player(pid)
                if p:
                    cap_count = intel.league_captains.get(pid, 0)
                    target_players.append(
                        f"{p.web_name} ({cap_count}/{len(intel.rivals)} rivals)"
                    )
            if target_players:
                lines.append(f"    TARGET (differential): {', '.join(target_players)}")

        # Differentials vs league
        lines.append("\n  DIFFERENTIALS VS YOUR LEAGUE:")
        diff_players = []
        for pid in intel.differential_vs_league[:8]:
            p = self.fpl.get_player(pid)
            if p:
                pct = intel.league_ownership.get(pid, 0)
                diff_players.append((p, pct))

        diff_players.sort(key=lambda x: x[0].form, reverse=True)
        for p, pct in diff_players[:5]:
            lines.append(
                f"    {p.web_name:15} ({p.team}) form:{p.form} - {pct:.0f}% of rivals own"
            )

        # Global vs League discrepancy analysis
        if intel.safe_differentials or intel.league_overweight or intel.trending_hidden:
            lines.append("\n  GLOBAL vs LEAGUE ANALYSIS:")

        if intel.safe_differentials:
            lines.append("\n  SAFE DIFFERENTIALS (world owns, league doesn't):")
            for entry in intel.safe_differentials[:5]:
                lines.append(
                    f"    {entry.web_name:15} ({entry.team}) "
                    f"Global: {entry.global_ownership:.1f}% | League: {entry.league_ownership:.0f}% | "
                    f"Gap: +{entry.discrepancy:.0f}% | Form: {entry.form}"
                )

        if intel.league_overweight:
            lines.append("\n  LEAGUE OVERWEIGHT (rivals over-indexing vs world):")
            for entry in intel.league_overweight[:5]:
                lines.append(
                    f"    {entry.web_name:15} ({entry.team}) "
                    f"Global: {entry.global_ownership:.1f}% | League: {entry.league_ownership:.0f}% | "
                    f"Gap: {entry.discrepancy:.0f}% | Form: {entry.form}"
                )

        if intel.trending_hidden:
            lines.append("\n  TRENDING HIDDEN GEMS (high transfers in, low ownership):")
            for entry in intel.trending_hidden[:5]:
                xfers = entry.transfers_in_event
                xfers_str = f"{xfers / 1000:.0f}k" if xfers >= 1000 else str(xfers)
                lines.append(
                    f"    {entry.web_name:15} ({entry.team}) "
                    f"Transfers in: {xfers_str} | Global: {entry.global_ownership:.1f}% | "
                    f"League: {entry.league_ownership:.0f}% | Form: {entry.form}"
                )

        return "\n".join(lines)
