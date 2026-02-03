"""
Mirror Manager - Copy the most consistent FPL manager's moves

Ian Foster (Entry ID: 8142824) - "The Honest Trundlers"
- Never dropped below #1,248 all season
- Was #1 for GW16-17
- Zero hits taken
- Template, high-ownership strategy

This module fetches his squad and suggests moves to mirror him exactly.
"""

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import aiohttp

# Ian Foster's entry ID - the most consistent manager
MIRROR_MANAGER_ID = 8142824
MIRROR_MANAGER_NAME = "Ian Foster"
MIRROR_MANAGER_TEAM = "The Honest Trundlers"


@dataclass
class MirrorAnalysis:
    """Analysis of how to mirror the target manager"""
    target_name: str
    target_team: str
    target_rank: int
    target_points: int
    target_gw_points: int

    target_squad: List[int]  # Player IDs
    target_captain: int
    target_vice_captain: int

    your_squad: List[int]

    players_to_buy: List[int]
    players_to_sell: List[int]
    transfers_needed: int

    recommend_free_hit: bool
    free_hit_threshold: int  # Transfers above this = use free hit

    captain_match: bool

    target_chips_used: List[dict]
    target_chip_this_gw: Optional[str]


async def fetch_manager_squad(
    session: aiohttp.ClientSession,
    entry_id: int,
    gameweek: int,
) -> Tuple[List[int], int, int, Optional[str], int]:
    """Fetch a manager's squad for a specific gameweek.

    Returns: (squad_ids, captain_id, vice_captain_id, chip_used, gw_points)
    """
    url = f'https://fantasy.premierleague.com/api/entry/{entry_id}/event/{gameweek}/picks/'

    try:
        async with session.get(url, timeout=15) as resp:
            if resp.status == 200:
                data = await resp.json()
                picks = data.get('picks', [])

                squad = [p['element'] for p in picks]
                captain = next((p['element'] for p in picks if p['is_captain']), None)
                vice = next((p['element'] for p in picks if p['is_vice_captain']), None)

                chip = data.get('active_chip')
                gw_points = data.get('entry_history', {}).get('points', 0)

                return squad, captain, vice, chip, gw_points
    except Exception as e:
        print(f"   Error fetching squad: {e}")

    return [], None, None, None, 0


async def fetch_manager_info(
    session: aiohttp.ClientSession,
    entry_id: int,
) -> Tuple[int, int, List[dict]]:
    """Fetch manager's current rank, total points, and chips used.

    Returns: (rank, total_points, chips_used)
    """
    url = f'https://fantasy.premierleague.com/api/entry/{entry_id}/'

    try:
        async with session.get(url, timeout=15) as resp:
            if resp.status == 200:
                data = await resp.json()
                rank = data.get('summary_overall_rank', 0)
                points = data.get('summary_overall_points', 0)

        # Get chips from history
        history_url = f'https://fantasy.premierleague.com/api/entry/{entry_id}/history/'
        async with session.get(history_url, timeout=15) as resp:
            if resp.status == 200:
                data = await resp.json()
                chips = data.get('chips', [])
                return rank, points, chips

    except Exception as e:
        print(f"   Error fetching manager info: {e}")

    return 0, 0, []


async def fetch_manager_transfers(
    session: aiohttp.ClientSession,
    entry_id: int,
) -> List[dict]:
    """Fetch manager's transfer history."""
    url = f'https://fantasy.premierleague.com/api/entry/{entry_id}/transfers/'

    try:
        async with session.get(url, timeout=15) as resp:
            if resp.status == 200:
                return await resp.json()
    except Exception:
        pass

    return []


async def analyze_mirror(
    your_squad: List[int],
    current_gw: int,
    free_hit_threshold: int = 4,
) -> Optional[MirrorAnalysis]:
    """Analyze how to mirror Ian Foster's squad.

    Args:
        your_squad: Your current squad (list of player IDs)
        current_gw: Current gameweek
        free_hit_threshold: Number of transfers above which to recommend Free Hit

    Returns:
        MirrorAnalysis with recommendations
    """
    print(f"🔄 Fetching {MIRROR_MANAGER_NAME}'s squad...")

    async with aiohttp.ClientSession() as session:
        # Fetch Ian's current squad
        target_squad, captain, vice, chip, gw_pts = await fetch_manager_squad(
            session, MIRROR_MANAGER_ID, current_gw
        )

        if not target_squad:
            print(f"   ❌ Could not fetch {MIRROR_MANAGER_NAME}'s squad")
            return None

        # Fetch his rank and chips
        rank, total_pts, chips_used = await fetch_manager_info(session, MIRROR_MANAGER_ID)

        print(f"✓ {MIRROR_MANAGER_NAME} is rank #{rank:,} with {total_pts} pts")

    # Calculate differences
    your_set = set(your_squad)
    target_set = set(target_squad)

    players_to_buy = list(target_set - your_set)
    players_to_sell = list(your_set - target_set)
    transfers_needed = len(players_to_buy)

    # Check if captain matches
    captain_match = captain in your_squad

    # Recommend free hit if too many transfers
    recommend_free_hit = transfers_needed >= free_hit_threshold

    return MirrorAnalysis(
        target_name=MIRROR_MANAGER_NAME,
        target_team=MIRROR_MANAGER_TEAM,
        target_rank=rank,
        target_points=total_pts,
        target_gw_points=gw_pts,
        target_squad=target_squad,
        target_captain=captain,
        target_vice_captain=vice,
        your_squad=your_squad,
        players_to_buy=players_to_buy,
        players_to_sell=players_to_sell,
        transfers_needed=transfers_needed,
        recommend_free_hit=recommend_free_hit,
        free_hit_threshold=free_hit_threshold,
        captain_match=captain_match,
        target_chips_used=chips_used,
        target_chip_this_gw=chip,
    )


def format_mirror_analysis(
    analysis: MirrorAnalysis,
    player_names: Dict[int, str],
) -> str:
    """Format mirror analysis for display."""
    lines = []

    lines.append("")
    lines.append("=" * 65)
    lines.append(f"  MIRROR MODE: {analysis.target_name}")
    lines.append(f"  \"{analysis.target_team}\"")
    lines.append("=" * 65)
    lines.append("")
    lines.append(f"  Current Rank:  #{analysis.target_rank:,}")
    lines.append(f"  Total Points:  {analysis.target_points}")
    lines.append(f"  GW Points:     {analysis.target_gw_points}")
    lines.append("")

    # Chip status
    if analysis.target_chip_this_gw:
        lines.append(f"  🎯 CHIP ACTIVE: {analysis.target_chip_this_gw.upper()}")
        lines.append("")

    # Squad comparison
    lines.append(f"  SQUAD COMPARISON:")
    lines.append(f"    Transfers needed:  {analysis.transfers_needed}")
    lines.append("")

    if analysis.transfers_needed == 0:
        lines.append("  ✅ Your squad matches perfectly!")
    else:
        if analysis.recommend_free_hit:
            lines.append(f"  ⚠️  RECOMMEND FREE HIT ({analysis.transfers_needed} transfers needed)")
            lines.append(f"     Threshold: {analysis.free_hit_threshold}+ transfers = use Free Hit")
            lines.append("")

        lines.append("  TRANSFERS TO MATCH:")
        lines.append("")
        lines.append("    OUT:")
        for pid in analysis.players_to_sell:
            name = player_names.get(pid, f"Player {pid}")
            lines.append(f"      - {name}")

        lines.append("")
        lines.append("    IN:")
        for pid in analysis.players_to_buy:
            name = player_names.get(pid, f"Player {pid}")
            lines.append(f"      + {name}")

    # Captain
    lines.append("")
    captain_name = player_names.get(analysis.target_captain, f"Player {analysis.target_captain}")
    vice_name = player_names.get(analysis.target_vice_captain, f"Player {analysis.target_vice_captain}")

    lines.append(f"  CAPTAIN: {captain_name}")
    lines.append(f"  VICE:    {vice_name}")

    if analysis.captain_match:
        lines.append("  ✅ You have the captain in your squad")
    else:
        lines.append("  ⚠️  You don't have the captain - need to transfer in!")

    # Ian's full squad
    lines.append("")
    lines.append(f"  {analysis.target_name.upper()}'S FULL SQUAD:")
    lines.append("    " + "-" * 40)

    for i, pid in enumerate(analysis.target_squad):
        name = player_names.get(pid, f"Player {pid}")
        marker = ""
        if pid == analysis.target_captain:
            marker = " (C)"
        elif pid == analysis.target_vice_captain:
            marker = " (V)"

        if i < 11:
            lines.append(f"    {i+1:2}. {name}{marker}")
        elif i == 11:
            lines.append("    --- Bench ---")
            lines.append(f"    {i+1:2}. {name}{marker}")
        else:
            lines.append(f"    {i+1:2}. {name}{marker}")

    # Chips used
    lines.append("")
    lines.append(f"  CHIPS USED BY {analysis.target_name.upper()}:")
    if analysis.target_chips_used:
        for chip in analysis.target_chips_used:
            lines.append(f"    GW{chip['event']}: {chip['name']}")
    else:
        lines.append("    None yet")

    lines.append("")

    return "\n".join(lines)


async def get_ian_weekly_moves(current_gw: int) -> Optional[dict]:
    """Get Ian Foster's moves for the current gameweek.

    Returns dict with:
    - transfers_in: players bought
    - transfers_out: players sold
    - captain: captain pick
    - chip: chip used (if any)
    """
    async with aiohttp.ClientSession() as session:
        # Get transfers
        transfers = await fetch_manager_transfers(session, MIRROR_MANAGER_ID)

        # Filter to this GW
        gw_transfers = [t for t in transfers if t.get('event') == current_gw]

        # Get squad info
        squad, captain, vice, chip, pts = await fetch_manager_squad(
            session, MIRROR_MANAGER_ID, current_gw
        )

        return {
            'transfers_in': [t['element_in'] for t in gw_transfers],
            'transfers_out': [t['element_out'] for t in gw_transfers],
            'captain': captain,
            'vice_captain': vice,
            'chip': chip,
            'gw_points': pts,
            'squad': squad,
        }


if __name__ == "__main__":
    # Test the module
    async def test():
        # Dummy squad for testing
        test_squad = [1, 5, 106, 373, 151, 283, 328, 355, 401, 427, 579, 116, 245, 308, 446]

        analysis = await analyze_mirror(test_squad, current_gw=24)

        if analysis:
            # Build simple player names map
            print(f"\nTransfers needed: {analysis.transfers_needed}")
            print(f"Recommend Free Hit: {analysis.recommend_free_hit}")
            print(f"Captain: {analysis.target_captain}")

    asyncio.run(test())
