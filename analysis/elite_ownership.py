"""
Elite Ownership Calculator - Top 10% manager ownership analysis

Instead of using global ownership (11M+ managers including casuals),
this module calculates ownership among top-ranked managers who make
better decisions.
"""

import asyncio
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional
import aiohttp


@dataclass
class EliteOwnership:
    """Elite ownership data for all players"""

    player_ownership: Dict[int, float]  # player_id -> ownership %
    sample_size: int
    avg_rank: float


async def fetch_top_manager_squads(
    session: aiohttp.ClientSession,
    num_managers: int = 1000,
    current_gw: int = 23,
) -> List[List[int]]:
    """Fetch squads of top-ranked managers.

    Args:
        session: aiohttp session
        num_managers: Number of top managers to sample
        current_gw: Current gameweek for picks

    Returns:
        List of squads (each squad is list of player IDs)
    """
    squads = []
    manager_ids = []

    # Fetch top manager IDs from overall standings
    # API returns 50 per page
    pages_needed = (num_managers + 49) // 50

    for page in range(1, pages_needed + 1):
        url = f"https://fantasy.premierleague.com/api/leagues-classic/314/standings/?page_standings={page}"
        try:
            async with session.get(url, timeout=15) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results = data.get("standings", {}).get("results", [])
                    for m in results:
                        if len(manager_ids) < num_managers:
                            manager_ids.append(m["entry"])
        except Exception:
            continue

        # Rate limiting
        await asyncio.sleep(0.5)

    print(f"   Fetching squads for {len(manager_ids)} top managers...")

    # Fetch squads in batches
    batch_size = 20
    for i in range(0, len(manager_ids), batch_size):
        batch = manager_ids[i : i + batch_size]
        tasks = []

        for entry_id in batch:
            url = f"https://fantasy.premierleague.com/api/entry/{entry_id}/event/{current_gw}/picks/"
            tasks.append(fetch_squad(session, url))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list) and result:
                squads.append(result)

        # Progress update
        if (i + batch_size) % 100 == 0:
            print(
                f"   Processed {min(i + batch_size, len(manager_ids))}/{len(manager_ids)} managers..."
            )

        # Rate limiting
        await asyncio.sleep(1.0)

    return squads


async def fetch_squad(session: aiohttp.ClientSession, url: str) -> List[int]:
    """Fetch a single squad."""
    try:
        async with session.get(url, timeout=10) as resp:
            if resp.status == 200:
                data = await resp.json()
                return [p["element"] for p in data.get("picks", [])]
    except Exception:
        pass
    return []


def calculate_elite_ownership(squads: List[List[int]]) -> Dict[int, float]:
    """Calculate ownership percentages from sampled squads.

    Args:
        squads: List of squads (each is list of player IDs)

    Returns:
        Dict mapping player_id -> ownership percentage
    """
    if not squads:
        return {}

    # Count occurrences of each player
    player_counts = Counter()
    for squad in squads:
        for player_id in squad:
            player_counts[player_id] += 1

    # Convert to percentages
    total_squads = len(squads)
    ownership = {
        player_id: (count / total_squads) * 100
        for player_id, count in player_counts.items()
    }

    return ownership


async def get_elite_ownership(
    current_gw: int = 23,
    num_managers: int = 500,
    cache_file: Optional[str] = None,
) -> EliteOwnership:
    """Get elite ownership data for top managers.

    Args:
        current_gw: Current gameweek
        num_managers: Number of top managers to sample (default 500 for speed)
        cache_file: Optional cache file path

    Returns:
        EliteOwnership with player ownership percentages
    """
    print(f"🔄 Calculating elite ownership (top {num_managers} managers)...")

    async with aiohttp.ClientSession() as session:
        squads = await fetch_top_manager_squads(
            session, num_managers=num_managers, current_gw=current_gw
        )

    ownership = calculate_elite_ownership(squads)

    print(
        f"✓ Elite ownership calculated ({len(squads)} squads, {len(ownership)} players)"
    )

    return EliteOwnership(
        player_ownership=ownership,
        sample_size=len(squads),
        avg_rank=num_managers / 2,  # Approximate
    )


def compare_elite_vs_global(
    elite: Dict[int, float],
    global_ownership: Dict[int, float],
    player_names: Dict[int, str],
    top_n: int = 20,
) -> None:
    """Print comparison of elite vs global ownership."""

    print("\n" + "=" * 65)
    print("  ELITE vs GLOBAL OWNERSHIP COMPARISON")
    print("=" * 65)

    # Find biggest differences
    diffs = []
    for pid, elite_own in elite.items():
        global_own = global_ownership.get(pid, 0)
        diff = elite_own - global_own
        name = player_names.get(pid, f"Player {pid}")
        diffs.append((pid, name, elite_own, global_own, diff))

    # Sort by absolute difference
    diffs.sort(key=lambda x: abs(x[4]), reverse=True)

    print("\n  ELITE OVERWEIGHTS (top managers own more than casuals):")
    print("    Player              Elite    Global   Diff")
    print("    " + "-" * 45)
    over = [d for d in diffs if d[4] > 5][:10]
    for _, name, elite_own, global_own, diff in over:
        print(
            f"    {name:<18} {elite_own:>5.1f}%   {global_own:>5.1f}%  +{diff:>4.1f}%"
        )

    print("\n  ELITE UNDERWEIGHTS (casuals own more than top managers):")
    print("    Player              Elite    Global   Diff")
    print("    " + "-" * 45)
    under = [d for d in diffs if d[4] < -5][:10]
    for _, name, elite_own, global_own, diff in under:
        print(f"    {name:<18} {elite_own:>5.1f}%   {global_own:>5.1f}%  {diff:>5.1f}%")


if __name__ == "__main__":
    # Test the module
    async def test():
        elite = await get_elite_ownership(current_gw=23, num_managers=100)
        print("\nTop owned by elite:")
        sorted_own = sorted(
            elite.player_ownership.items(), key=lambda x: x[1], reverse=True
        )
        for pid, own in sorted_own[:15]:
            print(f"  Player {pid}: {own:.1f}%")

    asyncio.run(test())
