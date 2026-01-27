#!/usr/bin/env python3
"""
FPL Comeback Assistant - Main Entry Point

Generates comprehensive FPL recommendations to help you climb the rankings.
"""

import argparse
import asyncio
import sys
from typing import List, Optional

import config
from data.fpl_api import FPLDataFetcher
from data.news_scraper import NewsScraper
from analysis.fixture_analyzer import FixtureAnalyzer
from analysis.player_scorer import PlayerScorer
from analysis.differential import DifferentialFinder
from analysis.chip_optimizer import ChipOptimizer
from output.recommendations import RecommendationEngine


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="FPL Comeback Assistant - Get recommendations to climb the rankings"
    )

    parser.add_argument(
        "--team-id",
        type=int,
        help="Your FPL team ID (find in your team URL)",
        default=config.TEAM_ID,
    )

    parser.add_argument(
        "--gameweek",
        "-gw",
        type=int,
        help="Gameweek to analyze (default: current)",
        default=None,
    )

    parser.add_argument(
        "--chips",
        nargs="+",
        choices=["wildcard", "free_hit", "triple_captain", "bench_boost", "all", "none"],
        help="Available chips (default: all)",
        default=["all"],
    )

    parser.add_argument(
        "--differentials",
        "-d",
        action="store_true",
        help="Show detailed differential analysis",
    )

    parser.add_argument(
        "--fixtures",
        "-f",
        action="store_true",
        help="Show detailed fixture analysis",
    )

    parser.add_argument(
        "--top-players",
        "-t",
        action="store_true",
        help="Show top players by position",
    )

    parser.add_argument(
        "--chip-strategy",
        "-c",
        action="store_true",
        help="Show detailed chip strategy",
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass cache and fetch fresh data",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Minimal output",
    )

    return parser.parse_args()


async def fetch_team_squad(fpl: FPLDataFetcher, team_id: int) -> Optional[List[int]]:
    """Fetch current squad for a team"""
    try:
        current_gw = fpl.get_current_gameweek()
        picks_data = await fpl.fetch_team_picks(team_id, current_gw)

        if not picks_data or "picks" not in picks_data:
            # Try previous gameweek
            picks_data = await fpl.fetch_team_picks(team_id, current_gw - 1)

        if picks_data and "picks" in picks_data:
            return [p["element"] for p in picks_data["picks"]]

    except Exception as e:
        print(f"Could not fetch team data: {e}")

    return None


def get_available_chips(chips_arg: List[str]) -> List[str]:
    """Parse chips argument"""
    if "all" in chips_arg:
        return ["wildcard", "free_hit", "triple_captain", "bench_boost"]
    elif "none" in chips_arg:
        return []
    else:
        return chips_arg


async def main_async(args):
    """Main async function"""
    print("\n🔄 Fetching FPL data...")

    # Initialize data fetcher
    fpl = FPLDataFetcher()

    # Clear cache if requested
    if args.no_cache:
        fpl.cache.clear()

    # Load all data
    await fpl.load_all_data()

    current_gw = fpl.get_current_gameweek()
    print(f"✓ Data loaded for GW{current_gw}")

    # Initialize news scraper
    news = NewsScraper()

    # Initialize analyzers
    fixtures = FixtureAnalyzer(fpl)
    scorer = PlayerScorer(fpl, fixtures, news)
    differential_finder = DifferentialFinder(fpl, scorer, fixtures)
    chip_optimizer = ChipOptimizer(fpl, fixtures)
    recommender = RecommendationEngine(
        fpl, scorer, fixtures, differential_finder, chip_optimizer, news
    )

    # Fetch team if provided
    squad_ids = None
    if args.team_id:
        print(f"🔄 Fetching team {args.team_id}...")
        squad_ids = await fetch_team_squad(fpl, args.team_id)
        if squad_ids:
            print(f"✓ Found {len(squad_ids)} players in squad")
        else:
            print("⚠️  Could not fetch team - using general recommendations")

    # Get available chips
    available_chips = get_available_chips(args.chips)

    # Generate and display recommendations
    if args.quiet:
        # Minimal output - just key recommendations
        print("\n" + "=" * 40)
        captains = recommender.get_captain_picks(squad_ids, limit=3)
        print("Captain Picks:")
        for i, cp in enumerate(captains, 1):
            print(f"  {i}. {cp.player.player.web_name} - {cp.expected_points:.1f} exp")

        print("\nTop Differential:")
        diffs = differential_finder.find_differentials(position="MID", limit=1)
        if diffs:
            d = diffs[0]
            print(f"  {d.scored_player.player.web_name} - {d.ownership:.1f}% owned")

    elif args.differentials:
        # Detailed differential analysis
        print("\n" + differential_finder.get_differential_summary())

    elif args.fixtures:
        # Detailed fixture analysis
        print("\n📅 FIXTURE DIFFICULTY RANKINGS")
        print("=" * 50)
        print("\nBEST FIXTURES (Next 5 GWs):")
        for run in fixtures.get_best_fixtures(10):
            print(f"  {fixtures.format_fixture_run(run)}")

        print("\nWORST FIXTURES:")
        for run in fixtures.get_worst_fixtures(5):
            print(f"  {fixtures.format_fixture_run(run)}")

        # DGW/BGW info
        dgw = fixtures.get_double_gameweeks()
        bgw = fixtures.get_blank_gameweeks()

        if dgw:
            print("\nDOUBLE GAMEWEEKS:")
            for gw, team_ids in sorted(dgw.items()):
                teams = [fpl.get_team(t).short_name for t in team_ids if fpl.get_team(t)]
                print(f"  GW{gw}: {', '.join(teams)}")

        if bgw:
            print("\nBLANK GAMEWEEKS:")
            for gw, team_ids in sorted(bgw.items()):
                teams = [fpl.get_team(t).short_name for t in team_ids if fpl.get_team(t)]
                print(f"  GW{gw}: {', '.join(teams)} blanking")

    elif args.top_players:
        # Top players by position
        print("\n🏆 TOP PLAYERS BY POSITION")
        print("=" * 50)

        top_by_pos = scorer.get_top_by_position(limit=10)
        for pos, players in top_by_pos.items():
            print(f"\n{pos}:")
            for i, sp in enumerate(players[:5], 1):
                p = sp.player
                avail = "" if sp.availability == "fit" else f" [{sp.availability}]"
                print(
                    f"  {i}. {p.web_name} ({p.team}) £{p.price}m - "
                    f"Score: {sp.overall_score:.2f}{avail}"
                )

    elif args.chip_strategy:
        # Detailed chip strategy
        strategy = chip_optimizer.get_chip_strategy(available_chips)
        print("\n" + strategy.summary)

    else:
        # Full recommendations
        rec = recommender.get_full_recommendations(squad_ids, available_chips)
        print(recommender.format_full_recommendations(rec))


def main():
    """Main entry point"""
    args = parse_args()

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nCancelled")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if "--debug" in sys.argv:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
