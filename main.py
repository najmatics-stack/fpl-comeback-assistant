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
from data.fpl_actions import FPLActions, prompt_credentials
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

    parser.add_argument(
        "--my-team",
        "-m",
        action="store_true",
        help="Show your current squad with ratings",
    )

    parser.add_argument(
        "--auto",
        "-a",
        action="store_true",
        help="Auto-pilot: execute transfers, set captain, activate chips",
    )

    return parser.parse_args()


async def fetch_team_squad(fpl: FPLDataFetcher, team_id: int) -> Optional[List[dict]]:
    """Fetch current squad for a team with full pick data"""
    try:
        current_gw = fpl.get_current_gameweek()
        picks_data = await fpl.fetch_team_picks(team_id, current_gw)

        if not picks_data or "picks" not in picks_data:
            # Try previous gameweek
            picks_data = await fpl.fetch_team_picks(team_id, current_gw - 1)

        if picks_data and "picks" in picks_data:
            return picks_data["picks"]

    except Exception as e:
        print(f"Could not fetch team data: {e}")

    return None


async def fetch_team_info(fpl: FPLDataFetcher, team_id: int) -> Optional[dict]:
    """Fetch team manager info (name, rank, points, etc.)"""
    try:
        return await fpl.fetch_team_data(team_id)
    except Exception as e:
        print(f"Could not fetch team info: {e}")
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
    squad_picks = None
    team_info = None
    if args.team_id:
        print(f"🔄 Fetching team {args.team_id}...")
        squad_picks = await fetch_team_squad(fpl, args.team_id)
        team_info = await fetch_team_info(fpl, args.team_id)
        if squad_picks:
            squad_ids = [p["element"] for p in squad_picks]
            print(f"✓ Found {len(squad_ids)} players in squad")
            if team_info:
                name = f"{team_info.get('player_first_name', '')} {team_info.get('player_last_name', '')}".strip()
                team_name = team_info.get('name', 'Unknown')
                print(f"✓ Manager: {name} | Team: {team_name}")
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

    elif args.my_team:
        # Show current squad with ratings
        if not squad_picks:
            print("\n❌ No team data available. Please provide --team-id")
        else:
            print("\n" + "=" * 60)
            print("  YOUR SQUAD ANALYSIS")
            print("=" * 60)

            # Team info header
            if team_info:
                name = f"{team_info.get('player_first_name', '')} {team_info.get('player_last_name', '')}".strip()
                team_name = team_info.get('name', 'Unknown')
                overall_rank = team_info.get('summary_overall_rank', 'N/A')
                overall_points = team_info.get('summary_overall_points', 0)
                gw_points = team_info.get('summary_event_points', 0)

                # Format rank with commas
                if isinstance(overall_rank, int):
                    rank_str = f"{overall_rank:,}"
                else:
                    rank_str = str(overall_rank)

                print(f"\n👤 {name}")
                print(f"🏆 {team_name}")
                print(f"📊 Overall Rank: {rank_str} | Total Points: {overall_points} | GW{current_gw}: {gw_points} pts")

            # Group players by position
            positions = {"GKP": [], "DEF": [], "MID": [], "FWD": []}

            for pick in squad_picks:
                player = fpl.get_player(pick["element"])
                if player:
                    sp = scorer.score_player(player)
                    positions[player.position].append({
                        "player": player,
                        "scored": sp,
                        "is_captain": pick.get("is_captain", False),
                        "is_vice": pick.get("is_vice_captain", False),
                        "multiplier": pick.get("multiplier", 1),
                        "position": pick.get("position", 0),
                    })

            # Sort starting XI (positions 1-11) vs bench (12-15)
            print("\n" + "-" * 60)
            print("STARTING XI")
            print("-" * 60)

            total_score = 0
            for pos in ["GKP", "DEF", "MID", "FWD"]:
                starters = [p for p in positions[pos] if p["position"] <= 11]
                if starters:
                    print(f"\n{pos}:")
                    for p in sorted(starters, key=lambda x: x["scored"].overall_score, reverse=True):
                        player = p["player"]
                        sp = p["scored"]
                        total_score += sp.overall_score

                        # Captain/vice badges
                        badge = ""
                        if p["is_captain"]:
                            badge = " 👑(C)"
                        elif p["is_vice"]:
                            badge = " (VC)"

                        # Availability icon
                        avail_icon = {"fit": "✓", "doubt": "⚠️", "injured": "❌", "suspended": "🚫"}.get(sp.availability, "")

                        # Fixture info
                        fixture_run = fixtures.get_fixture_run(player.team_id)
                        next_fix = fixture_run.fixtures[0][1] if fixture_run.fixtures else "?"

                        print(f"   {avail_icon} {player.web_name:15} ({player.team}) £{player.price}m | "
                              f"Form: {player.form} | Score: {sp.overall_score:.1f} | Next: {next_fix}{badge}")

            print("\n" + "-" * 60)
            print("BENCH")
            print("-" * 60)

            for pos in ["GKP", "DEF", "MID", "FWD"]:
                bench = [p for p in positions[pos] if p["position"] > 11]
                for p in bench:
                    player = p["player"]
                    sp = p["scored"]
                    avail_icon = {"fit": "✓", "doubt": "⚠️", "injured": "❌", "suspended": "🚫"}.get(sp.availability, "")
                    print(f"   {avail_icon} {player.web_name:15} ({player.team}) £{player.price}m | "
                          f"Form: {player.form} | Score: {sp.overall_score:.1f}")

            print("\n" + "-" * 60)
            print(f"SQUAD STRENGTH: {total_score:.1f} (Starting XI total score)")
            print("-" * 60)

            # Weakest link
            all_starters = []
            for pos in positions.values():
                all_starters.extend([p for p in pos if p["position"] <= 11])

            if all_starters:
                weakest = min(all_starters, key=lambda x: x["scored"].overall_score)
                print(f"\n⚠️  Weakest link: {weakest['player'].web_name} (Score: {weakest['scored'].overall_score:.1f})")
                print(f"   Consider replacing in upcoming transfers")

    elif args.auto:
        # Auto-pilot mode
        if not args.team_id:
            print("❌ --team-id required for auto mode")
            return

        if not squad_ids:
            print("❌ Could not fetch squad")
            return

        # Generate recommendations first
        rec = recommender.get_full_recommendations(squad_ids, available_chips)
        print(recommender.format_full_recommendations(rec))

        # Confirm before executing
        print("\n" + "=" * 60)
        print("  AUTO-PILOT MODE")
        print("=" * 60)

        # Summarize planned actions
        actions = []
        if rec.transfers:
            for tr in rec.transfers[:2]:  # Max 2 transfers
                actions.append(
                    f"Transfer: {tr.player_out.player.web_name} → {tr.player_in.player.web_name}"
                )

        if rec.captain_picks:
            cp = rec.captain_picks[0]
            actions.append(f"Captain: {cp.player.player.web_name}")
            if len(rec.captain_picks) > 1:
                vc = rec.captain_picks[1]
                actions.append(f"Vice Captain: {vc.player.player.web_name}")

        chip_to_play = None
        for chip_rec in rec.chip_strategy.recommendations:
            if chip_rec.priority == 1 and chip_rec.recommended_gw == current_gw + 1:
                chip_to_play = chip_rec
                actions.append(f"Activate: {chip_rec.chip.value.replace('_', ' ').title()}")

        if not actions:
            print("\nNo actions to execute this week.")
            return

        print("\nPlanned actions:")
        for i, action in enumerate(actions, 1):
            print(f"  {i}. {action}")

        confirm = input("\nExecute these actions? (yes/no): ").strip().lower()
        if confirm not in ("yes", "y"):
            print("Cancelled.")
            return

        # Login
        email, password = prompt_credentials()
        fpl_actions = FPLActions(args.team_id)

        try:
            if not await fpl_actions.login(email, password):
                print("❌ Cannot proceed without login")
                return

            # Execute transfers
            if rec.transfers:
                transfers = rec.transfers[:2]
                outs = [tr.player_out.player.id for tr in transfers]
                ins = [tr.player_in.player.id for tr in transfers]

                use_wc = chip_to_play and chip_to_play.chip.value == "wildcard"
                use_fh = chip_to_play and chip_to_play.chip.value == "free_hit"

                await fpl_actions.make_transfers(outs, ins, wildcard=use_wc, free_hit=use_fh)

            # Set captain
            if rec.captain_picks:
                captain_id = rec.captain_picks[0].player.player.id
                vc_id = rec.captain_picks[1].player.player.id if len(rec.captain_picks) > 1 else captain_id
                await fpl_actions.set_captain(captain_id, vc_id, current_gw)

            # Activate chip (if not already used via transfer)
            if chip_to_play and chip_to_play.chip.value in ("triple_captain", "bench_boost"):
                await fpl_actions.activate_chip(chip_to_play.chip.value, current_gw + 1)

            print("\n✓ All actions executed successfully!")

        finally:
            await fpl_actions.close()

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
