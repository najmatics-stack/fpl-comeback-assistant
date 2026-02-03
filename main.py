#!/usr/bin/env python3
"""
FPL Comeback Assistant - Main Entry Point

Generates comprehensive FPL recommendations to help you climb the rankings.
"""

import argparse
import asyncio
import sys
from typing import List, Optional, Set

import config
from data.fpl_api import FPLDataFetcher
from data.fpl_actions import FPLActions
from data.news_scraper import NewsScraper
from analysis.fixture_analyzer import FixtureAnalyzer
from analysis.player_scorer import PlayerScorer
from analysis.differential import DifferentialFinder
from analysis.chip_optimizer import ChipOptimizer
from analysis.backtester import Backtester
from analysis.evaluator import ModelEvaluator
from analysis.comparative_backtest import ComparativeBacktester
from analysis.league_spy import LeagueSpy
from output.recommendations import RecommendationEngine, TransferPlan, TransferRecommendation, CaptainPick


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

    parser.add_argument(
        "--backtest",
        "-b",
        type=str,
        metavar="GW_RANGE",
        help="Blind backtest: single GW (e.g. '20') or range (e.g. '18-22')",
    )

    parser.add_argument(
        "--evaluate",
        "-e",
        type=int,
        metavar="GW",
        help="Evaluate model on completed GW and auto-tune weights",
    )

    parser.add_argument(
        "--compare",
        type=str,
        metavar="GW_LIST",
        help="Compare our model vs baselines (e.g. '18,19,20,21,22')",
    )

    parser.add_argument(
        "--league-spy",
        "-l",
        action="store_true",
        help="Analyze rival squads in your mini-league",
    )

    parser.add_argument(
        "--league-id",
        type=int,
        help="Mini-league ID (auto-detected if not provided)",
        default=None,
    )

    parser.add_argument(
        "--test-login",
        action="store_true",
        help="Test the login flow without making any changes",
    )

    return parser.parse_args()


async def fetch_team_squad(fpl: FPLDataFetcher, team_id: int) -> Optional[List[dict]]:
    """Fetch current squad for a team, accounting for pending transfers"""
    try:
        return await fpl.fetch_current_squad(team_id)
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


def prompt_settings() -> dict:
    """Phase 0: Let user adjust auto-mode settings or press Enter to skip."""
    settings = {
        "max_hits": config.AUTO_MAX_HITS,
        "min_gain_free": config.AUTO_MIN_SCORE_GAIN_FREE,
        "min_gain_hit": config.AUTO_MIN_SCORE_GAIN_HIT,
        "max_transfers": config.AUTO_MAX_TRANSFERS,
        "risk": config.AUTO_RISK_LEVEL,
    }

    print("\n" + "=" * 60)
    print("  PHASE 0: SETTINGS")
    print("=" * 60)
    print(
        f"\n  max_hits={settings['max_hits']} | "
        f"min_gain_free={settings['min_gain_free']} | "
        f"min_gain_hit={settings['min_gain_hit']} | "
        f"max_transfers={settings['max_transfers']} | "
        f"risk={settings['risk']}"
    )

    choice = input("\nAdjust? [Enter=skip, s=settings]: ").strip().lower()
    if choice != "s":
        return settings

    setting_keys = list(settings.keys())
    while True:
        print("\nSettings:")
        for i, key in enumerate(setting_keys, 1):
            print(f"  {i}. {key} = {settings[key]}")
        print()
        cmd = input("Change (<number> <value>, or Enter=done): ").strip()
        if not cmd:
            break

        parts = cmd.split(None, 1)
        if len(parts) != 2:
            print("  Format: <number> <value>")
            continue

        try:
            idx = int(parts[0]) - 1
            if idx < 0 or idx >= len(setting_keys):
                print(f"  Pick 1-{len(setting_keys)}")
                continue
            key = setting_keys[idx]
            if key == "risk":
                if parts[1] not in ("conservative", "balanced", "aggressive"):
                    print("  Must be: conservative, balanced, or aggressive")
                    continue
                settings[key] = parts[1]
            elif key in ("max_hits", "max_transfers"):
                settings[key] = int(parts[1])
            else:
                settings[key] = float(parts[1])
            print(f"  {key} = {settings[key]}")
        except (ValueError, IndexError):
            print("  Invalid input")

    return settings


def prompt_chip(available_chips: List[str]) -> Optional[str]:
    """Phase 0.5: Let user select a chip to play or skip."""
    print("\n" + "=" * 60)
    print("  PHASE 0.5: CHIP SELECTION")
    print("=" * 60)

    chip_map = {
        "f": ("free_hit", "Free Hit"),
        "w": ("wildcard", "Wildcard"),
        "t": ("triple_captain", "Triple Captain"),
        "b": ("bench_boost", "Bench Boost"),
    }

    # Build prompt showing only available chips
    options = []
    for key, (chip_val, chip_label) in chip_map.items():
        if chip_val in available_chips:
            options.append(f"{key}={chip_label.lower()}")

    if not options:
        print("\n  No chips available.")
        return None

    prompt_str = ", ".join(options)
    print(f"\n  Play a chip? [Enter=none, {prompt_str}]: ", end="")
    choice = input().strip().lower()

    if choice in chip_map:
        chip_val, chip_label = chip_map[choice]
        if chip_val in available_chips:
            print(f"  Selected: {chip_label}")
            return chip_val
        else:
            print(f"  {chip_label} not available. Proceeding without chip.")
            return None

    print("  No chip selected.")
    return None


def prompt_locked_players(
    fpl: FPLDataFetcher,
    scorer: PlayerScorer,
    squad_ids: List[int],
) -> Set[int]:
    """Phase 0.75: Let user mark players as non-negotiable (cannot be sold).

    Returns set of locked player IDs.
    """
    print("\n" + "=" * 60)
    print("  PHASE 0.75: LOCK PLAYERS")
    print("=" * 60)

    # Group players by position with injury analysis
    positions = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
    injury_count = 0
    doubt_count = 0

    for pid in squad_ids:
        player = fpl.get_player(pid)
        if player:
            sp = scorer.score_player(player)
            positions[player.position].append({
                "id": pid,
                "player": player,
                "scored": sp,
            })
            if sp.availability == "injured" or sp.availability == "suspended":
                injury_count += 1
            elif sp.availability == "doubt":
                doubt_count += 1

    # Sort each position by score descending
    for pos in positions:
        positions[pos].sort(key=lambda x: x["scored"].overall_score, reverse=True)

    # Injury summary header
    if injury_count > 0 or doubt_count > 0:
        print(f"\n  ⚠️  INJURY ALERT: {injury_count} out, {doubt_count} doubtful")

    # Display with numbers and injury status
    print("\n  Your squad:")
    numbered = []
    idx = 1
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        if positions[pos]:
            print(f"\n  {pos}:")
            for p in positions[pos]:
                player = p["player"]
                sp = p["scored"]

                # Availability indicator with details
                avail_icon = ""
                avail_detail = ""
                if sp.availability == "injured":
                    avail_icon = "❌"
                    avail_detail = f" [OUT: {sp.injury_details or player.news or 'Injured'}]"
                elif sp.availability == "suspended":
                    avail_icon = "🚫"
                    avail_detail = f" [SUSPENDED: {player.news or 'Red card'}]"
                elif sp.availability == "doubt":
                    # Show chance of playing if available
                    if player.chance_of_playing is not None:
                        avail_icon = "⚠️"
                        avail_detail = f" [{player.chance_of_playing}%: {player.news or 'Doubt'}]"
                    else:
                        avail_icon = "⚠️"
                        avail_detail = f" [DOUBT: {sp.injury_details or player.news or 'Knock'}]"
                else:
                    avail_icon = "✓"

                print(
                    f"   {idx:2}. {avail_icon} {player.web_name:15} ({player.team:3}) "
                    f"Form: {player.form} | Score: {sp.overall_score:.1f}{avail_detail}"
                )
                numbered.append(p["id"])
                idx += 1

    print("\n  Lock players you don't want to sell (injured players shown for awareness).")
    choice = input("  [Enter=none, e.g. 1,3,5]: ").strip()

    if not choice:
        print("  No players locked.")
        return set()

    locked_ids: Set[int] = set()
    parts = choice.replace(" ", "").split(",")
    for part in parts:
        try:
            num = int(part)
            if 1 <= num <= len(numbered):
                pid = numbered[num - 1]
                locked_ids.add(pid)
                player = fpl.get_player(pid)
                if player:
                    print(f"  Locked: {player.web_name}")
            else:
                print(f"  Invalid number: {num}")
        except ValueError:
            print(f"  Invalid input: {part}")

    if locked_ids:
        print(f"\n  {len(locked_ids)} player(s) locked.")
    else:
        print("  No players locked.")

    return locked_ids


def prompt_free_hit_squad(
    recommender: RecommendationEngine,
    fpl: FPLDataFetcher,
    squad_ids: List[int],
) -> tuple:
    """Phase 1-FH: Build and display optimal Free Hit squad.

    Returns (chosen_transfers, new_squad_ids) or ([], squad_ids) if skipped.
    """
    print("\n" + "=" * 60)
    print("  PHASE 1-FH: FREE HIT SQUAD")
    print("=" * 60)

    # Calculate total budget from current squad prices
    total_budget = 0.0
    for pid in squad_ids:
        player = fpl.get_player(pid)
        if player:
            total_budget += player.price
    if total_budget < 50.0:
        total_budget = 100.0  # Fallback

    print(f"\n  Budget: £{total_budget:.1f}m (current squad value)")

    squad, transfers = recommender.get_free_hit_squad(squad_ids, total_budget)

    if not squad or len(squad) < 15:
        print("  Could not build a full 15-player squad.")
        return [], squad_ids

    new_squad_ids = [sp.player.id for sp in squad]
    print(recommender.format_free_hit_squad(squad, transfers, total_budget))

    while True:
        print(f"\n  [Enter=accept, e=edit, s=skip]: ", end="")
        choice = input().strip().lower()

        if choice == "s":
            print("  Skipping Free Hit.")
            return [], squad_ids

        if choice == "e":
            # Edit sub-loop
            print("  Commands: swap <pos_num> <player_name>, show <name>, done")
            # Number the squad for reference
            numbered = []
            for pos in ["GKP", "DEF", "MID", "FWD"]:
                for sp in squad:
                    if sp.player.position == pos:
                        numbered.append(sp)

            while True:
                print()
                for i, sp in enumerate(numbered, 1):
                    p = sp.player
                    print(f"   {i:2}. {p.position} {p.web_name:15} ({p.team}) £{p.price}m | Score: {sp.overall_score:.1f}")

                cmd = input("\nedit-fh> ").strip()
                if not cmd or cmd == "done":
                    break

                parts = cmd.split(None, 2)
                verb = parts[0].lower()

                if verb == "swap" and len(parts) >= 3:
                    try:
                        idx = int(parts[1]) - 1
                        name_query = parts[2]
                        if idx < 0 or idx >= len(numbered):
                            print(f"  Invalid number (1-{len(numbered)})")
                            continue

                        old_sp = numbered[idx]
                        new_player = recommender.find_player_by_name(name_query)
                        if not new_player:
                            print(f"  Player '{name_query}' not found")
                            continue

                        if new_player.position != old_sp.player.position:
                            print(
                                f"  Position mismatch: {new_player.web_name} is {new_player.position}, "
                                f"need {old_sp.player.position}"
                            )
                            continue

                        # Check 3-per-team (excluding old player)
                        temp_ids = [sp.player.id for sp in numbered if sp.player.id != old_sp.player.id]
                        team_count = sum(1 for pid in temp_ids if fpl.get_player(pid) and fpl.get_player(pid).team_id == new_player.team_id)
                        if team_count >= 3:
                            print(f"  {new_player.web_name} would exceed 3-per-team limit")
                            continue

                        # Check budget
                        old_cost = sum(sp.player.price for sp in numbered)
                        new_cost = old_cost - old_sp.player.price + new_player.price
                        if new_cost > total_budget:
                            print(f"  £{new_player.price}m too expensive (would need £{new_cost:.1f}m, budget £{total_budget:.1f}m)")
                            continue

                        new_scored = recommender.scorer.score_player(new_player)
                        numbered[idx] = new_scored
                        print(f"  Swapped: {old_sp.player.web_name} -> {new_player.web_name}")

                    except (ValueError, IndexError):
                        print("  Usage: swap <number> <player name>")

                elif verb == "show" and len(parts) >= 2:
                    name_query = " ".join(parts[1:])
                    player = recommender.find_player_by_name(name_query)
                    if player:
                        sp = recommender.scorer.score_player(player)
                        fixture_run = recommender.fixtures.get_fixture_run(player.team_id)
                        next_fix = fixture_run.fixtures[0][1] if fixture_run.fixtures else "?"
                        print(
                            f"  {player.web_name} ({player.team}) {player.position} "
                            f"£{player.price}m | Form: {player.form} | "
                            f"Score: {sp.overall_score:.1f} | Next: {next_fix}"
                        )
                    else:
                        print(f"  Player '{name_query}' not found")
                else:
                    print("  Commands: swap <pos_num> <player_name>, show <name>, done")

            # Rebuild squad and transfers from edited numbered list
            squad = list(numbered)
            new_squad_ids = [sp.player.id for sp in squad]

            # Recalculate transfers
            current_set = set(squad_ids)
            new_set = set(new_squad_ids)
            transfers = []
            outs_by_pos = {}
            ins_by_pos = {}

            for pid in squad_ids:
                if pid not in new_set:
                    p = fpl.get_player(pid)
                    if p:
                        sp = recommender.scorer.score_player(p)
                        outs_by_pos.setdefault(p.position, []).append(sp)

            for sp in squad:
                if sp.player.id not in current_set:
                    ins_by_pos.setdefault(sp.player.position, []).append(sp)

            for pos in ["GKP", "DEF", "MID", "FWD"]:
                pos_outs = outs_by_pos.get(pos, [])
                pos_ins = ins_by_pos.get(pos, [])
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

            print(recommender.format_free_hit_squad(squad, transfers, total_budget))
            continue  # Show squad again for accept/skip

        # Enter or unrecognized → accept
        print(f"  Accepted Free Hit squad ({len(transfers)} transfers)")
        return transfers, new_squad_ids


def prompt_transfer_plan(
    recommender: RecommendationEngine,
    plans: List[TransferPlan],
    squad_ids: List[int],
) -> List[TransferRecommendation]:
    """Phase 1: Show transfer plans, let user pick/edit/skip."""
    print("\n" + "=" * 60)
    print("  PHASE 1: TRANSFER PLAN")
    print("=" * 60)

    print(recommender.format_transfer_plans(plans))

    # Build letter-to-plan mapping
    plan_map = {}
    recommended_plan = None
    for plan in plans:
        letter = plan.display_label[0].lower()  # 'a', 'b', 'c'
        plan_map[letter] = plan
        if plan.is_recommended:
            recommended_plan = plan

    print(
        f"\nPick plan [Enter={recommended_plan.display_label[0] if recommended_plan else 'B'}, "
        f"a/b/c=select, e=edit, s=skip]: ",
        end="",
    )
    choice = input().strip().lower()

    if choice == "s":
        print("  Skipping transfers.")
        return []

    # Select base plan
    if choice == "e":
        selected = recommended_plan or plans[1] if len(plans) > 1 else plans[0]
    elif choice in plan_map:
        selected = plan_map[choice]
    else:
        # Enter or unrecognized → recommended
        selected = recommended_plan or (plans[1] if len(plans) > 1 else plans[0])

    chosen_transfers = list(selected.transfers)

    if choice != "e":
        print(f"\n  Selected: {selected.display_label}")
        return chosen_transfers

    # Edit sub-loop
    print(f"\n  Editing: {selected.display_label}")
    print("  Commands: drop <n>, swap <n> <name>, add <out> <in>, show <name>, done")

    while True:
        # Display current transfers
        if chosen_transfers:
            for i, tr in enumerate(chosen_transfers, 1):
                print(
                    f"   {i}. {tr.player_out.player.web_name} -> {tr.player_in.player.web_name} "
                    f"[+{tr.score_gain:.1f}]"
                )
        else:
            print("   (no transfers)")

        cmd = input("\nedit> ").strip()
        if not cmd or cmd == "done":
            break

        parts = cmd.split(None, 2)
        verb = parts[0].lower()

        if verb == "drop" and len(parts) >= 2:
            try:
                idx = int(parts[1]) - 1
                if 0 <= idx < len(chosen_transfers):
                    removed = chosen_transfers.pop(idx)
                    print(f"  Dropped: {removed.player_out.player.web_name} -> {removed.player_in.player.web_name}")
                else:
                    print(f"  Invalid number (1-{len(chosen_transfers)})")
            except ValueError:
                print("  Usage: drop <number>")

        elif verb == "swap" and len(parts) >= 3:
            try:
                idx = int(parts[1]) - 1
                name_query = parts[2]
                if idx < 0 or idx >= len(chosen_transfers):
                    print(f"  Invalid number (1-{len(chosen_transfers)})")
                    continue

                new_player = recommender.find_player_by_name(name_query)
                if not new_player:
                    print(f"  Player '{name_query}' not found")
                    continue

                old_tr = chosen_transfers[idx]
                if new_player.position != old_tr.player_out.player.position:
                    print(
                        f"  Position mismatch: {new_player.web_name} is {new_player.position}, "
                        f"need {old_tr.player_out.player.position}"
                    )
                    continue

                # Check FPL rules
                pending = [t for j, t in enumerate(chosen_transfers) if j != idx]
                if recommender._would_violate_rules(
                    new_player.id, old_tr.player_out.player.id, squad_ids, pending
                ):
                    print(f"  {new_player.web_name} would violate FPL rules (3-per-team/position limits)")
                    continue

                new_scored = recommender.scorer.score_player(new_player)
                new_gain = new_scored.overall_score - old_tr.player_out.overall_score
                new_price_diff = new_player.price - old_tr.player_out.player.price

                chosen_transfers[idx] = TransferRecommendation(
                    player_out=old_tr.player_out,
                    player_in=new_scored,
                    score_gain=new_gain,
                    price_diff=new_price_diff,
                    reason="manual override",
                )
                print(
                    f"  Swapped in: {new_player.web_name} ({new_player.team}) "
                    f"[+{new_gain:.1f}]"
                )
            except (ValueError, IndexError):
                print("  Usage: swap <number> <player name>")

        elif verb == "add" and len(parts) >= 3:
            names = parts[1:]
            if len(names) < 2:
                # Try splitting the second part
                sub_parts = parts[1].split(None, 1) if len(parts) == 2 else names
                if len(sub_parts) < 2:
                    print("  Usage: add <out_name> <in_name>")
                    continue
                names = sub_parts

            out_player = recommender.find_player_by_name(names[0])
            in_player = recommender.find_player_by_name(names[1])

            if not out_player:
                print(f"  Player '{names[0]}' not found")
                continue
            if not in_player:
                print(f"  Player '{names[1]}' not found")
                continue

            if out_player.id not in squad_ids:
                print(f"  {out_player.web_name} is not in your squad")
                continue
            if out_player.position != in_player.position:
                print(
                    f"  Position mismatch: {out_player.web_name} ({out_player.position}) vs "
                    f"{in_player.web_name} ({in_player.position})"
                )
                continue

            if recommender._would_violate_rules(
                in_player.id, out_player.id, squad_ids, chosen_transfers
            ):
                print(f"  Would violate FPL rules")
                continue

            out_scored = recommender.scorer.score_player(out_player)
            in_scored = recommender.scorer.score_player(in_player)
            gain = in_scored.overall_score - out_scored.overall_score
            price_diff = in_player.price - out_player.price

            chosen_transfers.append(TransferRecommendation(
                player_out=out_scored,
                player_in=in_scored,
                score_gain=gain,
                price_diff=price_diff,
                reason="manual add",
            ))
            print(
                f"  Added: {out_player.web_name} -> {in_player.web_name} "
                f"[+{gain:.1f}]"
            )

        elif verb == "show" and len(parts) >= 2:
            name_query = " ".join(parts[1:])
            player = recommender.find_player_by_name(name_query)
            if player:
                sp = recommender.scorer.score_player(player)
                fixture_run = recommender.fixtures.get_fixture_run(player.team_id)
                next_fix = fixture_run.fixtures[0][1] if fixture_run.fixtures else "?"
                print(
                    f"  {player.web_name} ({player.team}) {player.position} "
                    f"£{player.price}m | Form: {player.form} | "
                    f"Score: {sp.overall_score:.1f} | Next: {next_fix}"
                )
            else:
                print(f"  Player '{name_query}' not found")

        else:
            print("  Commands: drop <n>, swap <n> <name>, add <out> <in>, show <name>, done")

    return chosen_transfers


def prompt_captain(
    recommender: RecommendationEngine,
    captain_picks: List[CaptainPick],
    squad_ids: List[int],
) -> tuple:
    """Phase 2: Captain selection. Returns (captain_id, vc_id)."""
    print("\n" + "=" * 60)
    print("  PHASE 2: CAPTAIN")
    print("=" * 60)

    if not captain_picks:
        print("  No captain picks available.")
        return (None, None)

    print("\n  Top captain picks:")
    for i, cp in enumerate(captain_picks, 1):
        p = cp.player.player
        avail = "" if cp.player.availability == "fit" else f" [{cp.player.availability}]"
        print(
            f"   {i}. {p.web_name} ({p.team}) - "
            f"{cp.expected_points:.1f} exp pts{avail}"
        )
        print(f"      vs {cp.fixture_info} | {cp.reason}")

    print(
        f"\nCaptain [Enter=#1, 1-{len(captain_picks)}=select, o=override]: ",
        end="",
    )
    choice = input().strip().lower()

    if choice == "o":
        name_query = input("  Captain name: ").strip()
        player = recommender.find_player_by_name(name_query)
        if player and player.id in squad_ids:
            captain_id = player.id
            print(f"  Captain: {player.web_name}")
        else:
            if player:
                print(f"  {player.web_name} not in squad, using #1")
            else:
                print(f"  Player not found, using #1")
            captain_id = captain_picks[0].player.player.id
    elif choice.isdigit() and 1 <= int(choice) <= len(captain_picks):
        idx = int(choice) - 1
        captain_id = captain_picks[idx].player.player.id
        print(f"  Captain: {captain_picks[idx].player.player.web_name}")
    else:
        captain_id = captain_picks[0].player.player.id
        print(f"  Captain: {captain_picks[0].player.player.web_name}")

    # Vice captain
    remaining = [cp for cp in captain_picks if cp.player.player.id != captain_id]
    if remaining:
        print(
            f"\nVice captain [Enter=#1 remaining, 1-{len(remaining)}=select]: ",
            end="",
        )
        for i, cp in enumerate(remaining, 1):
            p = cp.player.player
            print(f"\n   {i}. {p.web_name} ({p.team}) - {cp.expected_points:.1f} exp", end="")
        print()
        vc_choice = input("  > ").strip()
        if vc_choice.isdigit() and 1 <= int(vc_choice) <= len(remaining):
            vc_id = remaining[int(vc_choice) - 1].player.player.id
        else:
            vc_id = remaining[0].player.player.id
        vc_name = next(
            (cp.player.player.web_name for cp in remaining if cp.player.player.id == vc_id),
            "?",
        )
        print(f"  Vice captain: {vc_name}")
    else:
        vc_id = captain_id

    return (captain_id, vc_id)


async def prompt_review_and_execute(
    fpl_actions_cls,
    team_id: int,
    transfers: List[TransferRecommendation],
    captain_id: Optional[int],
    vc_id: Optional[int],
    chip_to_play,
    current_gw: int,
):
    """Phase 3: Final review and execute."""
    print("\n" + "=" * 60)
    print("  PHASE 3: REVIEW & EXECUTE")
    print("=" * 60)

    actions = []
    if transfers:
        for tr in transfers:
            actions.append(
                f"Transfer: {tr.player_out.player.web_name} -> {tr.player_in.player.web_name} "
                f"[+{tr.score_gain:.1f}]"
            )

    if captain_id:
        actions.append(f"Captain: player #{captain_id}")
    if vc_id and vc_id != captain_id:
        actions.append(f"Vice captain: player #{vc_id}")

    # Normalize chip_to_play to a string value (or None)
    # It can be a string from prompt_chip() or a ChipRecommendation object
    chip_value = None
    if chip_to_play:
        if isinstance(chip_to_play, str):
            chip_value = chip_to_play
        else:
            chip_value = chip_to_play.chip.value

    if chip_value:
        actions.append(f"Activate: {chip_value.replace('_', ' ').title()}")

    if not actions:
        print("\n  No actions to execute.")
        return

    print("\n  Planned actions:")
    for i, action in enumerate(actions, 1):
        print(f"   {i}. {action}")

    confirm = input("\nExecute? [Enter/y=yes, n=cancel]: ").strip().lower()
    if confirm == "n":
        print("  Cancelled.")
        return

    # Login and execute
    fpl_actions = fpl_actions_cls(team_id)
    all_ok = True
    try:
        if not await fpl_actions.login():
            print("❌ Cannot proceed without login")
            return

        # Execute transfers
        if transfers:
            outs = [tr.player_out.player.id for tr in transfers]
            ins = [tr.player_in.player.id for tr in transfers]

            use_wc = chip_value == "wildcard"
            use_fh = chip_value == "free_hit"

            if not await fpl_actions.make_transfers(outs, ins, current_gw=current_gw, wildcard=use_wc, free_hit=use_fh):
                all_ok = False

        # Set captain
        if captain_id:
            if not await fpl_actions.set_captain(
                captain_id, vc_id or captain_id, current_gw
            ):
                all_ok = False

        # Activate chip (if not already used via transfer)
        if chip_value in ("triple_captain", "bench_boost"):
            if not await fpl_actions.activate_chip(chip_value, current_gw + 1):
                all_ok = False

        if all_ok:
            print("\n✓ All actions executed successfully!")
        else:
            print("\n⚠️  Some actions failed — check output above")

    finally:
        await fpl_actions.close()


async def interactive_auto_mode(
    recommender: RecommendationEngine,
    fpl: FPLDataFetcher,
    squad_ids: List[int],
    available_chips: List[str],
    team_id: int,
    current_gw: int,
):
    """Interactive 4-phase auto-pilot mode."""
    print("\n" + "=" * 60)
    print("  AUTO-PILOT MODE (Interactive)")
    print("=" * 60)

    # Pre-phase: Login and fetch REAL current squad
    # The public API shows the squad at the last GW deadline, but we need
    # the actual current state including any pending transfers
    print("\n🔐 Authenticating to get real-time squad state...")
    fpl_actions = FPLActions(team_id)
    try:
        if await fpl_actions.login():
            real_squad = await fpl_actions.get_current_squad()
            if real_squad and len(real_squad) == 15:
                if set(real_squad) != set(squad_ids):
                    print("✓ Updated squad with pending transfers")
                    # Show what changed
                    old_set = set(squad_ids)
                    new_set = set(real_squad)
                    added = new_set - old_set
                    removed = old_set - new_set
                    for pid in removed:
                        p = fpl.get_player(pid)
                        if p:
                            print(f"   - {p.web_name} (transferred out)")
                    for pid in added:
                        p = fpl.get_player(pid)
                        if p:
                            print(f"   + {p.web_name} (transferred in)")
                    squad_ids = real_squad
                else:
                    print("✓ Squad is up to date")
            else:
                print("⚠️  Could not fetch real squad, using cached data")
        else:
            print("⚠️  Login failed, using cached squad data")
    except Exception as e:
        print(f"⚠️  Auth error: {e}, using cached squad data")
    finally:
        await fpl_actions.close()

    # Phase 0: Settings
    settings = prompt_settings()

    # Phase 0.5: Chip selection
    selected_chip = prompt_chip(available_chips)

    # Phase 0.75: Lock non-negotiable players
    locked_ids = prompt_locked_players(fpl, recommender.scorer, squad_ids)

    # Determine flow based on chip
    captain_squad_ids = squad_ids  # IDs used for captain picks
    chip_to_play = None

    if selected_chip == "free_hit":
        # Phase 1-FH: Free Hit squad builder (locked players shown but can be overridden)
        chosen_transfers, captain_squad_ids = prompt_free_hit_squad(
            recommender, fpl, squad_ids
        )
        if chosen_transfers:
            # Build a chip_to_play object for review phase
            chip_to_play = selected_chip
        else:
            # User skipped — no chip
            selected_chip = None

    elif selected_chip == "wildcard":
        # Wildcard uses same Free Hit flow (rebuild from scratch, but permanent)
        chosen_transfers, captain_squad_ids = prompt_free_hit_squad(
            recommender, fpl, squad_ids
        )
        if chosen_transfers:
            chip_to_play = selected_chip
        else:
            selected_chip = None

    else:
        # Normal transfer plan flow (no chip, triple captain, or bench boost)
        if selected_chip in ("triple_captain", "bench_boost"):
            chip_to_play = selected_chip

        # Generate transfer plans (locked players excluded from candidates)
        plans = recommender.get_transfer_plans(
            squad_ids,
            free_transfers=1,
            max_hits=settings["max_hits"],
            min_gain_free=settings["min_gain_free"],
            min_gain_hit=settings["min_gain_hit"],
            risk_level=settings["risk"],
            locked_ids=locked_ids,
        )

        # Phase 1: Transfer plan
        chosen_transfers = prompt_transfer_plan(recommender, plans, squad_ids)

    # Update captain squad IDs to reflect post-transfer squad
    if chosen_transfers and captain_squad_ids is squad_ids:
        captain_squad_ids = list(squad_ids)
        for tr in chosen_transfers:
            out_id = tr.player_out.player.id
            in_id = tr.player_in.player.id
            if out_id in captain_squad_ids:
                captain_squad_ids.remove(out_id)
                captain_squad_ids.append(in_id)

    # Phase 2: Captain (uses post-transfer squad IDs)
    captain_picks = recommender.get_captain_picks(
        captain_squad_ids, limit=config.TOP_CAPTAINS
    )
    captain_id, vc_id = prompt_captain(recommender, captain_picks, captain_squad_ids)

    # Phase 3: Review and execute
    await prompt_review_and_execute(
        FPLActions, team_id, chosen_transfers,
        captain_id, vc_id, chip_to_play, current_gw,
    )


async def main_async(args):
    """Main async function"""

    if args.test_login:
        print("\n🔄 Testing login flow...")
        fpl_actions = FPLActions(args.team_id)
        try:
            if await fpl_actions.login():
                print("✓ Login test passed — session is valid")
            else:
                print("❌ Login test failed")
        finally:
            await fpl_actions.close()
        return

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
            print(f"   [debug] squad IDs (from GW picks): {squad_ids}")
            if team_info:
                name = f"{team_info.get('player_first_name', '')} {team_info.get('player_last_name', '')}".strip()
                team_name = team_info.get('name', 'Unknown')
                print(f"✓ Manager: {name} | Team: {team_name}")
        else:
            print("⚠️  Could not fetch team - using general recommendations")

    # Get available chips
    available_chips = get_available_chips(args.chips)

    # Run league spy for auto/default modes when team_id is available
    league_intel = None
    is_auto_or_default = not any([
        args.evaluate, args.compare, args.backtest, args.league_spy,
        args.quiet, args.differentials, args.fixtures, args.top_players,
        args.chip_strategy, args.my_team,
    ])
    if args.team_id and is_auto_or_default:
        try:
            print("🔄 Scanning rival squads...")
            spy = LeagueSpy(fpl, args.team_id)
            league_intel = await spy.analyze_league(league_id=args.league_id)
            if league_intel:
                print(f"✓ League intel: {league_intel.league_name} ({league_intel.total_managers} managers)")
            else:
                print("⚠️  Could not gather league intel - continuing without")
        except Exception as e:
            print(f"⚠️  League spy failed: {e} - continuing without")

    # Build recommender with league intel
    recommender = RecommendationEngine(
        fpl, scorer, fixtures, differential_finder, chip_optimizer, news,
        league_intel=league_intel,
    )

    # Generate and display recommendations
    if args.evaluate:
        evaluator = ModelEvaluator(fpl)
        result = await evaluator.evaluate_and_save(args.evaluate)
        print(evaluator.format_result(result))
        return

    if args.compare:
        gws = [int(g.strip()) for g in args.compare.split(",")]
        comp = ComparativeBacktester(fpl)
        result = await comp.run_comparison(gws)
        print(comp.format_comparison(result))
        return

    if args.backtest:
        # Parse GW range
        bt = args.backtest
        backtester = Backtester(fpl)
        if "-" in bt:
            start, end = bt.split("-")
            await backtester.run_multi_gw_backtest(int(start), int(end))
        else:
            await backtester.run_backtest(int(bt))
        return

    if args.league_spy:
        if not args.team_id:
            print("❌ --team-id required for league spy")
            return

        print("🔄 Scanning rival squads...")
        spy = LeagueSpy(fpl, args.team_id)
        intel = await spy.analyze_league(league_id=args.league_id)
        if intel:
            print(spy.format_intel(intel))
        else:
            print("❌ Could not gather league intelligence")
        return

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
        # Interactive auto-pilot mode
        if not args.team_id:
            print("❌ --team-id required for auto mode")
            return

        if not squad_ids:
            print("❌ Could not fetch squad")
            return

        await interactive_auto_mode(
            recommender, fpl, squad_ids, available_chips,
            args.team_id, current_gw,
        )

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
